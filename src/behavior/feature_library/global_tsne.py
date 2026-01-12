"""
GlobalTSNE feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple
import re
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from openTSNE import TSNEEmbedding, affinity, initialization

from behavior.dataset import register_feature
from .helpers import _build_path_sequence_map, _load_array_from_spec
from behavior.dataset import _latest_feature_run_root, _resolve_inputs, _feature_run_root
from behavior.helpers import to_safe_name


@register_feature
class GlobalTSNE: 
    """
    Global t-SNE over multiple prior features discovered from the dataset's feature indexes.
    - inputs: a list of dicts describing which feature outputs to include.
      Each input supports:
        {
          "feature": "pair-wavelet",  # prior feature name
          "run_id":  None,                         # optional, pick latest if None
          "pattern": "wavelet_social_seq=*persp=*.npz",  # files to glob inside the run folder
          "load": { "kind": "npz", "key": "spectrogram", "transpose": True }  # loader spec
        }
    The loaded arrays are treated as (T x D); arrays from all inputs with the same sequence
    are length-aligned (min T) and concatenated horizontally.

    Heavy work happens in fit(); transform() returns a tiny stub DF.
    """
    name: str = "global-tsne"
    version: str = "0.2"
    params: dict

    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            # Multi-input spec (good defaults for your current pipeline):
            inputs=[
                {
                    "feature": "pair-wavelet",
                    "run_id": None,  # pick latest
                    "pattern": "wavelet_social_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
                {
                    "feature": "pair-egocentric-wavelet",
                    "run_id": None,
                    "pattern": "wavelet_ego_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
            ],
            inputset=None,
            map_existing_inputs=False,
            reuse_embedding=None,
            artifact=None,
            scaler=None,
            random_state=42,
            r_scaler=200_000,          # cap for standardizer fit sample
            total_templates=2000,      # farthest-first target
            pre_quota_per_key=50,      # pre-sample per key
            perplexity=50,
            n_jobs=8,
            partial_k=25,
            partial_iters=100,
            partial_lr=1.0,
            map_chunk=20_000,
        )
        self.params = {**defaults, **(params or {})}
        self._inputs_overridden = bool(params and "inputs" in params)
        self._rng = np.random.default_rng(self.params["random_state"])
        self._scaler: Optional[StandardScaler] = None
        self._embedding: Optional[TSNEEmbedding] = None
        self._artifacts: Dict[str, Any] = {}
        self._ds = None  # set by bind_dataset
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        self._inputs_meta: Dict[str, Any] = {}
        self._resolved_inputs: List[dict] = []
        self._scope_filter: Optional[dict] = None

    # dataset hook
    def bind_dataset(self, ds) -> None:
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ---- Feature protocol ----
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def partial_fit(self, X: pd.DataFrame) -> None: raise NotImplementedError

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("GlobalTSNE requires dataset binding. Ensure Dataset.run_feature calls bind_dataset().")

        # 0) Gather inputs from feature indexes
        inputset_name = self.params.get("inputset")
        explicit_inputs = self.params["inputs"] if (self._inputs_overridden or not inputset_name) else None
        inputs, inputs_meta = _resolve_inputs(
            self._ds, explicit_inputs, inputset_name,
            explicit_override=self._inputs_overridden
        )
        self._resolved_inputs = inputs
        self._inputs_meta = inputs_meta
        # key is safe_seq (derived from dataset index or filename)
        per_key_parts: Dict[str, List[np.ndarray]] = {}

        allowed_safe = None
        if self._scope_filter:
            allowed_safe = set(self._scope_filter.get("safe_sequences") or [])

        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(self._ds, feat_name)
            else:
                run_root = _feature_run_root(self._ds, feat_name, run_id)

            pattern = spec.get("pattern", "*.npz")
            loader  = spec.get("load", {"kind": "npz", "key": "spectrogram", "transpose": True})

            files = sorted(run_root.glob(pattern))
            if not files:
                print(f"[global-tsne] WARN: no files for {feat_name} ({run_id}) with pattern {pattern}", file=sys.stderr)
                continue

            seq_map = self._feature_seq_map(feat_name, run_id)

            def parse_key(path: Path) -> str:
                stem = path.stem
                m_seq = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
                if m_seq:
                    safe_seq = m_seq.group(1)
                else:
                    safe_seq = seq_map.get(path.resolve())
                if not safe_seq:
                    safe_seq = to_safe_name(stem)
                return str(safe_seq)

            for pth in files:
                key = parse_key(pth)
                if allowed_safe is not None and key not in allowed_safe:
                    continue
                arr = self._load_matrix(pth, loader)
                if arr is None or arr.size == 0:
                    continue
                if key not in per_key_parts:
                    per_key_parts[key] = []
                per_key_parts[key].append(arr)

        if not per_key_parts:
            raise RuntimeError("GlobalTSNE found no usable inputs from the specified prior features.")

        # 1) Concatenate inputs horizontally per key (align on min T)
        features_per_key: Dict[str, np.ndarray] = {}
        n_total = 0
        for key, mats in per_key_parts.items():
            if not mats:
                continue
            T_min = min(m.shape[0] for m in mats)
            mats = [m[:T_min] for m in mats]
            X = np.hstack(mats)  # (T_min, D_total)
            features_per_key[key] = X
            n_total += X.shape[0]

        keys = list(features_per_key.keys())
        if not keys:
            raise RuntimeError("No combined feature frames after alignment.")

        if bool(self.params.get("map_existing_inputs")):
            self._prepare_reuse_artifacts()
            mapped = self._map_sequences(features_per_key)
            self._artifacts["mapped_coords"] = mapped
            self._artifacts["keys"] = keys
            self._artifacts["inputs_meta"] = inputs_meta
            return

        # 2) Global standardizer
        r_scaler = int(self.params["r_scaler"])
        pools = []
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            take = min(X.shape[0], max(1000, int(0.05 * X.shape[0])))
            idx = self._rng.choice(X.shape[0], size=take, replace=False)
            pools.append(X[idx])
        Xsamp = np.vstack(pools)
        if Xsamp.shape[0] > r_scaler:
            idx = self._rng.choice(Xsamp.shape[0], size=r_scaler, replace=False)
            Xsamp = Xsamp[idx]
        scaler = StandardScaler().fit(Xsamp)
        self._scaler = scaler

        # 3) Template selection (farthest-first over pre-sample)
        T_target = int(self.params["total_templates"])
        quota = max(int(self.params["pre_quota_per_key"]), T_target // max(1, len(keys)))
        pre = []
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            take = min(X.shape[0], quota * 3)
            idx = self._rng.choice(X.shape[0], size=take, replace=False)
            pre.append(scaler.transform(X[idx]))
        X_pre = np.vstack(pre)

        sel = [int(self._rng.integers(0, X_pre.shape[0]))]
        d2  = np.sum((X_pre - X_pre[sel[0]])**2, axis=1)
        while len(sel) < min(T_target, X_pre.shape[0]):
            i = int(np.argmax(d2))
            sel.append(i)
            d2 = np.minimum(d2, np.sum((X_pre - X_pre[i])**2, axis=1))
        templates = X_pre[np.array(sel)]

        # 4) Fit openTSNE on templates
        aff = affinity.PerplexityBasedNN(
            templates,
            perplexity=int(self.params["perplexity"]),
            metric="euclidean",
            method="annoy",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        init = initialization.pca(templates, random_state=int(self.params["random_state"]))
        emb = TSNEEmbedding(
            init, aff,
            negative_gradient_method="fft",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        emb.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True, verbose=False)
        emb.optimize(n_iter=750, momentum=0.8, inplace=True, verbose=False)
        self._embedding = emb

        # 5) Map all frames in chunks
        mapped = self._map_sequences(features_per_key)

        # hold artifacts for save_model
        self._artifacts["keys"] = keys
        self._artifacts["templates"] = templates
        self._artifacts["template_indices"] = np.array(sel)
        self._artifacts["mapped_coords"] = mapped
        self._artifacts["inputs_meta"] = inputs_meta

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Just a stub so Dataset.run_feature can index a parquet per (group,sequence)
        return pd.DataFrame({"global_tsne_done": [True]})

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # Save templates
        templates = self._artifacts.get("templates")
        if templates is not None:
            np.savez_compressed(run_root / "global_templates_features.npz", templates=templates)

        # Save tsne coords of templates + selection indices
        if self._embedding is not None:
            Y_templates = np.asarray(self._embedding)
            np.savez_compressed(run_root / "global_tsne_templates.npz",
                                Y=Y_templates, sel=self._artifacts.get("template_indices", np.array([], int)))

        # Save embedding + scaler
        joblib.dump(
            {
                "embedding": self._embedding,
                "scaler": self._scaler,
                "params": self.params,
            },
            run_root / "global_opentsne_embedding.joblib"
        )

        # Save per-key coords
        mapped = self._artifacts.get("mapped_coords", {})
        for safe_seq, Y in mapped.items():
            out_name = f"global_tsne_coords_seq={safe_seq}.npz"
            np.savez_compressed(run_root / out_name, Y=Y)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._embedding = bundle.get("embedding", None)
        self._scaler = bundle.get("scaler", None)

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _resolve_feature_run_root(self, feature_name: str, run_id: str | None) -> tuple[str, Path]:
        if self._ds is None:
            raise RuntimeError("GlobalTSNE requires dataset binding before resolving feature runs.")
        if run_id is None:
            run_id, run_root = _latest_feature_run_root(self._ds, feature_name)
        else:
            run_root = _feature_run_root(self._ds, feature_name, run_id)
        return str(run_id), run_root

    def _load_embedding_from_spec(self, spec: dict) -> tuple[Any, Any]:
        feature = spec.get("feature")
        if not feature:
            raise ValueError("reuse_embedding spec requires 'feature'.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"reuse_embedding: no files matching '{pattern}' in {run_root}")
        bundle = joblib.load(files[0])
        embedding = bundle.get("embedding")
        scaler = bundle.get("scaler")
        if embedding is None:
            raise ValueError(f"reuse_embedding bundle missing 'embedding' object: {files[0]}")
        self._artifacts.setdefault("reuse_sources", {})["embedding"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return embedding, scaler

    def _load_scaler_from_spec(self, spec: dict):
        feature = spec.get("feature")
        if not feature:
            raise ValueError("scaler spec requires 'feature'.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"scaler spec: no files matching '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        scaler = obj if key is None else obj[key]
        self._artifacts.setdefault("reuse_sources", {})["scaler"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return scaler

    def _load_templates_from_spec(self, spec: dict) -> Optional[np.ndarray]:
        feature = spec.get("feature")
        if not feature:
            raise ValueError("artifact spec requires 'feature'.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_templates_features.npz")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"artifact spec: no files matching '{pattern}' in {run_root}")
        load_spec = spec.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        arr = _load_array_from_spec(files[0], load_spec)
        if arr is None:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        self._artifacts.setdefault("reuse_sources", {})["templates"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return arr

    def _prepare_reuse_artifacts(self) -> None:
        emb_spec = self.params.get("reuse_embedding")
        if not emb_spec:
            raise ValueError("map_existing_inputs=True requires params['reuse_embedding'].")
        embedding, scaler = self._load_embedding_from_spec(emb_spec)
        self._embedding = embedding
        self._scaler = scaler
        scaler_override = self.params.get("scaler")
        if scaler_override:
            self._scaler = self._load_scaler_from_spec(scaler_override)
        if self._scaler is None:
            raise ValueError("Reusable embedding bundle missing scaler; provide params['scaler'].")
        art_spec = self.params.get("artifact")
        if art_spec:
            templates = self._load_templates_from_spec(art_spec)
            if templates is not None:
                self._artifacts["templates"] = templates

    def _map_sequences(self, features_per_key: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._embedding is None or self._scaler is None:
            raise RuntimeError("GlobalTSNE mapping requires both embedding and scaler.")
        embedding = self._embedding
        scaler = self._scaler
        CHUNK = int(self.params["map_chunk"])
        Kp = int(self.params["partial_k"])
        Pp = int(self.params["perplexity"])
        It = int(self.params["partial_iters"])
        Lr = float(self.params["partial_lr"])

        def map_chunk_block(X_chunk_std: np.ndarray) -> np.ndarray:
            part = embedding.prepare_partial(X_chunk_std, initialization="median", k=Kp, perplexity=Pp)
            part = part.optimize(
                n_iter=It,
                learning_rate=Lr,
                exaggeration=2.0,
                momentum=0.0,
                inplace=False,
                verbose=False,
            )
            return np.asarray(part)

        mapped: dict[str, np.ndarray] = {}
        for key, X in features_per_key.items():
            if X.shape[0] == 0:
                continue
            Xs = scaler.transform(X)
            blocks = [map_chunk_block(Xs[i:i + CHUNK]) for i in range(0, Xs.shape[0], CHUNK)]
            mapped[key] = np.vstack(blocks) if blocks else np.empty((0, 2), dtype=np.float32)
        return mapped

    # ------- loaders -------
    def _load_matrix(self, pth: Path, spec: dict) -> np.ndarray | None:
        """
        Load a 2D matrix from a file according to a 'load' spec:
        - kind: 'npz' or 'parquet'
        - key:  for npz (required), ignored for parquet
        - columns: optional list of columns for parquet
        - drop_columns: optional list of columns to drop before numeric-only selection
        - numeric_only: default True for parquet; selects only numeric dtypes
        - transpose: bool; if True, returns A.T
        """
        kind = str(spec.get("kind", "parquet")).lower()
        transpose = bool(spec.get("transpose", False))

        if kind == "npz":
            key = spec.get("key", None)
            if not key:
                raise ValueError("load.kind='npz' requires 'key' in load spec")
            npz = np.load(pth, allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"Key '{key}' not found in {pth.name}; available: {list(npz.files)}")
            A = np.asarray(npz[key])
            if A.ndim == 1:  # ensure 2D
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            return A.T if transpose else A

        elif kind == "parquet":
            import pandas as pd
            df = pd.read_parquet(pth)

            # Optional: drop explicitly unwanted columns first
            drop_cols = set(spec.get("drop_columns", []))
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")

            cols = spec.get("columns", None)
            if cols is not None:
                # Use exactly these columns
                df_num = df[cols]
            else:
                # Default: keep numeric columns only
                if spec.get("numeric_only", True):
                    df_num = df.select_dtypes(include=["number"])
                else:
                    # Try to coerce everything except the obvious non-numeric
                    df_num = df.apply(pd.to_numeric, errors="coerce")

            A = df_num.to_numpy(dtype=np.float32, copy=False)
            if A.ndim == 1:
                A = A[None, :]
            return A.T if transpose else A

        else:
            raise ValueError(f"Unknown load.kind='{kind}'")
