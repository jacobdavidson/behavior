"""
WardAssignClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import fcluster

from behavior.dataset import register_feature, _resolve_inputs
from .helpers import StreamingFeatureHelper


@register_feature
class WardAssignClustering:
    """
    Assign Ward clusters (cut at n_clusters) to full-frame feature streams by stacking
    multiple prior features (e.g., pair-wavelet social + ego) and reusing the scaler
    from GlobalTSNE.

    Params
    ------
    ward_model : dict
        { "feature": "global-ward__from__global-tsne",
          "run_id": None,
          "pattern": "model.joblib" }
    artifact : dict
        Same structure as GlobalWardClustering.artifact (used to reload the template matrix).
    scaler : optional dict
        Same contract as GlobalKMeans.assign.scaler (joblib w/ StandardScaler).
    inputs : list[dict]
        Feature specs to concatenate per sequence. All inputs are loaded from disk
        (typically resolved via an inputset) and aligned per sequence before assignment.
        To retain real frame indices in the outputs, set `load.frame_column` (defaults
        to "frame" when present) for at least one input.
    n_clusters : int
        Desired Ward cut.
    recalc : bool
        If True, force recomputation even if outputs already exist (pass overwrite=True to
        dataset.run_feature when rerunning). Defaults to False.
    """

    name = "ward-assign"
    version = "0.1"
    parallelizable = True
    output_type = "global"

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "ward_model": {
                "feature": "global-ward__from__global-tsne",
                "run_id": None,
                "pattern": "model.joblib",
            },
            "artifact": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            "scaler": None,
            "inputs": [],
            "inputset": None,
            "n_clusters": 20,
            "recalc": False,
        }
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    d = dict(self.params[k])
                    d.update(v)
                    self.params[k] = d
                else:
                    self.params[k] = v

        ward_feat_name = self.params["ward_model"].get("feature", "global-ward")
        self.storage_feature_name = f"ward-assign__from__{ward_feat_name}"
        self.storage_use_input_suffix = False
        self.skip_existing_outputs = True

        self._ds = None
        self._Z = None
        self._templates = None
        self._cluster_ids = None
        self._assign_nn = None
        self._scaler = None
        self._inputs = []
        self._processed_sequences: set[str] = set()  # Track which sequences were written
        self._run_root: Optional[Path] = None  # Set by dataset.run_feature before fit()
        self._allowed_safe_sequences: Optional[set[str]] = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._scope_filter: Optional[dict] = None
        self._inputs_overridden = bool(params and "inputs" in params)
        self._inputs_meta: Dict[str, Any] = {}

    def bind_dataset(self, ds):
        self._ds = ds

    def set_run_root(self, path: Path) -> None:
        """Set the output directory for immediate file writes during fit()."""
        self._run_root = path

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}
        safe_sequences = scope.get("safe_sequences") if scope else None
        if safe_sequences:
            self._allowed_safe_sequences = {str(s) for s in safe_sequences}
        else:
            self._allowed_safe_sequences = None
        pair_safe_map = scope.get("pair_safe_map") if scope else None
        if pair_safe_map:
            inv = {}
            for pair, safe in pair_safe_map.items():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                inv[str(safe)] = (str(pair[0]), str(pair[1]))
            self._pair_map = inv
        else:
            self._pair_map = {}

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def loads_own_data(self): return True  # Skip run_feature pre-loading; we load from artifacts
    def partial_fit(self, X): raise NotImplementedError

    # ---------- helpers ----------
    def _resolve_feature_run_root(self, feature_name: str, run_id: Optional[str]) -> tuple[str, Path]:
        froot = self._ds.get_root("features") / feature_name
        idx_path = froot / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feature_name}' at {idx_path}")
        df = pd.read_csv(idx_path)
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                      ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            if df.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(df.iloc[0]["run_id"])
        resolved = str(run_id)
        run_root = froot / resolved
        if not run_root.exists():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        return resolved, run_root

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.get("artifact", {})
        feature = art.get("feature")
        if not feature:
            raise ValueError("[ward-assign] artifact.feature required.")
        run_id, run_root = self._resolve_feature_run_root(feature, art.get("run_id"))
        pattern = art.get("pattern", "*.npz")
        loader = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No files matching '{pattern}' in {run_root}")
        kind = str(loader.get("kind", "npz")).lower()
        if kind == "npz":
            key = loader.get("key")
            npz = np.load(files[0], allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"[ward-assign] Key '{key}' missing in {files[0].name}")
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            return A.astype(np.float32, copy=False)
        elif kind == "parquet":
            df = pd.read_parquet(files[0])
            if loader.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
                # Drop metadata columns that are numeric but not features
                for mc in ("frame", "time", "id1", "id2"):
                    if mc in df.columns:
                        df = df.drop(columns=[mc])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            return df.to_numpy(dtype=np.float32, copy=False)
        else:
            raise ValueError(f"[ward-assign] Unsupported artifact load.kind='{kind}'")

    def _load_scaler(self, spec: Optional[dict]):
        if not spec:
            return None
        feat_name = spec.get("feature")
        if not feat_name:
            raise ValueError("[ward-assign] scaler.feature required.")
        _, run_root = self._resolve_feature_run_root(feat_name, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] Could not locate scaler '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        return obj if key is None else obj[key]

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        import gc
        import pyarrow as pa
        if self._ds is None:
            raise RuntimeError("[ward-assign] Dataset not bound. Use dataset.run_feature(...).")

        inputs = self.params.get("inputs") or []
        inputset_name = self.params.get("inputset")
        explicit_inputs = inputs if (self._inputs_overridden or not inputset_name) else None
        self._inputs, self._inputs_meta = _resolve_inputs(
            self._ds,
            explicit_inputs,
            inputset_name,
            explicit_override=self._inputs_overridden,
        )

        # Load Ward linkage
        ward_spec = self.params["ward_model"]
        _, ward_root = self._resolve_feature_run_root(ward_spec["feature"], ward_spec.get("run_id"))
        pattern = ward_spec.get("pattern", "model.joblib")
        files = sorted(ward_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No Ward model '{pattern}' in {ward_root}")
        bundle = joblib.load(files[0])
        self._Z = bundle.get("linkage_matrix")
        if self._Z is None:
            raise ValueError("[ward-assign] Ward model missing linkage_matrix.")

        # Reload artifact matrix (templates) to derive centroids
        self._templates = self._load_artifact_matrix()
        n_clusters = int(self.params.get("n_clusters", 20))
        labels_templates = fcluster(self._Z, n_clusters, criterion="maxclust")
        uniq = np.unique(labels_templates)
        centroids = []
        for cid in uniq:
            mask = labels_templates == cid
            if not mask.any():
                continue
            centroids.append(self._templates[mask].mean(axis=0))
        centroids = np.vstack(centroids)
        self._cluster_ids = uniq.astype(int)
        self._assign_nn = NearestNeighbors(n_neighbors=1).fit(centroids)

        # Free templates and linkage matrix - no longer needed after centroid computation
        del self._templates, self._Z, labels_templates, centroids
        self._templates = None
        self._Z = None
        gc.collect()

        # optional scaler
        self._scaler = self._load_scaler(self.params.get("scaler"))

        # Use StreamingFeatureHelper for manifest building and data loading
        helper = StreamingFeatureHelper(self._ds, "ward-assign")
        scope_filter = {"safe_sequences": self._allowed_safe_sequences} if self._allowed_safe_sequences else None
        manifest = helper.build_manifest(self._inputs, scope_filter=scope_filter)
        if not manifest:
            raise RuntimeError("[ward-assign] No usable inputs found for assignment.")

        # Process sequences one at a time using direct loading
        # (avoids generator pattern which holds extra reference to data)
        keys = list(manifest.keys())
        n_keys = len(keys)
        for i, safe_seq in enumerate(keys):
            X_full, frames = helper.load_key_data(manifest[safe_seq], extract_frames=True)
            if X_full is None:
                continue

            if self._scaler is not None:
                if not hasattr(self._scaler, "transform"):
                    raise ValueError("[ward-assign] scaler object missing transform().")
                X_use = self._scaler.transform(X_full)
                del X_full  # free raw data immediately
            else:
                X_use = X_full
                # Note: X_use IS X_full here, so don't delete

            idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
            labels = self._cluster_ids[idxs.ravel()]

            # Write immediately to disk instead of storing in memory
            if self._run_root is not None:
                out_path = self._run_root / f"global_ward_labels_seq={safe_seq}.npz"
                np.savez_compressed(out_path, labels=labels.astype(np.int32))
                self._processed_sequences.add(safe_seq)
            else:
                # Fallback: warn but don't crash (save_model will handle it)
                print(f"[ward-assign] WARN: _run_root not set, labels for {safe_seq} not written", file=sys.stderr)

            # Free memory after each sequence
            del X_use, idxs, labels
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % 10 == 0 or i == n_keys - 1:
                print(f"[ward-assign] Processed {i + 1}/{n_keys} sequences", file=sys.stderr)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Labels are written directly to disk during fit(); transform returns a stub
        if not self._processed_sequences:
            return pd.DataFrame(index=[])
        # Return a simple marker row (actual labels are in NPZ files)
        return pd.DataFrame({"ward_assign_done": [True]})

    def save_model(self, path: Path) -> None:
        # Labels are already written to disk during fit(); just save model params
        joblib.dump({
            "params": self.params,
            "n_clusters": int(self.params.get("n_clusters", 20)),
            "processed_sequences": list(self._processed_sequences),
        }, path)


#### VISUALIZATION
