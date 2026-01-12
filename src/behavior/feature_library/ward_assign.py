"""
WardAssignClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import fcluster

from behavior.dataset import register_feature
from .helpers import _build_path_sequence_map
from behavior.dataset import _resolve_inputs, _feature_index_path, _feature_run_root
from behavior.helpers import to_safe_name


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
        self._sequence_label_store: dict[str, dict[str, Any]] = {}
        self._allowed_safe_sequences: Optional[set[str]] = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._scope_filter: Optional[dict] = None
        self._inputs_overridden = bool(params and "inputs" in params)
        self._inputs_meta: Dict[str, Any] = {}

    def bind_dataset(self, ds):
        self._ds = ds

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

    def _feature_path_metadata(self, feature_name: str, run_id: str) -> dict[Path, dict[str, str]]:
        mapping: dict[Path, dict[str, str]] = {}
        if self._ds is None:
            return mapping
        idx_path = _feature_index_path(self._ds, feature_name)
        if not idx_path.exists():
            return mapping
        try:
            df = pd.read_csv(idx_path)
        except Exception:
            return mapping
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            return mapping
        df["group"] = df["group"].fillna("").astype(str)
        df["sequence"] = df["sequence"].fillna("").astype(str)
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].apply(lambda v: to_safe_name(v) if v else "")

        for _, row in df.iterrows():
            abs_raw = row.get("abs_path")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            try:
                abs_path = Path(abs_raw).resolve()
            except Exception:
                abs_path = Path(abs_raw)
            mapping[abs_path] = {
                "group": str(row.get("group", "") or ""),
                "sequence": str(row.get("sequence", "") or ""),
                "sequence_safe": str(row.get("sequence_safe", "") or ""),
            }
        return mapping

    def _load_one_general(self, path: Path, spec: dict) -> tuple[np.ndarray, Optional[np.ndarray]]:
        frame_vals: Optional[np.ndarray] = None
        kind = str(spec.get("kind", "parquet")).lower()
        if kind == "npz":
            key = spec.get("key")
            npz = np.load(path, allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"[ward-assign] Key '{key}' missing in {path.name}")
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            return A.astype(np.float32, copy=False), None
        elif kind == "parquet":
            df = pd.read_parquet(path)
            frame_col = spec.get("frame_column")
            if frame_col is None:
                frame_col = "frame"
            if frame_col and frame_col in df.columns:
                try:
                    frame_vals = df[frame_col].to_numpy(dtype=np.int64, copy=False)
                except Exception:
                    frame_vals = df[frame_col].to_numpy(copy=False)
                drop_frame = bool(spec.get("drop_frame_column", False))
                if drop_frame and not (spec.get("columns") and frame_col in spec.get("columns", [])):
                    df = df.drop(columns=[frame_col])
            drop_cols = spec.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            cols = spec.get("columns")
            if cols:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return np.empty((0, 0), dtype=np.float32)
                df = df[use]
            elif spec.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if spec.get("transpose"):
                arr = arr.T
            return arr, frame_vals
        else:
            raise ValueError(f"[ward-assign] Unknown load.kind='{kind}'")

    def _infer_sequence(self, df: pd.DataFrame) -> str:
        if "sequence" in df.columns and not df["sequence"].empty:
            val = str(df["sequence"].iloc[0])
            if val:
                return val
        if "sequence_safe" in df.columns and not df["sequence_safe"].empty:
            safe = str(df["sequence_safe"].iloc[0])
            if safe:
                pair = self._pair_map.get(safe)
                return pair[1] if pair else safe
        raise ValueError("[ward-assign] Input DataFrame missing 'sequence' column.")

    def _collect_sequence_inputs(self) -> dict[str, dict[str, Any]]:
        if not self._inputs:
            return {}
        n_inputs = len(self._inputs)
        per_seq: dict[str, dict[str, Any]] = {}
        allowed_safe = set(self._allowed_safe_sequences or [])
        enforce_scope = bool(allowed_safe)

        for idx, spec in enumerate(self._inputs):
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            resolved_run_id, run_root = self._resolve_feature_run_root(feat_name, run_id)
            pattern = spec.get("pattern", "*.parquet")
            files = sorted(run_root.glob(pattern))
            if not files:
                print(f"[ward-assign] WARN: no files for {feat_name} ({resolved_run_id}) pattern={pattern}", file=sys.stderr)
                continue
            seq_map = _build_path_sequence_map(self._ds, feat_name, resolved_run_id)
            meta_map = self._feature_path_metadata(feat_name, resolved_run_id)
            load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
            for pth in files:
                try:
                    abs_path = pth.resolve()
                except Exception:
                    abs_path = pth
                safe_seq = seq_map.get(abs_path)
                if not safe_seq:
                    safe_seq = to_safe_name(pth.stem)
                safe_seq = str(safe_seq)
                if enforce_scope and safe_seq not in allowed_safe:
                    continue
                arr, frame_vals = self._load_one_general(pth, load_spec)
                if arr is None or arr.size == 0:
                    continue
                entry = per_seq.setdefault(safe_seq, {
                    "mats": [None] * n_inputs,
                    "frames": None,
                    "sequence": None,
                    "group": None,
                })
                entry["mats"][idx] = arr
                meta = meta_map.get(abs_path, {})
                if entry["sequence"] is None:
                    entry["sequence"] = meta.get("sequence") or self._pair_map.get(safe_seq, ("", safe_seq))[1]
                if entry["group"] is None:
                    entry["group"] = meta.get("group") or self._pair_map.get(safe_seq, ("", ""))[0]
                if entry["frames"] is None and frame_vals is not None:
                    entry["frames"] = frame_vals

        clean = {}
        for safe_seq, payload in per_seq.items():
            mats = payload["mats"]
            if any((m is None or m.size == 0) for m in mats):
                print(f"[ward-assign] WARN: missing inputs for sequence '{safe_seq}', skipping.", file=sys.stderr)
                continue
            clean[safe_seq] = payload
        return clean

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
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

        self._sequence_label_store.clear()

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

        # optional scaler
        self._scaler = self._load_scaler(self.params.get("scaler"))

        seq_payloads = self._collect_sequence_inputs()
        if not seq_payloads:
            raise RuntimeError("[ward-assign] No usable inputs found for assignment.")

        for safe_seq, payload in seq_payloads.items():
            mats = payload["mats"]
            lengths = [m.shape[0] for m in mats if m is not None]
            if not lengths:
                continue
            T_min = min(lengths)
            mats_trim = [m[:T_min] for m in mats]
            X_full = np.hstack(mats_trim)
            if self._scaler is not None:
                if not hasattr(self._scaler, "transform"):
                    raise ValueError("[ward-assign] scaler object missing transform().")
                X_use = self._scaler.transform(X_full)
            else:
                X_use = X_full
            idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
            labels = self._cluster_ids[idxs.ravel()]
            frames = payload.get("frames")
            if frames is None or len(frames) < T_min:
                frames = np.arange(T_min, dtype=int)
            else:
                frames = frames[:T_min]
            seq_name = payload.get("sequence") or self._pair_map.get(safe_seq, ("", safe_seq))[1]
            self._sequence_label_store[safe_seq] = {
                "sequence": seq_name,
                "frames": frames.astype(int, copy=False),
                "labels": labels.astype(int, copy=False),
            }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._sequence_label_store:
            return pd.DataFrame(index=[])
        seq = self._infer_sequence(df)
        safe_seq = to_safe_name(seq)
        bundle = self._sequence_label_store.get(safe_seq)
        if not bundle:
            return pd.DataFrame(index=[])
        out = pd.DataFrame({
            "frame": bundle["frames"],
            "cluster": bundle["labels"],
            "sequence": seq,
        })
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        joblib.dump({
            "params": self.params,
            "n_clusters": int(self.params.get("n_clusters", 20)),
        }, path)
        for safe_seq, bundle in self._sequence_label_store.items():
            labels = bundle.get("labels")
            if labels is None:
                continue
            np.savez_compressed(run_root / f"global_ward_labels_seq={safe_seq}.npz", labels=labels)


#### VISUALIZATION
