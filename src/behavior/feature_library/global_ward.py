"""
GlobalWardClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import joblib

from scipy.cluster.hierarchy import linkage as _sch_linkage

from behavior.dataset import register_feature
from .helpers import _collect_sequence_blocks
from behavior.dataset import _resolve_inputs


@register_feature
class GlobalWardClustering:
    """
    Ward hierarchical clustering on a global feature artifact (e.g. global t-SNE templates).

    Params
    ------
    artifact : dict (required)
        {
          "feature": "global-tsne",            # feature that produced the artifact
          "run_id": None,                      # None => latest finished run
          "pattern": "global_templates_features.npz",
          "load": {"kind": "npz", "key": "templates", "transpose": False}
        }
    method : str = "ward"
        Linkage method (Ward requires Euclidean distances).
    """

    name    = "global-ward"
    version = "0.1"

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "artifact": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            "method": "ward",
        }
        if params:
            # shallow-merge
            self.params.update({k: v for k, v in params.items() if k != "artifact"})
            if "artifact" in params:
                a = dict(self.params["artifact"])
                a.update(params["artifact"])
                # nested load merge
                if "load" in params["artifact"]:
                    ld = dict(a.get("load", {}))
                    ld.update(params["artifact"]["load"])
                    a["load"] = ld
                self.params["artifact"] = a

        self._ds = None               # bound Dataset
        self._Z  = None               # linkage matrix (np.ndarray)
        self._X_shape = None          # (n_samples, n_features)
        self._marker_written = False  # ensure only one parquet marker row gets written
        self._artifact_inputs_meta: Dict[str, Any] = {}

    # ---------- framework API ----------
    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        # Ignore X_iter; we load from the declared artifact to avoid accidental wrong inputs
        X = self._load_artifact_matrix()
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(f"[global-ward] Need a 2D matrix with >=2 samples; got shape={X.shape}")
        method = str(self.params.get("method", "ward")).lower()
        if method != "ward":
            # You could allow other methods, but Ward is the intended one
            raise ValueError(f"[global-ward] Only 'ward' is supported here, got '{method}'.")

        # SciPy linkage expects samples as rows
        self._Z = _sch_linkage(X, method=method)
        self._X_shape = tuple(X.shape)

    def partial_fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        We don't produce per-(group,sequence) outputs. We emit a single 1-row marker
        the first time transform is called so the run is indexed; subsequent calls return
        an empty DF (so nothing else is written).
        """
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        ns, nf = (self._X_shape or (np.nan, np.nan))
        return pd.DataFrame([{
            "linkage_method": str(self.params.get("method", "ward")),
            "n_samples": int(ns) if ns == ns else -1,     # handle NaN
            "n_features": int(nf) if nf == nf else -1,
            "model_file": "model.joblib",
        }])

    def save_model(self, path: Path) -> None:
        """
        Persist the linkage and minimal provenance to model.joblib.
        Also write a human-usable .npz copy (optional).
        """
        if self._Z is None:
            # nothing fitted; skip
            return
        # Ensure parent exists and path is file path (run_feature passes a file path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "linkage_matrix": self._Z,
            "method": str(self.params.get("method", "ward")),
            "n_samples": None if self._X_shape is None else int(self._X_shape[0]),
            "n_features": None if self._X_shape is None else int(self._X_shape[1]),
            "version": self.version,
            "params": self.params,
        }, path)

        # Optional: also store as npz right next to model (convenient for quick numpy loads)
        np.savez_compressed(path.with_suffix(".npz"),
                            linkage_matrix=self._Z,
                            method=str(self.params.get("method", "ward")),
                            n_samples=(None if self._X_shape is None else int(self._X_shape[0])),
                            n_features=(None if self._X_shape is None else int(self._X_shape[1])))

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._Z = bundle.get("linkage_matrix", None)
        self._X_shape = (bundle.get("n_samples", None), bundle.get("n_features", None))

    def _load_artifact_matrix(self) -> np.ndarray:
        """
        Resolve the artifact (feature/run_id), glob the pattern, and load to a single (N,D) matrix.
        """
        if self._ds is None:
            raise RuntimeError("[global-ward] Feature not bound to a Dataset; call via dataset.run_feature(...)")

        art = self.params.get("artifact", {})
        inputset_name = art.get("inputset")
        explicit_inputs = art.get("inputs") if ("inputs" in art) else None
        if inputset_name or explicit_inputs:
            specs, meta = _resolve_inputs(
                self._ds,
                explicit_inputs,
                inputset_name,
                explicit_override=("inputs" in art),
            )
            blocks = _collect_sequence_blocks(self._ds, specs)
            if not blocks:
                raise RuntimeError("[global-ward] Inputset produced no usable matrices.")
            X = np.vstack(list(blocks.values()))
            if X.ndim != 2:
                raise ValueError(f"[global-ward] Loaded array must be 2D; got shape={X.shape}")
            self._artifact_inputs_meta = meta
            return X.astype(np.float64, copy=False)

        feat_name = art.get("feature", None)
        run_id    = art.get("run_id", None)
        pattern   = art.get("pattern", None)
        load_spec = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})

        if not feat_name or not pattern:
            raise ValueError("[global-ward] 'artifact.feature' and 'artifact.pattern' are required in params.")

        idx_path = self._ds.get_root("features") / feat_name / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"[global-ward] No index for feature '{feat_name}'. Expected: {idx_path}")

        df_idx = pd.read_csv(idx_path)
        if run_id is None:
            if "finished_at" in df_idx.columns:
                cand = df_idx[df_idx["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df_idx
                base = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                        ascending=False, kind="stable")
            else:
                base = df_idx.sort_values(by=["started_at"], ascending=False, kind="stable")
            if base.empty:
                raise ValueError(f"[global-ward] No runs found for feature '{feat_name}'.")
            run_id = str(base.iloc[0]["run_id"])

        run_root = self._ds.get_root("features") / feat_name / run_id
        if not run_root.exists():
            raise FileNotFoundError(f"[global-ward] run root not found: {run_root}")

        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[global-ward] No files matching '{pattern}' in {run_root}")

        mats = []
        for fp in files:
            kind = (load_spec.get("kind") or "npz").lower()
            transpose = bool(load_spec.get("transpose", False))
            if kind == "npz":
                key = load_spec.get("key", None)
                if key is None:
                    raise ValueError("[global-ward] 'load.key' required for npz")
                npz = np.load(fp, allow_pickle=True)
                if key not in npz.files:
                    raise KeyError(f"[global-ward] Key '{key}' not found in {fp.name}")
                A = np.asarray(npz[key])
            elif kind == "parquet":
                df = pd.read_parquet(fp)
                cols = load_spec.get("columns", None)
                if cols:
                    A = df[cols].to_numpy(dtype=np.float32)
                else:
                    if load_spec.get("numeric_only", True):
                        A = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                    else:
                        A = df.to_numpy(dtype=np.float32)
            else:
                raise ValueError(f"[global-ward] Unsupported load.kind='{kind}'")

            if A.ndim == 1:
                A = A[None, :]
            if transpose:
                A = A.T
            mats.append(A.astype(np.float64, copy=False))

        if not mats:
            raise RuntimeError(f"[global-ward] No matrices loaded from {run_root} with pattern {pattern}")

        X = mats[0] if len(mats) == 1 else np.vstack(mats)
        if X.ndim != 2:
            raise ValueError(f"[global-ward] Loaded array must be 2D; got {X.shape} from {files[0].name}")
        return X
