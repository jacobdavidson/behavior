"""
GlobalKMeansClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
import re

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from behavior.dataset import register_feature
from .helpers import _build_path_sequence_map, _load_array_from_spec
from behavior.dataset import _resolve_inputs, _feature_index_path, _feature_run_root
from behavior.helpers import to_safe_name


@register_feature
class GlobalKMeansClustering:
    """
    Global K-Means that fits **only** on an artifact matrix resolved from the dataset's
    feature index (no frame-stream fitting). This avoids fit/transform shape mismatch.

    params:
      k: int
      random_state: int
      n_init: int|"auto"
      max_iter: int
      artifact: {
        feature: "global-tsne" | "pair-egocentric" | ...   # required
        run_id: str | None                                 # None => latest
        pattern: str                                       # files inside run root
        load: {                                            # loader spec
          kind: "npz"|"parquet",
          key: str               # for npz
          transpose: bool,       # optional
          columns: [str]|None,   # for parquet; if None -> numeric_only
          drop_columns: [str]|None,
          numeric_only: bool,    # default True for parquet
        }
      }

    Optional assign block:
      assign: {
        scaler: {
          feature: <prior feature name>,      # e.g. "global-tsne"
          run_id: str | None,                # None => latest
          pattern: str,                      # e.g. "global_opentsne_embedding.joblib"
          key: str | None,                   # key to extract from joblib, or None for full object
        },
        inputs: [                           # multi-input spec, same as GlobalTSNE.inputs
          {
            feature: <prior feature name>,
            run_id: str | None,
            pattern: str,
            load: {kind, key, transpose, columns, ...}
          },
          ...
        ]
      }
    If assign is specified, after fitting KMeans, clusters are assigned to full-frame features
    using the provided scaler and multi-inputs. Assigned labels per sequence are saved as
    global_kmeans_labels_seq=<safe_seq>.npz in the run folder.

    Notes:
      • The `assign.scaler` entry is optional. If omitted, assignment is performed in the raw feature space.
        In that case, the concatenated input dimensionality must exactly match the dimensionality used to fit K-Means.

    Fit-time artifacts saved in run folder:
      - kmeans.joblib
      - artifact_meta.json (fit provenance + columns + feature_dim)
      - cluster_centers.npy
      - artifact_labels.npz (labels over the artifact matrix, if computed)
      - cluster_sizes.csv (label,count) if labels computed
      - (if assign) global_kmeans_labels_seq=<safe_seq>.npz (per key)
    """

    name: str = "global-kmeans"
    version: str = "0.3"
    output_type: str = "global"

    def __init__(self, params: dict | None = None):
        self.params = {
            "k": 100,
            "random_state": 42,
            "n_init": "auto",
            "max_iter": 300,
            "artifact": {
                "feature": None,
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            # optional: compute labels on the artifact used for fitting
            "label_artifact_points": True,
            # optional: assign block for assigning clusters to full-frame features
            "assign": None,
        }
        if params:
            self.params.update(params)

        self._ds = None
        self._kmeans = None
        self._fit_dim = None              # int
        self._fit_columns = None          # list[str] when parquet
        self._fit_artifact_info = None    # dict saved to JSON
        self._artifact_labels = None      # np.ndarray[int] (optional)
        self._assign_labels = {}          # dict[safe_seq] = labels (int32)
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        assign_config = self.params.get("assign") or {}
        self._assign_inputs_override = "inputs" in assign_config
        self._assign_inputs_meta: Dict[str, Any] = {}
        self._scope_constraints: Optional[dict[str, set[str]]] = None

    def _load_one_general(self, path: Path, spec: dict) -> np.ndarray:
        arr = _load_array_from_spec(path, spec)
        if arr is None:
            return np.empty((0, 0), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"_load_one_general: Loaded array is not 2D: {arr.shape}")
        return arr

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key_from_path(self, path: Path, seq_map: dict[Path, str]) -> str:
        """
        Extract safe_seq from file path using naming convention or dataset index fallback.
        """
        stem = path.stem
        m_seq = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
        if m_seq:
            safe_seq = m_seq.group(1)
        else:
            safe_seq = seq_map.get(path.resolve())
        if not safe_seq:
            safe_seq = to_safe_name(stem)
        return str(safe_seq)

    def _collect_inputs_per_key(self, inputs: list[dict]) -> dict[str, list[np.ndarray]]:
        """
        For each input spec, glob pattern, load each file as matrix, group by sequence.
        Returns dict[safe_seq] -> list[np.ndarray]
        """
        per_key_parts = {}
        scope_active = bool(self._scope_constraints)
        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            resolved_run_id, run_root = self._resolve_feature_run_root(feat_name, run_id)
            pattern = spec.get("pattern", "*.npz")
            load_spec = spec.get("load", {"kind": "npz", "key": "spectrogram", "transpose": True})
            files = sorted(run_root.glob(pattern))
            seq_map = self._feature_seq_map(feat_name, resolved_run_id)
            meta_map = self._feature_path_metadata(feat_name, resolved_run_id) if scope_active else {}
            for pth in files:
                try:
                    abs_path = pth.resolve()
                except Exception:
                    abs_path = pth
                meta = meta_map.get(abs_path, {})
                group_val = str(meta.get("group", "") or "")
                seq_val = str(meta.get("sequence", "") or "")
                safe_seq = str(meta.get("sequence_safe", "") or "")
                if not safe_seq:
                    safe_seq = self._extract_key_from_path(pth, seq_map)
                safe_group = to_safe_name(group_val) if group_val else self._extract_safe_group_from_path(pth)
                if not self._is_scope_allowed(group_val, seq_val, safe_seq, safe_group):
                    continue
                key = safe_seq
                arr = self._load_one_general(pth, load_spec)
                if arr is None or arr.size == 0:
                    continue
                if key not in per_key_parts:
                    per_key_parts[key] = []
                per_key_parts[key].append(arr)
        return per_key_parts

    def set_scope_constraints(self, scope: Optional[dict]) -> None:
        """
        Capture dataset-level group/sequence filters so assignment can respect them.
        """
        if not scope:
            self._scope_constraints = None
            return

        def _to_norm_set(values: Any) -> Optional[set[str]]:
            if not values:
                return None
            out = {str(v) for v in values if isinstance(v, (str, int)) or v}
            out = {v for v in out if v}
            return out or None

        groups = _to_norm_set(scope.get("groups"))
        safe_groups = _to_norm_set(scope.get("safe_groups"))
        if groups and not safe_groups:
            safe_groups = {to_safe_name(g) for g in groups}

        sequences = _to_norm_set(scope.get("sequences"))
        safe_sequences = _to_norm_set(scope.get("safe_sequences"))
        if sequences and not safe_sequences:
            safe_sequences = {to_safe_name(s) for s in sequences}

        scoped = {
            k: v for k, v in {
                "groups": groups,
                "safe_groups": safe_groups,
                "sequences": sequences,
                "safe_sequences": safe_sequences,
            }.items() if v
        }
        self._scope_constraints = scoped or None

    def _feature_path_metadata(self, feature_name: str, run_id: str) -> dict[Path, dict[str, str]]:
        """
        Returns {abs_path -> {group, sequence, sequence_safe}} for a prior feature run.
        """
        mapping: dict[Path, dict[str, str]] = {}
        if self._ds is None:
            return mapping
        try:
            idx_path = _feature_index_path(self._ds, feature_name)
        except Exception:
            return mapping
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

    def _extract_safe_group_from_path(self, path: Path) -> str:
        stem = path.stem
        if "__" in stem:
            return stem.split("__", 1)[0]
        return ""

    def _is_scope_allowed(self,
                          group: str,
                          sequence: str,
                          safe_sequence: str,
                          safe_group: Optional[str]) -> bool:
        if not self._scope_constraints:
            return True
        scope = self._scope_constraints
        group = group or ""
        sequence = sequence or ""
        safe_sequence = safe_sequence or ""
        safe_group = safe_group or ""
        if scope.get("groups") and group not in scope["groups"]:
            return False
        if scope.get("safe_groups") and safe_group not in scope["safe_groups"]:
            return False
        if scope.get("sequences") and sequence not in scope["sequences"]:
            return False
        if scope.get("safe_sequences") and safe_sequence not in scope["safe_sequences"]:
            return False
        return True

    def _load_scaler(self, spec: dict):
        """
        Loads a scaler (e.g. sklearn StandardScaler) from a joblib file.
        spec: {
          feature: str,
          run_id: str|None,
          pattern: str (default: "global_opentsne_embedding.joblib"),
          key: str|None,
        }
        Returns: scaler object, or obj[key] if key is not None.
        """
        feat_name = spec["feature"]
        run_id = spec.get("run_id")
        _, run_root = self._resolve_feature_run_root(feat_name, run_id)
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"_load_scaler: No files matching '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        return obj if key is None else obj[key]

    # Dataset binding
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def bind_dataset(self, ds): self._ds = ds

    # ----------------- Fit helpers -----------------
    def _resolve_feature_run_root(self, feature_name: str, run_id: str | None) -> tuple[str, Path]:
        froot = self._ds.get_root("features") / feature_name
        idx_path = froot / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feature_name}' at {idx_path}")
        idx = pd.read_csv(idx_path)
        if run_id is None:
            # pick latest finished if present; else started_at
            if "finished_at" in idx.columns:
                idx = idx.sort_values("finished_at", ascending=False, na_position="last")
            elif "started_at" in idx.columns:
                idx = idx.sort_values("started_at", ascending=False, na_position="last")
            if idx.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(idx.iloc[0]["run_id"])
        resolved = str(run_id)
        run_root = froot / resolved
        if not run_root.exists():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        return resolved, run_root

    def _load_npz_matrix(self, files: list[Path], key: str, transpose: bool) -> tuple[np.ndarray, dict]:
        mats = []
        for p in files:
            npz = np.load(p, allow_pickle=True)
            if key not in npz.files:
                # skip silently; some *.npz may not contain the requested key
                continue
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            A = A.T if transpose else A
            mats.append(A)
        if not mats:
            raise FileNotFoundError(f"No NPZ containing key '{key}' among: {[p.name for p in files]}")
        X = np.vstack(mats)
        meta = {"loader_kind": "npz", "key": key, "transpose": bool(transpose)}
        return X, meta

    def _load_parquet_matrix(self, files: list[Path], spec: dict) -> tuple[np.ndarray, dict, list[str]]:
        cols = spec.get("columns")
        drop_cols = set(spec.get("drop_columns", []))
        numeric_only = bool(spec.get("numeric_only", True))

        def load_df(p: Path) -> pd.DataFrame:
            df = pd.read_parquet(p)
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")
            if cols is not None:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return pd.DataFrame()
                df = df[use]
            else:
                df = df.select_dtypes(include=["number"]) if numeric_only else df.apply(pd.to_numeric, errors="coerce")
            return df

        dfs = []
        first_cols = None
        for p in files:
            df = load_df(p)
            if df.empty:
                continue
            if first_cols is None:
                first_cols = df.columns.tolist()
            else:
                # align to first set of columns
                df = df.reindex(columns=first_cols)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No Parquet files with usable numeric columns among: {[p.name for p in files]}")

        D = pd.concat(dfs, ignore_index=True)
        A = D.to_numpy(dtype=np.float32, copy=False)
        meta = {
            "loader_kind": "parquet",
            "columns": first_cols,
            "drop_columns": list(drop_cols) if drop_cols else [],
            "numeric_only": numeric_only,
        }
        return A, meta, first_cols

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.get("artifact") or {}
        feature = art.get("feature")
        if not feature:
            raise ValueError("GlobalKMeansClustering: params['artifact']['feature'] is required.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, art.get("run_id"))
        pattern = art.get("pattern", "*.npz")
        loader  = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {run_root}")

        kind = str(loader.get("kind", "npz")).lower()
        self._fit_columns = None
        if kind == "npz":
            key = loader.get("key")
            if not key:
                raise ValueError("artifact.load.kind='npz' requires 'key'")
            X, meta = self._load_npz_matrix(files, key, bool(loader.get("transpose", False)))
        elif kind == "parquet":
            X, meta, cols = self._load_parquet_matrix(files, loader)
            self._fit_columns = cols
        else:
            raise ValueError(f"Unknown artifact load.kind='{kind}'")

        # for provenance
        self._fit_artifact_info = {
            "feature": feature,
            "run_id": resolved_run_id,
            "run_root": str(run_root),
            "pattern": pattern,
            "loader": meta,
        }
        return X

    # ----------------- Fit / Transform / Save -----------------
    def fit(self, _X_iter_unused: Iterable[pd.DataFrame]) -> None:
        # Fit purely from artifact
        X = self._load_artifact_matrix()
        self._fit_dim = int(X.shape[1])

        if X.shape[0] < int(self.params["k"]):
            raise ValueError(f"Not enough samples to fit KMeans: n={X.shape[0]} < k={self.params['k']}")

        self._kmeans = KMeans(
            n_clusters=int(self.params["k"]),
            n_init=self.params.get("n_init", "auto"),
            random_state=self.params.get("random_state", 42),
            max_iter=int(self.params.get("max_iter", 300)),
        ).fit(X)

        # Optional: label the artifact points used for training (nice to have)
        if bool(self.params.get("label_artifact_points", True)):
            self._artifact_labels = self._kmeans.predict(X)

        # Optional: assign clusters to full-frame features using assign block
        assign = self.params.get("assign")
        if assign:
            # Optional scaler: if provided, standardize before prediction; else predict in raw space.
            scaler = None
            if "scaler" in assign and assign["scaler"]:
                scaler = self._load_scaler(assign["scaler"])

            assign_inputset = assign.get("inputset")
            explicit_assign_inputs = assign.get("inputs") if (self._assign_inputs_override or not assign_inputset) else None
            resolved_assign_inputs, assign_inputs_meta = _resolve_inputs(
                self._ds,
                explicit_assign_inputs,
                assign_inputset,
                explicit_override=self._assign_inputs_override,
            )
            self._assign_inputs_meta = assign_inputs_meta

            # Collect per-key inputs (may be empty)
            per_key_parts = self._collect_inputs_per_key(resolved_assign_inputs)
            for key, mats in per_key_parts.items():
                if not mats:
                    continue
                T_min = min(m.shape[0] for m in mats)
                mats_trim = [m[:T_min] for m in mats]
                X_full = np.hstack(mats_trim)
                D_total = X_full.shape[1]

                if scaler is not None:
                    # Validate scaler dimensionality
                    if not hasattr(scaler, "n_features_in_"):
                        raise ValueError("Scaler object must have n_features_in_ (e.g. sklearn StandardScaler)")
                    if scaler.n_features_in_ != D_total:
                        raise ValueError(
                            f"Scaler expects n_features_in_={getattr(scaler, 'n_features_in_', None)}, "
                            f"got {D_total} columns for key={key}"
                        )
                    X_use = scaler.transform(X_full)
                else:
                    # No scaler: require that assign inputs match the KMeans fit dimensionality.
                    if self._fit_dim is None:
                        raise RuntimeError("GlobalKMeansClustering internal error: fit_dim is None before assignment.")
                    if D_total != self._fit_dim:
                        raise ValueError(
                            f"Assign-without-scaler requires feature dim {self._fit_dim}, "
                            f"but got {D_total} for key={key}. Provide a scaler or align inputs."
                        )
                    X_use = X_full

                labels = self._kmeans.predict(X_use)
                self._assign_labels[key] = labels.astype(np.int32)
            if self._fit_artifact_info is None:
                self._fit_artifact_info = {}
            self._fit_artifact_info["assign_inputs_meta"] = assign_inputs_meta

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Assign clusters to an input DataFrame **only if** its columns can be aligned to the
        fitted feature space:
          - If we fitted on parquet with explicit/numeric columns, require those columns.
          - Otherwise, require numeric matrix with matching dimensionality.
        If alignment fails, return an **empty** DataFrame (no error).
        """
        if self._kmeans is None or self._fit_dim is None:
            raise RuntimeError("GlobalKMeansClustering not fitted yet.")

        # Case 1: we know the exact columns from parquet fit
        if self._fit_columns:
            missing = [c for c in self._fit_columns if c not in X.columns]
            if missing:
                return pd.DataFrame(columns=["frame", "cluster"])  # silently skip
            A = X[self._fit_columns].to_numpy(dtype=np.float32, copy=False)
        else:
            # Case 2: generic numeric-only fallback requiring same dimensionality
            num = X.select_dtypes(include=["number"])
            if num.shape[1] != self._fit_dim:
                return pd.DataFrame(columns=["frame", "cluster"])  # silently skip
            A = num.to_numpy(dtype=np.float32, copy=False)

        mask = np.isfinite(A).all(axis=1)
        labels = np.full(A.shape[0], -1, dtype=np.int32)
        if mask.any():
            labels[mask] = self._kmeans.predict(A[mask])

        out = pd.DataFrame({
            "frame": X["frame"].astype(int, errors="ignore") if "frame" in X.columns else np.arange(len(X), dtype=int),
            "cluster": labels,
        })
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # 1) Save the fitted model
        joblib.dump({
            "kmeans": self._kmeans,
            "k": int(self.params["k"]),
            "random_state": self.params.get("random_state", 42),
            "n_init": self.params.get("n_init", "auto"),
            "max_iter": int(self.params.get("max_iter", 300)),
            "fit_dim": int(self._fit_dim or 0),
            "fit_columns": self._fit_columns,
            "artifact_info": self._fit_artifact_info,
            "version": self.version,
        }, path)

        # 2) Save human-usable artifacts
        centers = np.asarray(self._kmeans.cluster_centers_, dtype=np.float32)
        np.save(run_root / "cluster_centers.npy", centers)

        # template/artifact labels + counts (if computed)
        if self._artifact_labels is not None:
            np.savez_compressed(run_root / "artifact_labels.npz", labels=self._artifact_labels)
            # counts for quick inspection
            uniq, cnt = np.unique(self._artifact_labels, return_counts=True)
            pd.DataFrame({"cluster": uniq.astype(int), "count": cnt.astype(int)}) \
              .to_csv(run_root / "cluster_sizes.csv", index=False)

        # 3) Save assigned labels per key if assign block was used
        for safe_seq, labels in self._assign_labels.items():
            fname = f"global_kmeans_labels_seq={safe_seq}.npz"
            np.savez_compressed(run_root / fname, labels=labels)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._kmeans = bundle["kmeans"]
        self._fit_dim = int(bundle.get("fit_dim") or 0)
        self._fit_columns = bundle.get("fit_columns")
        self._fit_artifact_info = bundle.get("artifact_info", {})
