"""
VizGlobalColored feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
import re
import sys
import fnmatch

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from behavior.dataset import register_feature
from .helpers import _build_path_sequence_map
from behavior.helpers import to_safe_name

# Sentinel value to indicate label_key_regex should inherit from coord_key_regex
INHERIT_REGEX = object()


@register_feature
class VizGlobalColored:
    """
    Generic wrapper to visualize any global embedding (t-SNE, PCA, UMAP, etc.)
    colored by arbitrary labels.

    Params (defaults target the global t-SNE + k-means workflow):
      coords: {feature, run_id, pattern, load:{kind:"npz", key:"Y"}}
      labels: optional spec matching coords (same structure as coords) OR
              {"source": "labels", "kind": "<label_kind>", "load": {...}}
              to load from labels/<kind>/index.csv (e.g., CalMS21 ground truth)
      coord_key_regex: regex applied to coord filenames (default extracts the sequence name)
      label_key_regex: regex for label filenames (defaults to coord regex)
      label_missing_value: sentinel for missing labels (default -1 -> gray)
      label_order: optional list of label values to fix the color ordering
      label_name_map: optional dict mapping label ids -> display names
      plot_max: max points to scatter
      palette: seaborn palette name or list
      title: plot title
      point_size / point_alpha: scatter style tweaks
      debug_save_arrays: if True, save Y_all/L_all (aligned, pre-subsample) as debug_viz_arrays.npz
    Output:
      - PNG in run_root plus a single marker parquet row for indexing.
    """

    name = "viz-global-colored"
    version = "0.1"

    def __init__(self, params=None):
        defaults = {
            "coords": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_tsne_coords_seq=*.npz",
                "load": {"kind": "npz", "key": "Y", "transpose": False},
            },
            "labels": None,
            "coord_key_regex": r"seq=(.+?)(?:_persp=.*)?$",
            "label_key_regex": INHERIT_REGEX,
            "label_missing_value": -1,
            "label_order": None,
            "label_name_map": None,
            "plot_max": 300_000,
            "palette": "tab20",
            "title": "Global embedding colored scatter",
            "point_size": 2.0,
            "point_alpha": 0.35,
            "debug_save_arrays": False,
        }
        self.params = dict(defaults)
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    merged = dict(self.params[k])
                    merged.update(v)
                    self.params[k] = merged
                else:
                    self.params[k] = v
        self._ds = None
        self._figs = []
        self._marker_written = False
        self._summary = {}
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        self._scope_constraints: Optional[dict] = None
        self._debug_arrays: Optional[dict] = None

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_constraints(self, scope: Optional[dict]) -> None:
        self._scope_constraints = scope or {}

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def partial_fit(self, X): raise NotImplementedError
    def finalize_fit(self): pass

    def _load_artifacts_glob(self, spec):
        ds = self._ds
        idx = ds.get_root("features") / spec["feature"] / "index.csv"
        df = pd.read_csv(idx)
        run_id = spec.get("run_id")
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(by=["finished_at" if len(cand) else "started_at"], ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            run_id = str(df.iloc[0]["run_id"])
        run_root = ds.get_root("features") / spec["feature"] / run_id
        files = sorted(run_root.glob(spec["pattern"]))
        return run_id, run_root, files

    def _load_one(self, path, load_spec):
        kind = load_spec.get("kind", "npz").lower()
        if kind == "npz":
            npz = np.load(path, allow_pickle=True)
            A = np.asarray(npz[load_spec["key"]])
            if load_spec.get("transpose"):
                A = A.T
            return A
        if kind == "parquet":
            df = pd.read_parquet(path)
            drop_cols = load_spec.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            cols = load_spec.get("columns")
            if cols:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return np.empty((0, len(cols)), dtype=np.float32)
                df = df[use]
            elif load_spec.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                # coerce everything numeric-style to avoid object columns
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if load_spec.get("transpose"):
                arr = arr.T
            return arr
        if kind == "joblib":
            obj = joblib.load(path)
            key = load_spec.get("key")
            return obj if key is None else obj[key]
        raise ValueError(f"Unsupported load.kind='{kind}'")

    def _load_labels_from_index(self, spec: dict) -> list[tuple[str, Path]]:
        """
        Load (sequence_safe, path) pairs from labels/<kind>/index.csv.
        Supports optional glob-style filtering via spec['pattern'] on filename.
        """
        if self._ds is None:
            raise RuntimeError("VizGlobalColored requires dataset binding to load labels.")
        kind = spec.get("kind")
        if not kind:
            raise ValueError("labels spec with source='labels' requires 'kind'.")
        labels_root = self._ds.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Labels index not found: {idx_path}")
        df = pd.read_csv(idx_path)
        pattern = spec.get("pattern")
        entries: list[tuple[str, Path]] = []
        for _, row in df.iterrows():
            abs_raw = row.get("abs_path", "")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            pth = Path(abs_raw)
            if pattern and not fnmatch.fnmatch(pth.name, pattern):
                continue
            seq_safe = str(row.get("sequence_safe") or row.get("sequence") or "").strip()
            if not seq_safe:
                seq_safe = to_safe_name(pth.stem)
            entries.append((seq_safe, pth))
        return entries

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key(self, path: Path, regex: Optional[str], seq_map: dict[Path, str]):
        stem = path.stem
        m = re.search(regex, stem) if regex else None
        if not m:
            safe_seq = seq_map.get(path.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(stem)
            return str(safe_seq)
        if m.lastindex is None:
            return m.group(0)
        if m.lastindex == 1:
            return m.group(1)
        return tuple(m.groups())

    def _prepare_color_map(self, labels):
        unique_vals = list(pd.unique(pd.Series(labels, dtype="object")))
        missing = self.params.get("label_missing_value", -1)
        order = self.params.get("label_order")
        if order:
            order_list = [o for o in order if o in unique_vals]
            unique_vals = order_list + [u for u in unique_vals if u not in order_list]

        palette = sns.color_palette(self.params.get("palette", "tab20"),
                                    max(1, len([u for u in unique_vals if u != missing])))
        color_map = {}
        idx = 0
        for val in unique_vals:
            if val == missing:
                continue
            color_map[val] = palette[idx % len(palette)]
            idx += 1
        if missing in unique_vals:
            color_map[missing] = (0.7, 0.7, 0.7)
        return color_map

    def _label_display(self, val):
        name_map = self.params.get("label_name_map") or {}
        if isinstance(name_map, dict):
            if val in name_map:
                return str(name_map[val])
            try:
                ival = int(val)
                if ival in name_map:
                    return str(name_map[ival])
            except Exception:
                pass
        return str(val)

    def fit(self, X_iter):
        coord_spec = self.params["coords"]
        coord_run_id, _, coord_files = self._load_artifacts_glob(coord_spec)
        if not coord_files:
            raise FileNotFoundError("[viz-global-colored] No coordinate files found.")

        scope = self._scope_constraints or {}
        allowed_safe = set(scope.get("safe_sequences") or [])
        allowed_sequences = set(scope.get("sequences") or [])
        allowed_any = set()
        if allowed_sequences:
            allowed_any.update(allowed_sequences)
            allowed_any.update(to_safe_name(s) for s in allowed_sequences)
        allowed_any.update(allowed_safe)

        key_regex = self.params.get("coord_key_regex")
        coord_seq_map = self._feature_seq_map(coord_spec["feature"], coord_run_id)
        Y_list, key_list, n_list = [], [], []
        for f in coord_files:
            key = self._extract_key(f, key_regex, coord_seq_map)
            key_safe = to_safe_name(key)
            if allowed_any and key not in allowed_any and key_safe not in allowed_any:
                continue
            arr = self._load_one(f, coord_spec["load"])
            if arr.ndim != 2:
                arr = np.atleast_2d(arr)
            Y_list.append(arr)
            key_list.append(key)
            n_list.append(arr.shape[0])
        if not Y_list:
            raise RuntimeError("No coordinate arrays loaded.")

        labels_spec = self.params.get("labels")
        missing_value = self.params.get("label_missing_value", -1)
        if labels_spec:
            label_regex_param = self.params.get("label_key_regex", INHERIT_REGEX)
            label_regex = key_regex if label_regex_param == INHERIT_REGEX else label_regex_param
            label_source = str(labels_spec.get("source", "feature")).lower()
            label_load_spec = labels_spec.get("load") or {"kind": "npz", "key": "labels"}
            lab_map: dict[Any, np.ndarray] = {}
            if label_source == "labels":
                entries = self._load_labels_from_index(labels_spec)
                if allowed_any:
                    entries = [(s, p) for s, p in entries if s in allowed_any]
                seq_map = {pth.resolve(): safe_seq for safe_seq, pth in entries}
                for safe_seq, pth in entries:
                    lab_key = self._extract_key(pth, label_regex, seq_map)
                    lab_map[lab_key] = np.asarray(self._load_one(pth, label_load_spec))
            else:
                label_run_id, _, label_files = self._load_artifacts_glob(labels_spec)
                label_seq_map = self._feature_seq_map(labels_spec["feature"], label_run_id)
                for lf in label_files:
                    lab_key = self._extract_key(lf, label_regex, label_seq_map)
                    lab_map[lab_key] = np.asarray(self._load_one(lf, label_load_spec))
            L_parts = []
            for idx, (key, n) in enumerate(zip(key_list, n_list)):
                arr = lab_map.get(key)
                if arr is None:
                    arr = lab_map.get(to_safe_name(str(key)))
                if arr is None:
                    print(f"[viz-global-colored] WARN: missing labels for key={key}; assigning {missing_value}",
                          file=sys.stderr)
                    arr = np.full(n, missing_value, dtype=int)
                else:
                    arr = np.asarray(arr).ravel()
                    if arr.shape[0] < n:
                        if n % arr.shape[0] == 0:
                            # common case: coord stream duplicated per-id/perspective; repeat labels accordingly
                            factor = n // arr.shape[0]
                            if factor == 2:
                                print(f"[viz-global-colored] INFO: labels shorter than coords for key={key} "
                                      f"({arr.shape[0]} vs {n}); repeating labels x{factor} "
                                      "(likely per-id/perspective duplication in inputs).",
                                      file=sys.stderr)
                            arr = np.tile(arr, factor)
                        else:
                            # lengths differ but not a clean multiple; trim coords to label length
                            print(f"[viz-global-colored] INFO: trimming coords for key={key} "
                                  f"from {n} to {arr.shape[0]} (labels length) due to mismatch.",
                                  file=sys.stderr)
                            Y_list[idx] = Y_list[idx][:arr.shape[0]]
                            n_list[idx] = arr.shape[0]
                            n = arr.shape[0]
                    if arr.shape[0] != n:
                        nmin = min(arr.shape[0], n)
                        if nmin <= 0:
                            continue
                        if nmin < n:
                            # trim coords to match available labels
                            print(f"[viz-global-colored] INFO: trimming coords for key={key} "
                                  f"from {n} to {nmin} to match labels {arr.shape[0]}",
                                  file=sys.stderr)
                            Y_list[idx] = Y_list[idx][:nmin]
                            n_list[idx] = nmin
                        arr = arr[:nmin]
                L_parts.append(arr)
            if not L_parts:
                raise RuntimeError("No labels aligned with coordinates.")
            Y_all = np.vstack(Y_list)
            L_all = np.concatenate(L_parts)
        else:
            L_parts = [np.zeros(n, dtype=int) for n in n_list]
            Y_all = np.vstack(Y_list)
            L_all = np.concatenate(L_parts)

        if bool(self.params.get("debug_save_arrays")):
            self._debug_arrays = {
                "Y_all": Y_all.copy(),
                "L_all": L_all.copy(),
                "key_list": list(key_list),
                "n_list": list(n_list),
            }

        plot_max = int(self.params.get("plot_max", 300_000))
        if Y_all.shape[0] > plot_max:
            rng = np.random.default_rng(42)
            sel = rng.choice(Y_all.shape[0], size=plot_max, replace=False)
            Y_plot = Y_all[sel]
            L_plot = L_all[sel]
        else:
            Y_plot = Y_all
            L_plot = L_all

        color_map = self._prepare_color_map(L_all if labels_spec else L_plot)
        colors = np.array([color_map.get(val, (0.7, 0.7, 0.7)) for val in L_plot])

        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.scatter(
            Y_plot[:, 0],
            Y_plot[:, 1],
            c=colors,
            s=float(self.params.get("point_size", 2.0)),
            alpha=float(self.params.get("point_alpha", 0.35)),
            linewidths=0,
        )
        ax.set_title(self.params.get("title", "Global embedding colored scatter"))
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")

        unique_vals = pd.unique(pd.Series(L_plot, dtype="object"))
        if len(unique_vals) <= 15:
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                           markerfacecolor=color_map.get(val, (0.7, 0.7, 0.7)),
                           markeredgecolor="none", label=self._label_display(val))
                for val in unique_vals
            ]
            if handles:
                ax.legend(handles=handles, title="label", loc="best", fontsize=8)

        out_name = "global_colored.png"
        self._figs = [(out_name, fig)]
        self._summary = {
            "points": int(Y_all.shape[0]),
            "plotted": int(Y_plot.shape[0]),
            "labels_present": bool(labels_spec),
        }

    def transform(self, X):
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        return pd.DataFrame([{
            "outputs": ",".join(fname for fname, _ in self._figs),
            "labels_present": bool(self.params.get("labels")),
        }])

    def save_model(self, path: Path):
        run_root = path.parent
        for fname, fig in self._figs:
            fig.savefig(run_root / fname, dpi=150, bbox_inches="tight")
        if bool(self.params.get("debug_save_arrays")) and self._debug_arrays:
            try:
                np.savez_compressed(run_root / "debug_viz_arrays.npz", **self._debug_arrays)
            except Exception as exc:
                print(f"[viz-global-colored] WARN: failed to save debug arrays: {exc}", file=sys.stderr)
        joblib.dump(
            {"params": self.params, "summary": self._summary,
             "files": [fname for fname, _ in self._figs]},
            run_root / "viz.joblib"
        )
