"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict
import numpy as np
import pandas as pd

from behavior.helpers import to_safe_name


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user-provided parameters with defaults.

    Parameters
    ----------
    overrides : dict or None
        User-provided parameters that override defaults
    defaults : dict
        Default parameter values

    Returns
    -------
    dict
        Merged parameters with overrides taking precedence
    """
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _load_array_from_spec(path: Path, load_spec: dict) -> Optional[np.ndarray]:
    """
    Load a numpy array from a file according to a load specification.

    Parameters
    ----------
    path : Path
        Path to the file to load
    load_spec : dict
        Load specification with keys:
        - kind: "npz" or "parquet"
        - transpose: bool (optional)
        - key: str (required if kind="npz")
        - columns: list (optional, for parquet)
        - drop_columns: list (optional, for parquet)
        - numeric_only: bool (optional, for parquet)

    Returns
    -------
    np.ndarray or None
        Loaded array as float32, shape (T, F) or None if empty
    """
    kind = str(load_spec.get("kind", "parquet")).lower()
    transpose = bool(load_spec.get("transpose", False))

    if kind == "npz":
        key = load_spec.get("key")
        if not key:
            raise ValueError("load.kind='npz' requires 'key'")
        npz = np.load(path, allow_pickle=True)
        if key not in npz.files:
            return None
        A = np.asarray(npz[key])
        if A.ndim == 1:
            A = A[None, :]
    elif kind == "parquet":
        df = pd.read_parquet(path)
        drop_cols = load_spec.get("drop_columns")
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        cols = load_spec.get("columns")
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        elif load_spec.get("numeric_only", True):
            df = df.select_dtypes(include=[np.number])
        else:
            df = df.apply(pd.to_numeric, errors="coerce")
        A = df.to_numpy(dtype=np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported load.kind='{kind}'")

    if A.size == 0:
        return None
    if transpose:
        A = A.T
    if A.ndim == 1:
        A = A[None, :]
    return A.astype(np.float32, copy=False)


def _collect_sequence_blocks(ds, specs: list[dict]) -> dict[str, np.ndarray]:
    """
    Load per-sequence stacked matrices for a given list of input specs.

    Used by global features that need to collect data from multiple feature runs.

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    specs : list[dict]
        List of input specifications, each with:
        - feature: str (feature name)
        - run_id: str or None
        - pattern: str (glob pattern, default "*.parquet")
        - load: dict (load specification)

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from sequence_safe to concatenated feature matrix
    """
    # Import here to avoid circular import
    from behavior.dataset import _latest_feature_run_root, _feature_run_root

    per_seq: dict[str, list[np.ndarray]] = defaultdict(list)
    for spec in specs:
        feat_name = spec["feature"]
        run_id = spec.get("run_id")
        if run_id is None:
            run_id, run_root = _latest_feature_run_root(ds, feat_name)
        else:
            run_root = _feature_run_root(ds, feat_name, run_id)
        pattern = spec.get("pattern", "*.parquet")
        load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            continue
        seq_map = _build_path_sequence_map(ds, feat_name, run_id)
        for fp in files:
            arr = _load_array_from_spec(fp, load_spec)
            if arr is None:
                continue
            safe_seq = seq_map.get(fp.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(fp.stem)
            per_seq[safe_seq].append(arr)

    blocks: dict[str, np.ndarray] = {}
    for safe_seq, mats in per_seq.items():
        mats = [m for m in mats if m.size]
        if not mats:
            continue
        T_min = min(m.shape[0] for m in mats)
        if T_min <= 0:
            continue
        mats = [m[:T_min] for m in mats]
        blocks[safe_seq] = np.hstack(mats)
    return blocks


def _build_path_sequence_map(ds, feature_name: str, run_id: str | None) -> dict[Path, str]:
    """
    Build mapping from absolute file paths to sequence_safe names.

    Uses the dataset's feature index to create a lookup table.

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    feature_name : str
        Name of the feature
    run_id : str or None
        Run ID to filter by

    Returns
    -------
    dict[Path, str]
        Mapping from absolute path to sequence_safe name
    """
    # Import here to avoid circular import
    from behavior.dataset import _feature_index_path

    mapping: dict[Path, str] = {}
    if ds is None or not feature_name or run_id is None:
        return mapping

    try:
        idx_path = _feature_index_path(ds, feature_name)
    except Exception:
        return mapping
    if not idx_path.exists():
        return mapping

    try:
        df = pd.read_csv(idx_path)
    except Exception:
        return mapping

    run_id_str = str(run_id)
    df = df[df["run_id"].astype(str) == run_id_str]
    if df.empty:
        return mapping

    if "sequence_safe" not in df.columns:
        df["sequence_safe"] = df["sequence"].fillna("").apply(
            lambda v: to_safe_name(str(v)) if str(v).strip() else ""
        )

    for _, row in df.iterrows():
        abs_raw = row.get("abs_path")
        if not isinstance(abs_raw, str) or not abs_raw:
            continue
        try:
            abs_path = Path(abs_raw).resolve()
        except Exception:
            abs_path = Path(abs_raw)
        seq_val = (
            row.get("sequence_safe")
            or row.get("sequence")
            or row.get("group_safe")
            or row.get("group")
            or ""
        )
        seq_val = str(seq_val).strip()
        if not seq_val:
            seq_val = to_safe_name(Path(abs_raw).stem)
        mapping[abs_path] = seq_val
    return mapping
