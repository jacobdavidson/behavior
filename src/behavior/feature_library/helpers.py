"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Iterator, Callable, TypeVar
from collections import defaultdict
import gc
import re
import sys
import numpy as np
import pandas as pd

from behavior.helpers import to_safe_name

T = TypeVar('T')


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


def _load_array_from_spec(
    path: Path,
    load_spec: dict,
    extract_frame_col: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
    extract_frame_col : str, optional
        If provided, extract this column as frame indices before feature loading

    Returns
    -------
    Tuple[np.ndarray or None, np.ndarray or None]
        (features array as float32, frame indices or None)
    """
    import pyarrow as pa
    kind = str(load_spec.get("kind", "parquet")).lower()
    transpose = bool(load_spec.get("transpose", False))
    frames: Optional[np.ndarray] = None

    if kind == "npz":
        key = load_spec.get("key")
        if not key:
            raise ValueError("load.kind='npz' requires 'key'")
        npz = np.load(path, allow_pickle=True)
        if key not in npz.files:
            return None, None
        A = np.asarray(npz[key])
        if A.ndim == 1:
            A = A[None, :]
    elif kind == "parquet":
        df = pd.read_parquet(path)

        # Extract frame column if requested
        frame_col = extract_frame_col or load_spec.get("frame_column")
        if frame_col and frame_col in df.columns:
            try:
                frames = df[frame_col].to_numpy(dtype=np.int64, copy=True)
            except Exception:
                frames = df[frame_col].to_numpy(copy=True)

        drop_cols = load_spec.get("drop_columns")
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        cols = load_spec.get("columns")
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        elif load_spec.get("numeric_only", True):
            df = df.select_dtypes(include=[np.number])
            # Drop metadata columns that are numeric but not features
            for mc in ("frame", "time", "id1", "id2"):
                if mc in df.columns:
                    df = df.drop(columns=[mc])
        else:
            df = df.apply(pd.to_numeric, errors="coerce")
        # CRITICAL: copy=True to decouple from Arrow memory
        A = df.to_numpy(dtype=np.float32, copy=True)
        del df
        pa.default_memory_pool().release_unused()
    else:
        raise ValueError(f"Unsupported load.kind='{kind}'")

    if A.size == 0:
        return None, frames
    if transpose:
        A = A.T
    if A.ndim == 1:
        A = A[None, :]
    return A.astype(np.float32, copy=False), frames


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
            arr, _ = _load_array_from_spec(fp, load_spec)
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


# -----------------------------------------------------------------------------
# StreamingFeatureHelper - Unified streaming processor for global features
# -----------------------------------------------------------------------------

class StreamingFeatureHelper:
    """
    Unified streaming processor for global features.

    Provides common functionality for:
    - Building file manifests without loading data
    - Streaming data loading with automatic memory cleanup
    - Progress logging
    - Scope filtering

    Usage:
        helper = StreamingFeatureHelper(ds, "my-feature")
        manifest = helper.build_manifest(inputs, scope_filter)
        for key, X, frames in helper.iter_sequences(manifest, extract_frames=True):
            # process X, frames
            pass
    """

    def __init__(self, ds, feature_name: str):
        """
        Initialize the streaming helper.

        Parameters
        ----------
        ds : Dataset
            Dataset instance
        feature_name : str
            Name of the feature (for logging)
        """
        self.ds = ds
        self.feature_name = feature_name
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}

    def build_manifest(
        self,
        inputs: List[dict],
        scope_filter: Optional[dict] = None,
    ) -> Dict[str, List[Tuple[Path, dict]]]:
        """
        Build manifest of file paths per sequence WITHOUT loading data.

        Parameters
        ----------
        inputs : List[dict]
            List of input specifications, each with:
            - feature: str (feature name)
            - run_id: str or None (pick latest if None)
            - pattern: str (glob pattern, default "*.parquet")
            - load: dict (load specification)
        scope_filter : dict, optional
            Scope constraints with keys:
            - safe_sequences: set of allowed sequence names
            - groups: set of allowed groups
            - safe_groups: set of allowed safe group names

        Returns
        -------
        Dict[str, List[Tuple[Path, dict]]]
            Mapping from sequence key to list of (path, load_spec) tuples
        """
        from behavior.dataset import _latest_feature_run_root, _feature_run_root

        manifest: Dict[str, List[Tuple[Path, dict]]] = {}

        # Parse scope filter
        allowed_safe = None
        if scope_filter:
            safe_seqs = scope_filter.get("safe_sequences")
            if safe_seqs:
                allowed_safe = {str(s) for s in safe_seqs}

        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(self.ds, feat_name)
            else:
                run_root = _feature_run_root(self.ds, feat_name, run_id)

            pattern = spec.get("pattern", "*.parquet")
            load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
            files = sorted(run_root.glob(pattern))

            if not files:
                print(f"[{self.feature_name}] WARN: no files for {feat_name} ({run_id}) pattern={pattern}",
                      file=sys.stderr)
                continue

            seq_map = self._get_seq_map(feat_name, run_id)

            for pth in files:
                key = self._extract_key(pth, seq_map)
                if allowed_safe is not None and key not in allowed_safe:
                    continue
                if key not in manifest:
                    manifest[key] = []
                manifest[key].append((pth, load_spec))

        return manifest

    def iter_sequences(
        self,
        manifest: Dict[str, List[Tuple[Path, dict]]],
        extract_frames: bool = False,
        progress_interval: int = 10,
    ) -> Iterator[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
        """
        Iterate through manifest, yielding (key, X_combined, frames) one at a time.

        Automatically handles memory cleanup after each sequence.

        Parameters
        ----------
        manifest : Dict[str, List[Tuple[Path, dict]]]
            Manifest from build_manifest()
        extract_frames : bool, default False
            Whether to extract frame column from parquet files
        progress_interval : int, default 10
            Log progress every N sequences

        Yields
        ------
        Tuple[str, np.ndarray, Optional[np.ndarray]]
            (sequence_key, feature_matrix, frame_indices or None)
        """
        import pyarrow as pa

        n_keys = len(manifest)
        for i, (key, file_specs) in enumerate(manifest.items()):
            X, frames = self.load_key_data(file_specs, extract_frames=extract_frames)
            if X is None:
                continue

            yield key, X, frames

            # Memory cleanup after yielding
            del X
            if frames is not None:
                del frames
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % progress_interval == 0 or i == n_keys - 1:
                print(f"[{self.feature_name}] Processed {i + 1}/{n_keys} sequences", file=sys.stderr)

    def load_key_data(
        self,
        file_specs: List[Tuple[Path, dict]],
        extract_frames: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and concatenate data for a single key.

        Parameters
        ----------
        file_specs : List[Tuple[Path, dict]]
            List of (path, load_spec) tuples
        extract_frames : bool, default False
            Whether to extract frame column

        Returns
        -------
        Tuple[np.ndarray or None, np.ndarray or None]
            (combined_features, frame_indices or None)
        """
        import pyarrow as pa

        mats = []
        frames = None

        for pth, load_spec in file_specs:
            frame_col = "frame" if extract_frames else None
            arr, frame_vals = _load_array_from_spec(pth, load_spec, extract_frame_col=frame_col)
            if arr is None or arr.size == 0:
                continue
            mats.append(arr)
            if frames is None and frame_vals is not None:
                frames = frame_vals

        if not mats:
            return None, None

        T_min = min(m.shape[0] for m in mats)
        mats_trim = [m[:T_min] for m in mats]
        X_full = np.hstack(mats_trim)

        del mats, mats_trim
        gc.collect()
        pa.default_memory_pool().release_unused()

        if frames is not None and len(frames) >= T_min:
            frames = frames[:T_min]
        elif frames is None or len(frames) < T_min:
            frames = np.arange(T_min, dtype=np.int64) if extract_frames else None

        return X_full, frames

    def _get_seq_map(self, feature_name: str, run_id: str) -> Dict[Path, str]:
        """Get cached sequence path mapping."""
        cache_key = (feature_name, str(run_id))
        if cache_key in self._seq_path_cache:
            return self._seq_path_cache[cache_key]
        mapping = _build_path_sequence_map(self.ds, feature_name, run_id)
        self._seq_path_cache[cache_key] = mapping
        return mapping

    def _extract_key(self, path: Path, seq_map: Dict[Path, str]) -> str:
        """Extract sequence key from file path."""
        stem = path.stem
        # Try to extract from filename pattern: seq=<key>
        m = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
        if m:
            return m.group(1)
        # Fallback to sequence map
        key = seq_map.get(path.resolve())
        if key:
            return str(key)
        # Last resort: use safe name of stem
        return to_safe_name(stem)
