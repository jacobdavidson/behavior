"""CalMS21 behavior label converter."""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from behavior.helpers import to_safe_name


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to merge user params with defaults."""
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


# Import these from dataset after moving
def _get_behavior_label_map():
    """Get the BEHAVIOR_LABEL_MAP from dataset module."""
    from behavior.dataset import BEHAVIOR_LABEL_MAP
    return BEHAVIOR_LABEL_MAP


def _load_calms21(path: Path):
    """Load a CalMS21 file (.npy or .json)."""
    from behavior.dataset import load_calms21
    return load_calms21(path)


class CalMS21BehaviorConverter:
    """
    Convert CalMS21 npy/json behavior annotations to behavior dataset format.

    CalMS21 files contain nested structure: group -> sequence -> annotations.
    This converter extracts per-frame behavior labels and converts them to
    standardized npz files.
    """

    src_format = "calms21_npy"  # Also handles calms21_json via load_calms21
    label_kind = "behavior"
    label_format = "calms21_behavior_v1"

    _defaults = dict(
        group_from="filename",  # 'filename', 'infile', or 'both'
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the converter with parameters.

        Parameters
        ----------
        params : dict, optional
            Configuration parameters
        **kwargs : additional keyword arguments
            Can override params values
        """
        self.params = _merge_params(params, self._defaults)
        self.params.update(kwargs)

    def convert(self,
                src_path: Path,
                raw_row: pd.Series,
                labels_root: Path,
                params: dict,
                overwrite: bool,
                existing_pairs: set[tuple[str, str]]) -> list[dict]:
        """
        Convert one CalMS21 npy/json file into per-sequence behavior label npz files.

        Parameters
        ----------
        src_path : Path
            Path to CalMS21 source file (.npy or .json)
        raw_row : pd.Series
            Row from tracks_raw/index.csv with metadata
        labels_root : Path
            Output directory (e.g., dataset_root/labels/behavior/)
        params : dict
            Conversion parameters (merged with self.params)
        overwrite : bool
            Whether to overwrite existing files
        existing_pairs : set[tuple[str, str]]
            Set of (group, sequence) pairs already converted

        Returns
        -------
        list[dict]
            List of index row dicts to append to labels/index.csv
        """
        # Load CalMS21 file
        nested = _load_calms21(src_path)
        rows_out: list[dict] = []

        # Get label mapping
        BEHAVIOR_LABEL_MAP = _get_behavior_label_map()
        label_ids = np.array(list(BEHAVIOR_LABEL_MAP.keys()), dtype=int)
        label_names = np.array(list(BEHAVIOR_LABEL_MAP.values()), dtype=object)

        raw_group_hint = str(raw_row.get("group", "") or "")
        group_from = self.params["group_from"]

        # Process each group and sequence in the CalMS21 file
        for group_name, seqs in nested.items():
            group_val_infile = str(group_name or "")

            # Determine output group name based on group_from parameter
            if group_from == "filename" and raw_group_hint:
                group_val = raw_group_hint
            elif group_from == "both" and raw_group_hint:
                group_val = raw_group_hint
            else:
                group_val = group_val_infile

            for seq_key, seq_dict in seqs.items():
                if "annotations" not in seq_dict:
                    continue

                # Extract labels array
                labels = np.asarray(seq_dict["annotations"])
                if labels.ndim > 1:
                    # Handle multi-annotator case: take first column
                    labels = labels[:, 0]
                labels = labels.astype(int, copy=False)

                # Create frame index
                frames = np.arange(labels.shape[0], dtype=np.int32)

                # Determine sequence name
                seq_val = str(seq_key)
                pair = (group_val, seq_val)

                # Create safe filenames
                safe_group = to_safe_name(group_val) if group_val else ""
                safe_seq = to_safe_name(seq_val)
                fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
                out_path = labels_root / fname

                # Skip if already exists and not overwriting
                if not overwrite and pair in existing_pairs and out_path.exists():
                    continue

                # Build npz payload
                payload = {
                    "group": group_val,
                    "sequence": seq_val,
                    "sequence_key": seq_val,
                    "frames": frames,
                    "labels": labels,
                    "label_ids": label_ids,
                    "label_names": label_names,
                }

                # Add source group if group_from is 'filename' or 'both' and differs
                if group_from in {"filename", "both"} and group_val_infile and group_val_infile != group_val:
                    payload["source_group"] = group_val_infile

                # Save npz file
                np.savez_compressed(out_path, **payload)
                existing_pairs.add(pair)

                # Build index row
                rows_out.append({
                    "kind": "behavior",
                    "label_format": "calms21_behavior_v1",
                    "group": group_val,
                    "sequence": seq_val,
                    "group_safe": safe_group,
                    "sequence_safe": safe_seq,
                    "abs_path": str(out_path.resolve()),
                    "source_abs_path": str(src_path.resolve()),
                    "source_md5": raw_row.get("md5", ""),
                    "n_frames": int(labels.shape[0]),
                    "label_ids": ",".join(map(str, BEHAVIOR_LABEL_MAP.keys())),
                    "label_names": ",".join(BEHAVIOR_LABEL_MAP.values()),
                })

        return rows_out

    def get_metadata(self) -> dict:
        """
        Return CalMS21-specific metadata for dataset.meta['labels'][kind].

        Returns
        -------
        dict
            Metadata to merge into dataset.meta['labels']['behavior']
        """
        BEHAVIOR_LABEL_MAP = _get_behavior_label_map()
        return {
            "label_ids": list(BEHAVIOR_LABEL_MAP.keys()),
            "label_names": list(BEHAVIOR_LABEL_MAP.values()),
        }
