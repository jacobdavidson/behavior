from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

from behavior.dataset import Dataset
from behavior.helpers import to_safe_name


def cluster_gt_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Quantify agreement between ground-truth labels and cluster assignments.

    Returns a dict containing:
      - counts: n_samples, n_classes, n_clusters
      - information/pair metrics: ARI, AMI, homogeneity, completeness, v_measure
      - purity (cluster-majority score)
      - hungarian_accuracy + best class->cluster mapping
      - confusion matrix (rows=true classes, cols=clusters)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    n_samples = y_true.shape[0]
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    if n_samples == 0 or classes.size == 0 or clusters.size == 0:
        empty_cm = np.zeros((classes.size or 1, clusters.size or 1), dtype=int)
        return {
            "n_samples": int(n_samples),
            "n_classes": int(classes.size),
            "n_clusters": int(clusters.size),
            "ARI": 0.0,
            "AMI": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "purity": 0.0,
            "hungarian_accuracy": 0.0,
            "confusion": empty_cm,
            "mapping": {},
        }

    cm = _contingency_matrix(y_true, y_pred, classes, clusters)
    if cm.size and cm.sum() > 0:
        purity = (cm.max(axis=0).sum() / cm.sum())
    else:
        purity = 0.0

    cost = cm.max() - cm  # maximize cm via minimizing transformed cost
    r_ind, c_ind = linear_sum_assignment(cost)
    best_correct = cm[r_ind, c_ind].sum()
    hung_acc = best_correct / cm.sum() if cm.sum() else 0.0
    mapping = {int(classes[r]): int(clusters[c]) for r, c in zip(r_ind, c_ind)}

    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)

    return {
        "n_samples": int(n_samples),
        "n_classes": int(classes.size),
        "n_clusters": int(clusters.size),
        "ARI": float(ari),
        "AMI": float(ami),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "purity": float(purity),
        "hungarian_accuracy": float(hung_acc),
        "confusion": cm,
        "mapping": mapping,
    }


@dataclass(frozen=True)
class SequenceBundle:
    sequence_safe: str
    path: Path
    sequence: str = ""
    group: str = ""


def compute_cluster_label_agreement(
    ds: Dataset,
    cluster_feature: str,
    cluster_run_id: Optional[str] = None,
    label_kind: str = "behavior",
    cluster_column: str = "cluster",
    sequences: Optional[Sequence[str]] = None,
    max_frames: Optional[int] = None,
    rng_seed: Optional[int] = 42,
    include_per_sequence: bool = False,
) -> dict:
    """
    Align Ward/K-Means (or any per-frame clustering feature) with ground-truth labels
    and report agreement metrics.

    Parameters
    ----------
    ds : Dataset
        Bound dataset instance.
    cluster_feature : str
        Feature folder under features/ (e.g. "ward-assign__from__global-ward__from__global-tsne__from__social+ego@v1").
    cluster_run_id : str, optional
        Specific run_id to use. If None, the most recent finished run is selected.
    label_kind : str
        Folder under labels/ (defaults to "behavior").
    cluster_column : str
        Column name inside the feature parquet holding cluster IDs (default "cluster").
    sequences : Sequence[str], optional
        Optional subset of sequences to analyze. Accepts raw names or already-safe names.
    max_frames : int, optional
        If provided, subsample the aligned frames to this many rows for faster metrics.
    rng_seed : int, optional
        Seed passed to numpy.random.default_rng for subsampling.
    include_per_sequence : bool
        If True, returns per-sequence metrics (with confusion matrices stripped).
    """
    cluster_map, resolved_run_id, run_root = _load_feature_sequence_index(ds, cluster_feature, cluster_run_id)
    label_map = _load_label_sequence_index(ds, label_kind)
    if not cluster_map:
        raise RuntimeError(f"No sequences found for feature '{cluster_feature}' (run_id={cluster_run_id}).")
    if not label_map:
        raise RuntimeError(f"No labels found for kind '{label_kind}'. Run dataset.convert_all_labels first?")

    if sequences:
        wanted = _normalize_sequence_filters(sequences)
        cluster_map = {k: v for k, v in cluster_map.items() if k in wanted}
        if not cluster_map:
            raise RuntimeError("No cluster outputs match the requested sequences.")

    shared_keys = sorted(set(cluster_map).intersection(label_map))
    if not shared_keys:
        raise RuntimeError("No overlap between cluster outputs and labels.")

    rng = np.random.default_rng(rng_seed) if max_frames else None
    y_true_blocks, y_pred_blocks = [], []
    aligned_rows = []
    per_sequence = []

    for safe_seq in shared_keys:
        c_bundle = cluster_map[safe_seq]
        l_bundle = label_map[safe_seq]
        if not c_bundle.path.exists() or not l_bundle.path.exists():
            continue

        pred = _load_cluster_labels(c_bundle.path, cluster_column)
        true = _load_label_npz(l_bundle.path)
        n = min(len(pred), len(true))
        if n == 0:
            continue

        pred = pred[:n].astype(int, copy=False)
        true = true[:n].astype(int, copy=False)
        y_true_blocks.append(true)
        y_pred_blocks.append(pred)
        aligned_rows.append({
            "sequence_safe": safe_seq,
            "sequence": c_bundle.sequence or l_bundle.sequence or safe_seq,
            "group": c_bundle.group or l_bundle.group or "",
            "n_frames": int(n),
            "cluster_path": str(c_bundle.path),
            "label_path": str(l_bundle.path),
        })

        if include_per_sequence:
            seq_metrics = cluster_gt_agreement(true, pred)
            per_sequence.append({
                "sequence_safe": safe_seq,
                "sequence": c_bundle.sequence or l_bundle.sequence or safe_seq,
                "group": c_bundle.group or l_bundle.group or "",
                **_strip_confusion(seq_metrics),
            })

    if not y_true_blocks:
        raise RuntimeError("All overlapping sequences were empty after alignment.")

    y_true_full = np.concatenate(y_true_blocks)
    y_pred_full = np.concatenate(y_pred_blocks)
    total_frames = int(y_true_full.shape[0])

    if max_frames and total_frames > max_frames:
        idx = rng.choice(total_frames, size=max_frames, replace=False)
        y_true_use = y_true_full[idx]
        y_pred_use = y_pred_full[idx]
        sampled = True
    else:
        y_true_use = y_true_full
        y_pred_use = y_pred_full
        sampled = False

    overall_metrics = cluster_gt_agreement(y_true_use, y_pred_use)
    return {
        "cluster_feature": cluster_feature,
        "cluster_run_id": resolved_run_id,
        "label_kind": label_kind,
        "n_sequences": len(aligned_rows),
        "n_frames_total": total_frames,
        "n_frames_used": int(y_true_use.shape[0]),
        "sampled": sampled,
        "metrics": overall_metrics,
        "aligned_sequences": aligned_rows,
        "per_sequence": per_sequence if include_per_sequence else None,
    }


def list_feature_runs(ds: Dataset, feature_name: str) -> pd.DataFrame:
    """
    Inspect features/<feature_name>/index.csv and summarize cached runs (run_id,
    timestamps, model params). Useful to discover available run_ids before looping
    over compute_cluster_label_agreement.
    """
    root = ds.get_root("features") / feature_name
    idx_path = root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Feature index missing: {idx_path}")
    df = pd.read_csv(idx_path)
    if df.empty:
        return pd.DataFrame(columns=[
            "feature", "run_id", "started_at", "finished_at", "n_entries",
            "params_hash", "model_path", "k", "n_clusters", "params",
        ])

    runs = []
    for run_id, group in df.groupby("run_id"):
        run_root = root / str(run_id)
        started = _first_value(group.get("started_at"))
        finished = _first_value(group.get("finished_at"))
        params_hash = _first_value(group.get("params_hash"))
        model_info = _load_model_summary(run_root)
        runs.append({
            "feature": feature_name,
            "run_id": str(run_id),
            "started_at": started,
            "finished_at": finished,
            "n_entries": int(group.shape[0]),
            "params_hash": params_hash,
            **model_info,
        })

    runs_df = pd.DataFrame(runs)
    if not runs_df.empty and "finished_at" in runs_df.columns:
        runs_df = runs_df.sort_values(by="finished_at", ascending=False, kind="stable")
    return runs_df.reset_index(drop=True)


def _load_feature_sequence_index(
    ds: Dataset, feature: str, run_id: Optional[str]
) -> tuple[Dict[str, SequenceBundle], str, Path]:
    root = ds.get_root("features") / feature
    idx_path = root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Feature index missing: {idx_path}")
    df = pd.read_csv(idx_path)
    resolved_run = _resolve_run_id(df, run_id)
    df = df[df["run_id"].astype(str) == resolved_run]
    run_root = root / resolved_run
    mapping: Dict[str, SequenceBundle] = {}
    for _, row in df.iterrows():
        abs_raw = str(row.get("abs_path", "")).strip()
        if not abs_raw:
            continue
        safe_seq = _safe_sequence_name(row)
        if not safe_seq:
            continue
        mapping[safe_seq] = SequenceBundle(
            sequence_safe=safe_seq,
            path=Path(abs_raw),
            sequence=str(row.get("sequence", "") or ""),
            group=str(row.get("group", "") or ""),
        )
    _augment_with_saved_sequences(mapping, run_root)
    return mapping, resolved_run, run_root


def _load_label_sequence_index(ds: Dataset, kind: str) -> Dict[str, SequenceBundle]:
    labels_root = ds.get_root("labels") / kind
    idx_path = labels_root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Labels index missing: {idx_path}")
    df = pd.read_csv(idx_path)
    mapping: Dict[str, SequenceBundle] = {}
    for _, row in df.iterrows():
        abs_raw = str(row.get("abs_path", "")).strip()
        if not abs_raw:
            continue
        safe_seq = _safe_sequence_name(row)
        if not safe_seq:
            continue
        mapping[safe_seq] = SequenceBundle(
            sequence_safe=safe_seq,
            path=Path(abs_raw),
            sequence=str(row.get("sequence", "") or ""),
            group=str(row.get("group", "") or ""),
        )
    return mapping


def _resolve_run_id(df: pd.DataFrame, run_id: Optional[str]) -> str:
    if run_id is not None:
        return str(run_id)
    if "finished_at" in df.columns:
        done = df[df["finished_at"].fillna("").astype(str) != ""]
        if not done.empty:
            df = done
            order_col = "finished_at"
        else:
            order_col = "started_at" if "started_at" in df.columns else None
    else:
        order_col = "started_at" if "started_at" in df.columns else None

    if order_col:
        df = df.sort_values(by=[order_col], ascending=False, kind="stable")
    if df.empty:
        raise RuntimeError("Feature index is empty; run the feature first.")
    return str(df.iloc[0]["run_id"])


def _safe_sequence_name(row: pd.Series) -> str:
    safe_val = str(row.get("sequence_safe", "") or "").strip()
    if safe_val:
        return safe_val
    raw = str(row.get("sequence", "") or "").strip()
    if raw:
        return to_safe_name(raw)
    abs_path = str(row.get("abs_path", "") or "")
    if abs_path:
        return to_safe_name(Path(abs_path).stem)
    return ""


def _load_cluster_labels(path: Path, column: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path, columns=[column])
        if column not in df.columns:
            raise KeyError(f"Column '{column}' missing in {path}")
        data = df[column].to_numpy(copy=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            key = column if column in npz.files else "labels"
            if key not in npz.files:
                raise KeyError(f"Neither '{column}' nor 'labels' present in {path.name}")
            data = npz[key]
    else:
        raise ValueError(f"Unsupported cluster artifact format: {path.suffix}")
    return np.asarray(data, dtype=int).ravel()


def _load_label_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        if "labels" not in npz.files:
            raise KeyError(f"'labels' key missing in {path.name}")
        return np.asarray(npz["labels"], dtype=int).ravel()


def _normalize_sequence_filters(seq_iter: Sequence[str]) -> set[str]:
    safe = set()
    for seq in seq_iter:
        if seq is None:
            continue
        seq_str = str(seq)
        safe.add(seq_str)
        safe.add(to_safe_name(seq_str))
    return safe


def _strip_confusion(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in {"confusion", "mapping"}}


def _contingency_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    if not classes.size or not clusters.size:
        return np.zeros((max(1, classes.size), max(1, clusters.size)), dtype=int)
    row_idx = np.searchsorted(classes, y_true)
    col_idx = np.searchsorted(clusters, y_pred)
    cm = np.zeros((classes.size, clusters.size), dtype=int)
    np.add.at(cm, (row_idx, col_idx), 1)
    return cm


def _augment_with_saved_sequences(mapping: Dict[str, SequenceBundle], run_root: Path) -> None:
    """
    Some features (e.g., GlobalKMeans assign) persist sequence npz files
    directly in the run folder rather than via transform outputs.
    Detect *_labels_seq=*.npz files and add them to the mapping.
    """
    if not run_root or not run_root.exists():
        return
    pattern_files = run_root.glob("*_labels_seq=*.npz")
    for fp in pattern_files:
        if not fp.is_file():
            continue
        stem = fp.stem
        if "labels_seq=" not in stem:
            continue
        safe_seq = stem.split("labels_seq=", 1)[1].strip()
        if not safe_seq or safe_seq in mapping:
            continue
        mapping[safe_seq] = SequenceBundle(sequence_safe=safe_seq, path=fp)


def _load_model_summary(run_root: Path) -> dict:
    model_path = run_root / "model.joblib"
    summary = {
        "model_path": str(model_path) if model_path.exists() else "",
        "k": None,
        "n_clusters": None,
        "params": None,
    }
    if not model_path.exists():
        return summary
    try:
        bundle = joblib.load(model_path)
    except Exception:
        return summary
    params_info = _summarize_model_params(bundle)
    summary.update(params_info)
    summary["model_path"] = str(model_path)
    return summary


def _summarize_model_params(bundle: dict) -> dict:
    info = {"k": None, "n_clusters": None, "params": None}
    if not isinstance(bundle, dict):
        return info
    params = bundle.get("params")
    if isinstance(params, dict):
        info["params"] = params
        info["k"] = params.get("k", info["k"])
        info["n_clusters"] = params.get("n_clusters", info["n_clusters"])
    info["k"] = bundle.get("k", info["k"])
    info["n_clusters"] = bundle.get("n_clusters", info["n_clusters"])
    return info


def _first_value(series: Optional[pd.Series]) -> Optional[str]:
    if series is None or series.empty:
        return None
    for val in series:
        if pd.notna(val):
            return val
    return None
