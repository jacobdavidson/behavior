from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import numpy as np
import pandas as pd

from behavior.dataset import register_feature


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _mean_angle(angles: np.ndarray) -> float:
    return float(np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles))))


@register_feature
class FFGroupsMetrics:
    """
    Per-sequence summary of focal-fish group metrics.

    Per-frame computed (internal):
      - distance_from_centroid, xrot_to_centroid, yrot_to_centroid, dev_speed_to_mean
    Summaries (output: one row per id within sequence):
      - fractime_norm2
      - avg_duration_frame
      - med_duration_frame
      - ftime_periphery
      - ftime_periphery_norm
    """

    name = "ffgroups-metrics"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        id_col="id",
        seq_col="sequence",
        group_col="group",
        x_col="x",
        y_col="y",
        heading_col="ANGLE",
        speed_col="speed",
        order_pref=("frame", "time"),
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path: Path) -> None:
        return

    def load_model(self, path: Path) -> None:
        return

    # ----------------------- Core logic --------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = self._order_col(df)
        df = df.sort_values(order_col).reset_index(drop=True)

        required = [p["x_col"], p["y_col"], p["heading_col"], p["speed_col"], p["id_col"]]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for FFGroupsMetrics: {missing}")

        # Group per frame within group_col if present
        group_keys = []
        if p["group_col"] in df.columns:
            group_keys.append(p["group_col"])
        frame_key = "frame" if "frame" in df.columns else "time"
        if frame_key not in df.columns:
            raise ValueError("Need either 'frame' or 'time' column.")
        group_keys.append(frame_key)

        df = self._compute_per_frame(df, p, group_keys)
        summary = self._compute_summary(df, p)

        return summary

    # ------------------ Internal helpers ------------------------
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _compute_per_frame(self, df: pd.DataFrame, p: dict, group_keys: list) -> pd.DataFrame:
        # Compute centroid, heading, speed mean per frame(group) and attach deltas.
        grouped = df.groupby(group_keys, sort=False)
        records = []
        for _, g in grouped:
            cx = g[p["x_col"]].mean()
            cy = g[p["y_col"]].mean()
            chead = _mean_angle(g[p["heading_col"]].to_numpy(dtype=float))
            mean_speed = g[p["speed_col"]].mean()

            dx = g[p["x_col"]] - cx
            dy = g[p["y_col"]] - cy
            ct, st = np.cos(-chead), np.sin(-chead)
            xrot = dx * ct - dy * st
            yrot = dx * st + dy * ct

            rec = g.copy()
            rec["distance_from_centroid"] = np.sqrt(dx ** 2 + dy ** 2)
            rec["xrot_to_centroid"] = xrot
            rec["yrot_to_centroid"] = yrot
            rec["dev_speed_to_mean"] = g[p["speed_col"]] - mean_speed
            records.append(rec)
        return pd.concat(records, axis=0).reset_index(drop=True)

    def _compute_summary(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        id_col = p["id_col"]
        frame_key = "frame" if "frame" in df.columns else "time"

        # Total frames per fish
        total_frames = df.groupby(id_col)[frame_key].count()

        # fractime_norm2: fraction of frames in each group_size (computed as count of frames sharing group in frame)
        if "group_size" in df.columns:
            gsize = df["group_size"]
        else:
            # infer group_size per frame group
            df["group_size"] = df.groupby(frame_key)[id_col].transform("count")
            gsize = df["group_size"]

        frame_counts = df.groupby([id_col, "group_size"])[frame_key].count()
        fractime_norm2 = (frame_counts / total_frames.reindex(frame_counts.index.get_level_values(id_col)).to_numpy())
        fractime_norm2 = fractime_norm2.reset_index(name="fractime_norm2")

        # durations of contiguous group_size runs per fish
        dur_rows = []
        for fish, sub in df.sort_values(frame_key).groupby(id_col):
            gsz = sub["group_size"].to_numpy()
            frames = sub[frame_key].to_numpy()
            if len(gsz) == 0:
                continue
            # find change points
            change = np.where(np.diff(gsz) != 0)[0] + 1
            starts = np.r_[0, change]
            ends = np.r_[change, len(gsz)]
            for s, e in zip(starts, ends):
                dur = frames[e - 1] - frames[s] + 1
                dur_rows.append((fish, gsz[s], dur))
        durations = pd.DataFrame(dur_rows, columns=[id_col, "group_size", "duration"])
        if durations.empty:
            durations["avg_duration_frame"] = []
            durations["med_duration_frame"] = []
        agg_dur = durations.groupby([id_col, "group_size"])["duration"].agg(
            avg_duration_frame="mean", med_duration_frame="median"
        ).reset_index()

        # periphery time: rank by distance, count frames where farthest
        df["rank_centroid_distance"] = df.groupby(frame_key)["distance_from_centroid"].rank(
            ascending=True, method="min"
        )
        farthest = df["rank_centroid_distance"] == df["group_size"]
        ftime_periphery = df.loc[farthest].groupby(id_col)[frame_key].count() / total_frames
        ftime_periphery = ftime_periphery.fillna(0)
        ftime_periphery_norm = (
            df.loc[farthest, "group_size"].groupby(df.loc[farthest, id_col]).sum() / total_frames
        ).reindex(total_frames.index).fillna(0)

        # Merge summaries
        out = (
            fractime_norm2.merge(agg_dur, on=[id_col, "group_size"], how="left")
            .merge(
                ftime_periphery.rename("ftime_periphery"),
                on=id_col,
                how="left",
            )
            .merge(
                ftime_periphery_norm.rename("ftime_periphery_norm"),
                on=id_col,
                how="left",
            )
        )

        # Attach meta
        for meta_col in (p["seq_col"], p["group_col"]):
            if meta_col in df.columns and meta_col not in out.columns:
                out[meta_col] = df[meta_col].iloc[0]

        return out.reset_index(drop=True)
