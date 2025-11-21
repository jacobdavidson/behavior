from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
import pandas as pd
import numpy as np

from behavior.dataset import register_feature
from behavior.helpers import to_safe_name

def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


@register_feature
class MyNewFeature:
    """
    Template for a per-sequence feature.

    Input:
      - A DataFrame `df` for a single (group, sequence) coming from either:
          * tracks (input_kind="tracks"), e.g. columns: frame, time, id, group, sequence, ...
          * another feature (input_kind="feature"), e.g. PC0.., A_speed.. etc.
    Output:
      - A DataFrame with one row per frame (or per whatever your logic chooses),
        plus standard meta columns:
          * frame or time
          * group, sequence (if available)
        and your feature columns:
          * e.g. "myfeat_x", "myfeat_y", ...
    """

    # how it will be stored under dataset_root/features/<name>/
    name = "my-new-feature"
    version = "0.1"
    parallelizable = True  # safe if transform(df) only depends on df

    # default configuration
    _defaults = dict(
        # which columns to read/use; you can override via params at runtime
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),   # what to use to order rows
        # any algorithm-specific parameters:
        window_size=15,
        smooth_sigma=0.0,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None  # dataset, if you need it in transform()

        # Optional storage overrides:
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True   # append "__from__<input_feature>" in run dir
        self.skip_existing_outputs = False     # set True if idempotent + heavy

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        """Called by Dataset.run_feature before any fit/transform."""
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        """
        Optional: used to restrict which sequences are processed.
        You can ignore this if you don't need it.
        """
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------

    def needs_fit(self) -> bool:
        # If your feature is purely stateless (no learning), return False.
        # If it needs a global fit over all sequences (e.g., PCA), return True.
        return False

    def supports_partial_fit(self) -> bool:
        # If needs_fit() is True and you want to stream sequences for training,
        # return True and implement partial_fit().
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        """
        Optional global fit over all sequences.
        Only called if needs_fit() == True.
        X_iter yields per-sequence DataFrames.
        """
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        """
        Optional streaming fit per sequence.
        Used when supports_partial_fit() == True.
        """
        return

    def finalize_fit(self) -> None:
        """
        Optional hook called after all fit/partial_fit calls.
        """
        return

    def save_model(self, path: Path) -> None:
        """
        Optional. Save any learned parameters to `path`.
        For stateless features, you can either omit or raise NotImplementedError.
        """
        return

    def load_model(self, path: Path) -> None:
        """
        Optional. Restore learned parameters from `path`.
        Needed if you want to re-use the feature model across datasets/runs.
        """
        return

    # ----------------------- Core logic --------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The main workhorse.

        Input: DataFrame for a single sequence (from tracks or another feature).
        Output: DataFrame with your computed features.
        """
        if df is None or df.empty:
            # returning empty is fine; Dataset will just skip writing
            return pd.DataFrame()

        p = self.params
        id_col = p["id_col"]
        seq_col = p["seq_col"]
        group_col = p["group_col"]
        order_col = self._order_col(df)

        # Recover group / sequence for bookkeeping
        sequence = str(df[seq_col].iloc[0]) if seq_col in df.columns else ""
        group = str(df[group_col].iloc[0]) if group_col in df.columns else ""
        safe_seq = to_safe_name(sequence) if sequence else ""

        # Sort by time/frame
        df = df.sort_values(order_col).reset_index(drop=True)

        # ----------- SELECT INPUT COLUMNS ------------------------
        # Example: use all numeric columns except basic meta
        meta_like = {seq_col, group_col, "frame", "time", "id", "perspective"}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in meta_like
        ]
        X = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)

        # ----------- CALL YOUR LOGIC / EXTERNAL LIBS -------------
        # Here is where you focus purely on the science:
        features = self._compute_features_for_sequence(X, p)

        # features should be shape (T_out, F)
        if features.ndim == 1:
            features = features[:, None]
        T_out = features.shape[0]

        # If the length matches original rows, we can reuse frame/time index
        if T_out != len(df):
            # you can decide how to downsample / slice, but simplest is:
            #   assume it's aligned and take the last T_out frames
            base = df.iloc[-T_out:].reset_index(drop=True)
        else:
            base = df.reset_index(drop=True)

        # ----------- BUILD OUTPUT DATAFRAME ----------------------
        out_cols = [f"myfeat_f{i}" for i in range(features.shape[1])]
        out = pd.DataFrame(features, columns=out_cols)

        # attach meta columns
        if "frame" in base.columns:
            out["frame"] = base["frame"].to_numpy()
        if "time" in base.columns:
            out["time"] = base["time"].to_numpy()
        if seq_col in base.columns:
            out[seq_col] = base[seq_col].iloc[0]
        else:
            out[seq_col] = sequence
        if group_col in base.columns:
            out[group_col] = base[group_col].iloc[0]
        else:
            out[group_col] = group

        # Example: if you want one row per frame, do nothing else.
        # If you want per-id, you could explode by id and so on.
        return out

    # ------------------ Internal helpers ------------------------

    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _compute_features_for_sequence(self, X: np.ndarray, params: dict) -> np.ndarray:
        """
        Pure computational logic. X is your numeric matrix for one sequence.

        Here you can call external libraries / your scattered functions.
        This function *does not* touch DataFrame, dataset, or file IO.
        """
        # ---- EXAMPLE: trivial feature = mean over a sliding window ----
        win = int(params["window_size"])
        if win <= 1:
            return X  # identity

        # simple moving average along time axis
        T, D = X.shape
        out = np.zeros_like(X)
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            out[t] = X[lo:hi].mean(axis=0)
        return out
