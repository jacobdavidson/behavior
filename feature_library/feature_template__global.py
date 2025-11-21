from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
import pandas as pd
import numpy as np

from behavior.dataset import (
    register_feature,
    _feature_run_root,
    _latest_feature_run_root,
    _resolve_inputs,
)
from behavior.features import (
    _build_path_sequence_map,
    _load_array_from_spec,
)
from behavior.helpers import to_safe_name

def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out

@register_feature
class MyGlobalFeature:
    """
    Template for a global feature that works on artifacts from prior features,
    e.g. clustering, dimensionality reduction, or any global mapping.

    It does not use per-sequence frames from Dataset.run_feature; instead it
    loads all its inputs directly from the feature run folders.
    """

    name = "my-global-feature"
    version = "0.1"
    parallelizable = False  # usually global stuff is non-parallel per sequence

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = dict(
            inputs=[],          # or inputset=<name>
            inputset=None,
            random_state=42,
            # add algo params here (k, perplexity, etc)
        )
        self.params = _merge_params(params, defaults)
        self._ds = None
        self._artifacts: Dict[str, Any] = {}

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError("MyGlobalFeature uses only fit() and artifacts.")

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("MyGlobalFeature requires dataset binding.")

        # 1) Resolve which prior feature outputs to use
        explicit_inputs = self.params.get("inputs") or []
        inputset_name = self.params.get("inputset")
        explicit_override = bool(explicit_inputs)
        inputs, inputs_meta = _resolve_inputs(
            self._ds,
            explicit_inputs,
            inputset_name,
            explicit_override=explicit_override,
        )

        # 2) Load per-sequence matrices into a dict[str, np.ndarray]
        features_per_key = self._collect_per_sequence_matrices(inputs)

        # 3) Run your global computation (clustering, low-dim embedding, etc)
        artifacts = self._run_global_algorithm(features_per_key)

        # 4) Stash artifacts for save_model()
        artifacts["inputs_meta"] = inputs_meta
        self._artifacts = artifacts

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stub – Dataset.run_feature still expects transform, but all the work
        for this feature is in fit()/save_model(). We can return a tiny marker.
        """
        return pd.DataFrame({"my_global_feature_done": [True]})

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # Example: save global artifacts
        if "templates" in self._artifacts:
            np.savez_compressed(run_root / "my_templates.npz",
                                templates=self._artifacts["templates"])

        if "coords_per_key" in self._artifacts:
            for safe_seq, Y in self._artifacts["coords_per_key"].items():
                np.savez_compressed(run_root / f"my_coords_seq={safe_seq}.npz", Y=Y)

        # You can also dump a joblib bundle if needed
        joblib.dump(
            {
                "params": self.params,
                "meta": self._artifacts.get("meta", {}),
            },
            run_root / "my_global_bundle.joblib",
        )

    def load_model(self, path: Path) -> None:
        # Optional: if you need to re-use artifacts programmatically
        obj = joblib.load(path)
        self.params.update(obj.get("params", {}))
        # reload anything else you care about

    # ----------- internal helpers ------------

    def _collect_per_sequence_matrices(self, inputs: List[dict]) -> Dict[str, np.ndarray]:
        """
        Very similar to GlobalTSNE: load matrices from prior feature runs,
        align by sequence, concatenate horizontally, return dict[safe_seq] -> array.
        """
        per_key_parts: Dict[str, List[np.ndarray]] = {}
        allowed_safe = set(self._scope_filter.get("safe_sequences") or []) if getattr(self, "_scope_filter", None) else None

        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(self._ds, feat_name)
            else:
                run_root = _feature_run_root(self._ds, feat_name, run_id)

            pattern = spec.get("pattern", "*.parquet")
            load_spec = spec.get("load", {"kind": "parquet", "numeric_only": True})
            files = sorted(run_root.glob(pattern))
            if not files:
                continue

            seq_map = _build_path_sequence_map(self._ds, feat_name, run_id)

            for fp in files:
                safe_seq = seq_map.get(fp.resolve()) or to_safe_name(fp.stem)
                if allowed_safe is not None and safe_seq not in allowed_safe:
                    continue
                arr = _load_array_from_spec(fp, load_spec)
                if arr is None or arr.size == 0:
                    continue
                per_key_parts.setdefault(safe_seq, []).append(arr)

        features_per_key: Dict[str, np.ndarray] = {}
        for key, mats in per_key_parts.items():
            mats = [m for m in mats if m.size]
            if not mats:
                continue
            T_min = min(m.shape[0] for m in mats)
            mats = [m[:T_min] for m in mats]
            features_per_key[key] = np.hstack(mats).astype(np.float32, copy=False)
        return features_per_key

    def _run_global_algorithm(self, features_per_key: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Pure computation over the dict[safe_seq] -> array.
        Plug in your own global method here (k-means, t-SNE, etc.)
        """
        # Example: trivial 2D PCA-like projection placeholder
        coords_per_key: Dict[str, np.ndarray] = {}
        for key, X in features_per_key.items():
            # dummy embedding: just take first two columns or pad
            if X.shape[1] >= 2:
                Y = X[:, :2]
            elif X.shape[1] == 1:
                Y = np.hstack([X, np.zeros_like(X)])
            else:
                Y = np.zeros((X.shape[0], 2), dtype=np.float32)
            coords_per_key[key] = Y.astype(np.float32, copy=False)

        artifacts = dict(
            coords_per_key=coords_per_key,
            templates=None,
            meta={},
        )
        return artifacts
