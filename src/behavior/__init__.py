"""Public package surface for behavior utilities."""

from .dataset import Dataset, register_feature
from .helpers import from_safe_name, to_safe_name

__all__ = [
    "Dataset",
    "register_feature",
    "to_safe_name",
    "from_safe_name",
]
