"""Distance-based anomaly detection models."""

import importlib
from typing import Any

__all__ = [
    "KNN",
    "LocalOutlierFactor",
    "SDOStream",
]


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    if name == "KNN":
        module = importlib.import_module("aberrant.model.distance.knn")
        return module.KNN
    if name == "LocalOutlierFactor":
        module = importlib.import_module("aberrant.model.distance.lof")
        return module.LocalOutlierFactor
    if name == "SDOStream":
        module = importlib.import_module("aberrant.model.distance.sdostream")
        return module.SDOStream
    raise AttributeError(f"module 'aberrant.model.distance' has no attribute '{name}'")
