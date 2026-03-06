"""Distance-based anomaly detection models."""

import importlib
from typing import Any

__all__ = [
    "KNN",
    "LocalOutlierFactor",
]


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    if name == "KNN":
        module = importlib.import_module("onad.model.distance.knn")
        return module.KNN
    if name == "LocalOutlierFactor":
        module = importlib.import_module("onad.model.distance.lof")
        return module.LocalOutlierFactor
    raise AttributeError(f"module 'onad.model.distance' has no attribute '{name}'")
