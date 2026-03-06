"""Isolation forest variants for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "ASDIsolationForest",
    "HalfSpaceTrees",
    "MondrianForest",
    "OnlineIsolationForest",
    "StreamRandomHistogramForest",
]


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    if name == "ASDIsolationForest":
        module = importlib.import_module("onad.model.iforest.asd")
        return module.ASDIsolationForest
    if name == "HalfSpaceTrees":
        module = importlib.import_module("onad.model.iforest.halfspace")
        return module.HalfSpaceTrees
    if name == "MondrianForest":
        module = importlib.import_module("onad.model.iforest.mondrian")
        return module.MondrianForest
    if name == "OnlineIsolationForest":
        module = importlib.import_module("onad.model.iforest.online")
        return module.OnlineIsolationForest
    if name == "StreamRandomHistogramForest":
        module = importlib.import_module("onad.model.iforest.rand_hist")
        return module.StreamRandomHistogramForest
    raise AttributeError(f"module 'onad.model.iforest' has no attribute '{name}'")
