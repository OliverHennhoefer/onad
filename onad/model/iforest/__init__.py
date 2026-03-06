"""Isolation forest variants for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "ASDIsolationForest",
    "HalfSpaceTrees",
    "MondrianForest",
    "OnlineIsolationForest",
    "RandomCutForest",
    "StreamRandomHistogramForest",
    "XStream",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ASDIsolationForest": ("onad.model.iforest.asd", "ASDIsolationForest"),
    "HalfSpaceTrees": ("onad.model.iforest.halfspace", "HalfSpaceTrees"),
    "MondrianForest": ("onad.model.iforest.mondrian", "MondrianForest"),
    "OnlineIsolationForest": ("onad.model.iforest.online", "OnlineIsolationForest"),
    "RandomCutForest": ("onad.model.iforest.random_cut", "RandomCutForest"),
    "StreamRandomHistogramForest": (
        "onad.model.iforest.rand_hist",
        "StreamRandomHistogramForest",
    ),
    "XStream": ("onad.model.iforest.xstream", "XStream"),
}


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'onad.model.iforest' has no attribute '{name}'"
        ) from exc

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
