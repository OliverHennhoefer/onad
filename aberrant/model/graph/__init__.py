"""Graph-stream anomaly detection models."""

import importlib
from typing import Any

__all__ = [
    "ISCONNA",
]


def __getattr__(name: str) -> Any:
    """Lazy import of graph model classes."""
    if name == "ISCONNA":
        module = importlib.import_module("aberrant.model.graph.isconna")
        return module.ISCONNA
    raise AttributeError(f"module 'aberrant.model.graph' has no attribute '{name}'")
