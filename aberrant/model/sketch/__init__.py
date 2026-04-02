"""Sketch-based models for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "LODA",
    "MStream",
    "RSHash",
]


def __getattr__(name: str) -> Any:
    """Lazy import of sketch model classes."""
    if name == "LODA":
        module = importlib.import_module("aberrant.model.sketch.loda")
        return module.LODA
    if name == "MStream":
        module = importlib.import_module("aberrant.model.sketch.mstream")
        return module.MStream
    if name == "RSHash":
        module = importlib.import_module("aberrant.model.sketch.rshash")
        return module.RSHash
    raise AttributeError(f"module 'aberrant.model.sketch' has no attribute '{name}'")
