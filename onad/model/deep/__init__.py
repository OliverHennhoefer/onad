"""Deep learning models for anomaly detection (optional torch dependency)."""

import importlib
from typing import Any

__all__ = [
    "Autoencoder",
]


def __getattr__(name: str) -> Any:
    """Lazy import of deep model classes."""
    if name == "Autoencoder":
        module = importlib.import_module("onad.model.deep.autoencoder")
        return module.Autoencoder
    raise AttributeError(f"module 'onad.model.deep' has no attribute '{name}'")
