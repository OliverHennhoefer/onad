"""SVM-based models for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "GADGETSVM",
    "IncrementalOneClassSVMAdaptiveKernel",
]


def __getattr__(name: str) -> Any:
    """Lazy import of SVM model classes."""
    if name == "IncrementalOneClassSVMAdaptiveKernel":
        module = importlib.import_module("onad.model.svm.adaptive")
        return module.IncrementalOneClassSVMAdaptiveKernel
    if name == "GADGETSVM":
        module = importlib.import_module("onad.model.svm.gadget")
        return module.GADGETSVM
    raise AttributeError(f"module 'onad.model.svm' has no attribute '{name}'")
