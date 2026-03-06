"""Base classes for online anomaly detection models and components.

This module provides the fundamental abstract base classes that define the
interfaces for models, transformers, pipelines, and other core components
in the aberrant library.
"""

import importlib
from typing import Any

from aberrant.base.exceptions import (
    AberrantError,
    IncompatibleComponentError,
    ModelNotFittedError,
    PipelineError,
    TransformationError,
    UnsupportedFeatureError,
    ValidationError,
)
from aberrant.base.model import BaseModel
from aberrant.base.pipeline import Pipeline
from aberrant.base.similarity import BaseSimilaritySearchEngine
from aberrant.base.transformer import BaseTransformer

__all__ = [
    "Architecture",
    "BaseModel",
    "BaseSimilaritySearchEngine",
    "BaseTransformer",
    "IncompatibleComponentError",
    "ModelNotFittedError",
    "AberrantError",
    "Pipeline",
    "PipelineError",
    "TransformationError",
    "UnsupportedFeatureError",
    "ValidationError",
]


def __getattr__(name: str) -> Any:
    """Lazy import of Architecture to avoid torch dependency."""
    if name == "Architecture":
        module = importlib.import_module("aberrant.base.architecture")
        return module.Architecture
    raise AttributeError(f"module 'aberrant.base' has no attribute '{name}'")
