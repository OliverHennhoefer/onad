"""Base classes for online anomaly detection models and components.

This module provides the fundamental abstract base classes that define the
interfaces for models, transformers, pipelines, and other core components
in the onad library.
"""

import importlib
from typing import Any

from onad.base.exceptions import (
    IncompatibleComponentError,
    ModelNotFittedError,
    OnadError,
    PipelineError,
    TransformationError,
    UnsupportedFeatureError,
    ValidationError,
)
from onad.base.model import BaseModel
from onad.base.pipeline import Pipeline
from onad.base.similarity import BaseSimilaritySearchEngine
from onad.base.transformer import BaseTransformer

__all__ = [
    "Architecture",
    "BaseModel",
    "BaseSimilaritySearchEngine",
    "BaseTransformer",
    "IncompatibleComponentError",
    "ModelNotFittedError",
    "OnadError",
    "Pipeline",
    "PipelineError",
    "TransformationError",
    "UnsupportedFeatureError",
    "ValidationError",
]


def __getattr__(name: str) -> Any:
    """Lazy import of Architecture to avoid torch dependency."""
    if name == "Architecture":
        module = importlib.import_module("onad.base.architecture")
        return module.Architecture
    raise AttributeError(f"module 'onad.base' has no attribute '{name}'")
