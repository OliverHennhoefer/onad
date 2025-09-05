"""Base classes for online anomaly detection models and components.

This module provides the fundamental abstract base classes that define the
interfaces for models, transformers, pipelines, and other core components
in the onad library.
"""

from .architecture import Architecture
from .exceptions import (
    IncompatibleComponentError,
    ModelNotFittedError,
    OnadError,
    PipelineError,
    TransformationError,
    UnsupportedFeatureError,
    ValidationError,
)
from .model import BaseModel
from .pipeline import Pipeline
from .similarity import BaseSimilaritySearchEngine
from .transformer import BaseTransformer

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
