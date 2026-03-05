"""Projection transformers for dimensionality reduction."""

from onad.transform.projection.incremental_pca import IncrementalPCA
from onad.transform.projection.random_projection import RandomProjection

__all__ = [
    "IncrementalPCA",
    "RandomProjection",
]
