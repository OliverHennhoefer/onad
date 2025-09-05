"""Random projection transformer for dimensionality reduction."""

import numpy as np

from onad.base.transformer import BaseTransformer


class RandomProjection(BaseTransformer):
    def __init__(
        self, n_components: int, keys: list[str] | None = None, seed: int | None = None
    ) -> None:
        """
        Initialize the RandomProjection transformer.
        
        Implements the binary approach for random projections using sparse random matrices.
        This provides a computationally efficient way to reduce dimensionality while
        approximately preserving distances (Johnson-Lindenstrauss lemma).
        
        Reference:
            Achlioptas D. (2003) "Database-friendly random projections: 
            Johnson-Lindenstrauss with binary coins"

        Args:
            n_components: Target number of dimensions after transformation.
            keys: Feature names. If None, inferred from first sample.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If n_components is greater than the number of features.
        """
        super().__init__()
        
        if n_components < 1:
            raise ValueError("n_components must be greater than 0")
        self.n_components = n_components
        self.feature_names = keys
        self.seed = seed

        self.n_dimensions = 0
        self.random_matrix: np.ndarray = np.array([])

        if self.feature_names is not None:
            if len(self.feature_names) != len(set(self.feature_names)):
                raise ValueError("Feature names cannot contain duplicates")
            self._initialize_random_matrix()

    def _initialize_random_matrix(self) -> None:
        """
        Initialize the random projection matrix.
        
        Raises:
            ValueError: If feature names are not set.
        """
        if self.feature_names is None:
            raise ValueError(
                "Feature names must be set before initializing random matrix"
            )
        self.n_dimensions = len(self.feature_names)
        if self.n_components > self.n_dimensions:
            raise ValueError(
                f"The number of n_components ({self.n_components}) has to be less or equal to the number of features ({self.n_dimensions})"
            )
        else:
            rng = np.random.default_rng(self.seed)
            self.random_matrix = 3 ** (0.5) * rng.choice(
                [-1, 0, 1],
                size=(self.n_dimensions, self.n_components),
                p=[1 / 6, 2 / 3, 1 / 6],
            )

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Learn the number of dimensions from the first data point.

        Args:
            x: A dictionary with feature names as keys and values as data point dimensions.

        Raises:
            ValueError: If n_components is greater than the number of features in x.
        """
        if self.feature_names is None and len(x) >= 1:
            self.feature_names = list(x.keys())
            self._initialize_random_matrix()

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point using random projection.

        Args:
            x: A dictionary with feature names as keys and values as data point dimensions.

        Returns:
            Transformed data point as dictionary with component names as keys.
            
        Raises:
            RuntimeError: If called before learning feature names.
        """

        if self.feature_names is None:
            raise RuntimeError(
                "Cannot transform before learning. Call learn_one() first or provide keys."
            )
        
        data_vector = np.array([x[key] for key in self.feature_names])
        transformed_x = self.random_matrix.T @ data_vector
        return {f"component_{i}": float(val) for i, val in enumerate(transformed_x)}
    
    def __repr__(self) -> str:
        """Return string representation of the transformer."""
        return (
            f"RandomProjection(n_components={self.n_components}, "
            f"seed={self.seed})"
        )
