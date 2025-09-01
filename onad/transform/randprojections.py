import numpy as np
from typing import Dict, Optional

class RandomProjections:
    def __init__(self, n_components: int, keys: Optional[list[str]] = None, seed=None) -> None:
        """
    Initialize the RandomProjections transformer.

    This constructor sets up an instance of the RandomProjections class to perform
    dimensionality reduction on data points using random projections.

    Implements the binary approach for random projections from:
        - Achlioptas D. (2003) "Database-friendly random projections: Johnson-Lindenstrauss with binary coins"

    Args:
        n_components (int): The target number of dimensions after transformation. Must be less than or equal to the number of original features if `keys` are provided.

        keys (Optional[list[str]]): An optional list of strings representing the
            initial set of feature names.
        seed (Optional[int]): Optional seed for random matrix

    Raises:
        ValueError: If `n_components` is greater than the number of features when
            `keys` are provided.
    """ 
        if n_components < 1:
            raise ValueError("n_components has to be greater then 0")
        self.n_components = n_components
        self.feature_names = keys
        self.seed = seed

        self.n_dimensions = 0
        self.random_matrix = np.array([])

        if self.feature_names is not None:
            if len(self.feature_names) != len(set(self.feature_names)):
                raise ValueError("keys contains duplicates")
            self._initialize_random_matrix()

    def _initialize_random_matrix(self):
        assert self.feature_names, "_initialize_random_matrix should not be called before assigning self.feature_names"
        self.n_dimensions = len(self.feature_names)
        if self.n_components > self.n_dimensions:
            raise ValueError(
                f"The number of n_components ({self.n_components}) has to be less or equal to the number of features ({self.n_dimensions})"
            )
        else:
            np.random.seed(self.seed)
            self.random_matrix = 3**(0.5) * np.random.choice([-1, 0, 1], size=(self.n_dimensions,self.n_components), p = [1/6, 2/3, 1/6])

    def learn_one(self, x: Dict[str, float]) -> None:
        """
        Lern the number of dimension in in the datapoint using a single sample (only with the first).

        Args:
            x (Dict[str, float]): A dictionary with feature names as keys and values as the data point dimensions.

        Raises:
            ValueError: If `n_components` is greater than the number of features in `x`.
        """
        if self.feature_names is None and len(x) >= 1:
            self.feature_names = list(x.keys())
            self._initialize_random_matrix()
    
    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        """
        Transform a single data point using the learned PCA components.

        Args:
            x (Dict[str, float]): A dictionary with feature names as keys and values as the data point dimensions.

        Returns:
            Dict[int, float]: Transformed data point as a dictionary with reduced dimensions.
        """

        if self.feature_names is None:
            raise RuntimeError(
                "You can't call transform_one() before assigning feature names manually or at least once learn_one()"
            )
        else:
            data_vector = np.array([x[key] for key in self.feature_names])
        transformed_x = self.random_matrix.T @ data_vector
        return {f"component_{i}": val for i, val in enumerate(transformed_x)}
