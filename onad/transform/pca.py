import numpy as np
from typing import Dict, Optional


class IncrementalPCA:
    def __init__(
        self,
        n_components: int,
        n0: int = 50,
        keys: Optional[list[str]] = None,
        tol=1e-7,
    ) -> None:
        """
        Initialize the IncrementalPCA transformer.

        Args:
            n_components (int): Number of principal components to keep.
            center (bool): Whether to center data before applying PCA. Default is False.
            n0 (int): Initial number of samples for warm-up phase before switching to online mode. Default is 50.
            keys (Optional[list[str]]): List of feature names. If None, they will be inferred from the first sample. Default is None.
            tol (float): Tolerance for considering whether a new data point contributes significantly. Default is 1e-7.

        Implements an online PCA algorithm similar to the 'incRpca' from the R Package
        'onlinePCA: Online Principal Component Analysis'. This method is based on the incremental Singular Value Decomposition (SVD) approach
        proposed by Brand (2002) and Arora et al. (2012).

        The decision to adopt this algorithm was informed by insights from
        'Online Principal Component Analysis in High Dimension: Which Algorithm to Choose?'
        by Hervé Cardot and David Degras (2015).
        """
        self.n_components: int = n_components
        self.n0: int = n0
        self.feature_names: Optional[list[str]] = keys
        self.tol = tol

        self.window: list = []  # saving datapoints during inital warm up phase
        self.n0_reached: bool = False
        self.n_samples_seen = 0
        self.n_features: int = 0
        self.dim_change = False

        self.values = np.array([])
        self.vectors = np.array([])

        self._check_n_features()

        # handling missing data settings -> learn_one() and transform_one()
        # ...

    def _check_n_features(self):
        if self.feature_names is not None:
            self.n_features = len(self.feature_names)
            if self.n_components > len(self.feature_names):
                raise ValueError(
                    "The number of primary components has to be less or equal to the number of features"
                )

    def _incPCA(self):
        pass

    def learn_one(self, x: Dict[str, float]) -> None:
        """
        Update PCA components incrementally using a single sample.

        Args:
            x (Dict[str, float]): A dictionary with feature names as keys and values as the data point dimensions.

        Raises:
            ValueError: If `n_components` is greater than the number of features in `x`.
        """
        if self.feature_names is None:
            self.feature_names = list(x.keys())
            self._check_n_features()

        # Convert input dictionary to NumPy array
        datapoint = np.array([x[key] for key in self.feature_names])
        # handling missing data ...

        if self.n0_reached:  # online PCA
            values = (1 - 1 / self.n_samples_seen) * self.values
            datapoint = datapoint * (1 / self.n_samples_seen) ** (1 / 2)
            xhat = self.vectors @ datapoint
            datapoint = datapoint - self.vectors.T @ xhat
            norm_x = np.linalg.norm(datapoint)
            if norm_x >= self.tol:
                self.values = np.append(values, 0)
                xhat = np.append(xhat, norm_x)
                self.vectors = np.append(self.vectors, datapoint / norm_x)
                self.vectors = self.vectors.reshape(
                    self.n_components + 1, self.n_features
                )
                dim_change = True
            else:
                dim_change = False
            diag_matrix = np.diag(self.values)
            tcrossprod_xhat = np.outer(xhat, xhat)
            resulting_matrix = diag_matrix + tcrossprod_xhat
            eigenvalues, eigenvectors = np.linalg.eig(resulting_matrix)
            if dim_change:
                self.values = eigenvalues[: self.n_components]
                eigenvectors = eigenvectors.T[: self.n_components].T
            else:
                self.values = eigenvalues
                self.vectors = eigenvectors.T
            self.vectors = self.vectors.T @ eigenvectors
            self.vectors = self.vectors.T

        else:  # initialisation phase
            self.window.append(datapoint)
            if len(self.window) >= self.n0:
                # initial full pca and switching to online mode
                initial_data = np.array(self.window)
                u, s, vt = np.linalg.svd(initial_data, full_matrices=False)
                s = s / np.sqrt(max(1, initial_data.shape[0] - 1))
                rotation = vt
                self.values = s[: self.n_components] ** 2
                self.vectors = rotation[: self.n_components]
                self.n0_reached = True
                self.window = []
        self.n_samples_seen += 1

    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        """
        Transform a single data point using the learned PCA components.

        Args:
            x (Dict[str, float]): A dictionary with feature names as keys and values as the data point dimensions.

        Returns:
            Dict[int, float]: Transformed data point as a dictionary with reduced dimensions.
        """

        if self.n0_reached:
            if self.feature_names is None:
                raise RuntimeError(
                    "You can't call transform_one() before assigning feature names manually or at least once learn_one()"
                )
            else:
                datapoint = np.array([x[key] for key in self.feature_names])
                # handling missing values...?
            transformed_x = self.vectors @ datapoint
            return {f"component_{i}": val for i, val in enumerate(transformed_x)}
        else:
            return {f"component_{i}": 0.0 for i in range(self.n_components)}
