import math
import random
from collections import deque

import numpy as np

from onad.base.model import BaseModel


class ASDIsolationForest(BaseModel):
    """
    Optimized Online ASD Isolation Forest for anomaly detection using NumPy for efficient computations.

    Args:
        n_estimators (int): Number of trees in the forest. Default is 100.
        max_samples (int): Number of samples used to build each tree. Default is 256.
        seed (Optional[int]): Random seed for reproducibility. Default is None.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feature_names: list[str] | None = None
        self.buffer: np.ndarray = np.empty((0, 0))
        self.buffer_count: int = 0
        self.trees: deque = deque()
        self.c_n: float = self._compute_c(max_samples)
        self.seed = seed

        # Initialize random number generator
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility (deprecated - use rng instead)."""
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _compute_c(n: int) -> float:
        """Compute the average path length adjustment term c(n)."""
        if n <= 1:
            return 0.0
        harmonic = math.log(n - 1) + 0.5772156649
        return 2.0 * harmonic - 2.0 * (n - 1) / n

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single data point."""
        if self.feature_names is None:
            self.feature_names = list(x.keys())
            self.buffer = np.zeros((self.max_samples, len(self.feature_names)))
            self.buffer_count = 0

        x_converted = np.array([x.get(f, 0.0) for f in self.feature_names])

        if self.buffer_count < self.max_samples:
            self.buffer[self.buffer_count] = x_converted
            self.buffer_count += 1
        else:
            # Build new tree and manage buffer
            new_tree = self._build_tree(self.buffer)
            self.trees.append(new_tree)
            if len(self.trees) > self.n_estimators:
                self.trees.popleft()
            # Reset buffer with current sample
            self.buffer_count = 0
            self.buffer[self.buffer_count] = x_converted
            self.buffer_count += 1

    def _build_tree(self, data_arr: np.ndarray) -> dict:
        """Build an isolation tree from a NumPy array buffer."""
        n_samples = data_arr.shape[0]
        indices = np.arange(n_samples)
        max_height = math.ceil(math.log2(n_samples))
        return self._build_tree_helper(data_arr, indices, max_height)

    def _build_tree_helper(
        self,
        data_arr: np.ndarray,
        indices: np.ndarray,
        max_height: int,
        current_height: int = 0,
    ) -> dict:
        """Iteratively build isolation tree nodes with NumPy optimizations."""
        n = len(indices)
        if n <= 1 or current_height >= max_height:
            return {"size": n, "c": self._compute_c(n)}

        feature_idx = random.randint(0, data_arr.shape[1] - 1)
        feature_vals = data_arr[indices, feature_idx]
        min_val, max_val = np.min(feature_vals), np.max(feature_vals)

        if min_val == max_val:
            return {"size": n, "c": self._compute_c(n)}

        split_val = random.uniform(min_val, max_val)
        mask = feature_vals < split_val
        left_indices = indices[mask]
        right_indices = indices[~mask]

        return {
            "split_feature": self.feature_names[feature_idx],
            "split_val": split_val,
            "left": self._build_tree_helper(
                data_arr, left_indices, max_height, current_height + 1
            ),
            "right": self._build_tree_helper(
                data_arr, right_indices, max_height, current_height + 1
            ),
            "size": n,  # For consistency, though mainly used in leaves
        }

    def _compute_path_length(self, x: dict[str, float], tree: dict) -> float:
        """Iteratively compute path length with precomputed c values."""
        depth = 0
        current_node = tree
        while True:
            if "split_feature" not in current_node:
                return depth + current_node["c"]
            feature_val = x.get(current_node["split_feature"], 0.0)
            if feature_val < current_node["split_val"]:
                current_node = current_node["left"]
            else:
                current_node = current_node["right"]
            depth += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score using optimized path calculations."""
        if not self.trees:
            return 0.0

        total_path = 0.0
        for tree in self.trees:
            total_path += self._compute_path_length(x, tree)

        avg_path = total_path / len(self.trees)
        return 2.0 ** (-avg_path / self.c_n)
