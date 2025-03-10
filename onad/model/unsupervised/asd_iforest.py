import math
import random
from collections import deque
from typing import Dict, Optional

import numpy as np

from onad.base.model import BaseModel


class ASDIsolationForest(BaseModel):
    """
    Online ASD Isolation Forest for anomaly detection.

    This algorithm adapts the Isolation Forest to an online setting by incrementally
    updating a collection of trees built on recent resources points. Each tree is constructed
    from a subsample of the resources stream, and older trees are replaced as new resources arrives.

    Args:
        n_estimators (int): Number of trees in the forest. Default is 100.
        max_samples (int): Number of samples used to build each tree. Default is 256.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.buffer = []
        self.trees = deque()
        self.c_n = self._compute_c(max_samples)
        self.seed = seed

        if self.seed is not None:
            self._set_seed(seed)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def _compute_c(self, n: int) -> float:
        """Compute the average path length adjustment term c(n) for a given sample size n."""
        if n <= 1:
            return 0.0
        harmonic = math.log(n - 1) + 0.5772156649
        return 2.0 * harmonic - 2.0 * (n - 1) / n

    def _build_tree(self, data: list) -> Dict:
        """Build an isolation tree from a subsample of resources."""
        max_height = math.ceil(math.log2(len(data)))
        return self._build_tree_helper(data, current_height=0, max_height=max_height)

    def _build_tree_helper(
        self, data: list, current_height: int, max_height: int
    ) -> Dict:
        """Recursively build an isolation tree node."""
        if len(data) <= 1 or current_height >= max_height:
            return {"size": len(data)}
        # Select a random feature
        feature = random.choice(list(data[0].keys()))
        # Extract feature values and determine split value
        values = [d[feature] for d in data]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {"size": len(data)}
        split_val = random.uniform(min_val, max_val)
        # Split resources into left and right
        left_data = [d for d in data if d[feature] < split_val]
        right_data = [d for d in data if d[feature] >= split_val]
        # Recursively build left and right subtrees
        left = self._build_tree_helper(left_data, current_height + 1, max_height)
        right = self._build_tree_helper(right_data, current_height + 1, max_height)
        return {
            "split_feature": feature,
            "split_val": split_val,
            "left": left,
            "right": right,
            "size": len(data),
        }

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resources point."""
        self.buffer.append(x)
        if len(self.buffer) >= self.max_samples:
            # Build a new tree and update the forest
            new_tree = self._build_tree(self.buffer)
            self.trees.append(new_tree)
            # Remove the oldest tree if exceeding n_estimators
            if len(self.trees) > self.n_estimators:
                self.trees.popleft()
            # Reset buffer for new samples
            self.buffer = []

    def _compute_path_length(self, x: Dict[str, float], tree: Dict) -> float:
        """Compute the path length of a resources point through a single tree."""

        def traverse(node: Dict, depth: int) -> float:
            if "split_feature" not in node:
                # Leaf node: adjust path length by c(node['size'])
                return depth + self._compute_c(node["size"])
            # Traverse left or right child based on split value
            if x.get(node["split_feature"], 0) < node["split_val"]:
                return traverse(node["left"], depth + 1)
            else:
                return traverse(node["right"], depth + 1)

        return traverse(tree, 0)

    def score_one(self, x: Dict[str, float]) -> float:
        """Compute the anomaly score for a single resources point."""
        if not self.trees:
            # Return a default score if no trees are available
            return 0.0
        # Calculate average path length across all trees
        total_path_length = 0.0
        for tree in self.trees:
            total_path_length += self._compute_path_length(x, tree)
        avg_path_length = total_path_length / len(self.trees)
        # Compute anomaly score
        anomaly_score = 2.0 ** (-avg_path_length / self.c_n)
        return anomaly_score
