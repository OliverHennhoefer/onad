import math
import numpy as np
from typing import Dict
from onad.base.model import BaseModel


class MondrianNode:
    __slots__ = [
        'split_feature', 'split_threshold', 'left_child', 'right_child',
        'is_leaf_', 'min', 'max', 'count'
    ]

    def __init__(self):
        self.split_feature = None  # Index in the projected feature vector
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf_ = True
        self.min = None  # NumPy array of minima
        self.max = None  # NumPy array of maxima
        self.count = 0

    def is_leaf(self):
        return self.is_leaf_

    def update_stats(self, x_values: np.ndarray) -> None:
        if self.count == 0:
            self.min = x_values.copy()
            self.max = x_values.copy()
        else:
            np.minimum(self.min, x_values, out=self.min)
            np.maximum(self.max, x_values, out=self.max)
        self.count += 1

    def attempt_split(self, lambda_: float, rng: np.random.Generator) -> bool:
        ranges = self.max - self.min
        volume = np.prod(ranges)
        if volume <= 0:
            return False

        if rng.random() < 1 - np.exp(-lambda_ * volume):
            probs = ranges / np.sum(ranges)
            split_feature = rng.choice(len(probs), p=probs)
            split_threshold = rng.uniform(self.min[split_feature], self.max[split_feature])

            self.left_child = MondrianNode()
            self.right_child = MondrianNode()

            self.left_child.min = self.min.copy()
            self.left_child.max = self.max.copy()
            self.left_child.max[split_feature] = split_threshold

            self.right_child.min = self.min.copy()
            self.right_child.min[split_feature] = split_threshold
            self.right_child.max = self.max.copy()

            self.split_feature = split_feature
            self.split_threshold = split_threshold
            self.is_leaf_ = False
            return True
        return False


class MondrianTree:
    def __init__(self, selected_indices: np.ndarray, lambda_: float, rng: np.random.Generator):
        self.selected_indices = selected_indices
        self.lambda_ = lambda_
        self.rng = rng
        self.root = MondrianNode()
        self.n_samples = 0

    def learn_one(self, x_projected: np.ndarray) -> None:
        self.n_samples += 1
        current_node = self.root

        while True:
            if current_node.is_leaf():
                current_node.update_stats(x_projected)
                if current_node.attempt_split(self.lambda_, self.rng):
                    continue  # Continue to traverse after split
                else:
                    break
            else:
                if x_projected[current_node.split_feature] <= current_node.split_threshold:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

    def score_one(self, x_projected: np.ndarray) -> int:
        path_length = 0
        current_node = self.root

        while not current_node.is_leaf():
            path_length += 1
            if x_projected[current_node.split_feature] <= current_node.split_threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return path_length


class MondrianForest(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        subspace_size: int = 256,
        lambda_: float = 1.0,
        random_state: int = None
    ):
        self.number_of_trees = n_estimators
        self.subspace_size = subspace_size
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.trees = []
        self.n_samples = 0
        self.features_ = None
        self.feature_to_index = None

    def learn_one(self, x: Dict[str, float]) -> None:
        if self.features_ is None:
            self._initialize_features(x)
        global_features = np.array([x[f] for f in self.features_])

        for tree in self.trees:
            x_projected = global_features[tree.selected_indices]
            tree.learn_one(x_projected)

        self.n_samples += 1

    def _initialize_features(self, x: Dict[str, float]) -> None:
        self.features_ = sorted(x.keys())
        self.subspace_size = min(self.subspace_size, len(self.features_))
        self.feature_to_index = {f: i for i, f in enumerate(self.features_)}

        global_features = np.array([x[f] for f in self.features_])

        for _ in range(self.number_of_trees):
            selected_features = self.rng_.choice(
                self.features_, size=self.subspace_size, replace=False
            )
            selected_indices = np.array([self.feature_to_index[f] for f in selected_features])
            tree = MondrianTree(selected_indices, self.lambda_, self.rng_)
            x_projected = global_features[selected_indices]
            tree.learn_one(x_projected)
            self.trees.append(tree)

    def score_one(self, x: Dict[str, float]) -> float:
        global_features = np.array([x[f] for f in self.features_])
        path_lengths = []

        for tree in self.trees:
            x_projected = global_features[tree.selected_indices]
            path_lengths.append(tree.score_one(x_projected))

        avg_path_length = np.mean(path_lengths)
        c = 1.0 if self.n_samples <= 1 else self._compute_c_factor()
        return 2 ** (-avg_path_length / c)

    def _compute_c_factor(self) -> float:
        n = self.n_samples
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n