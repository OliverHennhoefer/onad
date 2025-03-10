import math
import numpy as np

# from sklearn.utils import check_random_state
from typing import Dict

from onad.base.model import BaseModel


class MondrianNode:
    def __init__(self, selected_features):
        self.selected_features = selected_features  # List of feature names, ordered
        self.split_feature = None  # Index of the feature in selected_features
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf_ = True
        self.min = None  # List of minima for each feature (in selected_features order)
        self.max = None  # List of maxima for each feature
        self.count = 0  # Number of samples in this node

    def is_leaf(self):
        return self.is_leaf_

    def update_stats(self, x_values):
        if self.count == 0:
            self.min = list(x_values)
            self.max = list(x_values)
        else:
            for i in range(len(x_values)):
                if x_values[i] < self.min[i]:
                    self.min[i] = x_values[i]
                if x_values[i] > self.max[i]:
                    self.max[i] = x_values[i]
        self.count += 1

    def attempt_split(self, lambda_, rng):
        ranges = [self.max[i] - self.min[i] for i in range(len(self.min))]
        volume = np.prod(ranges)
        if volume <= 0:
            return False

        # if rng.rand() < 1 - np.exp(-lambda_ * volume):
        if rng.integers(1) < 1 - np.exp(-lambda_ * volume):
            probs = np.array(ranges) / np.sum(ranges)
            split_feature = rng.choice(len(probs), p=probs)
            min_val = self.min[split_feature]
            max_val = self.max[split_feature]
            split_threshold = rng.uniform(min_val, max_val)

            left_child = MondrianNode(self.selected_features)
            right_child = MondrianNode(self.selected_features)

            left_min = self.min.copy()
            left_max = self.max.copy()
            left_max[split_feature] = split_threshold
            left_child.min = left_min
            left_child.max = left_max

            right_min = self.min.copy()
            right_min[split_feature] = split_threshold
            right_child.min = right_min
            right_child.max = self.max.copy()

            self.split_feature = split_feature
            self.split_threshold = split_threshold
            self.left_child = left_child
            self.right_child = right_child
            self.is_leaf_ = False
            return True
        return False


class MondrianTree:
    def __init__(self, selected_features, lambda_, rng):
        self.selected_features = sorted(selected_features)
        self.lambda_ = lambda_
        self.rng = rng
        self.root = MondrianNode(self.selected_features)
        self.n_samples = 0

    def learn_one(self, x_projected):
        x_values = [x_projected[f] for f in self.selected_features]
        self.n_samples += 1
        current_node = self.root

        while True:
            if current_node.is_leaf():
                current_node.update_stats(x_values)
                if current_node.attempt_split(self.lambda_, self.rng):
                    continue
                else:
                    break
            else:
                if x_values[current_node.split_feature] <= current_node.split_threshold:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

    def score_one(self, x_projected):
        x_values = [x_projected[f] for f in self.selected_features]
        path_length = 0
        current_node = self.root

        while not current_node.is_leaf():
            path_length += 1
            if x_values[current_node.split_feature] <= current_node.split_threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return path_length


class MondrianForest(BaseModel):
    def __init__(
        self, n_estimators=100, subspace_size=256, lambda_=1.0, random_state=None
    ):
        self.number_of_trees = n_estimators
        self.subspace_size = subspace_size
        self.lambda_ = lambda_
        self.random_state = random_state
        # self.rng_ = check_random_state(random_state)
        self.rng_ = np.random.default_rng(random_state)
        self.trees = []
        self.n_samples = 0
        self.features_ = None

    def learn_one(self, x: Dict[str, float]) -> None:
        if self.features_ is None:
            self.features_ = sorted(x.keys())
            self.subspace_size = min(self.subspace_size, len(self.features_))

            for _ in range(self.number_of_trees):
                selected_features = list(
                    self.rng_.choice(
                        self.features_, size=self.subspace_size, replace=False
                    )
                )
                tree = MondrianTree(selected_features, self.lambda_, self.rng_)
                x_projected = {f: x[f] for f in selected_features}
                tree.learn_one(x_projected)
                self.trees.append(tree)
        else:
            for tree in self.trees:
                x_projected = {f: x[f] for f in tree.selected_features}
                tree.learn_one(x_projected)
        self.n_samples += 1

    def score_one(self, x: Dict[str, float]) -> float:
        path_lengths = []
        for tree in self.trees:
            x_projected = {f: x[f] for f in tree.selected_features}
            path_length = tree.score_one(x_projected)
            path_lengths.append(path_length)
        average_path_length = np.mean(path_lengths)

        if self.n_samples <= 1:
            c = 1.0
        else:
            c = (
                2 * (math.log(self.n_samples - 1) + 0.5772156649)
                - 2 * (self.n_samples - 1) / self.n_samples
            )
        anomaly_score = 2 ** (-average_path_length / c)
        return anomaly_score
