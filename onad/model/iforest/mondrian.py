"""Online Mondrian Forest for anomaly detection."""

import math

import numpy as np

from onad.base.model import BaseModel


class MondrianNode:
    """
    Node in a Mondrian tree.

    Represents either a leaf or internal node in the Mondrian tree structure,
    maintaining bounding box statistics and split information.
    """

    __slots__ = [
        "split_feature",
        "split_threshold",
        "left_child",
        "right_child",
        "is_leaf_",
        "min",
        "max",
        "count",
    ]

    def __init__(self) -> None:
        """Initialize a new Mondrian node."""
        self.split_feature: int | None = None
        self.split_threshold: float | None = None
        self.left_child: MondrianNode | None = None
        self.right_child: MondrianNode | None = None
        self.is_leaf_: bool = True
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.count: int = 0

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.is_leaf_

    def update_stats(self, x_values: np.ndarray) -> None:
        """
        Update node statistics with a new data point.

        Args:
            x_values: Feature values to incorporate into statistics.
        """
        if self.count == 0:
            self.min = x_values.copy()
            self.max = x_values.copy()
        else:
            np.minimum(self.min, x_values, out=self.min)
            np.maximum(self.max, x_values, out=self.max)
        self.count += 1

    def attempt_split(self, lambda_: float, rng: np.random.Generator) -> bool:
        """
        Attempt to split this node based on Mondrian process.

        Args:
            lambda_: Rate parameter for the Mondrian process.
            rng: Random number generator.

        Returns:
            True if the node was split, False otherwise.
        """
        ranges = self.max - self.min
        volume = np.prod(ranges)
        if volume <= 0:
            return False

        if rng.random() < 1 - np.exp(-lambda_ * volume):
            probs = ranges / np.sum(ranges)
            split_feature = rng.choice(len(probs), p=probs)
            split_threshold = rng.uniform(
                self.min[split_feature], self.max[split_feature]
            )

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
    """
    Individual Mondrian tree for isolation-based anomaly detection.

    Implements a single tree in the Mondrian Forest, using a Mondrian process
    to generate splits in the feature space.

    Args:
        selected_indices: Indices of features to use for this tree.
        lambda_: Rate parameter for the Mondrian process.
        rng: Random number generator.
    """

    def __init__(
        self, selected_indices: np.ndarray, lambda_: float, rng: np.random.Generator
    ) -> None:
        """Initialize a new Mondrian tree."""
        self.selected_indices = selected_indices
        self.lambda_ = lambda_
        self.rng = rng
        self.root = MondrianNode()
        self.n_samples = 0

    def learn_one(self, x_projected: np.ndarray) -> None:
        """
        Update the tree with a single data point.

        Args:
            x_projected: Projected feature values for the selected subspace.
        """
        self.n_samples += 1
        current_node = self.root

        while True:
            if current_node.is_leaf():
                current_node.update_stats(x_projected)
                if current_node.attempt_split(self.lambda_, self.rng):
                    continue  # Continue to traverse after split
                else:
                    break
            elif (
                x_projected[current_node.split_feature] <= current_node.split_threshold
            ):
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

    def score_one(self, x_projected: np.ndarray) -> int:
        """
        Compute the path length for a single data point.

        Args:
            x_projected: Projected feature values for the selected subspace.

        Returns:
            Path length from root to leaf.
        """
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
    """
    Online Mondrian Forest for anomaly detection.

    Implements an ensemble of Mondrian trees that learn incrementally from
    streaming data. Each tree operates on a random subspace of features,
    and anomaly scores are based on average path lengths.

    Args:
        n_estimators: Number of trees in the forest.
        subspace_size: Number of features per tree.
        lambda_: Rate parameter for the Mondrian process.
        seed: Random seed for reproducibility.

    Example:
        >>> forest = MondrianForest(n_estimators=100, subspace_size=50)
        >>> for x, y in data_stream:
        ...     forest.learn_one(x)
        ...     score = forest.score_one(x)
    """

    # Euler-Mascheroni constant for anomaly score computation
    EULER_MASCHERONI = 0.5772156649015329

    def __init__(
        self,
        n_estimators: int = 100,
        subspace_size: int = 256,
        lambda_: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the Mondrian Forest."""
        super().__init__()

        # Input validation
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if subspace_size <= 0:
            raise ValueError("subspace_size must be positive")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.lambda_ = lambda_
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trees: list[MondrianTree] = []
        self.n_samples = 0
        self._feature_order: list[str] | None = None
        self._feature_to_index: dict[str, int] | None = None

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the forest with a single data point.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        if self._feature_order is None:
            self._initialize_features(x)

        # Use original approach to maintain exact behavior
        global_features = np.array([x[f] for f in self._feature_order])

        for tree in self.trees:
            x_projected = global_features[tree.selected_indices]
            tree.learn_one(x_projected)

        self.n_samples += 1

    def _initialize_features(self, x: dict[str, float]) -> None:
        """
        Initialize feature ordering and create trees.

        Args:
            x: First data point to establish feature order.
        """
        if not x:
            raise ValueError("Cannot initialize forest with empty feature dictionary")
            
        self._feature_order = sorted(x.keys())
        self.subspace_size = min(self.subspace_size, len(self._feature_order))
        self._feature_to_index = {f: i for i, f in enumerate(self._feature_order)}

        # Use original approach for exact behavior match
        global_features = np.array([x[f] for f in self._feature_order])

        for _ in range(self.n_estimators):
            selected_features = self.rng.choice(
                self._feature_order, size=self.subspace_size, replace=False
            )
            selected_indices = np.array(
                [self._feature_to_index[f] for f in selected_features]
            )
            tree = MondrianTree(selected_indices, self.lambda_, self.rng)
            x_projected = global_features[selected_indices]
            tree.learn_one(x_projected)
            self.trees.append(tree)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a single data point.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Anomaly score between 0 and 1 (higher = more anomalous).
        """
        if self._feature_order is None:
            return 0.0

        # Use original approach for exact behavior match
        global_features = np.array([x[f] for f in self._feature_order])
        path_lengths = []

        for tree in self.trees:
            x_projected = global_features[tree.selected_indices]
            path_lengths.append(tree.score_one(x_projected))

        avg_path_length = np.mean(path_lengths)
        c = 1.0 if self.n_samples <= 1 else self._compute_c_factor()
        return 2.0 ** (-avg_path_length / c)

    def _compute_c_factor(self) -> float:
        """
        Compute the normalization factor for anomaly scores.

        Returns:
            Average path length of unsuccessful search in BST.
        """
        n = self.n_samples
        if n <= 1:
            return 1.0
        return 2 * (math.log(n - 1) + self.EULER_MASCHERONI) - 2 * (n - 1) / n

    def __repr__(self) -> str:
        """Return a string representation of the Mondrian Forest."""
        return (
            f"MondrianForest(n_estimators={self.n_estimators}, "
            f"subspace_size={self.subspace_size}, "
            f"lambda_={self.lambda_}, seed={self.seed})"
        )
