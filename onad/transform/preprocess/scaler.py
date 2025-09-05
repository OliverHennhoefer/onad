import math
from collections import Counter, defaultdict

import numpy as np

from onad.base.transformer import BaseTransformer


class MinMaxScaler(BaseTransformer):
    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        """
        Initialize the MinMaxScaler.

        Args:
            feature_range (tuple): The desired range of transformed resources (default is (0, 1)).
        """
        self.feature_range = feature_range
        self.min: dict[str, float] = {}
        self.max: dict[str, float] = {}

    def learn_one(self, x: dict[str, float | np.float64]) -> None:
        """
        Update the min and max values for each feature in the input resources.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.
        """
        for feature, value in x.items():
            if feature not in self.min:
                self.min[feature] = math.inf
                self.max[feature] = -math.inf

            self.min[feature] = min(self.min[feature], value)
            self.max[feature] = max(self.max[feature], value)

    def transform_one(self, x: dict[str, float | np.float64]) -> dict[str, float]:
        """
        Scale the input resources to the specified feature range.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.

        Returns:
            Dict[str, float]: The scaled feature-value pairs.
        """
        scaled_x = {}
        for feature, value in x.items():
            if feature not in self.min or feature not in self.max:
                raise ValueError(
                    f"Feature '{feature}' has not been seen during learning."
                )

            if self.min[feature] == self.max[feature]:
                scaled_x[feature] = float(
                    self.feature_range[0]
                )  # Convert range to float
            else:
                scaled_value = (value - self.min[feature]) / (
                    self.max[feature] - self.min[feature]
                )
                scaled_value = (
                    scaled_value * (self.feature_range[1] - self.feature_range[0])
                    + self.feature_range[0]
                )
                scaled_x[feature] = float(scaled_value)  # Ensure output is float

        return scaled_x


class StandardScaler(BaseTransformer):
    def __init__(self, with_std: bool = True) -> None:
        """
        Initialize the StandardScaler.

        Args:
            with_std (bool): If the normalization should be divided by  standard deviation (default is True)
        """
        self.with_std = with_std
        self.counts: Counter = Counter()
        self.means: defaultdict = defaultdict(float)
        self.sum_sq_diffs: defaultdict = defaultdict(float)

    def learn_one(self, x: dict[str, float | np.float64]) -> None:
        """
        Update the mean and standard deviation for each feature in the input resources.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.
        """
        for feature, value in x.items():
            self.counts[feature] += 1
            old_mean = self.means[feature]
            self.means[feature] += (value - old_mean) / self.counts[feature]
            if self.with_std:
                self.sum_sq_diffs[feature] += (value - old_mean) * (
                    value - self.means[feature]
                )

    def _safe_div(self, a, b) -> float:
        """Returns 0.0 if b is zero or False, else divides a by b."""
        return a / b if b else 0.0

    def transform_one(self, x: dict[str, float | np.float64]) -> dict[str, float]:
        """
        Scale the input resources to standard score.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.

        Returns:
            Dict[str, float]: The scaled feature-value pairs.
        """
        scaled_x = {}
        for feature, value in x.items():
            if feature not in self.means:
                raise ValueError(
                    f"Feature '{feature}' has not been seen during learning."
                )

            if self.with_std:
                variance = (
                    self.sum_sq_diffs[feature] / self.counts[feature]
                    if self.counts[feature] > 0
                    else 0.0
                )
                std_dev = variance**0.5
                scaled_x[feature] = self._safe_div(value - self.means[feature], std_dev)
            else:
                scaled_x[feature] = value - self.means[feature]

        return scaled_x
