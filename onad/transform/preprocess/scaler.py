"""Preprocessing transformers for data scaling and normalization."""

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from onad.base.transformer import BaseTransformer


class MinMaxScaler(BaseTransformer):
    """
    Min-max scaler for normalizing features to a specified range.
    
    Scales each feature linearly to a target range, typically [0, 1]. This scaling is useful
    for algorithms that are sensitive to the scale of input features.
    
    Args:
        feature_range: Target range for scaled features.
        
    Example:
        >>> scaler = MinMaxScaler(feature_range=(0, 1))
        >>> scaler.learn_one({"x": 5.0, "y": 10.0})
        >>> scaler.transform_one({"x": 3.0, "y": 8.0})
    """
    
    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        """
        Initialize the MinMaxScaler.

        Args:
            feature_range: The desired range of transformed features (default is (0, 1)).
        """
        super().__init__()
        self.feature_range = feature_range
        self.min: dict[str, float] = {}
        self.max: dict[str, float] = {}

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the min and max values for each feature in the input data.

        Args:
            x: A dictionary of feature-value pairs.
        """
        for feature, value in x.items():
            if feature not in self.min:
                self.min[feature] = math.inf
                self.max[feature] = -math.inf

            self.min[feature] = min(self.min[feature], value)
            self.max[feature] = max(self.max[feature], value)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Scale the input data to the specified feature range.

        Args:
            x: A dictionary of feature-value pairs.

        Returns:
            The scaled feature-value pairs.
            
        Raises:
            ValueError: If feature hasn't been seen during learning.
        """
        scaled_x = {}
        for feature, value in x.items():
            if feature not in self.min or feature not in self.max:
                raise ValueError(
                    f"Feature '{feature}' has not been seen during learning."
                )

            if self.min[feature] == self.max[feature]:
                # If min == max, assign the lower bound of the range
                scaled_x[feature] = float(self.feature_range[0])
            else:
                # Standard min-max scaling formula
                scaled_value = (value - self.min[feature]) / (
                    self.max[feature] - self.min[feature]
                )
                scaled_value = (
                    scaled_value * (self.feature_range[1] - self.feature_range[0])
                    + self.feature_range[0]
                )
                scaled_x[feature] = float(scaled_value)

        return scaled_x
    
    def __repr__(self) -> str:
        """Return string representation of the scaler."""
        return f"MinMaxScaler(feature_range={self.feature_range})"


class StandardScaler(BaseTransformer):
    """
    Standard scaler for normalizing features using z-score normalization.
    
    Transforms features to have zero mean and unit variance (when with_std=True).
    This is useful for algorithms that assume features are normally distributed
    and on similar scales.
    
    Args:
        with_std: Whether to scale to unit variance.
        
    Example:
        >>> scaler = StandardScaler()
        >>> scaler.learn_one({"x": 5.0, "y": 10.0})
        >>> scaler.transform_one({"x": 3.0, "y": 8.0})
    """
    
    def __init__(self, with_std: bool = True) -> None:
        """
        Initialize the StandardScaler.

        Args:
            with_std: Whether normalization should be divided by standard deviation.
        """
        super().__init__()
        self.with_std = with_std
        self.counts: Counter[str] = Counter()
        self.means: defaultdict[str, float] = defaultdict(float)
        self.sum_sq_diffs: defaultdict[str, float] = defaultdict(float)

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the mean and standard deviation for each feature incrementally.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: A dictionary of feature-value pairs.
        """
        for feature, value in x.items():
            self.counts[feature] += 1
            old_mean = self.means[feature]
            
            # Welford's online algorithm for mean and variance
            self.means[feature] += (value - old_mean) / self.counts[feature]
            if self.with_std:
                self.sum_sq_diffs[feature] += (value - old_mean) * (
                    value - self.means[feature]
                )

    def _safe_div(self, a: float, b: float) -> float:
        """Safely divide two numbers, returning 0.0 if divisor is zero."""
        return a / b if b != 0.0 else 0.0

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform input data to z-scores (standard scores).

        Args:
            x: A dictionary of feature-value pairs.

        Returns:
            The standardized feature-value pairs.
            
        Raises:
            ValueError: If feature hasn't been seen during learning.
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
    
    def __repr__(self) -> str:
        """Return string representation of the scaler."""
        return f"StandardScaler(with_std={self.with_std})"
