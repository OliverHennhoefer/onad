import math
import numpy as np
from typing import Dict, Union

from onad.base.transformer import BaseTransformer


class StandardScaler(BaseTransformer):
    def __init__(self):
        """
        Initialize the StandardScaler.
        """
        self.mean: Dict[str, float] = {}
        self.variance: Dict[str, float] = {}
        self.n_samples: Dict[str, int] = {}

    def learn_one(self, x: Dict[str, Union[float, np.float64]]) -> None:
        """
        Update the mean and variance for each feature in the input data using Welford's online algorithm.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.
        """
        for feature, value in x.items():
            value = float(value)  # Convert np.float64 to float explicitly

            if feature not in self.mean:
                self.mean[feature] = value
                self.variance[feature] = 0.0
                self.n_samples[feature] = 1
            else:
                n = self.n_samples[feature]
                mean_prev = self.mean[feature]
                variance_prev = self.variance[feature]

                # Update mean and variance using Welford's online algorithm
                self.mean[feature] = mean_prev + (value - mean_prev) / (n + 1)
                self.variance[feature] = variance_prev + (value - mean_prev) * (
                    value - self.mean[feature]
                )
                self.n_samples[feature] += 1

    def transform_one(self, x: Dict[str, Union[float, np.float64]]) -> Dict[str, float]:
        """
        Standardize the input data by subtracting the mean and dividing by the standard deviation.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.

        Returns:
            Dict[str, float]: The standardized feature-value pairs.
        """
        standardized_x = {}
        for feature, value in x.items():
            if (
                feature not in self.mean
                or feature not in self.variance
                or feature not in self.n_samples
            ):
                raise ValueError(
                    f"Feature '{feature}' has not been seen during learning."
                )

            value = float(value)  # Ensure value is a native Python float

            if self.n_samples[feature] < 2:
                # If there's only one sample, the variance is zero, so we can't scale
                standardized_x[feature] = 0.0
            else:
                std_dev = math.sqrt(
                    self.variance[feature] / (self.n_samples[feature] - 1)
                )
                if std_dev == 0:
                    standardized_x[feature] = 0.0
                else:
                    standardized_x[feature] = (value - self.mean[feature]) / std_dev

        return standardized_x
