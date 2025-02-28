import math
from typing import Dict, Union

import numpy as np  # Explicitly import numpy

from onad.base.transformer import BaseTransformer

class MinMaxScaler(BaseTransformer):
    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        """
        Initialize the MinMaxScaler.

        Args:
            feature_range (tuple): The desired range of transformed data (default is (0, 1)).
        """
        self.feature_range = feature_range
        self.min: Dict[str, float] = {}
        self.max: Dict[str, float] = {}

    def learn_one(self, x: Dict[str, Union[float, np.float64]]) -> None:
        """
        Update the min and max values for each feature in the input data.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.
        """
        for feature, value in x.items():
            value = float(value)  # Convert np.float64 to float explicitly

            if feature not in self.min:
                self.min[feature] = math.inf
                self.max[feature] = -math.inf

            self.min[feature] = min(self.min[feature], value)
            self.max[feature] = max(self.max[feature], value)

    def transform_one(self, x: Dict[str, Union[float, np.float64]]) -> Dict[str, float]:
        """
        Scale the input data to the specified feature range.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.

        Returns:
            Dict[str, float]: The scaled feature-value pairs.
        """
        scaled_x = {}
        for feature, value in x.items():
            if feature not in self.min or feature not in self.max:
                raise ValueError(f"Feature '{feature}' has not been seen during learning.")

            value = float(value)  # Ensure value is a native Python float

            if self.min[feature] == self.max[feature]:
                scaled_x[feature] = float(self.feature_range[0])  # Convert range to float
            else:
                scaled_value = (value - self.min[feature]) / (self.max[feature] - self.min[feature])
                scaled_value = (
                    scaled_value * (self.feature_range[1] - self.feature_range[0])
                    + self.feature_range[0]
                )
                scaled_x[feature] = float(scaled_value)  # Ensure output is float

        return scaled_x
