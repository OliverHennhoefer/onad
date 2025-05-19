from typing import Dict, List, Optional
from onad.base.model import BaseModel
from collections import deque
import numpy as np


class MovingCovariance(BaseModel):
    """
    A simple moving model that calculates the covariance of the most recent values.
    """

    def __init__(
        self, window_size: int, bessel=True, keys: Optional[list[str]] = None
    ) -> None:
        """Initialize a new instance of MovingCovariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving covariance.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: Dict = {}  # {key: deque([], maxlen=window_size)}
        self.feature_names: Optional[list[str]] = keys
        self.bessel = bessel

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains other than two key-value pairs.
        """
        assert len(x) == 2, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
            self.window[self.feature_names[0]] = deque([], maxlen=self.window_size)
            self.window[self.feature_names[1]] = deque([], maxlen=self.window_size)
        if isinstance(x[self.feature_names[0]], (int, float)) and isinstance(
            x[self.feature_names[0]], (int, float)
        ):
            self.window[self.feature_names[0]].append(x[self.feature_names[0]])
            self.window[self.feature_names[1]].append(x[self.feature_names[1]])

    def score_one(self) -> float:
        """Calculate and return the covariance of the values in the window.
        Returns:
            float: The covariance of the values in the window. 0 if the window is empty or has less then 2 data points.
        """
        if self.feature_names == None:
            return 0
        len_0 = len(self.window[self.feature_names[0]])
        len_1 = len(self.window[self.feature_names[1]])
        if len_0 != len_1:
            raise ValueError("Both windows must have the same length.")
        if len_0 < 2:
            return 0

        if self.bessel:
            n = len_0 - 1
        else:
            n = len_0
        mean_0 = sum(self.window[self.feature_names[0]]) / len_0
        mean_1 = sum(self.window[self.feature_names[1]]) / len_1
        return (
            sum(
                (self.window[self.feature_names[0]][i] - mean_0)
                * (self.window[self.feature_names[1]][i] - mean_1)
                for i in range(len_0)
            )
            / n
        )


class MovingCorrelationCoefficient(BaseModel):
    """
    A simple moving model that calculates the correlation coefficient of the most recent values.
    """

    def __init__(
        self, window_size: int, bessel=True, keys: Optional[list[str]] = None
    ) -> None:
        """Initialize a new instance of MovingCorrelationCoefficient.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving correlation coefficient.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: Dict = {}  # {key: deque([], maxlen=window_size)}
        self.feature_names: Optional[list[str]] = keys
        self.bessel = bessel

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains other than two key-value pairs.
        """
        assert len(x) == 2, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
            self.window[self.feature_names[0]] = deque([], maxlen=self.window_size)
            self.window[self.feature_names[1]] = deque([], maxlen=self.window_size)
        if isinstance(x[self.feature_names[0]], (int, float)) and isinstance(
            x[self.feature_names[0]], (int, float)
        ):
            self.window[self.feature_names[0]].append(x[self.feature_names[0]])
            self.window[self.feature_names[1]].append(x[self.feature_names[1]])

    def score_one(self) -> float:
        """Calculate and return the covaricorrelation coefficientance of the values in the window.
        Returns:
            float: The correlation coefficient of the values in the window. 0 if the window is empty or has less then 2 data points.
        """
        if self.feature_names == None:
            return 0
        len_0 = len(self.window[self.feature_names[0]])
        len_1 = len(self.window[self.feature_names[1]])
        if len_0 != len_1:
            raise ValueError("Both windows must have the same length.")
        if len_0 < 2:
            return 0

        if self.bessel:
            n = len_0 - 1
        else:
            n = len_0
        mean_0 = sum(self.window[self.feature_names[0]]) / len_0
        mean_1 = sum(self.window[self.feature_names[1]]) / len_1
        cov_01 = (
            sum(
                (self.window[self.feature_names[0]][i] - mean_0)
                * (self.window[self.feature_names[1]][i] - mean_1)
                for i in range(len_0)
            )
            / n
        )
        std_0 = (
            sum((x - mean_0) ** 2 for x in self.window[self.feature_names[0]]) / n
        ) ** 0.5
        std_1 = (
            sum((x - mean_1) ** 2 for x in self.window[self.feature_names[1]]) / n
        ) ** 0.5
        if std_0 == 0 or std_1 == 0:
            return 0
        else:
            return cov_01 / (std_0 * std_1)


class MovingMahalanobisDistance(BaseModel):
    """
    A simple moving model that calculates the mahalanobis distance of the last two values
    and the corellation matrix of most recent values.
    """

    def __init__(
        self, window_size: int, bias=True, keys: Optional[list[str]] = None
    ) -> None:
        """Initialize a new instance of MovingMahalanobisDistance.
        Args:
            window_size (int): The number of recent values to consider for calculating the mahalanobis distance.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: deque[list[float]] = deque([], maxlen=window_size)
        self.feature_names: Optional[list[str]] = keys
        self.bias = bias

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        """
        if self.feature_names is None:
            self.feature_names = list(x.keys())
        datapoint = [x[key] for key in self.feature_names]
        if all([isinstance(x, (int, float)) for x in datapoint]):
            self.window.append(datapoint)

    def score_one(self) -> float:
        """Calculate and return the covariance of the values in the window.
        Returns:
            float: The covariance of the values in the window. 0 if the window is empty or has less then 2 data points.
        """
        if self.feature_names is None or len(self.window) < 3:
            return 0
        previous_points = np.array(list(self.window)[:-1])
        cov_matrix = np.cov(previous_points, rowvar=False)
        if cov_matrix.shape[0] == cov_matrix.shape[1]:
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                cov_matrix = cov_matrix + 0.001 * min(np.diag(cov_matrix)) * np.eye(
                    cov_matrix.shape[0]
                )
                inv_cov_matrix = np.linalg.inv(cov_matrix)

        feature_mean = np.mean(previous_points, axis=0)
        x = np.array(self.window[-1])
        diff = x - feature_mean
        return float(diff.T @ inv_cov_matrix @ diff)
