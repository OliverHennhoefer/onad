from typing import Dict, Optional
from onad.base.model import BaseModel
from collections import deque
import numpy as np


def _covariance(x, y, ddof=1):
    """
    Calculate the covariance between two arrays.

    Args:
        x (array-like): First dataset.
        y (array-like): Second dataset.
        ddof int:  1/(n-ddof) for bessel correction.

    Returns:
        float: Covariance value
    """
    if len(x) != len(y):
        raise ValueError("Both datasets must have the same length.")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / (
        len(x) - ddof
    )
    return cov


class MovingCovariance(BaseModel):
    """
    A simple moving model that calculates the difference between the covariance from the window with a new value
    to the covariance from the window of the most recent values.
    """

    def __init__(
        self,
        window_size: int,
        bias=True,
        keys: Optional[list[str]] = None,
        abs_diff=True,
    ) -> None:
        """Initialize a new instance of MovingCovariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving covariance.
            bias (bool): False (default) if bessel corrction should not be used.
            keys (str): Keys for the moving window. If None, the first keys learned are used.
            abs_diff (bool): If True absolute is given back, else covariance(window + score) - covariance(window)
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: Dict = {}  # {key: deque([], maxlen=window_size)}
        self.feature_names: Optional[list[str]] = keys
        self.bias = bias
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains other than two key-value pairs.
        """
        assert len(x) == 2, "Dictionary has other than two key-value pair."
        if self.feature_names is None:
            self.feature_names = list(x.keys())
            self.window[self.feature_names[0]] = deque([], maxlen=self.window_size)
            self.window[self.feature_names[1]] = deque([], maxlen=self.window_size)
        if isinstance(x[self.feature_names[0]], (int, float)) and isinstance(
            x[self.feature_names[1]], (int, float)
        ):
            self.window[self.feature_names[0]].append(x[self.feature_names[0]])
            self.window[self.feature_names[1]].append(x[self.feature_names[1]])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference of the covariance of the values in the window with and without the new point.
            covariance(window + score) - covariance(window)
        Args:
            x (Dict): Single Datapoint to added temporarily to calculate the Covariance.
        Returns:
            float: Difference in the windows. 0 if the window is empty or has less then 2 data points.
        """
        if self.feature_names is None:
            return 0
        score_window_0 = list(self.window[self.feature_names[0]])
        score_window_1 = list(self.window[self.feature_names[1]])
        score_window_0.append(x[self.feature_names[0]])
        score_window_1.append(x[self.feature_names[1]])
        len_window_0 = len(self.window[self.feature_names[0]])
        len_window_1 = len(self.window[self.feature_names[1]])
        len_score_0 = len(score_window_0)
        len_score_1 = len(score_window_1)

        if len_window_0 != len_window_1:
            raise ValueError("Both windows must have the same length.")
        if len_window_0 < 2:
            return 0
        if len_score_0 != len_score_1:
            raise ValueError("Both score windows must have the same length.")
        if len_score_0 < 2:
            return 0

        if self.bias:
            score_cov = _covariance(score_window_0, score_window_1, ddof=0)
            window_cov = _covariance(
                self.window[self.feature_names[0]],
                self.window[self.feature_names[1]],
                ddof=0,
            )
        else:
            score_cov = _covariance(score_window_0, score_window_1, ddof=1)
            window_cov = _covariance(
                self.window[self.feature_names[0]],
                self.window[self.feature_names[1]],
                ddof=1,
            )

        covariance_difference = score_cov - window_cov
        return abs(covariance_difference) if self.abs_diff else covariance_difference


class MovingCorrelationCoefficient(BaseModel):
    """
    A simple moving model that calculates the difference between the correlation coefficient from the window with a new value
    to the correlation coefficient from the window of the most recent values.
    """

    def __init__(
        self,
        window_size: int,
        bias=True,
        keys: Optional[list[str]] = None,
        abs_diff=True,
    ) -> None:
        """Initialize a new instance of MovingCorrelationCoefficient.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving correlation coefficient.
            bias (bool): False if bessel corrction should not be used.
            keys (str): Keys for the moving window. If None, the first keys learned are used.
            abs_diff (bool): If True absolute is given back, else covariance(window + score) - covariance(window)
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: Dict = {}  # {key: deque([], maxlen=window_size)}
        self.feature_names: Optional[list[str]] = keys
        self.bias = bias
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains other than two key-value pairs.
        """
        assert len(x) == 2, "Dictionary has other than two key-value pair."
        if self.feature_names is None:
            self.feature_names = list(x.keys())
            self.window[self.feature_names[0]] = deque([], maxlen=self.window_size)
            self.window[self.feature_names[1]] = deque([], maxlen=self.window_size)
        if isinstance(x[self.feature_names[0]], (int, float)) and isinstance(
            x[self.feature_names[1]], (int, float)
        ):
            self.window[self.feature_names[0]].append(x[self.feature_names[0]])
            self.window[self.feature_names[1]].append(x[self.feature_names[1]])

    def _correlation_coefficient(self, window_0, window_1) -> float:
        len_0 = len(window_0)
        len_1 = len(window_1)
        if len_0 != len_1:
            raise ValueError("Both windows must have the same length.")
        if len_0 < 2:
            return 0
        if self.bias:
            n = len_0
        else:
            n = len_0 - 1
        mean_0 = sum(window_0) / len_0
        mean_1 = sum(window_1) / len_1
        cov = _covariance(window_0, window_1, ddof=0 if self.bias else 1)
        std_0 = (sum((_ - mean_0) ** 2 for _ in window_0) / n) ** 0.5
        std_1 = (sum((_ - mean_1) ** 2 for _ in window_1) / n) ** 0.5
        if std_0 == 0 or std_1 == 0:
            return 0
        else:
            return (
                abs(cov / (std_0 * std_1)) if self.abs_diff else cov / (std_0 * std_1)
            )

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the covaricorrelation coefficientance of the values in the windows.
        Args:
            x (Dict): Single Datapoint to added temporarily to calculate the correlation coefficient.
        Returns:
            float: The correlation coefficient differnce of the values in the window. 0 if the window is empty or has less then 2 data points.
        """
        if self.feature_names is None:
            return 0
        score_window_0 = list(self.window[self.feature_names[0]])
        score_window_1 = list(self.window[self.feature_names[1]])
        score_window_0.append(x[self.feature_names[0]])
        score_window_1.append(x[self.feature_names[1]])
        corr_coeff_diff = self._correlation_coefficient(
            score_window_0, score_window_1
        ) - self._correlation_coefficient(
            self.window[self.feature_names[0]], self.window[self.feature_names[1]]
        )
        return corr_coeff_diff if self.abs_diff else abs(corr_coeff_diff)


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
            bias (bool): False if bessel corrction should not be used.
            keys (str): Keys for the moving window. If None, the first keys learned are used.
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

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the mahalanobis distance from one given point to the windows feature mean.
        Args:
            x (Dict): Single Datapoint.
        Returns:
            float: The mahalanobis distance. 0 if the window is empty or has less then 2 data points.
        """
        if self.feature_names is None or len(self.window) < 3:
            return 0
        previous_points = np.array(list(self.window))
        cov_matrix = np.cov(previous_points, rowvar=False)
        if (
            cov_matrix.shape[0] == cov_matrix.shape[1]
        ):  # test for singular matrix and change diagonal by 0.1% of the minimum diagonal value
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                cov_matrix = cov_matrix + 0.001 * min(np.diag(cov_matrix)) * np.eye(
                    cov_matrix.shape[0]
                )
                inv_cov_matrix = np.linalg.inv(cov_matrix)

        feature_mean = np.mean(previous_points, axis=0)
        x_vector = np.array([x[key] for key in self.feature_names])
        diff = x_vector - feature_mean
        return float(diff.T @ inv_cov_matrix @ diff)
