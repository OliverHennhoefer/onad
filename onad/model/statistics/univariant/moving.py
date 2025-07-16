from typing import Dict, Optional
import numpy as np
from onad.base.model import BaseModel
from collections import deque


class MovingAverage(BaseModel):
    """A simple moving model that calculates the difference between arithmetic average of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingAverage.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving arithmetic average.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the moving average of the current window (including the new data point) 
        and the moving average of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving average (including `x`) and the current moving average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0 
        score_window = list(self.window)
        score_window.append(x[self.feature_name])
        score = sum(score_window) / len(score_window) - sum(self.window)/actual_window_length
        return abs(score) if self.abs_diff else score


class MovingHarmonicAverage(BaseModel):
    """A simple moving model that calculates the difference between harmonic average of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingHarmonicAverage.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving harmonic average.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. 0 will be ignored. 
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        if x[self.feature_name] != 0:
            self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the harmonic average of the current window (including the new data point) 
        and the moving average of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new harmonic average (including `x`) and the current harmonic average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0
        
        score_window = list(self.window)
        score_window.append(x[self.feature_name])
        
        # Calculate harmonic means with zero-division protection
        score_denominator = sum(1 / x for x in score_window if x != 0)
        window_denominator = sum(1 / x for x in self.window if x != 0)
        
        if score_denominator == 0 or window_denominator == 0:
            return 0
            
        score = len(score_window) / score_denominator - actual_window_length / window_denominator 
        return abs(score) if self.abs_diff else score


class MovingGeometricAverage(BaseModel):
    """A simple moving model that calculates the difference between geometric average of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, absolute_values: bool = False, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingGeometricAverage.
        Args:
            window_size (int): The number of recent absolute values to consider for calculating the moving geometric average.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            absolute_values (bool): Default is False. If True, the class calculates the growth between data points and then calculates 
                the geometric average from window_size - 1 values.
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.absolute_values = absolute_values
        self.abs_diff: bool = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values. The value has to be greater then 0.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        if x[self.feature_name] > 0:
            self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the geometric average of the current window (including the new data point) 
        and the geometric average of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new geometric average (including `x`) and the current geometric average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 1 or (
            actual_window_length <= 2 and self.absolute_values
        ):
            return 0
        else:
            if self.absolute_values:
                window_growth = [self.window[i + 1] / self.window[i] for i in range(actual_window_length - 1)]
                score_factor = x[self.feature_name] / self.window[-1]
            else:
                window_growth = list(self.window)
                score_factor = x[self.feature_name]
            
            window_geo = np.prod(window_growth) ** (1 / len(window_growth))
            score_geo = (np.prod(window_growth) * score_factor) ** (1 / (len(window_growth) + 1))
            return abs(score_geo - window_geo) if self.abs_diff else score_geo - window_geo


class MovingMedian(BaseModel):
    """A simple moving model that calculates the difference between median of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingMedian.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving median.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the median of the current window (including the new data point) 
        and the median of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new median (including `x`) and the current median of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0
        sorted_data = sorted(self.window)
        window_score = sorted_data.copy()
        window_score.append(x[self.feature_name])
        window_score.sort()

        mid_index_old = actual_window_length // 2
        mid_index_new = len(window_score) // 2

        if actual_window_length % 2 == 1:
            median_old = sorted_data[mid_index_old]
            median_new = (window_score[mid_index_new - 1] + window_score[mid_index_new]) / 2
        else:
            median_old = (sorted_data[mid_index_old - 1] + sorted_data[mid_index_old]) / 2
            median_new = window_score[mid_index_new]
        score = median_new - median_old
        return abs(score) if self.abs_diff else score


class MovingQuantile(BaseModel):
    """A simple moving model that calculates the difference between quantile of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, quantile: float = 0.5, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingQuantile.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving quantile.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            quantile (float): The quantile set for this instance. Default is 0.5
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.quantile = quantile
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def _quantile(self, sorted_list: list) -> float:
        window_length = len(sorted_list)
        rank = (
            window_length - 1
        ) * self.quantile
        lower_index = int(rank)
        upper_index = min(lower_index + 1, window_length - 1)
        fraction = rank - lower_index
        if lower_index == upper_index:
            return sorted_list[lower_index]
        else:
            return (1 - fraction) * sorted_list[lower_index] + fraction * sorted_list[upper_index]

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the quantile of the current window (including the new data point) 
        and the quantile of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new quantile (including `x`) and the current quantile of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0

        sorted_window: list = sorted(self.window)
        score_data: list = sorted_window.copy()
        score_data.append(x[self.feature_name])
        score_data.sort()

        score = self._quantile(score_data) - self._quantile(sorted_window)
        return abs(score) if self.abs_diff else score


class MovingVariance(BaseModel):
    """A simple moving model that calculates the difference between variance of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingVariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving variance.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the moving variance of the current window (including the new data point) 
        and the moving variance of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving variance (including `x`) and the current moving variance of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length < 1:
            return 0
        else:
            score_window = list(self.window)
            score_window.append(x[self.feature_name])

            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)

            squared_diffs_window = [(value - mean_window) ** 2 for value in self.window]
            squared_diffs_score = [(value - mean_score) ** 2 for value in score_window]

            variance_window = sum(squared_diffs_window) / len(self.window)
            variance_score = sum(squared_diffs_score) / len(score_window)
            score = variance_score - variance_window
            return abs(score) if self.abs_diff else score


class MovingInterquartileRange(BaseModel):
    """A simple moving model that calculates the difference between interquartile range of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingInterquartileRange.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving interquartile range.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def _score_one_quantile(self, sorted_list: list, quantile: float) -> float:
        """Calculate and return the value of a quantile of the values in the window.
        Returns:
            float: The quantile of the values in the window. 0 if the window is empty"""
        list_len = len(sorted_list)
        rank = (
            list_len - 1
        ) * quantile
        lower_index = int(rank)
        upper_index = min(lower_index + 1, list_len - 1)
        fraction = rank - lower_index
        if lower_index == upper_index:
            return sorted_list[lower_index]
        else:
            return (1 - fraction) * sorted_list[
                lower_index
            ] + fraction * sorted_list[upper_index]

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the interquartile range of the current window (including the new data point) 
        and the moving interquartile range of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving interquartile range (including `x`) and the current moving average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0
        
        sorted_data = sorted(self.window)
        score_window = sorted_data.copy()
        score_window.append(x[self.feature_name])
        score_window.sort()
        iqr_window = self._score_one_quantile(sorted_data, 0.75) - self._score_one_quantile(sorted_data, 0.25)
        iqr_score = self._score_one_quantile(score_window, 0.75) - self._score_one_quantile(score_window, 0.25)
        score = iqr_score - iqr_window
        return abs(score) if self.abs_diff else score


class MovingAverageAbsoluteDeviation(BaseModel):
    """A simple moving model that calculates the difference between absolute deviation of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingAverageAbsoluteDeviation.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving average absolute deviation.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the moving average absolute deviation of the current window (including the new data point) 
        and the moving average absolute deviation of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving average (including `x`) and the current moving average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            score_window = list(self.window)
            score_window.append(x[self.feature_name])
            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)
            dev_window = sum(abs(value - mean_window) for value in self.window) / actual_window_length
            dev_score = sum(abs(value - mean_score) for value in score_window) / len(score_window)

            score = dev_score - dev_window
            return abs(score) if self.abs_diff else score


class MovingKurtosis(BaseModel):
    """A simple moving model that calculates the difference between kurtosis of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingKurtosis.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving kurtosis.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the kurtosis of the current window (including the new data point) 
        and the kurtosis of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new kurtosis (including `x`) and the current kurtosis of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            score_window = list(self.window)
            score_window.append(x[self.feature_name])
            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)

            central_moment_4_window = (sum((value - mean_window) ** 4 for value in self.window) / actual_window_length)
            central_moment_4_score = (sum((value - mean_score) ** 4 for value in score_window) / len(score_window))

            std_4_window = (sum((value - mean_window) ** 2 for value in self.window) / actual_window_length) ** 2
            std_4_score = (sum((value - mean_score) ** 2 for value in score_window) / len(score_window)) ** 2

            if std_4_window == 0 or std_4_score == 0:
                return 0
            else:
                kurtosis_window = central_moment_4_window / std_4_window
                kurtosis_score = central_moment_4_score / std_4_score
                score = kurtosis_score - kurtosis_window
                return abs(score) if self.abs_diff else score


class MovingSkewness(BaseModel):
    """A simple moving model that calculates the difference between skewness of a window + new value and the window."""

    def __init__(self, window_size: int, key: Optional[str] = None, abs_diff: bool = True) -> None:
        """Initialize a new instance of MovingSkewness.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving skewness.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: Optional[str] = key
        self.abs_diff = abs_diff

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_name is None:
            self.feature_name = next(iter(x.keys()))
        self.window.append(x[self.feature_name])

    def score_one(self, x: Dict[str, float]) -> float:
        """Calculate and return the difference between the skewness of the current window (including the new data point) 
        and the skewness of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new skewness (including `x`) and the current skewness of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            score_window = list(self.window)
            score_window.append(x[self.feature_name])

            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)

            central_moment_3_window = (sum((value - mean_window) ** 3 for value in self.window) / actual_window_length)
            central_moment_3_score = (sum((value - mean_score) ** 3 for value in score_window) / len(score_window))

            std_3_window = (sum((value - mean_window) ** 2 for value in self.window) / actual_window_length) ** (3 / 2)
            std_3_score = (sum((value - mean_score) ** 2 for value in score_window) / len(score_window)) ** (3 / 2)
            if std_3_score == 0 or std_3_window == 0:
                return 0
            else:
                skewness_window = central_moment_3_window / std_3_window
                skewness_score = central_moment_3_score / std_3_score
                score = skewness_score - skewness_window
                return abs(score) if self.abs_diff else score