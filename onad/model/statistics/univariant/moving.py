from typing import Dict, List, Optional
from onad.base.model import BaseModel
from collections import deque


class MovingAverage(BaseModel):
    """A simple moving model that calculates the arithmetic average of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingAverage.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving arithmetic average.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the arithmetic avarage of the values in the window.
        Returns:
            float: The arithmetic average of the values in the window. 0 if the window is empty.
        """
        actual_window_length = len(self.window)
        return (
            0 if actual_window_length == 0 else sum(self.window) / actual_window_length
        )


class MovingHarmonicAverage(BaseModel):
    """A simple moving model that calculates the harmonic average of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingHarmonicAverage.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving harmonic average.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. 0 will be ignored. 
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        if x[self.feature_names[0]] != 0:
            self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the harmonic avarage of the values in the window.
        Returns:
            float: The harmonic average of the values in the window. 0 if the window is empty
        """
        actual_window_length = len(self.window)
        return (
            0
            if actual_window_length == 0
            else actual_window_length / sum(1 / x for x in self.window)
        )


class MovingGeometricAverage(BaseModel):
    """A simple moving model that calculates the geometric average of the most recent (absolute) values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int, absoluteValues=False) -> None:
        """Initialize a new instance of MovingGeometricAverage.
        Args:
            window_size (int): The number of recent absolute values to consider for calculating the moving geometric average.
            absoluteValues (bool): Default is False. If True, the class is calculating the growth between the data point and then calculating 
                the geometric avarage from window_size - 1 values. 
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None
        self.absoluteValues: bool = absoluteValues

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values. The value has to be greater then 0.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        if x[self.feature_names[0]] > 0:
            self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the geometric avarage of the values in the window.
        Returns:
            float: The geometric average of the values in the window. 1 if the window is empty
        """
        actual_window_length = len(self.window)

        if actual_window_length <= 1 or (
            actual_window_length <= 2 and self.absoluteValues
        ):
            return 1
        else:
            if self.absoluteValues:
                window_growth = [
                    self.window[i + 1] / self.window[i]
                    for i in range(actual_window_length - 1)
                ]
            else:
                window_growth = self.window
            window_product = 1
            for value in window_growth:
                window_product *= value
            return window_product ** (1 / (actual_window_length - 1))


class MovingMedian(BaseModel):
    """A simple moving model that calculates the median of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingMedian.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving median.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the median of the values in the window.
        Returns:
            float: The median of the values in the window. 0 if the window is empty"""
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0
        sorted_data = sorted(self.window)
        mid_index = actual_window_length // 2

        if actual_window_length % 2 == 1:
            # If odd, return the middle element
            median = sorted_data[mid_index]
        else:
            # If even, return the average of the two middle elements
            median = (sorted_data[mid_index - 1] + sorted_data[mid_index]) / 2
        return median


class MovingQuantile(BaseModel):
    """A simple moving model that calculates the given quantile of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int, quantile=0.5) -> None:
        """Initialize a new instance of MovingQuantile.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving quantle.
            quantile (float): The quantile set for this instance. Default is 0.5
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None
        self.quantile = quantile

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the median of the values in the window.
        Returns:
            float: The median of the values in the window. 0 if the window is empty"""
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0

        sorted_data = sorted(self.window)
        rank = (
            actual_window_length - 1
        ) * self.quantile  # Calculate the index of the quantile
        lower_index = int(rank)
        upper_index = min(lower_index + 1, actual_window_length - 1)
        fraction = rank - lower_index  # Calculate the fractional part
        if lower_index == upper_index:
            return sorted_data[lower_index]
        else:
            # Interpolation is necessary
            return (1 - fraction) * sorted_data[lower_index] + fraction * sorted_data[
                upper_index
            ]


class MovingVariance(BaseModel):
    """A simple moving model that calculates the variance  of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingVariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving arithmetic average.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the variance of the values in the window.
        Returns:
            float: The variance of the values in the window. 0 if the window is empty.
        """
        actual_window_length = len(self.window)
        if actual_window_length < 1:
            return 0
        else:
            mean = sum(self.window) / actual_window_length
            squared_diffs = [(x - mean) ** 2 for x in self.window]
            variance = sum(squared_diffs) / len(self.window)
            return variance


class MovingInterquartileRange(BaseModel):
    """A simple moving model that calculates the interquartile range of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingInterquartileRange.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving median.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def __score_one_quantile(self, quantile) -> float:
        """Calculate and return the value of a quantile of the values in the window.
        Returns:
            float: The quantile of the values in the window. 0 if the window is empty"""
        rank = (
            self.actual_window_length - 1
        ) * quantile  # Calculate the index of the quantile
        lower_index = int(rank)
        upper_index = min(lower_index + 1, self.actual_window_length - 1)
        fraction = rank - lower_index
        if lower_index == upper_index:
            return self.sorted_data[lower_index]
        else:
            # Interpolation is necessary
            return (1 - fraction) * self.sorted_data[
                lower_index
            ] + fraction * self.sorted_data[upper_index]

    def score_one(self) -> float:
        """Calculate and return the interquartile range of the values in the window.
        Returns:
            float: The interquantile range of the values in the window. 0 if the window is empty
        """
        self.actual_window_length = len(self.window)
        if self.actual_window_length <= 0:
            return 0
        self.sorted_data = sorted(self.window)
        return self.__score_one_quantile(0.75) - self.__score_one_quantile(0.25)


class MovingAverageAbsoluteDeviation(BaseModel):
    """A simple moving model that calculates the average absolute deviation of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingAverageAbsoluteDeviation.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving average absolute deviation.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the average absolute deviation of the values in the window.
        Returns:
            float: The average absolute deviation of the values in the window. 0 if the window is empty.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            mean = sum(self.window) / actual_window_length
            return (
                sum(abs(value - mean) for value in self.window) / actual_window_length
            )


class MovingKurtosis(BaseModel):
    """A simple moving model that calculates the kurtosis of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int, fisher: bool = True) -> None:
        """Initialize a new instance of MovingKurtosis.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving kurtosis.
            fisher (bool): If Fisher’s definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None
        self.fisher = fisher

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the kurtois of the values in the window.
        Returns:
            float: The kurtosis of the values in the window. 0 if the window is empty.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            mean = sum(self.window) / actual_window_length
            central_moment_4 = (
                sum((value - mean) ** 4 for value in self.window) / actual_window_length
            )
            std_4 = (
                sum((value - mean) ** 2 for value in self.window) / actual_window_length
            ) ** 2
            if std_4 == 0:
                return 0
            else:
                kurtosis = central_moment_4 / std_4
                return kurtosis - 3 if self.fisher else kurtosis


class MovingSkewness(BaseModel):
    """A simple moving model that calculates the skewness of the most recent values.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize a new instance of MovingSkewness.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving Skewness.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_names: Optional[List[str]] = None

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        assert len(x) == 1, "Dictionary has more than one key-value pair."
        if self.feature_names == None:
            self.feature_names = list(x.keys())
        self.window.append(x[self.feature_names[0]])

    def score_one(self) -> float:
        """Calculate and return the kurtois of the values in the window.
        Returns:
            float: The kurtosis of the values in the window. 0 if the window is empty.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            mean = sum(self.window) / actual_window_length
            central_moment_3 = (
                sum((value - mean) ** 3 for value in self.window) / actual_window_length
            )
            std_3 = (
                sum((value - mean) ** 2 for value in self.window) / actual_window_length
            ) ** (3 / 2)
            return 0 if std_3 == 0 else central_moment_3 / std_3


