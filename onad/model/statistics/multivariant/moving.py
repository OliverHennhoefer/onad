from typing import Dict, List, Optional
from onad.base.model import BaseModel
from collections import deque
import numpy as np


class MovingCovariance(BaseModel):
    """
    A simple moving model that calculates the covariance of the most recent values.
    """

    def __init__(self, window_size: int, bessel=True, keys: Optional[list[str]]=None) -> None:
        """Initialize a new instance of MovingCovariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving arithmetic average.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: Dict = {}  #{key: deque([], maxlen=window_size)}
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
        if isinstance(x[self.feature_names[0]],(int, float)) and isinstance(x[self.feature_names[0]],(int, float)):
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
        return sum((self.window[self.feature_names[0]][i] - mean_0) * (self.window[self.feature_names[1]][i] - mean_1) for i in range(len_0)) / n


class MovingCorrelationCoefficient(BaseModel):
    """
    A simple moving model that calculates the correlation coefficient of the most recent values.
    """

    def __init__(self, window_size: int, bessel=True, keys: Optional[list[str]]=None) -> None:
        """Initialize a new instance of MovingCovariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving arithmetic average.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: Dict = {}  #{key: deque([], maxlen=window_size)}
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
        if isinstance(x[self.feature_names[0]],(int, float)) and isinstance(x[self.feature_names[0]],(int, float)):
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
        cov_01 =  sum((self.window[self.feature_names[0]][i] - mean_0) * (self.window[self.feature_names[1]][i] - mean_1) for i in range(len_0)) / n
        std_0 = (sum((x - mean_0) ** 2 for x in self.window[self.feature_names[0]]) / n) ** 0.5
        std_1 = (sum((x - mean_1) ** 2 for x in self.window[self.feature_names[1]]) / n) ** 0.5
        if std_0==0 or std_1==0:
            return 0
        else:
            return cov_01 / (std_0 * std_1)







if __name__ == "__main__":
    data = [{"a": 1, "b": 3.3}, {"a": 2, "b": 4}, {"a": 1, "b": 2}, {"a": 3, "b": 6}, {"a": 1, "b": 3}, {"a": 1, "b": 3}]
    x = [dic["a"] for dic in data]
    y = [dic["b"] for dic in data]
    print(x)

    mc = MovingCorrelationCoefficient(10, bessel=False)
    
    for dat in data:
        mc.learn_one(dat)
    print(mc.window)
    print(mc.score_one())
    print(np.corrcoef(x, y))
