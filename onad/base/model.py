import abc
from typing import Dict


class BaseModel(abc.ABC):
    """
    Abstract base class for online anomaly detection model.

    This class defines the interface that all online anomaly detection model should implement.
    Online model process one data point at a time, updating their internal state and providing
    anomaly scores for each data point.

    Subclasses must implement the `learn_one` and `score_one` methods.
    """

    @abc.abstractmethod
    def learn_one(self, x: Dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x (dict): A dictionary representing a single data point. The keys are feature names,
                      and the values are the corresponding feature values.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score_one(self, x: Dict[str, float]) -> float:
        """
        Compute the anomaly score for a single data point.

        Args:
            x (dict): A dictionary representing a single data point. The keys are feature names,
                      and the values are the corresponding feature values.

        Returns:
            float: The anomaly score for the data point. Higher scores indicate greater anomaly.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
