"""Base model interface for online anomaly detection."""

import abc


class BaseModel(abc.ABC):
    """
    Abstract base class for online anomaly detection models.

    This class defines the interface that all online anomaly detection models should implement.
    Online models process one data point at a time, updating their internal state and providing
    anomaly scores for each data point.

    Subclasses must implement the `learn_one` and `score_one` methods.
    """

    @abc.abstractmethod
    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
               and the values are the corresponding feature values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute the anomaly score for a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
               and the values are the corresponding feature values.

        Returns:
            The anomaly score for the data point. Higher scores indicate greater anomaly.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the model."""
        return self.__repr__()
