"""Base transformer interface for online data transformation."""

import abc
from typing import TYPE_CHECKING, Any

from onad.base.pipeline import Pipeline

if TYPE_CHECKING:
    from onad.base.pipeline import Pipeline


class BaseTransformer(abc.ABC):
    """
    Abstract base class for online transformers.

    This class defines the interface for transformers that can learn from and transform
    data points incrementally. Transformers modify the input data while maintaining
    the streaming nature of the processing.

    Subclasses must implement the `learn_one` and `transform_one` methods.
    """

    @abc.abstractmethod
    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the transformer with a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
               and the values are the corresponding feature values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point.

        Args:
            x: A dictionary representing a single data point to transform.

        Returns:
            A dictionary with transformed feature values.
        """
        raise NotImplementedError

    def __or__(self, other: Any) -> "Pipeline":
        return Pipeline(self, other)

    def __repr__(self) -> str:
        """Return a string representation of the transformer."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the transformer."""
        return self.__repr__()
