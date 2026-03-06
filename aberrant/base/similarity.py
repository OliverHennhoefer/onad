"""Base similarity search engine interface."""

import abc


class BaseSimilaritySearchEngine(abc.ABC):
    """
    Abstract base class for similarity search engines.

    This class defines the interface for similarity search engines that can store
    data points and find similar neighbors for query points.

    Subclasses must implement the `append` and `search` methods.
    """

    @abc.abstractmethod
    def append(self, x: dict[str, float]) -> None:
        """
        Add a data point to the search engine.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
               and the values are the corresponding feature values.
        """
        pass

    @abc.abstractmethod
    def search(self, x: dict[str, float], n_neighbors: int) -> float:
        """
        Search for the n nearest neighbors of a data point.

        Args:
            x: A dictionary representing the query data point.
            n_neighbors: The number of nearest neighbors to find.

        Returns:
            A similarity or distance score based on the nearest neighbors.
            Should return a consistent numeric value (e.g., 0.0 if insufficient data).
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the search engine."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the search engine."""
        return self.__repr__()
