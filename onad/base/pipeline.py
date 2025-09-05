"""Pipeline for chaining transformers and models together."""

from typing import Any

from .exceptions import IncompatibleComponentError, PipelineError


class Pipeline:
    """
    Pipeline for chaining transformers and models.

    A pipeline allows you to chain multiple components together, where the output
    of one component becomes the input of the next. The pipeline can contain
    transformers (which modify data) and models (which score data).

    Args:
        first: The first component in the pipeline (transformer or model).
        second: The second component in the pipeline (transformer or model).

    Example:
        >>> scaler = MinMaxScaler()
        >>> model = RandomModel()
        >>> pipeline = Pipeline(scaler, model)
        >>> pipeline.learn_one({"feature": 1.0})
        >>> score = pipeline.score_one({"feature": 2.0})
    """

    def __init__(self, first: Any, second: Any) -> None:
        self._validate_components(first, second)
        self.first = first
        self.second = second

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Learn from a single data point through the pipeline.

        Args:
            x: A dictionary representing a single data point.
        """
        self.first.learn_one(x)
        transformed_x = self.first.transform_one(x)
        self.second.learn_one(transformed_x)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point through the pipeline.

        Args:
            x: A dictionary representing a single data point.

        Returns:
            The transformed data point.
        """
        transformed_x = self.first.transform_one(x)
        return self.second.transform_one(transformed_x)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Score a single data point using the pipeline.

        Args:
            x: A dictionary representing a single data point.

        Returns:
            The anomaly score for the data point.

        Raises:
            PipelineError: If the final component doesn't support scoring.
        """
        transformed_x = self.first.transform_one(x)

        if hasattr(self.second, "score_one"):
            return self.second.score_one(transformed_x)
        else:
            raise PipelineError(
                f"The final component ({self.second.__class__.__name__}) does not have a 'score_one' method."
            )

    def __or__(self, other: Any) -> "Pipeline":
        """Overload the | operator to allow further chaining of pipelines."""
        return Pipeline(self, other)

    def __repr__(self) -> str:
        """Return a string representation of the pipeline."""
        return f"Pipeline({self.first!r} | {self.second!r})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the pipeline."""
        return f"{self.first.__class__.__name__} | {self.second.__class__.__name__}"

    def _validate_components(self, first: Any, second: Any) -> None:
        """
        Validate that the pipeline components are compatible.

        Args:
            first: The first component.
            second: The second component.

        Raises:
            IncompatibleComponentError: If components are not compatible.
        """
        # Check first component has required methods
        if not hasattr(first, "learn_one"):
            raise IncompatibleComponentError(
                first.__class__.__name__, "component with 'learn_one' method"
            )

        if not hasattr(first, "transform_one"):
            raise IncompatibleComponentError(
                first.__class__.__name__, "component with 'transform_one' method"
            )

        # Check second component has required methods
        if not hasattr(second, "learn_one"):
            raise IncompatibleComponentError(
                second.__class__.__name__, "component with 'learn_one' method"
            )
