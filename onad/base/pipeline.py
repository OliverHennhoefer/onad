from typing import Dict


class Pipeline:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def learn_one(self, x: Dict[str, float]) -> None:
        """Learn from the input data."""
        self.first.learn_one(x)
        transformed_x = self.first.transform_one(x)
        self.second.learn_one(transformed_x)

    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        """Transform the input data."""
        transformed_x = self.first.transform_one(x)
        return self.second.transform_one(transformed_x)

    def score_one(self, x: Dict[str, float]) -> float:
        """
        Score the input data using the model in the pipeline.

        Args:
            x (Dict[str, float]): A dictionary of feature-value pairs.

        Returns:
            float: The score for the input data.
        """
        # Transform the input data using the preprocessor
        transformed_x = self.first.transform_one(x)

        # Check if the second component (model) has a score_one method
        if hasattr(self.second, "score_one"):
            return self.second.score_one(transformed_x)
        else:
            raise AttributeError(
                f"The second component ({self.second.__class__.__name__}) does not have a 'score_one' method."
            )
