import abc

from onad.base.pipeline import Pipeline


class BaseTransformer(abc.ABC):
    @abc.abstractmethod
    def learn_one(self, x: dict[str, float]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def __or__(self, other):
        """Overload the | operator to pipe the output of this transform to another transform or model."""
        return Pipeline(self, other)
