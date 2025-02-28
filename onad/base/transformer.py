import abc
from typing import Dict

from onad.base.pipeline import Pipeline


class BaseTransformer(abc.ABC):

    @abc.abstractmethod
    def learn_one(self, x: Dict[str, float]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def __or__(self, other):
        """Overload the | operator to pipe the output of this transformer to another transformer or model."""
        return Pipeline(self, other)
