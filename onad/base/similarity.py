import abc
from typing import Dict


class BaseSimilaritySearchEngine(abc.ABC):

    @abc.abstractmethod
    def append(self, x: Dict[str, float]) -> None:
        pass

    @abc.abstractmethod
    def search(self, x: Dict[str, float], n_neighbors: int) -> float:
        pass
