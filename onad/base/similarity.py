import abc


class BaseSimilaritySearchEngine(abc.ABC):
    @abc.abstractmethod
    def append(self, x: dict[str, float]) -> None:
        pass

    @abc.abstractmethod
    def search(self, x: dict[str, float], n_neighbors: int) -> float:
        pass
