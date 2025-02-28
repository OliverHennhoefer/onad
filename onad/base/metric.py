import abc


class BaseMetric(abc.ABC):

    @abc.abstractmethod
    def update(self, y_true: int, y_pred: float):
        raise NotImplementedError
