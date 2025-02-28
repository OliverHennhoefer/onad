import abc


class BaseStreamer(abc.ABC):

    def __iter__(self):
        raise NotImplementedError
