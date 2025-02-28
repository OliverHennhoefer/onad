import faiss
import collections

import numpy as np

from onad.base.similarity import BaseSimilaritySearchEngine


class FaissSimilaritySearchEngine(BaseSimilaritySearchEngine):

    def __init__(self, window_size: int, warm_up: int):
        self.window = collections.deque(maxlen=window_size)

        self._check_params(window_size, warm_up)
        self.warm_up: int = warm_up
        self.max_dim: int = 0

        self.index: faiss.Index | None = None
        self.keys: list | None = None

    def append(self, x: dict):
        self.window.append(x)
        self.keys, x = self._get_window_data()
        self.index = faiss.IndexFlatL2(len(self.keys))
        self.index.add(x.astype(dtype=np.float32))  # noqa

    def search(self, item: dict, n_neighbors: int):
        if len(self.window) >= self.warm_up:
            x = np.array(
                [item.get(key, np.nan) for key in self.keys], dtype=float
            ).reshape(1, -1)
            distances, _ = self.index.search(
                x.astype(np.float32), k=n_neighbors
            )  # noqa
            return np.mean(distances)
        return None

    def _get_window_data(self):
        keys = sorted({key for dict_ in self.window for key in dict_.keys()})
        return keys, np.array(
            [[dict_.get(key, np.nan) for key in keys] for dict_ in self.window]
        )

    @staticmethod
    def _check_params(window_size, warm_up):
        if window_size < warm_up:
            raise ValueError(
                f"""
                Invalid parameters window_size ({window_size}) and warm_up ({warm_up}).
                Parameter 'window_size' should be greater or equal to 'warm_up'.
                """
            )
        elif warm_up <= 0:
            raise ValueError(
                f"""
                Parameter warm_up ({warm_up}) should be larger than 0.
                """
            )
