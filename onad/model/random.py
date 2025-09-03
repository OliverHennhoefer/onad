import sys

import numpy as np

from onad.base.model import BaseModel


class RandomModel(BaseModel):
    def __init__(self, seed: int = 1):
        self._rng = np.random.default_rng(seed)

    def learn_one(self, x: dict[str, float]) -> None:
        return None

    def score_one(self, x: dict[str, float]) -> float:
        return self._rng.uniform(low=sys.float_info.min, high=1)
