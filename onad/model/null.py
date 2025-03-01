from typing import Dict

from onad.base.model import BaseModel


class NullModel(BaseModel):

    def learn_one(self, x: Dict[str, float]) -> None:
        return None

    def score_one(self, x: Dict[str, float]) -> float:
        return 0
