from onad.base.model import BaseModel


class NullModel(BaseModel):
    def learn_one(self, x: dict[str, float]) -> None:
        return None

    def score_one(self, x: dict[str, float]) -> float:
        return 0
