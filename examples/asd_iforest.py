from onad.metric.pr_auc import PRAUC
from onad.model.asd_iforest import ASDIsolationForest
from onad.utils.streamer.datasets import Dataset
from onad.utils.streamer.streamer import NPZStreamer

iforest = ASDIsolationForest(n_estimators=450, max_samples=2048)
metric = PRAUC(n_thresholds=10)

with NPZStreamer(Dataset.FRAUD) as streamer:
    for x, y in streamer:
        iforest.learn_one(x)
        score = iforest.score_one(x)
        metric.update(y, score)

print(metric.get())
