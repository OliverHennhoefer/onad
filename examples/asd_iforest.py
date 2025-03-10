from onad.metric.pr_auc import PRAUC
from onad.model.unsupervised.asd_iforest import ASDIsolationForest
from onad.stream.streamer import ParquetStreamer, Dataset

model = ASDIsolationForest(n_estimators=750, max_samples=2750, seed=1)

metric = PRAUC(n_thresholds=10)

with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
    for x, y in streamer:
        model.learn_one(x)
        score = model.score_one(x)
        metric.update(y, score)

print(metric.get())
