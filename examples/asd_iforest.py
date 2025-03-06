from onad.metric.pr_auc import PRAUC
from onad.model.asd_iforest import ASDIsolationForest
from onad.utils.streamer.streamer import NPZStreamer

model = ASDIsolationForest(n_estimators=750, max_samples=2750, seed=1)

metric = PRAUC(n_thresholds=10)

with NPZStreamer("./data/fraud.npz") as streamer:
    for x, y in streamer:
        model.learn_one(x)
        score = model.score_one(x)
        metric.update(y, score)

print(metric.get())
