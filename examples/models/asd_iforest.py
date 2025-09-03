from sklearn.metrics import average_precision_score

from onad.model.unsupervised.forest.asd_iforest import ASDIsolationForest
from onad.stream.streamer import Dataset, ParquetStreamer

model = ASDIsolationForest(n_estimators=750, max_samples=2750, seed=1)
labels, scores = [], []

with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer):
        if i < 10_000:
            if y == 0:
                model.learn_one(x)
            continue
        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.909
