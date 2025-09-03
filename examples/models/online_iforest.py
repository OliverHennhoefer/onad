from sklearn.metrics import average_precision_score, roc_auc_score

from onad.model.unsupervised.forest.online_iforest import OnlineIsolationForest
from onad.stream.streamer import Dataset, ParquetStreamer

# Create True Online Isolation Forest with original algorithm
model = OnlineIsolationForest(
    num_trees=20,
    max_leaf_samples=32,
    type="adaptive",
    subsample=1.0,
    window_size=512,
    branching_factor=2,
    metric="axisparallel",
    n_jobs=1,
)

labels, scores = [], []

with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer):
        if y == 0 and i < 5000:
            model.learn_one(x)
            continue

        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)


print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
