from sklearn.metrics import average_precision_score

from onad.model.unsupervised.forest.streamRHF import StreamRandomHistogramForest
from onad.stream.streamer import ParquetStreamer, Dataset

model = StreamRandomHistogramForest(
    n_estimators=25, max_bins=10, window_size=256, seed=1
)

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

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.46
