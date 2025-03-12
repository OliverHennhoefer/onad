from sklearn.metrics import average_precision_score

from onad.model.unsupervised.mondrian_iforest import MondrianForest
from onad.stream.streamer import ParquetStreamer, Dataset

model = MondrianForest(n_estimators=250, subspace_size=500, random_state=1)

labels, scores = [], []
with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer):
        if y == 0 and i < 10_000:
            model.learn_one(x)
            continue
        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.329
