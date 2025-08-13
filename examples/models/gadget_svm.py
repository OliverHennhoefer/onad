from sklearn.metrics import average_precision_score

from onad.model.unsupervised.svm.gadget_svm import GADGETSVM
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler

scaler = MinMaxScaler()
model = GADGETSVM()
pipeline = scaler | model
labels, scores = [], []

with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
    for i, (x, y) in enumerate(streamer):
        if i < 2_000:
            if y == 0:
                model.learn_one(x)
            continue
        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.373
