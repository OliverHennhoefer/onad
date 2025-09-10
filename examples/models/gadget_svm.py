from sklearn.metrics import average_precision_score

from onad.model.svm import GADGETSVM
from onad.stream.dataset import Dataset, load
from onad.transform.preprocessing.scaler import MinMaxScaler

scaler = MinMaxScaler()
model = GADGETSVM()
pipeline = scaler | model
labels, scores = [], []

# Load dataset using new API
dataset = load(Dataset.FRAUD)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 2_000:
        if y == 0:
            model.learn_one(x)
        continue
    model.learn_one(x)
    score = model.score_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.373
