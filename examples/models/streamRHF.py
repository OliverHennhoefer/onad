from sklearn.metrics import average_precision_score

from onad.dataset import Dataset, load
from onad.model.unsupervised.forest.streamRHF import StreamRandomHistogramForest

model = StreamRandomHistogramForest(
    n_estimators=25, max_bins=10, window_size=256, seed=1
)

labels, scores = [], []
# Load dataset using new API
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 10_000:
        if y == 0:
            model.learn_one(x)
        continue

    model.learn_one(x)
    score = model.score_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.46
