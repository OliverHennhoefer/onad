from sklearn.metrics import average_precision_score

from onad.model.unsupervised.svm.adaptive_svm import IncrementalOneClassSVMAdaptiveKernel
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler

# Initialisation
scaler = MinMaxScaler()
model = IncrementalOneClassSVMAdaptiveKernel(
    nu=0.1,
    sv_budget=25,
    initial_gamma=0.5,
    adaptation_rate=0.3,
    gamma_bounds=(0.1, 5.0)
)

# Build pipeline
pipeline = scaler | model

# Lists for evaluation
labels, scores = [], []

# Start data stream
with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
    for i, (x, y) in enumerate(streamer):
        # Only use normal classes (y == 0) for training first
        if y == 0 and i < 2_000:
            pipeline.learn_one(x)
            continue

        # Evaluate point
        score = pipeline.score_one(x)

        # Learn from the point
        pipeline.learn_one(x)

        # Store labels and scores for evaluation
        labels.append(y)
        scores.append(score)

# Calculate and print PR_AUC
pr_auc = average_precision_score(labels, scores)
print(f"PR_AUC: {round(pr_auc, 3)}")
