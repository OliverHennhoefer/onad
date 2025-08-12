from sklearn.metrics import average_precision_score, roc_auc_score

from onad.model.unsupervised.forest.online_iforest import OnlineIsolationForest
from onad.stream.streamer import ParquetStreamer, Dataset

# Create True Online Isolation Forest with original algorithm
model = OnlineIsolationForest(
    num_trees=20,
    max_leaf_samples=32,
    type='adaptive',
    subsample=1.0,
    window_size=512,
    branching_factor=2,
    metric='axisparallel',
    n_jobs=1
)

labels, scores = [], []

with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer):
        # Train only on normal data (label 0) for initial learning
        if y == 0 and i < 5000:
            model.learn_one(x)
            continue
        
        # Online learning and scoring
        model.learn_one(x)
        score = model.score_one(x)
        
        labels.append(y)
        scores.append(score)


# Calculate performance metrics
pr_auc = round(average_precision_score(labels, scores), 3)
roc_auc = round(roc_auc_score(labels, scores), 3)

print()
print("=== True Online Isolation Forest Results ===")
print(f"PR-AUC: {pr_auc}")
print(f"ROC-AUC: {roc_auc}")
print(f"Total samples processed: {len(labels)}")
print()

# Compare with some statistics
import numpy as np
print("=== Score Statistics ===")
print(f"Mean score: {np.mean(scores):.3f}")
print(f"Std score: {np.std(scores):.3f}")
print(f"Min score: {np.min(scores):.3f}")
print(f"Max score: {np.max(scores):.3f}")

# Show score distribution for normal vs anomaly
normal_scores = [score for score, label in zip(scores, labels) if label == 0]
anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]

if normal_scores and anomaly_scores:
    print()
    print("=== Score Distribution ===")
    print(f"Normal samples - Mean: {np.mean(normal_scores):.3f}, Std: {np.std(normal_scores):.3f}")
    print(f"Anomaly samples - Mean: {np.mean(anomaly_scores):.3f}, Std: {np.std(anomaly_scores):.3f}")
    print(f"Score separation: {np.mean(anomaly_scores) - np.mean(normal_scores):.3f}")