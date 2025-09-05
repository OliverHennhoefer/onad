"""
ASD Isolation Forest example for online anomaly detection.

This example demonstrates using ASD (Anomaly Subspace Detection) Isolation Forest
for anomaly detection on the SHUTTLE dataset, with online learning.
"""

from sklearn.metrics import average_precision_score

from onad.model.iforest.asd import ASDIsolationForest
from onad.stream.dataset import Dataset, load

# Hyperparameters
N_ESTIMATORS = 750  # Number of trees in the iforest
MAX_SAMPLES = 2750  # Samples per tree (buffer size)
SEED = 1  # Random seed for reproducibility
WARMUP_SAMPLES = 10_000  # Train only on normal samples during warmup

# Initialize the ASD Isolation Forest model
asd_iforest = ASDIsolationForest(
    n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, seed=SEED
)

# Storage for evaluation
labels, scores = [], []

# Load and process the dataset
print("Loading SHUTTLE dataset...")
dataset = load(Dataset.SHUTTLE)

print(f"Starting online learning with {WARMUP_SAMPLES} warmup samples...")
for i, (x, y) in enumerate(dataset.stream()):
    # Warmup phase: train only on normal samples (y == 0)
    if i < WARMUP_SAMPLES and y == 0:
        asd_iforest.learn_one(x)
        continue

    # Skip non-normal samples during warmup
    if i < WARMUP_SAMPLES:
        continue

    # Online phase: learn from all samples and compute scores
    asd_iforest.learn_one(x)
    score = asd_iforest.score_one(x)

    labels.append(y)
    scores.append(score)

# Evaluate performance
pr_auc = average_precision_score(labels, scores)
print(f"PR_AUC: {round(pr_auc, 3)}")  # Expected: ~0.792
print(f"Processed {len(labels)} test samples")
