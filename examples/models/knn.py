"""
K-Nearest Neighbors example for online anomaly detection.

This example demonstrates using KNN with FAISS similarity search for anomaly detection
on the SHUTTLE dataset, with online learning and preprocessing.
"""

from sklearn.metrics import average_precision_score

from onad.model.distance.knn import KNN
from onad.stream.dataset import Dataset, load
from onad.transform.preprocessing.scaler import MinMaxScaler
from onad.utils.similar.faiss_engine import FaissSimilaritySearchEngine

# Hyperparameters
K_NEIGHBORS = 45  # Number of nearest neighbors to consider
WINDOW_SIZE = 250  # Size of sliding window for FAISS engine
WARM_UP = 50  # Minimum samples before search can be performed
WARMUP_SAMPLES = 2_000  # Train only on normal samples during warmup

# Initialize the similarity search engine
engine = FaissSimilaritySearchEngine(window_size=WINDOW_SIZE, warm_up=WARM_UP)

# Create the KNN model with similarity engine
knn = KNN(k=K_NEIGHBORS, similarity_engine=engine)

# Create preprocessing pipeline: scaling -> KNN
scaler = MinMaxScaler()
pipeline = scaler | knn

# Storage for evaluation
labels, scores = [], []

# Load and process the dataset
print("Loading SHUTTLE dataset...")
dataset = load(Dataset.SHUTTLE)

print(f"Starting online learning with {WARMUP_SAMPLES} warmup samples...")
for i, (x, y) in enumerate(dataset.stream()):
    # Warmup phase: train only on normal samples (y == 0)
    if i < WARMUP_SAMPLES:
        if y == 0:
            pipeline.learn_one(x)
        continue

    # Online phase: learn from all samples and compute scores
    pipeline.learn_one(x)
    score = pipeline.score_one(x)

    labels.append(y)
    scores.append(score)

# Evaluate performance
pr_auc = average_precision_score(labels, scores)
print(f"PR_AUC: {round(pr_auc, 3)}")  # Expected: ~0.848
print(f"Processed {len(labels)} test samples")
