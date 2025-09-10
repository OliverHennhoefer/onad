"""
Autoencoder example for online anomaly detection.

This example demonstrates using a vanilla autoencoder for anomaly detection
on the SHUTTLE dataset, with online learning and preprocessing.
"""

from sklearn.metrics import average_precision_score
from torch import nn, optim

from onad.model.deep.autoencoder import Autoencoder
from onad.stream.dataset import Dataset, load
from onad.transform.preprocessing.scaler import MinMaxScaler
from onad.utils.deep.architecture import VanillaAutoencoder

# Hyperparameters
INPUT_SIZE = 9  # SHUTTLE dataset feature count
LEARNING_RATE = 0.005
SEED = 1
WARMUP_SAMPLES = 10_000  # Train only on normal samples during warmup

# Initialize the neural network architecture
architecture = VanillaAutoencoder(input_size=INPUT_SIZE, seed=SEED)

# Create the autoencoder model with optimizer and loss function
autoencoder = Autoencoder(
    model=architecture,
    optimizer=optim.Adam(architecture.parameters(), lr=LEARNING_RATE, weight_decay=0),
    criterion=nn.MSELoss(),
)

# Create preprocessing pipeline: scaling -> autoencoder
scaler = MinMaxScaler()
pipeline = scaler | autoencoder

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
print(f"PR_AUC: {round(pr_auc, 3)}")  # Expected: ~0.298
print(f"Processed {len(labels)} test samples")
