"""Real dataset test for the Autoencoder deep learning model."""

import unittest

try:
    from torch import nn, optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics import average_precision_score

from onad.stream.dataset import Dataset, get_dataset_info, load

if TORCH_AVAILABLE:
    from onad.model.deep.autoencoder import Autoencoder
    from onad.utils.deep.architecture import VanillaAutoencoder


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAutoencoderOnShuttle(unittest.TestCase):
    """Test Autoencoder with the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """
        Tests the Autoencoder model on the SHUTTLE dataset and snapshots the PR-AUC score.
        """
        # Test configuration
        WARMUP_SAMPLES = 1000
        MAX_TEST_SAMPLES = 2000
        SEED = 42
        DATASET = Dataset.SHUTTLE

        # Create model with the correct input size for the dataset
        dataset_info = get_dataset_info(DATASET)
        architecture = VanillaAutoencoder(input_size=dataset_info.n_features, seed=SEED)
        optimizer = optim.Adam(architecture.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        model = Autoencoder(
            model=architecture, optimizer=optimizer, criterion=criterion
        )

        # Load dataset
        dataset_stream = load(DATASET)

        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        # Process dataset stream
        for _i, (features, label) in enumerate(dataset_stream.stream()):
            # Warmup phase: train only on normal samples
            if warmup_count < WARMUP_SAMPLES:
                if label == 0:  # Normal sample
                    model.learn_one(features)
                    warmup_count += 1
                continue

            # Test phase: learn from all samples and collect scores
            if test_count >= MAX_TEST_SAMPLES:
                break

            model.learn_one(features)
            score = model.score_one(features)

            labels.append(label)
            scores.append(score)
            test_count += 1

        # Calculate and assert PR-AUC
        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)
        expected_pr_auc = 0.4000581594

        self.assertAlmostEqual(
            pr_auc,
            expected_pr_auc,
            places=6,
            msg=f"PR-AUC {pr_auc:.10f} should be exactly {expected_pr_auc:.10f}",
        )


if __name__ == "__main__":
    unittest.main()
