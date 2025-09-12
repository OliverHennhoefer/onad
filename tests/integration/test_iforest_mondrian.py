"""Integration test for the MondrianForest model."""

import unittest

from sklearn.metrics import average_precision_score

from onad.model.iforest.mondrian import MondrianForest
from onad.stream.dataset import Dataset, load


class TestMondrianForest(unittest.TestCase):
    """Test MondrianForest with the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """
        Tests the MondrianForest model on the SHUTTLE dataset and snapshots the PR-AUC score.
        """
        # Test configuration
        WARMUP_SAMPLES = 1000
        MAX_TEST_SAMPLES = 2000
        SEED = 42
        DATASET = Dataset.SHUTTLE

        # Create model
        model = MondrianForest(
            n_estimators=50, subspace_size=128, lambda_=1.0, seed=SEED
        )

        # Load dataset
        dataset_stream = load(DATASET)

        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        # Process dataset stream
        for _i, (features, label) in enumerate(dataset_stream.stream()):
            if warmup_count < WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(features)
                    warmup_count += 1
                continue

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
        expected_pr_auc = 0.098312

        self.assertAlmostEqual(
            pr_auc,
            expected_pr_auc,
            places=6,
            msg=f"PR-AUC {pr_auc:.10f} should be exactly {expected_pr_auc:.10f}",
        )


if __name__ == "__main__":
    unittest.main()
