"""Integration test for the OnlineIsolationForest model."""

import unittest

from sklearn.metrics import average_precision_score

from onad.model.iforest.online import OnlineIsolationForest
from onad.stream.dataset import Dataset, load


class TestOnlineIsolationForest(unittest.TestCase):
    """Test OnlineIsolationForest with the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """
        Tests the OnlineIsolationForest model on the SHUTTLE dataset and snapshots the PR-AUC score.
        """
        # Test configuration
        WARMUP_SAMPLES = 1000
        MAX_TEST_SAMPLES = 2000
        DATASET = Dataset.SHUTTLE

        # Create model (non-deterministic, no seed)
        model = OnlineIsolationForest(
            num_trees=25,
            max_leaf_samples=32,
            type="adaptive",
            subsample=1.0,
            window_size=1024,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
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

        # For non-deterministic models, assert a reasonable performance range
        lower_bound, upper_bound = 0.2, 0.98
        self.assertGreaterEqual(
            pr_auc,
            lower_bound,
            f"PR-AUC {pr_auc:.3f} is below expected range [{lower_bound}, {upper_bound}]",
        )
        self.assertLessEqual(
            pr_auc,
            upper_bound,
            f"PR-AUC {pr_auc:.3f} is above expected range [{lower_bound}, {upper_bound}]",
        )


if __name__ == "__main__":
    unittest.main()
