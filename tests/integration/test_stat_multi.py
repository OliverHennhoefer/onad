"""Integration tests for multivariate statistical models."""

import unittest

from sklearn.metrics import average_precision_score

from onad.model.stat.multi import MovingCovariance, MovingMahalanobisDistance
from onad.stream.dataset import Dataset, load


class TestMultivariateStatisticalModels(unittest.TestCase):
    """
    Tests multivariate statistical models on the SHUTTLE dataset.
    Each test prepares the feature dict according to the model's needs.
    """

    def setUp(self):
        """Set up test configuration."""
        self.WARMUP_SAMPLES = 1000
        self.MAX_TEST_SAMPLES = 2000

    def test_moving_covariance(self):
        """Test for the bivariate MovingCovariance model."""
        model = MovingCovariance(window_size=100)
        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        dataset_stream = load(Dataset.SHUTTLE)

        for _i, (features, label) in enumerate(dataset_stream.stream()):
            # Prepare bivariate data
            if len(features) < 2:
                continue

            feature_keys = list(features.keys())
            x_bi = {
                feature_keys[0]: features[feature_keys[0]],
                feature_keys[1]: features[feature_keys[1]],
            }

            if warmup_count < self.WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(x_bi)
                    warmup_count += 1
                continue

            if test_count >= self.MAX_TEST_SAMPLES:
                break

            model.learn_one(x_bi)
            score = model.score_one(x_bi)
            labels.append(label)
            scores.append(score)
            test_count += 1

        self.assertGreater(len(scores), 0, "No scores were generated.")
        pr_auc = average_precision_score(labels, scores)

        lower, upper = (0.3, 0.75)
        self.assertGreaterEqual(
            pr_auc,
            lower,
            f"PR-AUC {pr_auc:.3f} is below expected range [{lower}, {upper}]",
        )
        self.assertLessEqual(
            pr_auc,
            upper,
            f"PR-AUC {pr_auc:.3f} is above expected range [{lower}, {upper}]",
        )

    def test_moving_mahalanobis_distance(self):
        """Test for the multivariate MovingMahalanobisDistance model."""
        model = MovingMahalanobisDistance(window_size=100)
        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        dataset_stream = load(Dataset.SHUTTLE)

        for _i, (features, label) in enumerate(dataset_stream.stream()):
            if warmup_count < self.WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(features)
                    warmup_count += 1
                continue

            if test_count >= self.MAX_TEST_SAMPLES:
                break

            model.learn_one(features)
            score = model.score_one(features)
            labels.append(label)
            scores.append(score)
            test_count += 1

        self.assertGreater(len(scores), 0, "No scores were generated.")
        pr_auc = average_precision_score(labels, scores)

        lower, upper = (0.3, 0.8)
        self.assertGreaterEqual(
            pr_auc,
            lower,
            f"PR-AUC {pr_auc:.3f} is below expected range [{lower}, {upper}]",
        )
        self.assertLessEqual(
            pr_auc,
            upper,
            f"PR-AUC {pr_auc:.3f} is above expected range [{lower}, {upper}]",
        )


if __name__ == "__main__":
    unittest.main()
