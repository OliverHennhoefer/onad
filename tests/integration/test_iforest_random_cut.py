"""Integration test for the RandomCutForest model."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.iforest.random_cut import RandomCutForest
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestRandomCutForest(unittest.TestCase):
    """Test RandomCutForest on the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """Smoke-test RandomCutForest quality on SHUTTLE with bounded PR-AUC."""
        DATASET = Dataset.SHUTTLE

        model = RandomCutForest(
            n_trees=12,
            sample_size=256,
            shingle_size=1,
            warmup_samples=256,
            normalize_score=True,
            score_scale=8.0,
            seed=42,
        )

        dataset_stream = load(DATASET)
        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        for features, label in dataset_stream.stream():
            if warmup_count < WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(features)
                    warmup_count += 1
                continue

            if test_count >= MAX_TEST_SHORT:
                break

            score = model.score_one(features)
            model.learn_one(features)
            labels.append(label)
            scores.append(score)
            test_count += 1

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.72, 0.90
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
