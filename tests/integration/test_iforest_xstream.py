"""Integration test for the XStream model."""

import unittest

from sklearn.metrics import average_precision_score

from onad.model.iforest.xstream import XStream
from onad.stream.dataset import Dataset, load


class TestXStream(unittest.TestCase):
    """Test XStream on the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """Smoke-test XStream quality on SHUTTLE with bounded PR-AUC range."""
        WARMUP_SAMPLES = 1000
        MAX_TEST_SAMPLES = 2000
        DATASET = Dataset.SHUTTLE

        model = XStream(
            k=32,
            n_chains=32,
            depth=10,
            cms_width=256,
            cms_num_hashes=3,
            window_size=128,
            init_sample_size=128,
            density=0.25,
            seed=42,
        )

        dataset_stream = load(DATASET)
        labels, scores = [], []
        warmup_count = 0
        test_count = 0

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

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.05, 0.99
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
