"""Integration test for the MStream model."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.sketch import MStream
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestMStream(unittest.TestCase):
    """Test MStream on the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self) -> None:
        """Smoke-test MStream quality on SHUTTLE with bounded PR-AUC range."""
        dataset_stream = load(Dataset.SHUTTLE)
        model = MStream(
            rows=1,
            buckets=512,
            alpha=0.5,
            time_key="t",
            interaction_order=2,
            max_interactions=8,
            warm_up_buckets=4,
            seed=42,
        )

        labels: list[int] = []
        scores: list[float] = []
        warmup_count = 0
        test_count = 0

        for i, (features, label) in enumerate(dataset_stream.stream()):
            sample = dict(features)
            sample["t"] = float(i // 128)

            if warmup_count < WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(sample)
                    warmup_count += 1
                continue

            if test_count >= MAX_TEST_SHORT:
                break

            score = model.score_one(sample)
            model.learn_one(sample)
            labels.append(label)
            scores.append(score)
            test_count += 1

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.65, 0.90
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
