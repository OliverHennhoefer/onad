"""Integration test for the MIDAS graph model on a synthetic edge stream."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from aberrant.model.graph import MIDAS
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestMIDAS(unittest.TestCase):
    """Test MIDAS quality on a synthetic dynamic graph stream."""

    @staticmethod
    def _sample_edge(
        rng: np.random.Generator,
        t: int,
        anomaly_rate: float,
    ) -> tuple[dict[str, float], int]:
        bucket = t // 32
        if rng.random() < anomaly_rate:
            if rng.random() < 0.5:
                src = int(rng.integers(1_000, 1_100))
                dst = int(rng.integers(1_100, 1_200))
            else:
                src = int(rng.integers(200, 260))
                dst = int(rng.integers(260, 320))
            return {"src": float(src), "dst": float(dst), "t": float(bucket)}, 1

        # Nominal behavior: repeated community motifs.
        community = int(rng.integers(0, 4))
        start = community * 25
        src_local = int(rng.integers(0, 25))
        dst_local = (src_local + 1) % 25 if rng.random() < 0.9 else (src_local + 2) % 25
        src = start + src_local
        dst = start + dst_local
        return {"src": float(src), "dst": float(dst), "t": float(bucket)}, 0

    def test_synthetic_edge_stream_pr_auc(self) -> None:
        """Smoke-test MIDAS quality on a synthetic stream with broad PR-AUC bounds."""
        rng = np.random.default_rng(42)
        model = MIDAS(
            source_key="src",
            destination_key="dst",
            time_key="t",
            count_min_rows=4,
            count_min_cols=1024,
            warm_up_samples=128,
            use_relational=True,
            normalize_score=False,
            seed=42,
        )

        labels: list[int] = []
        scores: list[float] = []
        warmup_count = 0
        test_count = 0
        t = 0

        while warmup_count < WARMUP_SAMPLES:
            sample, label = self._sample_edge(rng, t=t, anomaly_rate=0.0)
            if label == 0:
                model.learn_one(sample)
                warmup_count += 1
            t += 1

        while test_count < MAX_TEST_SHORT:
            sample, label = self._sample_edge(rng, t=t, anomaly_rate=0.12)
            score = model.score_one(sample)
            model.learn_one(sample)
            labels.append(label)
            scores.append(score)
            test_count += 1
            t += 1

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        self.assertGreater(sum(labels), 0, "Test stream did not generate anomalies.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.60, 1.00
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
