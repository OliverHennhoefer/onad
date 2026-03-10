"""Integration test for the ISCONNA graph model on a synthetic edge stream."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from aberrant.model.graph import ISCONNA
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestISCONNA(unittest.TestCase):
    """Test ISCONNA quality on a synthetic dynamic graph stream."""

    @staticmethod
    def _sample_edge(
        rng: np.random.Generator,
        t: int,
        anomaly_rate: float,
    ) -> tuple[dict[str, float], int]:
        if rng.random() < anomaly_rate:
            if rng.random() < 0.5:
                # Sparse anomalies in mostly unseen node ranges.
                src = int(rng.integers(1_000, 1_100))
                dst = int(rng.integers(1_100, 1_200))
            else:
                # Cross-community anomalies with otherwise frequent nodes.
                src_community = int(rng.integers(0, 4))
                dst_community = (src_community + int(rng.integers(1, 4))) % 4
                src = src_community * 50 + int(rng.integers(0, 50))
                dst = dst_community * 50 + int(rng.integers(0, 50))
            return {"src": float(src), "dst": float(dst), "t": float(t)}, 1

        # Nominal behavior: repeated intra-community motifs.
        community = int(rng.integers(0, 4))
        start = community * 50
        src_local = int(rng.integers(0, 50))
        if rng.random() < 0.85:
            dst_local = (src_local + 1) % 50
        else:
            dst_local = (src_local + 2) % 50
        src = start + src_local
        dst = start + dst_local
        return {"src": float(src), "dst": float(dst), "t": float(t)}, 0

    def test_synthetic_edge_stream_pr_auc(self) -> None:
        """Smoke-test ISCONNA quality on a synthetic stream with broad PR-AUC bounds."""
        rng = np.random.default_rng(42)
        model = ISCONNA(
            source_key="src",
            destination_key="dst",
            time_key="t",
            count_min_rows=8,
            count_min_cols=1024,
            time_decay_factor=0.5,
            warm_up_samples=128,
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

        lower_bound, upper_bound = 0.65, 1.00
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
