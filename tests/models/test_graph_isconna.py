"""Unit tests for the ISCONNA graph-stream anomaly detector."""

import unittest

from aberrant.model.graph import ISCONNA


class TestISCONNA(unittest.TestCase):
    """Test suite for ISCONNA."""

    def create_model(self, **overrides: object) -> ISCONNA:
        defaults: dict[str, object] = {
            "source_key": "src",
            "destination_key": "dst",
            "time_key": "t",
            "count_min_rows": 4,
            "count_min_cols": 256,
            "time_decay_factor": 0.5,
            "warm_up_samples": 16,
            "normalize_score": False,
            "eps": 1e-9,
            "seed": 42,
        }
        defaults.update(overrides)
        return ISCONNA(**defaults)

    def test_initialization_defaults(self) -> None:
        model = ISCONNA()
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.count_min_rows, 8)
        self.assertEqual(model.count_min_cols, 1024)
        self.assertEqual(model.time_decay_factor, 0.5)
        self.assertEqual(model.warm_up_samples, 64)
        self.assertFalse(model.normalize_score)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            ISCONNA(source_key="")
        with self.assertRaises(ValueError):
            ISCONNA(destination_key="")
        with self.assertRaises(ValueError):
            ISCONNA(source_key="x", destination_key="x")
        with self.assertRaises(ValueError):
            ISCONNA(time_key="")
        with self.assertRaises(ValueError):
            ISCONNA(time_key="src")
        with self.assertRaises(ValueError):
            ISCONNA(count_min_rows=0)
        with self.assertRaises(ValueError):
            ISCONNA(count_min_cols=0)
        with self.assertRaises(ValueError):
            ISCONNA(time_decay_factor=0.0)
        with self.assertRaises(ValueError):
            ISCONNA(warm_up_samples=0)
        with self.assertRaises(ValueError):
            ISCONNA(eps=0.0)

    def test_input_validation(self) -> None:
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"dst": 1.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.0, "dst": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": "bad", "dst": 2.0, "t": 1.0})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.score_one({"src": 1.0, "dst": 2.0, "t": float("inf")})

    def test_score_is_zero_before_warmup(self) -> None:
        model = self.create_model(warm_up_samples=32)
        for i in range(8):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i)})
        self.assertEqual(model.score_one({"src": 1.0, "dst": 2.0, "t": 9.0}), 0.0)

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(time_key=None, warm_up_samples=4)
        sample = {"src": 1.0, "dst": 2.0}

        self.assertEqual(model.score_one(sample), 0.0)
        model.learn_one(sample)
        model.learn_one({"src": 1.0, "dst": 3.0})
        model.learn_one({"src": 2.0, "dst": 3.0})
        model.learn_one({"src": 2.0, "dst": 4.0})

        score = model.score_one({"src": 1.0, "dst": 2.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7, warm_up_samples=8)
        model2 = self.create_model(seed=7, warm_up_samples=8)

        for i in range(200):
            point = {
                "src": float((i * 3) % 23),
                "dst": float((i * 5) % 29),
                "t": float(i),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"src": 3.0, "dst": 17.0, "t": 201.0}
        self.assertAlmostEqual(model1.score_one(query), model2.score_one(query), places=12)

    def test_outlier_scores_higher_than_frequent_edge(self) -> None:
        model = self.create_model(
            count_min_rows=6,
            count_min_cols=512,
            warm_up_samples=32,
            seed=11,
        )
        for i in range(400):
            src = float(i % 20)
            dst = float((i % 20 + 1) % 20)
            model.learn_one({"src": src, "dst": dst, "t": float(i)})

        normal_score = model.score_one({"src": 2.0, "dst": 3.0, "t": 401.0})
        outlier_score = model.score_one({"src": 1000.0, "dst": 1001.0, "t": 401.0})
        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(count_min_rows=5, count_min_cols=128)
        baseline_shape = model._edge_current.table.shape

        for i in range(2_000):
            model.learn_one(
                {
                    "src": float(i % 37),
                    "dst": float((i * 2) % 41),
                    "t": float(i),
                }
            )

        self.assertEqual(model._edge_current.table.shape, baseline_shape)
        self.assertEqual(model._edge_total.table.shape, baseline_shape)
        self.assertEqual(model._source_current.table.shape, baseline_shape)
        self.assertEqual(model._destination_total.table.shape, baseline_shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model(warm_up_samples=8)
        for i in range(100):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertEqual(model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0}), 0.0)

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(normalize_score=True, warm_up_samples=8)
        for i in range(64):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i)})

        score = model.score_one({"src": 999.0, "dst": 998.0, "t": 65.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(count_min_rows=9, count_min_cols=257)
        output = repr(model)
        self.assertIn("ISCONNA", output)
        self.assertIn("count_min_rows=9", output)
        self.assertIn("count_min_cols=257", output)


if __name__ == "__main__":
    unittest.main()
