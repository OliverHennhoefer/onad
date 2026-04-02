"""Unit tests for the AnoEdgeL graph-stream anomaly detector."""

import unittest

import numpy as np

from aberrant.model.graph import AnoEdgeL


class TestAnoEdgeL(unittest.TestCase):
    """Test suite for AnoEdgeL."""

    def create_model(self, **overrides: object) -> AnoEdgeL:
        defaults: dict[str, object] = {
            "source_key": "src",
            "destination_key": "dst",
            "time_key": "t",
            "count_min_rows": 64,
            "count_min_cols": 64,
            "num_hashes": 4,
            "local_radius": 1,
            "time_decay_factor": 1.0,
            "warm_up_samples": 16,
            "normalize_score": False,
            "predict_threshold": 0.5,
            "eps": 1e-9,
            "seed": 42,
        }
        defaults.update(overrides)
        return AnoEdgeL(**defaults)

    def test_initialization_defaults(self) -> None:
        model = AnoEdgeL()
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.count_min_rows, 256)
        self.assertEqual(model.count_min_cols, 256)
        self.assertEqual(model.num_hashes, 4)
        self.assertEqual(model.local_radius, 2)
        self.assertEqual(model.time_decay_factor, 1.0)
        self.assertEqual(model.warm_up_samples, 128)
        self.assertFalse(model.normalize_score)
        self.assertEqual(model.predict_threshold, 0.5)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            AnoEdgeL(source_key="")
        with self.assertRaises(ValueError):
            AnoEdgeL(destination_key="")
        with self.assertRaises(ValueError):
            AnoEdgeL(source_key="x", destination_key="x")
        with self.assertRaises(ValueError):
            AnoEdgeL(time_key="")
        with self.assertRaises(ValueError):
            AnoEdgeL(time_key="src")
        with self.assertRaises(ValueError):
            AnoEdgeL(count_min_rows=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(count_min_cols=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(num_hashes=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(local_radius=-1)
        with self.assertRaises(ValueError):
            AnoEdgeL(time_decay_factor=0.0)
        with self.assertRaises(ValueError):
            AnoEdgeL(warm_up_samples=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(normalize_score=True, predict_threshold=1.1)
        with self.assertRaises(ValueError):
            AnoEdgeL(normalize_score=False, predict_threshold=-0.1)
        with self.assertRaises(ValueError):
            AnoEdgeL(eps=0.0)

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
        with self.assertRaises(ValueError):
            model.score_one({"src": 1.0, "dst": 2.0, "t": 1.5})

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

        for i in range(300):
            point = {
                "src": float((i * 3) % 31),
                "dst": float((i * 5) % 29),
                "t": float(i // 4),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"src": 3.0, "dst": 17.0, "t": 80.0}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_outlier_scores_higher_than_frequent_edge(self) -> None:
        model = self.create_model(
            count_min_rows=128,
            count_min_cols=128,
            warm_up_samples=32,
            seed=11,
        )
        for i in range(900):
            src = float(i % 20)
            dst = float((i % 20 + 1) % 20)
            model.learn_one({"src": src, "dst": dst, "t": float(i // 8)})

        normal_score = model.score_one({"src": 2.0, "dst": 3.0, "t": 113.0})
        outlier_score = model.score_one({"src": 999.0, "dst": 998.0, "t": 113.0})
        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(count_min_rows=12, count_min_cols=13, num_hashes=5)
        sketch_shape = model._sketch.shape
        row_shape = model._row_mass.shape
        col_shape = model._col_mass.shape
        total_shape = model._total_mass.shape

        for i in range(5_000):
            model.learn_one(
                {
                    "src": float(i % 101),
                    "dst": float((i * 7) % 103),
                    "t": float(i // 5),
                }
            )

        self.assertEqual(model._sketch.shape, sketch_shape)
        self.assertEqual(model._row_mass.shape, row_shape)
        self.assertEqual(model._col_mass.shape, col_shape)
        self.assertEqual(model._total_mass.shape, total_shape)

    def test_rollover_decay_on_bucket_jump(self) -> None:
        model = self.create_model(time_decay_factor=0.5, warm_up_samples=2)
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        mass_before = float(np.mean(model._total_mass))

        # Scoring triggers rollover when timestamps advance.
        model.score_one({"src": 1.0, "dst": 2.0, "t": 5.0})
        mass_after = float(np.mean(model._total_mass))
        self.assertLess(mass_after, mass_before)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model(warm_up_samples=8)
        for i in range(100):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i // 4)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertEqual(model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0}), 0.0)

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(normalize_score=True, warm_up_samples=8)
        for i in range(96):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i // 4)})

        score = model.score_one({"src": 999.0, "dst": 998.0, "t": 24.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_predict_one_is_binary(self) -> None:
        model = self.create_model(normalize_score=True, predict_threshold=0.2)
        for i in range(96):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i // 4)})

        prediction = model.predict_one({"src": 999.0, "dst": 998.0, "t": 24.0})
        self.assertIn(prediction, (0, 1))

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(count_min_rows=15, count_min_cols=17, num_hashes=6)
        output = repr(model)
        self.assertIn("AnoEdgeL", output)
        self.assertIn("count_min_rows=15", output)
        self.assertIn("count_min_cols=17", output)
        self.assertIn("num_hashes=6", output)


if __name__ == "__main__":
    unittest.main()
