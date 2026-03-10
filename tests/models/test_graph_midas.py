"""Unit tests for the MIDAS graph-stream anomaly detector."""

import unittest

from aberrant.model.graph import MIDAS


class TestMIDAS(unittest.TestCase):
    """Test suite for MIDAS."""

    def create_model(self, **overrides: object) -> MIDAS:
        defaults: dict[str, object] = {
            "source_key": "src",
            "destination_key": "dst",
            "time_key": "t",
            "count_min_rows": 4,
            "count_min_cols": 256,
            "warm_up_samples": 16,
            "use_relational": True,
            "normalize_score": False,
            "eps": 1e-9,
            "seed": 42,
        }
        defaults.update(overrides)
        return MIDAS(**defaults)

    def test_initialization_defaults(self) -> None:
        model = MIDAS()
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.count_min_rows, 4)
        self.assertEqual(model.count_min_cols, 2048)
        self.assertEqual(model.warm_up_samples, 128)
        self.assertTrue(model.use_relational)
        self.assertFalse(model.normalize_score)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            MIDAS(source_key="")
        with self.assertRaises(ValueError):
            MIDAS(destination_key="")
        with self.assertRaises(ValueError):
            MIDAS(source_key="x", destination_key="x")
        with self.assertRaises(ValueError):
            MIDAS(time_key="")
        with self.assertRaises(ValueError):
            MIDAS(time_key="src")
        with self.assertRaises(ValueError):
            MIDAS(count_min_rows=0)
        with self.assertRaises(ValueError):
            MIDAS(count_min_cols=0)
        with self.assertRaises(ValueError):
            MIDAS(warm_up_samples=0)
        with self.assertRaises(ValueError):
            MIDAS(eps=0.0)

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

    def test_relational_toggle_disables_node_sketches(self) -> None:
        model = self.create_model(use_relational=False)
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        score = model.score_one({"src": 1.0, "dst": 2.0, "t": 2.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertIsNone(model._source_current)
        self.assertIsNone(model._source_total)
        self.assertIsNone(model._destination_current)
        self.assertIsNone(model._destination_total)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7, warm_up_samples=8)
        model2 = self.create_model(seed=7, warm_up_samples=8)

        for i in range(240):
            point = {
                "src": float((i * 3) % 23),
                "dst": float((i * 5) % 29),
                "t": float(i // 8),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"src": 3.0, "dst": 17.0, "t": 31.0}
        self.assertAlmostEqual(
            model1.score_one(query), model2.score_one(query), places=12
        )

    def test_outlier_scores_higher_than_frequent_edge(self) -> None:
        model = self.create_model(
            count_min_rows=6,
            count_min_cols=512,
            warm_up_samples=32,
            seed=11,
        )
        for i in range(800):
            src = float(i % 10)
            dst = float((i % 10 + 1) % 10)
            model.learn_one({"src": src, "dst": dst, "t": float(i // 16)})

        normal_score = model.score_one({"src": 2.0, "dst": 3.0, "t": 49.0})
        outlier_score = model.score_one({"src": 999.0, "dst": 998.0, "t": 49.0})
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
                    "t": float(i // 4),
                }
            )

        self.assertEqual(model._edge_current.table.shape, baseline_shape)
        self.assertEqual(model._edge_total.table.shape, baseline_shape)
        if model._source_current is not None and model._destination_total is not None:
            self.assertEqual(model._source_current.table.shape, baseline_shape)
            self.assertEqual(model._destination_total.table.shape, baseline_shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model(warm_up_samples=8)
        for i in range(100):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i // 2)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertEqual(model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0}), 0.0)

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(normalize_score=True, warm_up_samples=8)
        for i in range(64):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i // 4)})

        score = model.score_one({"src": 999.0, "dst": 998.0, "t": 16.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(count_min_rows=9, count_min_cols=257)
        output = repr(model)
        self.assertIn("MIDAS", output)
        self.assertIn("count_min_rows=9", output)
        self.assertIn("count_min_cols=257", output)


if __name__ == "__main__":
    unittest.main()
