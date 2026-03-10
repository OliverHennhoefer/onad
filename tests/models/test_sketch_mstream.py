"""Unit tests for the MStream sketch-based anomaly detector."""

import unittest

from aberrant.model.sketch import MStream


class TestMStream(unittest.TestCase):
    """Test suite for MStream."""

    def create_model(self, **overrides: object) -> MStream:
        defaults: dict[str, object] = {
            "rows": 2,
            "buckets": 128,
            "alpha": 0.7,
            "time_key": "t",
            "interaction_order": 2,
            "max_interactions": 8,
            "warm_up_buckets": 1,
            "seed": 42,
        }
        defaults.update(overrides)
        return MStream(**defaults)

    def test_initialization_defaults(self) -> None:
        model = MStream()
        self.assertEqual(model.rows, 2)
        self.assertEqual(model.buckets, 1024)
        self.assertEqual(model.alpha, 0.6)
        self.assertEqual(model.interaction_order, 2)
        self.assertEqual(model.max_interactions, 64)
        self.assertEqual(model.warm_up_buckets, 1)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            MStream(rows=0)
        with self.assertRaises(ValueError):
            MStream(buckets=0)
        with self.assertRaises(ValueError):
            MStream(alpha=0.0)
        with self.assertRaises(ValueError):
            MStream(alpha=1.1)
        with self.assertRaises(ValueError):
            MStream(time_key="")
        with self.assertRaises(ValueError):
            MStream(interaction_order=3)
        with self.assertRaises(ValueError):
            MStream(max_interactions=-1)
        with self.assertRaises(ValueError):
            MStream(warm_up_buckets=0)
        with self.assertRaises(ValueError):
            MStream(eps=0.0)

    def test_input_validation(self) -> None:
        model = self.create_model()

        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0})  # Missing time key.
        with self.assertRaises(ValueError):
            model.learn_one({"t": float("nan"), "x": 1.0})
        with self.assertRaises(ValueError):
            model.score_one({"t": float("inf"), "x": 1.0})

    def test_score_is_zero_before_warmup(self) -> None:
        model = self.create_model(warm_up_buckets=2)
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

        score = model.score_one({"t": 2.0, "a": 1.1, "b": 2.1})
        self.assertEqual(score, 0.0)

    def test_feature_schema_mismatch_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "a": 1.0, "c": 2.0})

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})
        model.learn_one({"t": 2.0, "a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.score_one({"t": 1.0, "a": 1.0, "b": 2.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(
            time_key=None,
            interaction_order=1,
            warm_up_buckets=1,
            max_interactions=None,
        )

        first_point = {"a": 1.0, "b": 2.0}
        second_point = {"a": 1.1, "b": 2.1}

        self.assertEqual(model.score_one(first_point), 0.0)
        model.learn_one(first_point)

        score = model.score_one(second_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7)
        model2 = self.create_model(seed=7)

        for i in range(60):
            point = {
                "t": float(i // 4),
                "a": float(i % 5),
                "b": float((i * 3) % 7),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"t": 30.0, "a": 2.5, "b": 4.5}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_outlier_scores_higher_than_normal(self) -> None:
        model = self.create_model(
            rows=2,
            buckets=256,
            alpha=0.5,
            interaction_order=1,
            max_interactions=None,
            warm_up_buckets=3,
            seed=11,
        )

        for i in range(320):
            point = {
                "t": float(i // 16),
                "a": float((i % 7) * 0.02),
                "b": float((i % 5) * 0.02),
            }
            model.learn_one(point)

        normal_score = model.score_one({"t": 50.0, "a": 0.04, "b": 0.02})
        outlier_score = model.score_one({"t": 50.0, "a": 10.0, "b": -10.0})

        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_sketch_shapes_are_bounded(self) -> None:
        model = self.create_model(buckets=64, max_interactions=4)
        model.learn_one({"t": 0.0, "a": 1.0, "b": 2.0, "c": 3.0})

        if model._current_sketch is None or model._historical_sketch is None:
            self.fail("Sketch tensors were not initialized")

        current_shape = model._current_sketch.shape
        historical_shape = model._historical_sketch.shape

        for i in range(1, 300):
            model.learn_one(
                {
                    "t": float(i // 3),
                    "a": float(i % 10),
                    "b": float((i * 2) % 11),
                    "c": float((i * 3) % 13),
                }
            )

        self.assertEqual(model._current_sketch.shape, current_shape)
        self.assertEqual(model._historical_sketch.shape, historical_shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        for i in range(30):
            model.learn_one({"t": float(i // 2), "a": float(i), "b": float(i + 1)})

        self.assertTrue(model._ready)
        model.reset()

        self.assertFalse(model._ready)
        self.assertEqual(model._seen_buckets, 0)
        self.assertIsNone(model._feature_order)
        self.assertEqual(model.score_one({"t": 1.0, "a": 1.0, "b": 2.0}), 0.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(rows=3, buckets=256)
        output = repr(model)
        self.assertIn("MStream", output)
        self.assertIn("rows=3", output)
        self.assertIn("buckets=256", output)


if __name__ == "__main__":
    unittest.main()
