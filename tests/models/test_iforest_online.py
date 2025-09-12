"""Unit tests for Online Isolation Forest - Edge cases and initialization only.

Real dataset tests are in tests/integration/test_iforest_models.py
"""

import unittest

import numpy as np

from onad.model.iforest.online import OnlineIsolationForest


class TestOnlineIsolationForestEdgeCases(unittest.TestCase):
    """Test Online Isolation Forest edge cases and initialization."""

    def create_model(self):
        return OnlineIsolationForest(
            num_trees=3,  # Small for fast testing
            max_leaf_samples=8,
            type="fixed",
            subsample=1.0,
            window_size=50,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

    def setUp(self):
        self.model = self.create_model()

    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters."""
        model = OnlineIsolationForest(
            num_trees=5,
            max_leaf_samples=16,
            type="fixed",
            subsample=0.8,
            window_size=256,
            branching_factor=3,
            metric="axisparallel",
            n_jobs=1,
        )

        self.assertEqual(model.num_trees, 5)
        self.assertEqual(model.max_leaf_samples, 16)
        self.assertEqual(model.type, "fixed")
        self.assertEqual(model.subsample, 0.8)
        self.assertEqual(model.window_size, 256)
        self.assertEqual(model.branching_factor, 3)

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters raises errors."""
        # Invalid num_trees
        with self.assertRaises(ValueError):
            OnlineIsolationForest(num_trees=0)

        # Invalid max_leaf_samples
        with self.assertRaises(ValueError):
            OnlineIsolationForest(max_leaf_samples=0)

        # Invalid type
        with self.assertRaises(ValueError):
            OnlineIsolationForest(type="invalid_type")

        # Invalid subsample
        with self.assertRaises(ValueError):
            OnlineIsolationForest(subsample=0.0)

        with self.assertRaises(ValueError):
            OnlineIsolationForest(subsample=1.5)

        # Invalid branching_factor
        with self.assertRaises(ValueError):
            OnlineIsolationForest(branching_factor=1)

    def test_empty_dict_handling(self):
        """Test handling of empty feature dictionary."""
        model = self.create_model()

        # Test with empty dict - should handle gracefully
        try:
            model.learn_one({})
            score = model.score_one({})
            self.assertEqual(score, 0.0)
        except (ValueError, KeyError):
            # Acceptable to reject empty dict
            pass

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        model = self.create_model()

        # Test with NaN values
        nan_point = {"feature1": float("nan"), "feature2": 1.0}
        try:
            model.learn_one(nan_point)
            score = model.score_one(nan_point)
            # If it doesn't raise exception, score should be valid
            self.assertFalse(np.isnan(score))
        except (ValueError, RuntimeError):
            # Acceptable to reject NaN values
            pass

        # Test with Inf values
        inf_point = {"feature1": float("inf"), "feature2": 1.0}
        try:
            model.learn_one(inf_point)
            score = model.score_one(inf_point)
            self.assertFalse(np.isnan(score))
        except (ValueError, RuntimeError, OverflowError):
            # Acceptable to reject Inf values
            pass

    def test_window_size_boundary(self):
        """Test window size boundary behavior."""
        model = OnlineIsolationForest(
            num_trees=2,
            max_leaf_samples=4,
            type="fixed",
            window_size=10,  # Very small window for testing
            n_jobs=1,
        )

        # Add points up to window size
        for i in range(10):
            point = {"feature1": float(i), "feature2": float(i * 2)}
            model.learn_one(point)

        # Verify window size constraint
        self.assertLessEqual(model.data_size, 10)

        # Add more points (should trigger window sliding)
        for i in range(5):
            point = {"feature1": float(i + 10), "feature2": float((i + 10) * 2)}
            model.learn_one(point)

        # Window size should still be constrained
        self.assertLessEqual(model.data_size, 10)

        # Model should still function
        test_point = {"feature1": 99.0, "feature2": 198.0}
        score = model.score_one(test_point)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_type_variants(self):
        """Test different multiplier types (fixed vs adaptive)."""
        for type_name in ["fixed", "adaptive"]:
            with self.subTest(type=type_name):
                model = OnlineIsolationForest(
                    num_trees=2,
                    max_leaf_samples=8,
                    type=type_name,
                    window_size=50,
                    n_jobs=1,
                )

                # Should initialize without error
                self.assertEqual(model.type, type_name)

                # Basic functionality test
                point = {"feature1": 1.0, "feature2": 2.0}
                model.learn_one(point)
                score = model.score_one(point)

                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_feature_consistency(self):
        """Test behavior with inconsistent feature sets."""
        model = self.create_model()

        # Train with one feature set
        point1 = {"feature1": 1.0, "feature2": 2.0}
        model.learn_one(point1)

        # Score with different feature set - should handle gracefully
        point2 = {"feature3": 3.0, "feature4": 4.0}
        try:
            score = model.score_one(point2)
            self.assertIsInstance(score, (int, float))
        except (ValueError, KeyError):
            # Acceptable to reject inconsistent features
            pass

    def test_extreme_values(self):
        """Test with extreme numeric values."""
        model = self.create_model()

        # Very large values
        large_point = {"feature1": 1e10, "feature2": -1e10}
        try:
            model.learn_one(large_point)
            score = model.score_one(large_point)
            self.assertIsInstance(score, (int, float))
        except (ValueError, OverflowError):
            # Acceptable to have issues with extreme values
            pass

        # Very small values
        small_point = {"feature1": 1e-10, "feature2": -1e-10}
        try:
            model.learn_one(small_point)
            score = model.score_one(small_point)
            self.assertIsInstance(score, (int, float))
        except ValueError:
            # Acceptable to have issues with extreme values
            pass


if __name__ == "__main__":
    unittest.main()
