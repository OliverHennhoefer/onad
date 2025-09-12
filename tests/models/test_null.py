"""Tests for NullModel baseline anomaly detector."""

import unittest

from onad.model.null import NullModel
from tests.utils import DataGenerator


class TestNullModel(unittest.TestCase):
    """Test suite for NullModel."""

    def create_model(self) -> NullModel:
        """Create NullModel instance for testing."""
        return NullModel()

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_always_returns_zero(self):
        """Test that NullModel always returns score of 0.0."""
        model = NullModel()

        # Test without learning
        score = model.score_one({"feature1": 1.0})
        self.assertEqual(score, 0.0)

        # Test after learning
        test_data = self.data_generator.generate_streaming_data(n=100)
        for point in test_data:
            model.learn_one(point)

        # Should still return 0.0
        score = model.score_one({"feature1": 999.0})
        self.assertEqual(score, 0.0)

    def test_learn_one_does_nothing(self):
        """Test that learn_one doesn't change model state."""
        model = NullModel()

        # Get initial representation
        initial_repr = repr(model)

        # Learn some data
        test_data = self.data_generator.generate_streaming_data(n=100)
        for point in test_data:
            model.learn_one(point)

        # Representation should be unchanged
        final_repr = repr(model)
        self.assertEqual(initial_repr, final_repr)

    def test_consistency_across_data_types(self):
        """Test that model returns 0.0 for various data types."""
        model = NullModel()

        test_cases = [
            {"single_feature": 1.0},
            {"x": 0.0, "y": 100.0},
            {"a": -999.0, "b": 0.001, "c": 1e6},
            {"small": 1e-10},
            {"large": 1e10},
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                score = model.score_one(test_case)
                self.assertEqual(
                    score, 0.0, f"Expected 0.0 for {test_case}, got {score}"
                )

    def test_repr_string(self):
        """Test string representation."""
        model = NullModel()
        repr_str = repr(model)

        self.assertEqual(repr_str, "NullModel()")
        self.assertIsInstance(repr_str, str)

    def test_multiple_instances_identical(self):
        """Test that multiple NullModel instances behave identically."""
        model1 = NullModel()
        model2 = NullModel()

        test_data = self.data_generator.generate_streaming_data(n=50)

        scores1 = []
        scores2 = []

        for point in test_data:
            model1.learn_one(point)
            model2.learn_one(point)

            score1 = model1.score_one(point)
            score2 = model2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # All scores should be identical (0.0)
        self.assertEqual(scores1, scores2)
        self.assertTrue(all(score == 0.0 for score in scores1))

    def test_edge_cases(self):
        """Test edge cases for NullModel."""
        model = NullModel()

        # Test with extreme values
        extreme_cases = [
            {"feature": float("1e308")},  # Very large
            {"feature": float("1e-308")},  # Very small
            {"feature": 0.0},  # Zero
            {"feature": -1000.0},  # Negative
        ]

        for case in extreme_cases:
            with self.subTest(case=case):
                model.learn_one(case)
                score = model.score_one(case)
                self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
