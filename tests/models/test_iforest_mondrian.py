"""Tests for Mondrian Forest anomaly detection model."""

import unittest

import numpy as np

from onad.model.iforest.mondrian import MondrianForest, MondrianNode, MondrianTree
from tests.utils import DataGenerator


class TestMondrianNode(unittest.TestCase):
    """Test suite for MondrianNode."""

    def test_initialization(self):
        """Test MondrianNode initialization."""
        node = MondrianNode()

        self.assertIsNone(node.split_feature)
        self.assertIsNone(node.split_threshold)
        self.assertIsNone(node.left_child)
        self.assertIsNone(node.right_child)
        self.assertTrue(node.is_leaf_)
        self.assertIsNone(node.min)
        self.assertIsNone(node.max)
        self.assertEqual(node.count, 0)

    def test_is_leaf(self):
        """Test is_leaf method."""
        node = MondrianNode()
        self.assertTrue(node.is_leaf())

        # After splitting, should not be leaf
        node.is_leaf_ = False
        self.assertFalse(node.is_leaf())

    def test_update_stats_first_point(self):
        """Test updating statistics with first data point."""
        node = MondrianNode()
        x_values = np.array([1.0, 2.0, 3.0])

        node.update_stats(x_values)

        self.assertEqual(node.count, 1)
        np.testing.assert_array_equal(node.min, x_values)
        np.testing.assert_array_equal(node.max, x_values)

    def test_update_stats_multiple_points(self):
        """Test updating statistics with multiple data points."""
        node = MondrianNode()

        # First point
        node.update_stats(np.array([1.0, 5.0, 3.0]))

        # Second point with different extremes
        node.update_stats(np.array([3.0, 2.0, 1.0]))

        self.assertEqual(node.count, 2)
        np.testing.assert_array_equal(node.min, np.array([1.0, 2.0, 1.0]))
        np.testing.assert_array_equal(node.max, np.array([3.0, 5.0, 3.0]))

    def test_attempt_split_with_zero_volume(self):
        """Test attempt_split with zero volume (identical points)."""
        node = MondrianNode()
        rng = np.random.default_rng(42)

        # Add identical points (zero volume)
        same_point = np.array([1.0, 1.0])
        node.update_stats(same_point)
        node.update_stats(same_point)

        # Should not split with zero volume
        result = node.attempt_split(1.0, rng)
        self.assertFalse(result)
        self.assertTrue(node.is_leaf())

    def test_attempt_split_success(self):
        """Test successful split attempt."""
        node = MondrianNode()
        rng = np.random.default_rng(42)

        # Create node with sufficient volume
        node.update_stats(np.array([0.0, 0.0]))
        node.update_stats(np.array([10.0, 10.0]))

        # Force split with high lambda
        result = node.attempt_split(100.0, rng)  # Very high lambda for guaranteed split

        if result:  # Split may still be probabilistic
            self.assertFalse(node.is_leaf())
            self.assertIsNotNone(node.left_child)
            self.assertIsNotNone(node.right_child)
            self.assertIsNotNone(node.split_feature)
            self.assertIsNotNone(node.split_threshold)

            # Children should have proper bounding boxes
            self.assertIsNotNone(node.left_child.min)
            self.assertIsNotNone(node.left_child.max)
            self.assertIsNotNone(node.right_child.min)
            self.assertIsNotNone(node.right_child.max)


class TestMondrianTree(unittest.TestCase):
    """Test suite for MondrianTree."""

    def test_initialization(self):
        """Test MondrianTree initialization."""
        selected_indices = np.array([0, 2, 4])
        lambda_ = 1.0
        rng = np.random.default_rng(42)

        tree = MondrianTree(selected_indices, lambda_, rng)

        np.testing.assert_array_equal(tree.selected_indices, selected_indices)
        self.assertEqual(tree.lambda_, lambda_)
        self.assertIs(tree.rng, rng)
        self.assertIsInstance(tree.root, MondrianNode)
        self.assertEqual(tree.n_samples, 0)

    def test_learn_one_basic(self):
        """Test basic learn_one functionality."""
        selected_indices = np.array([0, 1])
        tree = MondrianTree(selected_indices, 1.0, np.random.default_rng(42))

        # Learn a data point
        x_projected = np.array([1.0, 2.0])
        tree.learn_one(x_projected)

        self.assertEqual(tree.n_samples, 1)
        self.assertEqual(tree.root.count, 1)

    def test_learn_one_multiple_points(self):
        """Test learning multiple data points."""
        selected_indices = np.array([0, 1, 2])
        tree = MondrianTree(selected_indices, 1.0, np.random.default_rng(42))

        # Learn multiple points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]

        for point in points:
            tree.learn_one(point)

        self.assertEqual(tree.n_samples, 3)

    def test_score_one_basic(self):
        """Test basic score_one functionality."""
        selected_indices = np.array([0, 1])
        tree = MondrianTree(selected_indices, 1.0, np.random.default_rng(42))

        # Learn some data first
        tree.learn_one(np.array([1.0, 2.0]))
        tree.learn_one(np.array([3.0, 4.0]))

        # Score a point
        score = tree.score_one(np.array([2.0, 3.0]))

        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)

    def test_score_one_without_learning(self):
        """Test scoring without any learning (should work on leaf)."""
        selected_indices = np.array([0, 1])
        tree = MondrianTree(selected_indices, 1.0, np.random.default_rng(42))

        # Score without learning - should return 0 (path length to root)
        score = tree.score_one(np.array([1.0, 2.0]))
        self.assertEqual(score, 0)


class TestMondrianForest(unittest.TestCase):
    """Test suite for MondrianForest."""

    def setUp(self):
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_with_default_params(self):
        """Test MondrianForest initialization with default parameters."""
        forest = MondrianForest()

        self.assertEqual(forest.n_estimators, 100)
        self.assertEqual(forest.subspace_size, 256)
        self.assertEqual(forest.lambda_, 1.0)
        self.assertIsNone(forest.seed)
        self.assertEqual(len(forest.trees), 0)  # Trees created on first data point
        self.assertEqual(forest.n_samples, 0)
        self.assertIsNone(forest._feature_order)

    def test_initialization_with_custom_params(self):
        """Test MondrianForest initialization with custom parameters."""
        forest = MondrianForest(n_estimators=50, subspace_size=10, lambda_=2.0, seed=42)

        self.assertEqual(forest.n_estimators, 50)
        self.assertEqual(forest.subspace_size, 10)
        self.assertEqual(forest.lambda_, 2.0)
        self.assertEqual(forest.seed, 42)

    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid n_estimators
        with self.assertRaises(ValueError) as context:
            MondrianForest(n_estimators=0)
        self.assertIn("n_estimators must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            MondrianForest(n_estimators=-1)
        self.assertIn("n_estimators must be positive", str(context.exception))

        # Test invalid subspace_size
        with self.assertRaises(ValueError) as context:
            MondrianForest(subspace_size=0)
        self.assertIn("subspace_size must be positive", str(context.exception))

        # Test invalid lambda_
        with self.assertRaises(ValueError) as context:
            MondrianForest(lambda_=0)
        self.assertIn("lambda_ must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            MondrianForest(lambda_=-1)
        self.assertIn("lambda_ must be positive", str(context.exception))

    def test_feature_initialization(self):
        """Test feature order establishment."""
        forest = MondrianForest(n_estimators=5, subspace_size=2, seed=42)

        # Before learning, no features established
        self.assertIsNone(forest._feature_order)
        self.assertEqual(len(forest.trees), 0)

        # First data point establishes features and creates trees
        first_point = {"c": 3.0, "a": 1.0, "b": 2.0}
        forest.learn_one(first_point)

        # Features should be sorted alphabetically
        expected_order = ["a", "b", "c"]
        self.assertEqual(forest._feature_order, expected_order)
        self.assertEqual(len(forest.trees), 5)  # n_estimators trees created

    def test_subspace_size_adaptation(self):
        """Test that subspace_size adapts to available features."""
        # Request more features than available
        forest = MondrianForest(n_estimators=3, subspace_size=10, seed=42)

        # Learn with only 2 features
        forest.learn_one({"x": 1.0, "y": 2.0})

        # subspace_size should be reduced to available features
        self.assertEqual(forest.subspace_size, 2)

    def test_score_normalization(self):
        """Test that scores are properly normalized between 0 and 1."""
        forest = MondrianForest(n_estimators=10, subspace_size=3, seed=42)

        # Train on normal data
        training_data = self.data_generator.generate_streaming_data(n=200, n_features=4)
        for point in training_data:
            forest.learn_one(point)

        # Score various points
        test_points = [
            {
                "feature_0": 0.0,
                "feature_1": 0.0,
                "feature_2": 0.0,
                "feature_3": 0.0,
            },  # Normal
            {
                "feature_0": 100.0,
                "feature_1": 100.0,
                "feature_2": 100.0,
                "feature_3": 100.0,
            },  # Anomaly
            training_data[0],  # Training point
        ]

        for point in test_points:
            score = forest.score_one(point)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(
                score, 0.0, f"Score {score} below 0 for point {point}"
            )
            self.assertLessEqual(score, 1.0, f"Score {score} above 1 for point {point}")

    def test_score_before_learning(self):
        """Test scoring before any learning returns 0."""
        forest = MondrianForest(n_estimators=5, subspace_size=2)

        score = forest.score_one({"feature": 1.0})
        self.assertEqual(score, 0.0)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces reproducible results."""
        # Create two forests with same seed
        forest1 = MondrianForest(n_estimators=5, subspace_size=2, seed=42)
        forest2 = MondrianForest(n_estimators=5, subspace_size=2, seed=42)

        # Generate test data
        test_data = self.data_generator.generate_streaming_data(n=50, n_features=3)

        scores1 = []
        scores2 = []

        # Train and score both forests identically
        for point in test_data:
            forest1.learn_one(point.copy())
            forest2.learn_one(point.copy())

            score1 = forest1.score_one(point)
            score2 = forest2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # Scores should be identical with same seed
        for s1, s2 in zip(scores1, scores2, strict=False):
            self.assertAlmostEqual(
                s1, s2, places=10, msg=f"Scores differ with same seed: {s1} != {s2}"
            )

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        forest1 = MondrianForest(n_estimators=10, subspace_size=3, seed=42)
        forest2 = MondrianForest(n_estimators=10, subspace_size=3, seed=123)

        test_data = self.data_generator.generate_streaming_data(n=30, n_features=4)

        scores1 = []
        scores2 = []

        for point in test_data:
            forest1.learn_one(point.copy())
            forest2.learn_one(point.copy())

            score1 = forest1.score_one(point)
            score2 = forest2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # Most scores should differ with different seeds
        different_count = sum(
            1 for s1, s2 in zip(scores1, scores2, strict=False) if abs(s1 - s2) > 1e-10
        )
        self.assertGreater(
            different_count,
            len(scores1) * 0.5,
            "Expected most scores to differ with different seeds",
        )

    def test_n_estimators_effect(self):
        """Test that different numbers of estimators affect results."""
        forest_few = MondrianForest(n_estimators=5, subspace_size=2, seed=42)
        forest_many = MondrianForest(n_estimators=50, subspace_size=2, seed=42)

        # Train on same data
        training_data = self.data_generator.generate_streaming_data(n=100, n_features=3)

        for point in training_data:
            forest_few.learn_one(point.copy())
            forest_many.learn_one(point.copy())

        # Score test points
        test_point = {"feature_0": 10.0, "feature_1": 10.0, "feature_2": 10.0}

        score_few = forest_few.score_one(test_point)
        score_many = forest_many.score_one(test_point)

        # Both should be valid scores
        self.assertIsInstance(score_few, float)
        self.assertIsInstance(score_many, float)
        self.assertGreaterEqual(score_few, 0.0)
        self.assertLessEqual(score_few, 1.0)
        self.assertGreaterEqual(score_many, 0.0)
        self.assertLessEqual(score_many, 1.0)

    def test_repr_string(self):
        """Test string representation includes key parameters."""
        forest = MondrianForest(n_estimators=25, subspace_size=8, lambda_=2.5, seed=999)

        repr_str = repr(forest)
        self.assertIn("MondrianForest", repr_str)
        self.assertIn("n_estimators=25", repr_str)
        self.assertIn("subspace_size=8", repr_str)
        self.assertIn("lambda_=2.5", repr_str)
        self.assertIn("seed=999", repr_str)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        forest = MondrianForest(n_estimators=5, subspace_size=2, seed=42)

        # Test with zero values
        zero_point = {"a": 0.0, "b": 0.0}
        forest.learn_one(zero_point)
        score = forest.score_one(zero_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with negative values
        negative_point = {"a": -5.0, "b": -10.0}
        forest.learn_one(negative_point)
        score = forest.score_one(negative_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with very small values
        small_point = {"a": 1e-10, "b": 1e-10}
        forest.learn_one(small_point)
        score = forest.score_one(small_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_c_factor_computation(self):
        """Test the c-factor computation for score normalization."""
        forest = MondrianForest(n_estimators=1, subspace_size=1)

        # Test with n_samples <= 1
        forest.n_samples = 1
        c = forest._compute_c_factor()
        self.assertEqual(c, 1.0)

        # Test with n_samples > 1
        forest.n_samples = 100
        c = forest._compute_c_factor()
        self.assertIsInstance(c, float)
        self.assertGreater(c, 0.0)

        # c should increase with sample size
        forest.n_samples = 1000
        c_large = forest._compute_c_factor()
        self.assertGreater(c_large, c)


if __name__ == "__main__":
    unittest.main()
