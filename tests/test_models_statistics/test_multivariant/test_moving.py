import unittest
import numpy as np
from collections import deque
from scipy.stats import kurtosis, skew
from onad.model.statistics.multivariant.moving import *

class TestMovingCovariance(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        model = MovingCovariance(window_size=5)
        self.assertEqual(model.window_size, 5)

    def test_initialization_with_negative_window_size(self):
        with self.assertRaises(ValueError):
            MovingCovariance(window_size=-1)

    def test_learn_one_with_valid_input(self):
        model = MovingCovariance(window_size=3)
        point = {'x': 1.0, 'y': 2.0}
        model.learn_one(point)
        self.assertEqual(model.window['x'], deque([1]))
        self.assertEqual(model.window['y'], deque([2]))

    def test_learn_one_with_multiple_points(self):
        model = MovingCovariance(window_size=3)
        points = [{'x': float(i), 'y': float(i) + 1} for i in range(5)]
        for point in points:
            model.learn_one(point)
        self.assertEqual(len(model.window['x']), 3)
        self.assertEqual(len(model.window['y']), 3)

    def test_score_one_with_empty_window(self):
        model = MovingCovariance(window_size=2)
        self.assertEqual(model.score_one({'x': 1, 'y': 1}), 0)

    def test_score_one_with_zero_window(self):
        model = MovingCovariance(window_size=5)
        for _ in range(4):
            model.learn_one({"a": 0, "b": 0})
        self.assertEqual(model.score_one({'a': 0, 'b': 0}), 0)

    def test_score_one_without_bessel(self):
        model = MovingCovariance(window_size=3, bias=True)
        points = [{'x': float(i), 'y': 2 * float(i)} for i in range(1, 3)]
        for point in points:
            model.learn_one(point)
        expected_covariance = ((1-2)*(2-4) + (2-2)*(4-4) + (3-2)*(6-4)) / 3
        self.assertAlmostEqual(model.score_one({'x': 3, 'y': 6}), expected_covariance)

    def test_score_one_with_bessel_correction(self):
        model = MovingCovariance(window_size=3, bias=False)
        points = [{'x': float(i), 'y': 2 * float(i)} for i in range(1, 3)]
        for point in points:
            model.learn_one(point)
        expected_covariance_bessel = ((1-2)*(2-4) + (2-2)*(4-4) + (3-2)*(6-4)) / 2
        self.assertAlmostEqual(model.score_one({'x': 3, 'y': 6}), expected_covariance_bessel)


class TestMovingCorrelationCoefficient(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        model = MovingCorrelationCoefficient(window_size=5)
        self.assertEqual(model.window_size, 5)

    def test_initialization_with_negative_window_size(self):
        with self.assertRaises(ValueError):
            MovingCorrelationCoefficient(window_size=-1)

    def test_learn_one_with_valid_input(self):
        model = MovingCorrelationCoefficient(window_size=3)
        point = {'x': 1.0, 'y': 2.0}
        model.learn_one(point)
        self.assertEqual(model.window['x'], deque([1]))
        self.assertEqual(model.window['y'], deque([2]))

    def test_learn_one_with_multiple_points(self):
        model = MovingCorrelationCoefficient(window_size=3)
        points = [{'x': float(i), 'y': float(i) + 1} for i in range(5)]
        for point in points:
            model.learn_one(point)
        self.assertEqual(len(model.window['x']), 3)
        self.assertEqual(len(model.window['y']), 3)

    def test_score_one_with_empty_window(self):
        model = MovingCorrelationCoefficient(window_size=2)
        self.assertEqual(model.score_one({'x': 3, 'y': 6}), 0)

    def test_score_one_with_zero_window(self):
        model = MovingCorrelationCoefficient(window_size=5)
        for _ in range(4):
            model.learn_one({"a": 0, "b": 0})
        self.assertEqual(model.score_one({"a": 0, "b": 0}), 0)

    def test_score_one_without_bessel(self):
        model = MovingCorrelationCoefficient(window_size=3, bias=True)
        points = [{'x': float(i), 'y': 2 * float(i)} for i in range(1, 3)]
        for point in points:
            model.learn_one(point)
        expected_covariance = ((1-2)*(2-4) + (2-2)*(4-4) + (3-2)*(6-4)) / 3
        std_xy = np.std([1, 2, 3]) * np.std([2, 4, 6])
        
        self.assertAlmostEqual(model.score_one({'x': 3, 'y': 6}), expected_covariance/std_xy)

    def test_score_one_with_bessel_correction(self):
        model = MovingCorrelationCoefficient(window_size=3, bias=False)
        points = [{'x': float(i), 'y': 2 * float(i)} for i in range(1, 3)]
        for point in points:
            model.learn_one(point)
        self.assertAlmostEqual(model.score_one({'x': 3, 'y': 6}), np.corrcoef(model.window["x"], model.window["y"])[1][0])


class TestMovingMahalanobisDistance(unittest.TestCase):

    def test_initialization(self):
        # Test valid initialization
        mmd = MovingMahalanobisDistance(window_size=3)
        self.assertEqual(mmd.window_size, 3)
        
        # Test invalid window size
        with self.assertRaises(ValueError):
            MovingMahalanobisDistance(window_size=-1)

    def test_learn_one(self):
        mmd = MovingMahalanobisDistance(window_size=3)
        
        # Test learning a single data point
        mmd.learn_one({'feature1': 1.0, 'feature2': 2.0})
        self.assertEqual(mmd.window, deque([[1.0, 2.0]]))

        # Test updating with another point
        mmd.learn_one({'feature1': 3.0, 'feature2': 4.0})
        self.assertEqual(len(mmd.window), 2)
        
        # Test handling of non-numeric data#
        mmd.learn_one({'feature1': 'a', 'feature2': 5.0})  # type: ignore
        self.assertEqual(len(mmd.window), 2)  # The invalid entry should not be added

    def test_score_one_insufficient_data_points(self):
        mmd = MovingMahalanobisDistance(window_size=2)
        
        # Test scoring with insufficient data points
        self.assertEqual(mmd.score_one({'feature1': 1.0, 'feature2': 2.0}), 0)

        # Add two valid data points and check the score
        mmd.learn_one({'feature1': 1.0, 'feature2': 2.0})
        mmd.learn_one({'feature1': 3.0, 'feature2': 4.0})
        
        # Test scoring with insufficient data points
        score = mmd.score_one({'feature1': 1.0, 'feature2': 2.0})
        self.assertGreaterEqual(score, 0)

    def test_score_one_singular_matrix(self):
        mmd = MovingMahalanobisDistance(window_size=5)
        mmd.learn_one({'feature1': 1.0, 'feature2': 1.0, 'feature3': 3.0, 'feature4': 4.0})
        mmd.learn_one({'feature1': 2.0, 'feature2': 4.0, 'feature3': 6.0, 'feature4': 8.0})
        mmd.learn_one({'feature1': 3.0, 'feature2': 6.0, 'feature3': 9.0, 'feature4': 12.0})
        self.assertGreaterEqual(mmd.score_one({'feature1': 4.0, 'feature2': 5.0, 'feature3': 6.0, 'feature4': 7.0}), 1)

    def test_score_one(self):
        mmd = MovingMahalanobisDistance(window_size=10)
        values = np.array([[1, 2], [2, 3], [2, 3.5], [3, 5], [5, 10]])
        for point in values:
            mmd.learn_one({"a": point[0], "b": point[1]})
        scored = mmd.score_one({'a': 6, 'b': 11})

        previous_points = np.array(list(values))
        cov_matrix = np.cov(previous_points, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        feature_mean = np.mean(previous_points, axis=0)
        x = np.array([6, 11])
        diff = x - feature_mean
        score =  float(diff.T @ inv_cov_matrix @ diff)
        
        self.assertEqual(scored, score)


if __name__ == '__main__':
    unittest.main()
