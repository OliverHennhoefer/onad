import unittest
import numpy as np
from collections import deque
from scipy.stats import kurtosis, skew
from onad.model.statistics.univariant.moving import *


class TestMovingAverage(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        ma = MovingAverage(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingAverage(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingAverage(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingAverage(3)
        with self.assertRaises(AssertionError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_mean_correctly(self):
        ma = MovingAverage(3)
        ma.learn_one({"value": 1.0})
        ma.learn_one({"value": 2.0})
        self.assertEqual(ma.score_one(), 1.5)

    def test_score_one_with_empty_window(self):
        ma = MovingAverage(3)
        self.assertEqual(ma.score_one(), 0)

    def test_window_shifting(self):
        ma = MovingAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingHarmonicAverage(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        mah = MovingHarmonicAverage(3)
        self.assertEqual(len(mah.window), 0)

    def test_initialization_with_zero_window_size_raises_error(self):
        with self.assertRaises(ValueError) as context:
            _ = MovingHarmonicAverage(0)
        self.assertEqual(
            str(context.exception), "Window size must be a positive integer."
        )

    def test_initialization_with_negative_window_size_raises_error(self):
        with self.assertRaises(ValueError) as context:
            _ = MovingHarmonicAverage(-5)
        self.assertEqual(
            str(context.exception), "Window size must be a positive integer."
        )

    def test_learn_one_with_single_value_dict(self):
        mah = MovingHarmonicAverage(3)
        mah.learn_one({"value": 10.0})
        self.assertEqual(len(mah.window), 1)
        if mah.feature_names:
            self.assertIn("value", mah.feature_names)

    def test_learn_one_raises_assertion_error_if_more_than_one_key(self):
        mah = MovingHarmonicAverage(3)
        with self.assertRaises(AssertionError) as context:
            mah.learn_one({"a": 10.0, "b": 20.0})
        self.assertEqual(
            str(context.exception), "Dictionary has more than one key-value pair."
        )

    def test_learn_one_raises_assertion_error_if_empty_dict(self):
        mah = MovingHarmonicAverage(3)
        with self.assertRaises(AssertionError) as context:
            mah.learn_one({})
        self.assertEqual(
            str(context.exception), "Dictionary has more than one key-value pair."
        )

    def test_score_one_with_no_values_returns_zero(self):
        mah = MovingHarmonicAverage(3)
        self.assertEqual(mah.score_one(), 0.0)

    def test_score_one_calculates_correct_harmonic_average(self):
        mah = MovingHarmonicAverage(3)
        mah.learn_one({"value": 1.0})
        mah.learn_one({"value": 2.0})
        mah.learn_one({"value": 4.0})
        self.assertAlmostEqual(mah.score_one(), 3 / (1 + 0.5 + 0.25), places=6)

    def test_score_one_with_window_full_updates_harmonic_average(self):
        mah = MovingHarmonicAverage(2)
        mah.learn_one({"value": 1.0})
        mah.learn_one({"value": 4.0})
        self.assertAlmostEqual(mah.score_one(), 2 / (1 + 0.25), places=6)
        mah.learn_one({"value": 16.0})
        # Now the window contains only [4.0, 16.0]
        self.assertAlmostEqual(mah.score_one(), 2 / (0.25 + 0.0625), places=6)

    def test_window_shifting(self):
        ma = MovingAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingGeometricAverage(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        m = MovingGeometricAverage(window_size=5)
        self.assertEqual(len(m.window), 0)

    def test_initialization_with_zero_or_negative_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            MovingGeometricAverage(window_size=0)
        with self.assertRaises(ValueError):
            MovingGeometricAverage(window_size=-1)

    def test_learn_one_with_valid_input(self):
        m = MovingGeometricAverage(window_size=3)
        m.learn_one({"feature": 2.0})
        self.assertEqual(len(m.window), 1)
        self.assertEqual(m.window[0], 2.0)

    def test_learn_one_raises_assertion_error_for_empty_dict(self):
        m = MovingGeometricAverage(window_size=3)
        with self.assertRaises(AssertionError):
            m.learn_one({})

    def test_learn_one_raises_assertion_error_for_multiple_keys(self):
        m = MovingGeometricAverage(window_size=3)
        with self.assertRaises(AssertionError):
            m.learn_one({"feature1": 2.0, "feature2": 4.0})

    def test_score_one_with_empty_window(self):
        m = MovingGeometricAverage(window_size=3)
        self.assertEqual(m.score_one(), 1)

    def test_score_one_with_single_value_in_window(self):
        m = MovingGeometricAverage(window_size=3)
        m.learn_one({"feature": 2.0})
        self.assertEqual(m.score_one(), 1)

    def test_score_one_with_two_values_no_absolute_values(self):
        m = MovingGeometricAverage(window_size=3)
        m.learn_one({"feature": 2.0})
        m.learn_one({"feature": 8.0})
        expected = (2.0 * 8.0) ** (
            1 / 1
        )  # Geometric mean of [2, 8] is sqrt(16) which is 4
        self.assertAlmostEqual(m.score_one(), expected)

    def test_score_one_with_multiple_values_no_absolute_values(self):
        m = MovingGeometricAverage(window_size=3)
        m.learn_one({"feature": 2.0})
        m.learn_one({"feature": 8.0})
        m.learn_one({"feature": 4.0})
        window_product = 2.0 * 8.0 * 4.0
        expected = (window_product) ** (
            1 / 2
        )  # Geometric mean of [2, 8, 4] is sqrt(64) which is 8
        self.assertAlmostEqual(m.score_one(), expected)

    def test_score_one_with_multiple_values_absolute_values(self):
        m = MovingGeometricAverage(window_size=3, absoluteValues=True)
        m.learn_one({"feature": 2.0})
        m.learn_one({"feature": 8.0})
        m.learn_one({"feature": 4.0})
        window_growth = [8.0 / 2.0, 4.0 / 8.0]  # Growth factors: [4.0, 0.5]
        window_product = 4.0 * 0.5
        expected = (window_product) ** (
            1 / 2
        )  # Geometric mean of growth factors [4.0, 0.5] is sqrt(2)
        self.assertAlmostEqual(m.score_one(), expected)

    def test_window_shifting(self):
        ma = MovingAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingMedian(unittest.TestCase):

    def test_initialization_invalid(self):
        # Testing if ValueError is raised for non-positive window size
        with self.assertRaises(ValueError):
            MovingMedian(0)
        with self.assertRaises(ValueError):
            MovingMedian(-5)

    def test_learn_one_valid_input(self):
        # Test learning a single valid input
        mm = MovingMedian(3)
        mm.learn_one({"value": 10})
        self.assertEqual(list(mm.window), [10])

    def test_learn_one_invalid_input_dict_length(self):
        # Test assertion error for dictionary with more than one key-value pair
        mm = MovingMedian(3)
        with self.assertRaises(AssertionError):
            mm.learn_one({"a": 1, "b": 2})

    def test_score_one_empty_window(self):
        # Test scoring when the window is empty
        mm = MovingMedian(3)
        self.assertEqual(mm.score_one(), 0)

    def test_score_one_single_element(self):
        # Test scoring with a single element in the window
        mm = MovingMedian(1)
        mm.learn_one({"value": 5})
        self.assertEqual(mm.score_one(), 5)

    def test_score_one_odd_window_size(self):
        # Test scoring for an odd-sized window
        mm = MovingMedian(3)
        mm.learn_one({"value": 10})
        mm.learn_one({"value": 20})
        mm.learn_one({"value": 30})
        self.assertEqual(mm.score_one(), 20)

    def test_score_one_even_window_size(self):
        # Test scoring for an even-sized window
        mm = MovingMedian(4)
        mm.learn_one({"value": 10})
        mm.learn_one({"value": 20})
        mm.learn_one({"value": 30})
        mm.learn_one({"value": 40})
        self.assertEqual(mm.score_one(), 25)

    def test_window_replacement(self):
        # Test that old values are replaced in a full window
        mm = MovingMedian(3)
        mm.learn_one({"value": 10})
        mm.learn_one({"value": 20})
        mm.learn_one({"value": 30})
        self.assertEqual(list(mm.window), [10, 20, 30])

        # Adding another value should remove the oldest (first added) one
        mm.learn_one({"value": 40})
        self.assertEqual(list(mm.window), [20, 30, 40])

    def test_window_shifting(self):
        ma = MovingAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingQuantile(unittest.TestCase):

    def test_initialization(self):
        with self.assertRaises(ValueError):
            MovingQuantile(window_size=0)

        model = MovingQuantile(window_size=3, quantile=0.5)
        self.assertEqual(model.window.maxlen, 3)
        self.assertEqual(model.quantile, 0.5)

    def test_learn_one_valid(self):
        model = MovingQuantile(window_size=3)
        model.learn_one({"value": 1})
        self.assertEqual(len(model.window), 1)

        model.learn_one({"value": 2})
        self.assertEqual(len(model.window), 2)

        model.learn_one({"value": 3})
        self.assertEqual(len(model.window), 3)

    def test_learn_one_invalid(self):
        model = MovingQuantile(window_size=3)
        with self.assertRaises(AssertionError):
            model.learn_one({})

        with self.assertRaises(AssertionError):
            model.learn_one({"value1": 1, "value2": 2})

    def test_score_empty_window(self):
        model = MovingQuantile(window_size=3)
        self.assertEqual(model.score_one(), 0)

    def test_score_single_value(self):
        model = MovingQuantile(window_size=1)
        model.learn_one({"value": 5})
        self.assertEqual(model.score_one(), 5)

    def test_score_median(self):
        model = MovingQuantile(window_size=3, quantile=0.5)
        model.learn_one({"value": 1})
        model.learn_one({"value": 3})
        model.learn_one({"value": 2})
        self.assertEqual(model.score_one(), 2)

    def test_score_quantile_below_50(self):
        model = MovingQuantile(window_size=4, quantile=0.25)
        model.learn_one({"value": 1})
        model.learn_one({"value": 3})
        model.learn_one({"value": 5})
        model.learn_one({"value": 2})
        self.assertEqual(model.score_one(), np.quantile(model.window, 0.25))

    def test_score_quantile_above_50(self):
        model = MovingQuantile(window_size=4, quantile=0.75)
        model.learn_one({"value": 1})
        model.learn_one({"value": 3})
        model.learn_one({"value": 5})
        model.learn_one({"value": 2})
        self.assertEqual(model.score_one(), np.quantile(model.window, 0.75))

    def test_interpolation(self):
        model = MovingQuantile(window_size=5, quantile=0.6)
        for value in [1, 3, 7, 9, 10]:
            model.learn_one({"value": value})
        # Expected: (7 + 0.8 * (9 - 7))
        self.assertAlmostEqual(model.score_one(), np.quantile(model.window, 0.6))

    def test_window_shifting(self):
        ma = MovingAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingVariance(unittest.TestCase):

    def test_initialization_with_valid_size(self):
        mv = MovingVariance(window_size=5)
        self.assertEqual(len(mv.window), 0)

    def test_initialization_with_invalid_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingVariance(window_size=-1)

    def test_learn_one_valid_input(self):
        mv = MovingVariance(window_size=3)
        mv.learn_one({"value": 10})
        mv.learn_one({"value": 20})
        self.assertEqual(list(mv.window), [10, 20])

    def test_learn_one_invalid_input_raises_assertion_error(self):
        mv = MovingVariance(window_size=3)
        with self.assertRaises(AssertionError):
            mv.learn_one({"a": 1, "b": 2})

    def test_score_one_empty_window_returns_zero(self):
        mv = MovingVariance(window_size=3)
        self.assertEqual(mv.score_one(), 0)

    def test_score_one_single_value_variance_is_zero(self):
        mv = MovingVariance(window_size=5)
        mv.learn_one({"value": 10})
        self.assertEqual(mv.score_one(), 0)

    def test_score_one_two_same_values_variance_is_zero(self):
        mv = MovingVariance(window_size=3)
        mv.learn_one({"value": 15})
        mv.learn_one({"value": 15})
        self.assertEqual(mv.score_one(), 0)

    def test_score_one_calculates_correct_variance(self):
        mv = MovingVariance(window_size=5)
        values = [10, 12, 23, 23, 16]
        for value in values:
            mv.learn_one({"value": value})

        expected_variance = (
            (10 - 16.8) ** 2
            + (12 - 16.8) ** 2
            + (23 - 16.8) ** 2
            + (23 - 16.8) ** 2
            + (16 - 16.8) ** 2
        ) / 5

        self.assertAlmostEqual(mv.score_one(), expected_variance)

    def test_score_one_with_window_rolling(self):
        mv = MovingVariance(window_size=3)
        values = [10, 12, 23]
        for value in values:
            mv.learn_one({"value": value})
        # Expect variance of window [10, 12, 23]
        expected_variance1 = np.var([10, 12, 23])
        self.assertAlmostEqual(mv.score_one(), expected_variance1)
        mv.learn_one({"value": 18})
        # Expect variance of window [12, 23, 18]
        expected_variance2 = np.var([12, 23, 18])
        self.assertAlmostEqual(mv.score_one(), expected_variance2)


class TestMovingInterquartileRange(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        ma = MovingInterquartileRange(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingInterquartileRange(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingInterquartileRange(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingInterquartileRange(3)
        with self.assertRaises(AssertionError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_mean_correctly(self):
        ma = MovingInterquartileRange(10)
        test_values = [1, 4, 2, 6, 7, 3, 9, 33, 34, -5]
        for x in test_values:
            ma.learn_one({"value": x})
        self.assertEqual(
            ma.score_one(),
            np.quantile(test_values, 0.75) - np.quantile(test_values, 0.25),
        )

    def test_score_one_with_empty_window(self):
        ma = MovingInterquartileRange(3)
        self.assertEqual(ma.score_one(), 0)

    def test_window_shifting(self):
        ma = MovingInterquartileRange(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingAverageAbsoluteDeviation(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        ma = MovingAverageAbsoluteDeviation(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingAverageAbsoluteDeviation(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingAverageAbsoluteDeviation(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingAverageAbsoluteDeviation(3)
        with self.assertRaises(AssertionError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_mean_correctly(self):
        ma = MovingAverageAbsoluteDeviation(10)
        test_values = [1, 4, 2, 6, 7, 3, 9, 33, 34, -5]
        for x in test_values:
            ma.learn_one({"value": x})
        self.assertEqual(
            ma.score_one(),
            sum(abs(x - np.mean(test_values)) for x in test_values) / len(test_values),
        )

    def test_score_one_with_empty_window(self):
        ma = MovingAverageAbsoluteDeviation(3)
        self.assertEqual(ma.score_one(), 0)

    def test_window_shifting(self):
        ma = MovingAverageAbsoluteDeviation(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingKurtosis(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        ma = MovingKurtosis(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingKurtosis(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingKurtosis(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingKurtosis(3)
        with self.assertRaises(AssertionError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_fisher_correctly(self):
        ma = MovingKurtosis(5)
        test_values = [1, 2, 3, 4]
        for x in test_values:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one(), kurtosis(test_values, fisher=True))

    def test_score_one_calculates_pearson_correctly(self):
        ma = MovingKurtosis(5, fisher=False)
        test_values = [1, 2, 3, 4]
        for x in test_values:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one(), kurtosis(test_values, fisher=False))

    def test_score_one_with_empty_window(self):
        ma = MovingKurtosis(3)
        self.assertEqual(ma.score_one(), 0)

    def test_window_shifting(self):
        ma = MovingKurtosis(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


class TestMovingSkewness(unittest.TestCase):

    def test_initialization_with_positive_window_size(self):
        ma = MovingSkewness(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingSkewness(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingSkewness(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingSkewness(3)
        with self.assertRaises(AssertionError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_skewness_correctly(self):
        ma = MovingSkewness(5)
        test_values = [1, 2, 3, 4, 43]
        for x in test_values:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one(), skew(test_values))

    def test_score_one_with_empty_window(self):
        ma = MovingSkewness(3)
        self.assertEqual(ma.score_one(), 0)

    def test_window_shifting(self):
        ma = MovingSkewness(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))


if __name__ == "__main__":
    unittest.main()
