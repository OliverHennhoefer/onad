import unittest
import numpy as np

from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler, StandardScaler


class TestMinMaxScaler(unittest.TestCase):
    """Comprehensive tests for MinMaxScaler with tight validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scaler = MinMaxScaler()
        self.test_data = [
            {'feature1': 10.0, 'feature2': 5.0},
            {'feature1': 20.0, 'feature2': 15.0},
            {'feature1': 30.0, 'feature2': 25.0},
        ]
    
    def test_basic_scaling_default_range(self):
        """Test basic scaling with default (0, 1) range."""
        # Learn from data
        for data in self.test_data:
            self.scaler.learn_one(data)
        
        # Test first point: min values should scale to 0
        result = self.scaler.transform_one({'feature1': 10.0, 'feature2': 5.0})
        self.assertAlmostEqual(result['feature1'], 0.0, places=10)
        self.assertAlmostEqual(result['feature2'], 0.0, places=10)
        
        # Test last point: max values should scale to 1
        result = self.scaler.transform_one({'feature1': 30.0, 'feature2': 25.0})
        self.assertAlmostEqual(result['feature1'], 1.0, places=10)
        self.assertAlmostEqual(result['feature2'], 1.0, places=10)
        
        # Test middle point: should scale to 0.5
        result = self.scaler.transform_one({'feature1': 20.0, 'feature2': 15.0})
        self.assertAlmostEqual(result['feature1'], 0.5, places=10)
        self.assertAlmostEqual(result['feature2'], 0.5, places=10)
    
    def test_custom_feature_range(self):
        """Test scaling with custom feature range."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        for data in self.test_data:
            scaler.learn_one(data)
        
        # Min values should scale to -1
        result = scaler.transform_one({'feature1': 10.0, 'feature2': 5.0})
        self.assertAlmostEqual(result['feature1'], -1.0, places=10)
        self.assertAlmostEqual(result['feature2'], -1.0, places=10)
        
        # Max values should scale to 1
        result = scaler.transform_one({'feature1': 30.0, 'feature2': 25.0})
        self.assertAlmostEqual(result['feature1'], 1.0, places=10)
        self.assertAlmostEqual(result['feature2'], 1.0, places=10)
        
        # Middle values should scale to 0
        result = scaler.transform_one({'feature1': 20.0, 'feature2': 15.0})
        self.assertAlmostEqual(result['feature1'], 0.0, places=10)
        self.assertAlmostEqual(result['feature2'], 0.0, places=10)
    
    def test_single_data_point(self):
        """Test behavior with single data point."""
        single_data = {'feature1': 42.0}
        self.scaler.learn_one(single_data)
        
        # With single point, min == max, should return feature_range[0]
        result = self.scaler.transform_one(single_data)
        self.assertAlmostEqual(result['feature1'], 0.0, places=10)
    
    def test_constant_feature_values(self):
        """Test scaling when all feature values are identical."""
        constant_data = [
            {'feature1': 5.0},
            {'feature1': 5.0},
            {'feature1': 5.0},
        ]
        
        for data in constant_data:
            self.scaler.learn_one(data)
        
        # All constant values should scale to feature_range[0]
        result = self.scaler.transform_one({'feature1': 5.0})
        self.assertAlmostEqual(result['feature1'], 0.0, places=10)
    
    def test_incremental_learning_correctness(self):
        """Test that incremental learning produces correct min/max values."""
        data_points = [
            {'x': 1.0}, {'x': 5.0}, {'x': 3.0}, {'x': 9.0}, {'x': 2.0}
        ]
        
        for data in data_points:
            self.scaler.learn_one(data)
        
        # Verify internal state
        self.assertEqual(self.scaler.min['x'], 1.0)
        self.assertEqual(self.scaler.max['x'], 9.0)
        
        # Test scaling correctness
        result = self.scaler.transform_one({'x': 1.0})  # min
        self.assertAlmostEqual(result['x'], 0.0, places=10)
        
        result = self.scaler.transform_one({'x': 9.0})  # max
        self.assertAlmostEqual(result['x'], 1.0, places=10)
        
        result = self.scaler.transform_one({'x': 5.0})  # (5-1)/(9-1) = 0.5
        self.assertAlmostEqual(result['x'], 0.5, places=10)
    
    def test_numpy_float64_handling(self):
        """Test proper handling of numpy float64 values."""
        data = {'feature1': np.float64(15.5)}
        self.scaler.learn_one(data)
        self.scaler.learn_one({'feature1': np.float64(25.5)})
        
        result = self.scaler.transform_one({'feature1': np.float64(20.5)})
        expected = (20.5 - 15.5) / (25.5 - 15.5)  # Should be 0.5
        self.assertAlmostEqual(result['feature1'], expected, places=10)
        self.assertIsInstance(result['feature1'], float)
    
    def test_unseen_feature_error(self):
        """Test error handling for unseen features during transform."""
        self.scaler.learn_one({'known_feature': 10.0})
        
        with self.assertRaises(ValueError) as context:
            self.scaler.transform_one({'unknown_feature': 5.0})
        
        self.assertIn("Feature 'unknown_feature' has not been seen during learning", str(context.exception))
    
    def test_mixed_features_error(self):
        """Test error when transform includes both known and unknown features."""
        self.scaler.learn_one({'feature1': 10.0})
        
        with self.assertRaises(ValueError):
            self.scaler.transform_one({'feature1': 15.0, 'unknown': 20.0})
    
    def test_empty_feature_dict(self):
        """Test behavior with empty feature dictionary."""
        result = self.scaler.transform_one({})
        self.assertEqual(result, {})


class TestStandardScaler(unittest.TestCase):
    """Comprehensive tests for StandardScaler with tight validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scaler = StandardScaler()
        self.test_values = [2.0, 4.0, 6.0, 8.0, 10.0]  # mean=6, var=8
        self.test_data = [{'x': val} for val in self.test_values]
    
    def test_basic_standardization_with_std(self):
        """Test basic standardization with standard deviation (default behavior)."""
        for data in self.test_data:
            self.scaler.learn_one(data)
        
        # Verify learned statistics
        expected_mean = 6.0
        expected_var = 8.0  # population variance
        
        self.assertAlmostEqual(self.scaler.means['x'], expected_mean, places=10)
        
        computed_var = self.scaler.sum_sq_diffs['x'] / self.scaler.counts['x']
        self.assertAlmostEqual(computed_var, expected_var, places=10)
        
        # Test transformations
        # For x=2: z = (2-6)/sqrt(8) = -4/2.828... ≈ -1.414
        result = self.scaler.transform_one({'x': 2.0})
        expected = (2.0 - 6.0) / (8.0 ** 0.5)
        self.assertAlmostEqual(result['x'], expected, places=10)
        
        # For x=10: z = (10-6)/sqrt(8) = 4/2.828... ≈ 1.414
        result = self.scaler.transform_one({'x': 10.0})
        expected = (10.0 - 6.0) / (8.0 ** 0.5)
        self.assertAlmostEqual(result['x'], expected, places=10)
        
        # Mean should transform to 0
        result = self.scaler.transform_one({'x': 6.0})
        self.assertAlmostEqual(result['x'], 0.0, places=10)
    
    def test_standardization_without_std(self):
        """Test standardization without standard deviation (mean centering only)."""
        scaler = StandardScaler(with_std=False)
        
        for data in self.test_data:
            scaler.learn_one(data)
        
        # Should only subtract mean, not divide by std
        result = scaler.transform_one({'x': 8.0})
        expected = 8.0 - 6.0  # No division by std
        self.assertAlmostEqual(result['x'], expected, places=10)
        
        result = scaler.transform_one({'x': 4.0})
        expected = 4.0 - 6.0
        self.assertAlmostEqual(result['x'], expected, places=10)
    
    def test_single_data_point(self):
        """Test behavior with single data point."""
        single_data = {'feature1': 42.0}
        self.scaler.learn_one(single_data)
        
        # With single point, variance is 0, should use safe division
        result = self.scaler.transform_one(single_data)
        self.assertAlmostEqual(result['feature1'], 0.0, places=10)  # Safe div returns 0
    
    def test_zero_variance_feature(self):
        """Test handling of features with zero variance."""
        constant_data = [{'x': 5.0}, {'x': 5.0}, {'x': 5.0}]
        
        for data in constant_data:
            self.scaler.learn_one(data)
        
        # Zero variance should use safe division (return 0)
        result = self.scaler.transform_one({'x': 5.0})
        self.assertAlmostEqual(result['x'], 0.0, places=10)
    
    def test_incremental_mean_calculation(self):
        """Test correctness of incremental mean calculation."""
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        scaler = StandardScaler()
        
        for val in values:
            scaler.learn_one({'x': val})
        
        expected_mean = sum(values) / len(values)
        self.assertAlmostEqual(scaler.means['x'], expected_mean, places=10)
    
    def test_incremental_variance_calculation(self):
        """Test correctness of incremental variance calculation using Welford's algorithm."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        scaler = StandardScaler()
        
        for val in values:
            scaler.learn_one({'x': val})
        
        # Verify against numpy variance (population variance)
        expected_var = np.var(values)
        computed_var = scaler.sum_sq_diffs['x'] / scaler.counts['x']
        self.assertAlmostEqual(computed_var, expected_var, places=10)
    
    def test_multiple_features(self):
        """Test scaling with multiple features simultaneously."""
        multi_data = [
            {'x': 1.0, 'y': 10.0},
            {'x': 2.0, 'y': 20.0},
            {'x': 3.0, 'y': 30.0},
        ]
        
        for data in multi_data:
            self.scaler.learn_one(data)
        
        # Test that each feature is scaled independently
        result = self.scaler.transform_one({'x': 2.0, 'y': 20.0})
        
        # x: mean=2, var=2/3, std=sqrt(2/3), z=(2-2)/sqrt(2/3)=0
        self.assertAlmostEqual(result['x'], 0.0, places=10)
        
        # y: mean=20, var=200/3, std=sqrt(200/3), z=(20-20)/sqrt(200/3)=0
        self.assertAlmostEqual(result['y'], 0.0, places=10)
    
    def test_numpy_float64_handling(self):
        """Test proper handling of numpy float64 values."""
        data = [{'x': np.float64(val)} for val in [1.0, 2.0, 3.0]]
        
        for d in data:
            self.scaler.learn_one(d)
        
        result = self.scaler.transform_one({'x': np.float64(2.0)})
        self.assertAlmostEqual(result['x'], 0.0, places=10)  # Mean value
        self.assertIsInstance(result['x'], float)
    
    def test_unseen_feature_error(self):
        """Test error handling for unseen features during transform."""
        self.scaler.learn_one({'known_feature': 10.0})
        
        with self.assertRaises(ValueError) as context:
            self.scaler.transform_one({'unknown_feature': 5.0})
        
        self.assertIn("Feature 'unknown_feature' has not been seen during learning", str(context.exception))
    
    def test_mixed_features_error(self):
        """Test error when transform includes both known and unknown features."""
        self.scaler.learn_one({'feature1': 10.0})
        
        with self.assertRaises(ValueError):
            self.scaler.transform_one({'feature1': 15.0, 'unknown': 20.0})
    
    def test_empty_feature_dict(self):
        """Test behavior with empty feature dictionary."""
        result = self.scaler.transform_one({})
        self.assertEqual(result, {})
    
    def test_safe_div_functionality(self):
        """Test the _safe_div helper method directly."""
        # Normal division
        self.assertAlmostEqual(self.scaler._safe_div(10.0, 2.0), 5.0, places=10)
        
        # Division by zero
        self.assertEqual(self.scaler._safe_div(10.0, 0.0), 0.0)
        
        # Division by False (falsy value)
        self.assertEqual(self.scaler._safe_div(10.0, False), 0.0)


class TestScalersIntegration(unittest.TestCase):
    """Integration tests using real data from the streaming framework."""
    
    def test_min_max_scaler_integration(self):
        """Integration test for MinMaxScaler with streaming data."""
        scaler = MinMaxScaler()
        
        normalized_vals = []
        sample_count = 0
        
        with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
            for x, y in streamer:
                scaler.learn_one(x)
                scaled_x = scaler.transform_one(x)
                normalized_vals.append(scaled_x)
                
                sample_count += 1
                if sample_count >= 100:  # Limit for faster testing
                    break
        
        # Verify all values are in [0, 1] range with tight validation
        for i, scaled_dict in enumerate(normalized_vals):
            for feature, value in scaled_dict.items():
                self.assertGreaterEqual(value, 0.0, 
                    f"Sample {i}, feature {feature}: {value} < 0.0")
                self.assertLessEqual(value, 1.0, 
                    f"Sample {i}, feature {feature}: {value} > 1.0")
                self.assertIsInstance(value, float,
                    f"Sample {i}, feature {feature}: {value} is not float")
    
    def test_standard_scaler_integration(self):
        """Integration test for StandardScaler with streaming data."""
        scaler = StandardScaler()
        
        scaled_vals = []
        sample_count = 0
        
        with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
            for x, y in streamer:
                scaler.learn_one(x)
                scaled_x = scaler.transform_one(x)
                scaled_vals.append(scaled_x)
                
                sample_count += 1
                if sample_count >= 100:  # Limit for faster testing
                    break
        
        # Verify all values are finite and properly typed
        for i, scaled_dict in enumerate(scaled_vals):
            for feature, value in scaled_dict.items():
                self.assertTrue(np.isfinite(value),
                    f"Sample {i}, feature {feature}: {value} is not finite")
                self.assertIsInstance(value, float,
                    f"Sample {i}, feature {feature}: {value} is not float")


if __name__ == "__main__":
    unittest.main()