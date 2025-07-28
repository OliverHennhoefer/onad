import unittest
import numpy as np

from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler, StandardScaler


class MyTestCase(unittest.TestCase):
    def test_min_max_scaler(self):

        scaler = MinMaxScaler()

        normalized_vals = []
        with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
            for x, y in streamer:
                scaler.learn_one(x)
                scaled_x = scaler.transform_one(x)
                normalized_vals.append(scaled_x)

        self.assertIsNotNone(normalized_vals)
        for i, (normalized_dict) in enumerate(normalized_vals):
            for key, value in normalized_dict.items():
                self.assertTrue(
                    0 <= value <= 1,
                    f"Instance {i} with value {value} for key {key} is not correctly normalized.",
                )

class TestStandardScaler(unittest.TestCase):
    def test_standard_scaler(self):
        base_values = [2.0, 3.0, 6.0, 3.0, 4.0]
        scaler = StandardScaler()
        X = [{'x': i, 'y': i*2} for i in base_values]
        for x in X:
            scaler.learn_one(x)
        print(scaler.means, scaler.vars)
        print(sum(base_values)/5, np.var(base_values))
        self.assertAlmostEqual(scaler.means['x'], sum(base_values)/5)  # mean: 3.6
        self.assertAlmostEqual(scaler.vars['x'], np.var(base_values))  #var: 1.84
        self.assertAlmostEqual(scaler.transform_one({'x': 2})['x'], (2-3.6)/1.84**0.5)

if __name__ == "__main__":
    unittest.main()
