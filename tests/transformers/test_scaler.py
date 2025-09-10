import unittest

from onad.stream.dataset import Dataset, load
from onad.transform.preprocessing.scaler import MinMaxScaler


class MyTestCase(unittest.TestCase):
    def test_min_max_scaler(self):
        scaler = MinMaxScaler()

        normalized_vals = []
        # Load dataset using new API
        dataset = load(Dataset.FRAUD)

        for x, _ in dataset.stream():
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


if __name__ == "__main__":
    unittest.main()
