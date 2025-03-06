import unittest
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_min_max_scaler(self):
        from onad.transformer.scaler.normalize import MinMaxScaler
        from onad.utils.streamer.streamer import NPZStreamer

        scaler = MinMaxScaler()

        normalized_vals = []
        with NPZStreamer("./data/shuttle.npz") as streamer:
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

    def test_standard_scaler(self):
        from onad.transformer.scaler.standardize import StandardScaler
        from onad.utils.streamer.streamer import NPZStreamer

        scaler = StandardScaler()

        standardized_vals = []
        with NPZStreamer("./data/shuttle.npz") as streamer:
            for i, (x, y) in enumerate(streamer):
                scaler.learn_one(x)
                scaled_x = scaler.transform_one(x)
                standardized_vals.append(scaled_x)

        # Flatten the standardized values for analysis
        all_values = []
        for scaled_x in standardized_vals:
            all_values.extend(scaled_x.values())
        all_values = np.array(all_values)

        # Check that the mean is approximately 0 and standard deviation is approximately 1
        self.assertAlmostEqual(np.mean(all_values), 0, delta=0.1)
        self.assertAlmostEqual(np.std(all_values), 1, delta=0.1)


if __name__ == "__main__":
    unittest.main()
