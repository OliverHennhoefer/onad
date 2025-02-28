import unittest


class MyTestCase(unittest.TestCase):
    def test_min_max_scaler(self):
        from onad.transformer.scaler.min_max import MinMaxScaler
        from onad.utils.streamer.datasets import Dataset
        from onad.utils.streamer.streamer import NPZStreamer

        scaler = MinMaxScaler()

        normalized_vals = []
        with NPZStreamer(Dataset.SHUTTLE) as streamer:
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


if __name__ == "__main__":
    unittest.main()
