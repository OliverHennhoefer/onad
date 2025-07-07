import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from onad.model.unsupervised.forest.online_iforest import BoundedRandomProjectionOnlineIForest
from onad.stream.streamer import ParquetStreamer, Dataset


class TestCaseBoundedOnlineIForest(unittest.TestCase):
    def test_shuttle(self):
        model = BoundedRandomProjectionOnlineIForest(
            num_trees=5,
            window_size=64,
            branching_factor=2,
            max_leaf_samples=2,
            type='fixed',
            subsample=1.0,
            n_jobs=-1
        )

        labels, scores = [], []

        with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
            for i, (x, y) in enumerate(streamer):
                # Train only on normal Data (Label 0) at the beginning
                if y == 0 and i < 500:
                    model.learn_one(x)
                    continue

                model.learn_one(x)
                score = model.score_one(x)

                labels.append(y)
                scores.append(score)

        # Calculate and check Metrics
        roc_auc = round(roc_auc_score(labels, scores), 3)
        avg_pre = round(average_precision_score(labels, scores), 3)

        print(f"ROC-AUC: {roc_auc}")
        print(f"Average Precision: {avg_pre}")

        self.assertGreater(roc_auc, 0.50)
        self.assertGreater(avg_pre, 0.05)

        # Optional: more checks on FPR/TPR Shape
        fpr, tpr, thresholds = roc_curve(labels, scores)
        self.assertEqual(len(fpr), len(tpr))
        self.assertEqual(len(thresholds), len(fpr))


if __name__ == "__main__":
    unittest.main()
