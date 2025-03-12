import unittest

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from onad.model.unsupervised.asd_iforest import ASDIsolationForest
from onad.stream.streamer import ParquetStreamer, Dataset


class TestCaseASDIsolationForest(unittest.TestCase):
    def test_shuttle(self):

        model = ASDIsolationForest(n_estimators=750, max_samples=2750, seed=1)

        labels, scores = [], []
        with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
            for i, (x, y) in enumerate(streamer):
                if y == 0 and i < 10_000:
                    model.learn_one(x)
                    continue
                model.learn_one(x)
                score = model.score_one(x)

                labels.append(y)
                scores.append(score)

        roc_auc = round(roc_auc_score(labels, scores), 3)
        self.assertEqual(roc_auc, 0.908)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        self.assertAlmostEqual(sum(fpr), 4983.480, places=1)
        self.assertAlmostEqual(sum(tpr), 9747.686, places=1)
        self.assertAlmostEqual(sum(thresholds[1:]), 4703.747, places=1)

        avg_pre = round(average_precision_score(labels, scores), 3)
        self.assertEqual(avg_pre, 0.749)


if __name__ == "__main__":
    unittest.main()
