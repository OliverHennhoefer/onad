import unittest

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from onad.model.unsupervised.autoencoder import Autoencoder
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler


class TestCaseAutoencoder(unittest.TestCase):
    def test_shuttle(self):

        scaler = MinMaxScaler()

        model = Autoencoder(hidden_sizes=[140, 110], latent_size=3, dropout=0.15, learning_rate=0.005, seed=1)

        model = scaler | model

        labels, scores = [], []
        with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
            for i, (x, y) in enumerate(streamer):
                if i < 10_000:
                    model.learn_one(x)
                    continue
                model.learn_one(x)
                score = model.score_one(x)

                labels.append(y)
                scores.append(score)

        roc_auc = round(roc_auc_score(labels, scores), 3)
        self.assertEqual(roc_auc, 0.985)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        self.assertAlmostEqual(sum(fpr), 100.999, places=1)
        self.assertAlmostEqual(sum(tpr), 351.246, places=1)
        self.assertAlmostEqual(sum(thresholds[1:]), 2.652, places=1)

        avg_pre = round(average_precision_score(labels, scores), 3)
        print(avg_pre)
        self.assertEqual(avg_pre, 0.957)


if __name__ == '__main__':
    unittest.main()
