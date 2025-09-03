import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from onad.dataset import Dataset, load
from onad.model.unsupervised.forest.online_iforest import OnlineIsolationForest


class TestTrueOnlineIForest(unittest.TestCase):
    def test_shuttle(self):
        """Test TrueOnlineIForest on SHUTTLE dataset."""
        model = OnlineIsolationForest(
            num_trees=10,
            max_leaf_samples=32,
            type="adaptive",
            subsample=1.0,
            window_size=512,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        labels, scores = [], []

        # Load dataset using new API
        dataset = load(Dataset.SHUTTLE)

        for i, (x, y) in enumerate(dataset.stream()):
            # Train only on normal data (Label 0) at the beginning
            if y == 0 and i < 1000:
                model.learn_one(x)
                continue

            model.learn_one(x)
            score = model.score_one(x)

            labels.append(y)
            scores.append(score)

        # Calculate and check metrics
        roc_auc = round(roc_auc_score(labels, scores), 3)
        avg_pre = round(average_precision_score(labels, scores), 3)

        print(f"True Online IForest - ROC-AUC: {roc_auc}")
        print(f"True Online IForest - Average Precision: {avg_pre}")

        self.assertGreater(roc_auc, 0.50)
        self.assertGreater(avg_pre, 0.05)

        # Optional: more checks on FPR/TPR shape
        fpr, tpr, thresholds = roc_curve(labels, scores)
        self.assertEqual(len(fpr), len(tpr))
        self.assertEqual(len(thresholds), len(fpr))

    def test_batch_learning(self):
        """Test batch learning functionality."""
        model = OnlineIsolationForest(
            num_trees=5,
            max_leaf_samples=16,
            type="fixed",
            subsample=1.0,
            window_size=256,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        # Generate synthetic data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (200, 5))
        anomaly_data = np.random.normal(3, 1, (20, 5))

        # Learn on normal data
        model.learn_batch(normal_data)

        # Score both normal and anomaly data
        normal_scores = model.score_batch(normal_data[:50])
        anomaly_scores = model.score_batch(anomaly_data)

        # Anomalies should have higher scores on average
        avg_normal_score = np.mean(normal_scores)
        avg_anomaly_score = np.mean(anomaly_scores)

        print(f"Average normal score: {avg_normal_score:.3f}")
        print(f"Average anomaly score: {avg_anomaly_score:.3f}")

        self.assertGreater(avg_anomaly_score, avg_normal_score)

    def test_incremental_learning(self):
        """Test incremental learning vs batch learning."""
        # Create two identical models
        model1 = OnlineIsolationForest(
            num_trees=3,
            max_leaf_samples=16,
            type="fixed",
            subsample=1.0,
            window_size=None,  # No sliding window
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        model2 = OnlineIsolationForest(
            num_trees=3,
            max_leaf_samples=16,
            type="fixed",
            subsample=1.0,
            window_size=None,  # No sliding window
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        # Set same random seed for reproducibility
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))

        # Learn incrementally
        for row in data:
            model1.learn_batch(row.reshape(1, -1))

        # Learn in batch
        model2.learn_batch(data)

        # Test data
        test_data = np.random.normal(0, 1, (20, 3))

        # Both should give similar results (not identical due to incremental updates)
        scores1 = model1.score_batch(test_data)
        scores2 = model2.score_batch(test_data)

        # Check that both models produce valid scores
        self.assertTrue(np.all(scores1 >= 0))
        self.assertTrue(np.all(scores1 <= 1))
        self.assertTrue(np.all(scores2 >= 0))
        self.assertTrue(np.all(scores2 <= 1))

    def test_sliding_window(self):
        """Test sliding window functionality."""
        model = OnlineIsolationForest(
            num_trees=3,
            max_leaf_samples=8,
            type="fixed",
            subsample=1.0,
            window_size=50,  # Small window
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        # Generate data in chunks
        np.random.seed(42)

        # Learn from first chunk
        chunk1 = np.random.normal(0, 1, (30, 3))
        model.learn_batch(chunk1)

        # Learn from second chunk (should fit in window)
        chunk2 = np.random.normal(0, 1, (20, 3))
        model.learn_batch(chunk2)

        # Check data size is within window
        self.assertEqual(model.data_size, 50)

        # Learn from third chunk (should trigger unlearning)
        chunk3 = np.random.normal(0, 1, (30, 3))
        model.learn_batch(chunk3)

        # Check data size is still within window
        self.assertEqual(model.data_size, 50)

        # Test scoring still works
        test_data = np.random.normal(0, 1, (10, 3))
        scores = model.score_batch(test_data)
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_different_branching_factors(self):
        """Test different branching factors."""
        for branching_factor in [2, 3, 4]:
            model = OnlineIsolationForest(
                num_trees=2,
                max_leaf_samples=8,
                type="fixed",
                subsample=1.0,
                window_size=100,
                branching_factor=branching_factor,
                metric="axisparallel",
                n_jobs=1,
            )

            # Generate test data
            np.random.seed(42)
            data = np.random.normal(0, 1, (50, 4))

            # Learn from data
            model.learn_batch(data)

            # Test scoring
            test_data = np.random.normal(0, 1, (10, 4))
            scores = model.score_batch(test_data)

            self.assertEqual(len(scores), 10)
            self.assertTrue(np.all(scores >= 0))
            self.assertTrue(np.all(scores <= 1))

    def test_adaptive_vs_fixed(self):
        """Test adaptive vs fixed multiplier types."""
        for type_name in ["fixed", "adaptive"]:
            model = OnlineIsolationForest(
                num_trees=3,
                max_leaf_samples=16,
                type=type_name,
                subsample=1.0,
                window_size=200,
                branching_factor=2,
                metric="axisparallel",
                n_jobs=1,
            )

            # Generate test data
            np.random.seed(42)
            data = np.random.normal(0, 1, (100, 3))

            # Learn from data
            model.learn_batch(data)

            # Test scoring
            test_data = np.random.normal(0, 1, (20, 3))
            scores = model.score_batch(test_data)

            self.assertEqual(len(scores), 20)
            self.assertTrue(np.all(scores >= 0))
            self.assertTrue(np.all(scores <= 1))

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        model = OnlineIsolationForest(
            num_trees=2,
            max_leaf_samples=8,
            type="fixed",
            subsample=1.0,
            window_size=100,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        # Test with empty dict
        model.learn_one({})
        score = model.score_one({})
        self.assertEqual(score, 0.0)

        # Test with empty array
        empty_array = np.array([]).reshape(0, 3)
        model.learn_batch(empty_array)
        scores = model.score_batch(empty_array)
        self.assertEqual(len(scores), 0)

    def test_single_point_learning(self):
        """Test learning from single data points."""
        model = OnlineIsolationForest(
            num_trees=2,
            max_leaf_samples=4,
            type="fixed",
            subsample=1.0,
            window_size=20,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
        )

        # Learn from individual points
        for i in range(10):
            point = {
                "feature1": float(i),
                "feature2": float(i * 2),
                "feature3": float(i * 3),
            }
            model.learn_one(point)

        # Test scoring
        test_point = {"feature1": 5.0, "feature2": 10.0, "feature3": 15.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
