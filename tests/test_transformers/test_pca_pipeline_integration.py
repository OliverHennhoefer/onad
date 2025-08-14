import unittest
from sklearn.metrics import average_precision_score
import numpy as np

from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import StandardScaler
from onad.transform.pca import IncrementalPCA
from onad.model.unsupervised.distance.knn import KNN
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine


class TestPCAPipelineIntegration(unittest.TestCase):
    """
    Integration tests for IncrementalPCA working as a transformer in complete pipelines
    using real streaming data from the Shuttle dataset.
    """

    def setUp(self):
        """Set up pipeline components for testing."""
        # Pipeline components
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=5, n0=50)  # Reduce to 5D from higher dim
        self.engine = FaissSimilaritySearchEngine(window_size=200, warm_up=25)
        self.knn = KNN(k=10, similarity_engine=self.engine)

        # Test different pipeline configurations
        # Now Pipeline supports | operator for chaining
        self.pipeline_scale_pca = self.scaler | self.pca
        self.pipeline_full = self.scaler | self.pca | self.knn

    def test_pipeline_construction(self):
        """Test that pipeline components can be chained using | operator."""
        # Test basic chaining (transformer | transformer)
        scaler_pca = self.scaler | self.pca
        self.assertIsNotNone(scaler_pca)
        self.assertTrue(hasattr(scaler_pca, "learn_one"))
        self.assertTrue(hasattr(scaler_pca, "transform_one"))
        self.assertTrue(
            hasattr(scaler_pca, "__or__")
        )  # Should support further chaining

        # Test full pipeline chaining (transformer | transformer | model)
        pipeline_direct = self.scaler | self.pca | self.knn
        self.assertIsNotNone(pipeline_direct)
        self.assertTrue(hasattr(pipeline_direct, "learn_one"))
        self.assertTrue(hasattr(pipeline_direct, "score_one"))

        # Test that both approaches create equivalent pipelines
        self.assertIsNotNone(self.pipeline_full)
        self.assertTrue(hasattr(self.pipeline_full, "learn_one"))
        self.assertTrue(hasattr(self.pipeline_full, "score_one"))

    def test_dimension_reduction_flow(self):
        """Test that data flows correctly through scaler → PCA with dimension reduction."""
        # Sample data to test dimension flow
        sample_data = [
            {
                "feat_0": 1.5,
                "feat_1": 2.3,
                "feat_2": -0.5,
                "feat_3": 4.1,
                "feat_4": 0.8,
                "feat_5": -1.2,
                "feat_6": 3.7,
                "feat_7": 2.9,
                "feat_8": -0.3,
            },
            {
                "feat_0": 2.1,
                "feat_1": 1.8,
                "feat_2": 0.7,
                "feat_3": 3.5,
                "feat_4": -0.2,
                "feat_5": 2.4,
                "feat_6": 1.9,
                "feat_7": -1.1,
                "feat_8": 2.6,
            },
        ]

        # Learn on sample data (more than n0 to trigger PCA)
        for _ in range(60):  # More than n0=50
            for sample in sample_data:
                self.pipeline_scale_pca.learn_one(sample)

        # Test transformation
        transformed = self.pipeline_scale_pca.transform_one(sample_data[0])

        # Should reduce from original dimensions to n_components=5
        self.assertEqual(len(transformed), 5)
        self.assertTrue(all(f"component_{i}" in transformed for i in range(5)))
        self.assertTrue(all(isinstance(v, float) for v in transformed.values()))

    def test_streaming_shuttle_dataset(self):
        """Test complete pipeline with streaming Shuttle dataset."""
        labels, scores = [], []
        processed_samples = 0
        max_samples = 500  # Process subset for test efficiency

        # Track PCA state evolution
        pca_values_evolution = []

        with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
            for i, (x, y) in enumerate(streamer):
                if processed_samples >= max_samples:
                    break

                # Learn from normal samples initially (similar to examples pattern)
                if y == 0 and i < 100:
                    self.pipeline_full.learn_one(x)
                    processed_samples += 1

                    # Track PCA evolution during learning
                    if self.pca.n0_reached and len(self.pca.values) > 0:
                        pca_values_evolution.append(self.pca.values.copy())
                    continue

                # Learn and score
                self.pipeline_full.learn_one(x)
                score = self.pipeline_full.score_one(x)

                # Skip if score is None (happens during warm-up phase)
                if score is not None:
                    labels.append(y)
                    scores.append(score)

                    # Validate score is reasonable
                    self.assertIsInstance(score, (int, float, np.floating))
                    self.assertFalse(np.isnan(score))
                    self.assertFalse(np.isinf(score))

                processed_samples += 1

        # Validate we processed data
        self.assertGreater(processed_samples, 100)
        self.assertGreater(len(labels), 50)
        self.assertGreater(len(scores), 50)

        # Validate PCA learned and evolved
        if len(pca_values_evolution) > 1:
            # Eigenvalues should change as PCA learns
            first_values = pca_values_evolution[0]
            last_values = pca_values_evolution[-1]
            self.assertFalse(np.allclose(first_values, last_values, rtol=1e-6))

        # Validate anomaly detection performance (basic sanity check)
        if len(set(labels)) > 1:  # If we have both normal and anomalous samples
            # Calculate PR-AUC (should be > random baseline of minority class %)
            pr_auc = average_precision_score(labels, scores)
            minority_class_ratio = sum(labels) / len(labels)
            self.assertGreater(
                pr_auc, minority_class_ratio * 0.5
            )  # At least half the random baseline

    def test_pca_state_consistency(self):
        """Test that PCA maintains consistent state during streaming."""
        sample_count = 0
        eigenvalue_snapshots = []

        with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
            for i, (x, y) in enumerate(streamer):
                if sample_count >= 150:
                    break

                self.pipeline_scale_pca.learn_one(x)
                sample_count += 1

                # Take snapshots after PCA initialization
                if self.pca.n0_reached and sample_count % 20 == 0:
                    eigenvalue_snapshots.append(
                        {
                            "sample_count": sample_count,
                            "values": self.pca.values.copy(),
                            "n_components": len(self.pca.values),
                        }
                    )

        # Validate PCA state evolution
        self.assertGreater(len(eigenvalue_snapshots), 3)

        # Check that number of components stays consistent
        n_components = eigenvalue_snapshots[0]["n_components"]
        for snapshot in eigenvalue_snapshots:
            self.assertEqual(snapshot["n_components"], n_components)
            self.assertEqual(len(snapshot["values"]), self.pca.n_components)

        # Check that eigenvalues are positive and ordered
        for snapshot in eigenvalue_snapshots:
            values = snapshot["values"]
            self.assertTrue(np.all(values > 0), "All eigenvalues should be positive")
            # Should be in descending order (approximately, allowing for small numerical errors)
            self.assertTrue(
                np.all(values[:-1] >= values[1:] - 1e-10),
                "Eigenvalues should be in descending order",
            )

    def test_error_handling_and_robustness(self):
        """Test pipeline robustness with edge cases."""
        # Test with minimal data
        minimal_pipeline = StandardScaler() | IncrementalPCA(n_components=2, n0=5)

        sample_data = {"feat_0": 1.0, "feat_1": 2.0, "feat_2": 3.0}

        # Should handle learning without errors
        for _ in range(10):
            minimal_pipeline.learn_one(sample_data)

        # Should handle transform_one
        if hasattr(minimal_pipeline, "transform_one"):
            result = minimal_pipeline.transform_one(sample_data)
            self.assertIsInstance(result, dict)

    def test_memory_efficiency(self):
        """Test that pipeline components don't grow memory unboundedly."""
        initial_pca_samples = self.pca.n_samples_seen

        # Process some samples
        with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
            for i, (x, y) in enumerate(streamer):
                if i >= 200:
                    break
                self.pipeline_full.learn_one(x)

        # PCA should have learned incrementally (samples seen should increase)
        self.assertGreater(self.pca.n_samples_seen, initial_pca_samples)

        # Engine should have bounded size due to window_size parameter
        if hasattr(self.engine, "vectors"):
            current_engine_size = len(self.engine.vectors)
            self.assertLessEqual(
                current_engine_size, self.engine.window_size + 50
            )  # Allow some tolerance


if __name__ == "__main__":
    unittest.main()
