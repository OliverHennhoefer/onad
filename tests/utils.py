"""Test utilities for ONAD test suite."""

import math
import time

import numpy as np


class DataGenerator:
    """Utility class for generating test data for anomaly detection models."""

    def __init__(self, seed: int = 42):
        """Initialize data generator with random seed."""
        self.rng = np.random.default_rng(seed)

    def generate_streaming_data(
        self, n: int = 1000, n_features: int = 3
    ) -> list[dict[str, float]]:
        """
        Generate synthetic streaming data for testing.

        Args:
            n: Number of data points to generate
            n_features: Number of features per data point

        Returns:
            List of feature dictionaries
        """
        data = []
        for i in range(n):
            point = {}
            for j in range(n_features):
                # Mix of different distributions
                if j == 0:
                    # Normal distribution
                    point[f"feature_{j}"] = float(self.rng.normal(0, 1))
                elif j == 1:
                    # Trending data
                    point[f"feature_{j}"] = float(i * 0.01 + self.rng.normal(0, 0.5))
                else:
                    # Uniform distribution
                    point[f"feature_{j}"] = float(self.rng.uniform(-2, 2))
            data.append(point)
        return data

    def create_anomaly_stream(
        self, n: int = 1000, anomaly_rate: float = 0.05, n_features: int = 2
    ) -> tuple[list[dict[str, float]], list[bool]]:
        """
        Create a data stream with known anomalies.

        Args:
            n: Total number of data points
            anomaly_rate: Fraction of points that are anomalies
            n_features: Number of features

        Returns:
            Tuple of (data_points, anomaly_labels)
        """
        data = []
        labels = []
        n_anomalies = int(n * anomaly_rate)

        # Generate normal points
        for _i in range(n - n_anomalies):
            point = {}
            for j in range(n_features):
                point[f"feature_{j}"] = float(self.rng.normal(0, 1))
            data.append(point)
            labels.append(False)

        # Generate anomalous points
        for _i in range(n_anomalies):
            point = {}
            for j in range(n_features):
                # Anomalies are further from center
                point[f"feature_{j}"] = float(self.rng.normal(0, 4))
            data.append(point)
            labels.append(True)

        # Shuffle the data
        indices = list(range(n))
        self.rng.shuffle(indices)

        shuffled_data = [data[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        return shuffled_data, shuffled_labels

    def generate_univariate_data(
        self, n: int = 1000, key: str = "value"
    ) -> list[dict[str, float]]:
        """
        Generate univariate streaming data.

        Args:
            n: Number of data points
            key: Feature key name

        Returns:
            List of single-feature dictionaries
        """
        data = []
        for i in range(n):
            # Mix sine wave with noise for interesting patterns
            value = math.sin(i * 0.1) + self.rng.normal(0, 0.3)
            data.append({key: float(value)})
        return data

    def generate_bivariate_data(self, n: int = 1000) -> list[dict[str, float]]:
        """
        Generate bivariate correlated data for covariance/correlation testing.

        Args:
            n: Number of data points

        Returns:
            List of two-feature dictionaries
        """
        data = []
        for _i in range(n):
            # Create correlated features
            x = self.rng.normal(0, 1)
            y = 0.7 * x + self.rng.normal(0, 0.5)  # Correlation ~0.7
            data.append({"x": float(x), "y": float(y)})
        return data

    def generate_concept_drift_data(self, n: int = 2000) -> list[dict[str, float]]:
        """
        Generate data with concept drift for testing adaptive models.

        Args:
            n: Number of data points

        Returns:
            List of feature dictionaries with shifting distribution
        """
        data = []
        for i in range(n):
            # Shift mean and variance over time
            shift = i / n  # 0 to 1 over the sequence
            mean = shift * 3  # Mean shifts from 0 to 3
            std = 1 + shift  # Std shifts from 1 to 2

            value = self.rng.normal(mean, std)
            data.append({"feature": float(value)})
        return data


class TestAssertions:
    """Utility class for common test assertions."""

    @staticmethod
    def assert_scores_valid(scores: list[float], min_score: float = 0.0):
        """
        Assert that anomaly scores are valid.

        Args:
            scores: List of anomaly scores
            min_score: Minimum valid score (default 0 for non-negative scores)
        """
        assert len(scores) > 0, "No scores provided"

        for i, score in enumerate(scores):
            assert isinstance(score, int | float), f"Score {i} is not numeric: {score}"
            assert not math.isnan(score), f"Score {i} is NaN"
            assert not math.isinf(score), f"Score {i} is infinite"
            assert score >= min_score, f"Score {i} below minimum: {score} < {min_score}"

    @staticmethod
    def assert_features_preserved(
        original: dict[str, float],
        transformed: dict[str, float],
        allow_new_features: bool = True,
    ):
        """
        Assert that feature transformation preserves expected properties.

        Args:
            original: Original feature dict
            transformed: Transformed feature dict
            allow_new_features: Whether new features can be added
        """
        assert isinstance(transformed, dict), "Transformed data must be dict"

        if not allow_new_features:
            assert len(transformed) == len(original), (
                f"Feature count changed: {len(original)} -> {len(transformed)}"
            )

        for key, value in transformed.items():
            assert isinstance(key, str), f"Feature key must be string: {key}"
            assert isinstance(value, int | float | np.number), (
                f"Feature value must be numeric: {key}={value}"
            )
            assert not math.isnan(float(value)), f"Feature {key} is NaN"
            assert not math.isinf(float(value)), f"Feature {key} is infinite"

    @staticmethod
    def assert_streaming_behavior(
        model, data: list[dict[str, float]], tolerance: float = 1e-10
    ):
        """
        Assert that model behaves deterministically with streaming data.

        Args:
            model: Model to test
            data: Streaming data
            tolerance: Tolerance for score differences
        """
        # Process data once
        scores1 = []
        for point in data:
            model.learn_one(point)
            score = model.score_one(point)
            scores1.append(score)

        # Reset and process again (if model supports reset)
        if hasattr(model, "reset"):
            model.reset()
            scores2 = []
            for point in data:
                model.learn_one(point)
                score = model.score_one(point)
                scores2.append(score)

            # Scores should be identical
            for i, (s1, s2) in enumerate(zip(scores1, scores2, strict=False)):
                assert abs(s1 - s2) <= tolerance, (
                    f"Score difference at point {i}: {s1} != {s2}"
                )


def benchmark_model_speed(
    model, data: list[dict[str, float]], warmup_points: int = 100
) -> dict[str, float]:
    """
    Simple speed benchmark for models (not for regular testing).

    Args:
        model: Model to benchmark
        data: Data to use for benchmarking
        warmup_points: Points to use for warmup

    Returns:
        Dictionary with timing information
    """
    # Warmup
    for point in data[:warmup_points]:
        model.learn_one(point)

    # Benchmark learning
    start_time = time.time()
    for point in data[warmup_points : warmup_points + 100]:
        model.learn_one(point)
    learn_time = time.time() - start_time

    # Benchmark scoring
    start_time = time.time()
    for point in data[warmup_points : warmup_points + 100]:
        model.score_one(point)
    score_time = time.time() - start_time

    return {
        "learn_time_per_point": learn_time / 100,
        "score_time_per_point": score_time / 100,
        "total_time": learn_time + score_time,
    }
