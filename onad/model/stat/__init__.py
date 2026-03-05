"""Statistical models for streaming anomaly detection."""

from onad.model.stat.multi import (
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)
from onad.model.stat.uni import (
    MovingAverage,
    MovingAverageAbsoluteDeviation,
    MovingGeometricAverage,
    MovingHarmonicAverage,
    MovingInterquartileRange,
    MovingKurtosis,
    MovingMedian,
    MovingQuantile,
    MovingSkewness,
    MovingVariance,
)

__all__ = [
    "MovingAverage",
    "MovingAverageAbsoluteDeviation",
    "MovingCorrelationCoefficient",
    "MovingCovariance",
    "MovingGeometricAverage",
    "MovingHarmonicAverage",
    "MovingInterquartileRange",
    "MovingKurtosis",
    "MovingMahalanobisDistance",
    "MovingMedian",
    "MovingQuantile",
    "MovingSkewness",
    "MovingVariance",
]
