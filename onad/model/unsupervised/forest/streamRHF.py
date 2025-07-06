import random
from collections import deque
from typing import Dict, Optional, List

import numpy as np

from onad.base.model import BaseModel


class StreamRandomHistogramForest(BaseModel):
    """
    Online Stream Random Histogram Forest for anomaly detection.

    Args:
        n_estimators (int): Number of histogram trees in the forest.
        max_bins (int): Number of bins per dimension.
        window_size (int): Max number of instances stored per histogram.
        seed (Optional[int]): Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_bins: int = 10,
        window_size: int = 256,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_bins = max_bins
        self.window_size = window_size
        self.seed = seed
        self.feature_names: Optional[List[str]] = None
        self.histograms: deque = deque()
        if self.seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def learn_one(self, x: Dict[str, float]) -> None:
        if self.feature_names is None:
            self.feature_names = list(x.keys())
            for _ in range(self.n_estimators):
                self.histograms.append(self._init_histogram())

        for histogram in self.histograms:
            self._update_histogram(histogram, x)

    def _init_histogram(self) -> Dict:
        """Initialize a random histogram tree."""
        bins = {}
        for feature in self.feature_names:
            min_val = random.uniform(0, 0.5)
            max_val = random.uniform(0.5, 1.0)
            bin_edges = np.linspace(min_val, max_val, self.max_bins + 1)
            bins[feature] = {
                "edges": bin_edges,
                "counts": np.zeros(self.max_bins),
                "min": min_val,
                "max": max_val,
            }
        return {"bins": bins, "count": 0}

    def _update_histogram(self, histogram: Dict, x: Dict[str, float]) -> None:
        for feature in self.feature_names:
            value = x.get(feature, 0.0)
            bin_data = histogram["bins"][feature]
            bin_idx = np.digitize(value, bin_data["edges"]) - 1
            bin_idx = np.clip(bin_idx, 0, self.max_bins - 1)
            bin_data["counts"][bin_idx] += 1

        histogram["count"] += 1
        if histogram["count"] > self.window_size:
            for feature in self.feature_names:
                histogram["bins"][feature]["counts"] *= 0.9  # exponential decay
            histogram["count"] = int(histogram["count"] * 0.9)

    def score_one(self, x: Dict[str, float]) -> float:
        if not self.histograms:
            return 0.0

        scores = []
        for histogram in self.histograms:
            density = self._estimate_density(histogram, x)
            scores.append(1.0 - density)  # lower density = higher anomaly

        return float(np.mean(scores))

    def _estimate_density(self, histogram: Dict, x: Dict[str, float]) -> float:
        prob = 1.0
        for feature in self.feature_names:
            value = x.get(feature, 0.0)
            bin_data = histogram["bins"][feature]
            bin_idx = np.digitize(value, bin_data["edges"]) - 1
            bin_idx = np.clip(bin_idx, 0, self.max_bins - 1)
            count = bin_data["counts"][bin_idx]
            total = np.sum(bin_data["counts"])
            p = (count + 1) / (total + self.max_bins)  # Laplace smoothing
            prob *= p
        return prob
