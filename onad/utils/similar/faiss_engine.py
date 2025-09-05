import collections

import faiss
import numpy as np

from onad.base.similarity import BaseSimilaritySearchEngine


class FaissSimilaritySearchEngine(BaseSimilaritySearchEngine):
    """
    FAISS-based similarity search engine for efficient nearest neighbor search.

    Uses a sliding window of data points and incrementally updates the FAISS index
    for efficient similarity search operations.

    Args:
        window_size: Maximum number of data points to keep in the sliding window.
        warm_up: Minimum number of data points required before search can be performed.
    """

    def __init__(self, window_size: int, warm_up: int) -> None:
        self.window: collections.deque = collections.deque(maxlen=window_size)
        self.window_size = window_size

        self._check_params(window_size, warm_up)
        self.warm_up: int = warm_up

        self.index: faiss.Index | None = None
        self.keys: list[str] | None = None
        self._index_needs_rebuild = True

    def append(self, x: dict[str, float]) -> None:
        """
        Add a data point to the search engine.

        Args:
            x: Dictionary representing a data point with feature names as keys.
        """
        # Check if we need to rebuild keys (new features or first data point)
        if self.keys is None:
            current_keys = sorted(x.keys())
            self.keys = current_keys
            self._index_needs_rebuild = True
        else:
            current_keys = sorted(x.keys())
            if current_keys != self.keys:
                # New features detected - need to rebuild everything
                self.keys = current_keys
                self._index_needs_rebuild = True

        self.window.append(x)

        # Only rebuild index if necessary (new features or index doesn't exist)
        if self._index_needs_rebuild or self.index is None:
            self._rebuild_index()
            self._index_needs_rebuild = False
        else:
            # For incremental updates, we unfortunately need to rebuild with FAISS IndexFlatL2
            # More sophisticated indices like IndexIVFFlat support add(), but IndexFlatL2 doesn't
            self._rebuild_index()

    def search(self, item: dict[str, float], n_neighbors: int) -> float:
        """
        Search for the n nearest neighbors of a data point.

        Args:
            item: Dictionary representing the query data point.
            n_neighbors: Number of nearest neighbors to find.

        Returns:
            Mean distance to the k nearest neighbors, or 0.0 if not enough data.

        Raises:
            ValueError: If n_neighbors is not positive or exceeds available data points.
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        if len(self.window) < self.warm_up:
            return 0.0

        if n_neighbors > len(self.window):
            raise ValueError(
                f"n_neighbors ({n_neighbors}) cannot exceed window size ({len(self.window)})"
            )

        # Convert query point to vector
        x = np.array(
            [item.get(key, 0.0) for key in self.keys], dtype=np.float32
        ).reshape(1, -1)

        # Search for nearest neighbors
        distances, _ = self.index.search(x, k=min(n_neighbors, self.index.ntotal))
        return float(np.mean(distances[0]))

    def _rebuild_index(self) -> None:
        """
        Rebuild the FAISS index from current window data.

        This is called when new features are detected or when the index needs initialization.
        """
        if not self.window or self.keys is None:
            return

        # Convert all window data to matrix
        data_matrix = np.array(
            [[point.get(key, 0.0) for key in self.keys] for point in self.window],
            dtype=np.float32,
        )

        # Create new index and add all data
        self.index = faiss.IndexFlatL2(len(self.keys))
        if len(data_matrix) > 0:
            self.index.add(data_matrix)

    def _get_window_data(self) -> tuple[list[str], np.ndarray]:
        """
        Extract keys and data matrix from current window.

        Returns:
            Tuple of (sorted feature keys, data matrix).
        """
        keys = sorted({key for dict_ in self.window for key in dict_})
        return keys, np.array(
            [[dict_.get(key, 0.0) for key in keys] for dict_ in self.window],
            dtype=np.float32,
        )

    @staticmethod
    def _check_params(window_size: int, warm_up: int) -> None:
        """
        Validate constructor parameters.

        Args:
            window_size: Maximum window size.
            warm_up: Minimum data points for search.

        Raises:
            ValueError: If parameters are invalid.
        """
        if window_size <= 0:
            raise ValueError(f"window_size ({window_size}) must be positive")
        if warm_up <= 0:
            raise ValueError(f"warm_up ({warm_up}) must be positive")
        if window_size < warm_up:
            raise ValueError(
                f"window_size ({window_size}) must be >= warm_up ({warm_up})"
            )

    def __repr__(self) -> str:
        """Return a string representation of the FAISS engine."""
        return (
            f"FaissSimilaritySearchEngine(window_size={self.window_size}, "
            f"warm_up={self.warm_up}, current_size={len(self.window)})"
        )
