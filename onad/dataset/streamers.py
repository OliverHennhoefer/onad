"""Dataset streaming utilities for efficient data iteration.

This module provides streaming interfaces for different data formats used in
anomaly detection datasets, with a focus on memory efficiency and compatibility.
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
from tqdm import tqdm

from .registry import Dataset, DatasetInfo


class NpzStreamer:
    """Memory-efficient streaming reader for NPZ files with configurable data handling.

    This class provides efficient row-by-row iteration over NumPy compressed (.npz) files
    while maintaining low memory usage. It supports flexible label handling and automatic
    type conversion for downstream compatibility.

    Args:
        file_path: Path to the NPZ file to stream
        dataset_info: Optional DatasetInfo for metadata (enables progress bars)
        feature_prefix: Prefix for feature column names (default: "feature_")
        label_column: Name of the label array in NPZ file (default: "y")
        feature_column: Name of the feature array in NPZ file (default: "X")
        sanitize_floats: Whether to convert all values to Python floats (default: True)

    Example:
        # Stream from NPZ file with progress bar
        info = get_dataset_info(Dataset.FRAUD)
        with NpzStreamer("/path/to/fraud.npz", info) as streamer:
            for features, label in streamer:
                print(f"Features: {features}, Label: {label}")

        # Stream without metadata
        with NpzStreamer("/path/to/data.npz") as streamer:
            for features, label in streamer:
                process_data(features, label)
    """

    def __init__(
        self,
        file_path: str,
        dataset_info: DatasetInfo | None = None,
        feature_prefix: str = "feature_",
        label_column: str = "y",
        feature_column: str = "X",
        sanitize_floats: bool = True,
    ) -> None:
        self.file_path = file_path
        self.dataset_info = dataset_info
        self.feature_prefix = feature_prefix
        self.label_column = label_column
        self.feature_column = feature_column
        self.sanitize_floats = sanitize_floats
        self.npz_file: np.lib.npyio.NpzFile | None = None

    def __enter__(self) -> "NpzStreamer":
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"NPZ file not found: {self.file_path}")

            self.npz_file = np.load(self.file_path)

            # Validate required arrays exist
            if self.feature_column not in self.npz_file:
                raise KeyError(
                    f"Feature array '{self.feature_column}' not found in NPZ file"
                )
            if self.label_column not in self.npz_file:
                raise KeyError(
                    f"Label array '{self.label_column}' not found in NPZ file"
                )

            return self

        except Exception as e:
            if self.npz_file:
                self.npz_file.close()
                self.npz_file = None
            raise RuntimeError(f"Failed to open NPZ file: {e}") from e

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.npz_file:
            self.npz_file.close()
            self.npz_file = None

    def __iter__(self) -> Iterator[tuple[dict[str, float], Any]]:
        if self.npz_file is None:
            raise RuntimeError("NPZ file not opened. Use with statement to open it.")

        # Load feature and label arrays
        features_array = self.npz_file[self.feature_column]
        labels_array = self.npz_file[self.label_column]

        # Validate array shapes
        if features_array.shape[0] != labels_array.shape[0]:
            raise ValueError(
                f"Feature and label arrays have different lengths: "
                f"{features_array.shape[0]} vs {labels_array.shape[0]}"
            )

        n_samples, n_features = features_array.shape

        # Set up progress bar if dataset info is available
        progress_desc = "Loading data"
        if self.dataset_info:
            progress_desc = f"Loading {self.dataset_info.name}"

        progress_bar = tqdm(total=n_samples, desc=progress_desc, unit="sample")

        try:
            # Generate feature column names
            feature_names = [f"{self.feature_prefix}{i}" for i in range(n_features)]

            # Stream data row by row
            for i in range(n_samples):
                # Extract feature vector and label
                feature_vector = features_array[i]
                label = labels_array[i]

                # Create feature dictionary
                features_dict = dict(zip(feature_names, feature_vector, strict=False))

                # Apply float sanitization if enabled
                if self.sanitize_floats:
                    features_dict = {k: float(v) for k, v in features_dict.items()}
                    # Convert label to appropriate Python type
                    if isinstance(label, np.integer):
                        label = int(label)
                    elif isinstance(label, np.floating):
                        label = float(label)

                yield features_dict, label
                progress_bar.update(1)

        finally:
            progress_bar.close()


class DatasetStreamer:
    """High-level dataset streaming interface that abstracts file formats.

    This class provides a unified interface for streaming different dataset formats
    and handles automatic format detection and streaming optimization.

    Args:
        dataset: Dataset enum value
        data_dir: Directory containing dataset files
        dataset_info: DatasetInfo object for metadata

    Example:
        info = get_dataset_info(Dataset.FRAUD)
        streamer = DatasetStreamer(Dataset.FRAUD, "/path/to/data", info)

        for features, label in streamer.stream():
            process_data(features, label)
    """

    def __init__(
        self, dataset: Dataset, data_dir: str, dataset_info: DatasetInfo, **kwargs
    ) -> None:
        self.dataset = dataset
        self.data_dir = data_dir
        self.dataset_info = dataset_info
        self.kwargs = kwargs

        # Determine file path and format
        self.file_path = os.path.join(data_dir, f"{dataset_info.filename}.npz")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}. "
                f"Please ensure the dataset is downloaded."
            )

    def stream(self) -> Iterator[tuple[dict[str, float], Any]]:
        """Stream the dataset data.

        Yields:
            Tuple of (features_dict, label) for each sample
        """
        with NpzStreamer(self.file_path, self.dataset_info, **self.kwargs) as streamer:
            yield from streamer

    def get_metadata(self) -> DatasetInfo:
        """Get dataset metadata.

        Returns:
            DatasetInfo object with dataset metadata
        """
        return self.dataset_info

    def __repr__(self) -> str:
        return (
            f"DatasetStreamer(dataset={self.dataset}, "
            f"n_samples={self.dataset_info.n_samples}, "
            f"n_features={self.dataset_info.n_features}, "
            f"anomaly_rate={self.dataset_info.anomaly_rate:.3f})"
        )


class BatchStreamer:
    """Batch streaming wrapper for efficient processing of large datasets.

    This class wraps any dataset streamer and provides batch iteration for
    more efficient processing of large datasets.

    Args:
        base_streamer: Base streamer to wrap
        batch_size: Number of samples per batch (default: 1000)

    Example:
        base_streamer = DatasetStreamer(Dataset.FRAUD, data_dir, info)
        batch_streamer = BatchStreamer(base_streamer, batch_size=5000)

        for feature_batch, label_batch in batch_streamer.stream():
            # feature_batch is a list of feature dicts
            # label_batch is a list of labels
            process_batch(feature_batch, label_batch)
    """

    def __init__(self, base_streamer: DatasetStreamer, batch_size: int = 1000) -> None:
        self.base_streamer = base_streamer
        self.batch_size = batch_size

    def stream(self) -> Iterator[tuple[list[dict[str, float]], list[Any]]]:
        """Stream data in batches.

        Yields:
            Tuple of (feature_batch, label_batch) where each batch contains
            up to batch_size samples
        """
        feature_batch = []
        label_batch = []

        for features, label in self.base_streamer.stream():
            feature_batch.append(features)
            label_batch.append(label)

            if len(feature_batch) >= self.batch_size:
                yield feature_batch, label_batch
                feature_batch = []
                label_batch = []

        # Yield remaining samples if any
        if feature_batch:
            yield feature_batch, label_batch

    def get_metadata(self) -> DatasetInfo:
        """Get dataset metadata from base streamer.

        Returns:
            DatasetInfo object with dataset metadata
        """
        return self.base_streamer.get_metadata()
