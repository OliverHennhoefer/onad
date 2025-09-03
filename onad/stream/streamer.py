import os
from collections.abc import Iterator
from enum import Enum
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


class Dataset(Enum):
    FRAUD = "./resources/fraud.parquet"
    SHUTTLE = "./resources/shuttle.parquet"
    SMD = "onad/stream/resources/smd.parquet"  # Entity 1


class ParquetStreamer:
    """
    Memory-efficient streaming reader for Parquet files with configurable label handling and data sanitization.

    This class provides efficient row-by-row iteration over Parquet files while maintaining low memory usage.
    It supports flexible label column specification and automatic type conversion to ensure downstream
    compatibility.

    Args:
        dataset: Either a Dataset enum value or a string path to a Parquet file
        label_column: Name of the column to use as labels. If None, uses the last column (default: None)
        sanitize_floats: Whether to convert all feature values to native Python floats (default: True)

    Example:
        # Using enum dataset with default label column (last column)
        with ParquetStreamer(Dataset.FRAUD) as streamer:
            for features, label in streamer:
                print(f"Features: {features}, Label: {label}")

        # Using custom label column and path
        with ParquetStreamer("/path/to/data.parquet", label_column="target") as streamer:
            for features, label in streamer:
                process_data(features, label)

        # Disable float sanitization for performance
        with ParquetStreamer(Dataset.SHUTTLE, sanitize_floats=False) as streamer:
            for features, label in streamer:
                train_model(features, label)
    """

    def __init__(
        self,
        dataset: str | Dataset,
        label_column: str | None = None,
        sanitize_floats: bool = True,
    ) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if isinstance(dataset, Dataset):
            # Make the dataset path relative to the current Python file
            self.file_path = os.path.join(current_dir, dataset.value)
        else:
            self.file_path = dataset

        self.label_column = label_column
        self.sanitize_floats = sanitize_floats
        self.file_handle: Any = None
        self.parquet_file: pq.ParquetFile | None = None

    def __enter__(self) -> "ParquetStreamer":
        try:
            # Explicitly open file handle for proper resource management
            self.file_handle = open(self.file_path, "rb")
            self.parquet_file = pq.ParquetFile(self.file_handle)
            return self
        except Exception as e:
            # Ensure file handle is closed if ParquetFile creation fails
            if self.file_handle:
                self.file_handle.close()
            raise RuntimeError(f"Failed to open Parquet file: {e}") from e

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Explicitly close file handle
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        self.parquet_file = None

    def __iter__(self) -> Iterator[tuple[dict[str, float], Any]]:
        if self.parquet_file is None:
            raise RuntimeError(
                "Parquet file not opened. Use with statement to open it."
            )

        # Get total rows efficiently from metadata (O(1) operation)
        total_rows = self.parquet_file.metadata.num_rows
        progress_bar = tqdm(total=total_rows, desc="Progress", unit="row")

        # Iterate through batches and process row by row
        for batch in self.parquet_file.iter_batches():
            # Extract column names and determine feature/label columns
            columns = batch.column_names
            if not columns:
                continue

            # Handle label column determination
            if self.label_column:
                if self.label_column not in columns:
                    raise ValueError(
                        f"Label column '{self.label_column}' not found in data. Available columns: {columns}"
                    )
                label_col = self.label_column
                feature_cols = [col for col in columns if col != label_col]
            else:
                # Default behavior: use last column as label
                label_col = columns[-1]
                feature_cols = columns[:-1]

            # Convert batch to pandas DataFrame for efficient row iteration
            df = batch.to_pandas()

            # Process each row individually
            for _, row in df.iterrows():
                # Create feature dictionary
                x = {col: row[col] for col in feature_cols}
                y = row[label_col]

                # Apply float sanitization if enabled
                if self.sanitize_floats:
                    x = {k: float(v) for k, v in x.items()}

                yield x, y
                progress_bar.update(1)

        progress_bar.close()
