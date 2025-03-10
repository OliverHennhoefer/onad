import os
from enum import Enum

import pyarrow.parquet as pq
from tqdm import tqdm


class Dataset(Enum):
    FRAUD = "./resources/fraud.parquet"
    SHUTTLE = "onad/stream/resources/shuttle.parquet"
    SMD = "onad/stream/resources/smd.parquet"  # Entity 1


class ParquetStreamer:
    def __init__(self, dataset: str | Dataset):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if isinstance(dataset, Dataset):
            # Make the dataset path relative to the current Python file
            self.file_path = os.path.join(current_dir, dataset.value)
        else:
            self.file_path = dataset

    def __enter__(self):
        try:
            self.parquet_file = pq.ParquetFile(self.file_path)
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to open Parquet file: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parquet_file:
            self.parquet_file = None  # Close or clean up any resources if necessary

    def __iter__(self):
        if self.parquet_file is None:
            raise RuntimeError(
                "Parquet file not opened. Use with statement to open it."
            )

        # Get the total number of rows
        total_rows = sum(batch.num_rows for batch in self.parquet_file.iter_batches())
        progress_bar = tqdm(total=total_rows, desc="Progress", unit="row")

        # Iterate through batches and then rows within each batch
        for batch in self.parquet_file.iter_batches():
            # Extract column names and split into features and label
            columns = batch.column_names
            if not columns:
                continue
            feature_cols = columns[:-1]
            label_col = columns[-1]

            # Convert each feature column and label column to Python lists
            feature_data = {col: batch[col].to_pylist() for col in feature_cols}
            label_data = batch[label_col].to_pylist()

            num_rows = batch.num_rows
            for i in range(num_rows):
                # Assemble x as a dictionary with feature column names as keys
                x = {col: feature_data[col][i] for col in feature_cols}
                y = label_data[i]
                yield x, y
                progress_bar.update(1)

        progress_bar.close()
