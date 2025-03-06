import io
import os
import csv
import zipfile

import numpy as np

from tqdm import tqdm


class NPZStreamer:
    def __init__(self, file_path: str, verbose=True):
        """
        Initializes the NPZStreamer with a given .npz file path.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, file_path)
        self.npz_file = None
        self.X = None
        self.Y = None
        self.index = 0
        self.length = 0  # Minimum length of X and Y to prevent out-of-bounds errors
        self.verbose = verbose  # Controls tqdm display

    def __enter__(self):
        """Enables usage with the 'with' statement."""
        self.npz_file = np.load(self.file_path, allow_pickle=True)

        # Ensure 'X' and 'y' exist in the .npz file
        if "X" not in self.npz_file or "y" not in self.npz_file:
            raise KeyError("The .npz file must contain both 'X' and 'y' arrays.")

        self.X = self.npz_file["X"]
        self.Y = self.npz_file["y"]

        # Ensure X and Y have the same length
        self.length = min(len(self.X), len(self.Y))
        self.index = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the .npz file when exiting the context."""
        if self.npz_file:
            self.npz_file.close()

    def __iter__(self):
        """Returns the iterator object and initializes tqdm if verbose."""
        self.index = 0
        self.pbar = tqdm(
            total=self.length, disable=not self.verbose, desc="Streaming NPZ"
        )
        return self

    def __next__(self):
        """Returns the next ({'key1': val1, 'key2': val2, ...}, number)."""
        if self.index >= self.length:
            self.pbar.close()  # Close tqdm when iteration ends
            raise StopIteration

        # Convert X[i] (array) to a dictionary
        x_dict = {f"key_{i}": val for i, val in enumerate(self.X[self.index])}

        # Get Y[i] as a plain number
        y_value = self.Y[
            self.index
        ].item()  # Ensures it's a regular number, not a numpy type

        self.index += 1
        self.pbar.update(1)  # Update tqdm progress
        return x_dict, y_value


class ZippedCSVStreamer:
    def __init__(self, file_path, verbose=True):
        """
        Initializes the CSVStreamer with a given file path.

        Args:
            file_path (str): Path to the zipped CSV file (relative or absolute).
            verbose (bool): Whether to display a progress bar.
        """
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # If file_path is not absolute, make it relative to script_dir
        if not os.path.isabs(file_path):
            self.file_path = os.path.join(script_dir, file_path)
        else:
            self.file_path = file_path

        self.zip_file = None
        self.csv_reader = None
        self.X = []
        self.Y = []
        self.index = 0
        self.length = 0
        self.verbose = verbose

        # Extract filename from the path
        _, self.filename = os.path.split(self.file_path)

    def __enter__(self):
        """Enables usage with the 'with' statement."""
        self.zip_file = zipfile.ZipFile(self.file_path, 'r')
        # Get the first file in the zip if it's the only one, otherwise use the base filename
        if len(self.zip_file.namelist()) == 1:
            csv_filename = self.zip_file.namelist()[0]
        else:
            # Remove .zip extension and try to find a matching CSV
            base_name = os.path.splitext(self.filename)[0]
            matching_files = [f for f in self.zip_file.namelist() if f.startswith(base_name)]
            if matching_files:
                csv_filename = matching_files[0]
            else:
                # If no matching filename, use the first CSV file
                csv_files = [f for f in self.zip_file.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV file found in the zip archive.")
                csv_filename = csv_files[0]

        # Open the zip file in binary mode and convert to text
        with self.zip_file.open(csv_filename, 'r') as binary_file:
            # Wrap the binary file with TextIOWrapper to convert to text mode
            text_file = io.TextIOWrapper(binary_file, encoding='utf-8')
            reader = csv.reader(text_file)
            data = list(reader)  # Load entire CSV into memory

        if len(data) < 2:
            raise ValueError("CSV file must have at least one data row and one header row.")

        self.X = [list(map(float, row[:-1])) for row in data[1:]]  # Convert feature values to floats
        self.Y = [float(row[-1]) for row in data[1:]]  # Convert labels to floats

        self.length = min(len(self.X), len(self.Y))
        self.index = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the zip file when exiting the context."""
        if self.zip_file:
            self.zip_file.close()

    def __iter__(self):
        """Returns the iterator object and initializes tqdm if verbose."""
        self.index = 0
        self.pbar = tqdm(total=self.length, disable=not self.verbose, desc="Streaming CSV")
        return self

    def __next__(self):
        """Returns the next ({'key1': val1, 'key2': val2, ...}, number)."""
        if self.index >= self.length:
            self.pbar.close()
            raise StopIteration

        # Convert X[i] (list) to a dictionary
        x_dict = {f"key_{i}": val for i, val in enumerate(self.X[self.index])}
        # Get Y[i] as a plain number
        y_value = self.Y[self.index]

        self.index += 1
        self.pbar.update(1)
        return x_dict, y_value