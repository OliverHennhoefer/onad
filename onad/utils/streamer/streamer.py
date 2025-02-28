import os
import numpy as np

from tqdm import tqdm

from onad.utils.streamer.datasets import Dataset


class NPZStreamer:
    def __init__(self, dataset: Dataset, verbose=True):
        """
        Initializes the NPZStreamer with a given .npz file path.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, dataset.value)
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
