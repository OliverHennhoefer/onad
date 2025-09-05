"""Dataset loading and management system with remote download capabilities.

This module provides automated dataset downloading, caching, and management
for anomaly detection benchmarks hosted on GitHub releases.
"""

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.error import URLError

import numpy as np
from tqdm import tqdm

from .registry import DATASET_REGISTRY, Dataset, get_dataset_info
from .streamers import DatasetStreamer


class DatasetManager:
    """Manages dataset downloading, caching, and loading.

    This class handles automatic dataset downloads from GitHub releases,
    local caching with version management, and provides a unified interface
    for dataset access.

    Args:
        cache_dir: Directory for local dataset cache (default: ~/.onad/datasets)
        github_repo: GitHub repository in format 'owner/repo' (default: ONAD repo)
        release_tag: GitHub release tag to download from (default: 'v0.1.0-datasets')

    Example:
        manager = DatasetManager()
        dataset = manager.load(Dataset.FRAUD)

        for features, label in dataset.stream():
            process_data(features, label)
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        github_repo: str = "OliverHennhoefer/nonconform",
        release_tag: str = "v0.9.17-datasets",
    ) -> None:
        self.github_repo = github_repo
        self.release_tag = release_tag

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".onad", "datasets")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {"version": "1.0", "datasets": {}}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except OSError as e:
            print(f"Warning: Could not save cache metadata: {e}")

    def _get_dataset_url(self, dataset: Dataset) -> str:
        """Get GitHub release download URL for a dataset."""
        info = get_dataset_info(dataset)
        filename = f"{info.filename}.npz"
        return (
            f"https://github.com/{self.github_repo}/releases/download/"
            f"{self.release_tag}/{filename}"
        )

    def _get_cache_path(self, dataset: Dataset) -> Path:
        """Get local cache path for a dataset."""
        info = get_dataset_info(dataset)
        return self.cache_dir / f"{info.filename}.npz"

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _download_with_progress(self, url: str, dest_path: Path) -> None:
        """Download a file with progress bar."""
        try:
            # Get file size for progress bar
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get("Content-Length", 0))

            # Download with progress bar
            with (
                urllib.request.urlopen(url) as response,
                open(dest_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dest_path.name}",
                ) as pbar,
            ):
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        except URLError as e:
            raise RuntimeError(f"Failed to download dataset: {e}") from e

    def _is_dataset_cached(self, dataset: Dataset) -> bool:
        """Check if dataset is cached locally."""
        cache_path = self._get_cache_path(dataset)
        return cache_path.exists()

    def _validate_cached_dataset(self, dataset: Dataset) -> bool:
        """Validate cached dataset integrity."""
        cache_path = self._get_cache_path(dataset)
        if not cache_path.exists():
            return False

        # Check if file is corrupted (basic validation)
        try:
            with np.load(cache_path) as npz_file:
                # Basic validation - check required arrays exist
                return "X" in npz_file and "y" in npz_file
        except Exception:
            return False

    def download(self, dataset: Dataset, force: bool = False) -> Path:
        """Download a dataset to local cache.

        Args:
            dataset: Dataset to download
            force: Force re-download even if cached (default: False)

        Returns:
            Path to cached dataset file

        Raises:
            RuntimeError: If download fails
            KeyError: If dataset not found in registry
        """
        if dataset not in DATASET_REGISTRY:
            raise KeyError(f"Dataset {dataset} not found in registry")

        cache_path = self._get_cache_path(dataset)

        # Check if already cached and valid
        if not force and self._is_dataset_cached(dataset):
            if self._validate_cached_dataset(dataset):
                print(f"Dataset {dataset.value} already cached at {cache_path}")
                return cache_path
            else:
                print(f"Cached dataset {dataset.value} is corrupted, re-downloading...")

        # Download dataset
        url = self._get_dataset_url(dataset)
        temp_path = cache_path.with_suffix(".tmp")

        try:
            print(f"Downloading {dataset.value} from {url}...")
            self._download_with_progress(url, temp_path)

            # Validate downloaded file
            if not self._validate_cached_dataset_at_path(temp_path):
                raise RuntimeError("Downloaded dataset failed validation")

            # Move to final location
            if cache_path.exists():
                cache_path.unlink()
            temp_path.rename(cache_path)

            # Update metadata
            file_hash = self._calculate_file_hash(cache_path)
            self.metadata["datasets"][dataset.value] = {
                "downloaded": True,
                "hash": file_hash,
                "size": cache_path.stat().st_size,
                "release_tag": self.release_tag,
            }
            self._save_metadata()

            print(f"Successfully downloaded {dataset.value} to {cache_path}")
            return cache_path

        except Exception as e:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(
                f"Failed to download dataset {dataset.value}: {e}"
            ) from e

    def _validate_cached_dataset_at_path(self, path: Path) -> bool:
        """Validate dataset at specific path."""
        try:
            with np.load(path) as npz_file:
                return "X" in npz_file and "y" in npz_file
        except Exception:
            return False

    def load(
        self, dataset: Dataset, auto_download: bool = True, **kwargs
    ) -> DatasetStreamer:
        """Load a dataset for streaming.

        Args:
            dataset: Dataset to load
            auto_download: Automatically download if not cached (default: True)
            **kwargs: Additional arguments passed to DatasetStreamer

        Returns:
            DatasetStreamer instance for the requested dataset

        Raises:
            FileNotFoundError: If dataset not cached and auto_download is False
            RuntimeError: If download fails
            KeyError: If dataset not found in registry
        """
        if dataset not in DATASET_REGISTRY:
            raise KeyError(f"Dataset {dataset} not found in registry")

        # Check if dataset is cached
        if not self._is_dataset_cached(dataset) or not self._validate_cached_dataset(
            dataset
        ):
            if auto_download:
                self.download(dataset)
            else:
                raise FileNotFoundError(
                    f"Dataset {dataset.value} not found in cache. "
                    f"Set auto_download=True or call download() first."
                )

        # Create and return streamer
        info = get_dataset_info(dataset)
        return DatasetStreamer(
            dataset=dataset, data_dir=str(self.cache_dir), dataset_info=info, **kwargs
        )

    def list_cached(self) -> dict[str, dict]:
        """List all cached datasets with metadata.

        Returns:
            Dictionary mapping dataset names to cache metadata
        """
        cached = {}
        for dataset_name, metadata in self.metadata.get("datasets", {}).items():
            if metadata.get("downloaded", False):
                cached[dataset_name] = metadata
        return cached

    def clear_cache(self, dataset: Dataset | None = None) -> None:
        """Clear dataset cache.

        Args:
            dataset: Specific dataset to remove from cache, or None to clear all
        """
        if dataset is not None:
            # Clear specific dataset
            cache_path = self._get_cache_path(dataset)
            if cache_path.exists():
                cache_path.unlink()
                print(f"Removed {dataset.value} from cache")

            # Update metadata
            if dataset.value in self.metadata.get("datasets", {}):
                del self.metadata["datasets"][dataset.value]
                self._save_metadata()
        else:
            # Clear entire cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Reset metadata
            self.metadata = {"version": "1.0", "datasets": {}}
            self._save_metadata()
            print("Cleared all dataset cache")

    def get_cache_size(self) -> int:
        """Get total size of dataset cache in bytes.

        Returns:
            Total cache size in bytes
        """
        total_size = 0
        if self.cache_dir.exists():
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file() and file_path.suffix == ".npz":
                    total_size += file_path.stat().st_size
        return total_size

    def __repr__(self) -> str:
        cache_size_mb = self.get_cache_size() / (1024 * 1024)
        cached_count = len(self.list_cached())
        return (
            f"DatasetManager(cache_dir='{self.cache_dir}', "
            f"cached_datasets={cached_count}, "
            f"cache_size={cache_size_mb:.1f}MB)"
        )


class _DefaultManagerHolder:
    """Internal singleton holder for default DatasetManager.

    This class encapsulates the singleton pattern for the default DatasetManager
    instance, eliminating the need for global statements while maintaining
    the same public API.
    """

    _instance: DatasetManager | None = None

    @classmethod
    def get(cls) -> DatasetManager:
        """Get or create the default DatasetManager instance."""
        if cls._instance is None:
            cls._instance = DatasetManager()
        return cls._instance

    @classmethod
    def set_cache_dir(cls, cache_dir: str) -> None:
        """Set the cache directory by creating a new DatasetManager instance."""
        cls._instance = DatasetManager(cache_dir=cache_dir)


def get_default_manager() -> DatasetManager:
    """Get the default global DatasetManager instance.

    Returns:
        Global DatasetManager instance
    """
    return _DefaultManagerHolder.get()


def set_cache_dir(cache_dir: str) -> None:
    """Set the global cache directory for datasets.

    Args:
        cache_dir: Path to cache directory
    """
    _DefaultManagerHolder.set_cache_dir(cache_dir)
