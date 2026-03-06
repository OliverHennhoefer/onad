"""Streaming data interfaces.

Public streaming dataset APIs are re-exported from ``onad.stream.dataset``.
"""

from onad.stream.dataset import (
    DATASET_REGISTRY,
    BatchStreamer,
    Dataset,
    DatasetInfo,
    DatasetManager,
    DatasetStreamer,
    NpzStreamer,
    clear_cache,
    download,
    get_cache_info,
    get_categories,
    get_dataset_info,
    get_default_manager,
    list_available,
    list_by_category,
    list_cached,
    load,
    set_cache_dir,
)

__all__ = [
    "BatchStreamer",
    "DATASET_REGISTRY",
    "Dataset",
    "DatasetInfo",
    "DatasetManager",
    "DatasetStreamer",
    "NpzStreamer",
    "clear_cache",
    "download",
    "get_cache_info",
    "get_categories",
    "get_dataset_info",
    "get_default_manager",
    "list_available",
    "list_by_category",
    "list_cached",
    "load",
    "set_cache_dir",
]
