import hashlib
import io
from dataclasses import replace
from http.client import RemoteDisconnected
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pytest

from aberrant.stream.dataset import Dataset
from aberrant.stream.dataset.loader import DatasetManager
from aberrant.stream.dataset.registry import get_dataset_info


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        start = self._offset
        end = min(len(self._payload), start + size)
        self._offset = end
        return self._payload[start:end]


def _valid_npz_payload() -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        y=np.array([0, 1], dtype=int),
    )
    return buffer.getvalue()


def test_download_retries_and_succeeds() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(
            cache_dir=str(tmp_path),
            download_retries=2,
            retry_backoff_seconds=0.0,
        )
        dest_path = tmp_path / "shuttle.tmp"

        with patch(
            "aberrant.stream.dataset.loader.urllib.request.urlopen",
            side_effect=[
                URLError("temporary network issue"),
                _FakeResponse(b"payload"),
            ],
        ) as mock_urlopen:
            manager._download_with_progress(
                "https://example.com/shuttle.npz",
                dest_path,
            )

        assert mock_urlopen.call_count == 2
        assert dest_path.read_bytes() == b"payload"


def test_download_raises_after_retry_exhaustion() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(
            cache_dir=str(tmp_path),
            download_retries=3,
            retry_backoff_seconds=0.0,
        )
        dest_path = tmp_path / "shuttle.tmp"

        with (
            patch(
                "aberrant.stream.dataset.loader.urllib.request.urlopen",
                side_effect=RemoteDisconnected("no response"),
            ) as mock_urlopen,
            pytest.raises(RuntimeError, match="after 3 attempts"),
        ):
            manager._download_with_progress(
                "https://example.com/shuttle.npz",
                dest_path,
            )

        assert mock_urlopen.call_count == 3
        assert not dest_path.exists()


def test_download_rejects_checksum_mismatch() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(
            cache_dir=str(tmp_path),
            download_retries=1,
            retry_backoff_seconds=0.0,
        )

        with (
            patch(
                "aberrant.stream.dataset.loader.urllib.request.urlopen",
                return_value=_FakeResponse(_valid_npz_payload()),
            ),
            pytest.raises(RuntimeError, match="failed validation"),
        ):
            manager.download(Dataset.SHUTTLE, force=True)


def test_load_fails_for_corrupted_cached_file_without_auto_download() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(cache_dir=str(tmp_path))
        cache_path = tmp_path / "shuttle.npz"
        cache_path.write_bytes(b"not-an-npz")

        with pytest.raises(FileNotFoundError, match="not found in cache"):
            manager.load(Dataset.SHUTTLE, auto_download=False)


def test_load_raises_runtime_error_when_offline() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(
            cache_dir=str(tmp_path),
            download_retries=2,
            retry_backoff_seconds=0.0,
        )

        with (
            patch(
                "aberrant.stream.dataset.loader.urllib.request.urlopen",
                side_effect=URLError("offline"),
            ) as mock_urlopen,
            pytest.raises(RuntimeError, match="Failed to download dataset"),
        ):
            manager.load(Dataset.SHUTTLE, auto_download=True)

        assert mock_urlopen.call_count == 2


def test_download_updates_stale_metadata_hash_after_successful_validation() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(
            cache_dir=str(tmp_path),
            download_retries=1,
            retry_backoff_seconds=0.0,
        )
        payload = _valid_npz_payload()
        payload_hash = hashlib.sha256(payload).hexdigest()
        original_info = get_dataset_info(Dataset.SHUTTLE)
        trusted_info = replace(original_info, sha256=payload_hash)
        manager.metadata["datasets"][Dataset.SHUTTLE.value] = {
            "downloaded": True,
            "hash": "stale-old-hash",
            "size": 123,
            "release_tag": "old-tag",
        }

        with (
            patch(
                "aberrant.stream.dataset.loader.urllib.request.urlopen",
                return_value=_FakeResponse(payload),
            ),
            patch(
                "aberrant.stream.dataset.loader.get_dataset_info",
                return_value=trusted_info,
            ),
        ):
            cached_path = manager.download(Dataset.SHUTTLE, force=True)

        assert cached_path.exists()
        assert (
            manager.metadata["datasets"][Dataset.SHUTTLE.value]["hash"] == payload_hash
        )


def test_download_reuses_verified_hash_without_rehashing() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        tmp_path = Path(temp_dir)
        manager = DatasetManager(
            cache_dir=str(tmp_path),
            download_retries=1,
            retry_backoff_seconds=0.0,
        )
        payload = _valid_npz_payload()
        payload_hash = hashlib.sha256(payload).hexdigest()
        original_info = get_dataset_info(Dataset.SHUTTLE)
        trusted_info = replace(original_info, sha256=payload_hash)

        with (
            patch(
                "aberrant.stream.dataset.loader.urllib.request.urlopen",
                return_value=_FakeResponse(payload),
            ),
            patch(
                "aberrant.stream.dataset.loader.get_dataset_info",
                return_value=trusted_info,
            ),
            patch.object(
                manager,
                "_calculate_file_hash",
                wraps=manager._calculate_file_hash,
            ) as hash_spy,
        ):
            cached_path = manager.download(Dataset.SHUTTLE, force=True)

        assert cached_path.exists()
        assert (
            manager.metadata["datasets"][Dataset.SHUTTLE.value]["hash"] == payload_hash
        )
        assert hash_spy.call_count == 1
