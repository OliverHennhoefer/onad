from http.client import RemoteDisconnected
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

import pytest

from aberrant.stream.dataset.loader import DatasetManager


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
