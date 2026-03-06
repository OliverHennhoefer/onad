"""Integration test configuration.

Integration tests rely on optional evaluation dependencies and external datasets.
"""

import importlib.util
from pathlib import Path

import pytest

_INTEGRATION_ROOT = Path(__file__).resolve().parent


def _integration_skip_reason() -> str | None:
    """Return skip reason for integration tests, or None if enabled."""
    if importlib.util.find_spec("sklearn") is None:
        return "Integration tests require scikit-learn. Install dev dependencies."
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark integration tests as skipped without aborting test session startup."""
    reason = _integration_skip_reason()
    if reason is None:
        return

    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        item_path = Path(str(getattr(item, "path", item.fspath))).resolve()
        if item_path == _INTEGRATION_ROOT or _INTEGRATION_ROOT in item_path.parents:
            item.add_marker(skip_marker)
