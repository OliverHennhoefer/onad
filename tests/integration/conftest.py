"""Integration test configuration.

Integration tests rely on optional evaluation dependencies and external datasets.
They are skipped by default unless explicitly enabled.
"""

import os

import pytest

pytest.importorskip("sklearn")

if os.getenv("ONAD_RUN_INTEGRATION") != "1":
    pytest.skip(
        "Integration tests are disabled by default. "
        "Set ONAD_RUN_INTEGRATION=1 to execute dataset-backed tests.",
        allow_module_level=True,
    )
