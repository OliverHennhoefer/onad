"""Shared sample budgets for integration tests.

Budgets are intentionally non-uniform:
- standard models use a medium test window
- slower models use a shorter test window
- concept-drift-sensitive models use a longer window
"""

WARMUP_SAMPLES = 1000
MAX_TEST_STANDARD = 600
MAX_TEST_SHORT = 500
MAX_TEST_LONG = 700
