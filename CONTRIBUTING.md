# Contributing to ONAD

Thanks for contributing.

## Development setup

```bash
uv sync --extra dev --extra docs --extra eval
```

## Quality gates

Run these before opening a PR:

```bash
uv run python -m ruff check .
uv run python -m mypy onad/base/model.py onad/base/transformer.py onad/base/pipeline.py onad/model/threshold.py onad/model/quantile_threshold.py onad/model/__init__.py onad/model/iforest/__init__.py onad/model/distance/__init__.py onad/model/svm/__init__.py onad/model/deep/__init__.py
uv run python -m pytest -q
uv run python -m build
```

To run dataset-backed integration tests:

```bash
ONAD_RUN_INTEGRATION=1 uv run python -m pytest tests/integration -q
```

## PR expectations

- Keep changes scoped and reviewable.
- Add or update tests for behavior changes.
- Update docs for user-facing API changes.
- Add a changelog entry under `Unreleased` in `CHANGELOG.md`.

## Optional dependency policy

- Base test suite must pass without optional extras.
- Tests requiring optional dependencies must skip when dependency is absent.

## Commit guidance

- Use clear, imperative commit messages.
- Avoid mixing unrelated refactors with bug fixes.
