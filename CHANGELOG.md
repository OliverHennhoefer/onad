# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project aims to follow
Semantic Versioning.

## [Unreleased]

### Added

- Public API exports for `aberrant.model.svm`, `aberrant.model.stat`, `aberrant.stream`,
  `aberrant.transform`, and deep lazy exports.
- Regression tests for feature-order stability in `OnlineIsolationForest`.
- Regression tests for `keys=` initialization in multivariate statistical models.
- Optional dependency extras for `parquet` and benchmark tooling.
- Repository standards and CI/security/release workflows.
- `KitNET` online deep detector with phased warm-up, tests, docs, and example script.
- `MStream` sketch-based streaming detector with `aberrant.model.sketch` public API,
  unit/integration tests, docs, and example script.

### Changed

- `OnlineIsolationForest` now enforces deterministic feature ordering and key-set checks.
- Dataset module version now uses `aberrant.__version__` as single source of truth.
- Documentation was rewritten to match the current public API.
- `aberrant[dev]` now includes `torch` and `scikit-learn` so the full test suite can
  run from the dev environment.
- Integration tests now run by default (no environment-variable gate).

### Fixed

- `keys=` initialization path in multivariate statistical detectors.
- Deep tests now skip gracefully when `torch` is unavailable.
- Legacy parquet streamer now fails with a clear optional-dependency message.
- Packaging metadata now correctly declares MIT classifier.
