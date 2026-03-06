# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project aims to follow
Semantic Versioning.

## [Unreleased]

### Added

- Public API exports for `onad.model.svm`, `onad.model.stat`, `onad.stream`,
  `onad.transform`, and deep lazy exports.
- Regression tests for feature-order stability in `OnlineIsolationForest`.
- Regression tests for `keys=` initialization in multivariate statistical models.
- Optional dependency extras for `parquet` and benchmark tooling.
- Repository standards and CI/security/release workflows.

### Changed

- `OnlineIsolationForest` now enforces deterministic feature ordering and key-set checks.
- Dataset module version now uses `onad.__version__` as single source of truth.
- Documentation was rewritten to match the current public API.

### Fixed

- `keys=` initialization path in multivariate statistical detectors.
- Deep tests now skip gracefully when `torch` is unavailable.
- Legacy parquet streamer now fails with a clear optional-dependency message.
- Packaging metadata now correctly declares MIT classifier.
