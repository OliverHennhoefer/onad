# API Reference

This section documents the supported ONAD public surface.

## Public modules

- `onad.base`
- `onad.drift`
- `onad.model`
- `onad.model.iforest`
- `onad.model.distance`
- `onad.model.svm`
- `onad.model.stat`
- `onad.transform`
- `onad.transform.preprocessing`
- `onad.transform.projection`
- `onad.stream.dataset`

## Scoring conventions

- Binary: `ThresholdModel` and threshold wrappers.
- Bounded `[0, 1]`: isolation-forest variants.
- Continuous/unbounded: distance, SVM, and statistical models.

## Stability policy

- Objects exported through package `__init__.py` files above are considered public.
- Internal helpers and private names (`_...`) can change between releases.
