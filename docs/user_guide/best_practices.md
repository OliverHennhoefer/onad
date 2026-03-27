# Best Practices

## Input contract

- Keep a fixed feature key set per model instance.
- Keep feature scales stable; add preprocessing where needed.
- Avoid silent missing-key defaults in your own wrappers.

## Warmup and thresholds

- Define a warmup phase explicitly.
- Track threshold drift over time; avoid static global thresholds in non-stationary streams.
- Prefer adaptive thresholding (`QuantileThreshold`) for score distributions that move.

## Concept drift

- Pair a detector with drift monitoring from `aberrant.drift`.
- When drift is detected, reset or reinitialize selected models.

## Reliability

- Persist model state regularly (pickle/joblib where supported).
- Add deterministic seeds for reproducibility in experiments.
- Keep optional dependencies explicit (`eval`, `dl`, `faiss` extras).

## Deployment checklist

1. Validate schema and key order before inference.
2. Log score distributions and alert rates.
3. Run canary streams before full rollout.
4. Add regression tests for every production incident.
