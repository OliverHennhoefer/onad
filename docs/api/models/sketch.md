# Sketch Models

Public objects from `aberrant.model.sketch`:

- `MStream`
- `RSHash`

`MStream` is a bounded-memory streaming detector based on fixed-size sketches.

`RSHash` is a bounded-memory randomized subspace hashing detector.

Notes:
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
