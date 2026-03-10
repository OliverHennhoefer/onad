# Sketch Models

Public objects from `aberrant.model.sketch`:

- `MStream`

`MStream` is a bounded-memory streaming detector based on fixed-size sketches.

Notes:
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
