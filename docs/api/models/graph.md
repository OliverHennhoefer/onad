# Graph Models

Public objects from `aberrant.model.graph`:

- `ISCONNA`

`ISCONNA` is a bounded-memory conditional detector for dynamic edge streams.

Notes:
- Expects source and destination node identifiers per sample.
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
