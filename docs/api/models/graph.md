# Graph Models

Public objects from `aberrant.model.graph`:

- `AnoEdgeL`
- `ISCONNA`
- `MIDAS`
- `StreamSpot`

`AnoEdgeL` is a bounded-memory local-density detector for dynamic edge streams.

`ISCONNA` is a bounded-memory conditional detector for dynamic edge streams.

`MIDAS` is a bounded-memory microcluster detector for dynamic edge streams.

`StreamSpot` is a bounded-memory structural detector for per-graph edge streams.

Notes:
- Expects source and destination node identifiers per sample.
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
