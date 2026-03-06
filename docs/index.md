# ABERRANT

ABERRANT is a Python library for online anomaly detection on streaming data.
Models consume one sample at a time and keep bounded state where possible.

## Why ABERRANT

- Streaming-first APIs (`learn_one`, `score_one`)
- Multiple detector families (forest, distance, SVM, statistical)
- Dataset streaming utilities for repeatable experiments
- Composable preprocessing and projection transforms

## Install

```bash
pip install aberrant
```

Optional extras:

- `aberrant[eval]`: evaluation tools (`scikit-learn`)
- `aberrant[dl]`: deep-learning model support (`torch`)
- `aberrant[parquet]`: legacy parquet streamer support (`pyarrow`)
- `aberrant[docs]`, `aberrant[dev]`, `aberrant[benchmark]`, `aberrant[all]`

## Minimal example

```python
from aberrant.model.iforest import OnlineIsolationForest

model = OnlineIsolationForest(window_size=512, num_trees=50)

for point in stream_of_feature_dicts:
    model.learn_one(point)
    score = model.score_one(point)
    if score > 0.8:
        print("anomaly", score)
```

## Stable import surface

ABERRANT intentionally exposes a small public surface:

- `aberrant.drift`
- `aberrant.model.iforest`
- `aberrant.model.distance`
- `aberrant.model.svm`
- `aberrant.model.stat`
- `aberrant.transform.preprocessing`
- `aberrant.transform.projection`
- `aberrant.stream.dataset`

See [API Reference](api/index.md) for exact exports.
