# ONAD

ONAD is a Python library for online anomaly detection on streaming data.
Models consume one sample at a time and keep bounded state where possible.

## Why ONAD

- Streaming-first APIs (`learn_one`, `score_one`)
- Multiple detector families (forest, distance, SVM, statistical)
- Dataset streaming utilities for repeatable experiments
- Composable preprocessing and projection transforms

## Install

```bash
pip install onad
```

Optional extras:

- `onad[eval]`: evaluation tools (`scikit-learn`)
- `onad[dl]`: deep-learning model support (`torch`)
- `onad[parquet]`: legacy parquet streamer support (`pyarrow`)
- `onad[docs]`, `onad[dev]`, `onad[benchmark]`, `onad[all]`

## Minimal example

```python
from onad.model.iforest import OnlineIsolationForest

model = OnlineIsolationForest(window_size=512, num_trees=50)

for point in stream_of_feature_dicts:
    model.learn_one(point)
    score = model.score_one(point)
    if score > 0.8:
        print("anomaly", score)
```

## Stable import surface

ONAD intentionally exposes a small public surface:

- `onad.drift`
- `onad.model.iforest`
- `onad.model.distance`
- `onad.model.svm`
- `onad.model.stat`
- `onad.transform.preprocessing`
- `onad.transform.projection`
- `onad.stream.dataset`

See [API Reference](api/index.md) for exact exports.
