# Transformers

ONAD supports streaming transformers that compose with models using `|`.

## Preprocessing

```python
from onad.transform.preprocessing import MinMaxScaler, StandardScaler
```

## Projection

```python
from onad.transform.projection import IncrementalPCA, RandomProjection
```

## Pipeline example

```python
from onad.model.iforest import OnlineIsolationForest
from onad.transform.preprocessing import StandardScaler

pipeline = StandardScaler() | OnlineIsolationForest(window_size=512)

for x in stream_of_feature_dicts:
    pipeline.learn_one(x)
    score = pipeline.score_one(x)
```

## Notes

- Keep feature keys consistent across the stream.
- `HalfSpaceTrees` works best after min-max scaling to `[0, 1]`.
