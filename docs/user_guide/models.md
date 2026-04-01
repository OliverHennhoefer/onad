# Models

ABERRANT ships multiple online detector families.

## Isolation forest family

Imports:

```python
from aberrant.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianForest,
    OnlineIsolationForest,
    RandomCutForest,
    StreamRandomHistogramForest,
    XStream,
)
```

Use these for general-purpose unsupervised streaming anomaly detection.

`MondrianForest` uses `lambda_` as the Mondrian lifetime budget:
- smaller `lambda_` yields coarser partitions
- larger `lambda_` yields finer partitions

For evaluation on a stream, prefer scoring before learning each sample to avoid
training on the point being evaluated.

## Distance family

Imports:

```python
from aberrant.model.distance import KNN, LocalOutlierFactor, NETS, SDOStream, STARE
```

`KNN` requires a similarity engine (for example FAISS):

```python
from aberrant.model.distance import KNN
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=25, similarity_engine=engine)
```

`LocalOutlierFactor` is a bounded sliding-window local-density detector:

```python
from aberrant.model.distance import LocalOutlierFactor

model = LocalOutlierFactor(k=10, window_size=1000, distance="euclidean")
```

- Returns a continuous LOF score, where values above `1` are more outlier-like.
- Uses bounded-memory window state (`window_size`).

`NETS` is a bounded streaming detector using cell-level net-effect pruning:

```python
from aberrant.model.distance import NETS

model = NETS(
    k=30,
    radius=1.5,
    window_size=2048,
    slide_size=128,
    subspace_dim=3,
    seed=42,
)
```

- Returns bounded scores in `[0, 1]`.
- Supports optional explicit time handling through `time_key`.
- Uses bounded-memory full/subspace cell state over a sliding window.

`SDOStream` is a bounded-memory observer-based online detector:

```python
from aberrant.model.distance import SDOStream

model = SDOStream(k=128, T=256.0, qv=0.3, x_neighbors=8, seed=42)
```

- Returns a continuous non-negative anomaly score (distance-based).
- Supports optional explicit time handling through `time_key`.
- Uses fixed-size observer state (`k`) with exponential fading (`T`).

`STARE` is a bounded sliding-window local outlier detector:

```python
from aberrant.model.distance import STARE

model = STARE(
    k=40,
    radius=1.5,
    window_size=2048,
    slide_size=128,
    skip_threshold=0.1,
)
```

- Returns bounded scores in `[0, 1]`.
- Supports optional explicit time handling through `time_key`.
- Uses bounded-memory sliding window state (`window_size`).

## Sketch family

Imports:

```python
from aberrant.model.sketch import MStream, RSHash
```

Use `MStream` and `RSHash` for bounded-memory sketch-based streaming detection.

- Supports `time_key` for explicit bucketed time updates.
- If `time_key=None`, it uses arrival order as the implicit time axis.
- Returns a continuous non-negative anomaly score.

## Graph family

Imports:

```python
from aberrant.model.graph import ISCONNA, MIDAS
```

Use `ISCONNA` and `MIDAS` for dynamic edge streams where each sample encodes
source and destination IDs (plus optional timestamp).

- Uses bounded-memory count-min sketches.
- Supports optional explicit timestamp handling via `time_key`.
- Returns a continuous non-negative anomaly score.

## SVM family

Imports:

```python
from aberrant.model.svm import GADGETSVM, IncrementalOneClassSVMAdaptiveKernel
```

Use when margin-based decision boundaries are preferred.

## Statistical family

Imports:

```python
from aberrant.model.stat import (
    MovingAverage,
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)
```

Use for compact, interpretable change detectors.

## Deep family

Imports:

```python
from aberrant.model.deep import Autoencoder, KitNET
```

- `Autoencoder` depends on `torch` (`aberrant[dl]`).
- `KitNET` is an online ensemble of lightweight autoencoders with explicit
  warm-up phases (`feature_map_grace`, `ad_grace`).

## Core utility models

```python
from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
```

- `ThresholdModel`: static rule-based boundary detector
- `QuantileThreshold`: adaptive threshold on a streaming score signal
