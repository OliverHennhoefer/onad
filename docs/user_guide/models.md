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
from aberrant.model.distance import KNN, LocalOutlierFactor
```

`KNN` requires a similarity engine (for example FAISS):

```python
from aberrant.model.distance import KNN
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=25, similarity_engine=engine)
```

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

## Core utility models

```python
from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
```

- `ThresholdModel`: static rule-based boundary detector
- `QuantileThreshold`: adaptive threshold on a streaming score signal
