# Models

ONAD ships multiple online detector families.

## Isolation forest family

Imports:

```python
from onad.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianForest,
    OnlineIsolationForest,
    StreamRandomHistogramForest,
    XStream,
)
```

Use these for general-purpose unsupervised streaming anomaly detection.

## Distance family

Imports:

```python
from onad.model.distance import KNN, LocalOutlierFactor
```

`KNN` requires a similarity engine (for example FAISS):

```python
from onad.model.distance import KNN
from onad.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=25, similarity_engine=engine)
```

## SVM family

Imports:

```python
from onad.model.svm import GADGETSVM, IncrementalOneClassSVMAdaptiveKernel
```

Use when margin-based decision boundaries are preferred.

## Statistical family

Imports:

```python
from onad.model.stat import (
    MovingAverage,
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)
```

Use for compact, interpretable change detectors.

## Core utility models

```python
from onad.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
```

- `ThresholdModel`: static rule-based boundary detector
- `QuantileThreshold`: adaptive threshold on a streaming score signal
