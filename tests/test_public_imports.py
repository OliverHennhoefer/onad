"""Public API import contract tests."""

from onad import __version__
from onad.base import BaseModel, BaseTransformer, Pipeline
from onad.drift import ADWIN, KSWIN, PageHinkley
from onad.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
from onad.model.distance import KNN, LocalOutlierFactor
from onad.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianForest,
    OnlineIsolationForest,
    StreamRandomHistogramForest,
)
from onad.model.stat import MovingAverage, MovingCovariance
from onad.model.svm import GADGETSVM, IncrementalOneClassSVMAdaptiveKernel
from onad.stream import Dataset, load
from onad.stream.dataset import BatchStreamer, DatasetStreamer
from onad.transform.preprocessing import MinMaxScaler, StandardScaler
from onad.transform.projection import IncrementalPCA, RandomProjection

try:
    from onad.model.deep import Autoencoder
except ImportError:
    Autoencoder = None  # type: ignore[assignment]


def test_public_imports_smoke() -> None:
    assert isinstance(__version__, str)
    assert BaseModel is not None
    assert BaseTransformer is not None
    assert Pipeline is not None
    assert ADWIN is not None
    assert KSWIN is not None
    assert PageHinkley is not None
    assert NullModel is not None
    assert RandomModel is not None
    assert ThresholdModel is not None
    assert QuantileThreshold is not None
    # Deep model imports are optional and depend on torch availability.
    if Autoencoder is not None:
        assert Autoencoder.__name__ == "Autoencoder"
    assert KNN is not None
    assert LocalOutlierFactor is not None
    assert ASDIsolationForest is not None
    assert HalfSpaceTrees is not None
    assert MondrianForest is not None
    assert OnlineIsolationForest is not None
    assert StreamRandomHistogramForest is not None
    assert IncrementalOneClassSVMAdaptiveKernel is not None
    assert GADGETSVM is not None
    assert MovingAverage is not None
    assert MovingCovariance is not None
    assert MinMaxScaler is not None
    assert StandardScaler is not None
    assert IncrementalPCA is not None
    assert RandomProjection is not None
    assert Dataset is not None
    assert load is not None
    assert BatchStreamer is not None
    assert DatasetStreamer is not None
