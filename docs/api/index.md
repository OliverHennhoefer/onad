# API Reference

This section provides comprehensive documentation for all ONAD classes, functions, and modules. The API is organized into logical groups based on functionality.

## Package Structure

ONAD follows a modular architecture with clear separation of concerns:

```
onad/
├── base/           # Abstract base classes and interfaces
├── model/          # Anomaly detection algorithms
│   ├── unsupervised/   # Unsupervised learning models
│   ├── supervised/     # Supervised learning models
│   └── statistics/     # Statistical methods
├── transform/      # Data preprocessing and transformation
├── stream/         # Data streaming and I/O utilities
└── utils/          # Helper functions and utilities
```

## Core Concepts

### Base Classes

All ONAD components inherit from well-defined base classes that establish consistent interfaces:

- **[BaseModel](base.md#onad.base.model.BaseModel)**: Foundation for all anomaly detection models
- **[BaseTransformer](base.md#onad.base.transformer.BaseTransformer)**: Base for data preprocessing components
- **[BaseSimilaritySearchEngine](base.md#onad.base.similarity.BaseSimilaritySearchEngine)**: Interface for similarity search implementations

### Model Categories

#### Unsupervised Models {#unsupervised}

Models that detect anomalies without labeled training data:

- **Forest-based**: Tree ensemble methods for anomaly detection
    - [OnlineIsolationForest](models/unsupervised.md#onad.model.unsupervised.forest.OnlineIsolationForest)
    - [MondrianIsolationForest](models/unsupervised.md#onad.model.unsupervised.forest.MondrianIsolationForest)

- **SVM-based**: Support Vector Machine approaches
    - [IncrementalOneClassSVMAdaptiveKernel](models/unsupervised.md#onad.model.unsupervised.svm.IncrementalOneClassSVMAdaptiveKernel)
    - [GADGETSVM](models/unsupervised.md#onad.model.unsupervised.svm.GADGETSVM)

- **Distance-based**: Proximity-based detection methods
    - [IncrementalKNN](models/unsupervised.md#onad.model.unsupervised.distance.IncrementalKNN)

#### Statistical Models {#statistics}

Classical statistical approaches for anomaly detection:

- **Univariate**: Single-variable statistical methods
- **Multivariate**: Multi-variable statistical methods
    - [MovingCovariance](models/statistics.md#onad.model.statistics.multivariant.MovingCovariance)
    - [MovingCorrelationCoefficient](models/statistics.md#onad.model.statistics.multivariant.MovingCorrelationCoefficient)
    - [MovingMahalanobisDistance](models/statistics.md#onad.model.statistics.multivariant.MovingMahalanobisDistance)

### Data Processing

#### Transformers {#transform}

Data preprocessing and feature engineering components:

- **[StandardScaler](transform.md#onad.transform.scale.StandardScaler)**: Z-score normalization
- **[MinMaxScaler](transform.md#onad.transform.scale.MinMaxScaler)**: Min-max scaling
- **[IncrementalPCA](transform.md#onad.transform.pca.IncrementalPCA)**: Online Principal Component Analysis

#### Streaming {#stream}

Efficient data streaming and I/O utilities:

- **[ParquetStreamer](stream.md#onad.stream.streamer.ParquetStreamer)**: Stream data from Parquet files
- **[Dataset](stream.md#onad.stream.streamer.Dataset)**: Built-in datasets for testing

### Utilities {#utils}

Helper functions and supporting components:

- **[Similarity Search](utils.md#similarity)**: Fast similarity search engines
- **[Architecture Utils](utils.md#architecture)**: Deep learning utilities

## Usage Patterns

### Basic Model Usage

All models follow a consistent interface:

```python
from onad.model.iforest import OnlineIsolationForest

# Initialize model
model = OnlineIsolationForest(window_size=1000)

# Process streaming data
for data_point in data_stream:
    # Learn from the data point
    model.learn_one(data_point)

    # Get anomaly score
    score = model.score_one(data_point)

    # Check if anomalous
    if score > threshold:
        handle_anomaly(data_point, score)
```

### Pipeline Construction

Combine transformers and models:

```python
from onad.transform.preprocessing.scaler import StandardScaler
from onad.model.svm import IncrementalOneClassSVMAdaptiveKernel

# Create pipeline components
scaler = StandardScaler()
detector = IncrementalOneClassSVMAdaptiveKernel()


# Process data through pipeline
def detect_anomaly(raw_data):
    # Transform data
    scaler.learn_one(raw_data)
    normalized_data = scaler.transform_one(raw_data)

    # Detect anomaly
    detector.learn_one(normalized_data)
    return detector.score_one(normalized_data)
```

## Memory Management

ONAD models are designed for streaming scenarios with memory constraints:

```python
# Configure memory limits
model = OnlineIsolationForest(
    window_size=1000,      # Limit memory window
    max_leaf_samples=32    # Control tree complexity
)

# Monitor memory usage
info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
print(f"Model state: {info}")
```

## Type Safety

All ONAD components include comprehensive type hints:

```python
from typing import Dict
from onad.model.iforest import OnlineIsolationForest

model: OnlineIsolationForest = OnlineIsolationForest()
features: Dict[str, float] = {"temperature": 23.5, "humidity": 0.65}
score: float = model.score_one(features)
```

## Error Handling

ONAD provides robust error handling with informative messages:

```python
try:
    model.learn_one(invalid_data)
except ValueError as e:
    print(f"Data validation error: {e}")
except RuntimeError as e:
    print(f"Model state error: {e}")
```

## Quick Reference

### Most Common Classes

| Class | Purpose | Location |
|-------|---------|----------|
| `OnlineIsolationForest` | Tree-based anomaly detection | `onad.model.unsupervised.forest` |
| `IncrementalOneClassSVMAdaptiveKernel` | SVM-based detection | `onad.model.unsupervised.svm` |
| `StandardScaler` | Data normalization | `onad.transform.scale` |
| `ParquetStreamer` | Data streaming | `onad.stream.streamer` |

### Key Methods

| Method | Purpose | Available On |
|--------|---------|--------------|
| `learn_one(x)` | Learn from one data point | All models and transformers |
| `score_one(x)` | Get anomaly score | All models |
| `transform_one(x)` | Transform one data point | All transformers |
| `get_model_info()` | Get model statistics | Selected models |

---

!!! info "API Stability"
    ONAD follows semantic versioning. The public API documented here is considered stable within major versions. Internal APIs (prefixed with `_`) may change without notice.