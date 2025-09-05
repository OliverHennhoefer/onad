# Models Overview

ONAD provides a comprehensive collection of anomaly detection algorithms optimized for streaming data. This guide covers all available models, their strengths, use cases, and configuration options.

## Model Categories

### Forest-based Models {#forest-models}

Tree ensemble methods that detect anomalies by isolating data points in random partitions.

#### OnlineIsolationForest

The flagship model for general-purpose anomaly detection.

**How it works:**
- Builds multiple random trees that partition the feature space
- Anomalies require fewer splits to isolate (shorter path lengths)
- Maintains a sliding window of recent data for concept drift adaptation

**Best for:**
- General-purpose anomaly detection
- High-dimensional data
- Mixed data types (numerical and categorical)
- Real-time applications with moderate memory constraints

**Configuration:**

```python
from onad.model.iforest import OnlineIsolationForest

model = OnlineIsolationForest(
    num_trees=100,  # More trees = better accuracy, higher memory
    window_size=1000,  # Memory window size
    max_leaf_samples=32,  # Tree complexity control
    random_state=42  # Reproducible results
)
```

**Memory usage:** O(num_trees × window_size)

#### MondrianIsolationForest

Advanced forest model with adaptive partitioning based on data distribution.

**How it works:**
- Uses Mondrian processes for data-adaptive tree construction
- Dynamically adjusts partitioning based on observed data patterns
- Better handling of non-uniform feature distributions

**Best for:**
- Data with varying feature densities
- Applications requiring adaptive partitioning
- Scenarios with strong feature interactions

```python
from onad.model.iforest import MondrianIsolationForest

model = MondrianIsolationForest(
    num_trees=50,
    window_size=1000,
    lambda_param=1.0  # Controls partitioning randomness
)
```

### SVM-based Models {#svm-models}

Support Vector Machine approaches that learn decision boundaries for anomaly detection.

#### IncrementalOneClassSVMAdaptiveKernel

Adaptive one-class SVM with evolving kernel parameters.

**How it works:**
- Maintains a one-class SVM decision boundary
- Adapts kernel parameters based on data distribution changes
- Uses sliding window for concept drift handling

**Best for:**
- Complex decision boundaries
- Low to medium dimensional data
- Applications requiring boundary adaptation

```python
from onad.model.svm import IncrementalOneClassSVMAdaptiveKernel

model = IncrementalOneClassSVMAdaptiveKernel(
    gamma='scale',  # Kernel parameter
    nu=0.1,  # Anomaly ratio estimate
    window_size=1000,  # Memory window
    adaptation_rate=0.01  # Kernel adaptation speed
)
```

**Memory usage:** O(window_size)

#### GADGETSVM

Graph-based SVM that leverages structural relationships in data.

**How it works:**
- Combines SVM with graph structure modeling
- Uses connectivity patterns to improve anomaly detection
- Suitable for data with inherent graph structure

**Best for:**
- Network data (social networks, communication graphs)
- Data with structural relationships
- Applications where connectivity matters

```python
from onad.model.svm import GADGETSVM

model = GADGETSVM(
    graph={0: [1], 1: [2], 2: []},  # Graph structure
    svm_params={'nu': 0.1, 'gamma': 'scale'}
)
```

### Distance-based Models {#distance-models}

Methods that detect anomalies based on proximity to normal data points.

#### IncrementalKNN

K-Nearest Neighbors approach with efficient similarity search.

**How it works:**
- Maintains a window of normal data points
- Computes distances to k-nearest neighbors
- Uses FAISS for efficient similarity search

**Best for:**
- Data with clear local structure
- Applications requiring interpretable results
- Medium to high-dimensional data with good distance metrics

```python
from onad.model.distance import IncrementalKNN

model = IncrementalKNN(
    k=10,  # Number of neighbors
    window_size=1000,  # Size of reference window
    distance_metric='euclidean'
)
```

**Memory usage:** O(window_size × dimensions)

### Statistical Models {#statistical-models}

Classical statistical approaches for detecting deviations from normal patterns.

#### Univariate Statistical Models

Single-variable statistical anomaly detection.

**Available models:**
- Moving average with standard deviation bounds
- Exponential smoothing with confidence intervals
- Seasonal decomposition with residual analysis

```python
from onad.model.stat.uni import MovingAverage

model = MovingAverage(
    window_size=100,
    n_std=2.0  # Standard deviation threshold
)
```

#### Multivariate Statistical Models

Multi-variable statistical methods for correlated features.

**MovingCovariance**
Tracks covariance matrix evolution for anomaly detection.

```python
from onad.model.stat.multi import MovingCovariance

model = MovingCovariance(
    window_size=200,
    regularization=0.01  # Covariance matrix regularization
)
```

**MovingMahalanobisDistance**
Uses Mahalanobis distance with evolving statistics.

```python
from onad.model.stat.multi import MovingMahalanobisDistance

model = MovingMahalanobisDistance(
    window_size=500,
    threshold_percentile=95
)
```

**MovingCorrelationCoefficient**
Monitors correlation structure changes.

```python
from onad.model.stat.multi import MovingCorrelationCoefficient

model = MovingCorrelationCoefficient(
    window_size=300,
    correlation_threshold=0.8
)
```

## Model Selection Guide

### By Data Characteristics

| Data Type | Recommended Models | Notes |
|-----------|-------------------|-------|
| **High-dimensional** | OnlineIsolationForest, IncrementalKNN | Forest models handle dimensionality well |
| **Low-dimensional** | SVM models, Statistical models | Better boundary learning in low dimensions |
| **Mixed types** | OnlineIsolationForest | Handles numerical and categorical features |
| **Temporal patterns** | Statistical models | Built-in time series understanding |
| **Graph/Network data** | GADGETSVM | Leverages structural relationships |

### By Performance Requirements

| Priority | Recommended Models | Trade-offs |
|----------|-------------------|------------|
| **Speed** | Statistical models, IncrementalKNN | Faster but may miss complex patterns |
| **Accuracy** | OnlineIsolationForest, SVM models | Higher computation but better detection |
| **Memory** | Statistical models | Constant memory usage |
| **Interpretability** | IncrementalKNN, Statistical models | Clear explanations for anomalies |

### By Use Case

| Use Case | Recommended Approach | Example Models |
|----------|---------------------|----------------|
| **IoT Sensors** | Statistical → Forest | MovingMahalanobis + OnlineIsolationForest |
| **Network Security** | Graph + Distance | GADGETSVM + IncrementalKNN |
| **Financial Fraud** | Forest + SVM | OnlineIsolationForest + AdaptiveSVM |
| **System Monitoring** | Statistical | MovingAverage, MovingCovariance |

## Common Parameters

### Memory Management
```python
# All models support window_size for memory control
model = OnlineIsolationForest(window_size=1000)  # Keep last 1000 points
```

### Performance Tuning
```python
# Forest models
model = OnlineIsolationForest(
    num_trees=100,          # More trees = better accuracy
    max_leaf_samples=32     # Smaller = more sensitive to anomalies
)

# SVM models
model = IncrementalOneClassSVMAdaptiveKernel(
    nu=0.1,                 # Expected anomaly ratio
    gamma='scale'           # Kernel bandwidth
)

# Distance models
model = IncrementalKNN(
    k=10,                   # More neighbors = smoother decision boundary
    distance_metric='euclidean'
)
```

### Reproducibility
```python
# Set random state for deterministic results
model = OnlineIsolationForest(random_state=42)
```

## Model Information and Monitoring

Most models provide introspection capabilities:

```python
# Get model statistics
if hasattr(model, 'get_model_info'):
    info = model.get_model_info()
    print(f"Model state: {info}")

# Check memory usage
if hasattr(model, 'get_memory_usage'):
    memory = model.get_memory_usage()
    print(f"Memory usage: {memory} MB")
```

## Advanced Configuration

### Adaptive Parameters
Some models support parameter adaptation:

```python
# SVM with adaptive kernel
model = IncrementalOneClassSVMAdaptiveKernel(
    adaptation_rate=0.01,   # How fast to adapt
    min_gamma=0.001,        # Minimum kernel bandwidth
    max_gamma=10.0          # Maximum kernel bandwidth
)
```

### Custom Similarity Metrics
```python
# KNN with custom distance function
def custom_distance(x, y):
    # Your custom distance calculation
    return np.linalg.norm(x - y)

model = IncrementalKNN(
    k=5,
    distance_function=custom_distance
)
```

---

!!! tip "Model Combination"
    Consider combining multiple models for robust detection:
    ```python
    # Use multiple models and aggregate scores
    scores = []
    for model in [forest_model, svm_model, stat_model]:
        model.learn_one(data_point)
        scores.append(model.score_one(data_point))
    
    # Combine scores (e.g., average, max, weighted)
    final_score = np.mean(scores)
    ```

!!! warning "Memory Considerations"
    Always set appropriate `window_size` parameters based on your memory constraints:
    - Start with smaller windows (100-1000) for testing
    - Monitor memory usage in production
    - Consider model complexity vs. available resources