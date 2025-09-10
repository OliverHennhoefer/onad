# Data Transformers

Data preprocessing is crucial for effective anomaly detection. ONAD provides streaming-optimized transformers that process data incrementally while maintaining constant memory usage.

## Scaling Transformers {#scaling}

Feature scaling ensures all features contribute equally to distance-based algorithms and prevents features with larger scales from dominating.

### StandardScaler

Z-score normalization that transforms features to have zero mean and unit variance.

**How it works:**
- Maintains running statistics (mean and standard deviation)
- Transforms features: `(x - mean) / std`
- Handles near-zero variance with configurable tolerance

**When to use:**
- Features have different scales (e.g., temperature in Celsius vs. pressure in Pascals)
- Using distance-based algorithms (KNN, SVM)
- Features are approximately normally distributed

```python
from onad.transform.preprocessing.scaler import StandardScaler

# Initialize scaler
scaler = StandardScaler(tolerance=1e-8)

# Process streaming data
for data_point in stream:
    # Learn statistics from data point
    scaler.learn_one(data_point)

    # Transform the data point
    normalized = scaler.transform_one(data_point)

    # Use normalized data for anomaly detection
    score = model.score_one(normalized)
```

**Configuration options:**
```python
scaler = StandardScaler(
    tolerance=1e-8,         # Minimum std dev to avoid division by zero
    with_mean=True,         # Center data (subtract mean)
    with_std=True           # Scale data (divide by std dev)
)
```

**Memory usage:** O(number of features)

### MinMaxScaler

Min-max normalization that scales features to a fixed range [0, 1] or custom range.

**How it works:**
- Tracks running minimum and maximum values
- Transforms features: `(x - min) / (max - min)`
- Optionally scales to custom range [a, b]

**When to use:**
- Need bounded feature ranges
- Preserving zero values is important
- Features have known or stable min/max bounds

```python
from onad.transform.preprocessing.scaler import MinMaxScaler

# Scale to [0, 1] range
scaler = MinMaxScaler()

# Scale to custom range
scaler = MinMaxScaler(feature_range=(-1, 1))

# Process data
for data_point in stream:
    scaler.learn_one(data_point)
    scaled = scaler.transform_one(data_point)
```

**Configuration options:**
```python
scaler = MinMaxScaler(
    feature_range=(0, 1),   # Output range
    clip=False              # Clip values outside observed range
)
```

## Dimensionality Reduction {#pca}

### IncrementalPCA

Online Principal Component Analysis for dimensionality reduction and feature extraction.

**How it works:**
- Incrementally updates eigenvectors and eigenvalues
- Projects data onto principal components
- Maintains specified number of components

**When to use:**
- High-dimensional data (hundreds of features)
- Features are correlated
- Need to reduce computational complexity
- Want to remove noise from data

```python
from onad.transform.projection.incremental_pca import IncrementalPCA

# Initialize PCA transformer
pca = IncrementalPCA(
    n_components=10,  # Keep top 10 components
    forgetting_factor=0.99  # Adaptation rate for concept drift
)

# Process streaming data
for data_point in stream:
    # Learn from data point
    pca.learn_one(data_point)

    # Transform to lower dimension
    reduced = pca.transform_one(data_point)

    # Use reduced data for anomaly detection
    score = model.score_one(reduced)
```

**Configuration options:**
```python
pca = IncrementalPCA(
    n_components=10,        # Number of components to keep
    forgetting_factor=0.99, # How fast to adapt (0.9-0.999)
    center_data=True,       # Center data before PCA
    whiten=False           # Normalize components to unit variance
)
```

**Accessing component information:**
```python
# Get explained variance ratio
if hasattr(pca, 'explained_variance_ratio_'):
    variance_explained = pca.explained_variance_ratio_
    print(f"Variance explained by components: {variance_explained}")

# Get principal components
if hasattr(pca, 'components_'):
    components = pca.components_
    print(f"Principal components shape: {components.shape}")
```

## Pipeline Construction

Transformers can be chained together for complex preprocessing:

### Basic Pipeline

```python
from onad.transform.preprocessing.scaler import StandardScaler
from onad.transform.projection.incremental_pca import IncrementalPCA

# Create pipeline components
scaler = StandardScaler()
pca = IncrementalPCA(n_components=5)


# Process data through pipeline
def preprocess_point(raw_data):
    # Step 1: Scale features
    scaler.learn_one(raw_data)
    scaled = scaler.transform_one(raw_data)

    # Step 2: Reduce dimensions
    pca.learn_one(scaled)
    reduced = pca.transform_one(scaled)

    return reduced


# Use in anomaly detection
for data_point in stream:
    processed = preprocess_point(data_point)
    model.learn_one(processed)
    score = model.score_one(processed)
```

### Advanced Pipeline with Error Handling

```python
from typing import Dict, Any
import logging

class PreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=10)
        self.logger = logging.getLogger(__name__)
    
    def process_point(self, data_point: Dict[str, float]) -> Dict[str, float]:
        try:
            # Scaling step
            self.scaler.learn_one(data_point)
            scaled = self.scaler.transform_one(data_point)
            
            # Dimensionality reduction step
            self.pca.learn_one(scaled)
            reduced = self.pca.transform_one(scaled)
            
            return reduced
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            'scaler_stats': getattr(self.scaler, 'get_stats', lambda: {})(),
            'pca_components': self.pca.n_components,
            'pca_variance_explained': getattr(self.pca, 'explained_variance_ratio_', None)
        }

# Usage
pipeline = PreprocessingPipeline()
for data_point in stream:
    processed = pipeline.process_point(data_point)
    score = model.score_one(processed)
```

## Best Practices

### Initialization and Warmup

```python
# Allow transformers to warm up with initial data
warmup_data = next_n_points(stream, 100)

for point in warmup_data:
    scaler.learn_one(point)
    pca.learn_one(point)

# Now start anomaly detection
for point in stream:
    scaled = scaler.transform_one(point)
    reduced = pca.transform_one(scaled)
    score = model.score_one(reduced)
```

### Monitoring Transformer State

```python
# Monitor scaler statistics
if hasattr(scaler, 'mean_') and hasattr(scaler, 'var_'):
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature variances: {scaler.var_}")

# Monitor PCA state
if hasattr(pca, 'explained_variance_ratio_'):
    total_variance = sum(pca.explained_variance_ratio_)
    print(f"Total variance explained: {total_variance:.2%}")
```

### Handling Missing Values

```python
import numpy as np

def safe_transform(transformer, data_point):
    # Handle missing values before transformation
    cleaned_data = {}
    for key, value in data_point.items():
        if np.isnan(value) or np.isinf(value):
            # Use previous value, zero, or skip feature
            cleaned_data[key] = 0.0  # or use interpolation
        else:
            cleaned_data[key] = value
    
    return transformer.transform_one(cleaned_data)
```

### Performance Optimization

```python
# Batch learning for better performance
batch_size = 50
batch = []

for data_point in stream:
    batch.append(data_point)
    
    if len(batch) >= batch_size:
        # Learn from batch
        for point in batch:
            scaler.learn_one(point)
            pca.learn_one(point)
        
        # Process batch
        processed_batch = []
        for point in batch:
            scaled = scaler.transform_one(point)
            reduced = pca.transform_one(scaled)
            processed_batch.append(reduced)
        
        # Anomaly detection on batch
        for point in processed_batch:
            score = model.score_one(point)
        
        batch = []
```

## Transformer Comparison

| Transformer | Use Case | Memory Usage | Computational Cost | Output Range |
|-------------|----------|--------------|-------------------|--------------|
| **StandardScaler** | Different feature scales | O(features) | Low | Unbounded, ~N(0,1) |
| **MinMaxScaler** | Need bounded ranges | O(features) | Low | [0,1] or custom |
| **IncrementalPCA** | High dimensions | O(features × components) | Medium | Unbounded |

## Common Issues and Solutions

### Numerical Instability

```python
# Use tolerance in StandardScaler
scaler = StandardScaler(tolerance=1e-6)

# Regularize PCA for better stability
pca = IncrementalPCA(n_components=10, regularization=1e-4)
```

### Concept Drift Adaptation

```python
# Use forgetting factor in PCA
pca = IncrementalPCA(
    n_components=10,
    forgetting_factor=0.95  # Faster adaptation to changes
)

# Reset scalers periodically
if data_point_count % 10000 == 0:
    scaler = StandardScaler()  # Fresh start
```

### Feature Selection

```python
# Select features before transformation
important_features = ['temperature', 'pressure', 'vibration']

def select_features(data_point, feature_names):
    return {k: v for k, v in data_point.items() if k in feature_names}

# Use in pipeline
for data_point in stream:
    selected = select_features(data_point, important_features)
    scaled = scaler.transform_one(selected)
    score = model.score_one(scaled)
```

---

!!! tip "Transformer Order"
    Apply transformers in this typical order:
    1. **Feature selection** (remove irrelevant features)
    2. **Missing value handling** (imputation or removal)
    3. **Scaling** (StandardScaler or MinMaxScaler)
    4. **Dimensionality reduction** (IncrementalPCA)
    5. **Anomaly detection model**

!!! warning "Data Leakage"
    Always learn transformer parameters from the same data point before transforming:
    ```python
    # Correct: learn then transform
    scaler.learn_one(data_point)
    transformed = scaler.transform_one(data_point)
    
    # Incorrect: transform without learning
    transformed = scaler.transform_one(data_point)  # Uses stale statistics
    ```