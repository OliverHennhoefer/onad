# User Guide

Welcome to the ONAD User Guide! This comprehensive guide will help you master online anomaly detection with ONAD, from basic concepts to advanced deployment strategies.

## What is Online Anomaly Detection?

Online anomaly detection is the process of identifying unusual patterns or outliers in streaming data as it arrives, without requiring the entire dataset to be available upfront. This approach is essential for:

- **Real-time systems** that need immediate responses to anomalies
- **Large-scale data** where storing everything is impractical
- **Evolving patterns** where the definition of "normal" changes over time
- **Resource-constrained environments** with limited memory and compute

## ONAD Philosophy

ONAD is built around several core principles:

### Streaming-First Design
Every component in ONAD is designed to process data one point at a time, maintaining constant memory usage regardless of stream length.

### Composable Architecture
Models, transformers, and utilities can be combined in flexible pipelines to solve complex problems.

### Production-Ready
Built-in logging, error handling, and memory management make ONAD suitable for production deployments.

### Type Safety
Comprehensive type hints ensure code reliability and better IDE support.

## Getting Started

### 1. Choose Your Model
ONAD provides several categories of anomaly detection models:

- **[Forest-based Models](models.md#forest-models)**: Tree ensemble methods like Isolation Forest
- **[SVM-based Models](models.md#svm-models)**: Support Vector Machine approaches
- **[Statistical Models](models.md#statistical-models)**: Classical statistical methods
- **[Distance-based Models](models.md#distance-models)**: K-NN and similarity search

### 2. Prepare Your Data
Use ONAD's data transformation tools:

- **[Scaling](transformers.md#scaling)**: Normalize features with StandardScaler or MinMaxScaler
- **[Dimensionality Reduction](transformers.md#pca)**: Reduce features with Incremental PCA
- **[Stream Processing](streaming.md)**: Efficiently load and process data streams

### 3. Build Your Pipeline
Combine components for sophisticated processing:

- **[Pipeline Construction](pipelines.md)**: Learn to chain transformers and models
- **[Memory Management](best_practices.md#memory)**: Configure memory limits and monitoring
- **[Error Handling](best_practices.md#error-handling)**: Implement robust error recovery

## Common Workflows

### Basic Anomaly Detection

```python
from onad.model.forest import OnlineIsolationForest

# Initialize model
model = OnlineIsolationForest()

# Process streaming data
for data_point in stream:
    model.learn_one(data_point)
    score = model.score_one(data_point)
    if score > threshold:
        handle_anomaly(data_point, score)
```

### Data Preprocessing Pipeline

```python
from onad.transform.preprocess.scaler import StandardScaler
from onad.model.svm import IncrementalOneClassSVMAdaptiveKernel

# Create pipeline components
scaler = StandardScaler()
detector = IncrementalOneClassSVMAdaptiveKernel()


# Process data through pipeline
def detect_anomaly(raw_data):
    scaler.learn_one(raw_data)
    normalized_data = scaler.transform_one(raw_data)
    detector.learn_one(normalized_data)
    return detector.score_one(normalized_data)
```

### Batch Processing with Streaming Interface
```python
from onad.stream import ParquetStreamer, Dataset

with ParquetStreamer(Dataset.FRAUD) as streamer:
    for features, label in streamer:
        # Process each data point
        model.learn_one(features)
        score = model.score_one(features)
```

## Topics Covered

### Core Concepts
- **[Models Overview](models.md)**: Complete guide to all anomaly detection algorithms
- **[Data Transformers](transformers.md)**: Preprocessing and feature engineering
- **[Stream Processing](streaming.md)**: Efficient data loading and processing

### Advanced Topics
- **[Pipeline Construction](pipelines.md)**: Building complex processing workflows
- **[Model Evaluation](evaluation.md)**: Testing and validating your models
- **[Best Practices](best_practices.md)**: Production deployment guidelines

## Quick Reference

### Essential Imports

```python
# Models
from onad.model.forest import OnlineIsolationForest
from onad.model.svm import IncrementalOneClassSVMAdaptiveKernel
from onad.model.stat.multi import MovingMahalanobisDistance

# Transformers
from onad.transform.preprocess.scaler import StandardScaler, MinMaxScaler
from onad.transform.project.incremental_pca import IncrementalPCA

# Streaming
from onad.stream import ParquetStreamer, Dataset
```

### Common Parameters
```python
# Memory management
model = OnlineIsolationForest(window_size=1000)

# Performance tuning
model = OnlineIsolationForest(num_trees=100, max_leaf_samples=32)

# Data preprocess
scaler = StandardScaler()
pca = IncrementalPCA(n_components=10)
```

### Typical Workflow
1. **Initialize** model and transformers
2. **Learn** from each data point with `learn_one()`
3. **Score** data points with `score_one()`
4. **Transform** data with `transform_one()` (if using transformers)
5. **Monitor** model state with logging and metrics

---

!!! tip "Next Steps"
    - Start with the [Models Overview](models.md) to understand available algorithms
    - Check out [Examples](../examples/index.md) for real-world use cases
    - Review [Best Practices](best_practices.md) for production deployment tips