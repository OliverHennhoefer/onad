# Quick Start Guide

This guide will get you up and running with ONAD in just a few minutes. We'll walk through a complete example of detecting anomalies in streaming data.

## Installation

First, install ONAD with evaluation tools:

```bash
pip install onad[eval]
```

## Your First Anomaly Detector

Let's create a simple anomaly detection system using the Online Isolation Forest algorithm:

### 1. Basic Setup

```python
from onad.model.unsupervised.forest import OnlineIsolationForest
import numpy as np

# Create an Online Isolation Forest model
model = OnlineIsolationForest(
    num_trees=50,           # Number of trees in the forest
    window_size=1000,       # Memory window size
    max_leaf_samples=32     # Samples per leaf node
)

print("✅ Model initialized successfully!")
```

### 2. Generate Sample Data

For this example, we'll create synthetic data with some anomalies:

```python
# Generate normal data (2D Gaussian)
np.random.seed(42)
normal_data = []

for i in range(500):
    # Normal points around (0, 0)
    point = {
        'feature_1': np.random.normal(0, 1),
        'feature_2': np.random.normal(0, 1)
    }
    normal_data.append(point)

# Generate some anomalies
anomaly_data = []
for i in range(20):
    # Anomalous points far from normal distribution
    point = {
        'feature_1': np.random.normal(5, 0.5),  # Far from normal center
        'feature_2': np.random.normal(5, 0.5)
    }
    anomaly_data.append(point)

print(f"Generated {len(normal_data)} normal points and {len(anomaly_data)} anomalies")
```

### 3. Train the Model

Train the model on normal data:

```python
# Train on normal data
for point in normal_data:
    model.learn_one(point)

print("🎓 Model training completed!")
```

### 4. Detect Anomalies

Now test the model on both normal and anomalous data:

```python
# Test on normal data
normal_scores = []
for point in normal_data[-50:]:  # Test on last 50 normal points
    score = model.score_one(point)
    normal_scores.append(score)

# Test on anomalies
anomaly_scores = []
for point in anomaly_data:
    score = model.score_one(point)
    anomaly_scores.append(score)

print(f"Normal data scores - Mean: {np.mean(normal_scores):.3f}, Max: {np.max(normal_scores):.3f}")
print(f"Anomaly scores - Mean: {np.mean(anomaly_scores):.3f}, Min: {np.min(anomaly_scores):.3f}")
```

### 5. Set Detection Threshold

Choose a threshold for anomaly detection:

```python
# Use 95th percentile of normal scores as threshold
threshold = np.percentile(normal_scores, 95)
print(f"🎯 Detection threshold: {threshold:.3f}")

# Detect anomalies
detected_anomalies = 0
for score in anomaly_scores:
    if score > threshold:
        detected_anomalies += 1

detection_rate = detected_anomalies / len(anomaly_scores)
print(f"🔍 Detected {detected_anomalies}/{len(anomaly_scores)} anomalies ({detection_rate:.1%} detection rate)")
```

## Complete Example

Here's the complete code in one block:

```python
from onad.model.unsupervised.forest import OnlineIsolationForest
import numpy as np

# Initialize model
model = OnlineIsolationForest(num_trees=50, window_size=1000, max_leaf_samples=32)

# Generate data
np.random.seed(42)
normal_data = [{'x': np.random.normal(0, 1), 'y': np.random.normal(0, 1)} 
               for _ in range(500)]
anomaly_data = [{'x': np.random.normal(5, 0.5), 'y': np.random.normal(5, 0.5)} 
                for _ in range(20)]

# Train model
for point in normal_data:
    model.learn_one(point)

# Evaluate
normal_scores = [model.score_one(point) for point in normal_data[-50:]]
anomaly_scores = [model.score_one(point) for point in anomaly_data]

# Set threshold and detect
threshold = np.percentile(normal_scores, 95)
detected = sum(1 for score in anomaly_scores if score > threshold)
print(f"Detected {detected}/{len(anomaly_data)} anomalies")
```

## Working with Real Data

### Using Built-in Datasets

ONAD comes with sample datasets for testing:

```python
from onad.stream import ParquetStreamer, Dataset

# Stream from built-in fraud detection dataset
with ParquetStreamer(Dataset.FRAUD) as streamer:
    model = OnlineIsolationForest()
    
    for features, label in streamer:
        # Learn from the data point
        model.learn_one(features)
        
        # Get anomaly score
        score = model.score_one(features)
        
        # Check if it's actually an anomaly (using label)
        is_anomaly = label == 1
        predicted_anomaly = score > 0.5
        
        if is_anomaly and predicted_anomaly:
            print(f"✅ Correctly detected anomaly: {score:.3f}")
        elif is_anomaly and not predicted_anomaly:
            print(f"❌ Missed anomaly: {score:.3f}")
```

### Custom Data Sources

You can easily adapt ONAD to your own data:

```python
# Your custom data source
def my_data_stream():
    """Generator that yields feature dictionaries"""
    for i in range(1000):
        yield {
            'cpu_usage': np.random.uniform(0, 100),
            'memory_usage': np.random.uniform(0, 100),
            'network_traffic': np.random.exponential(10)
        }

# Process your custom stream
model = OnlineIsolationForest()
anomaly_count = 0

for features in my_data_stream():
    model.learn_one(features)
    score = model.score_one(features)
    
    if score > 0.7:  # Your chosen threshold
        anomaly_count += 1
        print(f"Anomaly detected: {features}")

print(f"Total anomalies detected: {anomaly_count}")
```

## Building Pipelines

Combine multiple components for more sophisticated processing:

```python
from onad.transform.scale import StandardScaler
from onad.model.unsupervised.forest import OnlineIsolationForest

# Create a processing pipeline
scaler = StandardScaler()
model = OnlineIsolationForest()

# Process data through the pipeline
def process_point(raw_data):
    # 1. Learn and transform features
    scaler.learn_one(raw_data)
    normalized_data = scaler.transform_one(raw_data)
    
    # 2. Learn and score with the model
    model.learn_one(normalized_data)
    anomaly_score = model.score_one(normalized_data)
    
    return anomaly_score

# Use the pipeline
for point in my_data_stream():
    score = process_point(point)
    if score > threshold:
        print(f"Pipeline detected anomaly: {score:.3f}")
```

## Next Steps

Now that you've created your first anomaly detector, explore more advanced features:

### Learn More
- **[User Guide](user_guide/index.md)**: Comprehensive documentation
- **[Model Comparison](user_guide/models.md)**: Choose the right algorithm
- **[Pipeline Guide](user_guide/pipelines.md)**: Build complex processing workflows
- **[Best Practices](user_guide/best_practices.md)**: Tips for production deployment

### Try Different Models
- **[SVM Models](examples/svm_models.md)**: Support Vector Machine approaches
- **[Statistical Models](examples/statistical_models.md)**: Classical statistical methods
- **[Distance-based Models](api/models/unsupervised.md)**: KNN and similarity-based detection

### Real-world Examples
- **[IoT Monitoring](examples/basic_usage.md)**: Sensor anomaly detection
- **[Network Security](examples/custom_pipelines.md)**: Traffic analysis
- **[Financial Fraud](examples/forest_models.md)**: Transaction monitoring

---

!!! tip "Performance Tips"
    - Start with small `window_size` values (100-1000) for faster processing
    - Use `num_trees=50-100` for good accuracy/speed balance
    - Enable logging to monitor model performance in production