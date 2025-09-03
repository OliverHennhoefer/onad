# Examples

Welcome to the ONAD examples! This section provides practical, real-world examples of using ONAD for various anomaly detection scenarios. Each example includes complete code, explanations, and best practices.

## Example Categories

### Basic Usage
Perfect for getting started with ONAD:

- **[Basic Anomaly Detection](basic_usage.md)**: Simple streaming anomaly detection
- **[Data Preprocessing](basic_usage.md#preprocessing)**: Feature scaling and transformation
- **[Model Comparison](basic_usage.md#comparison)**: Compare different algorithms

### Model-Specific Examples
Deep dives into specific anomaly detection algorithms:

- **[Forest-based Models](forest_models.md)**: Isolation Forest and Mondrian Forest examples
- **[SVM-based Models](svm_models.md)**: Support Vector Machine approaches
- **[Statistical Models](statistical_models.md)**: Classical statistical methods

### Advanced Pipelines
Complex scenarios and production-ready examples:

- **[Custom Pipelines](custom_pipelines.md)**: Building sophisticated detection systems
- **[Multi-model Ensembles](custom_pipelines.md#ensembles)**: Combining multiple algorithms
- **[Real-time Processing](custom_pipelines.md#real-time)**: High-throughput streaming

## Quick Start Examples

### 30-Second Quick Start

```python
from onad.model.unsupervised.forest import OnlineIsolationForest
from onad.stream import ParquetStreamer, Dataset

# Initialize model
model = OnlineIsolationForest()

# Process built-in dataset
with ParquetStreamer(Dataset.FRAUD) as streamer:
    for features, label in streamer:
        model.learn_one(features)
        score = model.score_one(features)
        
        if score > 0.8:  # High anomaly threshold
            print(f"Anomaly detected! Score: {score:.3f}")
```

### 2-Minute Data Pipeline

```python
from onad.transform.scale import StandardScaler
from onad.model.unsupervised.forest import OnlineIsolationForest

# Create preprocessing pipeline
scaler = StandardScaler()
detector = OnlineIsolationForest(num_trees=100)

# Process your data
def detect_anomalies(data_stream, threshold=0.7):
    anomalies = []
    
    for data_point in data_stream:
        # Preprocess
        scaler.learn_one(data_point)
        scaled_data = scaler.transform_one(data_point)
        
        # Detect
        detector.learn_one(scaled_data)
        score = detector.score_one(scaled_data)
        
        if score > threshold:
            anomalies.append((data_point, score))
    
    return anomalies

# Your data stream (replace with your data source)
data_stream = [
    {'temperature': 25.5, 'pressure': 1013.2, 'humidity': 60.0},
    {'temperature': 80.0, 'pressure': 950.0, 'humidity': 90.0},  # Anomaly
    # ... more data points
]

anomalies = detect_anomalies(data_stream)
print(f"Found {len(anomalies)} anomalies")
```

### 5-Minute Production Setup

```python
import logging
from onad.transform.scale import StandardScaler
from onad.transform.pca import IncrementalPCA
from onad.model.unsupervised.forest import OnlineIsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAnomalyDetector:
    def __init__(self):
        # Preprocessing pipeline
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=10)
        
        # Detection model
        self.detector = OnlineIsolationForest(
            num_trees=100,
            window_size=2000,
            max_leaf_samples=32
        )
        
        # Monitoring
        self.processed_count = 0
        self.anomaly_count = 0
    
    def process_data_point(self, data_point, threshold=0.8):
        """Process single data point"""
        try:
            # Preprocessing
            self.scaler.learn_one(data_point)
            scaled = self.scaler.transform_one(data_point)
            
            self.pca.learn_one(scaled)
            reduced = self.pca.transform_one(scaled)
            
            # Anomaly detection
            self.detector.learn_one(reduced)
            score = self.detector.score_one(reduced)
            
            # Update counters
            self.processed_count += 1
            is_anomaly = score > threshold
            
            if is_anomaly:
                self.anomaly_count += 1
                logger.warning(f"Anomaly detected: {score:.3f}")
            
            # Periodic reporting
            if self.processed_count % 1000 == 0:
                rate = self.anomaly_count / self.processed_count
                logger.info(f"Processed {self.processed_count} points, {rate:.1%} anomalies")
            
            return {
                'score': score,
                'is_anomaly': is_anomaly,
                'processed_count': self.processed_count
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {'error': str(e)}

# Usage
detector = ProductionAnomalyDetector()

# Process your stream
for data_point in your_data_stream:
    result = detector.process_data_point(data_point)
    
    if result.get('is_anomaly'):
        handle_anomaly(data_point, result['score'])
```

## Use Case Examples

### IoT Sensor Monitoring

Monitor industrial sensors for equipment failures:

```python
# Sensor data example
sensor_data = {
    'temperature': 75.2,    # Celsius
    'pressure': 2.1,        # Bar
    'vibration': 0.05,      # G-force
    'current': 12.5,        # Amperes
    'voltage': 230.0        # Volts
}

# Detect sensor anomalies
model = OnlineIsolationForest(num_trees=50, window_size=1000)
model.learn_one(sensor_data)
score = model.score_one(sensor_data)

if score > 0.8:
    print("Equipment maintenance required!")
```

### Network Security

Detect network intrusions and unusual traffic:

```python
# Network connection data
connection_data = {
    'duration': 120,           # seconds
    'bytes_sent': 1024,       # bytes
    'bytes_received': 8192,   # bytes
    'packets_sent': 15,       # count
    'packets_received': 20,   # count
    'port': 80                # destination port
}

# Multi-model approach for security
statistical_model = MovingMahalanobisDistance(window_size=500)
forest_model = OnlineIsolationForest(num_trees=100)

# Get scores from both models
statistical_model.learn_one(connection_data)
stat_score = statistical_model.score_one(connection_data)

forest_model.learn_one(connection_data)
forest_score = forest_model.score_one(connection_data)

# Combined decision
combined_score = 0.6 * stat_score + 0.4 * forest_score
if combined_score > 0.7:
    print("Potential security threat detected!")
```

### Financial Fraud Detection

Monitor transactions for fraudulent activities:

```python
# Transaction data
transaction = {
    'amount': 1500.00,        # USD
    'merchant_category': 5814, # MCC code
    'hour_of_day': 14,        # 0-23
    'day_of_week': 2,         # 0-6
    'account_age_days': 450,  # days
    'location_risk': 0.2      # 0-1 risk score
}

# Feature scaling for mixed data types
scaler = StandardScaler()
fraud_detector = IncrementalOneClassSVMAdaptiveKernel(nu=0.05)

# Process transaction
scaler.learn_one(transaction)
scaled_transaction = scaler.transform_one(transaction)

fraud_detector.learn_one(scaled_transaction)
fraud_score = fraud_detector.score_one(scaled_transaction)

if fraud_score > 0.9:
    print("High-risk transaction - manual review required")
elif fraud_score > 0.7:
    print("Moderate risk - additional verification needed")
```

## Code Organization

All examples follow a consistent structure:

1. **Problem Description**: What we're trying to detect
2. **Data Setup**: How to prepare and load data
3. **Model Configuration**: Choosing and configuring algorithms
4. **Processing Pipeline**: Step-by-step data processing
5. **Results Interpretation**: Understanding and acting on results
6. **Extensions**: Ideas for further development

## Running the Examples

### Prerequisites

Make sure you have ONAD installed with example dependencies:

```bash
pip install onad[eval]
```

### Data Requirements

Most examples use:
- Built-in ONAD datasets (no additional data needed)
- Synthetic data generation (included in examples)
- Public datasets with download instructions

### Example Scripts

Each example page includes:
- Complete, runnable Python scripts
- Jupyter notebook versions (where applicable)
- Docker containerized examples for complex setups
- Command-line interfaces for batch processing

## Best Practices from Examples

### Model Selection
- Start with `OnlineIsolationForest` for general-purpose detection
- Use statistical models for time-series data
- Combine multiple models for higher accuracy

### Data Preprocessing
- Always scale features with different ranges
- Use PCA for high-dimensional data
- Validate data quality before processing

### Threshold Selection
- Start with 95th percentile of normal data scores
- Adjust based on business requirements (false positive vs false negative costs)
- Use adaptive thresholds for evolving data patterns

### Performance Optimization
- Process data in batches when possible
- Monitor memory usage for long-running processes
- Use appropriate window sizes for your data velocity

---

!!! tip "Getting Started"
    1. Start with [Basic Usage](basic_usage.md) for fundamental concepts
    2. Try [Forest Models](forest_models.md) for general-purpose anomaly detection
    3. Explore [Custom Pipelines](custom_pipelines.md) for production scenarios
    4. Check [Statistical Models](statistical_models.md) for time-series data

!!! note "Contributing Examples"
    Have an interesting use case? We welcome community contributions! See our [Contributing Guide](../contributing.md) for how to share your examples.