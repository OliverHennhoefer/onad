# Pipeline Construction

Building effective anomaly detection pipelines requires combining multiple components in a systematic way. This guide covers pipeline design patterns, composition techniques, and best practices for creating robust detection systems.

## Pipeline Architecture

### Basic Pipeline Structure

A typical ONAD pipeline consists of several stages:

1. **Data Ingestion**: Loading and streaming data
2. **Preprocessing**: Feature scaling and transformation
3. **Feature Engineering**: Dimensionality reduction and selection
4. **Anomaly Detection**: Model scoring and threshold application
5. **Post-processing**: Result interpretation and actions

```python
# Basic pipeline structure
from onad.transform.scale import StandardScaler
from onad.transform.pca import IncrementalPCA
from onad.model.unsupervised.forest import OnlineIsolationForest

# Create pipeline components
scaler = StandardScaler()
pca = IncrementalPCA(n_components=10)
detector = OnlineIsolationForest()

# Process data through pipeline
for data_point in stream:
    # Stage 1: Preprocessing
    scaler.learn_one(data_point)
    scaled_data = scaler.transform_one(data_point)
    
    # Stage 2: Feature engineering
    pca.learn_one(scaled_data)
    reduced_data = pca.transform_one(scaled_data)
    
    # Stage 3: Anomaly detection
    detector.learn_one(reduced_data)
    anomaly_score = detector.score_one(reduced_data)
    
    # Stage 4: Decision making
    if anomaly_score > threshold:
        handle_anomaly(data_point, anomaly_score)
```

## Pipeline Design Patterns

### Sequential Pipeline

The most common pattern chains components sequentially:

```python
class SequentialPipeline:
    def __init__(self, components):
        self.components = components
    
    def learn_one(self, x):
        """Learn from one data point through all components"""
        current_data = x
        for component in self.components:
            if hasattr(component, 'learn_one'):
                component.learn_one(current_data)
            if hasattr(component, 'transform_one'):
                current_data = component.transform_one(current_data)
    
    def score_one(self, x):
        """Score one data point through the pipeline"""
        current_data = x
        
        # Transform through preprocessing stages
        for component in self.components[:-1]:  # All except last
            if hasattr(component, 'transform_one'):
                current_data = component.transform_one(current_data)
        
        # Score with final model
        final_model = self.components[-1]
        return final_model.score_one(current_data)

# Usage
pipeline = SequentialPipeline([
    StandardScaler(),
    IncrementalPCA(n_components=5),
    OnlineIsolationForest()
])

for data_point in stream:
    pipeline.learn_one(data_point)
    score = pipeline.score_one(data_point)
```

### Parallel Pipeline

Process data through multiple parallel paths:

```python
from typing import List, Dict
import numpy as np

class ParallelPipeline:
    def __init__(self, pipelines: List[SequentialPipeline], weights: List[float] = None):
        self.pipelines = pipelines
        self.weights = weights or [1.0] * len(pipelines)
    
    def learn_one(self, x):
        """Learn from data point in all parallel pipelines"""
        for pipeline in self.pipelines:
            pipeline.learn_one(x)
    
    def score_one(self, x) -> float:
        """Get weighted average score from all pipelines"""
        scores = [pipeline.score_one(x) for pipeline in self.pipelines]
        weighted_scores = [s * w for s, w in zip(scores, self.weights)]
        return sum(weighted_scores) / sum(self.weights)

# Create parallel pipelines with different preprocessing
pipeline1 = SequentialPipeline([
    StandardScaler(),
    OnlineIsolationForest(num_trees=50)
])

pipeline2 = SequentialPipeline([
    MinMaxScaler(),
    IncrementalKNN(k=10)
])

# Combine pipelines
ensemble = ParallelPipeline([pipeline1, pipeline2], weights=[0.7, 0.3])

for data_point in stream:
    ensemble.learn_one(data_point)
    combined_score = ensemble.score_one(data_point)
```

### Conditional Pipeline

Apply different processing based on data characteristics:

```python
class ConditionalPipeline:
    def __init__(self, condition_func, pipeline_true, pipeline_false):
        self.condition_func = condition_func
        self.pipeline_true = pipeline_true
        self.pipeline_false = pipeline_false
    
    def _select_pipeline(self, x):
        """Select pipeline based on condition"""
        return self.pipeline_true if self.condition_func(x) else self.pipeline_false
    
    def learn_one(self, x):
        pipeline = self._select_pipeline(x)
        pipeline.learn_one(x)
    
    def score_one(self, x):
        pipeline = self._select_pipeline(x)
        return pipeline.score_one(x)

# Example: Different processing for high/low dimensional data
def is_high_dimensional(x):
    return len(x) > 50

high_dim_pipeline = SequentialPipeline([
    StandardScaler(),
    IncrementalPCA(n_components=20),
    OnlineIsolationForest()
])

low_dim_pipeline = SequentialPipeline([
    StandardScaler(),
    IncrementalKNN(k=5)
])

conditional = ConditionalPipeline(
    is_high_dimensional, 
    high_dim_pipeline, 
    low_dim_pipeline
)
```

## Advanced Pipeline Components

### Feature Selection Pipeline

```python
from typing import Set

class FeatureSelector:
    def __init__(self, selected_features: Set[str]):
        self.selected_features = selected_features
    
    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        """Select only specified features"""
        return {k: v for k, v in x.items() if k in self.selected_features}

# Usage in pipeline
important_features = {'temperature', 'pressure', 'vibration'}
pipeline = SequentialPipeline([
    FeatureSelector(important_features),
    StandardScaler(),
    OnlineIsolationForest()
])
```

### Validation Pipeline

```python
import numpy as np
import logging

class DataValidator:
    def __init__(self, required_features: Set[str], value_ranges: Dict[str, tuple]):
        self.required_features = required_features
        self.value_ranges = value_ranges
        self.logger = logging.getLogger(__name__)
    
    def validate_one(self, x: Dict[str, float]) -> bool:
        """Validate a single data point"""
        try:
            # Check required features
            missing_features = self.required_features - set(x.keys())
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                return False
            
            # Check value ranges
            for feature, value in x.items():
                if feature in self.value_ranges:
                    min_val, max_val = self.value_ranges[feature]
                    if not (min_val <= value <= max_val):
                        self.logger.warning(f"Feature {feature} value {value} outside range [{min_val}, {max_val}]")
                        return False
                
                # Check for invalid values
                if np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Invalid value for feature {feature}: {value}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

# Usage
validator = DataValidator(
    required_features={'temperature', 'pressure'},
    value_ranges={'temperature': (-50, 200), 'pressure': (0, 1000)}
)

for data_point in stream:
    if validator.validate_one(data_point):
        pipeline.learn_one(data_point)
        score = pipeline.score_one(data_point)
```

### Anomaly Explanation Pipeline

```python
class AnomalyExplainer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.baseline_stats = {}
    
    def learn_one(self, x: Dict[str, float]):
        """Learn baseline statistics"""
        for feature, value in x.items():
            if feature not in self.baseline_stats:
                self.baseline_stats[feature] = {'sum': 0, 'sum_sq': 0, 'count': 0}
            
            stats = self.baseline_stats[feature]
            stats['sum'] += value
            stats['sum_sq'] += value ** 2
            stats['count'] += 1
    
    def explain_anomaly(self, x: Dict[str, float]) -> Dict[str, float]:
        """Provide feature-level anomaly scores"""
        explanation = {}
        
        for feature, value in x.items():
            if feature in self.baseline_stats:
                stats = self.baseline_stats[feature]
                
                # Calculate z-score
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - mean ** 2
                std = np.sqrt(max(variance, 1e-8))
                
                z_score = abs(value - mean) / std
                explanation[feature] = z_score
        
        return explanation

# Usage
explainer = AnomalyExplainer(['temperature', 'pressure', 'vibration'])

for data_point in stream:
    # Normal processing
    pipeline.learn_one(data_point)
    score = pipeline.score_one(data_point)
    
    # Explanation for anomalies
    explainer.learn_one(data_point)
    if score > threshold:
        explanation = explainer.explain_anomaly(data_point)
        print(f"Anomaly explanation: {explanation}")
```

## Real-world Pipeline Examples

### IoT Sensor Monitoring

```python
class IoTMonitoringPipeline:
    def __init__(self):
        # Feature selection for sensor data
        self.feature_selector = FeatureSelector({
            'temperature', 'humidity', 'pressure', 'vibration'
        })
        
        # Data validation
        self.validator = DataValidator(
            required_features={'temperature', 'humidity'},
            value_ranges={
                'temperature': (-40, 85),
                'humidity': (0, 100),
                'pressure': (300, 1100)
            }
        )
        
        # Preprocessing pipeline
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=3)
        
        # Anomaly detection
        self.detector = OnlineIsolationForest(num_trees=100, window_size=2000)
        
        # Explanation
        self.explainer = AnomalyExplainer(['temperature', 'humidity', 'pressure', 'vibration'])
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def process_sensor_reading(self, reading: Dict[str, float]) -> Dict[str, any]:
        """Process a single sensor reading"""
        try:
            # Stage 1: Feature selection
            selected_features = self.feature_selector.transform_one(reading)
            
            # Stage 2: Validation
            if not self.validator.validate_one(selected_features):
                return {'status': 'invalid', 'score': 0.0}
            
            # Stage 3: Preprocessing
            self.scaler.learn_one(selected_features)
            scaled = self.scaler.transform_one(selected_features)
            
            self.pca.learn_one(scaled)
            reduced = self.pca.transform_one(scaled)
            
            # Stage 4: Anomaly detection
            self.detector.learn_one(reduced)
            score = self.detector.score_one(reduced)
            
            # Stage 5: Explanation
            self.explainer.learn_one(selected_features)
            explanation = None
            
            if score > 0.7:  # Anomaly threshold
                explanation = self.explainer.explain_anomaly(selected_features)
                self.logger.warning(f"Anomaly detected: score={score:.3f}, explanation={explanation}")
            
            return {
                'status': 'processed',
                'score': score,
                'is_anomaly': score > 0.7,
                'explanation': explanation,
                'processed_features': reduced
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return {'status': 'error', 'error': str(e)}

# Usage
pipeline = IoTMonitoringPipeline()

for sensor_reading in sensor_stream:
    result = pipeline.process_sensor_reading(sensor_reading)
    
    if result['is_anomaly']:
        print(f"ALERT: Sensor anomaly detected!")
        print(f"Score: {result['score']:.3f}")
        print(f"Explanation: {result['explanation']}")
```

### Network Security Pipeline

```python
class NetworkSecurityPipeline:
    def __init__(self):
        # Multiple detection models
        self.statistical_detector = MovingMahalanobisDistance(window_size=1000)
        self.forest_detector = OnlineIsolationForest(num_trees=200)
        self.knn_detector = IncrementalKNN(k=15)
        
        # Preprocessing
        self.scaler = StandardScaler()
        
        # Feature weights for different attack types
        self.feature_weights = {
            'packet_size': 1.5,
            'connection_duration': 2.0,
            'bytes_transferred': 1.2,
            'protocol_type': 0.8
        }
        
        self.attack_threshold = 0.8
        self.logger = logging.getLogger(__name__)
    
    def weight_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply feature weights"""
        return {k: v * self.feature_weights.get(k, 1.0) for k, v in features.items()}
    
    def detect_intrusion(self, network_data: Dict[str, float]) -> Dict[str, any]:
        """Detect network intrusions"""
        try:
            # Preprocessing
            weighted_features = self.weight_features(network_data)
            
            self.scaler.learn_one(weighted_features)
            scaled_features = self.scaler.transform_one(weighted_features)
            
            # Multiple detector scores
            scores = {}
            
            # Statistical detection
            self.statistical_detector.learn_one(scaled_features)
            scores['statistical'] = self.statistical_detector.score_one(scaled_features)
            
            # Forest-based detection
            self.forest_detector.learn_one(scaled_features)
            scores['forest'] = self.forest_detector.score_one(scaled_features)
            
            # Distance-based detection
            self.knn_detector.learn_one(scaled_features)
            scores['knn'] = self.knn_detector.score_one(scaled_features)
            
            # Combine scores (weighted average)
            weights = [0.4, 0.4, 0.2]  # statistical, forest, knn
            combined_score = sum(s * w for s, w in zip(scores.values(), weights))
            
            # Detection decision
            is_attack = combined_score > self.attack_threshold
            
            if is_attack:
                self.logger.critical(f"INTRUSION DETECTED: {combined_score:.3f}")
            
            return {
                'is_attack': is_attack,
                'combined_score': combined_score,
                'individual_scores': scores,
                'confidence': min(combined_score / self.attack_threshold, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return {'is_attack': False, 'error': str(e)}

# Usage
security_pipeline = NetworkSecurityPipeline()

for connection_data in network_stream:
    result = security_pipeline.detect_intrusion(connection_data)
    
    if result['is_attack']:
        # Trigger security response
        trigger_security_alert(connection_data, result)
```

## Pipeline Optimization

### Memory Optimization

```python
class MemoryOptimizedPipeline:
    def __init__(self, max_memory_mb=1000):
        self.max_memory_mb = max_memory_mb
        self.components = []
        self.processing_count = 0
        
    def add_component(self, component, reset_interval=10000):
        """Add component with memory management"""
        self.components.append({
            'component': component,
            'reset_interval': reset_interval,
            'last_reset': 0
        })
    
    def _check_memory_and_reset(self):
        """Check memory usage and reset components if needed"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        if memory_usage > self.max_memory_mb:
            for comp_info in self.components:
                # Reset component to original state
                component_class = type(comp_info['component'])
                comp_info['component'] = component_class()
                comp_info['last_reset'] = self.processing_count
            
            print(f"Memory limit reached ({memory_usage:.1f} MB), components reset")
    
    def process_one(self, x):
        """Process one data point through optimized pipeline"""
        current_data = x
        
        for comp_info in self.components:
            component = comp_info['component']
            
            # Regular processing
            if hasattr(component, 'learn_one'):
                component.learn_one(current_data)
            if hasattr(component, 'transform_one'):
                current_data = component.transform_one(current_data)
            elif hasattr(component, 'score_one'):
                return component.score_one(current_data)
        
        self.processing_count += 1
        
        # Periodic memory check
        if self.processing_count % 1000 == 0:
            self._check_memory_and_reset()
        
        return current_data
```

### Performance Monitoring

```python
import time
from collections import deque

class MonitoredPipeline:
    def __init__(self, pipeline, monitoring_window=1000):
        self.pipeline = pipeline
        self.processing_times = deque(maxlen=monitoring_window)
        self.error_count = 0
        self.total_processed = 0
    
    def process_one(self, x):
        """Process with performance monitoring"""
        start_time = time.time()
        
        try:
            result = self.pipeline.score_one(x)
            self.pipeline.learn_one(x)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.total_processed += 1
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.total_processed += 1
            raise
    
    def get_performance_stats(self):
        """Get pipeline performance statistics"""
        if not self.processing_times:
            return {}
        
        times = list(self.processing_times)
        return {
            'avg_processing_time': sum(times) / len(times),
            'max_processing_time': max(times),
            'min_processing_time': min(times),
            'throughput_per_sec': len(times) / sum(times) if sum(times) > 0 else 0,
            'error_rate': self.error_count / self.total_processed if self.total_processed > 0 else 0,
            'total_processed': self.total_processed
        }
```

---

!!! tip "Pipeline Design Best Practices"
    1. **Start Simple**: Begin with basic sequential pipelines, add complexity as needed
    2. **Validate Early**: Place validation components early in the pipeline
    3. **Monitor Performance**: Track processing times and memory usage
    4. **Handle Errors Gracefully**: Implement error handling at each stage
    5. **Document Components**: Clear documentation helps with maintenance and debugging

!!! warning "Common Pipeline Issues"
    - **Data Leakage**: Don't use future information for current predictions
    - **Memory Leaks**: Monitor memory usage, especially with long-running pipelines
    - **Component Mismatch**: Ensure output of one stage matches input of next
    - **Error Propagation**: One failing component shouldn't crash the entire pipeline