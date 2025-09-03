# Best Practices

This guide provides proven strategies for deploying ONAD in production environments, optimizing performance, and ensuring reliable anomaly detection systems.

## Production Deployment

### System Architecture

**Microservices Design**
```python
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class AnomalyDetectionConfig:
    model_type: str
    model_params: Dict[str, Any]
    preprocessing_steps: list
    thresholds: Dict[str, float]
    memory_limit_mb: int = 1000
    checkpoint_interval: int = 10000

class AnomalyDetectionService:
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model = self._create_model()
        self.preprocessors = self._create_preprocessors()
        
        # Monitoring
        self.processed_count = 0
        self.anomaly_count = 0
        self.last_checkpoint = 0
        
    def _create_model(self):
        """Factory method for model creation"""
        model_map = {
            'isolation_forest': OnlineIsolationForest,
            'adaptive_svm': IncrementalOneClassSVMAdaptiveKernel,
            'knn': IncrementalKNN
        }
        
        model_class = model_map.get(self.config.model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        return model_class(**self.config.model_params)
    
    def _create_preprocessors(self):
        """Create preprocessing pipeline"""
        preprocessors = []
        for step in self.config.preprocessing_steps:
            if step['type'] == 'scaler':
                from onad.transform.scale import StandardScaler
                preprocessors.append(StandardScaler(**step.get('params', {})))
            elif step['type'] == 'pca':
                from onad.transform.pca import IncrementalPCA
                preprocessors.append(IncrementalPCA(**step.get('params', {})))
        
        return preprocessors
    
    def process_data_point(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Process single data point"""
        try:
            # Preprocessing
            processed_data = data.copy()
            for preprocessor in self.preprocessors:
                preprocessor.learn_one(processed_data)
                processed_data = preprocessor.transform_one(processed_data)
            
            # Anomaly detection
            self.model.learn_one(processed_data)
            score = self.model.score_one(processed_data)
            
            # Thresholding
            is_anomaly = score > self.config.thresholds.get('anomaly', 0.7)
            severity = self._calculate_severity(score)
            
            # Update counters
            self.processed_count += 1
            if is_anomaly:
                self.anomaly_count += 1
            
            # Periodic operations
            if self.processed_count % self.config.checkpoint_interval == 0:
                self._checkpoint_model()
            
            result = {
                'anomaly_score': score,
                'is_anomaly': is_anomaly,
                'severity': severity,
                'timestamp': time.time(),
                'model_state': self._get_model_info()
            }
            
            if is_anomaly:
                self.logger.warning(f"Anomaly detected: score={score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity level"""
        thresholds = self.config.thresholds
        
        if score > thresholds.get('critical', 0.9):
            return 'critical'
        elif score > thresholds.get('high', 0.8):
            return 'high'
        elif score > thresholds.get('medium', 0.7):
            return 'medium'
        else:
            return 'low'
    
    def _checkpoint_model(self):
        """Save model checkpoint"""
        import pickle
        import os
        
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{checkpoint_dir}/model_{self.processed_count}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessors': self.preprocessors,
                'processed_count': self.processed_count,
                'config': self.config
            }, f)
        
        self.logger.info(f"Model checkpoint saved: {checkpoint_path}")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model state information"""
        return {
            'processed_count': self.processed_count,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.processed_count, 1),
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

# Configuration example
config = AnomalyDetectionConfig(
    model_type='isolation_forest',
    model_params={'num_trees': 100, 'window_size': 2000},
    preprocessing_steps=[
        {'type': 'scaler', 'params': {}},
        {'type': 'pca', 'params': {'n_components': 10}}
    ],
    thresholds={'medium': 0.6, 'high': 0.8, 'critical': 0.9}
)

# Service usage
service = AnomalyDetectionService(config)
result = service.process_data_point({'temperature': 25.5, 'pressure': 1013.25})
```

### Error Handling and Resilience

**Graceful Degradation**
```python
import time
from enum import Enum
from typing import Optional

class ServiceState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class ResilientAnomalyDetector:
    def __init__(self, primary_model, fallback_model=None):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.state = ServiceState.HEALTHY
        
        # Error tracking
        self.error_count = 0
        self.error_window = 100
        self.error_threshold = 0.1  # 10% error rate
        self.last_errors = deque(maxlen=self.error_window)
        
        # Circuit breaker
        self.circuit_breaker_open = False
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_opened_at = None
        
        self.logger = logging.getLogger(__name__)
    
    def process_with_resilience(self, data_point: Dict[str, float]) -> Dict[str, Any]:
        """Process data point with error handling and fallback"""
        try:
            # Check circuit breaker
            if self._should_use_circuit_breaker():
                return self._fallback_processing(data_point, "circuit_breaker")
            
            # Try primary model
            result = self._process_with_primary(data_point)
            self._record_success()
            
            return result
            
        except Exception as e:
            self._record_error(e)
            
            # Try fallback processing
            return self._fallback_processing(data_point, str(e))
    
    def _process_with_primary(self, data_point: Dict[str, float]) -> Dict[str, Any]:
        """Process with primary model"""
        self.primary_model.learn_one(data_point)
        score = self.primary_model.score_one(data_point)
        
        return {
            'score': score,
            'model': 'primary',
            'timestamp': time.time(),
            'state': self.state.value
        }
    
    def _fallback_processing(self, data_point: Dict[str, float], reason: str) -> Dict[str, Any]:
        """Fallback processing when primary fails"""
        self.logger.warning(f"Using fallback processing: {reason}")
        
        try:
            if self.fallback_model:
                self.fallback_model.learn_one(data_point)
                score = self.fallback_model.score_one(data_point)
                model_used = 'fallback'
            else:
                # Simple statistical fallback
                score = self._simple_statistical_anomaly_score(data_point)
                model_used = 'statistical'
            
            return {
                'score': score,
                'model': model_used,
                'timestamp': time.time(),
                'fallback_reason': reason,
                'state': ServiceState.DEGRADED.value
            }
            
        except Exception as e:
            self.logger.error(f"Fallback processing failed: {e}")
            return {
                'score': 0.0,
                'model': 'none',
                'error': str(e),
                'timestamp': time.time(),
                'state': ServiceState.FAILED.value
            }
    
    def _simple_statistical_anomaly_score(self, data_point: Dict[str, float]) -> float:
        """Simple statistical anomaly detection as last resort"""
        # Use z-score based on running statistics
        values = list(data_point.values())
        if not hasattr(self, '_running_mean'):
            self._running_mean = np.mean(values)
            self._running_std = 1.0
            self._count = 1
            return 0.0
        
        # Update running statistics
        current_mean = np.mean(values)
        self._count += 1
        alpha = 1.0 / self._count
        self._running_mean = (1 - alpha) * self._running_mean + alpha * current_mean
        
        # Calculate anomaly score based on deviation
        deviation = abs(current_mean - self._running_mean)
        return min(deviation / (self._running_std + 1e-8), 1.0)
    
    def _record_error(self, error: Exception):
        """Record error for circuit breaker logic"""
        self.error_count += 1
        self.last_errors.append(time.time())
        
        # Check if we should open circuit breaker
        recent_error_rate = len(self.last_errors) / self.error_window
        if recent_error_rate > self.error_threshold:
            self._open_circuit_breaker()
        
        self.logger.error(f"Model error: {error}")
    
    def _record_success(self):
        """Record successful processing"""
        if self.circuit_breaker_open:
            # Try to close circuit breaker
            self._close_circuit_breaker()
    
    def _should_use_circuit_breaker(self) -> bool:
        """Check if circuit breaker should prevent primary model use"""
        if not self.circuit_breaker_open:
            return False
        
        # Check if timeout has passed
        if (time.time() - self.circuit_breaker_opened_at) > self.circuit_breaker_timeout:
            self._close_circuit_breaker()
            return False
        
        return True
    
    def _open_circuit_breaker(self):
        """Open circuit breaker"""
        self.circuit_breaker_open = True
        self.circuit_breaker_opened_at = time.time()
        self.state = ServiceState.DEGRADED
        self.logger.warning("Circuit breaker opened - using fallback processing")
    
    def _close_circuit_breaker(self):
        """Close circuit breaker"""
        self.circuit_breaker_open = False
        self.circuit_breaker_opened_at = None
        self.state = ServiceState.HEALTHY
        self.logger.info("Circuit breaker closed - resuming normal processing")
```

### Monitoring and Observability

**Comprehensive Monitoring**
```python
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class PerformanceMetrics:
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    anomaly_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughput: deque = field(default_factory=lambda: deque(maxlen=1000))

class MonitoringService:
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.metrics = PerformanceMetrics()
        self.alert_thresholds = alert_thresholds
        self.logger = logging.getLogger(__name__)
        
        # Alert state tracking
        self.active_alerts = set()
        self.alert_cooldown = 300  # 5 minutes
        self.last_alert_times = defaultdict(float)
        
    def record_processing_time(self, processing_time: float):
        """Record processing time for monitoring"""
        self.metrics.processing_times.append(processing_time)
        
        # Check for performance degradation
        if len(self.metrics.processing_times) >= 100:
            avg_time = sum(list(self.metrics.processing_times)[-100:]) / 100
            if avg_time > self.alert_thresholds.get('slow_processing', 0.1):
                self._trigger_alert('slow_processing', f"Average processing time: {avg_time:.3f}s")
    
    def record_error_rate(self, error_count: int, total_count: int):
        """Record error rate"""
        error_rate = error_count / max(total_count, 1)
        self.metrics.error_rates.append(error_rate)
        
        if error_rate > self.alert_thresholds.get('high_error_rate', 0.05):
            self._trigger_alert('high_error_rate', f"Error rate: {error_rate:.1%}")
    
    def record_anomaly_rate(self, anomaly_count: int, total_count: int):
        """Record anomaly detection rate"""
        anomaly_rate = anomaly_count / max(total_count, 1)
        self.metrics.anomaly_rates.append(anomaly_rate)
        
        # Alert for unusual anomaly rates
        if len(self.metrics.anomaly_rates) >= 100:
            recent_rate = sum(list(self.metrics.anomaly_rates)[-100:]) / 100
            
            if recent_rate > self.alert_thresholds.get('high_anomaly_rate', 0.1):
                self._trigger_alert('high_anomaly_rate', f"Anomaly rate: {recent_rate:.1%}")
            elif recent_rate < self.alert_thresholds.get('low_anomaly_rate', 0.001):
                self._trigger_alert('low_anomaly_rate', f"Anomaly rate: {recent_rate:.1%}")
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        self.metrics.memory_usage.append(memory_mb)
        
        if memory_mb > self.alert_thresholds.get('high_memory', 1000):
            self._trigger_alert('high_memory', f"Memory usage: {memory_mb:.1f} MB")
    
    def record_throughput(self, points_per_second: float):
        """Record processing throughput"""
        self.metrics.throughput.append(points_per_second)
        
        if points_per_second < self.alert_thresholds.get('low_throughput', 10):
            self._trigger_alert('low_throughput', f"Throughput: {points_per_second:.1f} points/sec")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger alert with cooldown logic"""
        current_time = time.time()
        
        # Check cooldown
        if (current_time - self.last_alert_times[alert_type]) < self.alert_cooldown:
            return
        
        self.last_alert_times[alert_type] = current_time
        self.active_alerts.add(alert_type)
        
        # Log alert
        self.logger.critical(f"ALERT [{alert_type}]: {message}")
        
        # Send to external monitoring (implement as needed)
        self._send_external_alert(alert_type, message)
    
    def _send_external_alert(self, alert_type: str, message: str):
        """Send alert to external monitoring system"""
        # Implement integration with your monitoring system
        # Examples: Slack, PagerDuty, email, etc.
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        if not any(self.metrics.processing_times):
            return {'status': 'unknown', 'reason': 'no_data'}
        
        # Calculate current metrics
        recent_processing_time = sum(list(self.metrics.processing_times)[-10:]) / min(10, len(self.metrics.processing_times))
        recent_memory = list(self.metrics.memory_usage)[-1] if self.metrics.memory_usage else 0
        recent_throughput = list(self.metrics.throughput)[-1] if self.metrics.throughput else 0
        
        # Determine health status
        if self.active_alerts:
            status = 'unhealthy'
        elif (recent_processing_time > self.alert_thresholds.get('slow_processing', 0.1) * 0.8 or
              recent_memory > self.alert_thresholds.get('high_memory', 1000) * 0.8):
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'active_alerts': list(self.active_alerts),
            'metrics': {
                'avg_processing_time': recent_processing_time,
                'memory_usage_mb': recent_memory,
                'throughput_per_sec': recent_throughput
            },
            'timestamp': time.time()
        }

# Usage example
alert_thresholds = {
    'slow_processing': 0.1,      # seconds
    'high_error_rate': 0.05,     # 5%
    'high_anomaly_rate': 0.15,   # 15%
    'low_anomaly_rate': 0.001,   # 0.1%
    'high_memory': 1000,         # MB
    'low_throughput': 10         # points/sec
}

monitoring = MonitoringService(alert_thresholds)
```

## Memory Management {#memory}

### Memory-Efficient Processing

```python
import gc
import psutil
import os

class MemoryManager:
    def __init__(self, max_memory_mb=2000, cleanup_threshold=0.8):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.process = psutil.Process(os.getpid())
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        current_memory = self.get_memory_usage()
        return current_memory > (self.max_memory_mb * self.cleanup_threshold)
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        before_memory = self.get_memory_usage()
        
        # Run garbage collection
        collected = gc.collect()
        
        after_memory = self.get_memory_usage()
        freed_memory = before_memory - after_memory
        
        self.logger.info(
            f"Memory cleanup: freed {freed_memory:.1f} MB, "
            f"collected {collected} objects"
        )
        
        return freed_memory
    
    def monitor_component_memory(self, component, component_name: str):
        """Monitor memory usage of specific component"""
        if hasattr(component, '__sizeof__'):
            component_size = component.__sizeof__() / 1024 / 1024  # MB
            if component_size > 100:  # Log if component uses >100MB
                self.logger.warning(
                    f"Component {component_name} using {component_size:.1f} MB"
                )

class MemoryAwareModel:
    def __init__(self, base_model, memory_manager: MemoryManager, reset_threshold_mb=500):
        self.base_model = base_model
        self.memory_manager = memory_manager
        self.reset_threshold_mb = reset_threshold_mb
        self.processed_count = 0
        
    def learn_one(self, x):
        """Learn with memory monitoring"""
        self.base_model.learn_one(x)
        self.processed_count += 1
        
        # Periodic memory check
        if self.processed_count % 1000 == 0:
            current_memory = self.memory_manager.get_memory_usage()
            
            if current_memory > self.reset_threshold_mb:
                self._reset_model()
    
    def score_one(self, x):
        """Score with memory monitoring"""
        return self.base_model.score_one(x)
    
    def _reset_model(self):
        """Reset model to free memory"""
        model_class = type(self.base_model)
        self.base_model = model_class()
        self.memory_manager.force_cleanup()
        
        self.logger.info(f"Model reset after {self.processed_count} points")

# Usage
memory_manager = MemoryManager(max_memory_mb=2000)
memory_aware_model = MemoryAwareModel(OnlineIsolationForest(), memory_manager)
```

### Efficient Data Structures

```python
import numpy as np
from collections import deque
from typing import Dict, Any

class EfficientFeatureStore:
    """Memory-efficient storage for streaming features"""
    
    def __init__(self, max_size=10000, feature_types=None):
        self.max_size = max_size
        self.feature_types = feature_types or {}
        self.data = {}
        self.count = 0
        
    def add_point(self, features: Dict[str, Any]):
        """Add data point with automatic type optimization"""
        for key, value in features.items():
            if key not in self.data:
                # Initialize efficient storage based on data type
                dtype = self.feature_types.get(key, self._infer_dtype(value))
                if dtype == 'category':
                    self.data[key] = deque(maxlen=self.max_size)
                else:
                    self.data[key] = np.zeros(self.max_size, dtype=dtype)
                    self.data[key + '_idx'] = 0
            
            # Store value efficiently
            if key + '_idx' in self.data:  # Numeric array
                idx = self.data[key + '_idx']
                if idx < self.max_size:
                    self.data[key][idx] = value
                    self.data[key + '_idx'] += 1
                else:
                    # Shift array
                    self.data[key][:-1] = self.data[key][1:]
                    self.data[key][-1] = value
            else:  # Deque for categorical
                self.data[key].append(value)
        
        self.count += 1
    
    def _infer_dtype(self, value):
        """Infer efficient data type"""
        if isinstance(value, bool):
            return np.bool_
        elif isinstance(value, int) and -128 <= value <= 127:
            return np.int8
        elif isinstance(value, int) and -32768 <= value <= 32767:
            return np.int16
        elif isinstance(value, int):
            return np.int32
        elif isinstance(value, float):
            return np.float32
        else:
            return 'category'
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage by feature"""
        usage = {}
        for key, data in self.data.items():
            if isinstance(data, np.ndarray):
                usage[key] = data.nbytes / 1024 / 1024  # MB
            elif isinstance(data, deque):
                usage[key] = len(data) * 8 / 1024 / 1024  # Rough estimate
        return usage
```

## Performance Optimization

### Batch Processing

```python
from typing import List, Iterator
import time

class BatchProcessor:
    def __init__(self, batch_size=100, timeout_seconds=1.0):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.batch = []
        self.last_batch_time = time.time()
    
    def add_point(self, data_point: Dict[str, float]) -> Iterator[List[Dict[str, float]]]:
        """Add point to batch and yield when ready"""
        self.batch.append(data_point)
        current_time = time.time()
        
        # Yield batch if size or timeout reached
        if (len(self.batch) >= self.batch_size or 
            (current_time - self.last_batch_time) > self.timeout_seconds):
            
            batch_to_process = self.batch.copy()
            self.batch = []
            self.last_batch_time = current_time
            yield batch_to_process
    
    def flush(self) -> Iterator[List[Dict[str, float]]]:
        """Flush remaining batch"""
        if self.batch:
            yield self.batch
            self.batch = []

class BatchedAnomalyDetector:
    def __init__(self, model, batch_processor: BatchProcessor):
        self.model = model
        self.batch_processor = batch_processor
    
    def process_stream(self, data_stream):
        """Process stream in batches for better performance"""
        for data_point in data_stream:
            # Add to batch
            for batch in self.batch_processor.add_point(data_point):
                yield from self._process_batch(batch)
        
        # Process final batch
        for batch in self.batch_processor.flush():
            yield from self._process_batch(batch)
    
    def _process_batch(self, batch: List[Dict[str, float]]):
        """Process a batch of data points"""
        start_time = time.time()
        results = []
        
        for data_point in batch:
            # Learn and score
            self.model.learn_one(data_point)
            score = self.model.score_one(data_point)
            
            results.append({
                'data': data_point,
                'score': score,
                'timestamp': time.time()
            })
        
        processing_time = time.time() - start_time
        throughput = len(batch) / processing_time
        
        # Log batch performance
        if throughput < 100:  # Alert if <100 points/sec
            logging.warning(f"Low batch throughput: {throughput:.1f} points/sec")
        
        yield from results

# Usage
batch_processor = BatchProcessor(batch_size=50, timeout_seconds=0.5)
batched_detector = BatchedAnomalyDetector(model, batch_processor)

for result in batched_detector.process_stream(data_stream):
    if result['score'] > 0.8:
        handle_anomaly(result)
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib
import json

class SmartCacheAnomalyDetector:
    def __init__(self, model, cache_size=1000):
        self.model = model
        self.cache_size = cache_size
        
        # Feature preprocessing cache
        self._preprocess_cache = {}
        
        # Score cache for identical inputs
        self._score_cache = {}
        
    @lru_cache(maxsize=1000)
    def _cached_transform(self, data_hash: str, data_json: str) -> str:
        """Cache expensive transformations"""
        data = json.loads(data_json)
        # Expensive preprocessing here
        return json.dumps(self._expensive_preprocessing(data))
    
    def _expensive_preprocessing(self, data: Dict[str, float]) -> Dict[str, float]:
        """Placeholder for expensive preprocessing"""
        # Example: complex feature engineering
        result = data.copy()
        result['feature_sum'] = sum(data.values())
        result['feature_mean'] = sum(data.values()) / len(data)
        return result
    
    def _get_data_hash(self, data: Dict[str, float]) -> str:
        """Create hash for data point"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def score_one_cached(self, data: Dict[str, float]) -> float:
        """Score with caching"""
        data_hash = self._get_data_hash(data)
        
        # Check score cache
        if data_hash in self._score_cache:
            return self._score_cache[data_hash]
        
        # Check preprocessing cache
        data_json = json.dumps(data, sort_keys=True)
        if data_hash in self._preprocess_cache:
            processed_data = self._preprocess_cache[data_hash]
        else:
            processed_json = self._cached_transform(data_hash, data_json)
            processed_data = json.loads(processed_json)
            self._preprocess_cache[data_hash] = processed_data
        
        # Score with model
        score = self.model.score_one(processed_data)
        
        # Cache score
        if len(self._score_cache) < self.cache_size:
            self._score_cache[data_hash] = score
        
        return score
    
    def clear_caches(self):
        """Clear all caches"""
        self._preprocess_cache.clear()
        self._score_cache.clear()
        self._cached_transform.cache_clear()
```

## Security Considerations

### Input Validation and Sanitization

```python
import re
from typing import Set, Dict, Any, List

class SecurityValidator:
    def __init__(self):
        self.max_feature_count = 1000
        self.max_string_length = 10000
        self.allowed_feature_patterns = [
            r'^[a-zA-Z][a-zA-Z0-9_]*$',  # Valid identifier
            r'^\d+\.\d+\.\d+\.\d+$',      # IP address
            r'^[0-9a-fA-F-]{36}$'         # UUID
        ]
        self.blocked_patterns = [
            r'<script.*?>',               # Script tags
            r'javascript:',               # JavaScript URLs
            r'eval\(',                    # Eval calls
            r'__.*__'                     # Python special methods
        ]
        
    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input data"""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        
        if len(data) > self.max_feature_count:
            raise ValueError(f"Too many features: {len(data)} > {self.max_feature_count}")
        
        sanitized = {}
        
        for key, value in data.items():
            # Validate key
            self._validate_key(key)
            
            # Validate and sanitize value
            sanitized_value = self._validate_value(value)
            sanitized[key] = sanitized_value
        
        return sanitized
    
    def _validate_key(self, key: str):
        """Validate feature key"""
        if not isinstance(key, str):
            raise ValueError("Feature keys must be strings")
        
        if len(key) > 100:
            raise ValueError("Feature key too long")
        
        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, key, re.IGNORECASE):
                raise ValueError(f"Blocked pattern in key: {key}")
        
        # Check against allowed patterns
        key_valid = any(re.match(pattern, key) for pattern in self.allowed_feature_patterns)
        if not key_valid:
            raise ValueError(f"Invalid feature key format: {key}")
    
    def _validate_value(self, value: Any) -> float:
        """Validate and convert value to safe numeric type"""
        if isinstance(value, (int, float)):
            if not (-1e10 < value < 1e10):  # Reasonable range
                raise ValueError("Numeric value out of range")
            return float(value)
        
        elif isinstance(value, str):
            if len(value) > self.max_string_length:
                raise ValueError("String value too long")
            
            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValueError("Blocked content in string value")
            
            # Try to convert to float
            try:
                return float(value)
            except ValueError:
                # Hash string to numeric value
                import hashlib
                hash_val = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
                return float(hash_val % 10000)  # Normalize to reasonable range
        
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

class SecureAnomalyDetectionService:
    def __init__(self, model, rate_limit_per_minute=1000):
        self.model = model
        self.validator = SecurityValidator()
        self.rate_limiter = self._create_rate_limiter(rate_limit_per_minute)
        self.logger = logging.getLogger(__name__)
        
        # Security monitoring
        self.security_events = deque(maxlen=1000)
    
    def _create_rate_limiter(self, requests_per_minute):
        """Create rate limiter"""
        from collections import defaultdict
        import time
        
        class RateLimiter:
            def __init__(self, max_requests):
                self.max_requests = max_requests
                self.requests = defaultdict(list)
            
            def is_allowed(self, client_id: str) -> bool:
                now = time.time()
                minute_ago = now - 60
                
                # Clean old requests
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id] 
                    if req_time > minute_ago
                ]
                
                # Check limit
                if len(self.requests[client_id]) >= self.max_requests:
                    return False
                
                # Record request
                self.requests[client_id].append(now)
                return True
        
        return RateLimiter(requests_per_minute)
    
    def process_secure(self, data: Dict[str, Any], client_id: str = "default") -> Dict[str, Any]:
        """Process data with security validation"""
        try:
            # Rate limiting
            if not self.rate_limiter.is_allowed(client_id):
                self._log_security_event("rate_limit_exceeded", client_id)
                raise ValueError("Rate limit exceeded")
            
            # Input validation
            validated_data = self.validator.validate_input(data)
            
            # Normal processing
            self.model.learn_one(validated_data)
            score = self.model.score_one(validated_data)
            
            return {
                'score': score,
                'timestamp': time.time(),
                'status': 'success'
            }
            
        except ValueError as e:
            # Security violation
            self._log_security_event("validation_error", client_id, str(e))
            return {
                'error': 'Invalid input',
                'timestamp': time.time(),
                'status': 'rejected'
            }
        
        except Exception as e:
            # System error
            self.logger.error(f"Processing error: {e}")
            return {
                'error': 'Processing failed',
                'timestamp': time.time(),
                'status': 'error'
            }
    
    def _log_security_event(self, event_type: str, client_id: str, details: str = ""):
        """Log security event"""
        event = {
            'type': event_type,
            'client_id': client_id,
            'timestamp': time.time(),
            'details': details
        }
        
        self.security_events.append(event)
        self.logger.warning(f"Security event: {event}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security event summary"""
        if not self.security_events:
            return {'total_events': 0}
        
        event_types = {}
        for event in self.security_events:
            event_types[event['type']] = event_types.get(event['type'], 0) + 1
        
        return {
            'total_events': len(self.security_events),
            'event_types': event_types,
            'recent_events': list(self.security_events)[-10:]  # Last 10 events
        }

# Usage
secure_service = SecureAnomalyDetectionService(model, rate_limit_per_minute=500)
result = secure_service.process_secure(data_point, client_id="user_123")
```

---

!!! tip "Production Checklist"
    - [ ] Implement comprehensive error handling and circuit breakers
    - [ ] Set up monitoring and alerting for key metrics
    - [ ] Configure appropriate memory limits and cleanup
    - [ ] Validate and sanitize all inputs
    - [ ] Implement rate limiting and security measures
    - [ ] Set up checkpointing for long-running processes
    - [ ] Test failover and recovery procedures
    - [ ] Document configuration and operational procedures

!!! warning "Common Production Issues"
    - **Memory leaks**: Monitor memory usage and implement cleanup
    - **Performance degradation**: Use batch processing and caching
    - **Security vulnerabilities**: Always validate inputs
    - **Single points of failure**: Implement redundancy and failover