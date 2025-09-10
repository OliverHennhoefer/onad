# Stream Processing

ONAD is designed for processing continuous data streams efficiently. This guide covers data loading, streaming utilities, and best practices for handling real-time and batch data.

## Core Streaming Concepts

### Streaming vs Batch Processing

**Traditional Batch Processing:**
- Load entire dataset into memory
- Process all data at once
- Suitable for offline analysis

**Streaming Processing (ONAD approach):**
- Process one data point at a time
- Constant memory usage
- Real-time results
- Handles infinite streams

### Stream Interface

All ONAD components follow a consistent streaming interface:

```python
# Universal pattern for streaming processing
for data_point in stream:
    # Learn from the data point
    model.learn_one(data_point)
    
    # Get anomaly score
    score = model.score_one(data_point)
    
    # Process result
    if score > threshold:
        handle_anomaly(data_point, score)
```

## Built-in Data Sources

### ParquetStreamer

Efficiently stream data from Parquet files with memory optimization.

**Basic usage:**
```python
from onad.stream import ParquetStreamer

# Stream from file
with ParquetStreamer("data.parquet") as streamer:
    for features, label in streamer:
        model.learn_one(features)
        score = model.score_one(features)
```

**Advanced configuration:**
```python
# Stream with specific label column and data sanitization
with ParquetStreamer(
    "dataset.parquet",
    label_column="is_anomaly",    # Specify label column
    sanitize_floats=True          # Clean float values
) as streamer:
    for features, label in streamer:
        # features: dict of feature name -> value
        # label: value from label column (if specified)
        model.learn_one(features)
```

**Memory efficiency features:**
- Uses PyArrow for efficient Parquet reading
- Processes data in batches internally
- Converts to pandas iteratively to minimize memory usage
- Automatic cleanup of file handles

### Built-in Datasets

ONAD provides sample datasets for testing and experimentation:

```python
from onad.stream import Dataset, ParquetStreamer

# Available datasets
datasets = [
    Dataset.FRAUD,      # Financial fraud detection
    Dataset.NETWORK,    # Network intrusion detection
    Dataset.IOT,        # IoT sensor anomalies
    Dataset.SYSTEM      # System performance anomalies
]

# Use built-in dataset
with ParquetStreamer(Dataset.FRAUD) as streamer:
    for features, label in streamer:
        # Process fraud detection data
        model.learn_one(features)
        score = model.score_one(features)
```

## Custom Data Sources

### File-based Streams

**CSV files:**
```python
import csv
from typing import Iterator, Dict

def csv_stream(filename: str) -> Iterator[Dict[str, float]]:
    """Stream data from CSV file"""
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert string values to float
            yield {k: float(v) for k, v in row.items() if v}

# Usage
for data_point in csv_stream("sensor_data.csv"):
    model.learn_one(data_point)
```

**JSON Lines files:**
```python
import json

def jsonl_stream(filename: str) -> Iterator[Dict[str, float]]:
    """Stream data from JSON Lines file"""
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

# Usage
for data_point in jsonl_stream("events.jsonl"):
    model.learn_one(data_point)
```

### Database Streams

**SQL databases:**
```python
import sqlite3
from typing import Iterator

def sql_stream(db_path: str, query: str) -> Iterator[Dict[str, float]]:
    """Stream data from SQL database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column name access
    
    cursor = conn.execute(query)
    try:
        while True:
            rows = cursor.fetchmany(1000)  # Process in batches
            if not rows:
                break
            
            for row in rows:
                yield dict(row)
    finally:
        conn.close()

# Usage
query = "SELECT temperature, pressure, vibration FROM sensors ORDER BY timestamp"
for data_point in sql_stream("sensors.db", query):
    model.learn_one(data_point)
```

### Real-time Streams

**Message queues (Redis):**
```python
import redis
import json

def redis_stream(host: str, port: int, channel: str) -> Iterator[Dict[str, float]]:
    """Stream data from Redis pub/sub"""
    r = redis.Redis(host=host, port=port)
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                yield data
    finally:
        pubsub.close()

# Usage
for data_point in redis_stream("localhost", 6379, "sensor_data"):
    model.learn_one(data_point)
```

**Apache Kafka:**
```python
from kafka import KafkaConsumer
import json

def kafka_stream(topic: str, bootstrap_servers: list) -> Iterator[Dict[str, float]]:
    """Stream data from Kafka topic"""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    try:
        for message in consumer:
            yield message.value
    finally:
        consumer.close()

# Usage
servers = ['localhost:9092']
for data_point in kafka_stream("anomaly_data", servers):
    model.learn_one(data_point)
```

## Stream Processing Patterns

### Basic Processing Loop

```python
def process_stream(stream, model, threshold=0.7):
    """Basic anomaly detection on stream"""
    anomaly_count = 0
    total_count = 0
    
    for data_point in stream:
        # Update model
        model.learn_one(data_point)
        
        # Get anomaly score
        score = model.score_one(data_point)
        
        # Check for anomaly
        if score > threshold:
            anomaly_count += 1
            print(f"Anomaly detected: {data_point}, Score: {score:.3f}")
        
        total_count += 1
        
        # Report progress
        if total_count % 1000 == 0:
            rate = anomaly_count / total_count
            print(f"Processed {total_count} points, {rate:.1%} anomalies")

# Usage
process_stream(csv_stream("data.csv"), model)
```

### Windowed Processing

```python
from collections import deque
from typing import Deque

def windowed_processing(stream, model, window_size=100):
    """Process stream with sliding window statistics"""
    window: Deque[float] = deque(maxlen=window_size)
    
    for data_point in stream:
        # Update model
        model.learn_one(data_point)
        score = model.score_one(data_point)
        
        # Maintain sliding window of scores
        window.append(score)
        
        if len(window) >= window_size:
            # Compute window statistics
            window_mean = sum(window) / len(window)
            window_std = np.std(window)
            
            # Adaptive threshold based on recent scores
            threshold = window_mean + 2 * window_std
            
            if score > threshold:
                print(f"Anomaly detected with adaptive threshold: {score:.3f} > {threshold:.3f}")

# Usage
windowed_processing(csv_stream("sensor_data.csv"), model)
```

### Batch Processing Interface

Process streams in batches for improved performance:

```python
def batch_stream_processing(stream, model, batch_size=50):
    """Process stream in batches"""
    batch = []
    
    for data_point in stream:
        batch.append(data_point)
        
        if len(batch) >= batch_size:
            # Process batch
            scores = []
            for point in batch:
                model.learn_one(point)
                scores.append(model.score_one(point))
            
            # Analyze batch results
            max_score = max(scores)
            if max_score > 0.8:
                print(f"High anomaly score in batch: {max_score:.3f}")
            
            batch = []  # Reset batch

# Usage
batch_stream_processing(csv_stream("large_dataset.csv"), model, batch_size=100)
```

## Memory Management

### Monitoring Memory Usage

```python
import psutil
import os

def monitor_memory_usage(stream, model):
    """Monitor memory usage during stream processing"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    for i, data_point in enumerate(stream):
        model.learn_one(data_point)
        score = model.score_one(data_point)
        
        # Check memory every 1000 points
        if i % 1000 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            print(f"Point {i}: Memory usage: {current_memory:.1f} MB (+{memory_growth:.1f} MB)")
            
            # Alert if memory growth is excessive
            if memory_growth > 500:  # 500 MB growth
                print("WARNING: Excessive memory growth detected!")

# Usage
monitor_memory_usage(csv_stream("large_file.csv"), model)
```

### Memory-Efficient Streaming

```python
def memory_efficient_processing(stream, model, max_memory_mb=1000):
    """Process stream with memory limits"""
    process = psutil.Process(os.getpid())
    
    for data_point in stream:
        # Check memory before processing
        current_memory = process.memory_info().rss / 1024 / 1024
        
        if current_memory > max_memory_mb:
            # Reset model to free memory
            print(f"Memory limit reached ({current_memory:.1f} MB), resetting model")
            model = type(model)()  # Create fresh instance
        
        model.learn_one(data_point)
        score = model.score_one(data_point)
```

## Performance Optimization

### Parallel Processing

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import queue

def parallel_stream_processing(stream, model_class, num_workers=4):
    """Process stream with multiple workers"""
    
    def worker(data_queue, result_queue):
        """Worker process"""
        model = model_class()
        while True:
            try:
                data_point = data_queue.get(timeout=1)
                if data_point is None:  # Sentinel value
                    break
                
                model.learn_one(data_point)
                score = model.score_one(data_point)
                result_queue.put((data_point, score))
                
            except queue.Empty:
                continue
    
    # Create queues
    data_queue = multiprocessing.Queue(maxsize=1000)
    result_queue = multiprocessing.Queue()
    
    # Start workers
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(data_queue, result_queue))
        p.start()
        workers.append(p)
    
    # Feed data to workers
    for data_point in stream:
        data_queue.put(data_point)
    
    # Send stop signals
    for _ in workers:
        data_queue.put(None)
    
    # Collect results
    for _ in workers:
        p.join()
```

### Caching and Preprocessing

```python
from functools import lru_cache

class CachedPreprocessor:
    def __init__(self):
        self.feature_cache = {}
    
    @lru_cache(maxsize=1000)
    def preprocess_features(self, data_tuple):
        """Cache preprocessing results"""
        data_dict = dict(data_tuple)
        # Expensive preprocessing here
        return data_dict
    
    def process_stream(self, stream, model):
        for data_point in stream:
            # Convert to tuple for hashing
            data_tuple = tuple(sorted(data_point.items()))
            
            # Use cached preprocessing
            processed = self.preprocess_features(data_tuple)
            
            model.learn_one(processed)
            score = model.score_one(processed)
```

## Error Handling and Resilience

### Robust Stream Processing

```python
import logging
from typing import Optional

def robust_stream_processing(stream, model, error_threshold=0.01):
    """Process stream with error handling"""
    logger = logging.getLogger(__name__)
    
    total_points = 0
    error_count = 0
    
    for data_point in stream:
        try:
            # Validate data point
            if not isinstance(data_point, dict):
                raise ValueError("Data point must be a dictionary")
            
            if not data_point:
                raise ValueError("Data point cannot be empty")
            
            # Process point
            model.learn_one(data_point)
            score = model.score_one(data_point)
            
            total_points += 1
            
        except Exception as e:
            error_count += 1
            error_rate = error_count / (total_points + error_count)
            
            logger.error(f"Error processing data point: {e}")
            
            # Stop if error rate is too high
            if error_rate > error_threshold:
                logger.critical(f"Error rate {error_rate:.1%} exceeds threshold {error_threshold:.1%}")
                break
            
            continue

# Usage with logging
logging.basicConfig(level=logging.INFO)
robust_stream_processing(csv_stream("noisy_data.csv"), model)
```

### Checkpointing

```python
import pickle
import os

def checkpointed_processing(stream, model, checkpoint_interval=1000, checkpoint_file="model_checkpoint.pkl"):
    """Process stream with periodic checkpointing"""
    
    point_count = 0
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from checkpoint: {checkpoint_file}")
    
    try:
        for data_point in stream:
            model.learn_one(data_point)
            score = model.score_one(data_point)
            point_count += 1
            
            # Save checkpoint periodically
            if point_count % checkpoint_interval == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Checkpoint saved at point {point_count}")
                
    except KeyboardInterrupt:
        print("Processing interrupted, saving final checkpoint...")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(model, f)
        raise

# Usage
checkpointed_processing(csv_stream("large_dataset.csv"), model)
```

---

!!! tip "Stream Performance Tips"
    - Use generators instead of loading entire datasets into memory
    - Process data in batches when possible for better throughput
    - Monitor memory usage and reset models if needed
    - Implement checkpointing for long-running processes
    - Use appropriate data types (avoid unnecessary string conversions)

!!! warning "Common Pitfalls"
    - **Memory leaks**: Always close file handles and database connections
    - **Blocking operations**: Avoid synchronous I/O in streaming loops
    - **Error propagation**: Don't let single bad data points crash the entire stream
    - **Resource exhaustion**: Monitor and limit memory/CPU usage