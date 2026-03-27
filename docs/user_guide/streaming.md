# Streaming Datasets

Use `aberrant.stream.dataset` for built-in benchmark streams.

## Basic usage

```python
from aberrant.stream.dataset import Dataset, load

dataset = load(Dataset.FRAUD)

for x, y in dataset.stream():
    ...
```

## Batch streaming

```python
from aberrant.stream.dataset import BatchStreamer, Dataset, load

dataset = load(Dataset.SHUTTLE)
batch_streamer = BatchStreamer(dataset, batch_size=256)

for x_batch, y_batch in batch_streamer.stream():
    ...
```

## Cache management

```python
from aberrant.stream.dataset import get_cache_info, list_cached

print(get_cache_info())
print(list_cached())
```

## Progress bars

```python
from aberrant.stream.dataset import Dataset, DatasetManager

manager = DatasetManager(show_progress=True)  # download progress
dataset = manager.load(Dataset.FRAUD, show_progress=True)  # streaming progress

for x, y in dataset.stream():
    ...
```
