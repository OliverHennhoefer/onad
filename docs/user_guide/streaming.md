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

## Legacy parquet streamer

`aberrant.stream.streamer.ParquetStreamer` is legacy and requires `aberrant[parquet]`.
It is not part of the preferred public API.
