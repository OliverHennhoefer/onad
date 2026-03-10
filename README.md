# ABERRANT

Online anomaly detection for streaming data.

## Highlights

- Streaming-first model APIs: `learn_one` and `score_one`
- Detector families: isolation forests, distance, SVM, statistical, sketch
- Dataset streaming API with caching
- Composable transforms and pipelines

## Install

```bash
pip install aberrant
```

Optional extras:

- `aberrant[eval]` for evaluation metrics (`scikit-learn`)
- `aberrant[dl]` for deep models (`torch`)
- `aberrant[parquet]` for legacy parquet streaming (`pyarrow`)
- `aberrant[dev]`, `aberrant[docs]`, `aberrant[benchmark]`, `aberrant[all]`

## Quick example

```python
from aberrant.model.iforest import OnlineIsolationForest
from aberrant.stream.dataset import Dataset, load

model = OnlineIsolationForest(window_size=512, num_trees=50)
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 2000:
        if y == 0:
            model.learn_one(x)
        continue

    score = model.score_one(x)

    if score > 0.8:
        print("anomaly", score, y)

    model.learn_one(x)
```

## Stable public imports

- `aberrant.drift`
- `aberrant.model.iforest`
- `aberrant.model.distance`
- `aberrant.model.sketch`
- `aberrant.model.svm`
- `aberrant.model.stat`
- `aberrant.transform.preprocessing`
- `aberrant.transform.projection`
- `aberrant.stream.dataset`

## Score conventions

- `ThresholdModel`: binary score (`0.0` or `1.0`)
- Isolation forest variants: bounded score in `[0, 1]`
- Sketch/distance/SVM/statistical models: model-specific continuous scores

## Optional dependency behavior

- Deep models are optional (`aberrant[dl]`).
- Deep unit tests auto-skip when `torch` is unavailable.
- Integration tests require `aberrant[eval]`.

## Development

```bash
uv sync --extra dev --extra docs
uv run python -m ruff check .
uv run python -m pytest -q
```

## Project docs

- Docs site config: `docs/mkdocs.yml`
- Changelog: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`

## License

MIT (see `LICENSE`).
