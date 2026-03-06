# ONAD

Online anomaly detection for streaming data.

## Highlights

- Streaming-first model APIs: `learn_one` and `score_one`
- Detector families: isolation forests, distance, SVM, statistical
- Dataset streaming API with caching
- Composable transforms and pipelines

## Install

```bash
pip install onad
```

Optional extras:

- `onad[eval]` for evaluation metrics (`scikit-learn`)
- `onad[dl]` for deep models (`torch`)
- `onad[parquet]` for legacy parquet streaming (`pyarrow`)
- `onad[dev]`, `onad[docs]`, `onad[benchmark]`, `onad[all]`

## Quick example

```python
from onad.model.iforest import OnlineIsolationForest
from onad.stream.dataset import Dataset, load

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

- `onad.drift`
- `onad.model.iforest`
- `onad.model.distance`
- `onad.model.svm`
- `onad.model.stat`
- `onad.transform.preprocessing`
- `onad.transform.projection`
- `onad.stream.dataset`

## Score conventions

- `ThresholdModel`: binary score (`0.0` or `1.0`)
- Isolation forest variants: bounded score in `[0, 1]`
- Distance/SVM/statistical models: model-specific continuous scores

## Optional dependency behavior

- Deep models are optional (`onad[dl]`).
- Deep unit tests auto-skip when `torch` is unavailable.
- Integration tests require `onad[eval]`.

## Development

```bash
uv sync --extra dev --extra docs --extra eval
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
