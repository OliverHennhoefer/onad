# Installation

ONAD supports Python 3.10, 3.11, and 3.12.

## Basic install

```bash
pip install onad
```

## Optional extras

```bash
pip install "onad[eval]"      # scikit-learn metrics for evaluations
pip install "onad[dl]"        # deep model support (torch)
pip install "onad[parquet]"   # legacy parquet streamer (pyarrow)
pip install "onad[docs]"      # mkdocs + mkdocstrings
pip install "onad[dev]"       # test/lint/type-check tooling
pip install "onad[benchmark]" # river + pytest-benchmark
pip install "onad[all]"       # all extras
```

## Local development

```bash
git clone https://github.com/OliverHennhoefer/onad.git
cd onad
uv sync --extra dev --extra docs --extra eval
```

## Verify install

```python
import onad
from onad.model.iforest import OnlineIsolationForest

print(onad.__version__)
model = OnlineIsolationForest()
print(model)
```

## Optional dependency behavior

- `tests/models/test_deep_autoencoder.py` skips when `torch` is not installed.
- Integration tests require `scikit-learn` (`onad[eval]`).
