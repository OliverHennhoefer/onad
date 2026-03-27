# Installation

ABERRANT supports Python 3.10, 3.11, and 3.12.

## Basic install

```bash
pip install aberrant
```

## Optional extras

```bash
pip install "aberrant[eval]"      # scikit-learn metrics for evaluations
pip install "aberrant[dl]"        # deep model support (torch)
pip install "aberrant[faiss]"     # FAISS similarity engine support (faiss-cpu)
pip install "aberrant[docs]"      # mkdocs + mkdocstrings
pip install "aberrant[dev]"       # full test/lint/type-check stack (includes torch + scikit-learn)
pip install "aberrant[benchmark]" # river + pytest-benchmark
pip install "aberrant[all]"       # all extras
```

## Local development

```bash
git clone https://github.com/OliverHennhoefer/aberrant.git
cd aberrant
uv sync --extra dev --extra docs
```

## Verify install

```python
import aberrant
from aberrant.model.iforest import OnlineIsolationForest

print(aberrant.__version__)
model = OnlineIsolationForest()
print(model)
```

## Optional dependency behavior

- `tests/models/test_deep_autoencoder.py` skips when `torch` is not installed.
- `tests/models/test_distance_knn.py` skips when `faiss-cpu` is not installed.
- Integration tests require `scikit-learn` (included in `aberrant[dev]`).
