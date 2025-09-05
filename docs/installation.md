# Installation Guide

ONAD supports Python 3.10+ and can be installed via pip. This guide covers various installation options and requirements.

## Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: At least 512 MB RAM (more for larger datasets)

## Basic Installation

The simplest way to install ONAD is using pip:

```bash
pip install onad
```

This installs the core package with essential dependencies:

- `numpy>=2.2.3` - Numerical computing
- `faiss-cpu>=1.10.0` - Similarity search engine
- `tqdm>=4.67.1` - Progress bars
- `scipy>=1.15.2` - Scientific computing

## Installation Options

ONAD provides several optional dependency groups for different use cases:

### Evaluation Tools
For model evaluation and benchmarking:

```bash
pip install onad[eval]
```

Includes:
- `scikit-learn` - Machine learning utilities and metrics

### Deep Learning Support
For neural network-based models:

```bash
pip install onad[dl]
```

Includes:
- `torch` - PyTorch deep learning framework

### Benchmarking Tools
For comparing with other streaming libraries:

```bash
pip install onad[benchmark]
```

Includes:
- `river` - Online machine learning library

### Development Tools
For contributing to ONAD:

```bash
pip install onad[dev]
```

Includes:
- `ruff` - Fast Python linter and formatter
- `pre-commit` - Git hooks for code quality
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `mypy` - Static type checking

### Documentation Tools
For building documentation:

```bash
pip install onad[docs]
```

Includes:
- `mkdocs` - Documentation generator
- `mkdocs-material` - Material theme for MkDocs
- `mkdocstrings` - API documentation from docstrings

### Complete Installation
To install all optional dependencies:

```bash
pip install onad[all]
```

## Virtual Environment Setup

We recommend using a virtual environment to avoid dependency conflicts:

=== "conda"
    ```bash
    conda create -n onad python=3.10
    conda activate onad
    pip install onad[all]
    ```

=== "venv"
    ```bash
    python -m venv onad-env
    source onad-env/bin/activate  # On Windows: onad-env\Scripts\activate
    pip install onad[all]
    ```

=== "virtualenv"
    ```bash
    virtualenv onad-env
    source onad-env/bin/activate  # On Windows: onad-env\Scripts\activate
    pip install onad[all]
    ```

## Development Installation

If you want to contribute to ONAD or use the latest development version:

```bash
git clone https://github.com/OliverHennhoefer/onad.git
cd onad
pip install -e .[dev]
```

The `-e` flag installs in "editable" mode, so changes to the code take effect immediately.

## Verify Installation

Test your installation:

```python
import onad

print(f"ONAD version: {onad.__version__}")

# Test a simple model
from onad.model.iforest import OnlineIsolationForest

model = OnlineIsolationForest()
print("✅ Installation successful!")
```

## Common Issues

### ImportError: No module named 'faiss'

**Problem**: FAISS is not installed or incompatible with your system.

**Solution**: 
```bash
# Try installing FAISS separately
pip install faiss-cpu
# Or for GPU support (if available)
pip install faiss-gpu
```

### Memory Issues with Large Datasets

**Problem**: Out of memory errors when processing large streams.

**Solution**: Configure memory limits in your models:
```python
model = OnlineIsolationForest(window_size=1000)  # Limit window size
```

### Type Checking Errors

**Problem**: MyPy reports type errors in your code.

**Solution**: Ensure you're using Python 3.10+ and install type stubs:
```bash
pip install types-tqdm
```

## System Requirements

### Minimum Requirements
- **CPU**: Any modern processor
- **RAM**: 512 MB
- **Disk**: 100 MB free space
- **Python**: 3.10+

### Recommended Requirements
- **CPU**: Multi-core processor for parallel processing
- **RAM**: 2+ GB for large-scale streaming
- **Disk**: 1+ GB for datasets and models
- **Python**: 3.11+ for better performance

## Next Steps

After installation:

1. **[Quick Start](quickstart.md)**: Get started with your first anomaly detection model
2. **[User Guide](user_guide/index.md)**: Learn about ONAD's features in detail
3. **[Examples](examples/index.md)**: Explore real-world use cases

---

!!! question "Need Help?"
    If you encounter installation issues, please:
    
    1. Check the [troubleshooting section](user_guide/troubleshooting.md)
    2. Search [existing issues](https://github.com/OliverHennhoefer/onad/issues)
    3. Create a [new issue](https://github.com/OliverHennhoefer/onad/issues/new) with your system details