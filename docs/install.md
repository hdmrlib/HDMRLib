# Installation

## Prerequisites

Before installing HDMR-Lib, ensure you have the following installed:

- **Python**: 3.8 or higher
- **pip**: Package installer for Python (usually comes with Python)
- **git**: Required for development installation

### System Requirements

- RAM: Minimum 2GB (more recommended for large datasets)
- Storage: ~500MB for base installation + optional backends
- OS: Linux, macOS, or Windows

## Quick Install

The easiest way to install HDMR-Lib is from PyPI:

```bash
pip install hdmrlib
```

## Installation Options

### Stable Release from PyPI

```bash
pip install hdmrlib
```

This installs the latest stable version with NumPy as the default backend.

### Development Installation

For developers who want to contribute or use the latest development version:

```bash
git clone https://github.com/hdmrlib/HDMR-Lib.git
cd HDMR-Lib
pip install -e ".[dev]"
```

If your project does not define extras yet, use:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Optional Backends

HDMR-Lib supports optional compute backends for enhanced performance and functionality. If you do not install them, only the NumPy backend will be available.

### PyTorch Backend

PyTorch provides GPU acceleration and advanced tensor operations.

**Prerequisites**: Install PyTorch first by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your OS and CUDA version.

**Installation**:

```bash
pip install -r requirements-torch.txt
```

### TensorFlow Backend

TensorFlow offers GPU support and is optimized for production environments.

**Prerequisites**: Install TensorFlow first by following the [official TensorFlow installation guide](https://www.tensorflow.org/install) for your setup (CPU or GPU).

**Installation**:

```bash
pip install -r requirements-tensorflow.txt
```

## Backend Selection

You can dynamically select which backend to use. Here's a quick example:

```python
from hdmrlib.backends import set_backend, get_backend

# Set backend
set_backend("numpy")   # or "torch" / "tensorflow" if installed

# Verify the selected backend
print(get_backend())
```

**Supported backends**:
- "numpy" (default) - Standard backend, always available
- "torch" - Available after installing PyTorch backend
- "tensorflow" - Available after installing TensorFlow backend

## Troubleshooting

### Backend not available

If you get an error that a backend is not available:
1. Verify the backend is installed: `pip list | grep torch` or `pip list | grep tensorflow`
2. Reinstall the requirements file: `pip install -r requirements-{backend}.txt`
3. Restart your Python session or notebook kernel

### Installation issues

For common installation issues and solutions, please check our [GitHub Issues](https://github.com/hdmrlib/HDMR-Lib/issues) or open a new issue with details about your system and error message.