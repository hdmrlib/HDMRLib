# Installing HDMR-Lib

## Prerequisites

- **Python**: 3.8 or higher

## Quick Install

The easiest way to install HDMR-Lib is from PyPI:

```bash
pip install hdmrlib
```

This installs the latest stable version with NumPy as the default backend.

## Cloning the github repository

Use this option if you want to work with the codebase directly (e.g., to experiment with the latest changes or to contribute).

Clone the repository and move into it:
```bash
git clone https://github.com/hdmrlib/HDMR-Lib.git
cd HDMR-Lib
```

## Optional Backends

HDMR-Lib supports optional compute backends for enhanced performance and functionality. If you do not install them, only the NumPy backend will be available.

### PyTorch Backend

PyTorch provides GPU acceleration and advanced tensor operations.

**Prerequisites**: Install PyTorch first by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your OS and CUDA version.

**Installation**:

```bash
pip install hdmrlib[torch]
```

### TensorFlow Backend

TensorFlow offers GPU support and is optimized for production environments.

**Prerequisites**: Install TensorFlow first by following the [official TensorFlow installation guide](https://www.tensorflow.org/install) for your setup (CPU or GPU).

**Installation**:

```bash
pip install hdmrlib[tensorflow]
```

### All optional backends

```bash
pip install hdmrlib[all]
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


### Development Installation

For developers who want to contribute or use the latest development version:

```bash
git clone https://github.com/hdmrlib/HDMR-Lib.git
cd HDMR-Lib
pip install -e ".[dev]"
```

## Troubleshooting

### Backend not available

If you get an error that a backend is not available:
1. Verify the backend is installed: `pip list | grep torch` or `pip list | grep tensorflow`
2. Reinstall the requirements file: `pip install -r requirements-{backend}.txt`
3. Restart your Python session or notebook kernel

### Installation issues

For common installation issues and solutions, please check our [GitHub Issues](https://github.com/hdmrlib/HDMR-Lib/issues) or open a new issue with details about your system and error message.

