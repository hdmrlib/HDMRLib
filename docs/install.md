# Installation

## Prerequisites

- Python 3.8 or higher

---

## Install from PyPI

Install the base package with NumPy backend:

```bash
pip install hdmrlib
```

---

## Optional Backends

HDMR-Lib supports multiple computational backends. You can install them via extras:

### PyTorch backend

```bash
pip install hdmrlib[torch]
```

### TensorFlow backend

```bash
pip install hdmrlib[tensorflow]
```

### All backends

```bash
pip install hdmrlib[all]
```

---

## Install from Source

Use this option if you want to work with the latest version of the codebase:

```bash
git clone https://github.com/hdmrlib/HDMR-Lib.git
cd HDMR-Lib
pip install -e .
```

---

## Development Installation

For development and contributing:

```bash
pip install -e ".[dev]"
```

---

## Verify Installation

You can verify the installation and backend setup:

```python
from hdmrlib import EMPR

# Create sample data
import numpy as np
X = np.random.rand(10, 10)

# Run decomposition
empr = EMPR(X, order=2)
components = empr.components()

print(type(components))
```

---

## Backend Selection

You can explicitly set the backend:

```python
from hdmrlib.backend import set_backend

set_backend("numpy")      # default
set_backend("torch")      # requires PyTorch
set_backend("tensorflow") # requires TensorFlow
```

If a backend is not installed, an error will be raised.

---

## Troubleshooting

### Backend not found

Ensure the backend is installed:

```bash
pip install hdmrlib[torch]
```

or

```bash
pip install hdmrlib[tensorflow]
```

---

### Import errors

Check installation:

```bash
pip show hdmrlib
```

If needed, reinstall:

```bash
pip install --upgrade --force-reinstall hdmrlib
```
