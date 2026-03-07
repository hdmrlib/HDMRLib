# Quick Start

A short introduction to HDMR-Lib demonstrating the minimal workflow for getting started.

## Basic usage pattern

A typical workflow in HDMR-Lib consists of the following steps:

1. prepare a tensor input
2. choose a computational backend
3. create an EMPR or HDMR model
4. reconstruct an approximation at a chosen order
5. inspect the resulting components

## Minimal example

```python
import numpy as np
import hdmrlib as h

X = np.random.rand(8, 8, 8)

model = h.EMPR(X, order=2)
approx = model.reconstruct()
components = model.components()
```

## What this example does

In this example:

- `X` is the input tensor
- `EMPR(X, order=2)` ccreates an EMPR model with a second-order approximation
- `reconstruct()` computes the approximation
- `components()` returns the available decomposition components

You can also override the approximation order at reconstruction time:

```python
approx_order_1 = model.reconstruct(order=1)
approx_order_2 = model.reconstruct(order=2)
```

You can retrieve only selected components by passing their keys:

```python
selected = model.components(elements=[(0,), (1,), (0, 1)])
```
HDMR-Lib supports multiple computational backends such as NumPy, PyTorch, and TensorFlow. Backend selection is explained in the **Computational Backends** section.

## Using HDMR instead

The same workflow can also be used with HDMR:

```python
import numpy as np
import hdmrlib as h

X = np.random.rand(8, 8, 8)

model = h.HDMR(X, order=2)
approx = model.reconstruct()
components = model.components()
```

