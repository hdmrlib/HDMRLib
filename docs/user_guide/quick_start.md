# Quick Start

This page introduces the basic HDMR-Lib workflow: prepare input data, run a decomposition, inspect component terms, and reconstruct lower-order approximations.

HDMR-Lib supports two related decomposition formulations:

- **EMPR** is a convenient starting point when you want a direct workflow for decomposition and reconstruction
- **HDMR** follows the same high-level interface and is useful when working explicitly with HDMR-based formulations

In both cases, the workflow is the same:

**input data** → **decomposition** → **components** → **reconstruction**

## End-to-end example with EMPR

```python
import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)

X_reconstructed = empr.reconstruct()
components = empr.components()
```

This computes an EMPR decomposition of `X` up to order 2.

- `empr.reconstruct()` returns a reconstruction based on the computed decomposition
- `empr.components()` returns the extracted component terms

## Understanding the outputs

The `components()` method returns the component functions produced by the decomposition. These are typically organized by term name, such as lower-order individual terms and interaction terms.

For example, a call such as:

```python
selected = empr.components(elements=["g_0", "g_1", "g_2", "g_1,2"])
```

returns only the requested components.

This is useful when you want to inspect specific lower-order terms or selected interactions rather than the full decomposition output.

## Reconstructing lower-order approximations

```python
X_order1 = empr.reconstruct(order=1)
```

This reconstructs `X` using only terms up to order 1.

Use `reconstruct(order=...)` when you want to study how much of the data can be approximated using only lower-order structure. This is especially useful when comparing simplified approximations against the full reconstruction.

## Running the same workflow with HDMR

```python
import numpy as np
from hdmrlib import HDMR

X = np.random.rand(10, 10)

hdmr = HDMR(X, order=2)

X_reconstructed = hdmr.reconstruct()
components = hdmr.components()
```

The same workflow applies to HDMR: compute a decomposition, inspect the extracted components, and reconstruct the data from the resulting representation.

## Choosing between EMPR and HDMR

As a starting point:

- use **EMPR** when you want a simple entry point for decomposition and reconstruction
- use **HDMR** when you want to work specifically with HDMR-based formulations

Since both follow a consistent interface, it is easy to compare them within the same workflow.

