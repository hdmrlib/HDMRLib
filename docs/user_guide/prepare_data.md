# Prepare Input Data

## Provide a Numeric Tensor

`HDMR` and `EMPR` both take the input tensor as the first argument.

```python
import numpy as np

X = np.random.rand(10, 10)
```

Use dense numeric data for the input tensor.

## Shape Defines the Decomposition Structure

The shape of the input tensor determines the dimensional structure of the decomposition.

```python
print(X.shape)
```

Examples:

- `(10,)` for one-dimensional data
- `(10, 10)` for two-dimensional data
- `(10, 10, 10)` for three-dimensional data

## Use Small Tensors First

Start with small tensors when testing a new workflow.

```python
X = np.random.rand(5, 5)
```

This makes it easier to inspect outputs and catch shape-related issues early.

## Backend Conversion

The active backend converts the input internally:

- NumPy backend converts the input to a NumPy array
- PyTorch backend converts the input to a Torch tensor
- TensorFlow backend converts the input to a TensorFlow tensor

In all three backends, the internal tensor representation uses `float64`.

## Singleton Dimensions

Singleton dimensions are squeezed by the backend implementation.

For example, an input with shape `(1, 10, 10)` is treated as `(10, 10)` after conversion.

If you need a specific dimensional structure, check the shape before running the decomposition.

## Custom Supports and Weights

If you use custom supports or custom weights later in the decomposition workflow, they must match the number of input dimensions.

For example, a three-dimensional tensor requires three support vectors. HDMR custom weights also follow the same per-dimension rule.

## Before You Run a Decomposition

Check that:

- the input is numeric
- the shape matches the structure you want to analyze
- singleton dimensions are intentional
- the tensor size is reasonable for the decomposition order you plan to use

## Next

- **Run a Decomposition** shows how to create EMPR and HDMR decompositions
- **Inspect Components** explains how to work with decomposition outputs
- **Reconstruct Data** shows how to reconstruct lower-order approximations
