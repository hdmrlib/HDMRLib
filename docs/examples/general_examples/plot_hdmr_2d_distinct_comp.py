"""
HDMR on a 2D Tensor
===================

Run HDMR on a structured 2D tensor and inspect the resulting component terms.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import HDMR


def to_display_array(component):
    arr = np.asarray(component)

    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr

    return arr.reshape(arr.shape[0], -1)


# Build a tensor whose components are visually distinct.
x = np.linspace(-1.0, 1.0, 64)
y = np.linspace(-1.0, 1.0, 64)

X = (
    0.3
    + 1.2 * x[:, None]
    - 0.8 * y[None, :]
    + 1.5 * np.outer(x, y)
    + 0.3 * np.sin(3 * np.pi * x)[:, None]
    + 0.2 * np.cos(2 * np.pi * y)[None, :]
)

hdmr = HDMR(X, order=2)
components = hdmr.components()

print("Input shape:", X.shape)
print("Available component keys:", list(components.keys()))

g1 = to_display_array(components["g_1"])
g2 = to_display_array(components["g_2"])
g12 = to_display_array(components["g_1,2"])

# Use a shared symmetric color scale for the components.
component_max = max(
    np.abs(g1).max(),
    np.abs(g2).max(),
    np.abs(g12).max(),
)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

im0 = axes[0].imshow(X, aspect="auto", cmap="viridis")
axes[0].set_title("Original tensor")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(
    g1, aspect="auto", cmap="coolwarm", vmin=-component_max, vmax=component_max
)
axes[1].set_title("g_1")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(
    g2, aspect="auto", cmap="coolwarm", vmin=-component_max, vmax=component_max
)
axes[2].set_title("g_2")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

im3 = axes[3].imshow(
    g12, aspect="auto", cmap="coolwarm", vmin=-component_max, vmax=component_max
)
axes[3].set_title("g_1,2")
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
