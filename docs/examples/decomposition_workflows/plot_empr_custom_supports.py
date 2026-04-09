"""
EMPR with Custom Supports
=========================

Run EMPR with user-defined support vectors and compare the reconstruction with
the input tensor.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import EMPR

x = np.linspace(0.0, 1.0, 32)
y = np.linspace(0.0, 1.0, 32)

X = (
    0.5
    + np.sin(np.pi * x)[:, None]
    + np.cos(np.pi * y)[None, :]
    + 0.25 * np.outer(x, y)
)

support_x = np.linspace(0.5, 1.5, 32).reshape(-1, 1)
support_y = np.linspace(1.5, 0.5, 32).reshape(-1, 1)

empr = EMPR(
    X,
    order=2,
    supports="custom",
    custom_supports=[support_x, support_y],
)

X_reconstructed = empr.reconstruct()
error = np.abs(X - X_reconstructed)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im0 = axes[0].imshow(X, aspect="auto")
axes[0].set_title("Original tensor")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(X_reconstructed, aspect="auto")
axes[1].set_title("EMPR reconstruction")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(error, aspect="auto")
axes[2].set_title("Absolute error")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

print("Support shapes:", support_x.shape, support_y.shape)
print("Available component keys:", list(empr.components().keys()))
