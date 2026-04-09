"""
EMPR on a 2D Tensor
===================

Run EMPR on a small 2D tensor, reconstruct the approximation, and compare the
reconstructed tensor with the original input.
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

empr = EMPR(X, order=2)
X_reconstructed = empr.reconstruct()
components = empr.components()

print("Input shape:", X.shape)
print("Reconstructed shape:", X_reconstructed.shape)
print("Available component keys:", list(components.keys()))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

im0 = axes[0].imshow(X, aspect="auto")
axes[0].set_title("Original tensor")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(X_reconstructed, aspect="auto")
axes[1].set_title("EMPR reconstruction")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
