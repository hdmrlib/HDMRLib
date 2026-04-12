"""
HDMR on a 2D Tensor
===================

Run HDMR on a structured 2D tensor and inspect the resulting additive and
interaction terms.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import HDMR


def to_scalar(component):
    arr = np.asarray(component)
    return float(np.squeeze(arr))


def to_vector(component):
    arr = np.asarray(component)
    arr = np.squeeze(arr)

    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D component, got shape {arr.shape!r}")

    return arr


def to_matrix(component):
    arr = np.asarray(component)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D component, got shape {arr.shape!r}")

    return arr


# Build a tensor with clearly visible first-order and interaction structure.
x = np.linspace(-1.0, 1.0, 64)
y = np.linspace(-1.0, 1.0, 64)

X = (
    0.2
    + 0.9 * np.sin(np.pi * x)[:, None]
    + 0.7 * np.cos(np.pi * y)[None, :]
    + 1.2 * np.outer(x, y)
)

hdmr = HDMR(X, order=2)
components = hdmr.components()

print("Input shape:", X.shape)
print("Available component keys:", list(components.keys()))

g0 = to_scalar(components["g_0"])
g1 = to_vector(components["g_1"])
g2 = to_vector(components["g_2"])
g12 = to_matrix(components["g_1,2"])

additive = g0 + g1[:, None] + g2[None, :]

interaction_max = np.max(np.abs(g12))

fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(X, aspect="auto", cmap="viridis")
ax1.set_title("Original tensor")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(additive, aspect="auto", cmap="viridis")
ax2.set_title("Additive part: g_0 + g_1 + g_2")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(
    g12,
    aspect="auto",
    cmap="coolwarm",
    vmin=-interaction_max,
    vmax=interaction_max,
)
ax3.set_title("Interaction term: g_1,2")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

ax4 = plt.subplot(2, 3, 4)
ax4.plot(x, g1)
ax4.set_title("First-order term g_1")
ax4.set_xlabel("x")
ax4.set_ylabel("Contribution")
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
ax5.plot(y, g2)
ax5.set_title("First-order term g_2")
ax5.set_xlabel("y")
ax5.set_ylabel("Contribution")
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
residual = X - additive
im6 = ax6.imshow(residual, aspect="auto", cmap="coolwarm")
ax6.set_title("Residual after additive part")
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()