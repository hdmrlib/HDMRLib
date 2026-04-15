"""
EMPR vs HDMR on a 2D Tensor
===========================

Run EMPR and HDMR on the same small 2D tensor, reconstruct both
approximations, and compare them with the original input.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import EMPR, HDMR


x = np.linspace(0.0, 1.0, 32)
y = np.linspace(0.0, 1.0, 32)

X = (
    0.5
    + np.sin(np.pi * x)[:, None]
    + np.cos(np.pi * y)[None, :]
    + 0.25 * np.outer(x, y)
)

empr = EMPR(X, order=2)
X_empr = np.asarray(empr.reconstruct(), dtype=np.float64)
empr_components = empr.components()

hdmr = HDMR(X, order=2)
X_hdmr = np.asarray(hdmr.reconstruct(), dtype=np.float64)
hdmr_components = hdmr.components()

empr_mae = float(np.mean(np.abs(X - X_empr)))
hdmr_mae = float(np.mean(np.abs(X - X_hdmr)))

print("Input shape:", X.shape)
print("EMPR reconstructed shape:", X_empr.shape)
print("HDMR reconstructed shape:", X_hdmr.shape)
print("EMPR component keys:", list(empr_components.keys()))
print("HDMR component keys:", list(hdmr_components.keys()))
print("EMPR mean absolute error:", empr_mae)
print("HDMR mean absolute error:", hdmr_mae)

vmin = min(X.min(), X_empr.min(), X_hdmr.min())
vmax = max(X.max(), X_empr.max(), X_hdmr.max())

fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
cax = fig.add_subplot(gs[0, 3])

im0 = ax0.imshow(X, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
ax0.set_title("Original tensor")

im1 = ax1.imshow(X_empr, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
ax1.set_title("EMPR reconstruction")

im2 = ax2.imshow(X_hdmr, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
ax2.set_title("HDMR reconstruction")

fig.colorbar(im2, cax=cax)

plt.show()