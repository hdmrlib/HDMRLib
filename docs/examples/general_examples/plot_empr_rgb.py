from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

from hdmrlib import EMPR


candidates = [
    Path(os.environ.get("GITHUB_WORKSPACE", "")) / "docs" / "_static" / "examples" / "squirell.png",
    Path("docs/_static/examples/squirell.png"),
    Path("_static/examples/squirell.png"),
]

image_path = None
for candidate in candidates:
    if str(candidate) and candidate.exists():
        image_path = candidate
        break

if image_path is None:
    raise FileNotFoundError(
        "Could not find squirell.png. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )

# Load with PIL and resize properly
img = Image.open(image_path).convert("RGB")

max_size = 96
img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

X = np.asarray(img, dtype=np.float64) / 255.0

empr = EMPR(X, order=3)
X_reconstructed = np.asarray(empr.reconstruct(), dtype=np.float64)

X_display = np.clip(X, 0.0, 1.0)
X_reconstructed_display = np.clip(X_reconstructed, 0.0, 1.0)

error_map = np.mean(np.abs(X_display - X_reconstructed_display), axis=2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(X_display, interpolation="bilinear")
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(X_reconstructed_display, interpolation="bilinear")
axes[1].set_title("EMPR reconstruction (order=3)")
axes[1].axis("off")

im = axes[2].imshow(error_map, cmap="viridis")
axes[2].set_title("Mean absolute error")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()