"""Build a 3x3 mosaic of the 9 generated signature figures (just like the
README grid) with a dataset label under each panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultsw6"

# (filename, label) in the same layout order as the README grid
CELLS = [
    ("fig_fbm_h=0.1_n=2^15.png", "fBm H=0.1"),
    ("fig_fbm_h=0.5_n=2^15.png", "fBm H=0.5"),
    ("fig_fbm_h=0.9_n=2^15.png", "fBm H=0.9"),
    ("fig_henonmap.png", "Henon map"),
    ("fig_logisticmap.png", "Logistic map"),
    ("fig_skewtentmap.png", "Skew tent map"),
    ("fig_gaussian_2^15.png", "Gaussian"),
    ("fig_uniform_2^15.png", "Uniform"),
    ("fig_aami31a.png", "AAMI 31a (ECG)"),
]

fig = plt.figure(figsize=(15, 15))
gs = GridSpec(3, 3, wspace=0.05, hspace=0.12)

for idx, (fname, label) in enumerate(CELLS):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.imshow(mpimg.imread(OUT / fname))
    ax.axis("off")
    ax.set_title(label, fontsize=14, pad=4)

fig.savefig(OUT / "fig_mosaic_3x3.png", dpi=150, bbox_inches="tight")
# also JPEG version for README (like the original fig_*.jpg grid)
fig.savefig(OUT / "fig_mosaic_3x3.jpg", dpi=150, bbox_inches="tight")
print("mosaico salvo: resultsw6/fig_mosaic_3x3.png /.jpg")