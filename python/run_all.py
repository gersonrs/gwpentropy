"""Generate H/C/q signatures and 3D plots for every dataset (word size w=6).

Mirrors the R reference script (R/gwpe.R) and the original example
(python/gwpe.py): for q in [-10, 10] step 0.1 it computes the generalized
weighted permutation entropy H and complexity C, saves the signature as a
text table (same layout as resultsw6/*_gpentpy.txt) and renders the
complexity-entropy-scale 3D plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: save figures, never show windows

import matplotlib.pyplot as plt
import numpy as np

import gwpe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "resultsw6"
SIG_DIR = Path(__file__).resolve().parent / "results"  # regenerated .txt signatures

W = 6        # word size
QMIN = -10.0  # scaling parameter range
QMAX = 10.0
QDELTA = 0.1
# canonical resultsw6 files are printed with 6 decimals -> tol = 0.5*1e-6
REF_TOL = 5.1e-7

DATASETS = [
    "aami31a",
    "fbm_h=0.1_n=2^15",
    "fbm_h=0.5_n=2^15",
    "fbm_h=0.9_n=2^15",
    "gaussian_2^15",
    "henonmap",
    "logisticmap",
    "skewtentmap",
    "uniform_2^15",
]


def compute_hc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (H, C, q) signature curves, one point per q value."""
    n = x.shape[0]
    q = np.arange(QMIN, QMAX + QDELTA, QDELTA)
    h = np.empty_like(q)
    c = np.empty_like(q)
    for i, qi in enumerate(q):
        h[i], c[i] = gwpe.calc(x, n, W, float(qi))
    return h, c, q


def check_reference(name: str, h: np.ndarray, c: np.ndarray) -> None:
    """Compare against canonical resultsw6 files and report max deviation."""
    ref_path = OUT_DIR / f"{name}_gpentpy.txt"
    if not ref_path.exists():
        print(f"  [ref] {ref_path.name}: ausente — sem comparacao")
        return
    ref = np.loadtxt(ref_path, skiprows=1)  # skip 'H C q' header line
    ref_h, ref_c = ref[:, 0], ref[:, 1]
    if ref_h.shape != h.shape:
        print(f"  [ref] {ref_path.name}: shape {ref_h.shape} != {h.shape} [DIVERGE]")
        return
    dh = np.max(np.abs(ref_h - h))
    dc = np.max(np.abs(ref_c - c))
    status = "ok" if (dh <= REF_TOL and dc <= REF_TOL) else "DIVERGE"
    print(f"  [ref] {ref_path.name}: max|dH|={dh:.2e} max|dC|={dc:.2e} [{status}]")


def plot_signature(name: str, h: np.ndarray, c: np.ndarray, q: np.ndarray) -> None:
    """Save the 3D complexity-entropy-scale plot for one series."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(h, c, q, s=8, alpha=0.8)
    ax.set_xlabel("H")
    ax.set_ylabel("C")
    ax.set_zlabel("q")
    fig.savefig(OUT_DIR / f"fig_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    SIG_DIR.mkdir(exist_ok=True)
    for name in DATASETS:
        x = np.loadtxt(DATA_DIR / f"{name}.dat")
        h, c, q = compute_hc(x)
        check_reference(name, h, c)
        np.savetxt(
            SIG_DIR / f"{name}_gpentpy.txt",
            np.column_stack((h, c, q)),
            fmt="%.6f",
            header="H       C       q",
            comments="",
        )
        plot_signature(name, h, c, q)
        print(f"  [out] {name}: n={x.shape[0]} -> fig_{name}.png")


if __name__ == "__main__":
    main()