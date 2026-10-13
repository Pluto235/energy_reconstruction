#!/usr/bin/env python
"""Render the fitted 2D background-density surface for every cell as a grid.

Reconstructs each cell's fitted quadratic density q(x,y) = c0 + c1 x + c2 y +
c3 x^2 + c4 xy + c5 y^2 from the Stage-D ``surface_density_coefficients`` and
shows the relative deviation from the source-centered value, (q/q0 - 1) * 100 %,
over the source-centered ROI. This is the surface that sets B_on through its
radial trace tau = c_xx + c_yy (order-2 cells) or its tilt (order-1 cells).

Layout matches the report's (Nhit, predE) cell grids. Each panel is annotated
top-right with the cell id, the fitted surface order, and (order-2 only) tau.
The 7 order-2 SED-target cells get a red frame; the 6 order-2 donor-only cells
(pooled, sharing one surface) get an orange frame; order-1 planes are gray.

Per-panel colour autoscale (symmetric), so each panel shows its own shape; the
absolute deviation range is printed in each corner. Input is the small
coefficient NPZ dumped by ``extract_surf.py`` (or any Stage-D background NPZ).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ORDER2_TARGETS = {1, 2, 4, 5, 6, 19, 20}  # order-2 cells that enter the 44-cell SED fit
FIDUCIAL_DEG = 6.0


def evaluate_density(c: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="/tmp/surf_coeffs.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-points", type=int, default=140)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    cid = d["cell_id"].astype(int)
    nhit = np.asarray([str(v) for v in d["nhit_bin"]])
    predE = np.asarray([str(v) for v in d["predE_bin"]])
    order = d["surface_order"].astype(int)
    dens = d["surface_density_coefficients"].astype(float)
    shape = d["surface_shape_coefficients"].astype(float)
    inner = d["annulus_inner_deg"].astype(float)
    outer = d["annulus_outer_deg"].astype(float)
    r_opt = d["r_opt_deg"].astype(float)
    tau = shape[:, 3] + shape[:, 5]

    # (Nhit, predE) grid: high Nhit at top, predE increasing to the right.
    def low_edge(label: str) -> float:
        return float(label.lstrip("[(").split(",")[0])
    nhit_vals = sorted(np.unique(nhit).tolist(), key=low_edge, reverse=True)
    predE_vals = sorted(np.unique(predE).tolist(), key=low_edge)
    nrows, ncols = len(nhit_vals), len(predE_vals)
    row_of = {v: i for i, v in enumerate(nhit_vals)}
    col_of = {v: i for i, v in enumerate(predE_vals)}

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 1.9))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()

    for k in range(len(cid)):
        ax = axes[row_of[nhit[k]], col_of[predE[k]]]
        ax.set_axis_on()
        ax.set_xticks([]); ax.set_yticks([])
        R = min(float(outer[k]), FIDUCIAL_DEG)
        g = np.linspace(-R, R, args.grid_points)
        xx, yy = np.meshgrid(g, g)
        q = evaluate_density(dens[k], xx, yy)
        q0 = float(dens[k][0])
        dev = (q / q0 - 1.0) * 100.0 if q0 != 0 else np.zeros_like(q)
        rr = np.hypot(xx, yy)
        dev = np.where(rr <= R, dev, np.nan)          # circular window at the annulus outer edge
        vmax = float(np.nanmax(np.abs(dev)))
        vmax = max(vmax, 1e-6)
        ax.imshow(dev, origin="lower", extent=[-R, R, -R, R], cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="bilinear")
        # aperture + annulus overlays
        ax.add_patch(Circle((0, 0), float(r_opt[k]), fill=False, color="k", lw=0.8))
        ax.add_patch(Circle((0, 0), float(inner[k]), fill=False, color="0.35", lw=0.6, ls="--"))
        ax.add_patch(Circle((0, 0), min(float(outer[k]), R), fill=False, color="0.35", lw=0.6, ls="--"))
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")

        o = int(order[k])
        if o == 2 and cid[k] in ORDER2_TARGETS:
            edge, lw = "#d62728", 2.0
            label = f"#{cid[k]}  o2\n$\\tau$={tau[k]:+.4f}"
        elif o == 2:
            edge, lw = "#ff7f0e", 1.6
            label = f"#{cid[k]}  o2(d)\n$\\tau$={tau[k]:+.4f}"
        else:
            edge, lw = "0.6", 0.8
            label = f"#{cid[k]}  o{o}"
        for spine in ax.spines.values():
            spine.set_edgecolor(edge); spine.set_linewidth(lw)
        ax.text(0.97, 0.96, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=5.4, fontweight="bold", linespacing=0.95,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.text(0.03, 0.03, f"±{vmax:.2g}%", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=4.6, color="0.25")

    fig.suptitle(
        "Fitted background-density surfaces per cell  —  relative deviation (q/q$_0$$-$1) [%]  "
        "(Scheme R, double-Rayleigh, Poisson-unbinned)\n"
        "red = order-2 SED-target cell,  orange = order-2 pooled-donor cell,  gray = order-1 plane;  "
        "solid circle = ON aperture r$_{opt}$,  dashed = annulus [inner, outer].  "
        "Colour autoscaled per panel (range in lower-left).",
        fontsize=8.5, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out, "grid", (nrows, ncols), "cells", len(cid))


if __name__ == "__main__":
    main()
