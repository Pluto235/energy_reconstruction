#!/usr/bin/env python3
"""Redraw v6 report figures with 84-cell display IDs without changing data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-d-npz", type=Path, required=True)
    parser.add_argument("--stage-f-npz", type=Path, required=True)
    parser.add_argument("--stage-f-metadata", type=Path, required=True)
    parser.add_argument("--stage-d-output-dir", type=Path, required=True)
    parser.add_argument("--stage-f-output-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    return parser.parse_args()


def interval_key(label: str) -> float:
    if label.startswith("<"):
        return -np.inf
    if label.startswith(">="):
        return np.inf
    return float(label.split(",", 1)[0].lstrip("["))


def display_cell_id(cell_id: int, pred_bin: str) -> int | None:
    if pred_bin.strip().startswith(">="):
        return None
    return int(cell_id) - ((int(cell_id) - 1) // 13)


def plot_counts_grid(
    counts: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    xy_edges: np.ndarray,
    r_opt_deg: np.ndarray,
    output_path: Path,
    *,
    roi_fiducial_deg: float,
) -> None:
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(2.05 * len(ordered_pred), 1.75 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    logged = np.full(counts.shape, np.nan, dtype=np.float64)
    positive = counts > 0
    logged[positive] = np.log10(counts[positive])
    finite = logged[np.isfinite(logged)]
    vmin = float(np.percentile(finite, 5.0)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.0, 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    fiducial_x = roi_fiducial_deg * np.cos(theta)
    fiducial_y = roi_fiducial_deg * np.sin(theta)
    first_image = None

    for i, nhit in enumerate(ordered_nhit):
        for j, pred in enumerate(ordered_pred):
            ax = axes[i, j]
            idx = index_by_key.get((nhit, pred))
            if idx is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(
                logged[idx],
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if first_image is None:
                first_image = image
            ax.plot(fiducial_x, fiducial_y, color="white", linewidth=0.35, alpha=0.85)
            on_radius = float(r_opt_deg[idx])
            ax.plot(on_radius * np.cos(theta), on_radius * np.sin(theta), color="white", linewidth=0.45, alpha=0.9)
            ax.scatter([0.0], [0.0], marker="+", s=18, c="white", linewidths=0.6)
            shown = display_cell_id(int(cell_ids[idx]), pred)
            ax.set_title(f"cell {shown}: {pred}" if shown is not None else pred, fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit}\ny (deg)", fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)

    fig.suptitle("Stage D observed counts skymap before background subtraction", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_image is not None:
        colorbar = fig.colorbar(first_image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        colorbar.set_label("log10 observed counts", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_roi_grid(
    values: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    xy_edges: np.ndarray,
    r_opt_deg: np.ndarray,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    roi_fiducial_deg: float,
) -> None:
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(2.05 * len(ordered_pred), 1.75 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    fiducial_x = roi_fiducial_deg * np.cos(theta)
    fiducial_y = roi_fiducial_deg * np.sin(theta)
    first_image = None

    for i, nhit in enumerate(ordered_nhit):
        for j, pred in enumerate(ordered_pred):
            ax = axes[i, j]
            idx = index_by_key.get((nhit, pred))
            if idx is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(
                values[idx],
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            if first_image is None:
                first_image = image
            ax.plot(fiducial_x, fiducial_y, color="#222222", linewidth=0.35, alpha=0.8)
            on_radius = float(r_opt_deg[idx])
            ax.plot(on_radius * np.cos(theta), on_radius * np.sin(theta), color="#111111", linewidth=0.45, alpha=0.9)
            ax.scatter([0.0], [0.0], marker="+", s=18, c="#111111", linewidths=0.6)
            shown = display_cell_id(int(cell_ids[idx]), pred)
            ax.set_title(f"cell {shown}: {pred}" if shown is not None else pred, fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit}\ny (deg)", fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_image is not None:
        colorbar = fig.colorbar(first_image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_counts(stage_f: dict[str, np.ndarray], preferred_model: str, output_path: Path) -> None:
    cell_ids = stage_f["cell_id"]
    pred_bins = stage_f["predE_bin"].astype(str)
    labels = [str(display_cell_id(int(cell_id), pred)) for cell_id, pred in zip(cell_ids, pred_bins)]
    x = np.arange(cell_ids.size, dtype=np.float64) + 1.0
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.errorbar(
        x,
        stage_f["excess"],
        yerr=stage_f["excess_err_conservative"],
        fmt="o",
        color="#222222",
        markersize=4,
        label="Stage E excess",
    )
    ax.plot(x, stage_f["pl_conservative_model_counts"], "-o", color="#1f77b4", markersize=3, label="PL model")
    ax.plot(x, stage_f["logpar_conservative_model_counts"], "-o", color="#d62728", markersize=3, label="LogPar model")
    ax.set_xticks(x, labels)
    ax.set_xlabel("display cell ID (predE >= 6 tail excluded)")
    ax.set_ylabel("counts")
    ax.set_title(f"Stage F model counts vs excess; preferred={preferred_model}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def copy_to_assets(paths: list[Path], asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, asset_dir / path.name)


def main() -> None:
    args = parse_args()
    args.stage_d_output_dir.mkdir(parents=True, exist_ok=True)
    args.stage_f_output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.stage_d_npz, allow_pickle=False) as data:
        stage_d = {key: np.asarray(data[key]) for key in data.files}
    with np.load(args.stage_f_npz, allow_pickle=False) as data:
        stage_f = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads(args.stage_f_metadata.read_text(encoding="utf-8"))
    preferred_model = str((metadata.get("preferred_fit") or {}).get("model", "unknown"))
    roi_fiducial_deg = float(np.max(np.abs(stage_d["x_edges_deg"])))

    roi_counts = args.stage_d_output_dir / "roi_counts_grid.png"
    roi_excess = args.stage_d_output_dir / "roi_excess_grid.png"
    annulus_residual = args.stage_d_output_dir / "annulus_residual_grid.png"
    model_counts = args.stage_f_output_dir / "model_counts_vs_excess.png"
    common = (
        stage_d["cell_id"],
        stage_d["nhit_bin"].astype(str),
        stage_d["predE_bin"].astype(str),
        stage_d["x_edges_deg"],
        stage_d["r_opt_deg"],
    )
    plot_counts_grid(
        stage_d["counts_map"],
        *common,
        roi_counts,
        roi_fiducial_deg=roi_fiducial_deg,
    )
    plot_roi_grid(
        stage_d["excess_map"],
        *common,
        roi_excess,
        title="Stage D ROI-local counts minus annulus quadratic surface background",
        colorbar_label="counts - background",
        roi_fiducial_deg=roi_fiducial_deg,
    )
    plot_roi_grid(
        stage_d["annulus_residual_map"],
        *common,
        annulus_residual,
        title="Stage D annulus fit residuals",
        colorbar_label="annulus residual sigma",
        roi_fiducial_deg=roi_fiducial_deg,
    )
    plot_model_counts(stage_f, preferred_model, model_counts)
    copy_to_assets([roi_counts, roi_excess, annulus_residual, model_counts], args.asset_dir)
    print("Redrew display-only cell labels: 1-84; predE >= 6 tail unnumbered")


if __name__ == "__main__":
    main()
