#!/usr/bin/env python3
"""Plot normalized v6 true-energy distributions for the selected 2D cells."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-npz", type=Path, required=True)
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v6_64748_true_energy")
    return parser.parse_args()


def interval_key(label: str) -> float:
    if label.startswith("<"):
        return -np.inf
    if label.startswith(">="):
        return np.inf
    return float(label.split(",", 1)[0].lstrip("["))


def load_selected_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("include", "1")) == 1]
    if not rows:
        raise ValueError(f"No included cells found in {path}")
    return rows


def normalized_density(hist: np.ndarray, widths: np.ndarray) -> np.ndarray:
    area = float(np.sum(hist * widths))
    if not np.isfinite(area) or area <= 0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist / area


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"))
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    response_path = args.response_npz.resolve()
    selector_path = args.selector_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_selected_rows(selector_path)
    with np.load(response_path) as response:
        cell_ids = np.asarray(response["cell_id"], dtype=np.int64)
        loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
        numerator = np.asarray(response["numerator_sumw"], dtype=np.float64)

    if numerator.ndim != 3 or numerator.shape[0] != cell_ids.size:
        raise ValueError(
            "Expected numerator_sumw with shape (cell, true-energy, theta); "
            f"got {numerator.shape} for {cell_ids.size} cells"
        )
    if numerator.shape[1] != loge_edges.size - 1:
        raise ValueError("True-energy axis does not match logE_true_edges")

    id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    missing = sorted(int(row["cell_id"]) for row in rows if int(row["cell_id"]) not in id_to_index)
    if missing:
        raise ValueError(f"Selected cell IDs absent from response: {missing}")

    histograms: dict[tuple[str, str], np.ndarray] = {}
    pred_totals: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(numerator.shape[1], dtype=np.float64))
    for row in rows:
        key = (row["nhit_bin"], row["predE_bin"])
        hist = numerator[id_to_index[int(row["cell_id"])]].sum(axis=1)
        histograms[key] = hist
        pred_totals[row["predE_bin"]] += hist

    nhit_bins = sorted({row["nhit_bin"] for row in rows}, key=interval_key)
    pred_bins = sorted({row["predE_bin"] for row in rows}, key=interval_key)
    centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    widths = np.diff(loge_edges)
    colors = {pred: plt.get_cmap("viridis")(i / max(1, len(pred_bins) - 1)) for i, pred in enumerate(pred_bins)}

    configure_plotting()

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for pred_bin in pred_bins:
        density = normalized_density(pred_totals[pred_bin], widths)
        if np.any(density):
            ax.plot(centers, density, color=colors[pred_bin], label=pred_bin)
    ax.set_xlabel(r"$\log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$")
    ax.set_ylabel("Probability density")
    ax.set_title("v6 selected cells: normalized true-energy distributions")
    ax.legend(title=r"$\log_{10}(E_{\mathrm{pred}}/\mathrm{GeV})$ bin", ncol=2, frameon=False)
    fig.tight_layout()
    overlay_base = output_dir / f"{args.prefix}_predE_overlay"
    save_figure(fig, overlay_base)

    fig, axes = plt.subplots(4, 2, figsize=(12.0, 14.0), sharex=True, sharey=True, squeeze=False)
    for ax, nhit_bin in zip(axes.flat, nhit_bins):
        for pred_bin in pred_bins:
            hist = histograms.get((nhit_bin, pred_bin))
            if hist is None:
                continue
            density = normalized_density(hist, widths)
            if np.any(density):
                ax.plot(centers, density, color=colors[pred_bin], label=pred_bin)
        ax.set_title(f"Nhit {nhit_bin}")
        ax.set_ylabel("Probability density")
        ax.legend(title="PredE bin", ncol=2, frameon=False, fontsize=7, title_fontsize=8)
    axes.flat[-1].axis("off")
    fig.supxlabel(r"$\log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$", fontsize=11)
    fig.suptitle("v6 selected 2D bins: normalized true-energy distributions", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    grid_base = output_dir / f"{args.prefix}_nhit_grid"
    save_figure(fig, grid_base)

    metadata = {
        "response_npz": str(response_path),
        "selector_csv": str(selector_path),
        "normalization": "unit integral in log10(E_true/GeV) for each curve",
        "response_quantity": "numerator_sumw summed over theta bins",
        "included_cells": len(rows),
        "nhit_bins": nhit_bins,
        "predE_bins": pred_bins,
        "outputs": {
            "predE_overlay_png": str(overlay_base.with_suffix(".png")),
            "predE_overlay_pdf": str(overlay_base.with_suffix(".pdf")),
            "nhit_grid_png": str(grid_base.with_suffix(".png")),
            "nhit_grid_pdf": str(grid_base.with_suffix(".pdf")),
        },
    }
    metadata_path = output_dir / f"{args.prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
