#!/usr/bin/env python3
"""Plot normalized true-energy distributions for every v6 2D candidate cell."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-npz", type=Path, required=True)
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v6_64748_reselect44_true_energy_cell_grid")
    parser.add_argument("--y-max", type=float, default=4.2)
    parser.add_argument("--force-include-cell-ids", type=int, nargs="*", default=[])
    parser.add_argument("--force-exclude-cell-ids", type=int, nargs="*", default=[])
    return parser.parse_args()


def interval_key(label: str) -> float:
    if label.startswith("<"):
        return -np.inf
    if label.startswith(">="):
        return np.inf
    return float(label.split(",", 1)[0].lstrip("["))


def histogram_quantiles(counts: np.ndarray, centers: np.ndarray) -> tuple[float, float, float]:
    total = float(np.sum(counts))
    if total <= 0:
        return np.nan, np.nan, np.nan
    cdf = np.cumsum(counts) / total
    return tuple(float(value) for value in np.interp([0.16, 0.50, 0.84], cdf, centers))


def fmt(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    response_path = args.response_npz.resolve()
    selector_path = args.selector_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with selector_path.open("r", encoding="utf-8") as handle:
        selector_rows = list(csv.DictReader(handle))
    selector_by_id = {int(row["cell_id"]): row for row in selector_rows}
    included_ids = {int(row["cell_id"]) for row in selector_rows if int(row["include"]) == 1}
    included_ids.update(args.force_include_cell_ids)
    included_ids.difference_update(args.force_exclude_cell_ids)

    with np.load(response_path) as response:
        cell_ids = np.asarray(response["cell_id"], dtype=np.int64)
        nhit_labels = np.asarray(response["nhit_bin"]).astype(str)
        pred_labels = np.asarray(response["predE_bin"]).astype(str)
        loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
        numerator_count = np.asarray(response["numerator_count"], dtype=np.float64)

    if numerator_count.ndim != 3 or numerator_count.shape[0] != cell_ids.size:
        raise ValueError("numerator_count must have shape (cell, true-energy, theta)")

    if cell_ids.size != 91:
        raise ValueError(f"Expected 91 cells including the >=6 tail, found {cell_ids.size}")

    centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    widths = np.diff(loge_edges)
    hist_by_id: dict[int, np.ndarray] = {}
    metrics_by_id: dict[int, dict[str, object]] = {}

    for idx, cell_id_raw in enumerate(cell_ids):
        cell_id = int(cell_id_raw)
        hist = numerator_count[idx].sum(axis=1)
        count = int(round(float(hist.sum())))
        density = hist / (count * widths) if count > 0 else np.zeros_like(hist)
        q16, q50, q84 = histogram_quantiles(hist, centers)
        sigma68 = 0.5 * (q84 - q16) if np.isfinite(q16) and np.isfinite(q84) else np.nan
        selector = selector_by_id[cell_id]
        hist_by_id[cell_id] = density
        metrics_by_id[cell_id] = {
            "cell_id": cell_id,
            "include": int(cell_id in included_ids),
            "selector_include": int(selector["include"]),
            "nhit_bin": nhit_labels[idx],
            "predE_bin": pred_labels[idx],
            "mc_count_selector": int(selector["mc_count"]),
            "response_count_in_true_range": count,
            "logE_true_q16": q16,
            "logE_true_q50": q50,
            "logE_true_q84": q84,
            "logE_true_sigma68": sigma68,
            "peak_density": float(np.max(density)) if density.size else 0.0,
        }

    nhit_bins = sorted({row["nhit_bin"] for row in metrics_by_id.values()}, key=interval_key)
    pred_bins = sorted({row["predE_bin"] for row in metrics_by_id.values()}, key=interval_key)
    cell_for_key = {
        (str(row["nhit_bin"]), str(row["predE_bin"])): cell_id
        for cell_id, row in metrics_by_id.items()
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.titlesize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    )

    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(27.0, 14.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    selected_line = "#0072B2"
    selected_fill = "#56B4E9"
    excluded_line = "#6B7280"
    excluded_fill = "#D1D5DB"

    for row_idx, nhit_bin in enumerate(nhit_bins):
        for col_idx, pred_bin in enumerate(pred_bins):
            ax = axes[row_idx, col_idx]
            cell_id = cell_for_key[(nhit_bin, pred_bin)]
            metric = metrics_by_id[cell_id]
            included = int(metric["include"]) == 1
            density = hist_by_id[cell_id]
            line_color = selected_line if included else excluded_line
            fill_color = selected_fill if included else excluded_fill
            ax.plot(centers, density, color=line_color, linewidth=1.35 if included else 0.9)
            ax.fill_between(centers, density, color=fill_color, alpha=0.28 if included else 0.20)
            ax.set_xlim(float(loge_edges[0]), float(loge_edges[-1]))
            ax.set_ylim(0.0, args.y_max)
            ax.set_xticks([2, 3, 4, 5, 6])
            ax.set_yticks([0, 2, 4])
            for spine in ax.spines.values():
                spine.set_color("#009E73" if included else "#9CA3AF")
                spine.set_linewidth(1.35 if included else 0.65)
            status = "FIT" if included else "OUT"
            ax.text(
                0.03,
                0.96,
                f"C{cell_id} {status}  n={int(metric['response_count_in_true_range']):,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.6,
                color="#065F46" if included else "#4B5563",
                fontweight="bold" if included else "normal",
            )
            ax.text(
                0.03,
                0.82,
                f"m={fmt(float(metric['logE_true_q50']))}  s68={fmt(float(metric['logE_true_sigma68']))}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.2,
                color="#374151",
            )
            if float(metric["peak_density"]) > args.y_max:
                ax.text(0.97, 0.96, "clipped", transform=ax.transAxes, ha="right", va="top", fontsize=5, color="#D55E00")
            if row_idx == 0:
                ax.set_title(pred_bin, pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"Nhit {nhit_bin}", fontsize=7.5)

    fig.suptitle(
        "v6 64748: normalized true-energy distributions for 91 cells (including PredE >= 6)",
        fontsize=15,
        y=0.997,
    )
    fig.supxlabel(r"$\log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$", fontsize=11)
    fig.supylabel("Probability density", fontsize=11)
    fig.legend(
        handles=[
            Line2D([0], [0], color=selected_line, lw=2, label="Included in final 44-cell fit (green border)"),
            Line2D([0], [0], color=excluded_line, lw=1.5, label="Candidate excluded from fit"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.018, 0.025, 1, 0.962), w_pad=0.35, h_pad=0.45)

    figure_base = output_dir / args.prefix
    fig.savefig(figure_base.with_suffix(".png"))
    fig.savefig(figure_base.with_suffix(".pdf"))
    plt.close(fig)

    metric_rows = [metrics_by_id[cell_id] for cell_id in sorted(metrics_by_id)]
    metrics_csv = output_dir / f"{args.prefix}_cell_metrics.csv"
    write_csv(metrics_csv, metric_rows)

    adjacent_rows: list[dict[str, object]] = []
    for nhit_bin in nhit_bins:
        for left_pred, right_pred in zip(pred_bins[:-1], pred_bins[1:]):
            left_id = cell_for_key[(nhit_bin, left_pred)]
            right_id = cell_for_key[(nhit_bin, right_pred)]
            left = metrics_by_id[left_id]
            right = metrics_by_id[right_id]
            if int(left["include"]) != 1 or int(right["include"]) != 1:
                continue
            overlap = float(np.sum(np.minimum(hist_by_id[left_id], hist_by_id[right_id]) * widths))
            adjacent_rows.append(
                {
                    "nhit_bin": nhit_bin,
                    "left_cell_id": left_id,
                    "left_predE_bin": left_pred,
                    "right_cell_id": right_id,
                    "right_predE_bin": right_pred,
                    "true_energy_overlap": overlap,
                    "median_separation_dex": float(right["logE_true_q50"]) - float(left["logE_true_q50"]),
                    "high_overlap_ge_0p5": int(overlap >= 0.5),
                }
            )
    overlap_csv = output_dir / f"{args.prefix}_adjacent_overlap.csv"
    write_csv(overlap_csv, adjacent_rows)

    overlaps = np.asarray([float(row["true_energy_overlap"]) for row in adjacent_rows], dtype=np.float64)
    monotonic_violations = []
    for nhit_bin in nhit_bins:
        usable = [
            metrics_by_id[cell_for_key[(nhit_bin, pred_bin)]]
            for pred_bin in pred_bins
            if int(metrics_by_id[cell_for_key[(nhit_bin, pred_bin)]]["response_count_in_true_range"]) >= 1000
        ]
        for left, right in zip(usable[:-1], usable[1:]):
            if float(right["logE_true_q50"]) <= float(left["logE_true_q50"]):
                monotonic_violations.append([int(left["cell_id"]), int(right["cell_id"])])

    summary = {
        "response_npz": str(response_path),
        "selector_csv": str(selector_path),
        "plot_selection_overrides": {
            "force_include_cell_ids": args.force_include_cell_ids,
            "force_exclude_cell_ids": args.force_exclude_cell_ids,
        },
        "distribution": "unweighted numerator_count summed over theta bins",
        "normalization": "unit integral in log10(E_true/GeV) for each cell",
        "candidate_cells": len(metric_rows),
        "included_fit_cells": sum(int(row["include"]) for row in metric_rows),
        "nonempty_cells": sum(int(row["response_count_in_true_range"]) > 0 for row in metric_rows),
        "adjacent_included_pairs": len(adjacent_rows),
        "adjacent_overlap_median": float(np.median(overlaps)) if overlaps.size else None,
        "adjacent_overlap_max": float(np.max(overlaps)) if overlaps.size else None,
        "adjacent_pairs_overlap_ge_0p5": int(np.sum(overlaps >= 0.5)),
        "median_monotonicity_violations_for_cells_with_n_ge_1000": monotonic_violations,
        "outputs": {
            "png": str(figure_base.with_suffix(".png")),
            "pdf": str(figure_base.with_suffix(".pdf")),
            "cell_metrics_csv": str(metrics_csv),
            "adjacent_overlap_csv": str(overlap_csv),
        },
    }
    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
