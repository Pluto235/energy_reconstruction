#!/usr/bin/env python3
"""Fit cells 1-3 raw MC with a full-sphere Rayleigh plus spherical-King PSF."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

from apply.report.plot_v6_cell123_raw_mc_survival import (  # noqa: E402
    _read_cell,
    double_king_survival,
    empirical_survival,
    git_sha,
    load_double_king_models,
    path_sha256,
    setup_matplotlib,
    weighted_quantiles,
)
from apply.stages.psf_rayleigh_king import (  # noqa: E402
    fit_rayleigh_king_counts,
    kl_divergence,
    profile_probability,
    rayleigh_king_cdf,
)


BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
SOURCE_RUN_ID = "v6_64748_nhit100_highEplus1_split56"
FERMI_RUN_ID = f"{BASE_RUN_ID}_fermi_double_king"
TARGET_CONTAINMENT = 1.0 - math.exp(-0.5 * 1.58**2)

DEFAULT_BINNED_ROOT = Path(
    "/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate"
)
DEFAULT_CELL_LEDGER = REPO_ROOT / "apply" / "config" / f"cell_ledger_{SOURCE_RUN_ID}_candidate.csv"
DEFAULT_SOURCE_PSF = (
    REPO_ROOT
    / "apply"
    / "output"
    / f"stage_b_{BASE_RUN_ID}"
    / "runs"
    / f"{BASE_RUN_ID}_stage_b_psf"
    / f"psf_{BASE_RUN_ID}.npz"
)
DEFAULT_SOURCE_METADATA = DEFAULT_SOURCE_PSF.with_name(f"psf_{BASE_RUN_ID}_metadata.json")
DEFAULT_DOUBLE_KING = (
    REPO_ROOT
    / "apply"
    / "output"
    / f"stage_b_{FERMI_RUN_ID}"
    / "runs"
    / FERMI_RUN_ID
    / f"psf_{FERMI_RUN_ID}.npz"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "apply"
    / "report"
    / "assets"
    / "v6-64748-nhit100-reselect44-split56-miss030-spherical-rayleigh-king-fit"
)

COLORS = {
    "data": "#0072B2",
    "hybrid": "#009E73",
    "rayleigh": "#E69F00",
    "king": "#CC79A7",
    "double_king": "#D55E00",
    "cut": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binned-root", type=Path, default=DEFAULT_BINNED_ROOT)
    parser.add_argument("--cell-ledger", type=Path, default=DEFAULT_CELL_LEDGER)
    parser.add_argument("--source-psf", type=Path, default=DEFAULT_SOURCE_PSF)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--double-king-npz", type=Path, default=DEFAULT_DOUBLE_KING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-ids", default="1,2,3")
    parser.add_argument("--tree-name", default="t_eventout")
    parser.add_argument("--weight-branch", default="mc_weight")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--random-starts", type=int, default=24)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument("--angle-check-max-events", type=int, default=20000)
    return parser.parse_args()


def fit_edges_deg() -> np.ndarray:
    """Fine core bins and progressively coarser tail bins over the full sphere."""
    return np.concatenate(
        (
            np.linspace(0.0, 5.0, 201, dtype=np.float64),
            np.linspace(5.25, 20.0, 60, dtype=np.float64),
            np.linspace(21.0, 180.0, 160, dtype=np.float64),
        )
    )


def survival_grid_deg() -> np.ndarray:
    return np.unique(
        np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.linspace(0.0, 20.0, 2001, dtype=np.float64),
                np.geomspace(0.01, 180.0, 2600, dtype=np.float64),
                np.asarray([180.0], dtype=np.float64),
            )
        )
    )


def double_king_bin_probability(edges_deg: np.ndarray, model: dict[str, float]) -> np.ndarray:
    core_cdf = 1.0 - (
        1.0 + edges_deg**2 / (2.0 * model["gamma_core"] * model["sigma_core_deg"] ** 2)
    ) ** (1.0 - model["gamma_core"])
    tail_cdf = 1.0 - (
        1.0 + edges_deg**2 / (2.0 * model["gamma_tail"] * model["sigma_tail_deg"] ** 2)
    ) ** (1.0 - model["gamma_tail"])
    cdf = model["physical_core_fraction"] * core_cdf + (1.0 - model["physical_core_fraction"]) * tail_cdf
    mass = np.clip(np.diff(cdf), 0.0, None)
    if float(np.sum(mass)) <= 0.0:
        raise ValueError("Old double-King extrapolation has no mass on [0, 180 deg]")
    return mass / np.sum(mass)


def _fit_loaded_cell(task: dict[str, Any]) -> dict[str, Any]:
    row = dict(task["row"])
    radius = np.asarray(row.pop("radius_deg"), dtype=np.float64)
    weights = np.asarray(row.pop("baseline_weight"), dtype=np.float64)
    valid = (
        np.isfinite(radius)
        & (radius >= 0.0)
        & (radius <= 180.0)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    radius = radius[valid]
    weights = weights[valid]
    edges = np.asarray(task["fit_edges_deg"], dtype=np.float64)
    grid = np.asarray(task["survival_grid_deg"], dtype=np.float64)
    counts, _ = np.histogram(radius, bins=edges, weights=weights)
    fit, model_probability, core_probability, tail_probability = fit_rayleigh_king_counts(
        counts,
        edges,
        random_seed=int(task["random_seed"]),
        random_starts=int(task["random_starts"]),
        quadrature_order=int(task["quadrature_order"]),
    )
    model_cdf = rayleigh_king_cdf(
        grid,
        fit.core_fraction,
        fit.sigma_rayleigh_deg,
        fit.sigma_king_deg,
        fit.gamma_king,
        quadrature_order=int(task["quadrature_order"]),
    )
    thresholds = np.asarray(task["survival_thresholds_deg"], dtype=np.float64)
    model_at_thresholds = 1.0 - rayleigh_king_cdf(
        thresholds,
        fit.core_fraction,
        fit.sigma_rayleigh_deg,
        fit.sigma_king_deg,
        fit.gamma_king,
        quadrature_order=int(task["quadrature_order"]),
    )
    probabilities = np.asarray(task["quantile_probabilities"], dtype=np.float64)
    row.update(fit.to_dict())
    row.update(
        {
            "positive_fit_events": int(radius.size),
            "max_radius_deg": float(np.max(radius)),
            "weighted_counts": counts,
            "data_probability": profile_probability(counts),
            "model_probability": model_probability,
            "core_probability": core_probability,
            "tail_probability": tail_probability,
            "empirical_survival": empirical_survival(radius, weights, grid),
            "model_survival": np.clip(1.0 - model_cdf, 0.0, 1.0),
            "empirical_quantiles_deg": weighted_quantiles(radius, weights, probabilities),
            "empirical_survival_at_thresholds": empirical_survival(radius, weights, thresholds),
            "model_survival_at_thresholds": model_at_thresholds,
        }
    )
    return row


def draw_figure(
    rows: list[dict[str, Any]],
    edges: np.ndarray,
    grid: np.ndarray,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt = setup_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    widths = np.diff(edges)
    fig, axes = plt.subplots(2, len(rows), figsize=(12.8, 7.2), squeeze=False)
    for column, row in enumerate(rows):
        top = axes[0, column]
        bottom = axes[1, column]
        data_density = np.asarray(row["data_probability"], dtype=np.float64) / widths
        model_density = np.asarray(row["model_probability"], dtype=np.float64) / widths
        core_density = row["core_fraction"] * np.asarray(row["core_probability"], dtype=np.float64) / widths
        tail_density = (1.0 - row["core_fraction"]) * np.asarray(row["tail_probability"], dtype=np.float64) / widths
        top.stairs(data_density, edges, color=COLORS["data"], linewidth=1.15, label="weighted raw MC")
        top.stairs(model_density, edges, color=COLORS["hybrid"], linewidth=1.8, label="Rayleigh + spherical King")
        top.stairs(core_density, edges, color=COLORS["rayleigh"], linewidth=1.25, linestyle="--", label="Rayleigh component")
        top.stairs(tail_density, edges, color=COLORS["king"], linewidth=1.25, linestyle=":", label="spherical-King component")
        top.axvline(5.0, color=COLORS["cut"], linewidth=0.85, linestyle=(0, (2, 2)), label="former 5 deg edge")
        top.set_xlim(0.0, 20.0)
        top.set_yscale("log")
        top.set_ylim(1.0e-6, max(2.0, 1.4 * float(np.nanmax(data_density))))
        top.set_title(f"cell {row['cell_id']}: Nhit {row['nhit_bin']}, predE {row['predE_bin']}")
        top.text(
            0.97,
            0.96,
            f"f_R={row['core_fraction']:.3f}\n"
            f"sigma_R={row['sigma_rayleigh_deg']:.3f} deg\n"
            f"sigma_K={row['sigma_king_deg']:.2f} deg, gamma={row['gamma_king']:.3f}\n"
            f"KL: {row['kl_divergence']:.3e} (old {row['old_double_king_kl']:.3e})",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
        )

        empirical = np.asarray(row["empirical_survival"], dtype=np.float64)
        hybrid = np.asarray(row["model_survival"], dtype=np.float64)
        old = np.asarray(row["old_double_king_survival"], dtype=np.float64)
        bottom.plot(grid[empirical > 0.0], empirical[empirical > 0.0], color=COLORS["data"], linewidth=1.65, label="empirical survival")
        bottom.plot(grid[hybrid > 0.0], hybrid[hybrid > 0.0], color=COLORS["hybrid"], linewidth=1.8, label="new hybrid survival")
        bottom.plot(grid[old > 0.0], old[old > 0.0], color=COLORS["double_king"], linewidth=1.25, linestyle="--", label="old double-King extrapolation")
        bottom.axvline(5.0, color=COLORS["cut"], linewidth=0.85, linestyle=(0, (2, 2)))
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set_xlim(0.03, 180.0)
        bottom.set_ylim(1.0e-7, 1.15)
        bottom.set_xlabel("angular error, r [deg]")
        empirical_at = np.asarray(row["empirical_survival_at_thresholds"], dtype=np.float64)
        model_at = np.asarray(row["model_survival_at_thresholds"], dtype=np.float64)
        bottom.text(
            0.04,
            0.06,
            f"S(5 deg): data {empirical_at[2]:.3f}, model {model_at[2]:.3f}\n"
            f"S(45 deg): data {empirical_at[5]:.2e}, model {model_at[5]:.2e}",
            transform=bottom.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.3,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )

    axes[0, 0].set_ylabel("weighted radial density [deg$^{-1}$]")
    axes[1, 0].set_ylabel(r"$S(r)=P(\Delta\theta>r)$")
    handles: list[Any] = []
    labels: list[str] = []
    for axis in (axes[0, 0], axes[1, 0]):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Cells 1-3 full-sphere raw-MC PSF: Rayleigh core + spherical-King tail", fontsize=12, y=0.995)
    fig.text(
        0.5,
        0.962,
        "No angular truncation or 5 deg renormalization; weight = mc_weight x Crab-declination theta reweight.",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    requested_ids = {int(value.strip()) for value in args.cell_ids.split(",") if value.strip()}
    if not requested_ids:
        raise ValueError("--cell-ids selected no cells")
    required_paths = (args.cell_ledger, args.source_psf, args.source_metadata, args.double_king_npz)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    if not args.binned_root.is_dir():
        raise FileNotFoundError(f"Binned MC root does not exist: {args.binned_root}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import importlib

    stage02 = importlib.import_module("apply.stages.02_build_psf")
    cells = [cell for cell in stage02.load_cells(args.cell_ledger) if int(cell.cell_id) in requested_ids]
    if {int(cell.cell_id) for cell in cells} != requested_ids:
        raise KeyError("The cell ledger does not contain every requested cell")
    source_metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    loge_filter = source_metadata["logE_true_filter"]
    loge_min = float(loge_filter["min_inclusive"])
    loge_max = float(loge_filter["max_exclusive"])
    with np.load(args.source_psf, allow_pickle=False) as handle:
        theta_edges = np.asarray(handle["theta_edges_deg"], dtype=np.float64)
        crab_probability = np.asarray(handle["crab_theta_probability"], dtype=np.float64)

    read_tasks = [
        {
            "index": cell.index,
            "cell_id": cell.cell_id,
            "nhit_bin": cell.nhit_bin,
            "predE_bin": cell.predE_bin,
            "mc_count": cell.mc_count,
            "selection_version": cell.selection_version,
            "selection_reason": cell.selection_reason,
            "binned_root": str(args.binned_root),
            "tree_name": args.tree_name,
            "weight_branch": args.weight_branch,
            "file_progress_every": args.file_progress_every,
            "angle_check_max_events": args.angle_check_max_events,
            "loge_min": loge_min,
            "loge_max": loge_max,
            "theta_edges_deg": theta_edges,
            "crab_probability": crab_probability,
        }
        for cell in cells
    ]
    loaded_rows: list[dict[str, Any]] = []
    workers = max(1, min(int(args.workers), len(read_tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_read_cell, task): int(task["cell_id"]) for task in read_tasks}
        for future in as_completed(futures):
            row = future.result()
            loaded_rows.append(row)
            print(
                f"cell {row['cell_id']} loaded: files={row['input_files']}, selected={row['selected_events']}",
                flush=True,
            )

    edges = fit_edges_deg()
    grid = survival_grid_deg()
    probabilities = np.asarray([0.50, 0.68, TARGET_CONTAINMENT, 0.90], dtype=np.float64)
    thresholds = np.asarray([1.0, 3.0, 5.0, 10.0, 20.0, 45.0, 90.0], dtype=np.float64)
    fit_tasks = [
        {
            "row": row,
            "fit_edges_deg": edges,
            "survival_grid_deg": grid,
            "quantile_probabilities": probabilities,
            "survival_thresholds_deg": thresholds,
            "random_seed": 20260718 + int(row["cell_id"]),
            "random_starts": args.random_starts,
            "quadrature_order": args.quadrature_order,
        }
        for row in loaded_rows
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fit_loaded_cell, task): int(task["row"]["cell_id"]) for task in fit_tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"cell {row['cell_id']} fitted: f_R={row['core_fraction']:.4f}, "
                f"sigma_R={row['sigma_rayleigh_deg']:.4f}, sigma_K={row['sigma_king_deg']:.3f}, "
                f"gamma={row['gamma_king']:.4f}, KL={row['kl_divergence']:.4e}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["cell_id"]))

    old_models = load_double_king_models(args.double_king_npz, requested_ids)
    for row in rows:
        old_model = old_models[int(row["cell_id"])]
        old_probability = double_king_bin_probability(edges, old_model)
        row["old_double_king_kl"] = kl_divergence(row["data_probability"], old_probability)
        row["kl_improvement_factor"] = (
            row["old_double_king_kl"] / row["kl_divergence"]
            if row["kl_divergence"] > 0.0
            else float("inf")
        )
        row["old_double_king_survival"] = double_king_survival(grid, old_model)
        row["old_double_king_survival_at_thresholds"] = double_king_survival(thresholds, old_model)

    stem = f"{BASE_RUN_ID}_cell123_spherical_rayleigh_king_fit"
    outputs = {
        "png": args.output_dir / f"{stem}.png",
        "pdf": args.output_dir / f"{stem}.pdf",
        "npz": args.output_dir / f"{stem}.npz",
        "csv": args.output_dir / f"{stem}_summary.csv",
        "metadata": args.output_dir / f"{stem}_metadata.json",
    }
    draw_figure(rows, edges, grid, outputs["png"], outputs["pdf"])

    np.savez_compressed(
        outputs["npz"],
        cell_id=np.asarray([row["cell_id"] for row in rows], dtype=np.int32),
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        predE_bin=np.asarray([row["predE_bin"] for row in rows], dtype="U32"),
        fit_edges_deg=edges,
        data_probability=np.vstack([row["data_probability"] for row in rows]),
        model_probability=np.vstack([row["model_probability"] for row in rows]),
        rayleigh_probability=np.vstack([row["core_probability"] for row in rows]),
        spherical_king_probability=np.vstack([row["tail_probability"] for row in rows]),
        survival_grid_deg=grid,
        empirical_survival=np.vstack([row["empirical_survival"] for row in rows]),
        model_survival=np.vstack([row["model_survival"] for row in rows]),
        old_double_king_survival=np.vstack([row["old_double_king_survival"] for row in rows]),
        quantile_probabilities=probabilities,
        empirical_quantiles_deg=np.vstack([row["empirical_quantiles_deg"] for row in rows]),
        survival_thresholds_deg=thresholds,
        empirical_survival_at_thresholds=np.vstack([row["empirical_survival_at_thresholds"] for row in rows]),
        model_survival_at_thresholds=np.vstack([row["model_survival_at_thresholds"] for row in rows]),
        old_double_king_survival_at_thresholds=np.vstack(
            [row["old_double_king_survival_at_thresholds"] for row in rows]
        ),
        core_fraction=np.asarray([row["core_fraction"] for row in rows]),
        sigma_rayleigh_deg=np.asarray([row["sigma_rayleigh_deg"] for row in rows]),
        sigma_king_deg=np.asarray([row["sigma_king_deg"] for row in rows]),
        gamma_king=np.asarray([row["gamma_king"] for row in rows]),
    )

    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "input_files",
        "selected_events",
        "effective_events",
        "core_fraction",
        "sigma_rayleigh_deg",
        "sigma_king_deg",
        "gamma_king",
        "r50_deg",
        "r68_deg",
        "r712979_deg",
        "r90_deg",
        "kl_divergence",
        "old_double_king_kl",
        "kl_improvement_factor",
        "optimizer_success",
        "optimizer_message",
        "boundary_flags",
        "empirical_survival_5deg",
        "model_survival_5deg",
        "empirical_survival_45deg",
        "model_survival_45deg",
        "empirical_survival_90deg",
        "model_survival_90deg",
    ]
    with outputs["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            csv_row = {name: row.get(name) for name in fieldnames}
            csv_row.update(
                {
                    "boundary_flags": ";".join(row["boundary_flags"]),
                    "empirical_survival_5deg": row["empirical_survival_at_thresholds"][2],
                    "model_survival_5deg": row["model_survival_at_thresholds"][2],
                    "empirical_survival_45deg": row["empirical_survival_at_thresholds"][5],
                    "model_survival_45deg": row["model_survival_at_thresholds"][5],
                    "empirical_survival_90deg": row["empirical_survival_at_thresholds"][6],
                    "model_survival_90deg": row["model_survival_at_thresholds"][6],
                }
            )
            writer.writerow(csv_row)

    excluded_metadata_keys = {
        "weighted_counts",
        "data_probability",
        "model_probability",
        "core_probability",
        "tail_probability",
        "empirical_survival",
        "model_survival",
        "old_double_king_survival",
    }
    metadata_rows = []
    for row in rows:
        metadata_rows.append(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in row.items()
                if key not in excluded_metadata_keys
            }
        )
    metadata = {
        "description": "Cells 1-3 full-sphere raw-MC Rayleigh-core plus spherical-King-tail fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "model": {
            "formula": "f_R * Rayleigh(r; sigma_R)|[0,pi] + (1-f_R) * sin(r) * (1+r^2/(2*gamma*sigma_K^2))^(-gamma) / Z_K",
            "normalization_interval_deg": [0.0, 180.0],
            "constraints": ["0 < f_R < 1", "0 < sigma_R < sigma_K", "gamma_K > 1"],
            "objective": "fine-bin integrated Stage-B weighted negative log likelihood",
            "quadrature": f"Gauss-Legendre order {args.quadrature_order} per radial bin",
        },
        "selection": {
            "cell_ids": sorted(requested_ids),
            "logE_true_min_inclusive": loge_min,
            "logE_true_max_exclusive": loge_max,
            "angular_cut": None,
            "baseline_weight": f"{args.weight_branch} * crab_declination_theta_reweight",
        },
        "inputs": {
            "binned_root": str(args.binned_root.resolve()),
            "cell_ledger": str(args.cell_ledger.resolve()),
            "cell_ledger_sha256": path_sha256(args.cell_ledger),
            "source_psf": str(args.source_psf.resolve()),
            "source_psf_sha256": path_sha256(args.source_psf),
            "source_metadata": str(args.source_metadata.resolve()),
            "source_metadata_sha256": path_sha256(args.source_metadata),
            "double_king_npz": str(args.double_king_npz.resolve()),
            "double_king_npz_sha256": path_sha256(args.double_king_npz),
        },
        "fit_edges_deg": edges.tolist(),
        "quantile_probabilities": probabilities.tolist(),
        "survival_thresholds_deg": thresholds.tolist(),
        "cells": metadata_rows,
        "old_double_king_models": old_models,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": metadata_rows, "outputs": metadata["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
