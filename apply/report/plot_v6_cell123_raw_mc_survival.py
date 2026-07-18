#!/usr/bin/env python3
"""Plot untruncated raw-MC angular-error survival functions for cells 1-3."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    / "v6-64748-nhit100-reselect44-split56-miss030-fermi-double-king-tail-diagnostic"
)

COLORS = {
    "weighted": "#0072B2",
    "unweighted": "#6B7280",
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
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument("--angle-check-max-events", type=int, default=20000)
    return parser.parse_args()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def empirical_survival(radius_deg: np.ndarray, weights: np.ndarray, grid_deg: np.ndarray) -> np.ndarray:
    """Return P(radius > grid) using one normalization over all selected events."""
    radius = np.asarray(radius_deg, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    grid = np.asarray(grid_deg, dtype=np.float64)
    valid = np.isfinite(radius) & (radius >= 0.0) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        raise ValueError("No positive finite events are available for the survival function")
    order = np.argsort(radius[valid], kind="mergesort")
    sorted_radius = radius[valid][order]
    sorted_weight = weight[valid][order]
    cumulative = np.concatenate(([0.0], np.cumsum(sorted_weight, dtype=np.float64)))
    total = float(cumulative[-1])
    indices = np.searchsorted(sorted_radius, grid, side="right")
    survival = (total - cumulative[indices]) / total
    return np.clip(survival, 0.0, 1.0)


def weighted_quantiles(radius_deg: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_deg, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    valid = np.isfinite(radius) & (radius >= 0.0) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return np.full(probability.shape, np.nan, dtype=np.float64)
    order = np.argsort(radius[valid], kind="mergesort")
    sorted_radius = radius[valid][order]
    sorted_weight = weight[valid][order]
    cumulative = np.cumsum(sorted_weight, dtype=np.float64)
    cumulative /= cumulative[-1]
    return np.interp(probability, cumulative, sorted_radius, left=sorted_radius[0], right=sorted_radius[-1])


def king_cdf(radius_deg: np.ndarray, sigma_deg: float, gamma: float) -> np.ndarray:
    radius = np.asarray(radius_deg, dtype=np.float64)
    return 1.0 - (1.0 + radius**2 / (2.0 * gamma * sigma_deg**2)) ** (1.0 - gamma)


def double_king_survival(grid_deg: np.ndarray, model: dict[str, float]) -> np.ndarray:
    core = king_cdf(grid_deg, model["sigma_core_deg"], model["gamma_core"])
    tail = king_cdf(grid_deg, model["sigma_tail_deg"], model["gamma_tail"])
    cdf = model["physical_core_fraction"] * core + (1.0 - model["physical_core_fraction"]) * tail
    return np.clip(1.0 - cdf, 0.0, 1.0)


def effective_events(weights: np.ndarray) -> float:
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(weight) & (weight > 0.0)
    sumw = float(np.sum(weight[valid]))
    sumw2 = float(np.sum(weight[valid] ** 2))
    return sumw * sumw / sumw2 if sumw2 > 0.0 else 0.0


def _read_cell(task: dict[str, Any]) -> dict[str, Any]:
    stage02 = importlib.import_module("apply.stages.02_build_psf")
    cell = stage02.CellSpec(
        index=int(task["index"]),
        cell_id=int(task["cell_id"]),
        nhit_bin=str(task["nhit_bin"]),
        predE_bin=str(task["predE_bin"]),
        mc_count=int(task["mc_count"]),
        selection_version=str(task["selection_version"]),
        selection_reason=str(task["selection_reason"]),
    )
    cell_dir = stage02.binned_cell_dir(Path(task["binned_root"]), cell)
    files = stage02.discover_cell_files(cell_dir, None, allow_missing_cell_dirs=False)
    events = stage02.read_cell_events(
        files,
        tree_name=str(task["tree_name"]),
        weight_branch=str(task["weight_branch"]),
        allow_missing_weight=False,
        angle_check_max_events=int(task["angle_check_max_events"]),
        file_progress_every=int(task["file_progress_every"]),
        progress_label=f"cell {cell.cell_id}",
    )

    loge_min = float(task["loge_min"])
    loge_max = float(task["loge_max"])
    loge_valid = np.isfinite(events.loge_true) & (events.loge_true >= loge_min) & (events.loge_true < loge_max)
    ratio_support = (
        loge_valid
        & np.isfinite(events.dangle_rad)
        & (events.dangle_rad >= 0.0)
        & (events.dangle_rad <= math.pi + 1.0e-10)
        & np.isfinite(events.mc_weight)
        & (events.mc_weight > 0.0)
    )
    theta_edges = np.asarray(task["theta_edges_deg"], dtype=np.float64)
    crab_probability = np.asarray(task["crab_probability"], dtype=np.float64)
    theta_ratio, theta_meta = stage02.theta_reweight_ratio(
        events.mc_theta_deg,
        events.mc_weight,
        theta_edges,
        crab_probability,
        support_mask=ratio_support,
        allow_incomplete_theta_support=False,
    )
    theta_index, theta_valid = stage02.theta_bin_indices(events.mc_theta_deg, theta_edges)
    selected = theta_valid & ratio_support
    baseline_weight = np.zeros(events.dangle_rad.size, dtype=np.float64)
    baseline_weight[selected] = events.mc_weight[selected] * theta_ratio[theta_index[selected]]
    positive = selected & np.isfinite(baseline_weight) & (baseline_weight > 0.0)
    radius_deg = np.degrees(events.dangle_rad[selected])
    baseline_weight = baseline_weight[selected]
    angle_check = np.asarray(events.angle_check_absdiff_rad, dtype=np.float64)
    return {
        "cell_id": cell.cell_id,
        "nhit_bin": cell.nhit_bin,
        "predE_bin": cell.predE_bin,
        "cell_dir": str(cell_dir),
        "input_files": len(files),
        "raw_events": int(events.dangle_rad.size),
        "selected_events": int(radius_deg.size),
        "positive_baseline_weight_events": int(np.count_nonzero(positive)),
        "radius_deg": radius_deg,
        "baseline_weight": baseline_weight,
        "sum_baseline_weight": float(np.sum(baseline_weight)),
        "effective_events": effective_events(baseline_weight),
        "missing_crab_probability_mass": float(theta_meta.get("missing_crab_probability_mass") or 0.0),
        "angle_check_max_absdiff_rad": float(np.nanmax(angle_check)) if angle_check.size else None,
    }


def load_double_king_models(path: Path, cell_ids: set[int]) -> dict[int, dict[str, float]]:
    with np.load(path, allow_pickle=False) as handle:
        ids = np.asarray(handle["cell_id"], dtype=np.int64)
        fields = (
            "physical_core_fraction",
            "sigma_core_deg",
            "gamma_core",
            "sigma_tail_deg",
            "gamma_tail",
            "conditional_r_target_deg",
        )
        output: dict[int, dict[str, float]] = {}
        for index, cell_id in enumerate(ids):
            if int(cell_id) not in cell_ids:
                continue
            output[int(cell_id)] = {name: float(handle[name][index]) for name in fields}
    missing = cell_ids.difference(output)
    if missing:
        raise KeyError(f"Double-King NPZ is missing cells: {sorted(missing)}")
    return output


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    return plt


def draw_figure(
    rows: list[dict[str, Any]],
    grid_deg: np.ndarray,
    models: dict[int, dict[str, float]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, len(rows), figsize=(12.6, 6.8), sharey=True, squeeze=False)
    for column, row in enumerate(rows):
        cell_id = int(row["cell_id"])
        weighted = np.asarray(row["weighted_survival"], dtype=np.float64)
        unweighted = np.asarray(row["unweighted_survival"], dtype=np.float64)
        model_survival = double_king_survival(grid_deg, models[cell_id])
        for axis in (axes[0, column], axes[1, column]):
            valid_weighted = weighted > 0.0
            valid_unweighted = unweighted > 0.0
            valid_model = model_survival > 0.0
            axis.plot(
                grid_deg[valid_weighted], weighted[valid_weighted],
                color=COLORS["weighted"], linewidth=1.7, label="raw MC, Stage-B weight",
            )
            axis.plot(
                grid_deg[valid_unweighted], unweighted[valid_unweighted],
                color=COLORS["unweighted"], linewidth=1.15, linestyle="--", label="raw MC, unweighted",
            )
            axis.plot(
                grid_deg[valid_model], model_survival[valid_model],
                color=COLORS["double_king"], linewidth=1.35, label="current full-plane double-King",
            )
            axis.axvline(5.0, color=COLORS["cut"], linewidth=0.9, linestyle=":", label="former 5 deg edge")
            axis.set_yscale("log")
            axis.set_ylim(1.0e-7, 1.15)
        axes[0, column].set_xlim(0.0, 20.0)
        axes[0, column].set_title(
            f"cell {cell_id}: Nhit {row['nhit_bin']}, predE {row['predE_bin']}"
        )
        axes[0, column].text(
            0.97,
            0.96,
            f"S_w(5 deg)={row['weighted_survival_at_5deg']:.3e}\n"
            f"r71,w={row['weighted_r712979_deg']:.3f} deg\n"
            f"N={row['selected_events']:,}, N_eff={row['effective_events']:.0f}",
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        axes[1, column].set_xscale("log")
        axes[1, column].set_xlim(0.03, 180.0)
        axes[1, column].set_xlabel("angular error, r [deg]")
    axes[0, 0].set_ylabel("S(r) = P(Delta theta > r)")
    axes[1, 0].set_ylabel("S(r) = P(Delta theta > r)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Cells 1-3 raw-MC angular-error survival: no angular truncation or 5 deg renormalization",
        fontsize=12,
        y=0.995,
    )
    fig.text(
        0.5,
        0.962,
        "Formal logE_true selection; Stage-B weight = mc_weight x Crab-declination theta reweight. "
        "Top: 0-20 deg; bottom: full spherical range.",
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

    tasks = [
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
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(int(args.workers), len(tasks)))) as executor:
        futures = {executor.submit(_read_cell, task): int(task["cell_id"]) for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"cell {row['cell_id']} complete: files={row['input_files']}, "
                f"selected={row['selected_events']}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["cell_id"]))

    grid_deg = np.unique(
        np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.linspace(0.0, 20.0, 2001, dtype=np.float64),
                np.geomspace(0.01, 180.0, 2600, dtype=np.float64),
            )
        )
    )
    probabilities = np.asarray([0.50, 0.68, TARGET_CONTAINMENT, 0.90, 0.95, 0.99], dtype=np.float64)
    thresholds = np.asarray([1.0, 3.0, 5.0, 10.0, 20.0, 45.0, 90.0], dtype=np.float64)
    for row in rows:
        radius = np.asarray(row.pop("radius_deg"), dtype=np.float64)
        baseline_weight = np.asarray(row.pop("baseline_weight"), dtype=np.float64)
        ones = np.ones(radius.size, dtype=np.float64)
        weighted_survival = empirical_survival(radius, baseline_weight, grid_deg)
        unweighted_survival = empirical_survival(radius, ones, grid_deg)
        weighted_quantile = weighted_quantiles(radius, baseline_weight, probabilities)
        unweighted_quantile = weighted_quantiles(radius, ones, probabilities)
        weighted_at_threshold = empirical_survival(radius, baseline_weight, thresholds)
        unweighted_at_threshold = empirical_survival(radius, ones, thresholds)
        row.update(
            {
                "max_radius_deg": float(np.max(radius)),
                "weighted_survival": weighted_survival,
                "unweighted_survival": unweighted_survival,
                "weighted_quantiles_deg": weighted_quantile,
                "unweighted_quantiles_deg": unweighted_quantile,
                "weighted_survival_at_thresholds": weighted_at_threshold,
                "unweighted_survival_at_thresholds": unweighted_at_threshold,
                "weighted_survival_at_5deg": float(weighted_at_threshold[2]),
                "unweighted_survival_at_5deg": float(unweighted_at_threshold[2]),
                "weighted_r712979_deg": float(weighted_quantile[2]),
                "unweighted_r712979_deg": float(unweighted_quantile[2]),
            }
        )

    models = load_double_king_models(args.double_king_npz, requested_ids)
    stem = f"{BASE_RUN_ID}_cell123_raw_mc_survival"
    outputs = {
        "png": args.output_dir / f"{stem}.png",
        "pdf": args.output_dir / f"{stem}.pdf",
        "npz": args.output_dir / f"{stem}.npz",
        "csv": args.output_dir / f"{stem}_summary.csv",
        "metadata": args.output_dir / f"{stem}_metadata.json",
    }
    draw_figure(rows, grid_deg, models, outputs["png"], outputs["pdf"])

    np.savez_compressed(
        outputs["npz"],
        cell_id=np.asarray([row["cell_id"] for row in rows], dtype=np.int32),
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        predE_bin=np.asarray([row["predE_bin"] for row in rows], dtype="U32"),
        radius_grid_deg=grid_deg.astype(np.float64),
        weighted_survival=np.vstack([row["weighted_survival"] for row in rows]).astype(np.float64),
        unweighted_survival=np.vstack([row["unweighted_survival"] for row in rows]).astype(np.float64),
        quantile_probabilities=probabilities,
        weighted_quantiles_deg=np.vstack([row["weighted_quantiles_deg"] for row in rows]),
        unweighted_quantiles_deg=np.vstack([row["unweighted_quantiles_deg"] for row in rows]),
        survival_thresholds_deg=thresholds,
        weighted_survival_at_thresholds=np.vstack([row["weighted_survival_at_thresholds"] for row in rows]),
        unweighted_survival_at_thresholds=np.vstack([row["unweighted_survival_at_thresholds"] for row in rows]),
    )

    fieldnames = [
        "cell_id", "nhit_bin", "predE_bin", "input_files", "raw_events", "selected_events",
        "positive_baseline_weight_events", "sum_baseline_weight", "effective_events", "max_radius_deg", "weighted_r712979_deg",
        "unweighted_r712979_deg", "weighted_survival_at_5deg", "unweighted_survival_at_5deg",
        "missing_crab_probability_mass", "angle_check_max_absdiff_rad",
    ]
    with outputs["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    metadata_rows = []
    for row in rows:
        metadata_rows.append(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in row.items()
                if key not in {"weighted_survival", "unweighted_survival"}
            }
        )
    metadata = {
        "description": "Raw-MC angular-error survival without angular truncation or 5-degree renormalization",
        "git_sha": git_sha(),
        "selection": {
            "cell_ids": sorted(requested_ids),
            "logE_true_min_inclusive": loge_min,
            "logE_true_max_exclusive": loge_max,
            "angular_cut": None,
            "baseline_weight": f"{args.weight_branch} * crab_declination_theta_reweight",
            "survival_denominator": "all positive-weight formal selected events",
            "survival_definition": "sum(w_i * I(delta_theta_i > r)) / sum(w_i)",
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
        "quantile_probabilities": probabilities.tolist(),
        "survival_thresholds_deg": thresholds.tolist(),
        "cells": metadata_rows,
        "double_king_models": models,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": metadata_rows, "outputs": metadata["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
