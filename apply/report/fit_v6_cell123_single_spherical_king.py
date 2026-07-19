#!/usr/bin/env python3
"""Fit cells 1-3 with one full-sphere spherical King and compare with double-King."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

from apply.stages.psf_full_sphere_mixtures import fit_single_spherical_king_counts  # noqa: E402
from apply.stages.psf_rayleigh_king import spherical_king_cdf  # noqa: E402


BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
INPUT_RUN_LABEL = "v6-64748-nhit100-reselect44-split56-miss030-full-sphere-model-comparison"
OUTPUT_RUN_LABEL = "v6-64748-nhit100-reselect44-split56-miss030-single-spherical-king-fit"
INPUT_STEM = f"{BASE_RUN_ID}_cell123_full_sphere_psf_model_comparison"
OUTPUT_STEM = f"{BASE_RUN_ID}_cell123_single_spherical_king_fit"

DEFAULT_INPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / INPUT_RUN_LABEL
DEFAULT_INPUT_NPZ = DEFAULT_INPUT_DIR / f"{INPUT_STEM}.npz"
DEFAULT_INPUT_METADATA = DEFAULT_INPUT_DIR / f"{INPUT_STEM}_metadata.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / OUTPUT_RUN_LABEL

COLORS = {
    "data": "#0072B2",
    "single": "#009E73",
    "double": "#D55E00",
    "cut": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--random-starts", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=20)
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
            "lines.linewidth": 1.8,
        }
    )
    return plt


def survival_log_rmse(
    empirical: np.ndarray,
    model: np.ndarray,
    grid_deg: np.ndarray,
    effective_events: float,
) -> float:
    floor = 1.0 / max(float(effective_events), 1.0)
    valid = (
        (grid_deg >= 0.03)
        & np.isfinite(empirical)
        & (empirical >= floor)
        & np.isfinite(model)
        & (model > 0.0)
    )
    if not np.any(valid):
        return float("nan")
    residual = np.log10(model[valid]) - np.log10(empirical[valid])
    return float(np.sqrt(np.mean(residual**2)))


def _fit_cell(task: dict[str, Any]) -> dict[str, Any]:
    edges = np.asarray(task["fit_edges_deg"], dtype=np.float64)
    grid = np.asarray(task["survival_grid_deg"], dtype=np.float64)
    thresholds = np.asarray(task["survival_thresholds_deg"], dtype=np.float64)
    data_probability = np.asarray(task["data_probability"], dtype=np.float64)
    empirical_survival = np.asarray(task["empirical_survival"], dtype=np.float64)
    double_probability = np.asarray(task["double_probability"], dtype=np.float64)
    double_survival = np.asarray(task["double_survival"], dtype=np.float64)
    effective_events = float(task["effective_events"])

    fit, single_probability = fit_single_spherical_king_counts(
        data_probability,
        edges,
        random_seed=int(task["random_seed"]),
        random_starts=int(task["random_starts"]),
        quadrature_order=int(task["quadrature_order"]),
    )
    single_survival = 1.0 - spherical_king_cdf(
        grid,
        fit.sigma_deg,
        fit.gamma,
        quadrature_order=int(task["quadrature_order"]),
    )
    single_at_thresholds = 1.0 - spherical_king_cdf(
        thresholds,
        fit.sigma_deg,
        fit.gamma,
        quadrature_order=int(task["quadrature_order"]),
    )
    double_at_thresholds = np.interp(thresholds, grid, double_survival)

    single_kl = float(fit.kl_divergence)
    double_kl = float(task["double_kl_divergence"])
    single_aic = 2.0 * 2.0 + 2.0 * effective_events * single_kl
    double_aic = 2.0 * 5.0 + 2.0 * effective_events * double_kl
    single_bic = 2.0 * math.log(max(effective_events, 1.0)) + 2.0 * effective_events * single_kl
    double_bic = 5.0 * math.log(max(effective_events, 1.0)) + 2.0 * effective_events * double_kl
    return {
        "cell_id": int(task["cell_id"]),
        "nhit_bin": str(task["nhit_bin"]),
        "predE_bin": str(task["predE_bin"]),
        "effective_events": effective_events,
        "data_probability": data_probability,
        "empirical_survival": empirical_survival,
        "empirical_survival_at_thresholds": np.asarray(
            task["empirical_survival_at_thresholds"],
            dtype=np.float64,
        ),
        "single_probability": single_probability,
        "single_survival": np.clip(single_survival, 0.0, 1.0),
        "single_survival_at_thresholds": np.clip(single_at_thresholds, 0.0, 1.0),
        "double_probability": double_probability,
        "double_survival": double_survival,
        "double_survival_at_thresholds": double_at_thresholds,
        "single_fit": fit.to_dict(),
        "double_fit": dict(task["double_fit"]),
        "single_kl_divergence": single_kl,
        "double_kl_divergence": double_kl,
        "single_aic_relative_saturated": single_aic,
        "double_aic_relative_saturated": double_aic,
        "delta_aic_single_minus_double": single_aic - double_aic,
        "single_bic_relative_saturated": single_bic,
        "double_bic_relative_saturated": double_bic,
        "delta_bic_single_minus_double": single_bic - double_bic,
        "single_survival_log_rmse": survival_log_rmse(
            empirical_survival,
            single_survival,
            grid,
            effective_events,
        ),
        "double_survival_log_rmse": survival_log_rmse(
            empirical_survival,
            double_survival,
            grid,
            effective_events,
        ),
    }


def draw_figure(
    rows: list[dict[str, Any]],
    edges: np.ndarray,
    grid: np.ndarray,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt = setup_matplotlib()
    widths = np.diff(edges)
    fig, axes = plt.subplots(2, len(rows), figsize=(12.8, 7.2), squeeze=False)
    for column, row in enumerate(rows):
        top = axes[0, column]
        bottom = axes[1, column]
        data_density = np.asarray(row["data_probability"], dtype=np.float64) / widths
        single_density = np.asarray(row["single_probability"], dtype=np.float64) / widths
        double_density = np.asarray(row["double_probability"], dtype=np.float64) / widths
        top.stairs(
            data_density,
            edges,
            color=COLORS["data"],
            linewidth=1.15,
            label="weighted raw MC",
            zorder=2,
        )
        top.stairs(
            single_density,
            edges,
            color=COLORS["single"],
            linewidth=2.0,
            label="single spherical-King",
            zorder=4,
        )
        top.stairs(
            double_density,
            edges,
            color=COLORS["double"],
            linewidth=1.55,
            linestyle="--",
            label="double spherical-King reference",
            zorder=3,
        )
        top.axvline(
            5.0,
            color=COLORS["cut"],
            linewidth=0.85,
            linestyle=(0, (2, 2)),
            label="former 5 deg edge",
        )
        top.set_xlim(0.0, 20.0)
        top.set_yscale("log")
        top.set_ylim(1.0e-6, max(2.0, 1.4 * float(np.nanmax(data_density))))
        top.set_title(f"cell {row['cell_id']}: Nhit {row['nhit_bin']}, predE {row['predE_bin']}")
        single_fit = row["single_fit"]
        boundary_text = ", ".join(single_fit["boundary_flags"]) if single_fit["boundary_flags"] else "none"
        top.text(
            0.97,
            0.96,
            f"single K: sigma={single_fit['sigma_deg']:.3f} deg, gamma={single_fit['gamma']:.3f}\n"
            f"KL: single {row['single_kl_divergence']:.4e}, double {row['double_kl_divergence']:.4e}\n"
            f"Delta AIC(single-double)={row['delta_aic_single_minus_double']:+.1f}\n"
            f"single boundary={boundary_text}",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        )

        empirical = np.asarray(row["empirical_survival"], dtype=np.float64)
        single_survival = np.asarray(row["single_survival"], dtype=np.float64)
        double_survival = np.asarray(row["double_survival"], dtype=np.float64)
        bottom.plot(
            grid[empirical > 0.0],
            empirical[empirical > 0.0],
            color=COLORS["data"],
            linewidth=1.65,
            label="empirical survival",
            zorder=2,
        )
        bottom.plot(
            grid[single_survival > 0.0],
            single_survival[single_survival > 0.0],
            color=COLORS["single"],
            linewidth=2.0,
            label="single spherical-King survival",
            zorder=4,
        )
        bottom.plot(
            grid[double_survival > 0.0],
            double_survival[double_survival > 0.0],
            color=COLORS["double"],
            linewidth=1.55,
            linestyle="--",
            label="double spherical-King survival",
            zorder=3,
        )
        bottom.axvline(5.0, color=COLORS["cut"], linewidth=0.85, linestyle=(0, (2, 2)))
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set_xlim(0.03, 180.0)
        bottom.set_ylim(1.0e-7, 1.15)
        bottom.set_xlabel("angular error, r [deg]")
        empirical_at = np.asarray(row["empirical_survival_at_thresholds"], dtype=np.float64)
        single_at = np.asarray(row["single_survival_at_thresholds"], dtype=np.float64)
        double_at = np.asarray(row["double_survival_at_thresholds"], dtype=np.float64)
        bottom.text(
            0.04,
            0.06,
            f"log-S RMSE: single {row['single_survival_log_rmse']:.3f}, double {row['double_survival_log_rmse']:.3f}\n"
            f"S(45 deg): data {empirical_at[5]:.2e}, single {single_at[5]:.2e}, double {double_at[5]:.2e}\n"
            f"S(90 deg): data {empirical_at[6]:.2e}, single {single_at[6]:.2e}, double {double_at[6]:.2e}",
            transform=bottom.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
        )

    axes[0, 0].set_ylabel("weighted radial density [deg$^{-1}$]")
    axes[1, 0].set_ylabel(r"$S(r)=P(\Delta\theta>r)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Cells 1-3 full-sphere single spherical-King fit", fontsize=12, y=0.995)
    fig.text(
        0.5,
        0.962,
        "Single-King and double-King are evaluated on the same 0-180 deg Stage-B-weighted raw-MC distribution.",
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
    missing = [str(path) for path in (args.input_npz, args.input_metadata) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing single-King inputs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    metadata = json.loads(args.input_metadata.read_text(encoding="utf-8"))
    metadata_by_cell = {int(row["cell_id"]): row for row in metadata["cells"]}
    with np.load(args.input_npz, allow_pickle=False) as handle:
        source = {key: np.asarray(handle[key]) for key in handle.files}
    edges = np.asarray(source["fit_edges_deg"], dtype=np.float64)
    grid = np.asarray(source["survival_grid_deg"], dtype=np.float64)
    thresholds = np.asarray(source["survival_thresholds_deg"], dtype=np.float64)

    tasks: list[dict[str, Any]] = []
    for index, cell_value in enumerate(np.asarray(source["cell_id"], dtype=np.int64)):
        cell_id = int(cell_value)
        cell_metadata = metadata_by_cell[cell_id]
        double_metadata = cell_metadata["models"]["double_spherical_king"]
        tasks.append(
            {
                "cell_id": cell_id,
                "nhit_bin": str(source["nhit_bin"][index]),
                "predE_bin": str(source["predE_bin"][index]),
                "effective_events": float(cell_metadata["effective_events"]),
                "fit_edges_deg": edges,
                "survival_grid_deg": grid,
                "survival_thresholds_deg": thresholds,
                "data_probability": np.asarray(source["data_probability"][index], dtype=np.float64),
                "empirical_survival": np.asarray(source["empirical_survival"][index], dtype=np.float64),
                "empirical_survival_at_thresholds": np.asarray(
                    source["empirical_survival_at_thresholds"][index],
                    dtype=np.float64,
                ),
                "double_probability": np.asarray(
                    source["double_spherical_king_probability"][index],
                    dtype=np.float64,
                ),
                "double_survival": np.asarray(
                    source["double_spherical_king_survival"][index],
                    dtype=np.float64,
                ),
                "double_fit": dict(double_metadata["fit"]),
                "double_kl_divergence": float(double_metadata["kl_divergence"]),
                "random_seed": 20260719 + cell_id,
                "random_starts": args.random_starts,
                "quadrature_order": args.quadrature_order,
            }
        )

    rows: list[dict[str, Any]] = []
    workers = max(1, min(int(args.workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fit_cell, task): int(task["cell_id"]) for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"cell {row['cell_id']} complete: single KL={row['single_kl_divergence']:.4e}, "
                f"double KL={row['double_kl_divergence']:.4e}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["cell_id"]))

    outputs = {
        "png": args.output_dir / f"{OUTPUT_STEM}.png",
        "pdf": args.output_dir / f"{OUTPUT_STEM}.pdf",
        "npz": args.output_dir / f"{OUTPUT_STEM}.npz",
        "csv": args.output_dir / f"{OUTPUT_STEM}_summary.csv",
        "metadata": args.output_dir / f"{OUTPUT_STEM}_metadata.json",
    }
    draw_figure(rows, edges, grid, outputs["png"], outputs["pdf"])

    np.savez_compressed(
        outputs["npz"],
        cell_id=np.asarray([row["cell_id"] for row in rows], dtype=np.int32),
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        predE_bin=np.asarray([row["predE_bin"] for row in rows], dtype="U32"),
        fit_edges_deg=edges,
        survival_grid_deg=grid,
        survival_thresholds_deg=thresholds,
        data_probability=np.vstack([row["data_probability"] for row in rows]),
        empirical_survival=np.vstack([row["empirical_survival"] for row in rows]),
        single_king_probability=np.vstack([row["single_probability"] for row in rows]),
        double_king_probability=np.vstack([row["double_probability"] for row in rows]),
        single_king_survival=np.vstack([row["single_survival"] for row in rows]),
        double_king_survival=np.vstack([row["double_survival"] for row in rows]),
    )

    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "effective_events",
        "sigma_deg",
        "gamma",
        "single_kl_divergence",
        "double_kl_divergence",
        "single_aic_relative_saturated",
        "double_aic_relative_saturated",
        "delta_aic_single_minus_double",
        "single_bic_relative_saturated",
        "double_bic_relative_saturated",
        "delta_bic_single_minus_double",
        "single_survival_log_rmse",
        "double_survival_log_rmse",
        "empirical_survival_5deg",
        "single_survival_5deg",
        "double_survival_5deg",
        "empirical_survival_45deg",
        "single_survival_45deg",
        "double_survival_45deg",
        "empirical_survival_90deg",
        "single_survival_90deg",
        "double_survival_90deg",
        "optimizer_success",
        "optimizer_message",
        "boundary_flags",
    ]
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        fit = row["single_fit"]
        empirical_at = np.asarray(row["empirical_survival_at_thresholds"], dtype=np.float64)
        single_at = np.asarray(row["single_survival_at_thresholds"], dtype=np.float64)
        double_at = np.asarray(row["double_survival_at_thresholds"], dtype=np.float64)
        csv_rows.append(
            {
                "cell_id": row["cell_id"],
                "nhit_bin": row["nhit_bin"],
                "predE_bin": row["predE_bin"],
                "effective_events": row["effective_events"],
                "sigma_deg": fit["sigma_deg"],
                "gamma": fit["gamma"],
                "single_kl_divergence": row["single_kl_divergence"],
                "double_kl_divergence": row["double_kl_divergence"],
                "single_aic_relative_saturated": row["single_aic_relative_saturated"],
                "double_aic_relative_saturated": row["double_aic_relative_saturated"],
                "delta_aic_single_minus_double": row["delta_aic_single_minus_double"],
                "single_bic_relative_saturated": row["single_bic_relative_saturated"],
                "double_bic_relative_saturated": row["double_bic_relative_saturated"],
                "delta_bic_single_minus_double": row["delta_bic_single_minus_double"],
                "single_survival_log_rmse": row["single_survival_log_rmse"],
                "double_survival_log_rmse": row["double_survival_log_rmse"],
                "empirical_survival_5deg": empirical_at[2],
                "single_survival_5deg": single_at[2],
                "double_survival_5deg": double_at[2],
                "empirical_survival_45deg": empirical_at[5],
                "single_survival_45deg": single_at[5],
                "double_survival_45deg": double_at[5],
                "empirical_survival_90deg": empirical_at[6],
                "single_survival_90deg": single_at[6],
                "double_survival_90deg": double_at[6],
                "optimizer_success": fit["optimizer_success"],
                "optimizer_message": fit["optimizer_message"],
                "boundary_flags": ";".join(fit["boundary_flags"]),
            }
        )
    with outputs["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    metadata_rows = []
    excluded = {
        "data_probability",
        "empirical_survival",
        "single_probability",
        "single_survival",
        "double_probability",
        "double_survival",
    }
    for row in rows:
        metadata_rows.append(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in row.items()
                if key not in excluded
            }
        )
    output_metadata = {
        "description": "Cells 1-3 full-sphere single spherical-King fits compared with double spherical-King",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "single_king_parameter_count": 2,
        "double_king_parameter_count": 5,
        "normalization_support_deg": [0.0, 180.0],
        "information_criteria": {
            "aic_relative_saturated": "2*k + 2*N_eff*KL",
            "bic_relative_saturated": "k*ln(N_eff) + 2*N_eff*KL",
        },
        "quadrature": f"Gauss-Legendre order {args.quadrature_order} per radial bin",
        "inputs": {
            "npz": str(args.input_npz.resolve()),
            "npz_sha256": path_sha256(args.input_npz),
            "metadata": str(args.input_metadata.resolve()),
            "metadata_sha256": path_sha256(args.input_metadata),
        },
        "cells": metadata_rows,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": metadata_rows, "outputs": output_metadata["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
