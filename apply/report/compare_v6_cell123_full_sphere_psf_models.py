#!/usr/bin/env python3
"""Compare full-sphere Rayleigh+King, double-Rayleigh, and double-King fits."""

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

from apply.stages.psf_full_sphere_mixtures import (  # noqa: E402
    double_rayleigh_cdf,
    double_spherical_king_cdf,
    fit_double_rayleigh_counts,
    fit_double_spherical_king_counts,
)


BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
INPUT_RUN_LABEL = "v6-64748-nhit100-reselect44-split56-miss030-spherical-rayleigh-king-fit"
OUTPUT_RUN_LABEL = "v6-64748-nhit100-reselect44-split56-miss030-full-sphere-model-comparison"
INPUT_STEM = f"{BASE_RUN_ID}_cell123_spherical_rayleigh_king_fit"
OUTPUT_STEM = f"{BASE_RUN_ID}_cell123_full_sphere_psf_model_comparison"

DEFAULT_INPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / INPUT_RUN_LABEL
DEFAULT_INPUT_NPZ = DEFAULT_INPUT_DIR / f"{INPUT_STEM}.npz"
DEFAULT_INPUT_METADATA = DEFAULT_INPUT_DIR / f"{INPUT_STEM}_metadata.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / OUTPUT_RUN_LABEL

MODEL_ORDER = ("rayleigh_king", "double_rayleigh", "double_spherical_king")
MODEL_LABELS = {
    "rayleigh_king": "Rayleigh + spherical King",
    "double_rayleigh": "double-Rayleigh",
    "double_spherical_king": "double spherical-King",
}
MODEL_SHORT = {
    "rayleigh_king": "R+K",
    "double_rayleigh": "2R",
    "double_spherical_king": "2K",
}
MODEL_PARAMETERS = {
    "rayleigh_king": 4,
    "double_rayleigh": 3,
    "double_spherical_king": 5,
}
COLORS = {
    "data": "#0072B2",
    "rayleigh_king": "#009E73",
    "double_rayleigh": "#CC79A7",
    "double_spherical_king": "#D55E00",
    "cut": "#111827",
}
LINESTYLES = {
    "rayleigh_king": "-",
    "double_rayleigh": "-.",
    "double_spherical_king": "--",
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
    hybrid_probability = np.asarray(task["hybrid_probability"], dtype=np.float64)
    hybrid_survival = np.asarray(task["hybrid_survival"], dtype=np.float64)
    effective_events = float(task["effective_events"])

    double_rayleigh_fit, double_rayleigh_probability = fit_double_rayleigh_counts(
        data_probability,
        edges,
        random_seed=int(task["random_seed"]) + 1000,
        random_starts=int(task["random_starts"]),
    )
    double_king_fit, double_king_probability = fit_double_spherical_king_counts(
        data_probability,
        edges,
        random_seed=int(task["random_seed"]) + 2000,
        random_starts=int(task["random_starts"]),
        quadrature_order=int(task["quadrature_order"]),
    )
    double_rayleigh_survival = 1.0 - double_rayleigh_cdf(
        grid,
        double_rayleigh_fit.core_fraction,
        double_rayleigh_fit.sigma_core_deg,
        double_rayleigh_fit.sigma_tail_deg,
    )
    double_king_survival = 1.0 - double_spherical_king_cdf(
        grid,
        double_king_fit.core_fraction,
        double_king_fit.sigma_core_deg,
        double_king_fit.gamma_core,
        double_king_fit.sigma_tail_deg,
        double_king_fit.gamma_tail,
        quadrature_order=int(task["quadrature_order"]),
    )
    double_rayleigh_at_thresholds = 1.0 - double_rayleigh_cdf(
        thresholds,
        double_rayleigh_fit.core_fraction,
        double_rayleigh_fit.sigma_core_deg,
        double_rayleigh_fit.sigma_tail_deg,
    )
    double_king_at_thresholds = 1.0 - double_spherical_king_cdf(
        thresholds,
        double_king_fit.core_fraction,
        double_king_fit.sigma_core_deg,
        double_king_fit.gamma_core,
        double_king_fit.sigma_tail_deg,
        double_king_fit.gamma_tail,
        quadrature_order=int(task["quadrature_order"]),
    )

    models: dict[str, dict[str, Any]] = {
        "rayleigh_king": {
            "probability": hybrid_probability,
            "survival": hybrid_survival,
            "survival_at_thresholds": np.asarray(task["hybrid_survival_at_thresholds"], dtype=np.float64),
            "fit": dict(task["hybrid_fit"]),
            "kl_divergence": float(task["hybrid_kl_divergence"]),
        },
        "double_rayleigh": {
            "probability": double_rayleigh_probability,
            "survival": np.clip(double_rayleigh_survival, 0.0, 1.0),
            "survival_at_thresholds": np.clip(double_rayleigh_at_thresholds, 0.0, 1.0),
            "fit": double_rayleigh_fit.to_dict(),
            "kl_divergence": double_rayleigh_fit.kl_divergence,
        },
        "double_spherical_king": {
            "probability": double_king_probability,
            "survival": np.clip(double_king_survival, 0.0, 1.0),
            "survival_at_thresholds": np.clip(double_king_at_thresholds, 0.0, 1.0),
            "fit": double_king_fit.to_dict(),
            "kl_divergence": double_king_fit.kl_divergence,
        },
    }
    for name, model in models.items():
        parameters = MODEL_PARAMETERS[name]
        kl_value = float(model["kl_divergence"])
        model["parameter_count"] = parameters
        model["aic_relative_saturated"] = 2.0 * parameters + 2.0 * effective_events * kl_value
        model["bic_relative_saturated"] = (
            math.log(max(effective_events, 1.0)) * parameters + 2.0 * effective_events * kl_value
        )
        model["survival_log_rmse"] = survival_log_rmse(
            empirical_survival,
            np.asarray(model["survival"], dtype=np.float64),
            grid,
            effective_events,
        )

    for criterion in ("kl_divergence", "aic_relative_saturated", "bic_relative_saturated", "survival_log_rmse"):
        minimum = min(float(models[name][criterion]) for name in MODEL_ORDER)
        for name in MODEL_ORDER:
            models[name][f"delta_{criterion}"] = float(models[name][criterion]) - minimum

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
        "models": models,
        "best_kl_model": min(MODEL_ORDER, key=lambda name: models[name]["kl_divergence"]),
        "best_aic_model": min(MODEL_ORDER, key=lambda name: models[name]["aic_relative_saturated"]),
        "best_bic_model": min(MODEL_ORDER, key=lambda name: models[name]["bic_relative_saturated"]),
        "best_survival_model": min(MODEL_ORDER, key=lambda name: models[name]["survival_log_rmse"]),
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
        top.stairs(
            data_density,
            edges,
            color=COLORS["data"],
            linewidth=1.15,
            label="weighted raw MC",
            zorder=2,
        )
        for name in MODEL_ORDER:
            model_density = np.asarray(row["models"][name]["probability"], dtype=np.float64) / widths
            top.stairs(
                model_density,
                edges,
                color=COLORS[name],
                linewidth=1.85,
                linestyle=LINESTYLES[name],
                label=MODEL_LABELS[name],
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
        best_name = str(row["best_aic_model"])
        sorted_aic = sorted(float(row["models"][name]["aic_relative_saturated"]) for name in MODEL_ORDER)
        next_delta_aic = sorted_aic[1] - sorted_aic[0]
        annotation = ["full-sphere KL (lower is better)"]
        for name in MODEL_ORDER:
            annotation.append(f"{MODEL_SHORT[name]:>3s}: {row['models'][name]['kl_divergence']:.4e}")
        annotation.append(f"AIC best: {MODEL_SHORT[best_name]} (next +{next_delta_aic:.1f})")
        top.text(
            0.97,
            0.96,
            "\n".join(annotation),
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            family="monospace",
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        )

        empirical = np.asarray(row["empirical_survival"], dtype=np.float64)
        bottom.plot(
            grid[empirical > 0.0],
            empirical[empirical > 0.0],
            color=COLORS["data"],
            linewidth=1.65,
            label="empirical survival",
            zorder=2,
        )
        for name in MODEL_ORDER:
            survival = np.asarray(row["models"][name]["survival"], dtype=np.float64)
            bottom.plot(
                grid[survival > 0.0],
                survival[survival > 0.0],
                color=COLORS[name],
                linewidth=1.85,
                linestyle=LINESTYLES[name],
                label=f"{MODEL_LABELS[name]} survival",
                zorder=3,
            )
        bottom.axvline(5.0, color=COLORS["cut"], linewidth=0.85, linestyle=(0, (2, 2)))
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set_xlim(0.03, 180.0)
        bottom.set_ylim(1.0e-7, 1.15)
        bottom.set_xlabel("angular error, r [deg]")
        bottom.text(
            0.04,
            0.06,
            "log-survival RMSE\n"
            + "\n".join(
                f"{MODEL_SHORT[name]:>3s}: {row['models'][name]['survival_log_rmse']:.3f}"
                for name in MODEL_ORDER
            ),
            transform=bottom.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            family="monospace",
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
        )

    axes[0, 0].set_ylabel("weighted radial density [deg$^{-1}$]")
    axes[1, 0].set_ylabel(r"$S(r)=P(\Delta\theta>r)$")
    handles: list[Any] = []
    labels: list[str] = []
    for axis in (axes[0, 0], axes[1, 0]):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels and " survival" not in label:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Cells 1-3 full-sphere PSF model comparison", fontsize=12, y=0.995)
    fig.text(
        0.5,
        0.962,
        "All three models are refitted to the same 0-180 deg Stage-B-weighted raw-MC distribution.",
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
        raise FileNotFoundError(f"Missing comparison inputs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_metadata = json.loads(args.input_metadata.read_text(encoding="utf-8"))
    metadata_by_cell = {int(row["cell_id"]): row for row in input_metadata["cells"]}
    with np.load(args.input_npz, allow_pickle=False) as handle:
        source = {key: np.asarray(handle[key]) for key in handle.files}

    edges = np.asarray(source["fit_edges_deg"], dtype=np.float64)
    grid = np.asarray(source["survival_grid_deg"], dtype=np.float64)
    thresholds = np.asarray(source["survival_thresholds_deg"], dtype=np.float64)
    tasks: list[dict[str, Any]] = []
    for index, cell_value in enumerate(np.asarray(source["cell_id"], dtype=np.int64)):
        cell_id = int(cell_value)
        cell_metadata = metadata_by_cell[cell_id]
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
                "hybrid_probability": np.asarray(source["model_probability"][index], dtype=np.float64),
                "hybrid_survival": np.asarray(source["model_survival"][index], dtype=np.float64),
                "hybrid_survival_at_thresholds": np.asarray(
                    source["model_survival_at_thresholds"][index],
                    dtype=np.float64,
                ),
                "hybrid_kl_divergence": float(cell_metadata["kl_divergence"]),
                "hybrid_fit": {
                    "core_fraction": float(cell_metadata["core_fraction"]),
                    "sigma_core_deg": float(cell_metadata["sigma_rayleigh_deg"]),
                    "gamma_core": None,
                    "sigma_tail_deg": float(cell_metadata["sigma_king_deg"]),
                    "gamma_tail": float(cell_metadata["gamma_king"]),
                    "optimizer_success": bool(cell_metadata["optimizer_success"]),
                    "optimizer_message": str(cell_metadata["optimizer_message"]),
                    "boundary_flags": list(cell_metadata["boundary_flags"]),
                },
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
            summary = ", ".join(
                f"{MODEL_SHORT[name]} KL={row['models'][name]['kl_divergence']:.4e}"
                for name in MODEL_ORDER
            )
            print(f"cell {row['cell_id']} complete: {summary}", flush=True)
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
        empirical_survival_at_thresholds=np.vstack(
            [row["empirical_survival_at_thresholds"] for row in rows]
        ),
        rayleigh_king_probability=np.vstack(
            [row["models"]["rayleigh_king"]["probability"] for row in rows]
        ),
        double_rayleigh_probability=np.vstack(
            [row["models"]["double_rayleigh"]["probability"] for row in rows]
        ),
        double_spherical_king_probability=np.vstack(
            [row["models"]["double_spherical_king"]["probability"] for row in rows]
        ),
        rayleigh_king_survival=np.vstack(
            [row["models"]["rayleigh_king"]["survival"] for row in rows]
        ),
        double_rayleigh_survival=np.vstack(
            [row["models"]["double_rayleigh"]["survival"] for row in rows]
        ),
        double_spherical_king_survival=np.vstack(
            [row["models"]["double_spherical_king"]["survival"] for row in rows]
        ),
    )

    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "model",
        "model_label",
        "parameter_count",
        "effective_events",
        "kl_divergence",
        "delta_kl_divergence",
        "aic_relative_saturated",
        "delta_aic_relative_saturated",
        "bic_relative_saturated",
        "delta_bic_relative_saturated",
        "survival_log_rmse",
        "delta_survival_log_rmse",
        "best_kl",
        "best_aic",
        "best_bic",
        "best_survival",
        "core_fraction",
        "sigma_core_deg",
        "gamma_core",
        "sigma_tail_deg",
        "gamma_tail",
        "survival_5deg",
        "survival_45deg",
        "survival_90deg",
        "optimizer_success",
        "optimizer_message",
        "boundary_flags",
    ]
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        for name in MODEL_ORDER:
            model = row["models"][name]
            fit = model["fit"]
            survival_at = np.asarray(model["survival_at_thresholds"], dtype=np.float64)
            csv_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "nhit_bin": row["nhit_bin"],
                    "predE_bin": row["predE_bin"],
                    "model": name,
                    "model_label": MODEL_LABELS[name],
                    "parameter_count": model["parameter_count"],
                    "effective_events": row["effective_events"],
                    "kl_divergence": model["kl_divergence"],
                    "delta_kl_divergence": model["delta_kl_divergence"],
                    "aic_relative_saturated": model["aic_relative_saturated"],
                    "delta_aic_relative_saturated": model["delta_aic_relative_saturated"],
                    "bic_relative_saturated": model["bic_relative_saturated"],
                    "delta_bic_relative_saturated": model["delta_bic_relative_saturated"],
                    "survival_log_rmse": model["survival_log_rmse"],
                    "delta_survival_log_rmse": model["delta_survival_log_rmse"],
                    "best_kl": name == row["best_kl_model"],
                    "best_aic": name == row["best_aic_model"],
                    "best_bic": name == row["best_bic_model"],
                    "best_survival": name == row["best_survival_model"],
                    "core_fraction": fit.get("core_fraction"),
                    "sigma_core_deg": fit.get("sigma_core_deg"),
                    "gamma_core": fit.get("gamma_core"),
                    "sigma_tail_deg": fit.get("sigma_tail_deg"),
                    "gamma_tail": fit.get("gamma_tail"),
                    "survival_5deg": survival_at[2],
                    "survival_45deg": survival_at[5],
                    "survival_90deg": survival_at[6],
                    "optimizer_success": fit.get("optimizer_success"),
                    "optimizer_message": fit.get("optimizer_message"),
                    "boundary_flags": ";".join(fit.get("boundary_flags", [])),
                }
            )
    with outputs["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    metadata_rows = []
    for row in rows:
        metadata_rows.append(
            {
                "cell_id": row["cell_id"],
                "nhit_bin": row["nhit_bin"],
                "predE_bin": row["predE_bin"],
                "effective_events": row["effective_events"],
                "best_kl_model": row["best_kl_model"],
                "best_aic_model": row["best_aic_model"],
                "best_bic_model": row["best_bic_model"],
                "best_survival_model": row["best_survival_model"],
                "models": {
                    name: {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in row["models"][name].items()
                        if key not in {"probability", "survival"}
                    }
                    for name in MODEL_ORDER
                },
            }
        )
    metadata = {
        "description": "Fair full-sphere comparison of Rayleigh+King, double-Rayleigh, and double spherical-King for cells 1-3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "comparison_support_deg": [0.0, 180.0],
        "common_data": "Stage-B weighted raw-MC angular distances with no angular truncation or 5-degree renormalization",
        "model_parameter_counts": MODEL_PARAMETERS,
        "selection_criteria": {
            "kl_divergence": "binned distribution KL; lower is better",
            "aic_relative_saturated": "2*k + 2*N_eff*KL; common saturated-likelihood constant omitted",
            "bic_relative_saturated": "k*ln(N_eff) + 2*N_eff*KL; common saturated-likelihood constant omitted",
            "survival_log_rmse": "RMSE of log10(model survival)-log10(empirical survival) where empirical survival >= 1/N_eff",
        },
        "quadrature": f"Gauss-Legendre order {args.quadrature_order} per spherical-King radial bin",
        "inputs": {
            "npz": str(args.input_npz.resolve()),
            "npz_sha256": path_sha256(args.input_npz),
            "metadata": str(args.input_metadata.resolve()),
            "metadata_sha256": path_sha256(args.input_metadata),
        },
        "cells": metadata_rows,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": metadata_rows, "outputs": metadata["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
