#!/usr/bin/env python3
"""Compare one- and two-Rayleigh fits for seven predE-collapsed Nhit PSFs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize

from apply.report.fit_v6_nhit_folded_rayleigh import (
    BASE_RUN_ID,
    conditional_rayleigh_probability,
    fit_rayleigh_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_STEM = f"{BASE_RUN_ID}_nhit_folded_rayleigh"
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "apply"
    / "report"
    / "assets"
    / f"{BASE_RUN_ID.replace('_', '-')}-nhit-folded-rayleigh-exact"
)
DEFAULT_INPUT_NPZ = DEFAULT_INPUT_DIR / f"{INPUT_STEM}.npz"
OUTPUT_STEM = f"{BASE_RUN_ID}_nhit_folded_1r_2r_comparison"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / OUTPUT_STEM.replace("_", "-")

MC_COLOR = "#0072B2"
ONE_R_COLOR = "#D55E00"
TWO_R_COLOR = "#009E73"
GRID_COLOR = "#D7DCE1"


@dataclass(frozen=True)
class ConditionalDoubleRayleighFit:
    conditional_core_fraction: float
    sigma_core_deg: float
    sigma_tail_deg: float
    sigma_ratio: float
    kl_divergence: float
    total_variation: float
    max_cdf_difference: float
    multinomial_deviance: float
    multinomial_ndof: int
    optimizer_success: bool
    optimizer_message: str
    boundary_flags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-starts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _logistic(value: float) -> float:
    if value >= 0.0:
        exp_minus = math.exp(-value)
        return 1.0 / (1.0 + exp_minus)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def conditional_double_rayleigh_probability(
    edges_deg: np.ndarray,
    conditional_core_fraction: float,
    sigma_core_deg: float,
    sigma_tail_deg: float,
) -> np.ndarray:
    if not 0.0 < conditional_core_fraction < 1.0:
        raise ValueError("conditional_core_fraction must lie in (0, 1)")
    if not 0.0 < sigma_core_deg < sigma_tail_deg:
        raise ValueError("Require 0 < sigma_core_deg < sigma_tail_deg")
    core = conditional_rayleigh_probability(edges_deg, sigma_core_deg)
    tail = conditional_rayleigh_probability(edges_deg, sigma_tail_deg)
    return conditional_core_fraction * core + (1.0 - conditional_core_fraction) * tail


def fit_conditional_double_rayleigh(
    data_probability: np.ndarray,
    edges_deg: np.ndarray,
    *,
    effective_events: float = math.nan,
    one_rayleigh_sigma_deg: float | None = None,
    random_seed: int = 0,
    random_starts: int = 32,
) -> tuple[ConditionalDoubleRayleighFit, np.ndarray]:
    data = np.asarray(data_probability, dtype=np.float64)
    edges = np.asarray(edges_deg, dtype=np.float64)
    if data.shape != (edges.size - 1,) or np.any(~np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("data_probability is invalid")
    data_sum = float(np.sum(data))
    if data_sum <= 0.0:
        raise ValueError("data_probability has no mass")
    data = data / data_sum
    sigma_reference = float(one_rayleigh_sigma_deg or 0.25 * edges[-1])
    sigma_min = max(float(np.min(np.diff(edges))) * 0.02, 1.0e-5)
    sigma_core_max = max(float(edges[-1]), sigma_reference * 4.0)
    fraction_bounds = (1.0e-4, 1.0 - 1.0e-4)
    ratio_bounds = (1.001, 1000.0)
    bounds = (
        (_logit(fraction_bounds[0]), _logit(fraction_bounds[1])),
        (math.log(sigma_min), math.log(sigma_core_max)),
        (math.log(ratio_bounds[0]), math.log(ratio_bounds[1])),
    )

    def unpack(parameters: np.ndarray) -> tuple[float, float, float]:
        fraction = _logistic(float(parameters[0]))
        sigma_core = math.exp(float(parameters[1]))
        sigma_tail = sigma_core * math.exp(float(parameters[2]))
        return fraction, sigma_core, sigma_tail

    def objective(parameters: np.ndarray) -> float:
        fraction, sigma_core, sigma_tail = unpack(parameters)
        model = conditional_double_rayleigh_probability(edges, fraction, sigma_core, sigma_tail)
        return float(-np.sum(data * np.log(np.clip(model, 1.0e-300, None))))

    starts = [
        np.asarray([_logit(0.80), math.log(max(sigma_min, 0.55 * sigma_reference)), math.log(3.0)]),
        np.asarray([_logit(0.90), math.log(max(sigma_min, 0.40 * sigma_reference)), math.log(5.0)]),
        np.asarray([_logit(0.65), math.log(max(sigma_min, 0.70 * sigma_reference)), math.log(8.0)]),
        np.asarray([_logit(0.50), math.log(max(sigma_min, 0.35 * sigma_reference)), math.log(15.0)]),
    ]
    rng = np.random.default_rng(int(random_seed))
    for _ in range(max(0, int(random_starts) - len(starts))):
        starts.append(
            np.asarray(
                [
                    rng.uniform(_logit(0.15), _logit(0.98)),
                    rng.uniform(math.log(sigma_min), math.log(max(sigma_min * 1.01, sigma_reference * 1.5))),
                    rng.uniform(math.log(1.05), math.log(80.0)),
                ]
            )
        )

    results = []
    for start in starts:
        clipped_start = np.asarray(
            [np.clip(value, lower, upper) for value, (lower, upper) in zip(start, bounds)],
            dtype=np.float64,
        )
        result = minimize(
            objective,
            clipped_start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 4000, "ftol": 1.0e-14, "gtol": 1.0e-9},
        )
        if np.isfinite(result.fun):
            results.append(result)
    if not results:
        raise RuntimeError("Double-Rayleigh optimization did not produce a finite result")
    best = min(results, key=lambda result: float(result.fun))
    successful = [
        result
        for result in results
        if result.success and float(result.fun) <= float(best.fun) + 1.0e-9
    ]
    if successful:
        best = min(successful, key=lambda result: float(result.fun))

    fraction, sigma_core, sigma_tail = unpack(np.asarray(best.x, dtype=np.float64))
    model = conditional_double_rayleigh_probability(edges, fraction, sigma_core, sigma_tail)
    mask = data > 0.0
    kl = float(np.sum(data[mask] * np.log(data[mask] / np.clip(model[mask], 1.0e-300, None))))
    boundary_flags: list[str] = []
    checks = (
        ("conditional_core_fraction", fraction, *fraction_bounds),
        ("sigma_core", sigma_core, sigma_min, sigma_core_max),
        ("sigma_ratio", sigma_tail / sigma_core, *ratio_bounds),
    )
    for name, value, lower, upper in checks:
        scale = max(upper - lower, 1.0)
        if value - lower < 1.0e-4 * scale:
            boundary_flags.append(f"{name}:lower")
        if upper - value < 1.0e-4 * scale:
            boundary_flags.append(f"{name}:upper")
    fit = ConditionalDoubleRayleighFit(
        conditional_core_fraction=fraction,
        sigma_core_deg=sigma_core,
        sigma_tail_deg=sigma_tail,
        sigma_ratio=sigma_tail / sigma_core,
        kl_divergence=kl,
        total_variation=0.5 * float(np.sum(np.abs(data - model))),
        max_cdf_difference=float(np.max(np.abs(np.cumsum(data) - np.cumsum(model)))),
        multinomial_deviance=2.0 * effective_events * kl if np.isfinite(effective_events) else math.nan,
        multinomial_ndof=max(0, data.size - 4),
        optimizer_success=bool(best.success),
        optimizer_message=str(best.message),
        boundary_flags=tuple(boundary_flags),
    )
    return fit, model


def load_input(path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "nhit_bin",
            "profile_edges_deg",
            "profile_probability",
            "rayleigh_probability",
            "sigma_deg",
            "effective_events",
            "total_baseline_weight",
            "n_cells",
            "n_used_cells",
            "cell_ids",
            "used_cell_ids",
            "excluded_cell_ids",
        }
        missing = required.difference(source.files)
        if missing:
            raise KeyError(f"Nhit-folded input NPZ is missing arrays: {sorted(missing)}")
        edges = np.asarray(source["profile_edges_deg"], dtype=np.float64)
        rows = []
        for index, label in enumerate(source["nhit_bin"]):
            rows.append(
                {
                    "nhit_bin": str(label),
                    "data_probability": np.asarray(source["profile_probability"][index], dtype=np.float64),
                    "stored_one_r_probability": np.asarray(
                        source["rayleigh_probability"][index], dtype=np.float64
                    ),
                    "stored_one_r_sigma_deg": float(source["sigma_deg"][index]),
                    "effective_events": float(source["effective_events"][index]),
                    "total_baseline_weight": float(source["total_baseline_weight"][index]),
                    "n_cells": int(source["n_cells"][index]),
                    "n_used_cells": int(source["n_used_cells"][index]),
                    "cell_ids": str(source["cell_ids"][index]),
                    "used_cell_ids": str(source["used_cell_ids"][index]),
                    "excluded_cell_ids": str(source["excluded_cell_ids"][index]),
                }
            )
    if len(rows) != 7:
        raise ValueError(f"Expected seven Nhit profiles, found {len(rows)}")
    return edges, rows


def fit_rows(
    rows: list[dict[str, object]],
    edges_deg: np.ndarray,
    *,
    random_seed: int,
    random_starts: int,
) -> None:
    for index, row in enumerate(rows):
        data = np.asarray(row["data_probability"], dtype=np.float64)
        neff = float(row["effective_events"])
        one_fit, one_model = fit_rayleigh_profile(data, edges_deg, neff)
        stored_model = np.asarray(row.pop("stored_one_r_probability"), dtype=np.float64)
        if not np.allclose(one_model, stored_model, rtol=0.0, atol=5.0e-10):
            raise ValueError(f"Recomputed 1R model does not match stored input for {row['nhit_bin']}")
        two_fit, two_model = fit_conditional_double_rayleigh(
            data,
            edges_deg,
            effective_events=neff,
            one_rayleigh_sigma_deg=one_fit.sigma_deg,
            random_seed=int(random_seed) + index,
            random_starts=int(random_starts),
        )
        row.update(
            {
                "one_r_fit": one_fit,
                "one_r_probability": one_model,
                "two_r_fit": two_fit,
                "two_r_probability": two_model,
                "kl_improvement_factor": one_fit.kl_divergence / two_fit.kl_divergence,
                "kl_reduction_fraction": 1.0 - two_fit.kl_divergence / one_fit.kl_divergence,
                "delta_aic_2r_minus_1r": 4.0
                + 2.0 * neff * (two_fit.kl_divergence - one_fit.kl_divergence),
                "delta_bic_2r_minus_1r": 2.0 * math.log(neff)
                + 2.0 * neff * (two_fit.kl_divergence - one_fit.kl_divergence),
            }
        )


def draw_comparison_grid(
    rows: list[dict[str, object]],
    edges_deg: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.bbox": "tight",
        }
    )
    widths = np.diff(edges_deg)
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True)
    flat_axes = list(axes.flat)
    for axis, row in zip(flat_axes, rows):
        data_density = np.asarray(row["data_probability"], dtype=np.float64) / widths
        one_density = np.asarray(row["one_r_probability"], dtype=np.float64) / widths
        two_density = np.asarray(row["two_r_probability"], dtype=np.float64) / widths
        one_fit = row["one_r_fit"]
        two_fit = row["two_r_fit"]
        axis.stairs(data_density, edges_deg, color=MC_COLOR, linewidth=1.65, label="Weighted MC")
        axis.stairs(one_density, edges_deg, color=ONE_R_COLOR, linewidth=1.35, linestyle="--", label="1R")
        axis.stairs(two_density, edges_deg, color=TWO_R_COLOR, linewidth=1.55, label="2R")
        axis.set_yscale("log")
        axis.set_xlim(float(edges_deg[0]), float(edges_deg[-1]))
        positive = np.concatenate(
            (data_density[data_density > 0.0], one_density[one_density > 0.0], two_density[two_density > 0.0])
        )
        axis.set_ylim(max(float(np.min(positive)) * 0.55, 1.0e-6), float(np.max(positive)) * 2.0)
        axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.55, alpha=0.75)
        axis.grid(True, which="minor", axis="y", color=GRID_COLOR, linewidth=0.35, alpha=0.35)
        axis.set_title(
            f"Nhit {row['nhit_bin']}  |  {row['n_used_cells']}/{row['n_cells']} predE cells",
            fontsize=9.2,
            pad=5,
        )
        annotation = (
            rf"1R: $D_{{KL}}={one_fit.kl_divergence:.3f}$" + "\n"
            + rf"2R: $D_{{KL}}={two_fit.kl_divergence:.3f}$" + "\n"
            + rf"$f_c={two_fit.conditional_core_fraction:.2f}$, "
            + rf"$\sigma_c={two_fit.sigma_core_deg:.2f}\degree$, "
            + rf"$\sigma_t={two_fit.sigma_tail_deg:.2f}\degree$"
        )
        axis.text(
            0.96,
            0.95,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.25,
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "boxstyle": "square,pad=0.28", "alpha": 0.9},
        )
    flat_axes[-1].axis("off")
    for axis in flat_axes[:7]:
        axis.set_ylabel(r"Conditional radial density [deg$^{-1}$]")
    for axis in flat_axes[4:7]:
        axis.set_xlabel(r"Angular separation $\Delta\theta$ [deg]")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle("predE-collapsed Stage-B PSF: 1R versus 2R", fontsize=13.0, y=0.998)
    figure.text(
        0.995,
        0.012,
        "All models are conditional on 0-5 deg; predE cells use identical Stage-B sumw_baseline weights.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#4D555D",
    )
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.09, top=0.88, wspace=0.24, hspace=0.25)
    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)


def draw_survival_residual_grid(
    rows: list[dict[str, object]],
    edges_deg: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    figure, axes = plt.subplots(7, 2, figsize=(10.6, 14.2), sharex=True)
    for row_index, row in enumerate(rows):
        data = np.asarray(row["data_probability"], dtype=np.float64)
        one_model = np.asarray(row["one_r_probability"], dtype=np.float64)
        two_model = np.asarray(row["two_r_probability"], dtype=np.float64)
        survival_axis, residual_axis = axes[row_index]
        for probability, color, linestyle, label in (
            (data, MC_COLOR, "-", "Weighted MC"),
            (one_model, ONE_R_COLOR, "--", "1R"),
            (two_model, TWO_R_COLOR, "-", "2R"),
        ):
            survival = 1.0 - np.concatenate(([0.0], np.cumsum(probability)))
            survival_axis.stairs(
                np.clip(survival[:-1], 1.0e-12, None),
                edges_deg,
                color=color,
                linewidth=1.3,
                linestyle=linestyle,
                label=label,
            )
        survival_axis.set_yscale("log")
        survival_axis.set_ylim(1.0e-5, 1.05)
        survival_axis.set_ylabel(str(row["nhit_bin"]), fontsize=8.5)
        survival_axis.grid(True, which="both", color=GRID_COLOR, linewidth=0.4, alpha=0.6)
        for model, color, linestyle, label in (
            (one_model, ONE_R_COLOR, "--", "1R residual"),
            (two_model, TWO_R_COLOR, "-", "2R residual"),
        ):
            log_ratio = np.log10(np.clip(data, 1.0e-12, None) / np.clip(model, 1.0e-12, None))
            residual_axis.plot(centers, log_ratio, color=color, linewidth=1.05, linestyle=linestyle, label=label)
        residual_axis.axhline(0.0, color="#59636E", linewidth=0.75)
        residual_axis.set_ylim(-2.0, 2.0)
        residual_axis.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.65)
    axes[0, 0].set_title(r"Survival $P(\Delta\theta>r)$", fontsize=10.5)
    axes[0, 1].set_title(r"Bin residual $\log_{10}(P_{MC}/P_{model})$", fontsize=10.5)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=7.3, ncol=3)
    axes[0, 1].legend(loc="lower left", frameon=False, fontsize=7.3, ncol=2)
    axes[-1, 0].set_xlabel(r"Angular separation $r$ [deg]")
    axes[-1, 1].set_xlabel(r"Angular separation $r$ [deg]")
    figure.suptitle("Nhit-folded PSF diagnostics: 1R versus 2R", fontsize=13.0, y=0.997)
    figure.text(0.015, 0.5, "Nhit bin", rotation=90, va="center", ha="center", fontsize=9.0)
    figure.subplots_adjust(left=0.105, right=0.99, bottom=0.05, top=0.965, hspace=0.13, wspace=0.18)
    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)


def flattened_row(row: dict[str, object]) -> dict[str, object]:
    one_fit = row["one_r_fit"]
    two_fit = row["two_r_fit"]
    return {
        "nhit_bin": row["nhit_bin"],
        "n_cells": row["n_cells"],
        "n_used_cells": row["n_used_cells"],
        "used_cell_ids": row["used_cell_ids"],
        "excluded_cell_ids": row["excluded_cell_ids"],
        "effective_events": row["effective_events"],
        "one_r_sigma_deg": one_fit.sigma_deg,
        "one_r_kl": one_fit.kl_divergence,
        "one_r_max_cdf_difference": one_fit.max_cdf_difference,
        "two_r_conditional_core_fraction": two_fit.conditional_core_fraction,
        "two_r_sigma_core_deg": two_fit.sigma_core_deg,
        "two_r_sigma_tail_deg": two_fit.sigma_tail_deg,
        "two_r_sigma_ratio": two_fit.sigma_ratio,
        "two_r_kl": two_fit.kl_divergence,
        "two_r_max_cdf_difference": two_fit.max_cdf_difference,
        "kl_improvement_factor": row["kl_improvement_factor"],
        "kl_reduction_fraction": row["kl_reduction_fraction"],
        "delta_aic_2r_minus_1r": row["delta_aic_2r_minus_1r"],
        "delta_bic_2r_minus_1r": row["delta_bic_2r_minus_1r"],
        "two_r_optimizer_success": two_fit.optimizer_success,
        "two_r_boundary_flags": ";".join(two_fit.boundary_flags),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    flat = [flattened_row(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    args = parse_args()
    outputs = {
        "npz": args.output_dir / f"{OUTPUT_STEM}.npz",
        "csv": args.output_dir / f"{OUTPUT_STEM}_summary.csv",
        "metadata": args.output_dir / f"{OUTPUT_STEM}_metadata.json",
        "comparison_png": args.output_dir / f"{OUTPUT_STEM}_fit_grid.png",
        "comparison_pdf": args.output_dir / f"{OUTPUT_STEM}_fit_grid.pdf",
        "diagnostics_png": args.output_dir / f"{OUTPUT_STEM}_survival_residual_grid.png",
        "diagnostics_pdf": args.output_dir / f"{OUTPUT_STEM}_survival_residual_grid.pdf",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing outputs: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges, rows = load_input(args.input_npz)
    fit_rows(rows, edges, random_seed=args.seed, random_starts=args.random_starts)
    write_csv(rows, outputs["csv"])
    draw_comparison_grid(rows, edges, outputs["comparison_png"], outputs["comparison_pdf"])
    draw_survival_residual_grid(rows, edges, outputs["diagnostics_png"], outputs["diagnostics_pdf"])

    flat = [flattened_row(row) for row in rows]
    np.savez_compressed(
        outputs["npz"],
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        profile_edges_deg=edges.astype(np.float32),
        profile_probability=np.vstack([row["data_probability"] for row in rows]),
        one_r_probability=np.vstack([row["one_r_probability"] for row in rows]),
        two_r_probability=np.vstack([row["two_r_probability"] for row in rows]),
        one_r_sigma_deg=np.asarray([item["one_r_sigma_deg"] for item in flat]),
        one_r_kl=np.asarray([item["one_r_kl"] for item in flat]),
        two_r_conditional_core_fraction=np.asarray(
            [item["two_r_conditional_core_fraction"] for item in flat]
        ),
        two_r_sigma_core_deg=np.asarray([item["two_r_sigma_core_deg"] for item in flat]),
        two_r_sigma_tail_deg=np.asarray([item["two_r_sigma_tail_deg"] for item in flat]),
        two_r_kl=np.asarray([item["two_r_kl"] for item in flat]),
        kl_improvement_factor=np.asarray([item["kl_improvement_factor"] for item in flat]),
        delta_aic_2r_minus_1r=np.asarray([item["delta_aic_2r_minus_1r"] for item in flat]),
        delta_bic_2r_minus_1r=np.asarray([item["delta_bic_2r_minus_1r"] for item in flat]),
        effective_events=np.asarray([row["effective_events"] for row in rows]),
        n_used_cells=np.asarray([row["n_used_cells"] for row in rows], dtype=np.int32),
    )
    metadata = {
        "run_id": OUTPUT_STEM,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "definition": {
            "data": "unchanged seven Nhit profiles from exact predE-folded 1R comparison",
            "profile_window_deg": [float(edges[0]), float(edges[-1])],
            "normalization": "conditional_on_0_to_5_deg",
            "one_rayleigh_parameters": 1,
            "two_rayleigh_parameters": 3,
            "two_rayleigh_model": "f_cond*R(sigma_core)+(1-f_cond)*R(sigma_tail)",
            "constraints": "0<f_cond<1 and 0<sigma_core<sigma_tail",
            "fit_objective": "exact-bin multinomial cross-entropy (KL minimization)",
            "model_selection_note": "AIC/BIC use aggregate effective-events as approximate sample size",
        },
        "inputs": {
            "nhit_folded_npz": str(args.input_npz.resolve()),
            "nhit_folded_npz_sha256": path_sha256(args.input_npz),
        },
        "configuration": {"random_starts": int(args.random_starts), "seed": int(args.seed)},
        "summary": {
            "n_nhit_bins": len(rows),
            "n_two_r_optimizer_success": sum(item["two_r_optimizer_success"] for item in flat),
            "n_two_r_boundary_flagged": sum(bool(item["two_r_boundary_flags"]) for item in flat),
            "n_two_r_kl_better": sum(item["two_r_kl"] < item["one_r_kl"] for item in flat),
            "kl_improvement_factor_min": float(min(item["kl_improvement_factor"] for item in flat)),
            "kl_improvement_factor_median": float(np.median([item["kl_improvement_factor"] for item in flat])),
            "kl_improvement_factor_max": float(max(item["kl_improvement_factor"] for item in flat)),
        },
        "fits": [
            {
                **item,
                "one_r_fit": asdict(row["one_r_fit"]),
                "two_r_fit": asdict(row["two_r_fit"]),
            }
            for item, row in zip(flat, rows)
        ],
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2))
    for item in flat:
        print(
            f"Nhit {item['nhit_bin']}: KL 1R={item['one_r_kl']:.6f}, 2R={item['two_r_kl']:.6f}, "
            f"improvement={item['kl_improvement_factor']:.2f}x, "
            f"f={item['two_r_conditional_core_fraction']:.4f}, "
            f"sigma=({item['two_r_sigma_core_deg']:.4f},{item['two_r_sigma_tail_deg']:.4f}) deg"
        )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
