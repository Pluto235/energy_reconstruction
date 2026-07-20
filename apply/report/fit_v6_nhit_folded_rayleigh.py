#!/usr/bin/env python3
"""Collapse v6 Stage-B profiles over predE and fit seven Nhit Rayleigh PSFs."""

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
from scipy.optimize import minimize_scalar


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
STAGE_B_RUN = (
    REPO_ROOT
    / "apply"
    / "output"
    / f"stage_b_{BASE_RUN_ID}"
    / "runs"
    / f"{BASE_RUN_ID}_stage_b_psf"
)
DEFAULT_INPUT_NPZ = STAGE_B_RUN / f"psf_{BASE_RUN_ID}.npz"
DEFAULT_INPUT_METADATA = STAGE_B_RUN / f"psf_{BASE_RUN_ID}_metadata.json"
OUTPUT_STEM = f"{BASE_RUN_ID}_nhit_folded_rayleigh"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / OUTPUT_STEM.replace("_", "-")

MC_COLOR = "#0072B2"
MODEL_COLOR = "#D55E00"
RESIDUAL_COLOR = "#009E73"
GRID_COLOR = "#D7DCE1"


@dataclass(frozen=True)
class FoldedProfile:
    nhit_bin: str
    probability: np.ndarray
    total_baseline_weight: float
    effective_events: float
    cell_ids: tuple[int, ...]
    used_cell_ids: tuple[int, ...]
    excluded_cell_ids: tuple[int, ...]


@dataclass(frozen=True)
class RayleighFit:
    sigma_deg: float
    sigma_error_deg: float
    empirical_r68_deg: float
    model_r68_deg: float
    kl_divergence: float
    total_variation: float
    max_cdf_difference: float
    multinomial_deviance: float
    multinomial_ndof: int
    optimizer_success: bool
    boundary_flag: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def interval_lower(label: str) -> float:
    text = str(label).strip()
    if text.startswith("[") and "," in text:
        return float(text[1:].split(",", 1)[0])
    if text.startswith(">="):
        return float(text[2:])
    raise ValueError(f"Cannot order interval label: {label!r}")


def conditional_rayleigh_probability(edges_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    edges = np.asarray(edges_deg, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges_deg must be a strictly increasing one-dimensional array")
    if not np.isfinite(sigma_deg) or sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be finite and positive")
    exponent = 0.5 * (edges / sigma_deg) ** 2
    survival_left = np.exp(-exponent[:-1])
    exponent_step = np.diff(exponent)
    probability = survival_left * (-np.expm1(-exponent_step))
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("Rayleigh distribution has no mass in the profile window")
    return probability / total


def _profile_probability(density: np.ndarray, edges_deg: np.ndarray) -> np.ndarray | None:
    values = np.asarray(density, dtype=np.float64)
    widths = np.diff(np.asarray(edges_deg, dtype=np.float64))
    if values.shape != widths.shape:
        raise ValueError("profile_density and profile_edges_deg have incompatible shapes")
    probability = np.where(np.isfinite(values) & (values > 0.0), values, 0.0) * widths
    total = float(np.sum(probability))
    return probability / total if total > 0.0 else None


def fold_profiles_by_nhit(
    cell_id: np.ndarray,
    nhit_bin: np.ndarray,
    profile_density: np.ndarray,
    profile_edges_deg: np.ndarray,
    sumw_baseline: np.ndarray,
    effective_events: np.ndarray,
) -> list[FoldedProfile]:
    ids = np.asarray(cell_id, dtype=np.int64)
    labels = np.asarray(nhit_bin).astype(str)
    density = np.asarray(profile_density, dtype=np.float64)
    edges = np.asarray(profile_edges_deg, dtype=np.float64)
    weights = np.asarray(sumw_baseline, dtype=np.float64)
    neff = np.asarray(effective_events, dtype=np.float64)
    n_cells = ids.size
    if labels.shape != (n_cells,) or weights.shape != (n_cells,) or neff.shape != (n_cells,):
        raise ValueError("Cell-level arrays must have matching one-dimensional shapes")
    if density.shape != (n_cells, edges.size - 1):
        raise ValueError("profile_density has incompatible dimensions")
    if np.unique(ids).size != n_cells:
        raise ValueError("cell_id contains duplicates")

    folded: list[FoldedProfile] = []
    for label in sorted(set(labels), key=interval_lower):
        indices = np.flatnonzero(labels == label)
        numerator = np.zeros(edges.size - 1, dtype=np.float64)
        total_weight = 0.0
        sum_weight_squared = 0.0
        used: list[int] = []
        excluded: list[int] = []
        for index in indices:
            probability = _profile_probability(density[index], edges)
            weight = float(weights[index])
            cell_neff = float(neff[index])
            valid = probability is not None and np.isfinite(weight) and weight > 0.0
            if not valid:
                excluded.append(int(ids[index]))
                continue
            numerator += weight * probability
            total_weight += weight
            if np.isfinite(cell_neff) and cell_neff > 0.0:
                sum_weight_squared += weight * weight / cell_neff
            used.append(int(ids[index]))
        if total_weight <= 0.0:
            raise ValueError(f"Nhit bin {label} has no positive weighted Stage-B profiles")
        aggregate_neff = total_weight * total_weight / sum_weight_squared if sum_weight_squared > 0.0 else math.nan
        folded.append(
            FoldedProfile(
                nhit_bin=label,
                probability=numerator / total_weight,
                total_baseline_weight=total_weight,
                effective_events=aggregate_neff,
                cell_ids=tuple(int(ids[index]) for index in indices),
                used_cell_ids=tuple(used),
                excluded_cell_ids=tuple(excluded),
            )
        )
    return folded


def _quantile_from_binned_probability(probability: np.ndarray, edges_deg: np.ndarray, quantile: float) -> float:
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(probability, dtype=np.float64))))
    return float(np.interp(float(quantile), cumulative, np.asarray(edges_deg, dtype=np.float64)))


def _conditional_rayleigh_quantile(sigma_deg: float, max_radius_deg: float, quantile: float) -> float:
    window_mass = -math.expm1(-0.5 * (max_radius_deg / sigma_deg) ** 2)
    return sigma_deg * math.sqrt(-2.0 * math.log1p(-quantile * window_mass))


def fit_rayleigh_profile(
    data_probability: np.ndarray,
    edges_deg: np.ndarray,
    effective_events: float = math.nan,
) -> tuple[RayleighFit, np.ndarray]:
    data = np.asarray(data_probability, dtype=np.float64)
    edges = np.asarray(edges_deg, dtype=np.float64)
    if data.shape != (edges.size - 1,) or np.any(~np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("data_probability is invalid")
    total = float(np.sum(data))
    if total <= 0.0:
        raise ValueError("data_probability has no mass")
    data = data / total
    width_min = float(np.min(np.diff(edges)))
    sigma_min = max(width_min * 1.0e-3, 1.0e-6)
    sigma_max = max(float(edges[-1]) * 100.0, 10.0)
    log_bounds = (math.log(sigma_min), math.log(sigma_max))

    def objective(log_sigma: float) -> float:
        model = conditional_rayleigh_probability(edges, math.exp(log_sigma))
        return float(-np.sum(data * np.log(np.clip(model, 1.0e-300, None))))

    result = minimize_scalar(objective, bounds=log_bounds, method="bounded", options={"xatol": 1.0e-12})
    sigma = math.exp(float(result.x))
    model = conditional_rayleigh_probability(edges, sigma)
    mask = data > 0.0
    kl = float(np.sum(data[mask] * np.log(data[mask] / np.clip(model[mask], 1.0e-300, None))))
    total_variation = 0.5 * float(np.sum(np.abs(data - model)))
    max_cdf_difference = float(np.max(np.abs(np.cumsum(data) - np.cumsum(model))))
    if np.isfinite(effective_events) and effective_events > 0.0:
        multinomial_deviance = 2.0 * effective_events * kl
        h = 1.0e-3
        curvature = (objective(result.x + h) - 2.0 * objective(result.x) + objective(result.x - h)) / h**2
        sigma_error = sigma / math.sqrt(effective_events * curvature) if curvature > 0.0 else math.nan
    else:
        multinomial_deviance = math.nan
        sigma_error = math.nan
    tolerance = 1.0e-4 * (log_bounds[1] - log_bounds[0])
    fit = RayleighFit(
        sigma_deg=sigma,
        sigma_error_deg=sigma_error,
        empirical_r68_deg=_quantile_from_binned_probability(data, edges, 0.68),
        model_r68_deg=_conditional_rayleigh_quantile(sigma, float(edges[-1]), 0.68),
        kl_divergence=kl,
        total_variation=total_variation,
        max_cdf_difference=max_cdf_difference,
        multinomial_deviance=multinomial_deviance,
        multinomial_ndof=max(0, data.size - 2),
        optimizer_success=bool(result.success),
        boundary_flag=bool(result.x - log_bounds[0] < tolerance or log_bounds[1] - result.x < tolerance),
    )
    return fit, model


def load_and_fold(input_npz: Path) -> tuple[np.ndarray, list[FoldedProfile]]:
    with np.load(input_npz, allow_pickle=False) as source:
        required = {
            "cell_id",
            "nhit_bin",
            "profile_edges_deg",
            "profile_density",
            "sumw_baseline",
            "effective_events",
        }
        missing = required.difference(source.files)
        if missing:
            raise KeyError(f"Stage-B NPZ is missing arrays: {sorted(missing)}")
        edges = np.asarray(source["profile_edges_deg"], dtype=np.float64)
        folded = fold_profiles_by_nhit(
            source["cell_id"],
            source["nhit_bin"],
            source["profile_density"],
            edges,
            source["sumw_baseline"],
            source["effective_events"],
        )
    if len(folded) != 7:
        raise ValueError(f"Expected seven Nhit bins, found {len(folded)}")
    return edges, folded


def draw_fit_grid(
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
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True, constrained_layout=False)
    flat_axes = list(axes.flat)
    for axis, row in zip(flat_axes, rows):
        data_density = np.asarray(row["data_probability"], dtype=np.float64) / widths
        model_density = np.asarray(row["model_probability"], dtype=np.float64) / widths
        axis.stairs(data_density, edges_deg, color=MC_COLOR, linewidth=1.65, label="Weighted MC")
        axis.stairs(model_density, edges_deg, color=MODEL_COLOR, linewidth=1.4, label="Rayleigh fit")
        axis.set_yscale("log")
        axis.set_xlim(float(edges_deg[0]), float(edges_deg[-1]))
        positive = np.concatenate((data_density[data_density > 0.0], model_density[model_density > 0.0]))
        axis.set_ylim(max(float(np.min(positive)) * 0.55, 1.0e-6), float(np.max(positive)) * 2.0)
        axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.55, alpha=0.75)
        axis.grid(True, which="minor", axis="y", color=GRID_COLOR, linewidth=0.35, alpha=0.35)
        axis.set_title(
            f"Nhit {row['nhit_bin']}  |  {row['n_used_cells']}/{row['n_cells']} predE cells",
            fontsize=9.2,
            pad=5,
        )
        annotation = (
            rf"$\sigma={float(row['sigma_deg']):.3f}\degree$" + "\n"
            + rf"$D_{{KL}}={float(row['kl_divergence']):.3f}$" + "\n"
            + rf"$\max|\Delta CDF|={float(row['max_cdf_difference']):.3f}$"
        )
        axis.text(
            0.96,
            0.95,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "boxstyle": "square,pad=0.28", "alpha": 0.9},
        )
    flat_axes[-1].axis("off")
    for axis in flat_axes[:4]:
        axis.set_ylabel(r"Conditional radial density [deg$^{-1}$]")
    for axis in flat_axes[4:7]:
        axis.set_xlabel(r"Angular separation $\Delta\theta$ [deg]")
        axis.set_ylabel(r"Conditional radial density [deg$^{-1}$]")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle("predE-collapsed Stage-B PSF: one Rayleigh fit per Nhit bin", fontsize=13.0, y=0.998)
    figure.text(
        0.995,
        0.012,
        "Profiles are conditional on 0-5 deg; predE cells are mixed with Stage-B sumw_baseline.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#4D555D",
    )
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.09, top=0.88, wspace=0.24, hspace=0.25)
    figure.savefig(output_png, dpi=240)
    figure.savefig(output_pdf)
    plt.close(figure)


def draw_survival_residual_grid(rows: list[dict[str, object]], edges_deg: np.ndarray, output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    figure, axes = plt.subplots(7, 2, figsize=(10.6, 14.2), sharex=True)
    for row_index, row in enumerate(rows):
        data = np.asarray(row["data_probability"], dtype=np.float64)
        model = np.asarray(row["model_probability"], dtype=np.float64)
        data_survival = 1.0 - np.concatenate(([0.0], np.cumsum(data)))
        model_survival = 1.0 - np.concatenate(([0.0], np.cumsum(model)))
        survival_axis, residual_axis = axes[row_index]
        survival_axis.stairs(data_survival[:-1], edges_deg, color=MC_COLOR, linewidth=1.45, label="Weighted MC")
        survival_axis.stairs(model_survival[:-1], edges_deg, color=MODEL_COLOR, linewidth=1.25, label="Rayleigh")
        survival_axis.set_yscale("log")
        survival_axis.set_ylim(1.0e-5, 1.05)
        survival_axis.set_ylabel(str(row["nhit_bin"]), fontsize=8.5)
        survival_axis.grid(True, which="both", color=GRID_COLOR, linewidth=0.4, alpha=0.6)
        log_ratio = np.log10(np.clip(data, 1.0e-12, None) / np.clip(model, 1.0e-12, None))
        residual_axis.axhline(0.0, color="#59636E", linewidth=0.75)
        residual_axis.plot(centers, log_ratio, color=RESIDUAL_COLOR, linewidth=1.0)
        residual_axis.fill_between(centers, 0.0, log_ratio, color=RESIDUAL_COLOR, alpha=0.18)
        residual_axis.set_ylim(-2.0, 2.0)
        residual_axis.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.65)
    axes[0, 0].set_title(r"Survival $P(\Delta\theta>r)$", fontsize=10.5)
    axes[0, 1].set_title(r"Bin residual $\log_{10}(P_{MC}/P_{Rayleigh})$", fontsize=10.5)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=7.5)
    axes[-1, 0].set_xlabel(r"Angular separation $r$ [deg]")
    axes[-1, 1].set_xlabel(r"Angular separation $r$ [deg]")
    figure.suptitle("Nhit-folded Rayleigh fit diagnostics", fontsize=13.0, y=0.997)
    figure.text(0.015, 0.5, "Nhit bin", rotation=90, va="center", ha="center", fontsize=9.0)
    figure.subplots_adjust(left=0.105, right=0.99, bottom=0.05, top=0.965, hspace=0.13, wspace=0.18)
    figure.savefig(output_png, dpi=220)
    plt.close(figure)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "nhit_bin",
        "n_cells",
        "n_used_cells",
        "used_cell_ids",
        "excluded_cell_ids",
        "total_baseline_weight",
        "effective_events",
        "sigma_deg",
        "sigma_error_deg",
        "empirical_r68_deg",
        "model_r68_deg",
        "kl_divergence",
        "total_variation",
        "max_cdf_difference",
        "multinomial_deviance",
        "multinomial_ndof",
        "multinomial_deviance_per_ndof",
        "optimizer_success",
        "boundary_flag",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    args = parse_args()
    outputs = {
        "npz": args.output_dir / f"{OUTPUT_STEM}.npz",
        "csv": args.output_dir / f"{OUTPUT_STEM}_summary.csv",
        "metadata": args.output_dir / f"{OUTPUT_STEM}_metadata.json",
        "fit_grid_png": args.output_dir / f"{OUTPUT_STEM}_fit_grid.png",
        "fit_grid_pdf": args.output_dir / f"{OUTPUT_STEM}_fit_grid.pdf",
        "diagnostics_png": args.output_dir / f"{OUTPUT_STEM}_survival_residual_grid.png",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing outputs: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges, folded = load_and_fold(args.input_npz)
    rows: list[dict[str, object]] = []
    for profile in folded:
        fit, model = fit_rayleigh_profile(profile.probability, edges, profile.effective_events)
        fit_values = asdict(fit)
        row: dict[str, object] = {
            "nhit_bin": profile.nhit_bin,
            "n_cells": len(profile.cell_ids),
            "n_used_cells": len(profile.used_cell_ids),
            "cell_ids": ",".join(map(str, profile.cell_ids)),
            "used_cell_ids": ",".join(map(str, profile.used_cell_ids)),
            "excluded_cell_ids": ",".join(map(str, profile.excluded_cell_ids)),
            "total_baseline_weight": profile.total_baseline_weight,
            "effective_events": profile.effective_events,
            "data_probability": profile.probability,
            "model_probability": model,
            **fit_values,
        }
        row["multinomial_deviance_per_ndof"] = (
            fit.multinomial_deviance / fit.multinomial_ndof if fit.multinomial_ndof > 0 else math.nan
        )
        rows.append(row)

    np.savez_compressed(
        outputs["npz"],
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        profile_edges_deg=edges.astype(np.float32),
        profile_probability=np.vstack([row["data_probability"] for row in rows]).astype(np.float64),
        rayleigh_probability=np.vstack([row["model_probability"] for row in rows]).astype(np.float64),
        sigma_deg=np.asarray([row["sigma_deg"] for row in rows], dtype=np.float64),
        sigma_error_deg=np.asarray([row["sigma_error_deg"] for row in rows], dtype=np.float64),
        empirical_r68_deg=np.asarray([row["empirical_r68_deg"] for row in rows], dtype=np.float64),
        model_r68_deg=np.asarray([row["model_r68_deg"] for row in rows], dtype=np.float64),
        kl_divergence=np.asarray([row["kl_divergence"] for row in rows], dtype=np.float64),
        total_variation=np.asarray([row["total_variation"] for row in rows], dtype=np.float64),
        max_cdf_difference=np.asarray([row["max_cdf_difference"] for row in rows], dtype=np.float64),
        effective_events=np.asarray([row["effective_events"] for row in rows], dtype=np.float64),
        total_baseline_weight=np.asarray([row["total_baseline_weight"] for row in rows], dtype=np.float64),
        n_cells=np.asarray([row["n_cells"] for row in rows], dtype=np.int32),
        n_used_cells=np.asarray([row["n_used_cells"] for row in rows], dtype=np.int32),
        cell_ids=np.asarray([row["cell_ids"] for row in rows], dtype="U64"),
        used_cell_ids=np.asarray([row["used_cell_ids"] for row in rows], dtype="U256"),
        excluded_cell_ids=np.asarray([row["excluded_cell_ids"] for row in rows], dtype="U256"),
    )
    write_csv(rows, outputs["csv"])
    draw_fit_grid(rows, edges, outputs["fit_grid_png"], outputs["fit_grid_pdf"])
    draw_survival_residual_grid(rows, edges, outputs["diagnostics_png"])

    metadata = {
        "run_id": OUTPUT_STEM,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "definition": {
            "operation": "collapse Stage-B 2D (Nhit,predE) profiles over predE into seven Nhit profiles",
            "profile_window_deg": [float(edges[0]), float(edges[-1])],
            "profile_normalization": "conditional_on_0_to_5_deg_for_each_cell_and_folded_profile",
            "predE_cell_mixture_weight": "sumw_baseline",
            "zero_weight_policy": "retain_in_ledger_but_exclude_from_mixture",
            "unfiltered_diagnostic_policy": "not_used_because_it_has_no_physical_sumw_baseline",
            "fit_model": "conditional finite-window Rayleigh radial distribution",
            "fit_objective": "exact-bin multinomial cross-entropy (equivalently KL minimization)",
            "effective_events": "(sum_i sumw_i)^2 / sum_i(sumw_i^2 / Neff_i)",
            "deviance": "2 * aggregate_effective_events * KL; diagnostic effective-events approximation",
        },
        "inputs": {
            "stage_b_npz": str(args.input_npz.resolve()),
            "stage_b_npz_sha256": path_sha256(args.input_npz),
            "stage_b_metadata": str(args.input_metadata.resolve()),
            "stage_b_metadata_sha256": path_sha256(args.input_metadata),
        },
        "summary": {
            "n_input_cells": sum(int(row["n_cells"]) for row in rows),
            "n_positive_weight_profiles": sum(int(row["n_used_cells"]) for row in rows),
            "n_zero_weight_or_empty_profiles": sum(
                int(row["n_cells"]) - int(row["n_used_cells"]) for row in rows
            ),
            "n_nhit_bins": len(rows),
            "all_optimizer_success": all(bool(row["optimizer_success"]) for row in rows),
            "n_boundary_flagged": sum(bool(row["boundary_flag"]) for row in rows),
        },
        "fits": [
            {
                key: value
                for key, value in row.items()
                if key not in {"data_probability", "model_probability"}
            }
            for row in rows
        ],
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2))
    for row in rows:
        print(
            f"Nhit {row['nhit_bin']}: sigma={float(row['sigma_deg']):.5f} deg, "
            f"KL={float(row['kl_divergence']):.5f}, max|dCDF|={float(row['max_cdf_difference']):.5f}, "
            f"cells={row['n_used_cells']}/{row['n_cells']}"
        )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
