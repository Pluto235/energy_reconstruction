#!/usr/bin/env python3
"""Compare exact-GTI Pass5 and v6 Crab LogPar covariance results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


V6_COLOR = "#0072B2"
PASS5_COLOR = "#E69F00"
GAIN_COLOR = "#009E73"
LOSS_COLOR = "#D55E00"
GRID_COLOR = "#CBD5E1"
PARAMETER_LABELS = [r"$\phi_0(3\,\mathrm{TeV})$", r"$\alpha(3\,\mathrm{TeV})$", r"$\beta$"]
FIT_PARAMETER_NAMES = ["log10_phi0", "alpha", "beta"]
REFERENCE_ENERGIES_TEV = np.asarray([1.0, 3.0, 10.0, 30.0, 100.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass5-yaml", type=Path, required=True)
    parser.add_argument("--v6-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument("--pass5-live-days", type=float, required=True)
    parser.add_argument("--v6-live-days", type=float, required=True)
    parser.add_argument("--gti-manifest", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def require_covariance(
    matrix: object,
    label: str,
    reported_errors: list[float] | None = None,
    error_relative_tolerance: float = 1e-4,
) -> tuple[np.ndarray, dict[str, object]]:
    covariance = np.asarray(matrix, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{label} covariance is not a finite 3x3 matrix")
    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    symmetry_tolerance = max(1.0, float(np.max(np.abs(covariance)))) * 1e-12
    if asymmetry > symmetry_tolerance:
        raise ValueError(
            f"{label} covariance is not symmetric: max |C-C.T|={asymmetry}"
        )
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"{label} covariance is not positive definite: {eigenvalues}")
    diagonal_errors = np.sqrt(np.diag(covariance))
    audit: dict[str, object] = {
        "shape": list(covariance.shape),
        "finite": True,
        "symmetric": True,
        "max_abs_asymmetry": asymmetry,
        "positive_definite": True,
        "eigenvalues": eigenvalues.tolist(),
        "diagonal_errors": diagonal_errors.tolist(),
    }
    if reported_errors is not None:
        reported = np.asarray(reported_errors, dtype=np.float64)
        if reported.shape != (3,) or np.any(reported <= 0.0):
            raise ValueError(f"{label} reported errors are invalid: {reported}")
        relative_difference = np.abs(diagonal_errors - reported) / reported
        if np.any(relative_difference > error_relative_tolerance):
            raise ValueError(
                f"{label} covariance diagonal disagrees with reported errors: "
                f"relative differences={relative_difference}"
            )
        audit.update(
            {
                "reported_errors": reported.tolist(),
                "diagonal_errors_match_reported": True,
                "max_relative_error_difference": float(np.max(relative_difference)),
                "error_relative_tolerance": error_relative_tolerance,
            }
        )
    return covariance, audit


def load_v6(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = (payload.get("fits") or {}).get("logpar_conservative") or {}
    names = [str(name) for name in fit.get("fit_parameter_names") or []]
    if names != FIT_PARAMETER_NAMES:
        raise ValueError(f"v6 parameter order is {names}, expected {FIT_PARAMETER_NAMES}")
    parameters = fit.get("fit_parameters") or {}
    values = np.asarray([parameters[name] for name in FIT_PARAMETER_NAMES], dtype=np.float64)
    reported_errors = fit.get("fit_parameter_errors") or {}
    covariance, covariance_audit = require_covariance(
        fit.get("covariance"),
        "v6",
        [float(reported_errors[name]) for name in FIT_PARAMETER_NAMES],
    )
    pivot = float((payload.get("forward_folding") or {}).get("pivot_tev", math.nan))
    if pivot != 3.0:
        raise ValueError(f"v6 pivot is {pivot}, expected 3 TeV")
    diagnostics = {
        "chi2": float(fit.get("chi2", math.nan)),
        "ndof": int(fit.get("ndof", 0)),
        "chi2_over_ndof": float(fit.get("chi2_over_ndof", math.nan)),
        "p_value": float(fit.get("p_value", math.nan)),
        "minuit_status": fit.get("minuit_status") or {},
        "parameter_order": names,
        "pivot_tev": pivot,
        "covariance_audit": covariance_audit,
    }
    return values, covariance, diagnostics


def load_pass5(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = payload["source_dict"]["J0534+2200"]["sed_model"]
    output = payload["output_option"]["gtlike"]
    status = int(output["covariance_status"])
    if status != 3:
        raise ValueError(f"Pass5 covariance status is {status}, expected 3")

    names = [str(name) for name in output["covariance_parameter_names"]]
    target_names = ["J0534+2200_norm", "J0534+2200_index1", "J0534+2200_index2"]
    indices = [names.index(name) for name in target_names]
    full_covariance = np.asarray(output["covariance"], dtype=np.float64)
    native_covariance = full_covariance[np.ix_(indices, indices)]

    norm = float(source["norm"][0])
    norm_scale = float(source["norm"][3])
    alpha_native = float(source["index1"][0])
    beta = float(source["index2"][0])
    native_pivot = float(source["E_0"])
    if native_pivot != 3.0:
        raise ValueError(f"Pass5 pivot is {native_pivot}, expected 3 TeV")
    native_covariance, native_covariance_audit = require_covariance(
        native_covariance,
        "Pass5 native",
        [
            float(source["norm"][1]),
            float(source["index1"][1]),
            float(source["index2"][1]),
        ],
        error_relative_tolerance=3e-3,
    )

    log_pivot_ratio = math.log(3.0 / native_pivot)
    log10_phi0_3 = math.log10(norm * norm_scale) - (
        alpha_native * log_pivot_ratio + beta * log_pivot_ratio**2
    ) / math.log(10.0)
    alpha_3 = alpha_native + 2.0 * beta * log_pivot_ratio
    values = np.asarray([log10_phi0_3, alpha_3, beta], dtype=np.float64)

    jacobian = np.asarray(
        [
            [
                1.0 / (norm * math.log(10.0)),
                -log_pivot_ratio / math.log(10.0),
                -(log_pivot_ratio**2) / math.log(10.0),
            ],
            [0.0, 1.0, 2.0 * log_pivot_ratio],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    covariance, covariance_audit = require_covariance(
        jacobian @ native_covariance @ jacobian.T,
        "Pass5 transformed",
        [
            float(source["norm"][1]) / (norm * math.log(10.0)),
            float(source["index1"][1]),
            float(source["index2"][1]),
        ],
        error_relative_tolerance=3e-3,
    )
    diagnostics = {
        "covariance_status": status,
        "edm": float(output["edm"]),
        "minimum_value": float(output["minimum_value"]),
        "function_calls": int(output["function_calls"]),
        "native_parameter_names": target_names,
        "native_values": [norm, alpha_native, beta],
        "native_pivot_tev": native_pivot,
        "native_covariance": native_covariance.tolist(),
        "jacobian_native_to_3_tev": jacobian.tolist(),
        "all_free_parameter_names": names,
        "parameter_order": FIT_PARAMETER_NAMES,
        "pivot_tev": 3.0,
        "native_covariance_audit": native_covariance_audit,
        "covariance_audit": covariance_audit,
    }
    return values, covariance, diagnostics


def pdg_scale_factor(chi2_over_ndof: float) -> float:
    if not math.isfinite(chi2_over_ndof) or chi2_over_ndof <= 1.0:
        return 1.0
    return math.sqrt(chi2_over_ndof)


def correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigma, sigma)


def flux(values: np.ndarray, energies_tev: np.ndarray) -> np.ndarray:
    x = np.log(energies_tev / 3.0)
    ln_flux = math.log(10.0) * values[0] - values[1] * x - values[2] * x**2
    return np.exp(ln_flux)


def relative_uncertainty(covariance: np.ndarray, energies_tev: np.ndarray) -> np.ndarray:
    x = np.log(energies_tev / 3.0)
    gradients = np.column_stack([np.full_like(x, math.log(10.0)), -x, -(x**2)])
    variance = np.einsum("ij,jk,ik->i", gradients, covariance, gradients)
    return np.sqrt(np.clip(variance, 0.0, np.inf))


def physical_parameter_rows(values: np.ndarray, covariance: np.ndarray) -> list[tuple[float, float]]:
    sigma = np.sqrt(np.diag(covariance))
    phi0 = 10.0 ** values[0]
    return [
        (phi0, math.log(10.0) * phi0 * sigma[0]),
        (values[1], sigma[1]),
        (values[2], sigma[2]),
    ]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_parameter_table(
    output_dir: Path,
    v6_rows: list[tuple[float, float]],
    pass5_rows: list[tuple[float, float]],
    v6_scale_factor: float,
) -> None:
    rows = []
    labels = [r"$\phi_0$  [$10^{-12}\,\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]", r"$\alpha$", r"$\beta$"]
    for index, ((v6_value, v6_sigma), (p5_value, p5_sigma)) in enumerate(zip(v6_rows, pass5_rows)):
        scale = 1e12 if index == 0 else 1.0
        ratio = v6_sigma / p5_sigma
        v6_sigma_inflated = v6_sigma * v6_scale_factor
        ratio_inflated = v6_sigma_inflated / p5_sigma
        rows.append(
            [
                labels[index],
                f"{v6_value * scale:.5g}",
                f"{v6_sigma * scale:.4g}",
                f"{v6_sigma_inflated * scale:.4g}",
                f"{p5_value * scale:.5g}",
                f"{p5_sigma * scale:.4g}",
                f"{ratio:.3f}",
                f"{ratio_inflated:.3f}",
                f"{100.0 * (1.0 - ratio_inflated):+.1f}%",
            ]
        )

    fig, ax = plt.subplots(figsize=(15.0, 3.5))
    ax.axis("off")
    fig.suptitle("Crab LogPar covariance on the common-GTI selection", fontsize=17, fontweight="bold", y=0.98)
    columns = [
        "Parameter\nat 3 TeV",
        "v6\nbest fit",
        "v6 $1\\sigma$\n(raw HESSE)",
        f"v6 $1\\sigma$\n(inflated x{v6_scale_factor:.2f})",
        "Pass5\nbest fit",
        "Pass5\n$1\\sigma$",
        "$\\sigma_{v6}/\\sigma_{P5}$\n(raw)",
        "$\\sigma_{v6}/\\sigma_{P5}$\n(inflated)",
        "v6 reduction\n(inflated)",
    ]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center", bbox=[0.0, 0.16, 1.0, 0.72])
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    widths = [0.16, 0.09, 0.115, 0.15, 0.09, 0.09, 0.105, 0.11, 0.115]
    for (row, column), cell in table.get_celld().items():
        cell.set_width(widths[column])
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_height(cell.get_height() * 2.1)
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "white")
            if column == 8:
                value = float(rows[row - 1][8].rstrip("%"))
                cell.set_text_props(color=GAIN_COLOR if value >= 0.0 else LOSS_COLOR, fontweight="bold")
    fig.text(
        0.5,
        0.055,
        rf"Raw HESSE is shown for audit only. Inflated errors apply $S=\sqrt{{\chi^2/\mathrm{{ndof}}}}={v6_scale_factor:.3f}$; "
        "neither column proves smaller total uncertainty.",
        ha="center",
        color="#4B5563",
        fontsize=9.0,
    )
    save_figure(fig, output_dir / "v6_vs_pass5_parameter_uncertainty_table")


def plot_correlations(output_dir: Path, v6_cov: np.ndarray, pass5_cov: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
    fig.suptitle("LogPar parameter correlations at the common 3 TeV pivot", fontsize=16, fontweight="bold")
    image = None
    for ax, title, matrix in zip(axes, ["v6 2D Nhit x predE", "Official Pass5"], [correlation(v6_cov), correlation(pass5_cov)]):
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(3), [r"$\log_{10}\phi_0$", r"$\alpha$", r"$\beta$"])
        ax.set_yticks(range(3), [r"$\log_{10}\phi_0$", r"$\alpha$", r"$\beta$"])
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 else "#111827", fontweight="bold")
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.82, label="Correlation coefficient")
    save_figure(fig, output_dir / "v6_vs_pass5_correlation_matrices")


def plot_spectral_precision(
    output_dir: Path,
    energies: np.ndarray,
    v6_relative: np.ndarray,
    v6_relative_inflated: np.ndarray,
    pass5_relative: np.ndarray,
    volume_ratio: float,
    volume_ratio_inflated: float,
    v6_scale_factor: float,
) -> None:
    gain = 100.0 * (1.0 - v6_relative / pass5_relative)
    gain_inflated = 100.0 * (1.0 - v6_relative_inflated / pass5_relative)
    fig, (ax_main, ax_gain) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.1, 1.15]},
    )
    fig.suptitle("Crab formal covariance propagation: v6 2D versus official Pass5", fontsize=17, fontweight="bold")
    ax_main.plot(energies, 100.0 * pass5_relative, color=PASS5_COLOR, linewidth=2.2, label="Pass5 Nhit-only")
    ax_main.plot(energies, 100.0 * v6_relative, color=V6_COLOR, linewidth=2.4, label="v6 2D Nhit x predE (raw HESSE)")
    ax_main.plot(
        energies,
        100.0 * v6_relative_inflated,
        color=V6_COLOR,
        linewidth=2.0,
        linestyle="--",
        label=rf"v6 Birge/PDG inflated (x{v6_scale_factor:.3f})",
    )
    lower = np.minimum(v6_relative_inflated, pass5_relative) * 100.0
    upper = np.maximum(v6_relative_inflated, pass5_relative) * 100.0
    ax_main.fill_between(energies, lower, upper, color="#D1FAE5", alpha=0.45)
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r"Formal relative uncertainty  $\sigma_\phi(E)/\phi(E)$  [%]")
    ax_main.grid(True, which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.65)
    ax_main.legend(frameon=False, loc="upper left")
    ax_main.text(
        0.98,
        0.05,
        "Joint error-volume ratio\n"
        rf"raw={volume_ratio:.3f}; Birge/PDG={volume_ratio_inflated:.3f}",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    ax_gain.axhline(0.0, color="#6B7280", linewidth=0.9)
    ax_gain.plot(energies, gain, color=GAIN_COLOR, linewidth=1.6, linestyle=":", label="raw HESSE")
    ax_gain.plot(energies, gain_inflated, color=GAIN_COLOR, linewidth=2.0, label="Birge/PDG inflated")
    ax_gain.fill_between(energies, 0.0, gain_inflated, where=gain_inflated >= 0.0, color="#A7F3D0", alpha=0.65)
    ax_gain.fill_between(energies, 0.0, gain_inflated, where=gain_inflated < 0.0, color="#FED7AA", alpha=0.75)
    ax_gain.set_xscale("log")
    ax_gain.set_xlabel("Energy [TeV]")
    ax_gain.set_ylabel("v6 reduction [%]")
    ax_gain.grid(True, axis="x", which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.65)
    ax_gain.set_xticks(REFERENCE_ENERGIES_TEV, ["1", "3", "10", "30", "100"])
    ax_gain.legend(frameon=False, loc="best", fontsize=8.4)
    for energy in REFERENCE_ENERGIES_TEV:
        index = int(np.argmin(np.abs(energies - energy)))
        ax_gain.annotate(
            f"{gain_inflated[index]:+.1f}%",
            (energies[index], gain_inflated[index]),
            xytext=(0, 7 if gain_inflated[index] >= 0 else -13),
            textcoords="offset points",
            ha="center",
            color=GAIN_COLOR if gain_inflated[index] >= 0 else LOSS_COLOR,
            fontsize=8.8,
            fontweight="bold",
        )
    fig.text(
        0.5,
        -0.015,
        r"Full-covariance propagation. Inflated v6 errors scale by sqrt(chi2/ndof); "
        r"the raw curve is not evidence of superior precision.",
        ha="center",
        fontsize=9.1,
        color="#4B5563",
    )
    save_figure(fig, output_dir / "v6_vs_pass5_spectral_relative_uncertainty")


def plot_spectrum(
    output_dir: Path,
    energies: np.ndarray,
    v6_values: np.ndarray,
    v6_relative: np.ndarray,
    pass5_values: np.ndarray,
    pass5_relative: np.ndarray,
) -> None:
    v6_flux = flux(v6_values, energies)
    pass5_flux = flux(pass5_values, energies)
    fig, (ax_main, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.1, 1.0]},
    )
    fig.suptitle("Crab LogPar fits on the common-GTI selection", fontsize=17, fontweight="bold")
    for values, rel, color, label in [
        (v6_flux, v6_relative, V6_COLOR, "v6 2D Nhit x predE"),
        (pass5_flux, pass5_relative, PASS5_COLOR, "Pass5 Nhit-only"),
    ]:
        sed = energies**2 * values
        ax_main.plot(energies, sed, color=color, linewidth=2.3, label=label)
        ax_main.fill_between(energies, sed * np.exp(-rel), sed * np.exp(rel), color=color, alpha=0.18)
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$  [TeV cm$^{-2}$ s$^{-1}$]")
    ax_main.grid(True, which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.65)
    ax_main.legend(frameon=False)

    ratio = pass5_flux / v6_flux
    ratio_sigma = np.sqrt(v6_relative**2 + pass5_relative**2)
    ax_ratio.axhline(1.0, color="#6B7280", linewidth=0.9)
    ax_ratio.plot(energies, ratio, color=PASS5_COLOR, linewidth=2.1)
    ax_ratio.fill_between(energies, ratio * np.exp(-ratio_sigma), ratio * np.exp(ratio_sigma), color="#FDE68A", alpha=0.55)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Energy [TeV]")
    ax_ratio.set_ylabel("Pass5 / v6")
    ax_ratio.set_xticks(REFERENCE_ENERGIES_TEV, ["1", "3", "10", "30", "100"])
    ax_ratio.grid(True, axis="x", which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.65)
    fig.text(0.5, -0.015, "Bands show formal HESSE 1 sigma statistical propagation only; the ratio band ignores cross-method covariance.", ha="center", fontsize=9.2, color="#4B5563")
    save_figure(fig, output_dir / "v6_vs_pass5_logpar_spectrum")


def write_tables(
    output_dir: Path,
    energies: np.ndarray,
    v6_values: np.ndarray,
    v6_cov: np.ndarray,
    pass5_values: np.ndarray,
    pass5_cov: np.ndarray,
    v6_scale_factor: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    v6_rows = physical_parameter_rows(v6_values, v6_cov)
    pass5_rows = physical_parameter_rows(pass5_values, pass5_cov)
    with (output_dir / "v6_vs_pass5_logpar_parameters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "parameter",
                "v6_value",
                "v6_sigma_raw",
                "v6_sigma_birge_pdg",
                "pass5_value",
                "pass5_sigma",
                "sigma_v6_over_pass5_raw",
                "sigma_v6_over_pass5_birge_pdg",
                "v6_uncertainty_reduction_percent_raw",
                "v6_uncertainty_reduction_percent_birge_pdg",
            ]
        )
        for name, (v6_value, v6_sigma), (p5_value, p5_sigma) in zip(["phi0_at_3tev", "alpha_at_3tev", "beta"], v6_rows, pass5_rows):
            v6_sigma_inflated = v6_sigma * v6_scale_factor
            ratio = v6_sigma / p5_sigma
            ratio_inflated = v6_sigma_inflated / p5_sigma
            writer.writerow(
                [
                    name,
                    f"{v6_value:.12g}",
                    f"{v6_sigma:.12g}",
                    f"{v6_sigma_inflated:.12g}",
                    f"{p5_value:.12g}",
                    f"{p5_sigma:.12g}",
                    f"{ratio:.12g}",
                    f"{ratio_inflated:.12g}",
                    f"{100.0 * (1.0 - ratio):.12g}",
                    f"{100.0 * (1.0 - ratio_inflated):.12g}",
                ]
            )

    v6_relative = relative_uncertainty(v6_cov, energies)
    v6_relative_inflated = v6_relative * v6_scale_factor
    pass5_relative = relative_uncertainty(pass5_cov, energies)
    with (output_dir / "v6_vs_pass5_spectral_relative_uncertainty.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "energy_tev",
                "v6_relative_uncertainty_percent_raw",
                "v6_relative_uncertainty_percent_birge_pdg",
                "pass5_relative_uncertainty_percent",
                "v6_uncertainty_reduction_percent_raw",
                "v6_uncertainty_reduction_percent_birge_pdg",
            ]
        )
        for energy, v6_rel, v6_rel_inflated, p5_rel in zip(
            energies, v6_relative, v6_relative_inflated, pass5_relative
        ):
            writer.writerow(
                [
                    f"{energy:.10g}",
                    f"{100.0 * v6_rel:.10g}",
                    f"{100.0 * v6_rel_inflated:.10g}",
                    f"{100.0 * p5_rel:.10g}",
                    f"{100.0 * (1.0 - v6_rel / p5_rel):.10g}",
                    f"{100.0 * (1.0 - v6_rel_inflated / p5_rel):.10g}",
                ]
            )
    return v6_rows, pass5_rows


def write_html_report(output_dir: Path, report_path: Path, summary: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    asset_prefix = Path(os.path.relpath(output_dir, report_path.parent)).as_posix()

    def asset(name: str) -> str:
        return html.escape(f"{asset_prefix}/{name}", quote=True)

    p = summary["parameter_comparison"]
    audit = summary["scientific_audit"]
    sample = audit["sample_contract"]
    live = audit["live_time_comparison"]
    scope = audit["analysis_scope"]
    provenance = audit["pass5_provenance"]
    scale_factor = summary["v6_birge_pdg_scale_factor"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['parameter']))}</td>"
        f"<td>{row['v6_value']}</td><td>{row['v6_sigma']}</td>"
        f"<td>{row['v6_sigma_birge_pdg']}</td>"
        f"<td>{row['pass5_value']}</td><td>{row['pass5_sigma']}</td>"
        f"<td>{row['v6_reduction_percent']:+.1f}%</td>"
        f"<td>{row['v6_reduction_percent_birge_pdg']:+.1f}%</td>"
        "</tr>"
        for row in p
    )
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>v6 vs Pass5 Crab covariance comparison</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;color:#111827;background:#fff;line-height:1.6}}
main{{max-width:1180px;margin:0 auto;padding:36px 28px 64px}} h1{{font-size:30px;margin:0 0 8px}} h2{{margin-top:42px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}}
.meta{{color:#4b5563}} .callout{{border-left:5px solid {V6_COLOR};padding:12px 16px;background:#eff6ff;margin:22px 0}}
img{{display:block;width:100%;height:auto;margin:18px 0 28px}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;border:1px solid #d1d5db;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#1f2937;color:#fff}}
code{{background:#f3f4f6;padding:2px 5px}} .warn{{border-left-color:{LOSS_COLOR};background:#fff7ed}}
</style></head><body><main>
<h1>v6 与 official Pass5：Crab LogPar covariance 对比</h1>
<p class="meta">最终 common-GTI 选择：{summary['selected_hour_count']} accepted hours、{summary['gti_interval_count']} intervals、928 个 Pass5 J2000 maps。v6 GTI = {summary['v6_live_days']:.9f} d；Pass5 merged-map header = {summary['pass5_live_days']:.9f} d。</p>
<div class="callout"><strong>样本审计通过：</strong>strict recovery 为 1078 total / 928 accepted / 150 rejected / 0 remaining；<code>accepted_maps.list</code> 为 928 行且 928 个唯一 URI，与 EOS 上 928 个非空 J2000 ROOT 完全一致，merge 终态日志也记录 928 个输入。<code>common_gti.tsv</code> 的 4763 行 duration sum 与 manifest 一致。</div>
<div class="callout warn"><strong>“相同时间样本”的限定：</strong>两套管线共享相同 accepted-hour/common-GTI 选择，但有效 live time 并非逐秒相同。差值为 {live['v6_minus_pass5_seconds']:.3f} s（v6 的 {live['relative_difference_percent_of_v6']:.5f}%）。928 个 chunk 中 898 个在 ±0.2 s 内一致；14 个异常 chunk 贡献 {live['large_delta_share_percent']:.2f}% 的总差值。Pass5 以官方事件流经 GTI mask 后的 DI 时间占用计算 <code>EffLtime</code>，v6 则累加 recovered-time GTI 连续端点；20 个 accepted hours 的 Pass5 mask histogram 为零。故可称“common-GTI 选择样本”，不可称严格相同有效曝光。</div>
<div class="callout"><strong>共同参数空间：</strong>两套最终 covariance 均独立核验为 3×3、原矩阵对称、正定，且对角线平方根与各自参数误差一致；共同参数顺序为 <code>log10_phi0 / alpha / beta</code>，pivot 均为 3 TeV。</div>
<h2>参数误差</h2><img src="{asset('v6_vs_pass5_parameter_uncertainty_table.png')}" alt="LogPar parameter uncertainty table">
<table><thead><tr><th>参数</th><th>v6 值</th><th>v6 σ raw</th><th>v6 σ Birge/PDG</th><th>Pass5 值</th><th>Pass5 σ</th><th>v6 reduction raw</th><th>v6 reduction Birge/PDG</th></tr></thead><tbody>{rows}</tbody></table>
<p>v6 的 χ²/ndof = {summary['v6_diagnostics']['chi2_over_ndof']:.6f}，Birge/PDG 因子为 {scale_factor:.6f}。联合误差体积比 <code>sqrt(det C_v6 / det C_Pass5)</code>：raw HESSE = <strong>{summary['joint_error_volume_ratio_v6_over_pass5']:.3f}</strong>；按该因子膨胀 v6 三维误差体积后 = <strong>{summary['joint_error_volume_ratio_v6_over_pass5_birge_pdg']:.3f}</strong>。raw 数值 0.528 不能作为 v6 精度优于 Pass5 的证据。</p>
<h2>能谱相对不确定度</h2><img src="{asset('v6_vs_pass5_spectral_relative_uncertainty.png')}" alt="Spectral relative uncertainty comparison">
<p>使用 <code>Var[ln φ(E)] = g(E)^T C g(E)</code> 传播完整 covariance。raw 曲线仅复现形式 HESSE；虚线为 Birge/PDG 膨胀，二者都不包含系统误差。</p>
<h2>最佳拟合能谱</h2><img src="{asset('v6_vs_pass5_logpar_spectrum.png')}" alt="Best-fit LogPar spectra">
<h2>参数相关性</h2><img src="{asset('v6_vs_pass5_correlation_matrices.png')}" alt="Correlation matrices">
<h2>能区、binning 与 objective 审计</h2>
<table><thead><tr><th>项目</th><th>v6 2D</th><th>official Pass5</th></tr></thead><tbody>
<tr><td>Nhit</td><td>100≤Nhit&lt;3000；7 个 Nhit 带，44 个选中 Nhit×predE cells</td><td>30≤Nhit&lt;2000；edges = 30/60/100/200/300/500/800/2000</td></tr>
<tr><td>能量坐标</td><td>predE 选中包络 0.1–316.23 TeV；响应 true-E 0.1–1000 TeV；本报告传播 1–100 TeV</td><td>Nhit-only，无 event-level reconstructed-energy cut；7 个代表能量约 0.562–15.849 TeV</td></tr>
<tr><td>目标函数</td><td>44 个 Stage-E excess cells 上的 conservative χ²，σ=√(N_on+B_on)，ndof=41</td><td>7-bin spatial cube 的 Poisson likelihood；HESSE 中共 7 个 free parameters（Crab 3 + nuisance norms）</td></tr>
</tbody></table>
<div class="callout warn"><strong>科学解释限制：</strong>这是完整管线比较，不是 predE 独立增益实验。除 predE 外，两侧的 Nhit 覆盖/edges、cell selection、背景、PSF/IRF、objective 与 nuisance 处理均不同。要隔离 predE 增益，必须在同一 v6 管线内固定这些选择，只切换 predE 维度。当前结果不能支持“v6 总精度已证明优于 Pass5”，也不能把 covariance 差异归因于 predE。</div>
<div class="callout warn"><strong>Pass5 provenance 限定：</strong>merged map → data config → data.root → covariance YAML 的 SHA256、live time 与时间戳顺序已记录；但 <code>data_config.yaml</code> 和 <code>covariance_fit.yaml</code> 仍内嵌已不存在的 <code>common_gti_fit_interactive/</code> 路径，实际文件位于 <code>common_gti_fit/</code>。因此文件级 provenance 可审计，但路径元数据不是完全自洽闭环，原始 YAML 未被改写。</div>
</main></body></html>"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    v6_values, v6_cov, v6_diagnostics = load_v6(args.v6_json)
    pass5_values, pass5_cov, pass5_diagnostics = load_pass5(args.pass5_yaml)
    gti_manifest = json.loads(args.gti_manifest.read_text(encoding="utf-8"))
    scientific_audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    sample_contract = scientific_audit["sample_contract"]
    live_time_audit = scientific_audit["live_time_comparison"]
    if not all(
        [
            sample_contract["duration_sum_matches_manifest"],
            sample_contract["accepted_maps_matches"],
            sample_contract["eos_matches_accepted_maps"],
            sample_contract["merged_map_terminal_log_matches"],
        ]
    ):
        raise ValueError("common-GTI sample audit did not pass")
    if not math.isclose(
        args.v6_live_days,
        float(gti_manifest["common_gti_live_time_days"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("--v6-live-days differs from the terminal common-GTI manifest")
    if not math.isclose(
        args.pass5_live_days,
        float(live_time_audit["pass5_merged_header_days"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("--pass5-live-days differs from the merged-map header audit")
    energies = np.geomspace(1.0, 100.0, 301)
    v6_scale_factor = pdg_scale_factor(v6_diagnostics["chi2_over_ndof"])
    v6_relative = relative_uncertainty(v6_cov, energies)
    v6_relative_inflated = v6_relative * v6_scale_factor
    pass5_relative = relative_uncertainty(pass5_cov, energies)
    volume_ratio = math.sqrt(float(np.linalg.det(v6_cov) / np.linalg.det(pass5_cov)))
    volume_ratio_inflated = volume_ratio * v6_scale_factor**3
    v6_rows, pass5_rows = write_tables(
        args.output_dir,
        energies,
        v6_values,
        v6_cov,
        pass5_values,
        pass5_cov,
        v6_scale_factor,
    )

    plot_parameter_table(args.output_dir, v6_rows, pass5_rows, v6_scale_factor)
    plot_correlations(args.output_dir, v6_cov, pass5_cov)
    plot_spectral_precision(
        args.output_dir,
        energies,
        v6_relative,
        v6_relative_inflated,
        pass5_relative,
        volume_ratio,
        volume_ratio_inflated,
        v6_scale_factor,
    )
    plot_spectrum(args.output_dir, energies, v6_values, v6_relative, pass5_values, pass5_relative)

    parameter_comparison = []
    for name, (v6_value, v6_sigma), (p5_value, p5_sigma) in zip(["phi0_at_3tev", "alpha_at_3tev", "beta"], v6_rows, pass5_rows):
        v6_sigma_inflated = v6_sigma * v6_scale_factor
        parameter_comparison.append(
            {
                "parameter": name,
                "v6_value": v6_value,
                "v6_sigma": v6_sigma,
                "v6_sigma_birge_pdg": v6_sigma_inflated,
                "pass5_value": p5_value,
                "pass5_sigma": p5_sigma,
                "sigma_v6_over_pass5": v6_sigma / p5_sigma,
                "sigma_v6_over_pass5_birge_pdg": v6_sigma_inflated / p5_sigma,
                "v6_reduction_percent": 100.0 * (1.0 - v6_sigma / p5_sigma),
                "v6_reduction_percent_birge_pdg": 100.0
                * (1.0 - v6_sigma_inflated / p5_sigma),
            }
        )
    summary = {
        "sample": (
            "Pass5 and v6 use the same accepted-hour/common-GTI selection, with "
            "pipeline-specific effective live time."
        ),
        "selected_hour_count": int(gti_manifest["accepted_hour_count"]),
        "gti_interval_count": int(gti_manifest["common_gti_interval_count"]),
        "gti_manifest": str(args.gti_manifest),
        "v6_live_days": args.v6_live_days,
        "pass5_live_days": args.pass5_live_days,
        "common_parameterization": FIT_PARAMETER_NAMES,
        "common_pivot_tev": 3.0,
        "v6_values": v6_values.tolist(),
        "v6_covariance": v6_cov.tolist(),
        "v6_correlation": correlation(v6_cov).tolist(),
        "pass5_values": pass5_values.tolist(),
        "pass5_covariance": pass5_cov.tolist(),
        "pass5_correlation": correlation(pass5_cov).tolist(),
        "joint_error_volume_ratio_v6_over_pass5": volume_ratio,
        "joint_error_volume_ratio_v6_over_pass5_birge_pdg": volume_ratio_inflated,
        "v6_birge_pdg_scale_factor": v6_scale_factor,
        "parameter_comparison": parameter_comparison,
        "v6_diagnostics": v6_diagnostics,
        "pass5_diagnostics": pass5_diagnostics,
        "scientific_audit": scientific_audit,
        "covariance_validation_passed": True,
        "caveats": [
            "Formal HESSE statistical covariance only; systematic uncertainty is excluded.",
            "v6 has chi2/ndof = 20.943; its raw HESSE covariance is not demonstrated to be a reliable precision measure.",
            "The raw joint error-volume ratio is 0.528, but the Birge/PDG-scaled ratio is 50.617; neither establishes superior v6 total precision.",
            "v6 uses a conservative chi-square objective while Pass5 uses a Poisson likelihood, so the formal covariance definitions are not identical.",
            "Official Pass5 uses 30 <= Nhit < 2000 while the current v6 fit uses 100 <= Nhit < 3000 with different bin edges; this is a full-pipeline comparison, not an isolated predE ablation.",
            "The nominal common-GTI selection is shared, but the effective live times differ by about 4615 seconds because Pass5 counts official-event/DI occupancy while v6 sums recovered-time interval endpoints.",
            "No cross-method covariance is available for a rigorous difference significance.",
            "The plotted spectral-uncertainty range is 1-100 TeV. Pass5 remains an Nhit-only analysis and therefore has no event-by-event reconstructed-energy cut.",
            "Pass5 provenance retains stale common_gti_fit_interactive paths; file hashes and chronology are audited, but embedded path provenance is not fully self-contained.",
        ],
    }
    (args.output_dir / "v6_vs_pass5_covariance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_html = args.report_html or (args.output_dir / "v6_vs_pass5_sorted_gti_covariance_report.html")
    write_html_report(args.output_dir, report_html, summary)


if __name__ == "__main__":
    main()
