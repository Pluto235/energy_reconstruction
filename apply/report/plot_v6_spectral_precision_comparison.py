#!/usr/bin/env python3
"""Plot the controlled v6 1D-versus-2D spectral precision comparison."""

from __future__ import annotations

import argparse
import csv
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


ENERGIES_TEV = np.asarray([1.0, 3.0, 10.0, 30.0, 100.0], dtype=np.float64)
PIVOT_TEV = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d-json", type=Path, required=True)
    parser.add_argument("--two-d-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def logpar_covariance(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = (payload.get("fits") or {}).get("logpar_conservative") or {}
    names = [str(name) for name in fit.get("fit_parameter_names") or []]
    expected = ["log10_phi0", "alpha", "beta"]
    if names != expected:
        raise ValueError(f"{path} parameter order is {names}, expected {expected}")
    covariance = np.asarray(fit.get("covariance"), dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{path} has an invalid 3x3 LogPar covariance")
    return covariance


def relative_uncertainty_percent(covariance: np.ndarray, energies_tev: np.ndarray) -> np.ndarray:
    log_ratio = np.log(energies_tev / PIVOT_TEV)
    gradients = np.column_stack(
        [
            np.full(energies_tev.shape, math.log(10.0), dtype=np.float64),
            -log_ratio,
            -(log_ratio * log_ratio),
        ]
    )
    variance = np.einsum("ij,jk,ik->i", gradients, covariance, gradients)
    return 100.0 * np.sqrt(np.clip(variance, 0.0, np.inf))


def write_csv(
    path: Path,
    one_d_percent: np.ndarray,
    two_d_percent: np.ndarray,
    improvement_percent: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["energy_tev", "one_d_relative_uncertainty_percent", "two_d_relative_uncertainty_percent", "improvement_percent"])
        for energy, one_d, two_d, improvement in zip(
            ENERGIES_TEV,
            one_d_percent,
            two_d_percent,
            improvement_percent,
        ):
            writer.writerow([f"{energy:g}", f"{one_d:.8g}", f"{two_d:.8g}", f"{improvement:.8g}"])


def main() -> None:
    args = parse_args()
    covariance_1d = logpar_covariance(args.one_d_json)
    covariance_2d = logpar_covariance(args.two_d_json)
    one_d = relative_uncertainty_percent(covariance_1d, ENERGIES_TEV)
    two_d = relative_uncertainty_percent(covariance_2d, ENERGIES_TEV)
    improvement = 100.0 * (1.0 - two_d / one_d)
    volume_ratio = math.sqrt(float(np.linalg.det(covariance_2d) / np.linalg.det(covariance_1d)))
    volume_reduction = 100.0 * (1.0 - volume_ratio)

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

    fig, (ax_main, ax_gain) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.3),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.1, 1.0]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        r"Crab spectral precision: 1D $N_{\mathrm{hit}}$ versus 2D $N_{\mathrm{hit}}\times E_{\mathrm{pred}}$",
        fontsize=17,
        fontweight="bold",
    )

    one_color = "#6B7280"
    two_color = "#2563EB"
    gain_color = "#059669"
    ax_main.fill_between(
        ENERGIES_TEV,
        two_d,
        one_d,
        color="#A7F3D0",
        alpha=0.42,
        label="Precision gained by retaining predE",
        zorder=1,
    )
    ax_main.plot(
        ENERGIES_TEV,
        one_d,
        "o--",
        color=one_color,
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="1D Nhit",
        zorder=3,
    )
    ax_main.plot(
        ENERGIES_TEV,
        two_d,
        "s-",
        color=two_color,
        linewidth=2.3,
        markersize=6.2,
        label="2D Nhit x predE",
        zorder=4,
    )
    for energy, one_value, two_value in zip(ENERGIES_TEV, one_d, two_d):
        ax_main.annotate(
            f"{one_value:.2f}%",
            (energy, one_value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color=one_color,
        )
        ax_main.annotate(
            f"{two_value:.2f}%",
            (energy, two_value),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.7,
            color=two_color,
            fontweight="bold",
        )

    ax_main.set_xscale("log")
    ax_main.set_ylabel(r"Relative spectral uncertainty  $\sigma_{\phi}(E)/\phi(E)$  [%]")
    ax_main.set_ylim(0.0, max(one_d) * 1.22)
    ax_main.grid(True, which="both", color="#CBD5E1", alpha=0.48, linewidth=0.65)
    ax_main.legend(loc="upper left", frameon=False, ncol=1)
    ax_main.text(
        0.62,
        0.94,
        "Joint 3-parameter precision\n"
        + rf"$\sqrt{{\det C_{{2D}}/\det C_{{1D}}}}={volume_ratio:.3f}$"
        + "\n"
        + f"Error volume is {volume_reduction:.1f}% smaller",
        transform=ax_main.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#EFF6FF", "edgecolor": "#93C5FD", "linewidth": 1.0},
    )

    ax_gain.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax_gain.vlines(ENERGIES_TEV, 0.0, improvement, color="#6EE7B7", linewidth=5.0, zorder=2)
    ax_gain.scatter(ENERGIES_TEV, improvement, s=55, color=gain_color, edgecolor="white", linewidth=0.8, zorder=3)
    for energy, value in zip(ENERGIES_TEV, improvement):
        ax_gain.annotate(
            f"{value:.1f}%",
            (energy, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#047857",
            fontweight="bold",
        )
    ax_gain.set_xscale("log")
    ax_gain.set_ylim(0.0, max(improvement) * 1.35)
    ax_gain.set_ylabel("Improvement [%]")
    ax_gain.set_xlabel("Energy [TeV]")
    ax_gain.set_xticks(ENERGIES_TEV, ["1", "3", "10", "30", "100"])
    ax_gain.grid(True, axis="x", color="#CBD5E1", alpha=0.48, linewidth=0.65)

    fig.text(
        0.5,
        -0.012,
        r"Formal HESSE $1\sigma$ precision propagated with $\sigma^2_{\ln\phi(E)}=g(E)^T C g(E)$; systematic uncertainty and bias are not included.",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#4B5563",
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), one_d, two_d, improvement)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
