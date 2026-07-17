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
    return parser.parse_args()


def require_covariance(matrix: object, label: str) -> np.ndarray:
    covariance = np.asarray(matrix, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{label} covariance is not a finite 3x3 matrix")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"{label} covariance is not positive definite: {eigenvalues}")
    return covariance


def load_v6(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = (payload.get("fits") or {}).get("logpar_conservative") or {}
    names = [str(name) for name in fit.get("fit_parameter_names") or []]
    if names != FIT_PARAMETER_NAMES:
        raise ValueError(f"v6 parameter order is {names}, expected {FIT_PARAMETER_NAMES}")
    parameters = fit.get("fit_parameters") or {}
    values = np.asarray([parameters[name] for name in FIT_PARAMETER_NAMES], dtype=np.float64)
    covariance = require_covariance(fit.get("covariance"), "v6")
    diagnostics = {
        "chi2": float(fit.get("chi2", math.nan)),
        "ndof": int(fit.get("ndof", 0)),
        "chi2_over_ndof": float(fit.get("chi2_over_ndof", math.nan)),
        "p_value": float(fit.get("p_value", math.nan)),
        "minuit_status": fit.get("minuit_status") or {},
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
    if native_pivot <= 0.0:
        raise ValueError(f"Pass5 pivot must be positive, got {native_pivot}")

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
    covariance = require_covariance(jacobian @ native_covariance @ jacobian.T, "Pass5")
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
    }
    return values, covariance, diagnostics


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
) -> None:
    rows = []
    labels = [r"$\phi_0$  [$10^{-12}\,\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]", r"$\alpha$", r"$\beta$"]
    for index, ((v6_value, v6_sigma), (p5_value, p5_sigma)) in enumerate(zip(v6_rows, pass5_rows)):
        scale = 1e12 if index == 0 else 1.0
        ratio = v6_sigma / p5_sigma
        rows.append(
            [
                labels[index],
                f"{v6_value * scale:.5g}",
                f"{v6_sigma * scale:.4g}",
                f"{p5_value * scale:.5g}",
                f"{p5_sigma * scale:.4g}",
                f"{ratio:.3f}",
                f"{100.0 * (1.0 - ratio):+.1f}%",
            ]
        )

    fig, ax = plt.subplots(figsize=(12.0, 3.25))
    ax.axis("off")
    fig.suptitle("Crab LogPar precision on identical recovered-time GTIs", fontsize=17, fontweight="bold", y=0.98)
    columns = ["Parameter at 3 TeV", "v6 best fit", r"v6 $1\sigma$", "Pass5 best fit", r"Pass5 $1\sigma$", r"$\sigma_{v6}/\sigma_{P5}$", "v6 reduction"]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center", bbox=[0.0, 0.16, 1.0, 0.66])
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    widths = [0.25, 0.115, 0.105, 0.115, 0.105, 0.13, 0.13]
    for (row, column), cell in table.get_celld().items():
        cell.set_width(widths[column])
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "white")
            if column == 6:
                value = float(rows[row - 1][6].rstrip("%"))
                cell.set_text_props(color=GAIN_COLOR if value >= 0.0 else LOSS_COLOR, fontweight="bold")
    fig.text(
        0.5,
        0.055,
        "Positive reduction means v6 has a smaller formal statistical error; systematic uncertainties are not included.",
        ha="center",
        color="#4B5563",
        fontsize=9.4,
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
    pass5_relative: np.ndarray,
    volume_ratio: float,
) -> None:
    gain = 100.0 * (1.0 - v6_relative / pass5_relative)
    fig, (ax_main, ax_gain) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.1, 1.15]},
    )
    fig.suptitle("Crab spectral precision: v6 2D versus official Pass5", fontsize=17, fontweight="bold")
    ax_main.plot(energies, 100.0 * pass5_relative, color=PASS5_COLOR, linewidth=2.2, label="Pass5 Nhit-only")
    ax_main.plot(energies, 100.0 * v6_relative, color=V6_COLOR, linewidth=2.4, label="v6 2D Nhit x predE")
    lower = np.minimum(v6_relative, pass5_relative) * 100.0
    upper = np.maximum(v6_relative, pass5_relative) * 100.0
    ax_main.fill_between(energies, lower, upper, color="#D1FAE5", alpha=0.45)
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r"Formal relative uncertainty  $\sigma_\phi(E)/\phi(E)$  [%]")
    ax_main.grid(True, which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.65)
    ax_main.legend(frameon=False, loc="upper left")
    ax_main.text(
        0.98,
        0.05,
        rf"Joint error-volume ratio: $\sqrt{{\det C_{{v6}}/\det C_{{P5}}}}={volume_ratio:.3f}$",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    ax_gain.axhline(0.0, color="#6B7280", linewidth=0.9)
    ax_gain.plot(energies, gain, color=GAIN_COLOR, linewidth=2.0)
    ax_gain.fill_between(energies, 0.0, gain, where=gain >= 0.0, color="#A7F3D0", alpha=0.65)
    ax_gain.fill_between(energies, 0.0, gain, where=gain < 0.0, color="#FED7AA", alpha=0.75)
    ax_gain.set_xscale("log")
    ax_gain.set_xlabel("Energy [TeV]")
    ax_gain.set_ylabel("v6 reduction [%]")
    ax_gain.grid(True, axis="x", which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.65)
    ax_gain.set_xticks(REFERENCE_ENERGIES_TEV, ["1", "3", "10", "30", "100"])
    for energy in REFERENCE_ENERGIES_TEV:
        index = int(np.argmin(np.abs(energies - energy)))
        ax_gain.annotate(f"{gain[index]:+.1f}%", (energies[index], gain[index]), xytext=(0, 7 if gain[index] >= 0 else -13), textcoords="offset points", ha="center", color=GAIN_COLOR if gain[index] >= 0 else LOSS_COLOR, fontsize=8.8, fontweight="bold")
    fig.text(0.5, -0.015, r"Propagation: $\sigma^2_{\ln\phi(E)}=g(E)^T C g(E)$ with the full 3-parameter covariance.", ha="center", fontsize=9.3, color="#4B5563")
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
    fig.suptitle("Crab LogPar fits on identical recovered-time GTIs", fontsize=17, fontweight="bold")
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
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    v6_rows = physical_parameter_rows(v6_values, v6_cov)
    pass5_rows = physical_parameter_rows(pass5_values, pass5_cov)
    with (output_dir / "v6_vs_pass5_logpar_parameters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["parameter", "v6_value", "v6_sigma", "pass5_value", "pass5_sigma", "sigma_v6_over_pass5", "v6_uncertainty_reduction_percent"])
        for name, (v6_value, v6_sigma), (p5_value, p5_sigma) in zip(["phi0_at_3tev", "alpha_at_3tev", "beta"], v6_rows, pass5_rows):
            ratio = v6_sigma / p5_sigma
            writer.writerow([name, f"{v6_value:.12g}", f"{v6_sigma:.12g}", f"{p5_value:.12g}", f"{p5_sigma:.12g}", f"{ratio:.12g}", f"{100.0 * (1.0 - ratio):.12g}"])

    v6_relative = relative_uncertainty(v6_cov, energies)
    pass5_relative = relative_uncertainty(pass5_cov, energies)
    with (output_dir / "v6_vs_pass5_spectral_relative_uncertainty.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["energy_tev", "v6_relative_uncertainty_percent", "pass5_relative_uncertainty_percent", "v6_uncertainty_reduction_percent"])
        for energy, v6_rel, p5_rel in zip(energies, v6_relative, pass5_relative):
            writer.writerow([f"{energy:.10g}", f"{100.0 * v6_rel:.10g}", f"{100.0 * p5_rel:.10g}", f"{100.0 * (1.0 - v6_rel / p5_rel):.10g}"])
    return v6_rows, pass5_rows


def write_html_report(output_dir: Path, report_path: Path, summary: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    asset_prefix = Path(os.path.relpath(output_dir, report_path.parent)).as_posix()

    def asset(name: str) -> str:
        return html.escape(f"{asset_prefix}/{name}", quote=True)

    p = summary["parameter_comparison"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['parameter']))}</td>"
        f"<td>{row['v6_value']}</td><td>{row['v6_sigma']}</td>"
        f"<td>{row['pass5_value']}</td><td>{row['pass5_sigma']}</td>"
        f"<td>{row['v6_reduction_percent']:+.1f}%</td>"
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
<p class="meta">同一组排序后的 recovered-time GTI；3969 个小时文件、{summary['gti_interval_count']} 个 GTI。v6 GTI = {summary['v6_live_days']:.6f} d，Pass5 DI map header live time = {summary['pass5_live_days']:.6f} d。</p>
<div class="callout"><strong>比较口径：</strong>Pass5 的 10 TeV 参数和完整 HESSE covariance 已通过解析 Jacobian 变换到 v6 的 3 TeV 参数空间，再比较误差与能谱不确定度。</div>
<h2>参数误差</h2><img src="{asset('v6_vs_pass5_parameter_uncertainty_table.png')}" alt="LogPar parameter uncertainty table">
<table><thead><tr><th>参数</th><th>v6 值</th><th>v6 σ</th><th>Pass5 值</th><th>Pass5 σ</th><th>v6 误差减小</th></tr></thead><tbody>{rows}</tbody></table>
<h2>能谱相对不确定度</h2><img src="{asset('v6_vs_pass5_spectral_relative_uncertainty.png')}" alt="Spectral relative uncertainty comparison">
<p>使用 <code>Var[ln φ(E)] = g(E)^T C g(E)</code> 传播完整 covariance。正的 v6 reduction 表示 v6 的形式统计误差更小。</p>
<h2>最佳拟合能谱</h2><img src="{asset('v6_vs_pass5_logpar_spectrum.png')}" alt="Best-fit LogPar spectra">
<h2>参数相关性</h2><img src="{asset('v6_vs_pass5_correlation_matrices.png')}" alt="Correlation matrices">
<div class="callout warn"><strong>解释限制：</strong>这些是 HESSE 的形式统计误差，不含系统误差。v6 使用 conservative χ²，Pass5 使用 Poisson likelihood，目标函数并不完全相同；official Pass5 使用 30≤Nhit&lt;2000，而当前 v6 使用 100≤Nhit&lt;3000，分 bin 也不同。因此本图比较的是两条完整分析管线，不能把差异完全归因于 predE。v6 当前 Stage F 的 χ²/ndof = {summary['v6_diagnostics']['chi2_over_ndof']:.3f}，拟合优度较差，因此不能把较小 covariance 单独解释为最终总精度更高。两种方法使用同一观测样本，缺少跨方法 covariance，所以能谱比值带也不是严格的差异显著性。</div>
</main></body></html>"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    v6_values, v6_cov, v6_diagnostics = load_v6(args.v6_json)
    pass5_values, pass5_cov, pass5_diagnostics = load_pass5(args.pass5_yaml)
    gti_manifest = json.loads(args.gti_manifest.read_text(encoding="utf-8"))
    energies = np.geomspace(1.0, 100.0, 301)
    v6_relative = relative_uncertainty(v6_cov, energies)
    pass5_relative = relative_uncertainty(pass5_cov, energies)
    volume_ratio = math.sqrt(float(np.linalg.det(v6_cov) / np.linalg.det(pass5_cov)))
    v6_rows, pass5_rows = write_tables(args.output_dir, energies, v6_values, v6_cov, pass5_values, pass5_cov)

    plot_parameter_table(args.output_dir, v6_rows, pass5_rows)
    plot_correlations(args.output_dir, v6_cov, pass5_cov)
    plot_spectral_precision(args.output_dir, energies, v6_relative, pass5_relative, volume_ratio)
    plot_spectrum(args.output_dir, energies, v6_values, v6_relative, pass5_values, pass5_relative)

    parameter_comparison = []
    for name, (v6_value, v6_sigma), (p5_value, p5_sigma) in zip(["phi0_at_3tev", "alpha_at_3tev", "beta"], v6_rows, pass5_rows):
        parameter_comparison.append(
            {
                "parameter": name,
                "v6_value": v6_value,
                "v6_sigma": v6_sigma,
                "pass5_value": p5_value,
                "pass5_sigma": p5_sigma,
                "sigma_v6_over_pass5": v6_sigma / p5_sigma,
                "v6_reduction_percent": 100.0 * (1.0 - v6_sigma / p5_sigma),
            }
        )
    summary = {
        "sample": "Pass5 and v6 restricted to identical sorted recovered-time GTIs",
        "selected_hour_count": 3969,
        "gti_interval_count": int(gti_manifest["interval_count"]),
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
        "parameter_comparison": parameter_comparison,
        "v6_diagnostics": v6_diagnostics,
        "pass5_diagnostics": pass5_diagnostics,
        "caveats": [
            "Formal HESSE statistical covariance only; systematic uncertainty is excluded.",
            "v6 uses a conservative chi-square objective while Pass5 uses a Poisson likelihood, so the formal covariance definitions are not identical.",
            "Official Pass5 uses 30 <= Nhit < 2000 while the current v6 fit uses 100 <= Nhit < 3000 with different bin edges; this is a full-pipeline comparison, not an isolated predE ablation.",
            "The current v6 Stage F goodness of fit is poor, so a smaller formal covariance is not by itself proof of smaller total uncertainty.",
            "The methods use the same observation sample; no cross-method covariance is available for a rigorous difference significance.",
            "The plotted spectral-uncertainty range is 1-100 TeV. Pass5 remains an Nhit-only analysis and therefore has no event-by-event reconstructed-energy cut.",
        ],
    }
    (args.output_dir / "v6_vs_pass5_covariance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_html = args.report_html or (args.output_dir / "v6_vs_pass5_sorted_gti_covariance_report.html")
    write_html_report(args.output_dir, report_html, summary)


if __name__ == "__main__":
    main()
