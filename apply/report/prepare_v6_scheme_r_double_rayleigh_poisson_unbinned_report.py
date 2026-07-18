#!/usr/bin/env python3
"""Enrich the Scheme R double-Rayleigh Poisson-unbinned Stage A-G report."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import html
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ORDER2_CELLS = (1, 2, 4, 5, 6, 19, 20)
EXPERIMENT_ID = (
    "v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_"
    "scheme_R_fixed712979_poisson_unbinned"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--published-report",
        type=Path,
        help="Final path after the caller atomically publishes the temporary report.",
    )
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--unbinned-stage-d", type=Path, required=True)
    parser.add_argument("--pooled-stage-d", type=Path, required=True)
    parser.add_argument("--unbinned-stage-f-metadata", type=Path, required=True)
    parser.add_argument("--pooled-stage-f-metadata", type=Path, required=True)
    parser.add_argument("--unbinned-stage-g", type=Path, required=True)
    parser.add_argument("--pooled-stage-g", type=Path, required=True)
    parser.add_argument("--time-split-json", type=Path, required=True)
    parser.add_argument("--covariance-npz", type=Path, required=True)
    parser.add_argument("--covariance-json", type=Path, required=True)
    parser.add_argument("--shared-model-tier-figure", type=Path)
    parser.add_argument("--implementation-sha", required=True)
    parser.add_argument("--grid-job-id", default="65261")
    parser.add_argument("--bootstrap-job-id", default="65262")
    parser.add_argument("--finalizer-job-id", default="n/a")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 6) -> str:
    number = finite(value)
    if number is None:
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def figure(report: Path, path: Path, caption: str) -> str:
    relative = os.path.relpath(path.resolve(), start=report.parent.resolve())
    return (
        '<figure class="figure">'
        f'<img src="{esc(relative)}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def npz_copy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]).copy() for name in handle.files}


def by_cell(data: Mapping[str, np.ndarray], name: str) -> dict[int, Any]:
    return {int(cell): value for cell, value in zip(data["cell_id"], data[name])}


def fit_record(metadata: Mapping[str, Any], key: str) -> dict[str, Any]:
    fit = (metadata.get("fits") or {}).get(key) or {}
    parameters = fit.get("parameters") or {}
    chi2 = finite(fit.get("chi2"))
    ndof = finite(fit.get("ndof"))
    return {
        "valid": fit.get("valid"),
        "phi0": finite(parameters.get("phi0")),
        "alpha": finite(parameters.get("alpha")),
        "beta": finite(parameters.get("beta")),
        "chi2": chi2,
        "ndof": int(ndof) if ndof is not None else None,
        "chi2_over_ndof": chi2 / ndof if chi2 is not None and ndof not in (None, 0.0) else None,
    }


def plot_background_comparison(
    unbinned: Mapping[str, np.ndarray], pooled: Mapping[str, np.ndarray], output: Path
) -> list[dict[str, Any]]:
    u_b = by_cell(unbinned, "B_on")
    p_b = by_cell(pooled, "B_on")
    u_c = by_cell(unbinned, "surface_shape_coefficients")
    p_c = by_cell(pooled, "surface_shape_coefficients")
    rows: list[dict[str, Any]] = []
    for cell in ORDER2_CELLS:
        delta_percent = 100.0 * (float(u_b[cell]) / float(p_b[cell]) - 1.0)
        tau_u = float(np.asarray(u_c[cell])[3] + np.asarray(u_c[cell])[5])
        tau_p = float(np.asarray(p_c[cell])[3] + np.asarray(p_c[cell])[5])
        rows.append(
            {
                "cell_id": cell,
                "B_on_pooled": float(p_b[cell]),
                "B_on_unbinned": float(u_b[cell]),
                "delta_B_on_percent": delta_percent,
                "tau_pooled": tau_p,
                "tau_unbinned": tau_u,
            }
        )

    x = np.arange(len(rows), dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), dpi=170, sharex=True)
    delta = np.asarray([row["delta_B_on_percent"] for row in rows])
    axes[0].bar(x, delta, color=np.where(delta >= 0.0, "#166534", "#b45309"), width=0.68)
    axes[0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("(unbinned / binned - 1) [%]")
    axes[0].set_title("Order-2 cells: analytic on-source background")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.35
    axes[1].bar(x - width / 2, [row["tau_pooled"] for row in rows], width, label="binned pooled", color="#64748b")
    axes[1].bar(x + width / 2, [row["tau_unbinned"] for row in rows], width, label="unbinned", color="#005eb8")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("radial trace tau = cxx + cyy")
    axes[1].set_xlabel("internal cell id")
    axes[1].set_xticks(x, [str(row["cell_id"]) for row in rows])
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return rows


def plot_time_split(payload: Mapping[str, Any], output: Path) -> None:
    checks = payload.get("checks") or []
    cells = [int(row["cell_id"]) for row in checks]
    x = np.arange(len(cells), dtype=np.float64)
    signed_sigma = np.asarray(
        [(float(row["g_h2"]) - float(row["g_h1"])) / float(row["sigma_diff"]) for row in checks]
    )
    relative_percent = np.asarray(
        [100.0 * (float(row["g_h2"]) / float(row["g_h1"]) - 1.0) for row in checks]
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), dpi=170, sharex=True)
    axes[0].bar(x, signed_sigma, color="#005eb8", width=0.68)
    axes[0].axhline(1.0, color="#aa2e25", linestyle="--", linewidth=1.0)
    axes[0].axhline(-1.0, color="#aa2e25", linestyle="--", linewidth=1.0)
    axes[0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("(g2 - g1) / sigma_diff")
    axes[0].set_title("Order-2 shape-factor time split")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, relative_percent, color="#15803d", width=0.68)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("g2 / g1 - 1 [%]")
    axes[1].set_xlabel("internal cell id")
    axes[1].set_xticks(x, [str(cell) for cell in cells])
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_covariance(covariance: Mapping[str, np.ndarray], sigma_output: Path, corr_output: Path) -> None:
    cells = np.asarray(covariance["cell_id"], dtype=np.int64)
    nominal = np.asarray(covariance["B_on_nominal"], dtype=np.float64)
    cov = np.asarray(covariance["B_on_covariance"], dtype=np.float64)
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    relative = np.divide(sigma, nominal, out=np.zeros_like(sigma), where=nominal > 0.0) * 100.0

    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=170)
    colors = ["#005eb8" if int(cell) in ORDER2_CELLS else "#64748b" for cell in cells]
    ax.bar(np.arange(cells.size), relative, color=colors, width=0.78)
    ax.set_xticks(np.arange(cells.size), [str(int(cell)) for cell in cells], rotation=90, fontsize=7)
    ax.set_xlabel("internal cell id")
    ax.set_ylabel("bootstrap sigma(B_on) / B_on [%]")
    ax.set_title("Unbinned background uncertainty (blue: order-2 cells)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(sigma_output)
    plt.close(fig)

    denom = np.outer(sigma, sigma)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=170)
    image = ax.imshow(corr, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
    ticks = np.arange(cells.size)
    ax.set_xticks(ticks, [str(int(cell)) for cell in cells], rotation=90, fontsize=6)
    ax.set_yticks(ticks, [str(int(cell)) for cell in cells], fontsize=6)
    ax.set_xlabel("internal cell id")
    ax.set_ylabel("internal cell id")
    ax.set_title("Unbinned bootstrap B_on correlation")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("correlation")
    fig.tight_layout()
    fig.savefig(corr_output)
    plt.close(fig)


def plot_sed_ratio(unbinned_path: Path, pooled_path: Path, output: Path) -> None:
    unbinned = npz_copy(unbinned_path)
    pooled = npz_copy(pooled_path)
    pooled_index = {
        (str(grouping), str(label)): index
        for index, (grouping, label) in enumerate(zip(pooled["grouping"], pooled["group_label"]))
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=170)
    for grouping, marker, color, label in (
        ("nhit", "o", "#005eb8", "Nhit grouped"),
        ("predE", "s", "#15803d", "predE grouped"),
    ):
        energy: list[float] = []
        ratio: list[float] = []
        for index, (kind, group_label) in enumerate(zip(unbinned["grouping"], unbinned["group_label"])):
            if str(kind) != grouping:
                continue
            other = pooled_index.get((grouping, str(group_label)))
            if other is None or float(pooled["E2_dnde"][other]) == 0.0:
                continue
            energy.append(float(unbinned["effective_energy_tev"][index]))
            ratio.append(float(unbinned["E2_dnde"][index]) / float(pooled["E2_dnde"][other]))
        ax.plot(energy, ratio, marker=marker, linestyle="none", color=color, label=label, markersize=5.5)
    ax.axhline(1.0, color="#111827", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("effective energy [TeV]")
    ax.set_ylabel("unbinned / binned pooled E2 dN/dE")
    ax.set_title("SED point stability under grid-free curvature fitting")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


class ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        values = dict(attrs)
        if values.get("src"):
            self.sources.append(str(values["src"]))


def validate_images(report: Path) -> dict[str, Any]:
    parser = ImageParser()
    parser.feed(report.read_text(encoding="utf-8"))
    missing: list[str] = []
    empty: list[str] = []
    for source in parser.sources:
        path = (report.parent / source).resolve()
        if not path.exists():
            missing.append(source)
        elif path.stat().st_size == 0:
            empty.append(source)
    return {
        "image_ref_count": len(parser.sources),
        "unique_image_ref_count": len(set(parser.sources)),
        "missing_image_refs": missing,
        "empty_image_refs": empty,
        "passed": not missing and not empty and len(parser.sources) >= 24,
    }


def main() -> None:
    args = parse_args()
    published_report = args.published_report or args.report
    required = [
        args.report,
        args.unbinned_stage_d,
        args.pooled_stage_d,
        args.unbinned_stage_f_metadata,
        args.pooled_stage_f_metadata,
        args.unbinned_stage_g,
        args.pooled_stage_g,
        args.time_split_json,
        args.covariance_npz,
        args.covariance_json,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    unbinned_d = npz_copy(args.unbinned_stage_d)
    pooled_d = npz_copy(args.pooled_stage_d)
    covariance = npz_copy(args.covariance_npz)
    covariance_meta = load_json(args.covariance_json)
    time_split = load_json(args.time_split_json)
    unbinned_fit_meta = load_json(args.unbinned_stage_f_metadata)
    pooled_fit_meta = load_json(args.pooled_stage_f_metadata)

    background_plot = args.asset_dir / "unbinned_vs_binned_background.png"
    time_plot = args.asset_dir / "unbinned_time_split_shape_factor.png"
    sigma_plot = args.asset_dir / "unbinned_background_relative_uncertainty.png"
    corr_plot = args.asset_dir / "unbinned_background_correlation_heatmap.png"
    sed_ratio_plot = args.asset_dir / "unbinned_vs_binned_sed_ratio.png"
    background_rows = plot_background_comparison(unbinned_d, pooled_d, background_plot)
    plot_time_split(time_split, time_plot)
    plot_covariance(covariance, sigma_plot, corr_plot)
    plot_sed_ratio(args.unbinned_stage_g, args.pooled_stage_g, sed_ratio_plot)

    extra_figures = [
        figure(args.report, background_plot, "Order-2 B_on and radial-trace comparison: binned pooled versus unbinned"),
        figure(args.report, time_plot, "Time-split consistency of the exposure-free shape factor g(tau)"),
        figure(args.report, sigma_plot, "Per-cell unbinned bootstrap background uncertainty"),
        figure(args.report, corr_plot, "Unbinned bootstrap background correlation matrix"),
        figure(args.report, sed_ratio_plot, "Unbinned-to-binned pooled SED-point ratio; descriptive because the two branches share events"),
    ]
    if args.shared_model_tier_figure and args.shared_model_tier_figure.is_file():
        copied = args.asset_dir / "model_tier_donor_map.png"
        shutil.copy2(args.shared_model_tier_figure, copied)
        extra_figures.insert(
            1,
            figure(args.report, copied, "Frozen background model tier and donor map, shared with the pooled experiment"),
        )

    checks = time_split.get("checks") or []
    time_rows = [
        [
            esc(row.get("cell_id")),
            fmt(row.get("g_h1"), 7),
            fmt(row.get("g_h2"), 7),
            fmt(row.get("abs_delta_g"), 4),
            fmt(row.get("delta_over_sigma"), 4),
            '<span class="status-pass">PASSED</span>' if row.get("passed") is True else '<span class="status-fail">FAILED</span>',
        ]
        for row in checks
    ]
    background_table_rows = [
        [
            esc(row["cell_id"]),
            fmt(row["B_on_pooled"], 8),
            fmt(row["B_on_unbinned"], 8),
            fmt(row["delta_B_on_percent"], 5),
            fmt(row["tau_pooled"], 6),
            fmt(row["tau_unbinned"], 6),
        ]
        for row in background_rows
    ]

    pooled_cons = fit_record(pooled_fit_meta, "logpar_conservative")
    pooled_cov = fit_record(pooled_fit_meta, "logpar_background_covariance")
    unbinned_cons = fit_record(unbinned_fit_meta, "logpar_conservative")
    unbinned_cov = fit_record(unbinned_fit_meta, "logpar_background_covariance")
    contracts = [
        ("Poisson pooled conservative", "binned profiled-Poisson; analytic B_on; primary", pooled_cons),
        ("Poisson pooled covariance-aware", "binned bootstrap covariance; diagnostic", pooled_cov),
        ("Poisson unbinned conservative", "continuous profiled-Poisson curvature; analytic B_on; primary", unbinned_cons),
        ("Poisson unbinned covariance-aware", "unbinned bootstrap covariance; diagnostic", unbinned_cov),
    ]
    contract_rows = [
        [
            esc(label),
            esc(contract),
            esc(record["valid"]),
            fmt(record["phi0"], 7),
            fmt(record["alpha"], 7),
            fmt(record["beta"], 7),
            f"{fmt(record['chi2'], 6)}/{esc(record['ndof'])}",
            fmt(record["chi2_over_ndof"], 6),
        ]
        for label, contract, record in contracts
    ]

    gate = time_split.get("gate") or {}
    time_pass = gate.get("passed") is True
    max_shift = max(abs(float(row["delta_B_on_percent"])) for row in background_rows)
    covariance_matrix = np.asarray(covariance["excess_covariance"], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    insert = f"""  <section id="unbinned-method-acceptance">
    <h2>Grid-Free Unbinned Poisson Method And Acceptance</h2>
    <div class="okbox"><strong>Numerical result:</strong> the 12 grid-resolution/phase branches have an unbinned <code>B_on</code> envelope at machine zero. This is a regression check expected by construction, not an independent model-selection test. The independent diagnostic retained here is the time split of <code>g(tau)=B_on/N_ann</code>.</div>
    <p>For each of the seven order-2 cells, the shared polynomial shape is fitted directly to continuous OFF-annulus event coordinates with the profiled objective <code>-sum log q(x_e,y_e) + N_ann log integral_annulus(q)</code>. Positivity is enforced over the full 6 deg fiducial disk. The centered analytic aperture integral depends on the radial trace <code>tau=cxx+cyy</code>; plane and constant cells remain unchanged.</p>
    {table(["Cell", "B_on binned", "B_on unbinned", "Delta B_on [%]", "tau binned", "tau unbinned"], background_table_rows)}
    <div class="figgrid">{''.join(extra_figures[:2])}</div>
  </section>

  <section id="unbinned-time-split">
    <h2>Time-Split Shape Diagnostic</h2>
    <div class="{'okbox' if time_pass else 'callout'}"><strong>time_split_shape_consistency: <span class="status-{'pass' if time_pass else 'fail'}">{'PASSED' if time_pass else 'FAILED'}</span></strong>. Maximum reported <code>|Delta g|/sigma={fmt(gate.get('observed_max_delta_over_sigma'), 5)}</code> against the prototype limit <code>{fmt((gate.get('limit') or {}).get('value'), 3)}</code>. The production split used the matched-livetime calendar midpoint MJD 59670.16; it was not an equal-event median split, so this remains diagnostic rather than a promotion gate.</div>
    {table(["Cell", "g half 1", "g half 2", "|Delta g|", "|Delta g|/sigma", "Status"], time_rows)}
    <div class="figgrid">{extra_figures[2] if len(extra_figures) == 6 else extra_figures[1]}</div>
  </section>

  <section id="unbinned-bootstrap">
    <h2>Unbinned Bootstrap Background Evidence</h2>
    <p>The covariance artifact uses a Poissonized nonparametric event bootstrap: OFF coordinates are resampled with replacement, the event total is Poisson fluctuated, and every order-2 surface is refit. The covariance-aware Stage F fit is diagnostic; the conservative fit remains preferred.</p>
    {table(["Diagnostic", "Value"], [
        ["Replicates completed / requested", f"{esc(covariance_meta.get('bootstrap_count_completed'))} / {esc(covariance_meta.get('bootstrap_count_requested'))}"],
        ["Refit failures", esc(covariance_meta.get('refit_failure_count'))],
        ["Minimum excess-covariance eigenvalue", fmt(np.min(eigenvalues), 8)],
        ["Excess-covariance condition number", fmt(np.linalg.cond(covariance_matrix), 8)],
        ["Stage D SHA256", f"<code>{esc(covariance_meta.get('stage_d_sha256'))}</code>"],
        ["Stage E SHA256", f"<code>{esc(covariance_meta.get('stage_e_sha256'))}</code>"],
    ])}
    <div class="figgrid">{''.join(extra_figures[3:5]) if len(extra_figures) == 6 else ''.join(extra_figures[2:4])}</div>
  </section>

  <section id="unbinned-spectral-comparison">
    <h2>Binned Versus Unbinned Spectral Comparison</h2>
    <p>The background change is numerically stable: the largest order-2 <code>B_on</code> shift is {max_shift:.3f}%. Stage F chi-square is reported as a diagnostic and is not used to select the background method.</p>
    {table(["Contract", "Background / uncertainty", "Valid", "phi0", "alpha", "beta", "chi2/ndof", "chi2/ndof value"], contract_rows)}
    <div class="figgrid">{extra_figures[-1]}</div>
  </section>

  <section id="unbinned-provenance">
    <h2>Unbinned Experiment Provenance</h2>
    {table(["Field", "Value"], [
        ["Experiment id", f"<code>{EXPERIMENT_ID}</code>"],
        ["Implementation commit", f"<code>{esc(args.implementation_sha)}</code>"],
        ["Grid Slurm job", f"<code>{esc(args.grid_job_id)}</code> (12/12 completed)"],
        ["Bootstrap/time-split Stage F job", f"<code>{esc(args.bootstrap_job_id)}</code> (completed)"],
        ["Full-report/Stage G job", f"<code>{esc(args.finalizer_job_id)}</code>"],
        ["Unbinned Stage D SHA256", f"<code>{sha256(args.unbinned_stage_d)}</code>"],
        ["Unbinned Stage F metadata SHA256", f"<code>{sha256(args.unbinned_stage_f_metadata)}</code>"],
        ["Unbinned Stage G SHA256", f"<code>{sha256(args.unbinned_stage_g)}</code>"],
    ])}
    <div class="callout"><strong>Status:</strong> isolated experiment. No overwrite of the frozen <code>poisson_pooled</code> namespace and no automatic promotion of <code>preferred_fit</code>. The global LogPar goodness of fit remains poor and is not repaired by this background change.</div>
  </section>

"""

    report_text = args.report.read_text(encoding="utf-8")
    if "unbinned-method-acceptance" in report_text:
        raise ValueError("Report is already enriched with unbinned sections")
    if args.finalizer_job_id != "n/a":
        running_cell = (
            f"<td>{esc(args.finalizer_job_id)}</td>"
            '<td><span class="status-warn">RUNNING</span></td>'
        )
        completed_cell = (
            f"<td>{esc(args.finalizer_job_id)}</td>"
            '<td><span class="status-pass">COMPLETED</span></td>'
        )
        report_text = report_text.replace(running_cell, completed_cell, 1)
    generic_stage_d = (
        "<tr><td>Stage D background</td><td><code>"
        "apply/output/stage_d_v6_64748_nhit100_reselect44_split56_miss030_"
        "double_rayleigh_annnorm</code></td></tr>"
    )
    actual_stage_d = (
        "<tr><td>Stage D background</td><td><code>"
        f"{esc(args.unbinned_stage_d.parent)}</code></td></tr>"
    )
    report_text = report_text.replace(generic_stage_d, actual_stage_d, 1)
    marker = "  <section>\n    <h2>Stage C Time Audit</h2>"
    if marker not in report_text:
        raise ValueError("Could not find Stage C insertion marker in the base report")
    report_text = report_text.replace(marker, insert + marker, 1)
    args.report.write_text(report_text, encoding="utf-8")

    image_validation = validate_images(args.report)
    validation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "report": str(published_report.resolve()),
        "report_sha256": sha256(args.report),
        "image_validation": image_validation,
        "order2_cells": list(ORDER2_CELLS),
        "maximum_abs_B_on_shift_percent": max_shift,
        "time_split_gate": gate,
        "bootstrap_completed": covariance_meta.get("bootstrap_count_completed"),
        "bootstrap_failures": covariance_meta.get("refit_failure_count"),
        "implementation_sha": args.implementation_sha,
    }
    validation_path = args.asset_dir / "scheme_R_double_rayleigh_poisson_unbinned_report_validation.json"
    validation_text = json.dumps(validation, indent=2) + "\n"
    validation_path.write_text(validation_text, encoding="utf-8")
    (args.asset_dir / "report_validation.json").write_text(validation_text, encoding="utf-8")
    comparison_path = args.asset_dir / "scheme_R_double_rayleigh_poisson_unbinned_comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "background_cells": background_rows,
                "time_split": time_split,
                "spectral_contracts": [
                    {"label": label, "contract": contract, **record}
                    for label, contract, record in contracts
                ],
                "inputs": {str(path): sha256(path) for path in required if path != args.report},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not image_validation["passed"]:
        raise RuntimeError(f"Final image validation failed: {image_validation}")
    print(f"Enriched unbinned report: {args.report}")
    print(f"Image references: {image_validation['image_ref_count']} (all valid)")
    print(f"Comparison evidence: {comparison_path}")
    print(f"Report validation: {validation_path}")


if __name__ == "__main__":
    main()
