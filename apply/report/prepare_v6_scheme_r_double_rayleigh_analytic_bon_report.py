#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = "v6_64748_nhit100_reselect44_split56_miss030"
PSF2 = f"{BASE}_double_rayleigh"
A1 = f"{BASE}_analytic_bon"
A2 = f"{PSF2}_analytic_bon"
E1 = f"{BASE}_scheme_R_fixed712979_analytic_bon"
E2 = f"{PSF2}_scheme_R_fixed712979_analytic_bon"
TARGET = 0.7129790300890827
REPORT = REPO_ROOT / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_analytic_bon_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-analytic-bon"
COMPARISON = ASSET_DIR / "scheme_R_double_rayleigh_analytic_bon_comparison.json"

R1_LEGACY_ASSETS = REPO_ROOT / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030/scheme-R"
R2_LEGACY_ASSETS = REPO_ROOT / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R"
R1_ANALYTIC_F = REPO_ROOT / f"apply/output/stage_f_{E1}/runs/{E1}_stage_f"
R1_ANALYTIC_G = REPO_ROOT / f"apply/output/stage_g_{E1}/runs/{E1}_stage_g"

D1_LEGACY = REPO_ROOT / f"apply/output/stage_d_{BASE}_annnorm/runs/{BASE}_stage_d_annnorm/background_{BASE}_annnorm.npz"
D2_LEGACY = REPO_ROOT / f"apply/output/stage_d_{PSF2}_annnorm/runs/{PSF2}_stage_d_annnorm/background_{PSF2}_annnorm.npz"
D1_ANALYTIC = REPO_ROOT / f"apply/output/stage_d_{A1}_annnorm/runs/{A1}_stage_d_annnorm/background_{A1}_annnorm.npz"
D2_ANALYTIC = REPO_ROOT / f"apply/output/stage_d_{A2}_annnorm/runs/{A2}_stage_d_annnorm/background_{A2}_annnorm.npz"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def copy_reference_images() -> dict[str, str]:
    sources = {
        "legacy_r1_pull": R1_LEGACY_ASSETS / "pull_grid_logpar.png",
        "legacy_r1_ratio": R1_LEGACY_ASSETS / "sed_points_ratio.png",
        "legacy_r2_pull": R2_LEGACY_ASSETS / "pull_grid_logpar.png",
        "legacy_r2_ratio": R2_LEGACY_ASSETS / "sed_points_ratio.png",
        "analytic_r1_pull": R1_ANALYTIC_F / "pull_grid_logpar.png",
        "analytic_r1_ratio": R1_ANALYTIC_G / "sed_points_ratio.png",
    }
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    prefix = ASSET_DIR.relative_to(REPORT.parent).as_posix()
    for key, source in sources.items():
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing comparison image: {source}")
        name = f"reference_{key}.png"
        shutil.copy2(source, ASSET_DIR / name)
        result[key] = f"{prefix}/{name}"
    result["analytic_r2_pull"] = f"{prefix}/pull_grid_logpar.png"
    result["analytic_r2_ratio"] = f"{prefix}/sed_points_ratio.png"
    return result


def cell_index(data: dict[str, np.ndarray], cell_id: int) -> int:
    matches = np.flatnonzero(np.asarray(data["cell_id"], dtype=np.int64) == int(cell_id))
    if matches.size != 1:
        raise ValueError(f"Expected one cell {cell_id}, found {matches.size}")
    return int(matches[0])


def make_cell1_radius_plot() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = {
        "1R": (load_npz(D1_LEGACY), load_npz(D1_ANALYTIC), "#1f77b4"),
        "2R": (load_npz(D2_LEGACY), load_npz(D2_ANALYTIC), "#d62728"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=170)
    for label, (legacy, analytic, color) in datasets.items():
        old_idx = cell_index(legacy, 1)
        new_idx = cell_index(analytic, 1)
        target_r = float(analytic["r_opt_deg"][new_idx])
        radii = np.linspace(0.76, 0.92, 500)
        rho = np.asarray(legacy["rho_grid_deg"], dtype=np.float64)
        background = np.asarray(legacy["background_map"][old_idx], dtype=np.float64)
        pixel = np.asarray([np.nansum(background[rho <= radius]) for radius in radii])
        coeff = np.asarray(analytic["surface_coefficients"][new_idx], dtype=np.float64)
        scale = float(analytic["annulus_surface_scale"][new_idx])
        step = float(np.asarray(analytic["on_aperture_grid_step_deg"], dtype=np.float64)[0])
        smooth = scale / step**2 * (
            math.pi * radii**2 * coeff[0]
            + 0.25 * math.pi * radii**4 * (coeff[3] + coeff[5])
        )
        ax.step(radii, pixel, where="post", color=color, alpha=0.45, linewidth=1.0, label=f"{label} legacy pixel-center")
        ax.plot(radii, smooth, color=color, linewidth=1.8, label=f"{label} analytic disk")
        ax.axvline(target_r, color=color, linestyle=":", linewidth=0.9)
    ax.set_xlabel("Cell 1 aperture radius (deg)")
    ax.set_ylabel("B_on expected counts")
    ax.set_title("Cell 1 background aperture integration")
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = ASSET_DIR / "cell1_bon_vs_radius_legacy_analytic.png"
    fig.savefig(output)
    plt.close(fig)
    return f"{ASSET_DIR.relative_to(REPORT.parent).as_posix()}/{output.name}"


def build_insert(payload: dict[str, Any], images: dict[str, str], radius_plot: str) -> str:
    records = payload["comparison"]
    branch_ids = [record["branch_id"] for record in records]
    comparison_rows = [
        [
            esc(record["label"]), esc(record["background_integration"]), fmt(record["phi0"], 7),
            fmt(record["alpha"]), fmt(record["beta"]), f"{fmt(record['chi2'], 4)}/{esc(record['ndof'])}",
            fmt(record["max_abs_pull"]), fmt(record["total_obs_over_pass5"]),
        ]
        for record in records
    ]
    cell1 = next(row for row in payload["cell_rows"] if int(row["cell_id"]) == 1)
    cell1_rows = []
    for record in records:
        values = cell1["branches"][record["branch_id"]]
        cell1_rows.append([
            esc(record["label"]), fmt(values["N_on"], 2), fmt(values["B_on"], 3),
            fmt(values["excess"], 3), fmt(values["pull"], 4),
        ])
    large_rows = []
    for row in sorted(payload["large_pull_cells"], key=lambda item: float(item["max_abs_pull"]), reverse=True):
        large_rows.append([esc(row["cell_id"])] + [fmt(row["branches"][branch_id]["pull"], 4) for branch_id in branch_ids])
    all_rows = []
    for row in payload["cell_rows"]:
        values = [row["branches"][branch_id] for branch_id in branch_ids]
        all_rows.append([
            esc(row["cell_id"]),
            fmt(values[0]["B_on"], 3), fmt(values[1]["B_on"], 3),
            fmt(values[2]["B_on"], 3), fmt(values[3]["B_on"], 3),
            fmt(values[1]["excess"] - values[0]["excess"], 3),
            fmt(values[3]["excess"] - values[2]["excess"], 3),
            fmt(values[3]["pull"], 4),
        ])
    figures = "".join(
        f'<figure class="figure"><img src="{esc(images[key])}" alt="{esc(key)}"><figcaption>{esc(key.replace("_", " "))}</figcaption></figure>'
        for key in ["legacy_r1_pull", "analytic_r1_pull", "legacy_r2_pull", "analytic_r2_pull", "legacy_r1_ratio", "analytic_r1_ratio", "legacy_r2_ratio", "analytic_r2_ratio"]
    )
    return f"""<section id="analytic-bon-contract">
    <h2>Analytic B_on Contract</h2>
    <div class="okbox"><strong>Active contract:</strong> the <code>0.1 deg</code> map still fits the annulus-normalized quadratic background shape, but active <code>B_on</code> is the closed-form integral over the true centered disk:
    <code>scale/h^2 * [pi r^2 c0 + pi r^4 (cxx+cyy)/4]</code>. Event-level <code>N_on</code> remains a continuous spherical-distance cut. All 44 selector-cell fitted surfaces are positive inside their on apertures.</div>
    <p>The response remains strict Scheme R: <code>Aeff_R={TARGET}*Aeff_nominal</code>, applied exactly once, with downstream containment equal to 1.</p>
  </section>
  <section id="four-branch-comparison">
    <h2>Four-Branch Controlled Comparison</h2>
    <p>Legacy versus analytic within the same PSF isolates the aperture-integration algorithm. Analytic 1R versus analytic 2R isolates the PSF/aperture model under one fixed response and one continuous background integral.</p>
    {table(["Branch", "B_on integration", "phi0", "alpha", "beta", "chi2/ndof", "max |pull|", "total obs/pass5"], comparison_rows)}
  </section>
  <section id="cell1-analytic-diagnostic">
    <h2>Cell 1 Diagnostic</h2>
    {table(["Branch", "N_on", "B_on", "excess", "LogPar pull"], cell1_rows)}
    <figure class="figure wide"><img src="{esc(radius_plot)}" alt="Cell 1 B_on radius comparison"><figcaption>Legacy pixel-center integration changes in steps, while the analytic quadratic disk integral varies continuously with aperture radius.</figcaption></figure>
  </section>
  <section id="large-pull-migration">
    <h2>Large-Pull Migration</h2>
    {table(["Cell"] + [record["label"] for record in records], large_rows)}
    <details><summary>All 44 cells: B_on, excess shifts, and active R-2R pull</summary>
    {table(["Cell", "R1 legacy B", "R1 analytic B", "R2 legacy B", "R2 analytic B", "R1 delta excess", "R2 delta excess", "R2 analytic pull"], all_rows)}
    </details>
    <div class="figgrid">{figures}</div>
  </section>"""


def main() -> None:
    if not REPORT.exists() or not COMPARISON.exists():
        raise FileNotFoundError("The generated analytic report and comparison JSON are required")
    text = REPORT.read_text(encoding="utf-8")
    if "analytic-bon-contract" in text:
        raise ValueError("Analytic B_on report is already prepared")
    payload = json.loads(COMPARISON.read_text(encoding="utf-8"))
    images = copy_reference_images()
    radius_plot = make_cell1_radius_plot()
    marker = "<section>\n    <h2>Stage C Time Audit</h2>"
    if marker not in text:
        raise ValueError("Could not find Stage C insertion marker")
    text = text.replace(marker, build_insert(payload, images, radius_plot) + "\n" + marker, 1)
    temporary = REPORT.with_suffix(REPORT.suffix + ".analytic_bon_tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(REPORT)
    payload["report_path"] = str(REPORT)
    payload["report_sha256"] = sha256(REPORT)
    COMPARISON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared analytic B_on report: {REPORT}")


if __name__ == "__main__":
    main()
