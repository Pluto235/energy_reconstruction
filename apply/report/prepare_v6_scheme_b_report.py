#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "apply/report"
BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
RUN_ID = os.environ.get("V6_SCHEME_B_RUN_ID", f"{BASE_RUN_ID}_double_rayleigh")
SOURCE = Path(
    os.environ.get(
        "V6_SCHEME_B_SOURCE_REPORT",
        str(REPORT_DIR / f"crab_sed_{RUN_ID}_stage_a_to_g_report.html"),
    )
)
TARGET = REPORT_DIR / "crab_sed_v6_64748_nhit100_reselect44_scheme_B_report.html"
NEW_TITLE = "Crab SED v6 64748 reselect44 - Scheme B - Double-Rayleigh MC Aperture"
TARGET_CONTAINMENT = 0.7129790300890827
R_OPT_FACTOR = 1.58
M2_TO_CM2 = 1.0e4

SELECTOR = REPO_ROOT / "apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
ASSET_DIR = REPORT_DIR / "assets" / RUN_ID.replace("_", "-")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 5) -> str:
    number = finite(value)
    if number is None:
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e4 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def pass5_points() -> tuple[np.ndarray, np.ndarray]:
    energy: list[float] = []
    flux: list[float] = []
    with PASS5_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            e = finite(row.get("energy_tev"))
            dnde = finite(row.get("flux_per_tev_cm2_s"))
            if e is not None and dnde is not None and e > 0.0 and dnde > 0.0:
                energy.append(e)
                flux.append(dnde)
    return np.asarray(energy, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def loglog_interp(energy_tev: np.ndarray, knots_energy: np.ndarray, knots_flux: np.ndarray) -> np.ndarray:
    x = np.log(np.asarray(energy_tev, dtype=np.float64))
    xk = np.log(knots_energy)
    yk = np.log(knots_flux)
    slopes = np.diff(yk) / np.diff(xk)
    y = np.interp(x, xk, yk)
    low = x < xk[0]
    high = x > xk[-1]
    y[low] = yk[0] + slopes[0] * (x[low] - xk[0])
    y[high] = yk[-1] + slopes[-1] * (x[high] - xk[-1])
    return np.exp(y)


def integrate_pass5(loge_edges: np.ndarray) -> np.ndarray:
    energy, flux = pass5_points()
    nodes, weights = np.polynomial.legendre.leggauss(96)
    out = np.zeros(loge_edges.size - 1, dtype=np.float64)
    for idx, (lo, high) in enumerate(zip(loge_edges[:-1], loge_edges[1:])):
        loge = 0.5 * (high - lo) * nodes + 0.5 * (high + lo)
        energy_tev = np.power(10.0, loge) / 1000.0
        integrand = loglog_interp(energy_tev, energy, flux) * math.log(10.0) * energy_tev
        out[idx] = 0.5 * (high - lo) * float(np.sum(weights * integrand))
    return out


def official_expected_counts(response: dict[str, np.ndarray], fit: dict[str, np.ndarray]) -> np.ndarray:
    flux_integral = integrate_pass5(np.asarray(response["logE_true_edges"], dtype=np.float64))
    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    fit_ids = np.asarray(fit["cell_id"], dtype=np.int64)
    index_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(response_ids)}
    missing = [int(cell_id) for cell_id in fit_ids if int(cell_id) not in index_by_cell]
    if missing:
        raise ValueError(f"Response is missing Stage F cells: {missing}")
    indices = np.asarray([index_by_cell[int(cell_id)] for cell_id in fit_ids], dtype=np.int64)
    return M2_TO_CM2 * np.einsum(
        "bet,e,t->b",
        np.asarray(response["a_eff"], dtype=np.float64)[indices],
        flux_integral,
        np.asarray(fit["theta_exposure_sec"], dtype=np.float64),
    )


def branch_paths(run_id: str) -> tuple[Path, Path, Path]:
    response = REPO_ROOT / f"apply/output/stage_a_{run_id}_aperture_conditioned/response_2d_{run_id}_aperture_conditioned.npz"
    fit_dir = REPO_ROOT / f"apply/output/stage_f_{run_id}/runs/{run_id}_stage_f"
    return response, fit_dir / f"fit_{run_id}.npz", fit_dir / f"fit_{run_id}_metadata.json"


def branch_record(label: str, run_id: str) -> tuple[dict[str, Any], dict[str, float]]:
    response_path, fit_path, meta_path = branch_paths(run_id)
    response = read_npz(response_path)
    fit = read_npz(fit_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    logpar = (meta.get("fits") or {}).get("logpar_conservative") or {}
    params = logpar.get("parameters") or {}
    pulls = np.asarray(fit["logpar_conservative_pull"], dtype=np.float64)
    expected = official_expected_counts(response, fit)
    observed = float(np.nansum(np.asarray(fit["excess"], dtype=np.float64)))
    pass5 = float(np.nansum(expected))
    pull_by_cell = {
        int(cell_id): float(pull)
        for cell_id, pull in zip(np.asarray(fit["cell_id"], dtype=np.int64), pulls)
        if np.isfinite(pull)
    }
    record = {
        "label": label,
        "run_id": run_id,
        "phi0": finite(params.get("phi0")),
        "alpha": finite(params.get("alpha")),
        "beta": finite(params.get("beta")),
        "chi2": finite(logpar.get("chi2")),
        "ndof": int(logpar.get("ndof")) if finite(logpar.get("ndof")) is not None else None,
        "chi2_over_ndof": finite(logpar.get("chi2_over_ndof")),
        "max_abs_pull": float(np.nanmax(np.abs(pulls))),
        "total_observed_excess": observed,
        "total_pass5_expected": pass5,
        "total_obs_over_pass5": observed / pass5 if pass5 > 0.0 else None,
    }
    return record, pull_by_cell


def psf_rows() -> tuple[list[dict[str, str]], set[int]]:
    summary = REPO_ROOT / f"apply/output/stage_b_{RUN_ID}/runs/{RUN_ID}_stage_b_psf/psf_{RUN_ID}_summary.csv"
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with SELECTOR.open(newline="", encoding="utf-8") as handle:
        selected = {
            int(float(row["cell_id"]))
            for row in csv.DictReader(handle)
            if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
        }
    return rows, selected


def build_insert() -> tuple[str, dict[str, Any]]:
    old_record, old_pulls = branch_record("Old Scheme B single-Rayleigh", BASE_RUN_ID)
    new_record, new_pulls = branch_record("New Scheme B double-Rayleigh", RUN_ID)
    rows, selected = psf_rows()
    fallback_rows = [row for row in rows if str(row.get("double_rayleigh_fit_quality", "")).startswith("fallback:")]
    selected_fallback_rows = [row for row in fallback_rows if int(row["cell_id"]) in selected]
    old_psf_path = REPO_ROOT / f"apply/output/stage_b_{BASE_RUN_ID}/runs/{BASE_RUN_ID}_stage_b_psf/psf_{BASE_RUN_ID}.npz"
    old_psf = read_npz(old_psf_path)
    old_r_opt_by_cell = {
        int(cell_id): float(radius)
        for cell_id, radius in zip(old_psf["cell_id"], old_psf["r_opt_deg"])
    }

    comparison_rows = []
    for record in [old_record, new_record]:
        comparison_rows.append(
            [
                esc(record["label"]),
                fmt(record["phi0"], 7),
                fmt(record["alpha"], 6),
                fmt(record["beta"], 6),
                f"{fmt(record['chi2'], 5)}/{esc(record['ndof'])}",
                fmt(record["chi2_over_ndof"], 5),
                fmt(record["max_abs_pull"], 5),
                fmt(record["total_obs_over_pass5"], 5),
            ]
        )

    focus_ids = sorted(
        (cell_id for cell_id, pull in old_pulls.items() if abs(pull) >= 5.0),
        key=lambda cell_id: abs(old_pulls[cell_id]),
        reverse=True,
    )
    by_cell = {int(row["cell_id"]): row for row in rows}
    focus_rows = []
    for cell_id in focus_ids:
        row = by_cell[cell_id]
        focus_rows.append(
            [
                esc(cell_id),
                f"<code>{esc(row.get('nhit_bin'))}</code>",
                f"<code>{esc(row.get('predE_bin'))}</code>",
                fmt(old_pulls.get(cell_id), 5),
                fmt(new_pulls.get(cell_id), 5),
                fmt(old_r_opt_by_cell.get(cell_id), 5),
                fmt(row.get("double_rayleigh_r_opt_deg"), 5),
                esc(row.get("double_rayleigh_fit_quality") or "n/a"),
            ]
        )

    psf_table_rows = []
    for row in rows:
        cell_id = int(row["cell_id"])
        psf_table_rows.append(
            [
                f"<strong>{cell_id}</strong>" if cell_id in selected else esc(cell_id),
                "yes" if cell_id in selected else "no",
                f"<code>{esc(row.get('nhit_bin'))}</code>",
                f"<code>{esc(row.get('predE_bin'))}</code>",
                fmt(row.get("double_rayleigh_A"), 5),
                fmt(row.get("double_rayleigh_sigma1_deg"), 5),
                fmt(row.get("double_rayleigh_sigma2_deg"), 5),
                fmt(row.get("double_rayleigh_r_opt_deg"), 5),
                fmt(row.get("double_rayleigh_sigma_eq_deg"), 5),
                fmt(row.get("double_rayleigh_containment_r_opt"), 5),
                fmt(row.get("double_rayleigh_chi2_ndof"), 5),
                esc(row.get("double_rayleigh_fit_quality") or "n/a"),
                esc(row.get("double_rayleigh_fallback_reason") or ""),
            ]
        )

    contract = f"""<section id="double-rayleigh-contract">
    <h2>Double-Rayleigh PSF Contract</h2>
    <div class="okbox"><strong>Active Scheme B contract:</strong>
      Stage B fits the same Crab-theta-weighted MC radial profile as the former single-Rayleigh branch with
      <code>p(r)=A*r/sigma1^2*exp[-r^2/(2*sigma1^2)] + (1-A)*r/sigma2^2*exp[-r^2/(2*sigma2^2)]</code>,
      constrained by <code>0&lt;A&lt;1</code> and <code>0&lt;sigma1&lt;sigma2</code>. The aperture radius is the numerical solution of
      <code>F(r_opt)={TARGET_CONTAINMENT}</code>. <code>sigma_eq=r_opt/{R_OPT_FACTOR}</code> is report-only and never drives the aperture.
      Source low-stat/fallback cells and failed or unstable mixture fits retain the existing Rayleigh/psfborrow aperture with a recorded reason.
      The new aperture rebuilds the aperture-conditioned Stage A response; Stage C is reused byte-for-byte; Stage D is rerun with annnorm,
      Stage E uses containment=1, and Stage F/G keep the previous exposure, background, spectral, selector, and SED-grouping contracts.
    </div>
    <p><strong>Selector:</strong> <code>apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv</code>
      ({len(selected)} included cells). <strong>Fallbacks:</strong> {len(fallback_rows)} of {len(rows)} formal cells, including {len(selected_fallback_rows)} of the 44 selected cells; fallback cells outside the fit remain diagnostic.</p>
    <h3>Per-cell mixture fit and aperture</h3>
    {html_table(["Cell", "Fit", "Nhit", "predE", "A", "sigma1 deg", "sigma2 deg", "r_opt deg", "sigma_eq deg", "Empirical containment", "fit chi2/ndof", "Fit quality", "Fallback reason"], psf_table_rows)}
  </section>

  <section id="scheme-b-comparison">
    <h2>Old Rayleigh vs New Double-Rayleigh Scheme B</h2>
    <p><code>total obs/pass5</code> is the selected-cell Stage E excess divided by the official Pass5 spectrum forward-folded through each branch's own aperture-conditioned Stage A response and the unchanged Stage F exposure.</p>
    {html_table(["Branch", "phi0", "alpha", "beta", "chi2/ndof", "chi2/ndof value", "max |pull|", "total obs/pass5"], comparison_rows)}
    <h3>Original large-pull cells (old |pull| &gt;= 5)</h3>
    {html_table(["Cell", "Nhit", "predE", "Old pull", "New pull", "Old Rayleigh r_opt", "New r_opt", "New PSF status"], focus_rows)}
  </section>
"""
    payload = {
        "run_id": RUN_ID,
        "target_containment": TARGET_CONTAINMENT,
        "sigma_eq_contract": "r_opt/1.58, reporting only",
        "selector": str(SELECTOR),
        "selected_cells": sorted(selected),
        "fallback_cell_ids": [int(row["cell_id"]) for row in fallback_rows],
        "selected_fallback_cell_ids": [int(row["cell_id"]) for row in selected_fallback_rows],
        "comparison": [old_record, new_record],
        "old_large_pull_threshold": 5.0,
        "old_large_pull_cells": [
            {
                "cell_id": cell_id,
                "old_pull": old_pulls[cell_id],
                "new_pull": new_pulls.get(cell_id),
            }
            for cell_id in focus_ids
        ],
    }
    return contract, payload


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing double-Rayleigh Scheme B source report: {SOURCE}")

    before = sha256(TARGET) if TARGET.exists() else None
    text = SOURCE.read_text(encoding="utf-8")
    title_start = text.find("<title>")
    title_end = text.find("</title>", title_start)
    if title_start >= 0 and title_end >= 0:
        text = text[: title_start + len("<title>")] + esc(NEW_TITLE) + text[title_end:]
    h1_start = text.find("<h1>")
    h1_end = text.find("</h1>", h1_start)
    if h1_start >= 0 and h1_end >= 0:
        text = text[: h1_start + len("<h1>")] + esc(NEW_TITLE) + text[h1_end:]

    insert, payload = build_insert()
    marker = "<section>\n    <h2>Stage C Time Audit</h2>"
    if marker not in text:
        raise ValueError("Could not find Stage C insertion marker in the generated report")
    text = text.replace(marker, insert + "\n" + marker, 1)
    text = text.replace(
        "Shared Stage B and Stage D figures are scheme-independent inputs copied into this report's asset directory.",
        "Stage B and Stage D figures are regenerated from the double-Rayleigh branch and copied into this report's asset directory.",
    )

    tmp = TARGET.with_suffix(TARGET.suffix + ".double_rayleigh_tmp")
    tmp.write_text(text, encoding="utf-8")
    if "Double-Rayleigh PSF Contract" not in text or "Old Rayleigh vs New Double-Rayleigh Scheme B" not in text:
        raise ValueError("Prepared report is missing required double-Rayleigh sections")
    tmp.replace(TARGET)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = str(TARGET)
    payload["report_sha256"] = sha256(TARGET)
    payload["previous_report_sha256"] = before
    payload["input_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    comparison_json = ASSET_DIR / "scheme_B_double_rayleigh_comparison.json"
    comparison_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Previous B report sha256: {before or 'none'}")
    print(f"Double-Rayleigh B report: {TARGET}")
    print(f"Comparison payload: {comparison_json}")


if __name__ == "__main__":
    main()
