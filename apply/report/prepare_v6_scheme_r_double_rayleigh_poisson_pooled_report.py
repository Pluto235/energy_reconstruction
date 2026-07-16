#!/usr/bin/env python3
"""Prepare the independent Scheme R double-Rayleigh Poisson evidence report."""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = (
    "v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_"
    "scheme_R_fixed712979_poisson_pooled"
)
REPORT = Path(
    os.environ.get(
        "V6_REPORT_PATH",
        REPO_ROOT
        / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_"
        "double_rayleigh_poisson_pooled_report.html",
    )
)
ASSET_DIR = Path(
    os.environ.get(
        "V6_REPORT_ASSET_DIR",
        REPO_ROOT
        / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-"
        "double-rayleigh-scheme-R-poisson-pooled",
    )
)
COMPARISON_JSON = ASSET_DIR / "scheme_R_double_rayleigh_poisson_pooled_comparison.json"
ANALYTIC_COMPARISON = Path(
    os.environ.get(
        "V6_REPORT_ANALYTIC_COMPARISON",
        REPO_ROOT
        / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-"
        "double-rayleigh-scheme-R-analytic-bon/"
        "scheme_R_double_rayleigh_analytic_bon_comparison.json",
    )
)


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def require_env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for the Poisson report")
    path = Path(value).resolve()
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{name} is missing or empty: {path}")
    return path


def stage_f_metadata_path() -> Path:
    explicit = os.environ.get("V6_REPORT_STAGE_F_METADATA", "").strip()
    if explicit:
        return Path(explicit).resolve()
    run_dir = os.environ.get("V6_REPORT_STAGE_F_RUN_DIR", "").strip()
    stem = os.environ.get("V6_REPORT_STAGE_F_STEM", "").strip()
    if not run_dir or not stem:
        raise ValueError(
            "V6_REPORT_STAGE_F_RUN_DIR and V6_REPORT_STAGE_F_STEM are required "
            "unless V6_REPORT_STAGE_F_METADATA is set"
        )
    return (Path(run_dir) / f"{stem}_metadata.json").resolve()


def fit_record(label: str, contract: str, fit: Mapping[str, Any]) -> dict[str, Any]:
    parameters = fit.get("parameters") if isinstance(fit.get("parameters"), Mapping) else {}
    fit_parameters = fit.get("fit_parameters") if isinstance(fit.get("fit_parameters"), Mapping) else {}
    phi0 = finite(parameters.get("phi0"))
    if phi0 is None:
        log10_phi0 = finite(fit_parameters.get("log10_phi0"))
        phi0 = 10.0**log10_phi0 if log10_phi0 is not None else None
    chi2 = finite(fit.get("chi2"))
    ndof = finite(fit.get("ndof"))
    return {
        "label": label,
        "contract": contract,
        "valid": fit.get("valid"),
        "phi0": phi0,
        "alpha": finite(parameters.get("alpha", fit_parameters.get("alpha"))),
        "beta": finite(parameters.get("beta", fit_parameters.get("beta"))),
        "chi2": chi2,
        "ndof": int(ndof) if ndof is not None else None,
        "chi2_over_ndof": chi2 / ndof if chi2 is not None and ndof not in (None, 0.0) else None,
        "p_value": finite(fit.get("p_value")),
        "error_mode": fit.get("error_mode"),
    }


def baseline_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("comparison") or []
    if not isinstance(records, list):
        raise ValueError("Analytic comparison has no comparison list")
    legacy = next(
        (
            row for row in records
            if isinstance(row, Mapping)
            and str(row.get("label")) == "Scheme R 2R legacy"
            and row.get("background_integration") == "pixel-center"
        ),
        None,
    )
    analytic = next(
        (
            row for row in records
            if isinstance(row, Mapping)
            and str(row.get("label")) == "Scheme R 2R analytic B_on"
            and row.get("background_integration") == "analytic-quadratic"
        ),
        None,
    )
    if legacy is None or analytic is None:
        raise ValueError("Analytic comparison lacks the two required R-2R baselines")

    def convert(row: Mapping[str, Any], label: str, contract: str) -> dict[str, Any]:
        return {
            "label": label,
            "contract": contract,
            "valid": True,
            "phi0": finite(row.get("phi0")),
            "alpha": finite(row.get("alpha")),
            "beta": finite(row.get("beta")),
            "chi2": finite(row.get("chi2")),
            "ndof": row.get("ndof"),
            "chi2_over_ndof": finite(row.get("chi2_over_ndof")),
            "p_value": finite(row.get("p_value")),
            "error_mode": "legacy conservative",
            "source_branch_id": row.get("branch_id"),
            "fit_sha256": row.get("fit_sha256"),
            "response_sha256": row.get("response_sha256"),
        }

    return [
        convert(
            legacy,
            "Legacy old R-2R pixel-WLS",
            "weighted-LS annulus surface; pixel-center B_on; legacy conservative errors",
        ),
        convert(
            analytic,
            "Analytic-B_on R-2R WLS",
            "weighted-LS annulus surface; analytic circular B_on; legacy conservative errors",
        ),
    ]


def slurm_rows(value: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    items = value.items() if isinstance(value, Mapping) else enumerate(value) if isinstance(value, list) else []
    for role, raw in items:
        item = raw if isinstance(raw, Mapping) else {"state": raw}
        tasks = item.get("tasks")
        if isinstance(tasks, Mapping):
            for branch, task_raw in tasks.items():
                task = task_raw if isinstance(task_raw, Mapping) else {"state": task_raw}
                rows.append([esc(branch), esc(task.get("job_id") or item.get("job_id") or "n/a"), esc(task.get("state") or item.get("state") or "unknown")])
        else:
            rows.append([esc(item.get("role") or item.get("branch_id") or role), esc(item.get("job_id") or "n/a"), esc(item.get("state") or "unknown")])
    return rows


def cv_summary(row: Mapping[str, Any]) -> str:
    evidence = row.get("cross_validation") or row.get("cv_scores") or row.get("validation_scores")
    if not isinstance(evidence, Mapping):
        return "n/a"
    best = evidence.get("best_order")
    threshold = evidence.get("one_se_threshold")
    failures = evidence.get("failures") or {}
    failure_count = sum(len(value) for value in failures.values() if isinstance(value, list)) if isinstance(failures, Mapping) else 0
    return f"best={esc(best)}; 1-SE={fmt(threshold, 5)}; failed folds={failure_count}"


def diagnostic_figures(convergence: Mapping[str, Any], convergence_path: Path) -> str:
    filenames = convergence.get("plots") or []
    if not isinstance(filenames, list):
        return ""
    configured = os.environ.get("V6_REPORT_GRID_PLOT_DIR", "").strip()
    roots = [Path(configured)] if configured else [ASSET_DIR, convergence_path.parent]
    figures: list[str] = []
    for raw in filenames:
        name = str(raw)
        source = next((root / name for root in roots if (root / name).is_file()), None)
        if source is None or source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        relative = os.path.relpath(source.resolve(), start=REPORT.parent.resolve())
        caption = source.stem.replace("_", " ")
        figures.append(
            f'<figure class="figure"><img src="{esc(relative)}" alt="{esc(caption)}">'
            f"<figcaption>{esc(caption)}</figcaption></figure>"
        )
    return '<div class="figgrid">' + "".join(figures) + "</div>" if figures else ""


def nominal_positive_minima() -> list[dict[str, Any]]:
    explicit = os.environ.get("V6_REPORT_STAGE_D_NPZ", "").strip()
    if explicit:
        path = Path(explicit).resolve()
    else:
        run_dir = os.environ.get("V6_REPORT_STAGE_D_RUN_DIR", "").strip()
        stem = os.environ.get("V6_REPORT_STAGE_D_STEM", "").strip()
        if not run_dir or not stem:
            return []
        path = (Path(run_dir) / f"{stem}.npz").resolve()
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Nominal Stage D NPZ is missing or empty: {path}")
    with np.load(path, allow_pickle=False) as handle:
        cell_ids = np.asarray(handle["cell_id"], dtype=np.int64)
        minima = np.asarray(handle["positive_minimum"], dtype=np.float64)
        orders = np.asarray(handle["surface_order"], dtype=np.int64)
    if cell_ids.shape != minima.shape or cell_ids.shape != orders.shape:
        raise ValueError("Nominal Stage D positive-minimum arrays are not aligned")
    return [
        {"cell_id": int(cell_id), "positive_minimum": float(minimum), "surface_order": int(order)}
        for cell_id, minimum, order in zip(cell_ids, minima, orders)
    ]


def build_insert(
    manifest: Mapping[str, Any],
    convergence: Mapping[str, Any],
    covariance: Mapping[str, np.ndarray],
    covariance_metadata: Mapping[str, Any],
    fit_metadata: Mapping[str, Any],
    baselines: list[dict[str, Any]],
    convergence_path: Path,
    positive_minima: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    fits = fit_metadata.get("fits") or {}
    conservative = fits.get("logpar_conservative") if isinstance(fits, Mapping) else None
    covariance_aware = fits.get("logpar_background_covariance") if isinstance(fits, Mapping) else None
    if not isinstance(conservative, Mapping) or not isinstance(covariance_aware, Mapping):
        raise ValueError("Stage F metadata requires logpar_conservative and logpar_background_covariance")
    contracts = baselines + [
        fit_record(
            "Poisson pooled conservative",
            "positive profiled-Poisson pooled surface; analytic B_on; legacy conservative errors (primary)",
            conservative,
        ),
        fit_record(
            "Poisson pooled covariance-aware",
            "same nominal Poisson B_on; full bootstrap excess covariance (diagnostic only)",
            covariance_aware,
        ),
    ]
    contract_rows = [
        [
            esc(row["label"]), esc(row["contract"]), esc(row["valid"]), fmt(row["phi0"], 7),
            fmt(row["alpha"]), fmt(row["beta"]),
            f"{fmt(row['chi2'], 5)}/{esc(row['ndof'])}", fmt(row["chi2_over_ndof"], 5),
        ]
        for row in contracts
    ]

    branches = convergence.get("branches") or []
    branch_rows = []
    for row in branches:
        if not isinstance(row, Mapping):
            continue
        parameters = row.get("fit_parameters") or {}
        branch_rows.append([
            f"<code>{esc(row.get('branch_id'))}</code>", fmt(parameters.get("phi0"), 7),
            fmt(parameters.get("alpha")), fmt(parameters.get("beta")),
            fmt(row.get("mahalanobis_from_nominal"), 5), fmt(row.get("max_abs_delta_b_on_sigma"), 5),
            fmt(row.get("max_abs_delta_pull"), 5),
        ])

    cells = manifest.get("cells") or {}
    target_ids = [int(value) for value in manifest.get("target_cell_ids", [])]
    continuous = manifest.get("continuous_annulus_counts") or {}
    pooling_rows = []
    for cell_id in target_ids:
        row = cells.get(str(cell_id), {}) if isinstance(cells, Mapping) else {}
        donors = row.get("donor_cell_ids") or []
        pooling_rows.append([
            esc(cell_id), esc(continuous.get(str(cell_id), row.get("continuous_annulus_count"))),
            esc(row.get("mode")), esc(row.get("surface_order")),
            esc(", ".join(str(value) for value in donors)),
            esc(row.get("shape_contributor")), esc(cv_summary(row)),
        ])
    donor_count_rows = [
        [esc(cell_id), esc(continuous.get(str(cell_id)))]
        for cell_id in manifest.get("donor_universe_cell_ids", [])
    ]
    positive_minima = positive_minima or []
    minimum_rows = [
        [esc(row.get("cell_id")), esc(row.get("surface_order")), fmt(row.get("positive_minimum"), 8)]
        for row in positive_minima
    ]

    gate_rows = []
    failed_names = []
    for row in convergence.get("checks") or []:
        if not isinstance(row, Mapping):
            continue
        passed = row.get("passed") is True
        if not passed:
            failed_names.append(str(row.get("name")))
        gate_rows.append([
            esc(row.get("name")),
            f'<span class="status-{"pass" if passed else "fail"}">{"PASSED" if passed else "FAILED"}</span>',
            f"<code>{esc(json.dumps(row.get('observed'), sort_keys=True, separators=(',', ':')))}</code>",
            f"<code>{esc(json.dumps(row.get('limit'), sort_keys=True, separators=(',', ':')))}</code>",
        ])

    cell_envelopes = [row for row in convergence.get("cells") or [] if isinstance(row, Mapping)]
    envelope_rows = [
        [esc(row.get("cell_id")), fmt(row.get("B_on_nominal"), 7), fmt(row.get("B_on_envelope_sigma"), 5),
         fmt(row.get("phase_rms_envelope_sigma"), 5), fmt(row.get("pull_nominal"), 5), fmt(row.get("pull_envelope"), 5)]
        for row in cell_envelopes
    ]

    b_covariance = np.asarray(covariance["B_on_covariance"], dtype=np.float64)
    excess_covariance = np.asarray(covariance["excess_covariance"], dtype=np.float64)
    samples = np.asarray(covariance["B_on_bootstrap_samples"], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(excess_covariance)
    covariance_rows = [
        ["Replicates requested / completed", f"{esc(covariance_metadata.get('bootstrap_count_requested'))} / {esc(covariance_metadata.get('bootstrap_count_completed'))}"],
        ["Refit failures", esc(covariance_metadata.get("refit_failure_count"))],
        ["Seed", esc(covariance_metadata.get("seed"))],
        ["B_on covariance shape", esc(list(b_covariance.shape))],
        ["Bootstrap sample shape", esc(list(samples.shape))],
        ["Minimum excess-covariance eigenvalue", fmt(np.min(eigenvalues), 8)],
        ["Excess-covariance condition number", fmt(np.linalg.cond(excess_covariance), 8)],
        ["Stage D SHA", f"<code>{esc(covariance_metadata.get('stage_d_sha256'))}</code>"],
        ["Manifest SHA", f"<code>{esc(covariance_metadata.get('manifest_sha256'))}</code>"],
    ]

    provenance = manifest.get("provenance") or {}
    provenance_rows = [
        ["Manifest self hash", f"<code>{esc(manifest.get('manifest_sha256'))}</code>"],
        ["Analytic baseline commit", f"<code>{esc(manifest.get('analytic_bon_baseline_sha'))}</code>"],
        ["Implementation commit", f"<code>{esc(manifest.get('implementation_commit_sha'))}</code>"],
    ]
    if isinstance(provenance, Mapping):
        for name, raw in sorted(provenance.items()):
            item = raw if isinstance(raw, Mapping) else {}
            provenance_rows.append([esc(name), f"<code>{esc(item.get('sha256'))}</code>"])
    nominal = next((row for row in branches if isinstance(row, Mapping) and row.get("branch_id") == "h010_x0_y0"), {})
    for name, value in (nominal.get("provenance") or {}).items() if isinstance(nominal, Mapping) else []:
        provenance_rows.append([f"nominal {esc(name)}", f"<code>{esc(value)}</code>"])

    status_passed = convergence.get("passed") is True
    status = "PASSED" if status_passed else "FAILED"
    status_box = "okbox" if status_passed else "callout"
    slurm = slurm_rows(convergence.get("slurm_jobs"))
    figures = diagnostic_figures(convergence, convergence_path)

    insert = f"""<section id="poisson-four-contract-comparison">
    <h2>Four Background And Error Contracts</h2>
    <p>These rows are distinct scientific contracts. The old pixel-WLS and prerequisite analytic-B_on/WLS rows are read-only R-2R baselines. The Poisson conservative row is the primary new-versus-old comparison. The covariance-aware row is a bootstrap diagnostic and does not replace the conservative preferred fit. R-1R is historical context only and is intentionally absent from this primary table.</p>
    {table(["Contract", "Background / uncertainty", "Valid", "phi0", "alpha", "beta", "chi2/ndof", "chi2/ndof value"], contract_rows)}
  </section>

  <section id="poisson-pooling-model-table">
    <h2>Frozen Pooling, Model Tier, And CV Evidence</h2>
    <p>The donor universe is the 84 non-tail cells and the production targets are the frozen 44-cell selector. Pooling remains within Nhit. Continuous annulus counts drive the 20k/10k/100 rules; eight azimuthal folds and pooled leave-one-donor-out scores feed the one-standard-error selection.</p>
    {table(["Cell", "Continuous annulus N", "Tier", "Order", "Donors", "Shape contributor", "CV evidence"], pooling_rows)}
    <details><summary>All 84 non-tail donor continuous annulus counts</summary>
      {table(["Donor cell", "Continuous annulus N"], donor_count_rows)}
    </details>
    <h3>Exact positive minima</h3>
    <p>Every fitted intensity surface must have a strictly positive exact minimum over <code>rho &lt; 6 deg</code>; no clipping is permitted.</p>
    {table(["Cell", "Surface order", "Positive minimum"], minimum_rows) if minimum_rows else '<div class="callout">Numerical minima were not loaded; see the serialized all_stage_d_surfaces_positive gate below.</div>'}
  </section>

  <section id="poisson-12-branch-envelope">
    <h2>12 Branch Envelope</h2>
    <div class="{status_box}"><strong>Grid Convergence: {status}</strong>. Nominal production remains <code>h010_x0_y0</code> (0.1 deg, zero phase). Every failed and passed registered gate is retained below. Failure is evidence, not a reason to suppress this report or substitute another branch.{(' Failed gates: ' + esc(', '.join(failed_names))) if failed_names else ''}</div>
    {table(["Branch", "phi0", "alpha", "beta", "Mahalanobis", "max delta B/sigma", "max delta pull"], branch_rows)}
    <details><summary>Per-cell B_on, phase, and pull envelopes</summary>
      {table(["Cell", "Nominal B_on", "B_on envelope / sigma", "Phase RMS / sigma", "Nominal pull", "Pull envelope"], envelope_rows)}
    </details>
    {table(["Registered gate", "Status", "Observed", "Limit"], gate_rows)}
    {figures}
  </section>

  <section id="poisson-bootstrap-covariance-evidence">
    <h2>Bootstrap Covariance Evidence</h2>
    <p>The covariance-aware fit uses <code>diag(N_on) + Cov(B_on)</code> through a Cholesky solve. Its result is diagnostic; the legacy conservative-error Poisson fit remains primary.</p>
    {table(["Diagnostic", "Evidence"], covariance_rows)}
  </section>

  <section id="poisson-sha-slurm-provenance">
    <h2>Response, Selector, Manifest, And Slurm Provenance</h2>
    {table(["Artifact", "SHA"], provenance_rows)}
    {table(["Role / branch", "Job id", "State"], slurm) if slurm else '<div class="callout">No Slurm job registry was embedded in the convergence JSON.</div>'}
  </section>"""

    comparison = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "convergence_passed": status_passed,
        "failed_convergence_gates": failed_names,
        "contracts": contracts,
        "branches": branches,
        "cell_envelopes": cell_envelopes,
        "pooling_cells": {str(cell_id): cells.get(str(cell_id), {}) for cell_id in target_ids},
        "positive_minima": positive_minima,
        "provenance": {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "analytic_bon_baseline_sha": manifest.get("analytic_bon_baseline_sha"),
            "implementation_commit_sha": manifest.get("implementation_commit_sha"),
            "nominal_branch": nominal.get("provenance") if isinstance(nominal, Mapping) else None,
            "slurm_jobs": convergence.get("slurm_jobs"),
        },
        "bootstrap": {
            "count_requested": covariance_metadata.get("bootstrap_count_requested"),
            "count_completed": covariance_metadata.get("bootstrap_count_completed"),
            "refit_failure_count": covariance_metadata.get("refit_failure_count"),
            "seed": covariance_metadata.get("seed"),
            "minimum_excess_eigenvalue": float(np.min(eigenvalues)),
            "excess_condition_number": float(np.linalg.cond(excess_covariance)),
        },
    }
    return insert, comparison


def main() -> None:
    manifest_path = require_env_path("V6_REPORT_POISSON_MANIFEST")
    convergence_path = require_env_path("V6_REPORT_GRID_CONVERGENCE")
    covariance_path = require_env_path("V6_REPORT_BACKGROUND_COVARIANCE")
    fit_metadata_path = stage_f_metadata_path()
    required = [REPORT, ANALYTIC_COMPARISON, fit_metadata_path]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")

    report_text = REPORT.read_text(encoding="utf-8")
    if "poisson-four-contract-comparison" in report_text:
        raise ValueError("Poisson pooled report is already prepared")
    if "poisson-pooling-provenance" not in report_text:
        raise ValueError("Base report lacks optional Poisson sections; set all three V6_REPORT inputs")

    manifest = load_json(manifest_path)
    convergence = load_json(convergence_path)
    fit_metadata = load_json(fit_metadata_path)
    analytic = load_json(ANALYTIC_COMPARISON)
    covariance_metadata_path = Path(
        os.environ.get("V6_REPORT_BACKGROUND_COVARIANCE_METADATA", str(covariance_path.with_suffix(".json")))
    )
    covariance_metadata = load_json(covariance_metadata_path) if covariance_metadata_path.exists() else {}
    with np.load(covariance_path, allow_pickle=False) as handle:
        covariance = {name: handle[name].copy() for name in handle.files}

    insert, comparison = build_insert(
        manifest,
        convergence,
        covariance,
        covariance_metadata,
        fit_metadata,
        baseline_records(analytic),
        convergence_path,
        nominal_positive_minima(),
    )
    marker = '<section id="poisson-pooling-provenance">'
    report_text = report_text.replace(marker, insert + "\n\n  " + marker, 1)
    temporary = REPORT.with_suffix(REPORT.suffix + ".poisson_pooled_tmp")
    temporary.write_text(report_text, encoding="utf-8")
    temporary.replace(REPORT)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    comparison.update(
        {
            "report_path": str(REPORT),
            "report_sha256": sha256(REPORT),
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": sha256(manifest_path),
            "convergence_path": str(convergence_path),
            "convergence_sha256": sha256(convergence_path),
            "background_covariance_path": str(covariance_path),
            "background_covariance_sha256": sha256(covariance_path),
            "stage_f_metadata_path": str(fit_metadata_path),
            "stage_f_metadata_sha256": sha256(fit_metadata_path),
        }
    )
    COMPARISON_JSON.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared Poisson pooled report: {REPORT}")
    print(f"Comparison evidence: {COMPARISON_JSON}")
    print(f"Grid convergence: {'passed' if convergence.get('passed') is True else 'failed (report preserved)'}")


if __name__ == "__main__":
    main()
