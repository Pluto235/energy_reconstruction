#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the integrated v3 Crab SED Stage A-G HTML report.")
    parser.add_argument("--output-html", type=str, default="apply/report/crab_sed_v3_stage_a_to_g_report.html")
    parser.add_argument("--roadmap-html", type=str, default="apply/report/roadmap_v3.html")
    parser.add_argument("--selection-html", type=str, default="apply/report/v3_cell_selection_diagnostics.html")
    parser.add_argument("--stage-a-dir", type=str, default="apply/output/stage_a_v3_candidate")
    parser.add_argument("--stage-b-dir", type=str, default="apply/output/stage_b_v3_candidate/current")
    parser.add_argument("--stage-c-dir", type=str, default="apply/output/stage_c_v3_candidate/current")
    parser.add_argument("--stage-d-dir", type=str, default="apply/output/stage_d_v3_candidate/current")
    parser.add_argument("--stage-e-dir", type=str, default="apply/output/stage_e_v3_candidate/current")
    parser.add_argument("--stage-f-dir", type=str, default="apply/output/stage_f_v3_baseline/current")
    parser.add_argument("--stage-g-dir", type=str, default="apply/output/stage_g_v3_baseline/current")
    parser.add_argument("--stage-f-metadata-name", type=str, default="fit_v3_baseline_metadata.json")
    parser.add_argument("--stage-g-metadata-name", type=str, default="sed_points_v3_baseline_metadata.json")
    parser.add_argument("--stage-f-report-html", type=str, default="apply/report/stage_f_v3_baseline_report.html")
    parser.add_argument("--stage-g-report-html", type=str, default="apply/report/stage_g_v3_baseline_report.html")
    parser.add_argument("--raw-ledger-csv", type=str, default="apply/config/cell_ledger_v3_candidate.csv")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline.csv")
    parser.add_argument("--systematics-selector-csv", type=str, default="apply/config/cell_selector_v3_systematics.csv")
    parser.add_argument(
        "--high-energy-selector-csv",
        type=str,
        default="apply/config/cell_selector_v3_high_energy_probes.csv",
    )
    parser.add_argument("--baseline-name", type=str, default="v3_baseline")
    parser.add_argument("--fit-cell-counts-skymap", type=str, default="apply/report/assets/crab-v3-baseline-fit-cell-skymaps/crab_v3_baseline_fit_counts_grid.png")
    parser.add_argument("--fit-cell-excess-skymap", type=str, default="apply/report/assets/crab-v3-baseline-fit-cell-skymaps/crab_v3_baseline_fit_excess_grid.png")
    parser.add_argument("--fit-cell-ra-profile", type=str, default="apply/report/assets/crab-v3-baseline-fit-cell-profiles/crab_v3_baseline_fit_ra_normalized_counts_profiles.png")
    parser.add_argument("--fit-cell-dec-profile", type=str, default="apply/report/assets/crab-v3-baseline-fit-cell-profiles/crab_v3_baseline_fit_dec_normalized_counts_profiles.png")
    parser.add_argument("--fit-cell-excess-ra-profile", type=str, default="apply/report/assets/crab-v3-baseline-fit-cell-profiles/crab_v3_baseline_fit_ra_normalized_excess_profiles.png")
    parser.add_argument("--fit-cell-excess-dec-profile", type=str, default="apply/report/assets/crab-v3-baseline-fit-cell-profiles/crab_v3_baseline_fit_dec_normalized_excess_profiles.png")
    parser.add_argument("--selection-matrix-png", type=str, default="apply/report/assets/v3-cell-selection/v3_cell_selection_matrix.png")
    parser.add_argument("--mc-overlay-png", type=str, default="apply/report/assets/v3-cell-selection/v3_mc_true_energy_overlay.png")
    parser.add_argument("--central-mask-png", type=str, default="apply/report/assets/v3-cell-selection/v3_central99_mask.png")
    parser.add_argument("--ridge-fraction-png", type=str, default="apply/report/assets/v3-cell-selection/v3_mc_occupancy_ridge_fraction.png")
    parser.add_argument("--background-systematics-csv", type=str, default="apply/report/assets/v3-background-systematics/v3_background_systematics_summary.csv")
    parser.add_argument("--background-systematics-json", type=str, default="apply/report/assets/v3-background-systematics/v3_background_systematics_summary.json")
    parser.add_argument("--background-sensitivity-png", type=str, default="apply/report/assets/v3-background-systematics/v3_background_method_sensitivity_summary.png")
    parser.add_argument("--before-after-dec-profile-png", type=str, default="apply/report/assets/v3-background-systematics/v3_background_before_after_dec_profile.png")
    parser.add_argument("--validation-json", type=str, default="apply/report/assets/v3-validation/v3_validation_summary.json")
    parser.add_argument("--selector-systematics-csv", type=str, default="apply/report/assets/v3-validation/v3_selector_systematics_summary.csv")
    parser.add_argument("--selector-fit-comparison-csv", type=str, default="apply/report/assets/v3-validation/v3_selector_fit_comparison.csv")
    parser.add_argument("--response-closure-csv", type=str, default="apply/report/assets/v3-validation/v3_response_closure_summary.csv")
    parser.add_argument("--mc-reference-closure-csv", type=str, default="apply/report/assets/v3-validation/v3_mc_reference_forward_fold_closure.csv")
    parser.add_argument("--offsource-fake-source-csv", type=str, default="apply/report/assets/v3-validation/v3_offsource_fake_source_summary.csv")
    parser.add_argument("--time-split-csv", type=str, default="apply/report/assets/v3-validation/v3_time_split_summary.csv")
    parser.add_argument("--selector-sensitivity-png", type=str, default="apply/report/assets/v3-validation/v3_selector_sensitivity_summary.png")
    parser.add_argument("--response-closure-png", type=str, default="apply/report/assets/v3-validation/v3_response_closure_summary.png")
    return parser.parse_args()


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def rel(path: str | Path, start: Path) -> str:
    p = abs_path(path)
    try:
        return os.path.relpath(p, start=start.resolve())
    except ValueError:
        return str(p)


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def fmt(value: object, digits: int = 5) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


def h(value: object) -> str:
    return html.escape(str(value))


def spectrum_label(model_name: object) -> str:
    value = str(model_name or "").lower()
    if value == "pl":
        return "PL"
    if value == "logpar":
        return "LogPar"
    return str(model_name or "n/a")


def selector_ids(rows: Sequence[Dict[str, str]], include: bool) -> List[int]:
    ids: List[int] = []
    for row in rows:
        raw = str(row.get("include", "")).strip().lower()
        row_include = raw in {"1", "true", "yes", "y", "include"}
        if row_include == include:
            ids.append(int(row["cell_id"]))
    return ids


def stage_f_cell_ids(meta: Dict[str, object]) -> List[int]:
    validation = meta.get("validation") if isinstance(meta.get("validation"), dict) else {}
    subset = validation.get("cell_subset") if isinstance(validation.get("cell_subset"), dict) else {}
    ids = subset.get("included_cell_ids")
    if isinstance(ids, list):
        return [int(value) for value in ids]
    cells = meta.get("cells") if isinstance(meta.get("cells"), list) else []
    out: List[int] = []
    for cell in cells:
        if isinstance(cell, dict) and "cell_id" in cell:
            out.append(int(cell["cell_id"]))
    return out


def stage_g_required_cell_ids(meta: Dict[str, object]) -> List[int]:
    validation = meta.get("validation") if isinstance(meta.get("validation"), dict) else {}
    ids = validation.get("required_cell_ids")
    if isinstance(ids, list):
        return [int(value) for value in ids]
    return []


def table_from_rows(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return "<p>n/a</p>"
    head = "".join(f"<th>{h(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{h(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def figure(path: Path, caption: str, *, wide: bool = False, explanation: str = "") -> str:
    if not path.exists():
        return ""
    cls = ' class="wide"' if wide else ""
    explanation_html = f'<span class="figure-note">{h(explanation)}</span>' if explanation else ""
    return (
        f'<figure{cls}><img src="{h(rel(path, REPORT_DIR))}" alt="{h(caption)}">'
        f'<figcaption><strong>{h(caption)}</strong>{explanation_html}</figcaption></figure>'
    )


def stage_card(stage: str, title: str, meta: Dict[str, object], artifact: Path, notes: Sequence[str]) -> str:
    outputs = meta.get("outputs") if isinstance(meta.get("outputs"), dict) else {}
    promotion = meta.get("promotion") if isinstance(meta.get("promotion"), dict) else {}
    quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
    quality_gate = meta.get("quality_gate") if isinstance(meta.get("quality_gate"), dict) else {}
    status = (
        promotion.get("status")
        or quality.get("status")
        or quality_gate.get("status")
        or meta.get("absolute_effective_area_status")
        or ("missing" if not artifact.exists() else "available")
    )
    links = []
    if artifact.exists():
        links.append(f'<a href="{h(rel(artifact, REPORT_DIR))}">metadata</a>')
    for key in ["summary_md", "summary_csv", "report_html", "npz"]:
        value = outputs.get(key) if isinstance(outputs, dict) else None
        if value and abs_path(str(value)).exists():
            links.append(f'<a href="{h(rel(str(value), REPORT_DIR))}">{h(key)}</a>')
    note_html = "".join(f"<li>{h(note)}</li>" for note in notes)
    return f"""
    <article class="stage-card">
      <div class="stage-label">{h(stage)}</div>
      <h3>{h(title)}</h3>
      <dl>
        <dt>Run</dt><dd><code>{h(meta.get('run_id') or meta.get('slurm_job_id') or 'n/a')}</code></dd>
        <dt>Status</dt><dd><code>{h(status)}</code></dd>
        <dt>Artifacts</dt><dd>{' · '.join(links) if links else 'n/a'}</dd>
      </dl>
      <ul>{note_html}</ul>
    </article>
    """


def main() -> None:
    global REPORT_DIR
    args = parse_args()
    output_html = abs_path(args.output_html)
    REPORT_DIR = output_html.parent

    stage_a_dir = abs_path(args.stage_a_dir)
    stage_b_dir = abs_path(args.stage_b_dir)
    stage_c_dir = abs_path(args.stage_c_dir)
    stage_d_dir = abs_path(args.stage_d_dir)
    stage_e_dir = abs_path(args.stage_e_dir)
    stage_f_dir = abs_path(args.stage_f_dir)
    stage_g_dir = abs_path(args.stage_g_dir)

    raw_rows = read_csv_rows(abs_path(args.raw_ledger_csv))
    selector_rows = read_csv_rows(abs_path(args.baseline_selector_csv))
    systematics_rows = read_csv_rows(abs_path(args.systematics_selector_csv))
    high_energy_rows = read_csv_rows(abs_path(args.high_energy_selector_csv))
    included_ids = selector_ids(selector_rows, True)
    excluded_ids = selector_ids(selector_rows, False)
    systematics_ids = selector_ids(systematics_rows, True)
    high_energy_ids = selector_ids(high_energy_rows, True)
    role_counts: Dict[str, int] = {}
    for row in raw_rows:
        role = row.get("cell_role", "")
        role_counts[role] = role_counts.get(role, 0) + 1

    stage_a_meta_path = stage_a_dir / "response_2d_v3_candidate_metadata.json"
    stage_b_meta_path = stage_b_dir / "psf_v3_candidate_metadata.json"
    stage_c_meta_path = stage_c_dir / "obs_events_metadata.json"
    stage_d_meta_path = stage_d_dir / "background_v3_candidate_metadata.json"
    stage_e_meta_path = stage_e_dir / "signal_v3_candidate_metadata.json"
    stage_f_meta_path = stage_f_dir / args.stage_f_metadata_name
    stage_g_meta_path = stage_g_dir / args.stage_g_metadata_name

    stage_a = load_json(stage_a_meta_path)
    stage_b = load_json(stage_b_meta_path)
    stage_c = load_json(stage_c_meta_path)
    stage_d = load_json(stage_d_meta_path)
    stage_e = load_json(stage_e_meta_path)
    stage_f = load_json(stage_f_meta_path)
    stage_g = load_json(stage_g_meta_path)
    background_systematics = load_json(abs_path(args.background_systematics_json))
    background_systematics_rows = read_csv_rows(abs_path(args.background_systematics_csv))
    validation_summary = load_json(abs_path(args.validation_json))
    selector_systematics_rows = read_csv_rows(abs_path(args.selector_systematics_csv))
    selector_fit_rows = read_csv_rows(abs_path(args.selector_fit_comparison_csv))
    response_closure_rows = read_csv_rows(abs_path(args.response_closure_csv))
    mc_reference_closure_rows = read_csv_rows(abs_path(args.mc_reference_closure_csv))
    offsource_rows = read_csv_rows(abs_path(args.offsource_fake_source_csv))
    time_split_rows = read_csv_rows(abs_path(args.time_split_csv))

    totals_e = stage_e.get("totals") if isinstance(stage_e.get("totals"), dict) else {}
    contract_e = stage_e.get("stage_d_contract") if isinstance(stage_e.get("stage_d_contract"), dict) else {}
    quality_e = stage_e.get("quality_gate") if isinstance(stage_e.get("quality_gate"), dict) else {}
    preferred = stage_f.get("preferred_fit") if isinstance(stage_f.get("preferred_fit"), dict) else {}
    fits = stage_f.get("fits") if isinstance(stage_f.get("fits"), dict) else {}
    preferred_model = str(preferred.get("model", "pl") if isinstance(preferred, dict) else "pl").lower()
    preferred_error = str(preferred.get("error_mode", "conservative") if isinstance(preferred, dict) else "conservative").lower()
    preferred_key = f"{preferred_model}_{preferred_error}"
    fit_preferred = fits.get(preferred_key, {}) if isinstance(fits, dict) else {}
    fit_params = fit_preferred.get("parameters") if isinstance(fit_preferred.get("parameters"), dict) else {}
    stage_f_ids = stage_f_cell_ids(stage_f)
    stage_g_ids = stage_g_required_cell_ids(stage_g)
    selector_matches_stage_f = bool(included_ids and stage_f_ids and included_ids == stage_f_ids)
    selector_matches_stage_g = bool(included_ids and stage_g_ids and included_ids == stage_g_ids)
    selector_result_status = (
        "selector/result matched"
        if selector_matches_stage_f and selector_matches_stage_g
        else "selector frozen; fit/SED pending rerun"
    )
    selector_pending_ids = sorted(set(included_ids) - set(stage_f_ids))
    stale_result_ids = sorted(set(stage_f_ids) - set(included_ids))
    psf_followup_ids = [
        int(row["cell_id"])
        for row in selector_rows
        if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}
        and str(row.get("psf_quality_flag", "1")).strip().lower() not in {"1", "true", "yes", "y"}
    ]
    sed_points = stage_g.get("points", []) if isinstance(stage_g.get("points"), list) else []
    high_energy_ref = (
        background_systematics.get("high_energy_stage_g_reference")
        if isinstance(background_systematics.get("high_energy_stage_g_reference"), dict)
        else {}
    )
    high_energy_values = []
    if isinstance(high_energy_ref, dict) and isinstance(high_energy_ref.get("high_energy_effective_energy_tev"), dict):
        for value in high_energy_ref["high_energy_effective_energy_tev"].values():  # type: ignore[index]
            number = finite_float(value)
            if number is not None:
                high_energy_values.append(number)
    highest_high_energy_tev = max(high_energy_values) if high_energy_values else None
    high_energy_labels = (
        high_energy_ref.get("high_energy_labels", [])
        if isinstance(high_energy_ref, dict) and isinstance(high_energy_ref.get("high_energy_labels"), list)
        else []
    )

    role_table = [{"role": key, "cells": value} for key, value in sorted(role_counts.items())]
    systematics_table_rows = []
    for row in background_systematics_rows:
        systematics_table_rows.append(
            {
                "variant": row.get("variant", ""),
                "annulus": row.get("annulus", ""),
                "order": row.get("surface_order", ""),
                "fit family": row.get("fit_family", ""),
                "B_on": fmt(row.get("baseline_B_on"), 6),
                "excess": fmt(row.get("baseline_excess"), 6),
                "sigma": fmt(row.get("baseline_formal_sigma"), 5),
                "valid cells": row.get("valid_baseline_background_cells", ""),
                "LogPar phi0": fmt(row.get("logpar_phi0"), 6),
                "alpha": fmt(row.get("logpar_alpha"), 5),
                "beta": fmt(row.get("logpar_beta"), 5),
                "chi2": fmt(row.get("logpar_chi2"), 5),
            }
        )
    selector_table_rows = []
    for row in selector_systematics_rows:
        selector_table_rows.append(
            {
                "selector": row.get("selector", ""),
                "cells": row.get("included_cells", ""),
                "low Nhit": row.get("low_nhit_125_200_cells", ""),
                "HE overlap": row.get("high_energy_probe_overlap", ""),
                "added": row.get("added_vs_baseline", ""),
                "removed": row.get("removed_vs_baseline", ""),
                "added ids": row.get("added_ids", ""),
                "removed ids": row.get("removed_ids", ""),
            }
        )
    selector_fit_table_rows = []
    for row in selector_fit_rows:
        selector_fit_table_rows.append(
            {
                "fit": row.get("fit_label", ""),
                "status": row.get("status", ""),
                "cells": row.get("n_cells", ""),
                "model": spectrum_label(row.get("preferred_model", "")),
                "phi0": fmt(row.get("phi0"), 6),
                "gamma": fmt(row.get("gamma"), 5),
                "alpha": fmt(row.get("alpha"), 5),
                "beta": fmt(row.get("beta"), 5),
                "chi2/ndof": f"{fmt(row.get('chi2'), 5)} / {h(row.get('ndof', ''))}",
                "SED pts": row.get("stage_g_points", ""),
                "HE pts": row.get("stage_g_high_energy_predE_points", ""),
                "max Eeff TeV": fmt(row.get("stage_g_max_effective_energy_tev"), 5),
            }
        )
    closure_table_rows = []
    for row in response_closure_rows:
        closure_table_rows.append(
            {
                "selector": row.get("selector", ""),
                "cells": row.get("included_cells", ""),
                "rel count": fmt(row.get("rel_delta_count"), 4),
                "max cell count": fmt(row.get("max_abs_rel_delta_count_per_cell"), 4),
                "rel sumw": fmt(row.get("rel_delta_sumw"), 4),
                "max cell sumw": fmt(row.get("max_abs_rel_delta_sumw_per_cell"), 4),
                "max sum eta": fmt(row.get("max_sum_eta_over_true_bins"), 5),
            }
        )
    mc_closure_table_rows = []
    for row in mc_reference_closure_rows:
        mc_closure_table_rows.append(
            {
                "selector": row.get("selector", ""),
                "cells": row.get("included_cells", ""),
                "rel count": fmt(row.get("rel_delta_count"), 4),
                "rel sumw": fmt(row.get("rel_delta_sumw"), 4),
                "truth count": fmt(row.get("truth_numerator_count"), 6),
                "pred count": fmt(row.get("reconstructed_from_eta_count"), 6),
            }
        )
    signal_validation_table_rows = []
    for row in offsource_rows:
        signal_validation_table_rows.append(
            {
                "validation": row.get("validation", ""),
                "run": row.get("run", ""),
                "RA": fmt(row.get("ra_deg"), 5),
                "MJD min": fmt(row.get("mjd_min"), 6),
                "MJD max": fmt(row.get("mjd_max"), 6),
                "baseline N": fmt_int(row.get("baseline_N_on")),
                "baseline B": fmt(row.get("baseline_B_on"), 6),
                "baseline excess": fmt(row.get("baseline_excess"), 6),
                "baseline sigma": fmt(row.get("baseline_combined_sigma"), 5),
            }
        )
    time_split_table_rows = []
    for row in time_split_rows:
        time_split_table_rows.append(
            {
                "validation": row.get("validation", ""),
                "run": row.get("run", ""),
                "MJD min": fmt(row.get("mjd_min"), 6),
                "MJD max": fmt(row.get("mjd_max"), 6),
                "baseline N": fmt_int(row.get("baseline_N_on")),
                "baseline B": fmt(row.get("baseline_B_on"), 6),
                "baseline excess": fmt(row.get("baseline_excess"), 6),
                "baseline sigma": fmt(row.get("baseline_combined_sigma"), 5),
                "all sigma": fmt(row.get("all_formal_sigma"), 5),
            }
        )
    validation_status_rows = []
    status_items = validation_summary.get("status_items", []) if isinstance(validation_summary.get("status_items"), list) else []
    for item in status_items:
        if not isinstance(item, dict):
            continue
        validation_status_rows.append(
            {
                "item": item.get("item", ""),
                "status": item.get("status", ""),
                "evidence": item.get("evidence", ""),
            }
        )
    sed_table_rows = []
    for point in sed_points:
        if not isinstance(point, dict):
            continue
        sed_table_rows.append(
            {
                "grouping": point.get("grouping", ""),
                "group": point.get("group_label", ""),
                "cells": ",".join(str(v) for v in point.get("cell_ids", [])) if isinstance(point.get("cell_ids"), list) else "",
                "E_eff TeV": fmt(point.get("effective_energy_tev"), 5),
                "E2 dN/dE": fmt(point.get("E2_dnde"), 5),
                "err": fmt(point.get("E2_dnde_err"), 4),
                "TS/sigma": fmt(point.get("known_b_sigma_total", point.get("pull_vs_stage_f_model")), 4),
                "ratio StageF": fmt(point.get("ratio_to_stage_f_model", point.get("ratio_to_stage_f_pl")), 4),
            }
        )

    stage_b_warnings = stage_b.get("warning_rows", stage_b.get("warnings", []))
    surface = stage_d.get("background_model") if isinstance(stage_d.get("background_model"), dict) else {}
    stage_d_quality = stage_d.get("quality") if isinstance(stage_d.get("quality"), dict) else {}
    stage_d_warning_text = stage_d_quality.get("warnings", []) if isinstance(stage_d_quality, dict) else []
    stage_d_warning_ids = set()
    for item in stage_d_warning_text if isinstance(stage_d_warning_text, list) else []:
        text = str(item)
        if text.startswith("cell "):
            try:
                stage_d_warning_ids.add(int(text.split(":", 1)[0].split()[1]))
            except (IndexError, ValueError):
                pass
    stage_d_baseline_warning_ids = sorted(stage_d_warning_ids.intersection(included_ids))
    assignment_audit = stage_c.get("assignment_audit") if isinstance(stage_c.get("assignment_audit"), dict) else {}
    stage_cards = [
        stage_card(
            "A",
            "v3 candidate response",
            stage_a,
            stage_a_meta_path,
            [
                f"Cells: {fmt_int(len(raw_rows))}",
                f"Response type: {stage_a.get('response_type', 'n/a')}",
                f"S0: {fmt(stage_a.get('s0_m2'), 6)} m2",
            ],
        ),
        stage_card(
            "B",
            "v3 candidate PSF",
            stage_b,
            stage_b_meta_path,
            [
                f"Cells: {fmt_int(len(stage_b.get('cells', [])) if isinstance(stage_b.get('cells'), list) else len(raw_rows))}",
                f"Warning rows: {fmt_int(len(stage_b_warnings) if isinstance(stage_b_warnings, list) else 0)}",
                (
                    f"PSF follow-up baseline cells: {','.join(str(v) for v in psf_followup_ids) or 'none'}; "
                    "fit rerun is deferred until PSF is repaired."
                ),
            ],
        ),
        stage_card(
            "C",
            "Nhit >= 125 observation reduction",
            stage_c,
            stage_c_meta_path,
            [
                f"Output rows: {fmt_int(stage_c.get('processing', {}).get('selected_rows') if isinstance(stage_c.get('processing'), dict) else None)}",
                f"Nhit below candidate min: {fmt_int(assignment_audit.get('nhit_below_candidate_min_after_quality_cuts'))}",
                f"Out of ledger: {fmt_int(assignment_audit.get('out_of_ledger_after_finite'))}",
            ],
        ),
        stage_card(
            "D",
            "annulus 2D surface background",
            stage_d,
            stage_d_meta_path,
            [
                f"Method: {surface.get('method', 'n/a') if isinstance(surface, dict) else 'n/a'}",
                f"Surface order: {surface.get('surface_order', 'n/a') if isinstance(surface, dict) else 'n/a'}",
                f"Background form: {surface.get('background_form', 'n/a') if isinstance(surface, dict) else 'n/a'}",
                f"Candidate warnings: {fmt_int(len(stage_d_warning_ids))}; baseline warnings: {fmt_int(len(stage_d_baseline_warning_ids))}",
            ],
        ),
        stage_card(
            "E",
            "v3 candidate signal table",
            stage_e,
            stage_e_meta_path,
            [
                f"Quality: {quality_e.get('status', 'n/a') if isinstance(quality_e, dict) else 'n/a'}",
                f"Formal sigma: {fmt(totals_e.get('formal_sigma'), 5) if isinstance(totals_e, dict) else 'n/a'}",
                f"Excess: {fmt(totals_e.get('excess'), 5) if isinstance(totals_e, dict) else 'n/a'}",
            ],
        ),
        stage_card(
            "F",
            f"{args.baseline_name} forward folding",
            stage_f,
            stage_f_meta_path,
            [
                f"Included cells: {','.join(str(v) for v in included_ids) or 'n/a'}",
                f"Result status: {selector_result_status}",
                f"Preferred model: {spectrum_label(preferred_model)} / {preferred_error}",
                f"Preferred phi0: {fmt(fit_params.get('phi0'), 6) if isinstance(fit_params, dict) else 'n/a'}",
            ],
        ),
        stage_card(
            "G",
            f"{args.baseline_name} diagnostic SED",
            stage_g,
            stage_g_meta_path,
            [
                f"SED points: {fmt_int(len(sed_points))}",
                f"Result status: {selector_result_status}",
                "PredE grouping follows reconstructed-energy bins across contributing Nhit cells.",
            ],
        ),
    ]

    links = [
        ("Roadmap v3", abs_path(args.roadmap_html)),
        ("Cell-selection diagnostics", abs_path(args.selection_html)),
        ("Stage E report", abs_path("apply/report/stage_e_v3_candidate_report.html")),
        ("Stage F report", abs_path(args.stage_f_report_html)),
        ("Stage G report", abs_path(args.stage_g_report_html)),
    ]
    link_html = " · ".join(f'<a href="{h(rel(path, REPORT_DIR))}">{h(label)}</a>' for label, path in links if path.exists())

    figures = [
        figure(
            abs_path(args.selection_matrix_png),
            "v3 cell selection matrix",
            wide=True,
            explanation="Rows are Nhit bins and columns are predE bins. The colors separate baseline fit cells, systematics/probe cells, and cells excluded by prefit MC/response rules; this is the selector freeze audit, not a Crab-excess driven selection.",
        ),
        figure(
            abs_path(args.mc_overlay_png),
            "MC normalized true-energy distribution overlay",
            wide=True,
            explanation="Each colored curve is the normalized MC true-energy distribution for one v3 predE bin. Use it to see energy overlap, median shifts, and whether the mixed binning gives sensible true-energy coverage.",
        ),
        figure(
            abs_path(args.central_mask_png),
            "MC central-99% selection mask",
            wide=True,
            explanation="This shows which Nhit x predE cells fall inside the MC reconstructed-energy central-99% population. It is the first prefit selector cut and does not use Crab excess or fit residuals.",
        ),
        figure(
            abs_path(args.ridge_fraction_png),
            "MC occupancy ridge fraction",
            wide=True,
            explanation="Each cell is normalized by the maximum MC count in its Nhit row. The v3 baseline ridge is generated from this prefit occupancy fraction together with central99, MC-count, and high-energy-bin rules; highlighted baseline cells are not chosen from Crab on-source significance.",
        ),
        figure(
            abs_path(args.fit_cell_counts_skymap),
            f"{args.baseline_name} fit-cell Stage D counts skymap",
            wide=True,
            explanation="Observed Stage D counts maps for baseline fit cells. The white rho=6 deg circle marks the fiducial ROI and the center marker is Crab; this is the raw spatial distribution before background subtraction.",
        ),
        figure(
            abs_path(args.fit_cell_excess_skymap),
            f"{args.baseline_name} fit-cell Stage D excess skymap",
            wide=True,
            explanation="Counts minus the fitted local background for baseline fit cells. A compact positive structure near the center is Crab-like; broad gradients or edge structures point to background-model issues.",
        ),
        figure(
            abs_path(args.fit_cell_ra_profile),
            f"candidate-grid normalized RA-offset counts profiles ({args.baseline_name} fit cells highlighted)",
            wide=True,
            explanation="For every candidate cell, counts are summed in a |Dec offset|<1 deg band and divided by that cell's own peak. Empty panels mean the profile peak is zero or the cell has no usable events in the profile band; highlighted panels are baseline fit cells.",
        ),
        figure(
            abs_path(args.fit_cell_dec_profile),
            f"candidate-grid normalized Dec-offset counts profiles ({args.baseline_name} fit cells highlighted)",
            wide=True,
            explanation="For every candidate cell, counts are summed in a |RA offset|<1 deg band and normalized by the cell's peak. Compare this with the RA profile to check Dec-direction imbalance and PSF width changes across cells.",
        ),
        figure(
            abs_path(args.fit_cell_excess_ra_profile),
            f"candidate-grid normalized RA-offset excess profiles after background subtraction ({args.baseline_name} fit cells highlighted)",
            wide=True,
            explanation="Same RA profile diagnostic after subtracting the fitted background. A centered narrow excess is desirable; asymmetric tails indicate residual background or low-stat behavior.",
        ),
        figure(
            abs_path(args.fit_cell_excess_dec_profile),
            f"candidate-grid normalized Dec-offset excess profiles after background subtraction ({args.baseline_name} fit cells highlighted)",
            wide=True,
            explanation="Same Dec profile diagnostic after background subtraction. This is the direct check of whether the v3 annulus surface reduces Dec-direction background imbalance around Crab.",
        ),
        figure(
            stage_d_dir / "roi_counts_grid.png",
            "Stage D counts map grid with rho=6 deg circle",
            wide=True,
            explanation="Observed candidate-grid counts maps. The shared log color scale makes gross background levels comparable across cells, but low-stat cells can look blank because they contain few or no events.",
        ),
        figure(
            stage_d_dir / "annulus_training_mask_grid.png",
            "Annulus training mask grid",
            wide=True,
            explanation="Shows which pixels train the local background surface for each cell. The source/core region is excluded; shifted annuli appear at larger radius for broad-PSF or low-Nhit cells.",
        ),
        figure(
            stage_d_dir / "roi_background_grid.png",
            "Fitted 2D background surface grid",
            wide=True,
            explanation="This is the fitted background expectation, not data and not excess. Stage D fits a quadratic 2D surface on the annulus and extrapolates it across the ROI; smooth surfaces are good, while holes, crescent shapes, or sharp edge features usually flag low-stat or failed diagnostic cells.",
        ),
        figure(
            stage_d_dir / "annulus_residual_grid.png",
            "Annulus residual grid",
            wide=True,
            explanation="Residuals in the annulus training region after fitting the background surface. Values should be structureless around zero; coherent RA/Dec patterns mean the quadratic surface is not capturing the local background.",
        ),
        figure(
            stage_d_dir / "core_background_grid.png",
            "Core extrapolated background grid",
            wide=True,
            explanation="Background expectation restricted to the source/on aperture used for B_on. This view shows exactly what is integrated as background under the Crab region for each cell.",
        ),
        figure(
            abs_path(args.before_after_dec_profile_png),
            "Before/after Dec profile comparison for v3 baseline fit cells",
            wide=True,
            explanation="Compares Dec-direction behavior before and after applying the v3 background method. The goal is to reduce Dec-gradient residuals without erasing the central Crab excess.",
        ),
        figure(
            abs_path(args.background_sensitivity_png),
            "Background-method sensitivity summary",
            wide=True,
            explanation="Compares nominal and alternative background variants such as annulus placement and surface order. Large shifts in flux or sigma indicate background-model systematic risk.",
        ),
        figure(
            abs_path(args.selector_sensitivity_png),
            "Cell-selection sensitivity summary",
            wide=True,
            explanation="Shows how the fit changes under baseline versus expanded/systematic selectors. Stable spectral parameters mean the prefit selector is not driving the result artificially.",
        ),
        figure(
            abs_path(args.response_closure_png),
            "Stage A response histogram self-closure summary",
            wide=True,
            explanation="Checks whether Stage A response histograms fold back to their MC numerator truth. Large closure errors point to response binning, normalization, or bookkeeping problems.",
        ),
        figure(
            stage_d_dir / "roi_excess_grid.png",
            "Stage D excess map grid",
            wide=True,
            explanation="Candidate-grid counts minus fitted background. Use this after the counts, background, and residual grids: a credible Crab signal should be central and should not be accompanied by annulus residual structure.",
        ),
        figure(
            stage_f_dir / "model_counts_vs_excess.png",
            "Stage F model counts vs excess",
            explanation="Compares forward-folded model counts to Stage E excess per fit cell. Points far from the one-to-one trend identify cells that dominate chi-square or disagree with the fitted spectrum.",
        ),
        figure(
            stage_f_dir / ("pull_grid_logpar.png" if preferred_model == "logpar" else "pull_grid_pl.png"),
            "Stage F pull grid",
            explanation="Per-cell fit pull for the preferred spectral model. Random small pulls are expected; coherent regions in Nhit/predE space suggest response, background, or selector systematics.",
        ),
        figure(
            stage_g_dir / "sed_points_stage_f_fullarray_pool1.png",
            "Stage G SED points",
            explanation="Diagnostic SED points built by refitting normalization in reconstructed-energy groups with the Stage F spectral shape fixed. Compare high-energy points with external Crab references and upper limits.",
        ),
        figure(
            stage_g_dir / "sed_points_ratio.png",
            "Stage G SED ratios",
            explanation="Ratio of each diagnostic SED point to the Stage F reference model. A flat ratio near one means Stage G is consistent with the global fit; trends reveal curvature or bin-specific bias.",
        ),
        figure(
            stage_g_dir / "sed_point_cell_counts.png",
            "Stage G cell counts per point",
            explanation="Shows how many cells contribute to each Stage G SED point. High-energy points with few cells should be interpreted more conservatively.",
        ),
    ]

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crab SED v3 Stage A-G Report</title>
<style>
:root {{ --bg:#f6f7f8; --fg:#1f2933; --muted:#5d6b76; --panel:#ffffff; --panel2:#eef2f5; --border:#d8e0e6; --accent:#005f73; --warn:#b7791f; --code:#edf2f7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans CJK SC","Microsoft YaHei",sans-serif; line-height:1.62; }}
main {{ max-width:1240px; margin:0 auto; padding:42px 20px 72px; }}
header {{ border-bottom:1px solid var(--border); padding-bottom:24px; margin-bottom:30px; }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
h1 {{ margin:8px 0 12px; font-size:clamp(32px,5vw,52px); line-height:1.08; }}
h2 {{ margin:38px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--border); font-size:24px; }}
h3 {{ margin:4px 0 12px; font-size:20px; }}
p {{ margin:10px 0; }}
a {{ color:var(--accent); }}
code {{ background:var(--code); border-radius:4px; padding:2px 5px; font-size:13px; }}
.lead {{ max-width:980px; color:var(--muted); font-size:17px; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.metric {{ min-height:112px; padding:16px; border:1px solid var(--border); background:var(--panel); border-radius:8px; }}
.label {{ color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }}
.value {{ margin-top:8px; font-size:25px; font-weight:800; overflow-wrap:anywhere; }}
.note {{ margin-top:7px; color:var(--muted); font-size:13px; }}
.callout {{ margin:18px 0; padding:16px 18px; border:1px solid var(--border); border-left:4px solid var(--warn); background:var(--panel); border-radius:8px; }}
.stage-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.stage-card {{ position:relative; padding:18px 18px 16px 76px; border:1px solid var(--border); background:var(--panel); border-radius:8px; min-height:190px; }}
.stage-label {{ position:absolute; left:18px; top:18px; width:40px; height:40px; border-radius:50%; display:grid; place-items:center; background:var(--accent); color:#fff; font-weight:900; }}
dl {{ display:grid; grid-template-columns:86px 1fr; gap:4px 10px; margin:0 0 10px; }}
dt {{ color:var(--muted); }}
dd {{ margin:0; overflow-wrap:anywhere; }}
ul {{ margin:10px 0 0 18px; padding:0; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--border); background:var(--panel); border-radius:8px; margin:16px 0; }}
table {{ width:100%; min-width:880px; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
th {{ background:var(--panel2); white-space:nowrap; }}
.figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
figure {{ margin:0; padding:12px; border:1px solid var(--border); background:var(--panel); border-radius:8px; }}
.wide {{ grid-column:1 / -1; }}
img {{ display:block; width:100%; height:auto; background:#fff; border-radius:4px; }}
figcaption {{ margin-top:8px; color:var(--muted); font-size:13px; }}
figcaption strong {{ display:block; color:var(--fg); font-size:13px; margin-bottom:4px; }}
.figure-note {{ display:block; }}
footer {{ margin-top:48px; padding-top:18px; border-top:1px solid var(--border); color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
@media (max-width:900px) {{ .metric-grid,.stage-grid,.figure-grid {{ grid-template-columns:1fr; }} .stage-card {{ padding-left:18px; padding-top:68px; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">LHAASO-WCDA · Crab SED v3</div>
    <h1>Stage A-G 完整结果报告</h1>
    <p class="lead">本页汇总 v3 candidate grid、冻结的 HAWC-style prefit selector、Nhit>=125 观测规约、Crab-local annulus 二阶背景曲面、forward folding fit 和 diagnostic SED points。</p>
  </header>

  <section>
    <h2>Run Summary</h2>
    <div class="metric-grid">
      <div class="metric"><div class="label">candidate</div><div class="value">{fmt_int(len(raw_rows))}</div><div class="note">v3 candidate cells</div></div>
      <div class="metric"><div class="label">baseline</div><div class="value">{fmt_int(len(included_ids))}</div><div class="note">{h(args.baseline_name)} cells</div></div>
      <div class="metric"><div class="label">systematics</div><div class="value">{fmt_int(len(systematics_ids))}</div><div class="note">expanded prefit cells</div></div>
      <div class="metric"><div class="label">high-energy probes</div><div class="value">{fmt_int(len(high_energy_ids))}</div><div class="note">diagnostic / upper-limit candidates</div></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><div class="label">Stage E sigma</div><div class="value">{fmt(totals_e.get('formal_sigma'), 4) if isinstance(totals_e, dict) else 'n/a'}</div><div class="note">{h(contract_e.get('background_form', 'n/a') if isinstance(contract_e, dict) else 'n/a')}</div></div>
      <div class="metric"><div class="label">Stage F preferred</div><div class="value">{h(spectrum_label(preferred_model))}</div><div class="note">chi2/ndof {fmt(fit_preferred.get('chi2'), 4) if isinstance(fit_preferred, dict) else 'n/a'} / {h(fit_preferred.get('ndof', 'n/a') if isinstance(fit_preferred, dict) else 'n/a')}</div></div>
      <div class="metric"><div class="label">Nhit cut audit</div><div class="value">{fmt_int(assignment_audit.get('nhit_below_candidate_min_after_quality_cuts'))}</div><div class="note">excluded below candidate min</div></div>
      <div class="metric"><div class="label">Background</div><div class="value">{h(surface.get('method', 'n/a') if isinstance(surface, dict) else 'n/a')}</div><div class="note">annulus surface direct expectation</div></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><div class="label">Stage D candidate quality</div><div class="value">{h(stage_d_quality.get('status', 'n/a') if isinstance(stage_d_quality, dict) else 'n/a')}</div><div class="note">candidate-grid diagnostic status</div></div>
      <div class="metric"><div class="label">Stage D candidate warnings</div><div class="value">{fmt_int(len(stage_d_warning_ids))}</div><div class="note">mostly excluded low-stat / probe cells</div></div>
      <div class="metric"><div class="label">Stage D baseline warnings</div><div class="value">{fmt_int(len(stage_d_baseline_warning_ids))}</div><div class="note">{h(args.baseline_name)} fit cells affected</div></div>
      <div class="metric"><div class="label">Stage G points</div><div class="value">{fmt_int(len(sed_points))}</div><div class="note">diagnostic SED groups</div></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><div class="label">background systematics</div><div class="value">{fmt_int(len(background_systematics_rows))}</div><div class="note">annulus/order variants</div></div>
      <div class="metric"><div class="label">high-energy predE points</div><div class="value">{fmt_int(high_energy_ref.get('high_energy_points') if isinstance(high_energy_ref, dict) else None)}</div><div class="note">Stage G high-energy groups</div></div>
      <div class="metric"><div class="label">highest E_eff</div><div class="value">{fmt(highest_high_energy_tev, 4)}</div><div class="note">TeV, response-weighted</div></div>
      <div class="metric"><div class="label">systematics scope</div><div class="value">diagnostic</div><div class="note">derived from Stage D maps</div></div>
    </div>
    <div class="callout">
      Selector freeze audit: baseline/systematics selectors are read from CSV files and are not defined from Crab <code>N_on/B_on</code>, excess, significance, Stage F pulls, or Stage G residuals. Stage D uses <code>direct_expectation</code>; Li-Ma remains not applicable unless a future off-counts background is produced. Stage D candidate-grid quality may fail when excluded diagnostic/probe cells have fragile annulus fits; the frozen baseline warning count is reported separately. Background systematics compare default versus PSF-shifted annuli and first- versus second-order surfaces using the same Stage D counts maps.
    </div>
    <div class="callout">
      Current selector status: <strong>{h(selector_result_status)}</strong>. The frozen baseline selector now contains <code>{h(','.join(str(v) for v in included_ids) or 'n/a')}</code>. Existing Stage F/G artifacts contain <code>{h(','.join(str(v) for v in stage_f_ids) or 'n/a')}</code>; pending selector cells are <code>{h(','.join(str(v) for v in selector_pending_ids) or 'none')}</code>, stale result-only cells are <code>{h(','.join(str(v) for v in stale_result_ids) or 'none')}</code>. Treat Stage F/G plots and tables as the previous reference until the 30-cell fit is explicitly rerun.
    </div>
    <p>{link_html}</p>
  </section>

  <section>
    <h2>PSF Theta-Support Notes</h2>
    <div class="callout">
      <p><code>theta_missing_crab_probability_mass</code> measures the fraction of the Crab declination theta exposure for which a given cell has no MC support after applying that cell's Nhit/predE selection, true-energy range, finite-angle, and positive-weight requirements. It is not a measure of the total MC sample size; it is a coverage test on the conditional MC sample inside one cell.</p>
      <p>The important lesson for v3 is that many MC events globally do not guarantee full theta coverage inside every fine <code>Nhit x predE</code> cell. The ridge-left cells <code>39</code>, <code>52</code>, and <code>65</code> have visible Crab excess and satisfy the MC occupancy ridge rule, but their Stage B PSF falls back because their conditional MC theta support misses more than 10% of the Crab theta exposure. Current values are approximately <code>0.124</code>, <code>0.228</code>, and <code>0.208</code>, respectively.</p>
      <p>These cells are the left shoulder of the ridge: high Nhit but lower predicted energy than the row peak. That combination can be more sensitive to zenith angle, shower geometry, and NN response residuals, so its theta support can be less continuous than the neighboring right-shoulder cells. In contrast, adjacent cells <code>40</code>, <code>53</code>, and <code>66</code> pass with missing masses near <code>0.000</code>, <code>0.083</code>, and <code>0.062</code>.</p>
      <p>The selector decision is therefore: keep <code>39/52/65</code> in the frozen baseline cell list because they are physical-ridge cells, but do not trust the fallback PSF as a final fit input. Before the next Stage F/G rerun, repair or replace their PSF using a documented method such as neighboring-cell interpolation/borrowing and carry the choice as a PSF systematic.</p>
    </div>
  </section>

  <section>
    <h2>Ledger And Selector</h2>
    {table_from_rows(role_table, ['role', 'cells'])}
    <p>{h(args.baseline_name)} included cells: <code>{h(','.join(str(v) for v in included_ids) or 'n/a')}</code></p>
    <p>excluded / diagnostic cells: <code>{h(','.join(str(v) for v in excluded_ids) or 'n/a')}</code></p>
    <p>high-energy probe cells: <code>{h(','.join(str(v) for v in high_energy_ids) or 'n/a')}</code></p>
  </section>

  <section>
    <h2>Stages</h2>
    <div class="stage-grid">{''.join(stage_cards)}</div>
  </section>

  <section>
    <h2>Stage G SED Points</h2>
    {table_from_rows(sed_table_rows, ['grouping', 'group', 'cells', 'E_eff TeV', 'E2 dN/dE', 'err', 'TS/sigma', 'ratio StageF'])}
  </section>

  <section>
    <h2>Background Systematics</h2>
    {table_from_rows(systematics_table_rows, ['variant', 'annulus', 'order', 'fit family', 'B_on', 'excess', 'sigma', 'valid cells', 'LogPar phi0', 'alpha', 'beta', 'chi2'])}
    <p>High-energy predE groups: <code>{h(', '.join(str(v) for v in high_energy_labels) or 'n/a')}</code></p>
  </section>

  <section>
    <h2>Validation Status</h2>
    {table_from_rows(validation_status_rows, ['item', 'status', 'evidence'])}
    <h3>Cell-selection systematics</h3>
    {table_from_rows(selector_table_rows, ['selector', 'cells', 'low Nhit', 'HE overlap', 'added', 'removed', 'added ids', 'removed ids'])}
    <h3>Selector fit comparison</h3>
    {table_from_rows(selector_fit_table_rows, ['fit', 'status', 'cells', 'model', 'phi0', 'gamma', 'alpha', 'beta', 'chi2/ndof', 'SED pts', 'HE pts', 'max Eeff TeV'])}
    <h3>MC response closure</h3>
    {table_from_rows(closure_table_rows, ['selector', 'cells', 'rel count', 'max cell count', 'rel sumw', 'max cell sumw', 'max sum eta'])}
    <h3>MC reference forward-fold closure</h3>
    {table_from_rows(mc_closure_table_rows, ['selector', 'cells', 'rel count', 'rel sumw', 'truth count', 'pred count'])}
    <h3>Off-source fake-source controls</h3>
    {table_from_rows(signal_validation_table_rows, ['validation', 'run', 'RA', 'MJD min', 'MJD max', 'baseline N', 'baseline B', 'baseline excess', 'baseline sigma'])}
    <h3>Time-split background stability</h3>
    {table_from_rows(time_split_table_rows, ['validation', 'run', 'MJD min', 'MJD max', 'baseline N', 'baseline B', 'baseline excess', 'baseline sigma', 'all sigma'])}
    <p>The MC reference forward-fold closure uses the Stage A binned MC numerator as the truth definition and folds the denominator through the stored response; it is a production response closure, not an independent holdout sample. Off-source fake-source controls and time-split Stage D/E runs are listed explicitly; failed controls are kept in the report rather than folded back into the frozen selector.</p>
  </section>

  <section>
    <h2>Figures</h2>
    <div class="callout">
      图表读法：candidate-grid 图通常按行显示 Nhit bin、按列显示 predE bin；白色圆圈表示 <code>rho=6 deg</code> fiducial ROI，中心标记是 Crab。空白或破碎 panel 通常来自 excluded diagnostic/probe cells 的零统计或低统计，不代表 baseline fit cells 全部失败。先看 counts 和 training mask，再看 fitted background、annulus residual，最后看 excess 和 Stage F/G 结果。
    </div>
    <div class="figure-grid">{''.join(item for item in figures if item) or '<p>Figures are not available yet; rerun after Stage artifacts are produced.</p>'}</div>
  </section>

  <footer>Generated from Stage metadata under <code>{h(str(REPO_ROOT / 'apply/output'))}</code>.</footer>
</main>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(line.rstrip() for line in html_text.splitlines()) + "\n", encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
