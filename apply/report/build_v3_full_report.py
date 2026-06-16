#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(".")
ETO_APPLY_PREFIXES = (
    Path("/mnt/mydisk/server/projects/energy_reconstruction/apply"),
    Path("/home/server/projects/energy_reconstruction/apply"),
)


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
    parser.add_argument("--psfborrow-stage-b-dir", type=str, default="apply/output/stage_b_v3_candidate_psfborrow/current")
    parser.add_argument("--psfborrow-stage-d-dir", type=str, default="apply/output/stage_d_v3_candidate_psfborrow/current")
    parser.add_argument("--psfborrow-stage-e-dir", type=str, default="apply/output/stage_e_v3_candidate_psfborrow/current")
    parser.add_argument("--psfborrow-stage-f-dir", type=str, default="apply/output/stage_f_v3_baseline_psfborrow/current")
    parser.add_argument("--psfborrow-stage-g-dir", type=str, default="apply/output/stage_g_v3_baseline_psfborrow/current")
    parser.add_argument("--psfborrow-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline_psfborrow.csv")
    parser.add_argument("--psfborrow-stage-d-metadata-name", type=str, default="background_v3_candidate_psfborrow_metadata.json")
    parser.add_argument("--psfborrow-stage-e-metadata-name", type=str, default="signal_v3_candidate_psfborrow_metadata.json")
    parser.add_argument("--psfborrow-stage-f-metadata-name", type=str, default="fit_v3_baseline_psfborrow_metadata.json")
    parser.add_argument("--psfborrow-stage-g-metadata-name", type=str, default="sed_points_v3_baseline_psfborrow_metadata.json")
    parser.add_argument("--psfborrow-stage-f-report-html", type=str, default="apply/report/stage_f_v3_baseline_psfborrow_report.html")
    parser.add_argument("--psfborrow-stage-g-report-html", type=str, default="apply/report/stage_g_v3_baseline_psfborrow_report.html")
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
    parser.add_argument("--psfborrow-fit-cell-counts-skymap", type=str, default="apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_counts_grid.png")
    parser.add_argument("--psfborrow-fit-cell-excess-skymap", type=str, default="apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_excess_grid.png")
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
    parser.add_argument(
        "--official-pass5-sed-csv",
        type=str,
        default="apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv",
    )
    parser.add_argument(
        "--official-pass5-overlay-png",
        type=str,
        default="apply/report/assets/official-pass5/wcda_pass5_vs_v3_stage_g_sed_overlay.png",
    )
    parser.add_argument(
        "--stage-g-official-overlay-png",
        type=str,
        default="apply/report/assets/official-pass5/stage_g_sed_points_with_official_refs.png",
    )
    parser.add_argument(
        "--official-v099-sed-csv",
        type=str,
        default="apply/report/assets/official-v099/wcda_crab_sed_v099_20250731_20260616_123624.csv",
    )
    return parser.parse_args()


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    resolved = p if p.is_absolute() else (REPO_ROOT / p).resolve()
    if resolved.exists():
        return resolved
    for prefix in ETO_APPLY_PREFIXES:
        try:
            suffix = resolved.relative_to(prefix)
        except ValueError:
            continue
        local_candidate = REPO_ROOT / "apply" / suffix
        if local_candidate.exists():
            return local_candidate
    return resolved


def stage_dir(path: str | Path, metadata_name: str, preferred_run_ids: Sequence[str] = ()) -> Path:
    base = abs_path(path)
    if (base / metadata_name).exists():
        return base
    root = base.parent if base.name in {"current", "latest"} else base
    runs_root = root / "runs"
    if not runs_root.exists():
        return base
    for run_id in preferred_run_ids:
        candidate = runs_root / run_id
        if (candidate / metadata_name).exists():
            return candidate
    candidates = sorted(
        runs_root.glob(f"*/{metadata_name}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent
    return base


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


def sed_curve(
    energy_tev: np.ndarray,
    *,
    model_name: str,
    params: Dict[str, float],
    pivot_tev: float,
) -> np.ndarray:
    ratio = np.asarray(energy_tev, dtype=np.float64) / float(pivot_tev)
    if model_name == "pl":
        flux = float(params["phi0"]) * np.power(ratio, -float(params["gamma"]))
    elif model_name == "logpar":
        log_ratio = np.log(ratio)
        flux = float(params["phi0"]) * np.exp(
            (-float(params["alpha"]) - float(params["beta"]) * log_ratio) * log_ratio
        )
    else:
        raise ValueError(f"Unsupported spectrum model: {model_name}")
    return energy_tev * energy_tev * flux


def selector_ids(rows: Sequence[Dict[str, str]], include: bool) -> List[int]:
    ids: List[int] = []
    for row in rows:
        raw = str(row.get("include", "")).strip().lower()
        row_include = raw in {"1", "true", "yes", "y", "include"}
        if row_include == include:
            ids.append(int(row["cell_id"]))
    return ids


def selector_includes(row: Dict[str, str]) -> bool:
    return str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}


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


def preferred_fit_entry(meta: Dict[str, object]) -> Dict[str, object]:
    preferred = meta.get("preferred_fit") if isinstance(meta.get("preferred_fit"), dict) else {}
    fits = meta.get("fits") if isinstance(meta.get("fits"), dict) else {}
    model = str(preferred.get("model", "pl")).lower() if isinstance(preferred, dict) else "pl"
    error = str(preferred.get("error_mode", "conservative")).lower() if isinstance(preferred, dict) else "conservative"
    fit = fits.get(f"{model}_{error}", {}) if isinstance(fits, dict) else {}
    return fit if isinstance(fit, dict) else {}


def fit_parameters(meta: Dict[str, object]) -> Dict[str, object]:
    fit = preferred_fit_entry(meta)
    params = fit.get("parameters") if isinstance(fit.get("parameters"), dict) else {}
    return params if isinstance(params, dict) else {}


def fit_model(meta: Dict[str, object]) -> str:
    preferred = meta.get("preferred_fit") if isinstance(meta.get("preferred_fit"), dict) else {}
    return str(preferred.get("model", "n/a")) if isinstance(preferred, dict) else "n/a"


def fit_error_mode(meta: Dict[str, object]) -> str:
    preferred = meta.get("preferred_fit") if isinstance(meta.get("preferred_fit"), dict) else {}
    return str(preferred.get("error_mode", "n/a")) if isinstance(preferred, dict) else "n/a"


def metadata_run_label(meta: Dict[str, object]) -> str:
    return str(meta.get("run_id") or meta.get("slurm_job_id") or "n/a")


def compare_value(new_value: object, old_value: object, digits: int = 5) -> str:
    new_number = finite_float(new_value)
    old_number = finite_float(old_value)
    if new_number is None:
        return "n/a"
    if old_number is None:
        return fmt(new_number, digits)
    delta = new_number - old_number
    if old_number != 0:
        rel_delta = delta / old_number
        return f"{fmt(new_number, digits)} ({fmt(delta, digits)}, {fmt(rel_delta, 4)})"
    return f"{fmt(new_number, digits)} ({fmt(delta, digits)})"


def psf_borrow_records(meta: Dict[str, object]) -> List[Dict[str, object]]:
    borrowing = meta.get("psf_borrowing") if isinstance(meta.get("psf_borrowing"), dict) else {}
    records = borrowing.get("records", []) if isinstance(borrowing, dict) else []
    return [record for record in records if isinstance(record, dict)]


def make_psfborrow_fit_rows(
    nominal_f: Dict[str, object],
    nominal_g: Dict[str, object],
    psf_f: Dict[str, object],
    psf_g: Dict[str, object],
    nominal_ids: Sequence[int],
    psf_ids: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, fit_meta, sed_meta, ids, reference_fit in [
        ("nominal reference", nominal_f, nominal_g, nominal_ids, {}),
        ("PSF systematic", psf_f, psf_g, psf_ids, nominal_f),
    ]:
        fit = preferred_fit_entry(fit_meta)
        params = fit_parameters(fit_meta)
        ref_params = fit_parameters(reference_fit) if reference_fit else {}
        ref_fit = preferred_fit_entry(reference_fit) if reference_fit else {}
        rows.append(
            {
                "version": label,
                "run": metadata_run_label(fit_meta),
                "cells": len(ids),
                "added": ",".join(str(v) for v in sorted(set(ids) - set(nominal_ids))) if label != "nominal reference" else "",
                "removed": ",".join(str(v) for v in sorted(set(nominal_ids) - set(ids))) if label != "nominal reference" else "",
                "model": spectrum_label(fit_model(fit_meta)),
                "error": fit_error_mode(fit_meta),
                "phi0 (delta)": compare_value(params.get("phi0"), ref_params.get("phi0"), 6),
                "gamma/alpha (delta)": compare_value(
                    params.get("gamma", params.get("alpha")),
                    ref_params.get("gamma", ref_params.get("alpha")),
                    5,
                ),
                "beta (delta)": compare_value(params.get("beta"), ref_params.get("beta"), 5),
                "chi2/ndof": f"{fmt(fit.get('chi2'), 5)} / {h(fit.get('ndof', ''))}",
                "delta chi2": compare_value(fit.get("chi2"), ref_fit.get("chi2"), 5),
                "SED pts": len(sed_meta.get("points", [])) if isinstance(sed_meta.get("points"), list) else "",
            }
        )
    return rows


def make_sed_compare_rows(nominal_g: Dict[str, object], psf_g: Dict[str, object]) -> List[Dict[str, object]]:
    nominal_points = nominal_g.get("points", []) if isinstance(nominal_g.get("points"), list) else []
    psf_points = psf_g.get("points", []) if isinstance(psf_g.get("points"), list) else []
    nominal_by_key = {
        (str(point.get("grouping")), str(point.get("group_label"))): point
        for point in nominal_points
        if isinstance(point, dict)
    }
    rows: List[Dict[str, object]] = []
    for point in psf_points:
        if not isinstance(point, dict):
            continue
        key = (str(point.get("grouping")), str(point.get("group_label")))
        nominal = nominal_by_key.get(key, {})
        rows.append(
            {
                "grouping": key[0],
                "group": key[1],
                "cells": ",".join(str(v) for v in point.get("cell_ids", [])) if isinstance(point.get("cell_ids"), list) else "",
                "E_eff TeV": compare_value(point.get("effective_energy_tev"), nominal.get("effective_energy_tev"), 5),
                "E2 dN/dE": compare_value(point.get("E2_dnde"), nominal.get("E2_dnde"), 5),
                "err": fmt(point.get("E2_dnde_err"), 4),
                "ratio StageF": compare_value(
                    point.get("ratio_to_stage_f_model", point.get("ratio_to_stage_f_pl")),
                    nominal.get("ratio_to_stage_f_model", nominal.get("ratio_to_stage_f_pl")),
                    4,
                ),
            }
        )
    return rows


def official_pass5_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        energy = finite_float(row.get("energy_tev"))
        flux = finite_float(row.get("flux_per_tev_cm2_s"))
        e2_flux = energy * energy * flux if energy is not None and flux is not None else None
        out.append(
            {
                "E TeV": fmt(energy, 4),
                "dN/dE": fmt(flux, 4),
                "E2 dN/dE": fmt(e2_flux, 4),
                "TS": fmt(row.get("ts"), 5),
                "Nhit": row.get("nhit_bin", ""),
                "Error_status": row.get("error_status", ""),
                "upper limit": row.get("is_upper_limit", ""),
                "stderr empty": row.get("stderr_empty", ""),
            }
        )
    return out


def official_v099_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        energy = finite_float(row.get("energy_tev"))
        flux_scaled = finite_float(row.get("e2_flux_scaled_1e14_tev_cm2_s"))
        err_low_scaled = finite_float(row.get("e2_flux_err_low_scaled_1e14"))
        err_high_scaled = finite_float(row.get("e2_flux_err_high_scaled_1e14"))
        e2_flux = flux_scaled * 1.0e-14 if flux_scaled is not None else None
        err_low = err_low_scaled * 1.0e-14 if err_low_scaled is not None else None
        err_high = err_high_scaled * 1.0e-14 if err_high_scaled is not None else None
        out.append(
            {
                "E TeV": fmt(energy, 4),
                "E2 flux raw": fmt(flux_scaled, 6),
                "E2 dN/dE": fmt(e2_flux, 4),
                "err low": fmt(err_low, 4),
                "err high": fmt(err_high, 4),
                "TS": fmt(row.get("ts"), 5),
                "WCDAtag": row.get("wcda_tag", ""),
                "status": row.get("crab_status", ""),
            }
        )
    return out


def plot_sed_overlay(
    output_path: Path,
    nominal_g: Dict[str, object],
    psf_g: Dict[str, object],
) -> Optional[Path]:
    nominal_points = [p for p in nominal_g.get("points", []) if isinstance(p, dict)] if isinstance(nominal_g.get("points"), list) else []
    psf_points = [p for p in psf_g.get("points", []) if isinstance(p, dict)] if isinstance(psf_g.get("points"), list) else []
    if not nominal_points or not psf_points:
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def arrays(points: Sequence[Dict[str, object]], grouping: str):
        selected = [p for p in points if str(p.get("grouping")) == grouping]
        selected.sort(key=lambda p: finite_float(p.get("effective_energy_tev")) or 0.0)
        energy = [finite_float(p.get("effective_energy_tev")) for p in selected]
        flux = [finite_float(p.get("E2_dnde")) for p in selected]
        err = [finite_float(p.get("E2_dnde_err")) for p in selected]
        labels = [str(p.get("group_label")) for p in selected]
        valid = [i for i, (e, f) in enumerate(zip(energy, flux)) if e is not None and f is not None]
        return (
            np.asarray([energy[i] for i in valid], dtype=np.float64),
            np.asarray([flux[i] for i in valid], dtype=np.float64),
            np.asarray([err[i] if err[i] is not None else 0.0 for i in valid], dtype=np.float64),
            [labels[i] for i in valid],
        )

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    for grouping, marker in [("nhit", "o"), ("predE", "s")]:
        x, y, yerr, _ = arrays(nominal_points, grouping)
        if x.size:
            ax.errorbar(x, y, yerr=yerr, fmt=marker, markersize=4.5, capsize=2, label=f"nominal {grouping}", alpha=0.78)
        x, y, yerr, _ = arrays(psf_points, grouping)
        if x.size:
            ax.errorbar(x, y, yerr=yerr, fmt=marker, markersize=4.5, capsize=2, label=f"psfborrow {grouping}", alpha=0.78)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective energy (TeV)")
    ax.set_ylabel("E^2 dN/dE (TeV cm^-2 s^-1)")
    ax.set_title("Stage G SED points: nominal reference vs PSF borrowing systematic")
    ax.grid(alpha=0.25, which="both", linewidth=0.45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_official_sed_overlay(
    output_path: Path,
    nominal_g: Dict[str, object],
    psf_g: Dict[str, object],
    pass5_rows: Sequence[Dict[str, str]],
    v099_rows: Sequence[Dict[str, str]],
) -> Optional[Path]:
    nominal_points = [p for p in nominal_g.get("points", []) if isinstance(p, dict)] if isinstance(nominal_g.get("points"), list) else []
    psf_points = [p for p in psf_g.get("points", []) if isinstance(p, dict)] if isinstance(psf_g.get("points"), list) else []
    pass5_points = []
    for row in pass5_rows:
        energy = finite_float(row.get("energy_tev"))
        flux = finite_float(row.get("flux_per_tev_cm2_s"))
        if energy is None or flux is None:
            continue
        pass5_points.append((energy, energy * energy * flux))
    v099_points = []
    for row in v099_rows:
        energy = finite_float(row.get("energy_tev"))
        flux_scaled = finite_float(row.get("e2_flux_scaled_1e14_tev_cm2_s"))
        err_low_scaled = finite_float(row.get("e2_flux_err_low_scaled_1e14"))
        err_high_scaled = finite_float(row.get("e2_flux_err_high_scaled_1e14"))
        if energy is None or flux_scaled is None:
            continue
        v099_points.append(
            (
                energy,
                flux_scaled * 1.0e-14,
                (err_low_scaled or 0.0) * 1.0e-14,
                (err_high_scaled or 0.0) * 1.0e-14,
            )
        )
    if not pass5_points and not v099_points:
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def arrays(points: Sequence[Dict[str, object]], grouping: str):
        selected = [p for p in points if str(p.get("grouping")) == grouping]
        selected.sort(key=lambda p: finite_float(p.get("effective_energy_tev")) or 0.0)
        energy = [finite_float(p.get("effective_energy_tev")) for p in selected]
        flux = [finite_float(p.get("E2_dnde")) for p in selected]
        err = [finite_float(p.get("E2_dnde_err")) for p in selected]
        valid = [i for i, (e, f) in enumerate(zip(energy, flux)) if e is not None and f is not None]
        return (
            np.asarray([energy[i] for i in valid], dtype=np.float64),
            np.asarray([flux[i] for i in valid], dtype=np.float64),
            np.asarray([err[i] if err[i] is not None else 0.0 for i in valid], dtype=np.float64),
        )

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
    if pass5_points:
        x = np.asarray([p[0] for p in pass5_points], dtype=np.float64)
        y = np.asarray([p[1] for p in pass5_points], dtype=np.float64)
        ax.plot(x, y, "D-", color="#111827", markersize=5.0, linewidth=1.4, label="official pass5 Nhit SED")
    if v099_points:
        x = np.asarray([p[0] for p in v099_points], dtype=np.float64)
        y = np.asarray([p[1] for p in v099_points], dtype=np.float64)
        yerr = np.vstack(
            [
                np.asarray([p[2] for p in v099_points], dtype=np.float64),
                np.asarray([p[3] for p in v099_points], dtype=np.float64),
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="^-",
            color="#b45309",
            markersize=5.0,
            linewidth=1.25,
            capsize=2.5,
            label="tutorial v0.99 WCDA-only SED",
        )
    for grouping, marker, alpha in [("nhit", "o", 0.55), ("predE", "s", 0.42)]:
        sx, sy, syerr = arrays(nominal_points, grouping)
        if sx.size:
            ax.errorbar(
                sx,
                sy,
                yerr=syerr,
                fmt=marker,
                markersize=4.0,
                capsize=2,
                label=f"v3 nominal Stage G {grouping}",
                alpha=alpha,
            )
        sx, sy, syerr = arrays(psf_points, grouping)
        if sx.size:
            ax.errorbar(
                sx,
                sy,
                yerr=syerr,
                fmt=marker,
                markersize=4.0,
                capsize=2,
                label=f"v3 psfborrow Stage G {grouping}",
                alpha=alpha,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E^2 dN/dE (TeV cm^-2 s^-1)")
    ax.set_title("Official/tutorial WCDA Crab SEDs vs v3 Stage G diagnostics")
    ax.grid(alpha=0.25, which="both", linewidth=0.45)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_stage_g_with_official_refs(
    output_path: Path,
    stage_g: Dict[str, object],
    pass5_rows: Sequence[Dict[str, str]],
    v099_rows: Sequence[Dict[str, str]],
) -> Optional[Path]:
    points = [p for p in stage_g.get("points", []) if isinstance(p, dict)] if isinstance(stage_g.get("points"), list) else []
    if not points:
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    energies = [
        finite_float(point.get("effective_energy_tev"))
        for point in points
        if finite_float(point.get("effective_energy_tev")) is not None and finite_float(point.get("E2_dnde")) is not None
    ]
    if energies:
        emin = max(0.2, min(energies) / 1.8)
        emax = min(200.0, max(energies) * 1.8)
    else:
        emin, emax = 0.3, 80.0
    x = np.geomspace(emin, emax, 240)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=180, constrained_layout=True)

    frozen = stage_g.get("frozen_spectrum") if isinstance(stage_g.get("frozen_spectrum"), dict) else {}
    frozen_model = str(frozen.get("model", "")).lower()
    frozen_pivot = finite_float(frozen.get("pivot_tev")) or 3.0
    if frozen_model:
        frozen_params = {
            key: value
            for key, value in {
                "phi0": finite_float(frozen.get("phi0")),
                "gamma": finite_float(frozen.get("gamma")),
                "alpha": finite_float(frozen.get("alpha")),
                "beta": finite_float(frozen.get("beta")),
            }.items()
            if value is not None
        }
        try:
            ax.plot(
                x,
                sed_curve(x, model_name=frozen_model, params=frozen_params, pivot_tev=frozen_pivot),
                color="#1f77b4",
                lw=2.0,
                label=f"Stage F frozen {spectrum_label(frozen_model)}",
            )
        except Exception:
            pass

    reference = stage_g.get("reference_spectrum") if isinstance(stage_g.get("reference_spectrum"), dict) else {}
    ref_phi0 = finite_float(reference.get("phi0"))
    ref_gamma = finite_float(reference.get("gamma"))
    ref_pivot = finite_float(reference.get("pivot_tev")) or 3.0
    if ref_phi0 is not None and ref_gamma is not None:
        ax.plot(
            x,
            sed_curve(x, model_name="pl", params={"phi0": ref_phi0, "gamma": ref_gamma}, pivot_tev=ref_pivot),
            color="#555555",
            lw=1.8,
            ls="--",
            label="1LHAASO WCDA full-array PL",
        )

    pool1 = stage_g.get("wcda1_pool1_reference") if isinstance(stage_g.get("wcda1_pool1_reference"), dict) else {}
    pool1_points = [p for p in pool1.get("points", []) if isinstance(p, dict)] if isinstance(pool1.get("points"), list) else []
    pool1_x = [finite_float(p.get("emed_tev")) for p in pool1_points]
    pool1_y = [finite_float(p.get("E2_dnde")) for p in pool1_points]
    pool1_err = [finite_float(p.get("E2_dnde_err")) for p in pool1_points]
    valid = [i for i, (px, py) in enumerate(zip(pool1_x, pool1_y)) if px is not None and py is not None]
    if valid:
        ax.errorbar(
            [pool1_x[i] for i in valid],
            [pool1_y[i] for i in valid],
            yerr=[pool1_err[i] if pool1_err[i] is not None else 0.0 for i in valid],
            fmt="^",
            color="#7f3fbf",
            ecolor="#7f3fbf",
            capsize=3,
            ms=5,
            lw=0.9,
            label="WCDA-1 Pool-1 Table 1",
        )

    external = stage_g.get("external_crab_sed_references") if isinstance(stage_g.get("external_crab_sed_references"), dict) else {}
    external_points = [p for p in external.get("points", []) if isinstance(p, dict)] if isinstance(external.get("points"), list) else []
    external_styles = {
        "magic_joint_crab": {"fmt": "v", "color": "#9467bd", "label": "MAGIC"},
        "hess_2024_stereo": {"fmt": "D", "color": "#8c564b", "label": "H.E.S.S."},
        "hawc_2019_nn": {"fmt": "P", "color": "#17becf", "label": "HAWC NN"},
    }
    for dataset, style in external_styles.items():
        selected = [
            p
            for p in external_points
            if str(p.get("dataset")) == dataset
            and not str(p.get("is_upper_limit", "")).lower() == "true"
            and finite_float(p.get("energy_tev")) is not None
            and finite_float(p.get("e2_dnde")) is not None
        ]
        if not selected:
            continue
        ax.errorbar(
            [finite_float(p.get("energy_tev")) for p in selected],
            [finite_float(p.get("e2_dnde")) for p in selected],
            yerr=[finite_float(p.get("e2_dnde_err")) or 0.0 for p in selected],
            capsize=2,
            ms=4,
            lw=0.7,
            alpha=0.72,
            **style,
        )

    pass5_points = []
    for row in pass5_rows:
        energy = finite_float(row.get("energy_tev"))
        flux = finite_float(row.get("flux_per_tev_cm2_s"))
        if energy is not None and flux is not None:
            pass5_points.append((energy, energy * energy * flux))
    if pass5_points:
        ax.plot(
            [p[0] for p in pass5_points],
            [p[1] for p in pass5_points],
            "D-",
            color="#111827",
            markersize=4.6,
            linewidth=1.15,
            alpha=0.86,
            label="official pass5 Nhit SED",
        )

    v099_points = []
    for row in v099_rows:
        energy = finite_float(row.get("energy_tev"))
        flux_scaled = finite_float(row.get("e2_flux_scaled_1e14_tev_cm2_s"))
        err_low_scaled = finite_float(row.get("e2_flux_err_low_scaled_1e14"))
        err_high_scaled = finite_float(row.get("e2_flux_err_high_scaled_1e14"))
        if energy is None or flux_scaled is None:
            continue
        v099_points.append(
            (
                energy,
                flux_scaled * 1.0e-14,
                (err_low_scaled or 0.0) * 1.0e-14,
                (err_high_scaled or 0.0) * 1.0e-14,
            )
        )
    if v099_points:
        ax.errorbar(
            [p[0] for p in v099_points],
            [p[1] for p in v099_points],
            yerr=np.vstack(
                [
                    np.asarray([p[2] for p in v099_points], dtype=np.float64),
                    np.asarray([p[3] for p in v099_points], dtype=np.float64),
                ]
            ),
            fmt="^-",
            color="#b45309",
            markersize=4.6,
            linewidth=1.05,
            capsize=2.5,
            alpha=0.86,
            label="tutorial v0.99 WCDA-only SED",
        )

    styles = {
        "nhit": {"fmt": "o", "color": "#d62728", "label": "Nhit grouped"},
        "predE": {"fmt": "s", "color": "#2ca02c", "label": "predE grouped"},
    }
    for grouping, style in styles.items():
        selected = [
            p
            for p in points
            if str(p.get("grouping")) == grouping
            and finite_float(p.get("effective_energy_tev")) is not None
            and finite_float(p.get("E2_dnde")) is not None
        ]
        selected.sort(key=lambda p: finite_float(p.get("effective_energy_tev")) or 0.0)
        if not selected:
            continue
        ax.errorbar(
            [finite_float(p.get("effective_energy_tev")) for p in selected],
            [finite_float(p.get("E2_dnde")) for p in selected],
            yerr=[finite_float(p.get("E2_dnde_err")) or 0.0 for p in selected],
            capsize=3,
            ms=5,
            lw=0.9,
            **style,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective true energy [TeV]")
    ax.set_ylabel(r"$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    ax.set_title(f"Stage G diagnostic SED points, {stage_g.get('validation', {}).get('baseline', 'v3_baseline') if isinstance(stage_g.get('validation'), dict) else 'v3_baseline'}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def table_from_rows(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return "<p>n/a</p>"
    head = "".join(f"<th>{h(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{h(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def make_selection_by_nhit_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if not selector_includes(row):
            continue
        grouped.setdefault(row.get("nhit_bin", ""), []).append(row)

    out: List[Dict[str, object]] = []
    for nhit_bin, group in grouped.items():
        group = sorted(group, key=lambda row: int(row.get("cell_id", 0)))
        borrowed_notes = []
        psf_notes = []
        for row in group:
            borrowed_from = str(row.get("psf_borrowed_from", "")).strip()
            if borrowed_from:
                borrowed_notes.append(
                    f"{row.get('cell_id')} <- {borrowed_from} ({row.get('psf_borrow_method', '')})"
                )
            elif str(row.get("psf_quality_flag", "1")).strip().lower() not in {"1", "true", "yes", "y"}:
                psf_notes.append(f"{row.get('cell_id')} PSF follow-up")
        out.append(
            {
                "Nhit bin": nhit_bin,
                "kept cells": ",".join(row.get("cell_id", "") for row in group),
                "predE bins": "; ".join(row.get("predE_bin", "") for row in group),
                "ridge fractions": ", ".join(fmt(row.get("ridge_peak_fraction"), 3) for row in group),
                "MC counts": ", ".join(fmt_int(row.get("mc_count")) for row in group),
                "PSF note": "; ".join(borrowed_notes + psf_notes) or "direct Stage B PSF",
            }
        )
    return out


def make_special_selection_rows(rows: Sequence[Dict[str, str]], cell_ids: Sequence[int]) -> List[Dict[str, object]]:
    by_id = {int(row["cell_id"]): row for row in rows if row.get("cell_id", "").isdigit()}
    out: List[Dict[str, object]] = []
    for cell_id in cell_ids:
        row = by_id.get(cell_id)
        if not row:
            continue
        include = selector_includes(row)
        borrowed_from = str(row.get("psf_borrowed_from", "")).strip()
        if include and borrowed_from:
            decision = "included in v3_psfborrow"
            reason = f"physical MC ridge; nominal PSF repaired from {borrowed_from}"
        elif include:
            decision = "included"
            reason = row.get("subset_reason", "")
        else:
            decision = "excluded"
            reason = row.get("subset_reason") or row.get("exclusion_source", "")
        out.append(
            {
                "cell": cell_id,
                "Nhit bin": row.get("nhit_bin", ""),
                "predE bin": row.get("predE_bin", ""),
                "ridge frac": fmt(row.get("ridge_peak_fraction"), 3),
                "MC count": fmt_int(row.get("mc_count")),
                "decision": decision,
                "reason": reason,
            }
        )
    return out


def make_active_psf_rows(
    psf_rows: Sequence[Dict[str, str]],
    active_ids: Sequence[int],
) -> List[Dict[str, object]]:
    by_id = {int(row["cell_id"]): row for row in psf_rows if row.get("cell_id", "").isdigit()}
    out: List[Dict[str, object]] = []
    for cell_id in active_ids:
        row = by_id.get(cell_id)
        if not row:
            continue
        borrowed = str(row.get("psf_borrowed", "")).strip().lower() in {"1", "true", "yes", "y"}
        borrowed_from = row.get("borrowed_from", "")
        original_missing = row.get("original_theta_missing_crab_probability_mass") or row.get(
            "theta_missing_crab_probability_mass"
        )
        out.append(
            {
                "cell": cell_id,
                "Nhit bin": row.get("nhit_bin", ""),
                "predE bin": row.get("predE_bin", ""),
                "sigma deg": fmt(row.get("sigma_deg"), 5),
                "r_opt deg": fmt(row.get("r_opt_deg"), 5),
                "containment": fmt(row.get("containment_r_opt"), 5),
                "Neff": fmt(row.get("effective_events"), 5),
                "missing mass": fmt(row.get("theta_missing_crab_probability_mass"), 5),
                "PSF source": f"borrowed from {borrowed_from}" if borrowed else "direct",
                "orig missing": fmt(original_missing, 5) if borrowed else "",
            }
        )
    return out


def plot_active_psf_profiles(
    output_path: Path,
    npz_path: Path,
    active_ids: Sequence[int],
    psf_rows: Sequence[Dict[str, str]],
) -> Optional[Path]:
    if not npz_path.exists() or not active_ids:
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    by_id = {int(row["cell_id"]): row for row in psf_rows if row.get("cell_id", "").isdigit()}
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            ids = data["cell_id"].astype(int)
            centers = 0.5 * (data["profile_edges_deg"][:-1] + data["profile_edges_deg"][1:])
            density = data["profile_density"]
            sigma = data["sigma_deg"]
            r_opt = data["r_opt_deg"]
    except Exception:
        return None

    index = {int(cell_id): i for i, cell_id in enumerate(ids)}
    selected_ids = [cell_id for cell_id in active_ids if cell_id in index]
    if not selected_ids:
        return None
    ncols = 5
    nrows = int(np.ceil(len(selected_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 2.1 * nrows), dpi=160, sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.set_visible(False)
    for ax, cell_id in zip(axes_arr, selected_ids):
        ax.set_visible(True)
        i = index[cell_id]
        row = by_id.get(cell_id, {})
        borrowed = str(row.get("psf_borrowed", "")).strip().lower() in {"1", "true", "yes", "y"}
        color = "#d62728" if borrowed else "#1f77b4"
        ax.plot(centers, density[i], color=color, lw=1.25)
        r_value = float(r_opt[i])
        if np.isfinite(r_value):
            ax.axvline(r_value, color="#111827", lw=0.8, ls="--", alpha=0.65)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(bottom=0)
        source = f"borrow {row.get('borrowed_from', '')}" if borrowed else "direct"
        ax.set_title(
            f"{cell_id} {row.get('nhit_bin', '')}\\n{row.get('predE_bin', '')}",
            fontsize=7.0,
            color=color,
        )
        ax.text(
            0.98,
            0.92,
            f"s={sigma[i]:.3g} deg\nr={r_opt[i]:.3g} deg\n{source}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.86},
        )
        ax.grid(alpha=0.22, lw=0.45)
    for ax in axes_arr[-ncols:]:
        if ax.get_visible():
            ax.set_xlabel("offset angle [deg]", fontsize=7)
    for row_idx in range(nrows):
        ax = axes_arr[row_idx * ncols]
        if ax.get_visible():
            ax.set_ylabel("density", fontsize=7)
    fig.suptitle("Active 30-cell PSF radial profiles (v3_baseline_psfborrow)", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


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
    stage_b_dir = stage_dir(args.stage_b_dir, "psf_v3_candidate_metadata.json", ["slurm_42023"])
    stage_c_dir = stage_dir(args.stage_c_dir, "obs_events_metadata.json", ["v3_stage_c_slurm_42024"])
    stage_d_dir = stage_dir(args.stage_d_dir, "background_v3_candidate_metadata.json", ["v3_stage_d_slurm_42024"])
    stage_e_dir = stage_dir(args.stage_e_dir, "signal_v3_candidate_metadata.json", ["v3_stage_e_slurm_42024"])
    stage_f_dir = stage_dir(args.stage_f_dir, args.stage_f_metadata_name, ["v3_stage_f_slurm_42024"])
    stage_g_dir = stage_dir(args.stage_g_dir, args.stage_g_metadata_name, ["v3_stage_g_slurm_42024"])
    psfborrow_stage_b_dir = stage_dir(
        args.psfborrow_stage_b_dir,
        "psf_v3_candidate_metadata.json",
        ["v3_psfborrow_from_nominal"],
    )
    psfborrow_stage_d_dir = stage_dir(
        args.psfborrow_stage_d_dir,
        args.psfborrow_stage_d_metadata_name,
        ["v3_stage_d_psfborrow_slurm_42029"],
    )
    psfborrow_stage_e_dir = stage_dir(
        args.psfborrow_stage_e_dir,
        args.psfborrow_stage_e_metadata_name,
        ["v3_stage_e_psfborrow_slurm_42029"],
    )
    psfborrow_stage_f_dir = stage_dir(
        args.psfborrow_stage_f_dir,
        args.psfborrow_stage_f_metadata_name,
        ["v3_stage_f_psfborrow_slurm_42029"],
    )
    psfborrow_stage_g_dir = stage_dir(
        args.psfborrow_stage_g_dir,
        args.psfborrow_stage_g_metadata_name,
        ["v3_stage_g_psfborrow_slurm_42029"],
    )

    raw_rows = read_csv_rows(abs_path(args.raw_ledger_csv))
    selector_rows = read_csv_rows(abs_path(args.baseline_selector_csv))
    psfborrow_selector_rows = read_csv_rows(abs_path(args.psfborrow_selector_csv))
    systematics_rows = read_csv_rows(abs_path(args.systematics_selector_csv))
    high_energy_rows = read_csv_rows(abs_path(args.high_energy_selector_csv))
    included_ids = selector_ids(selector_rows, True)
    excluded_ids = selector_ids(selector_rows, False)
    psfborrow_included_ids = selector_ids(psfborrow_selector_rows, True)
    psfborrow_excluded_ids = selector_ids(psfborrow_selector_rows, False)
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
    psfborrow_stage_b_meta_path = psfborrow_stage_b_dir / "psf_v3_candidate_metadata.json"
    psfborrow_stage_d_meta_path = psfborrow_stage_d_dir / args.psfborrow_stage_d_metadata_name
    psfborrow_stage_e_meta_path = psfborrow_stage_e_dir / args.psfborrow_stage_e_metadata_name
    psfborrow_stage_f_meta_path = psfborrow_stage_f_dir / args.psfborrow_stage_f_metadata_name
    psfborrow_stage_g_meta_path = psfborrow_stage_g_dir / args.psfborrow_stage_g_metadata_name
    stage_b_summary_path = stage_b_dir / "psf_v3_candidate_summary.csv"
    psfborrow_stage_b_summary_path = psfborrow_stage_b_dir / "psf_v3_candidate_summary.csv"
    psfborrow_stage_b_npz_path = psfborrow_stage_b_dir / "psf_v3_candidate.npz"

    stage_a = load_json(stage_a_meta_path)
    stage_b = load_json(stage_b_meta_path)
    stage_c = load_json(stage_c_meta_path)
    stage_d = load_json(stage_d_meta_path)
    stage_e = load_json(stage_e_meta_path)
    stage_f = load_json(stage_f_meta_path)
    stage_g = load_json(stage_g_meta_path)
    psfborrow_stage_b = load_json(psfborrow_stage_b_meta_path)
    psfborrow_stage_d = load_json(psfborrow_stage_d_meta_path)
    psfborrow_stage_e = load_json(psfborrow_stage_e_meta_path)
    psfborrow_stage_f = load_json(psfborrow_stage_f_meta_path)
    psfborrow_stage_g = load_json(psfborrow_stage_g_meta_path)
    stage_b_psf_rows = read_csv_rows(stage_b_summary_path)
    psfborrow_stage_b_psf_rows = read_csv_rows(psfborrow_stage_b_summary_path)
    background_systematics = load_json(abs_path(args.background_systematics_json))
    background_systematics_rows = read_csv_rows(abs_path(args.background_systematics_csv))
    validation_summary = load_json(abs_path(args.validation_json))
    selector_systematics_rows = read_csv_rows(abs_path(args.selector_systematics_csv))
    selector_fit_rows = read_csv_rows(abs_path(args.selector_fit_comparison_csv))
    response_closure_rows = read_csv_rows(abs_path(args.response_closure_csv))
    mc_reference_closure_rows = read_csv_rows(abs_path(args.mc_reference_closure_csv))
    offsource_rows = read_csv_rows(abs_path(args.offsource_fake_source_csv))
    time_split_rows = read_csv_rows(abs_path(args.time_split_csv))
    official_pass5_csv = abs_path(args.official_pass5_sed_csv)
    official_pass5_raw_rows = read_csv_rows(official_pass5_csv)
    official_pass5_table_rows = official_pass5_rows(official_pass5_raw_rows)
    official_pass5_livetime_days = (
        finite_float(official_pass5_raw_rows[0].get("livetime_days")) if official_pass5_raw_rows else None
    )
    official_v099_csv = abs_path(args.official_v099_sed_csv)
    official_v099_raw_rows = read_csv_rows(official_v099_csv)
    official_v099_table_rows = official_v099_rows(official_v099_raw_rows)

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
    psfborrow_stage_f_ids = stage_f_cell_ids(psfborrow_stage_f)
    psfborrow_stage_g_ids = stage_g_required_cell_ids(psfborrow_stage_g)
    psfborrow_selector_matches_stage_f = bool(
        psfborrow_included_ids and psfborrow_stage_f_ids and psfborrow_included_ids == psfborrow_stage_f_ids
    )
    psfborrow_selector_matches_stage_g = bool(
        psfborrow_included_ids and psfborrow_stage_g_ids and psfborrow_included_ids == psfborrow_stage_g_ids
    )
    psfborrow_result_status = (
        "selector/result matched"
        if psfborrow_selector_matches_stage_f and psfborrow_selector_matches_stage_g
        else "PSF systematic pending rerun"
    )
    selector_matches_stage_f = bool(included_ids and stage_f_ids and included_ids == stage_f_ids)
    selector_matches_stage_g = bool(included_ids and stage_g_ids and included_ids == stage_g_ids)
    selector_result_status = (
        "selector/result matched"
        if selector_matches_stage_f and selector_matches_stage_g
        else "selector frozen; fit/SED pending rerun"
    )
    selector_pending_ids = sorted(set(included_ids) - set(stage_f_ids))
    stale_result_ids = sorted(set(stage_f_ids) - set(included_ids))
    psfborrow_added_vs_nominal_result = sorted(set(psfborrow_included_ids) - set(stage_f_ids))
    psfborrow_removed_vs_nominal_result = sorted(set(stage_f_ids) - set(psfborrow_included_ids))
    psfborrow_added_vs_selector = sorted(set(psfborrow_included_ids) - set(included_ids))
    psfborrow_removed_vs_selector = sorted(set(included_ids) - set(psfborrow_included_ids))
    psf_followup_ids = [
        int(row["cell_id"])
        for row in selector_rows
        if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}
        and str(row.get("psf_quality_flag", "1")).strip().lower() not in {"1", "true", "yes", "y"}
    ]
    sed_points = stage_g.get("points", []) if isinstance(stage_g.get("points"), list) else []
    psfborrow_sed_points = (
        psfborrow_stage_g.get("points", []) if isinstance(psfborrow_stage_g.get("points"), list) else []
    )
    psfborrow_records = psf_borrow_records(psfborrow_stage_b)
    psfborrow_record_rows = []
    for record in psfborrow_records:
        original = record.get("original") if isinstance(record.get("original"), dict) else {}
        borrowed = record.get("borrowed") if isinstance(record.get("borrowed"), dict) else {}
        sources = record.get("sources") if isinstance(record.get("sources"), list) else []
        source_text = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_text.append(
                f"{source.get('cell_id')} sigma={fmt(source.get('sigma_deg'), 5)} "
                f"r={fmt(source.get('r_opt_deg'), 5)} c={fmt(source.get('containment_r_opt'), 5)} "
                f"Neff={fmt(source.get('effective_events'), 5)}"
            )
        weights = record.get("weights") if isinstance(record.get("weights"), dict) else {}
        psfborrow_record_rows.append(
            {
                "cell": record.get("target_cell_id", ""),
                "method": record.get("method", ""),
                "borrowed_from": ",".join(str(v) for v in record.get("borrowed_from", []))
                if isinstance(record.get("borrowed_from"), list)
                else record.get("borrowed_from", ""),
                "weights": ",".join(f"{k}:{fmt(v, 4)}" for k, v in weights.items()),
                "orig missing": fmt(original.get("theta_missing_crab_probability_mass"), 5),
                "orig Neff": fmt(original.get("effective_events"), 5),
                "orig sigma": fmt(original.get("sigma_deg"), 5),
                "borrow sigma": fmt(borrowed.get("sigma_deg"), 5),
                "borrow r_opt": fmt(borrowed.get("r_opt_deg"), 5),
                "borrow containment": fmt(borrowed.get("containment_r_opt"), 5),
                "source PSF": "; ".join(source_text),
            }
        )
    psfborrow_fit_rows = make_psfborrow_fit_rows(
        stage_f,
        stage_g,
        psfborrow_stage_f,
        psfborrow_stage_g,
        stage_f_ids,
        psfborrow_stage_f_ids,
    )
    psfborrow_sed_compare_rows = make_sed_compare_rows(stage_g, psfborrow_stage_g)
    psfborrow_sed_overlay_path = plot_sed_overlay(
        REPORT_DIR / "assets/v3-psfborrow/v3_psfborrow_sed_overlay.png",
        stage_g,
        psfborrow_stage_g,
    )
    official_pass5_overlay_path = plot_official_sed_overlay(
        abs_path(args.official_pass5_overlay_png),
        stage_g,
        psfborrow_stage_g,
        official_pass5_raw_rows,
        official_v099_raw_rows,
    )
    stage_g_official_overlay_path = plot_stage_g_with_official_refs(
        abs_path(args.stage_g_official_overlay_png),
        stage_g,
        official_pass5_raw_rows,
        official_v099_raw_rows,
    )
    psfborrow_run_rows = [
        {
            "stage": label,
            "run": metadata_run_label(meta),
            "status": (
                meta.get("promotion", {}).get("status")
                if isinstance(meta.get("promotion"), dict)
                else meta.get("psf_systematic_variant", "missing" if not path.exists() else "available")
            ),
            "artifact": rel(path, REPORT_DIR) if path.exists() else "missing",
        }
        for label, meta, path in [
            ("B", psfborrow_stage_b, psfborrow_stage_b_meta_path),
            ("D", psfborrow_stage_d, psfborrow_stage_d_meta_path),
            ("E", psfborrow_stage_e, psfborrow_stage_e_meta_path),
            ("F", psfborrow_stage_f, psfborrow_stage_f_meta_path),
            ("G", psfborrow_stage_g, psfborrow_stage_g_meta_path),
        ]
    ]
    psfborrow_selector_table_rows = [
        {
            "selector": "nominal selector file",
            "cells": len(included_ids),
            "included": ",".join(str(v) for v in included_ids),
            "added vs nominal result": "",
            "removed vs nominal result": "",
            "added vs nominal selector": "",
            "removed vs nominal selector": "",
            "status": selector_result_status,
        },
        {
            "selector": "v3_baseline_psfborrow",
            "cells": len(psfborrow_included_ids),
            "included": ",".join(str(v) for v in psfborrow_included_ids),
            "added vs nominal result": ",".join(str(v) for v in psfborrow_added_vs_nominal_result) or "none",
            "removed vs nominal result": ",".join(str(v) for v in psfborrow_removed_vs_nominal_result) or "none",
            "added vs nominal selector": ",".join(str(v) for v in psfborrow_added_vs_selector) or "none",
            "removed vs nominal selector": ",".join(str(v) for v in psfborrow_removed_vs_selector) or "none",
            "status": psfborrow_result_status,
        },
    ]
    active_selection_ids = psfborrow_included_ids or included_ids
    active_selection_label = (
        "v3_baseline_psfborrow"
        if psfborrow_included_ids and psfborrow_selector_matches_stage_f and psfborrow_selector_matches_stage_g
        else args.baseline_name
    )
    active_selection_status = (
        "completed as PSF-borrowing systematic"
        if active_selection_label == "v3_baseline_psfborrow"
        else selector_result_status
    )
    selection_rows_for_active = (
        psfborrow_selector_rows if active_selection_label == "v3_baseline_psfborrow" else selector_rows
    )
    selection_by_nhit_rows = make_selection_by_nhit_rows(selection_rows_for_active)
    selection_special_rows = make_special_selection_rows(selection_rows_for_active, [39, 52, 65, 79, 80])
    active_psf_source_rows = psfborrow_stage_b_psf_rows or stage_b_psf_rows
    active_psf_rows = make_active_psf_rows(active_psf_source_rows, active_selection_ids)
    active_psf_profiles_path = plot_active_psf_profiles(
        REPORT_DIR / "assets/v3-psfborrow/v3_active_fit_cell_psf_profiles.png",
        psfborrow_stage_b_npz_path if psfborrow_stage_b_npz_path.exists() else stage_b_dir / "psf_v3_candidate.npz",
        active_selection_ids,
        active_psf_source_rows,
    )
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
                f"Result status: nominal reference; active 30-cell result is {active_selection_label}",
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
                f"Result status: nominal reference; active 30-cell result is {active_selection_label}",
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
        ("PSF-borrow Stage F report", abs_path(args.psfborrow_stage_f_report_html)),
        ("PSF-borrow Stage G report", abs_path(args.psfborrow_stage_g_report_html)),
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
            stage_d_dir / "roi_excess_grid.png",
            "Stage D counts minus fitted 2D background skymap",
            wide=True,
            explanation="Candidate-grid residual skymaps computed as observed Stage D counts minus the fitted 2D background surface. This is the background-subtracted counterpart to the counts map grid with the rho=6 deg circle.",
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
            stage_g_official_overlay_path or (stage_g_dir / "sed_points_stage_f_fullarray_pool1.png"),
            "Stage G SED points",
            wide=True,
            explanation="Diagnostic SED points built by refitting normalization in reconstructed-energy groups with the Stage F spectral shape fixed. This report version overlays the official pass5 Nhit SED and tutorial v0.99 WCDA-only SED directly on the Stage G SED plot; pass5 points are shown without error bars because the transferred summary did not include uncertainties.",
        ),
        figure(
            official_pass5_overlay_path or abs_path(args.official_pass5_overlay_png),
            "Official/tutorial WCDA SEDs versus v3 Stage G diagnostics",
            wide=True,
            explanation="Standalone comparison view for official pass5 Nhit SED and tutorial v0.99 WCDA-only SED points versus v3 Stage G diagnostics. Keep this separate view for checking the two external WCDA products against nominal and PSF-borrow v3 points.",
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
        figure(
            abs_path(args.psfborrow_fit_cell_counts_skymap),
            "PSF-borrow fit-cell Stage D counts skymap",
            wide=True,
            explanation="Observed counts maps for the PSF borrowing systematic selector. Cells 39/52/65 are included after neighboring-PSF repair, while high-Nhit edge cells 79/80 remain excluded.",
        ),
        figure(
            abs_path(args.psfborrow_fit_cell_excess_skymap),
            "PSF-borrow fit-cell Stage D excess skymap",
            wide=True,
            explanation="Counts minus fitted background for the PSF borrowing systematic selector. Compare against the nominal fit-cell excess map to isolate aperture/containment-driven changes.",
        ),
        figure(
            psfborrow_sed_overlay_path or (REPORT_DIR / "assets/v3-psfborrow/v3_psfborrow_sed_overlay.png"),
            "PSF-borrow versus nominal Stage G SED overlay",
            wide=True,
            explanation="Nominal reference Stage G points and PSF borrowing systematic points overlaid. This is a systematic comparison, not a new nominal promotion.",
        ),
        figure(
            psfborrow_stage_g_dir / "sed_points_stage_f_fullarray_pool1.png",
            "PSF-borrow Stage G SED points",
            explanation="Diagnostic SED points built from the PSF borrowing Stage F spectrum and Stage E signal table.",
        ),
        figure(
            psfborrow_stage_g_dir / "sed_points_ratio.png",
            "PSF-borrow Stage G SED ratios",
            explanation="Ratio of the PSF borrowing diagnostic SED points to the corresponding Stage F reference model.",
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
      <div class="metric"><div class="label">official pass5 points</div><div class="value">{fmt_int(len(official_pass5_table_rows))}</div><div class="note">WCDA Nhit SED, {fmt(official_pass5_livetime_days, 5)} days</div></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><div class="label">tutorial v0.99 points</div><div class="value">{fmt_int(len(official_v099_table_rows))}</div><div class="note">WCDA-only SED_Mor Crab_SED</div></div>
      <div class="metric"><div class="label">tutorial fit cluster</div><div class="value">2832848</div><div class="note">HepJob/HTCondor fit stage</div></div>
      <div class="metric"><div class="label">tutorial SED cluster</div><div class="value">2832858</div><div class="note">HepJob/HTCondor SED stage</div></div>
      <div class="metric"><div class="label">tutorial status</div><div class="value">OK</div><div class="note">7 WCDA points, SHA256 verified</div></div>
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
      Active 30-cell branch: <strong>{h(active_selection_label)}</strong> (<strong>{h(active_selection_status)}</strong>) with cells <code>{h(','.join(str(v) for v in active_selection_ids) or 'n/a')}</code>. The older nominal Stage F/G artifacts are kept as a reference; they contain <code>{h(','.join(str(v) for v in stage_f_ids) or 'n/a')}</code>, so their pending selector cells are <code>{h(','.join(str(v) for v in selector_pending_ids) or 'none')}</code> and stale result-only cells are <code>{h(','.join(str(v) for v in stale_result_ids) or 'none')}</code>.
    </div>
    <p>{link_html}</p>
  </section>

  <section>
    <h2>30-Cell Selection</h2>
    <div class="callout">
      <p>The final fit-cell set used in the PSF-borrowing systematic is a frozen prefit MC/response selection, not a Crab-significance selection. It starts from the 84-cell <code>Nhit x predE</code> candidate grid, keeps cells inside the MC central response support, follows the MC occupancy ridge in each Nhit row, requires adequate MC/statistical support and usable PSF information, and then applies the high-Nhit edge veto. The selector is therefore fixed before looking at Stage E excess, Stage F pulls, or Stage G residuals.</p>
      <p>The active selection contains <strong>{fmt_int(len(active_selection_ids))}</strong> cells: <code>{h(','.join(str(v) for v in active_selection_ids) or 'n/a')}</code>. Cells <code>39/52/65</code> are kept because they are ridge-left physical candidates; their nominal PSF fails the theta-support test, so the <code>v3_psfborrow</code> branch replaces only their PSF with neighboring-cell PSFs. Cells <code>79/80</code> stay excluded because they are high-Nhit edge cells with low-stat/PSF-untrusted behavior.</p>
    </div>
    <h3>Selection by Nhit row</h3>
    {table_from_rows(selection_by_nhit_rows, ['Nhit bin', 'kept cells', 'predE bins', 'ridge fractions', 'MC counts', 'PSF note'])}
    <h3>Special cell decisions</h3>
    {table_from_rows(selection_special_rows, ['cell', 'Nhit bin', 'predE bin', 'ridge frac', 'MC count', 'decision', 'reason'])}
    <p>Read this together with the <strong>MC occupancy ridge fraction</strong> figure: each row is normalized by its own Nhit-row peak, so a value near one marks the dominant MC response bin for that Nhit range, while accepted left/right shoulder cells are retained only when they remain inside the prefit response ridge and pass the additional quality rules.</p>
  </section>

  <section>
    <h2>Active Fit-Cell PSF Diagnostics</h2>
    <div class="callout">
      <p>This section shows the PSF actually used by the active <code>{h(active_selection_label)}</code> fit-cell branch. For direct cells the values come from Stage B; for cells <code>39/52/65</code> the active PSF is the borrowed/interpolated neighbor PSF while the original missing theta-support diagnostic is preserved in the table.</p>
    </div>
    {figure(active_psf_profiles_path or Path('__missing_active_psf_profiles.png'), 'Active 30-cell PSF radial profiles', wide=True, explanation='Radial PSF density profiles for the active fit cells. Red panels are cells whose active PSF is borrowed/interpolated from neighboring cells; the dashed vertical line marks r_opt used by the aperture optimization.')}
    {figure(stage_b_dir / 'psf_sigma_deg_grid.png', 'Stage B PSF sigma grid', wide=True, explanation='Candidate-grid Rayleigh-core PSF width sigma in degrees. Smaller sigma means a narrower reconstructed Crab response for that cell.')}
    {figure(stage_b_dir / 'psf_r_opt_deg_grid.png', 'Stage B PSF r_opt grid', wide=True, explanation='Candidate-grid optimized aperture radius r_opt in degrees, derived from the Stage B PSF model.')}
    {figure(stage_b_dir / 'psf_containment_grid.png', 'Stage B PSF containment at r_opt grid', wide=True, explanation='Fraction of the PSF contained inside r_opt for each candidate cell. Low containment or warnings indicate a broad tail or low-stat PSF behavior.')}
    {figure(stage_b_dir / 'psf_effective_events_grid.png', 'Stage B PSF effective-events grid', wide=True, explanation='Effective MC statistics after Crab-declination theta reweighting. Low values are the main reason some visually plausible cells need PSF follow-up or borrowing.')}
    <h3>Active fit-cell PSF table</h3>
    {table_from_rows(active_psf_rows, ['cell', 'Nhit bin', 'predE bin', 'sigma deg', 'r_opt deg', 'containment', 'Neff', 'missing mass', 'PSF source', 'orig missing'])}
  </section>

  <section>
    <h2>PSF Theta-Support Notes</h2>
    <div class="callout">
      <p><code>theta_missing_crab_probability_mass</code> measures the fraction of the Crab declination theta exposure for which a given cell has no MC support after applying that cell's Nhit/predE selection, true-energy range, finite-angle, and positive-weight requirements. It is not a measure of the total MC sample size; it is a coverage test on the conditional MC sample inside one cell.</p>
      <p>The important lesson for v3 is that many MC events globally do not guarantee full theta coverage inside every fine <code>Nhit x predE</code> cell. The ridge-left cells <code>39</code>, <code>52</code>, and <code>65</code> have visible Crab excess and satisfy the MC occupancy ridge rule, but their Stage B PSF falls back because their conditional MC theta support misses more than 10% of the Crab theta exposure. Current values are approximately <code>0.124</code>, <code>0.228</code>, and <code>0.208</code>, respectively.</p>
      <p>These cells are the left shoulder of the ridge: high Nhit but lower predicted energy than the row peak. That combination can be more sensitive to zenith angle, shower geometry, and NN response residuals, so its theta support can be less continuous than the neighboring right-shoulder cells. In contrast, adjacent cells <code>40</code>, <code>53</code>, and <code>66</code> pass with missing masses near <code>0.000</code>, <code>0.083</code>, and <code>0.062</code>.</p>
      <p>A follow-up audit of cell <code>39</code> shows that the fallback summary row can be misleading: fallback rows reset diagnostic counters such as <code>logE_range_events</code> to zero. The underlying ROOT files for cell <code>39</code> contain <code>27,733</code> positive-weight events in <code>log10(mc_energy)=[2,6)</code> out of <code>27,769</code> total events. The problem is therefore not the true-energy window; it is sparse high-theta support after Crab-declination reweighting.</p>
      <p>For cell <code>39</code>, the missing 1-degree theta bins are all in the high-theta tail: <code>40-41</code>, <code>41-42</code>, <code>43-44</code>, <code>45-46</code>, <code>46-47</code>, and <code>48-49 deg</code>, totaling <code>0.12449</code> Crab theta probability mass. With 2-degree theta bins the missing mass drops to <code>0.04146</code> and only <code>40-42 deg</code> is unsupported, but the reweighted effective event count falls to <code>108.29</code>, below the Stage B threshold of <code>200</code>. Wider theta bins remove the missing-bin flag, but the effective event count remains lower because a very small number of high-theta MC events receives large weights.</p>
      <p>The selector decision is therefore: keep <code>39/52/65</code> in the frozen physical-ridge cell list, but do not use their fallback PSF as nominal. The dedicated <code>v3_psfborrow</code> systematic replaces their active PSF with neighboring ridge PSFs while preserving the original missing-support diagnostics in metadata and summary tables.</p>
      <p>The active PSF systematic uses <code>39 -> 40</code>, <code>52 -> 2/3*53 + 1/3*54</code>, and <code>65 -> 2/3*66 + 1/3*67</code>. It keeps 39/52/65 in the fit selector and continues to exclude high-Nhit edge cells such as <code>79/80</code>. This branch is explicitly a PSF systematic, not a nominal v3 promotion.</p>
    </div>
  </section>

  <section>
    <h2>PSF Borrowing Systematic</h2>
    <div class="callout">
      This branch is labeled <code>v3_psfborrow</code>. It starts from nominal Stage B <code>psf_v3_candidate.npz</code>, writes a separate Stage B artifact under <code>stage_b_v3_candidate_psfborrow</code>, and reruns Stage D/E/F/G into <code>*_psfborrow</code> output roots. It does not overwrite nominal v3 or promote this result as nominal.
    </div>
    <h3>Borrowed PSF audit</h3>
    {table_from_rows(psfborrow_record_rows, ['cell', 'method', 'borrowed_from', 'weights', 'orig missing', 'orig Neff', 'orig sigma', 'borrow sigma', 'borrow r_opt', 'borrow containment', 'source PSF'])}
    <h3>Selector change</h3>
    {table_from_rows(psfborrow_selector_table_rows, ['selector', 'cells', 'included', 'added vs nominal result', 'removed vs nominal result', 'added vs nominal selector', 'removed vs nominal selector', 'status'])}
    <h3>PSF systematic run artifacts</h3>
    {table_from_rows(psfborrow_run_rows, ['stage', 'run', 'status', 'artifact'])}
    <h3>Stage F/G comparison</h3>
    {table_from_rows(psfborrow_fit_rows, ['version', 'run', 'cells', 'added', 'removed', 'model', 'error', 'phi0 (delta)', 'gamma/alpha (delta)', 'beta (delta)', 'chi2/ndof', 'delta chi2', 'SED pts'])}
    <h3>Stage G SED point comparison</h3>
    {table_from_rows(psfborrow_sed_compare_rows, ['grouping', 'group', 'cells', 'E_eff TeV', 'E2 dN/dE', 'err', 'ratio StageF'])}
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
    <h2>Official WCDA Pass5 SED</h2>
    <div class="callout">
      The official LHAASO-WCDA program fit on pass5 z50 data produced seven Crab Nhit SED points with WCDA livetime <code>{fmt(official_pass5_livetime_days, 5)} days</code>. All seven rows have <code>Error_status=3</code>, no upper-limit rows, and empty stderr in the external run summary. The table preserves the official <code>dN/dE</code> values and adds <code>E^2 dN/dE</code> only for comparison with this report's Stage G diagnostic convention. No error bars are drawn for these official points because no uncertainty columns were provided in the transferred summary.
    </div>
    <p>Source package: <code>wcda_crab_sed_pass5_20260616_104941/final_wcda_crab_sed_pass5.tar.gz</code>; source table: <code>wcda_crab_sed_pass5_20260616_104941/final_wcda_crab_sed_pass5/sed_J0534+2200.txt</code>.</p>
    {table_from_rows(official_pass5_table_rows, ['E TeV', 'dN/dE', 'E2 dN/dE', 'TS', 'Nhit', 'Error_status', 'upper limit', 'stderr empty'])}
  </section>

  <section>
    <h2>Tutorial v0.99 WCDA-Only SED</h2>
    <div class="callout">
      The second tutorial route <code>v0.99</code> produced seven WCDA-only Crab SED points from <code>results/SED_Mor/Crab_SED.txt</code>. The transferred table reports <code>energy flux ferrL ferrU TS WCDAtag</code>; the flux scale matches <code>E^2 dN/dE</code> in units of <code>1e-14 TeV cm^-2 s^-1</code>, so this report stores the raw scaled values and converts them to physical <code>E^2 dN/dE</code> for overlays. All points are WCDA tagged and Crab component/bin/flux-point validation is OK; ROOT output opens and SHA256 verification passed.
    </div>
    <p>Package: <code>/home/lhaaso/liushijie/energy/wcda_crab_sed_v099_20250731_20260616_123624/final_wcda_crab_sed_v099_20250731.tar.gz</code>. Fit cluster: <code>2832848</code>; SED cluster: <code>2832858</code>. README warnings for <code>Halo_2_Ecut</code> and <code>Halo_2_F0</code> are boundary/upper-limit related and do not affect the listed Crab SED/TS status.</p>
    {table_from_rows(official_v099_table_rows, ['E TeV', 'E2 flux raw', 'E2 dN/dE', 'err low', 'err high', 'TS', 'WCDAtag', 'status'])}
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
