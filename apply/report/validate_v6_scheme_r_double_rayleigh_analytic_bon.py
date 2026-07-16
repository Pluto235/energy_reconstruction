#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from apply.report.validate_v6_scheme_r_double_rayleigh import (
    branch_record,
    exact_array_equal,
    image_validation,
    load_json,
    load_npz,
    same_path,
    sha256,
)


REPO_ROOT = Path(os.environ.get("V6_VALIDATION_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
SOURCE_RUN_ID = "v6_64748_nhit100_highEplus1_split56"
PSF2_RUN_ID = f"{BASE_RUN_ID}_double_rayleigh"
ANALYTIC1_ID = f"{BASE_RUN_ID}_analytic_bon"
ANALYTIC2_ID = f"{PSF2_RUN_ID}_analytic_bon"
EXPERIMENT1_ID = f"{BASE_RUN_ID}_scheme_R_fixed712979_analytic_bon"
EXPERIMENT2_ID = f"{PSF2_RUN_ID}_scheme_R_fixed712979_analytic_bon"
TARGET_CONTAINMENT = 0.7129790300890827
SELECTOR_SHA256 = "c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6"

SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{BASE_RUN_ID}_fit.csv"
NOMINAL_RESPONSE = REPO_ROOT / f"apply/output/stage_a_{SOURCE_RUN_ID}/response_2d_{SOURCE_RUN_ID}.npz"
FIXED_RESPONSE_ROOT = REPO_ROOT / f"apply/output/stage_e_{BASE_RUN_ID}_fixed712979_rayleigh_annnorm"
FIXED_RESPONSE = FIXED_RESPONSE_ROOT / f"response_{BASE_RUN_ID}_fixed712979_rayleigh.npz"
FIXED_RESPONSE_META = FIXED_RESPONSE_ROOT / f"response_{BASE_RUN_ID}_fixed712979_rayleigh_metadata.json"
STAGE_C = REPO_ROOT / f"apply/output/stage_c_{SOURCE_RUN_ID}/runs/{SOURCE_RUN_ID}_stage_c_halfyear"

REPORT = REPO_ROOT / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_analytic_bon_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-analytic-bon"
VALIDATION_JSON = ASSET_DIR / "scheme_R_double_rayleigh_analytic_bon_validation.json"
COMPARISON_JSON = ASSET_DIR / "scheme_R_double_rayleigh_analytic_bon_comparison.json"
REPORT_VALIDATION_JSON = ASSET_DIR / "report_validation.json"


def branch_paths(kind: str, *, analytic: bool) -> dict[str, Path | str]:
    is_2r = kind == "2R"
    psf_id = PSF2_RUN_ID if is_2r else BASE_RUN_ID
    if analytic:
        analysis_id = ANALYTIC2_ID if is_2r else ANALYTIC1_ID
        experiment_id = EXPERIMENT2_ID if is_2r else EXPERIMENT1_ID
        d_root = REPO_ROOT / f"apply/output/stage_d_{analysis_id}_annnorm/runs/{analysis_id}_stage_d_annnorm"
        e_root = REPO_ROOT / f"apply/output/stage_e_{analysis_id}_containment1_annnorm/runs/{analysis_id}_stage_e_containment1_annnorm"
        f_root = REPO_ROOT / f"apply/output/stage_f_{experiment_id}/runs/{experiment_id}_stage_f"
        g_root = REPO_ROOT / f"apply/output/stage_g_{experiment_id}/runs/{experiment_id}_stage_g"
        return {
            "branch_id": experiment_id,
            "psf": REPO_ROOT / f"apply/output/stage_b_{psf_id}/runs/{psf_id}_stage_b_psf/psf_{psf_id}.npz",
            "psf_meta": REPO_ROOT / f"apply/output/stage_b_{psf_id}/runs/{psf_id}_stage_b_psf/psf_{psf_id}_metadata.json",
            "d": d_root / f"background_{analysis_id}_annnorm.npz",
            "d_meta": d_root / f"background_{analysis_id}_annnorm_metadata.json",
            "e": e_root / f"signal_{analysis_id}_containment1_annnorm.npz",
            "e_meta": e_root / f"signal_{analysis_id}_containment1_annnorm_metadata.json",
            "f": f_root / f"fit_{experiment_id}.npz",
            "f_meta": f_root / f"fit_{experiment_id}_metadata.json",
            "f_dir": f_root,
            "g": g_root / f"sed_points_{experiment_id}.npz",
            "g_meta": g_root / f"sed_points_{experiment_id}_metadata.json",
            "g_summary": g_root / f"sed_points_{experiment_id}_summary.json",
            "g_dir": g_root,
        }
    if is_2r:
        d_root = REPO_ROOT / f"apply/output/stage_d_{PSF2_RUN_ID}_annnorm/runs/{PSF2_RUN_ID}_stage_d_annnorm"
        e_root = REPO_ROOT / f"apply/output/stage_e_{PSF2_RUN_ID}_containment1_annnorm/runs/{PSF2_RUN_ID}_stage_e_containment1_annnorm"
        f_root = REPO_ROOT / f"apply/output/stage_f_{EXPERIMENT2_ID.removesuffix('_analytic_bon')}/runs/{EXPERIMENT2_ID.removesuffix('_analytic_bon')}_stage_f"
        g_root = REPO_ROOT / f"apply/output/stage_g_{EXPERIMENT2_ID.removesuffix('_analytic_bon')}/runs/{EXPERIMENT2_ID.removesuffix('_analytic_bon')}_stage_g"
        legacy_experiment = EXPERIMENT2_ID.removesuffix("_analytic_bon")
        return {
            "branch_id": legacy_experiment,
            "d": d_root / f"background_{PSF2_RUN_ID}_annnorm.npz",
            "d_meta": d_root / f"background_{PSF2_RUN_ID}_annnorm_metadata.json",
            "e": e_root / f"signal_{PSF2_RUN_ID}_containment1_annnorm.npz",
            "e_meta": e_root / f"signal_{PSF2_RUN_ID}_containment1_annnorm_metadata.json",
            "f": f_root / f"fit_{legacy_experiment}.npz",
            "f_meta": f_root / f"fit_{legacy_experiment}_metadata.json",
            "f_dir": f_root,
            "g_meta": g_root / f"sed_points_{legacy_experiment}_metadata.json",
            "g_dir": g_root,
        }
    legacy_experiment = f"{BASE_RUN_ID}_scheme_R_fixed712979"
    d_root = REPO_ROOT / f"apply/output/stage_d_{BASE_RUN_ID}_annnorm/runs/{BASE_RUN_ID}_stage_d_annnorm"
    e_root = FIXED_RESPONSE_ROOT / f"runs/{BASE_RUN_ID}_stage_e_fixed712979_rayleigh_annnorm"
    f_root = REPO_ROOT / f"apply/output/stage_f_{legacy_experiment}/runs/{BASE_RUN_ID}_stage_f_scheme_R_fixed712979"
    g_root = REPO_ROOT / f"apply/output/stage_g_{legacy_experiment}/runs/{BASE_RUN_ID}_stage_g_scheme_R_fixed712979"
    return {
        "branch_id": legacy_experiment,
        "d": d_root / f"background_{BASE_RUN_ID}_annnorm.npz",
        "d_meta": d_root / f"background_{BASE_RUN_ID}_annnorm_metadata.json",
        "e": e_root / f"signal_{BASE_RUN_ID}_fixed712979_rayleigh_annnorm.npz",
        "e_meta": e_root / f"signal_{BASE_RUN_ID}_fixed712979_rayleigh_annnorm_metadata.json",
        "f": f_root / f"fit_{legacy_experiment}.npz",
        "f_meta": f_root / f"fit_{legacy_experiment}_metadata.json",
        "f_dir": f_root,
        "g_meta": g_root / f"sed_points_{legacy_experiment}_metadata.json",
        "g_dir": g_root,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Scheme R analytic B_on 1R/2R experiments.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--branch", choices=["1R", "2R"])
    parser.add_argument("--require-report", action="store_true")
    return parser.parse_args()


def selected_cells() -> list[int]:
    with SELECTOR.open(newline="", encoding="utf-8") as handle:
        return sorted(int(float(row["cell_id"])) for row in csv.DictReader(handle) if row.get("include") == "1")


def subset_indices(cell_ids: np.ndarray, selected: list[int]) -> np.ndarray:
    index = {int(cell_id): idx for idx, cell_id in enumerate(np.asarray(cell_ids, dtype=np.int64))}
    missing = [cell_id for cell_id in selected if cell_id not in index]
    if missing:
        raise ValueError(f"Missing selected cells: {missing}")
    return np.asarray([index[cell_id] for cell_id in selected], dtype=np.int64)


def gauss_disk_integral(coefficients: np.ndarray, radius: float, step: float, scale: float) -> float:
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(48)
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(64)
    rho = 0.5 * radius * (radial_nodes + 1.0)
    theta = math.pi * (angular_nodes + 1.0)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    c = np.asarray(coefficients, dtype=np.float64)
    values = c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y
    weights = 0.5 * radius * radial_weights[:, None] * math.pi * angular_weights[None, :]
    return float(scale * np.sum(values * rr * weights) / step**2)


def check_branch(kind: str, selected: list[int], check: Any) -> None:
    analytic = branch_paths(kind, analytic=True)
    legacy = branch_paths(kind, analytic=False)
    required = [
        analytic["d"], analytic["d_meta"], analytic["e"], analytic["e_meta"],
        analytic["f"], analytic["f_meta"], analytic["g"], analytic["g_meta"], analytic["g_summary"],
        Path(analytic["f_dir"]) / "pull_grid_logpar.png",
        Path(analytic["g_dir"]) / "sed_points_ratio.png",
    ]
    missing = [str(path) for path in required if not Path(path).exists() or Path(path).stat().st_size == 0]
    check(f"{kind}_required_outputs", not missing, {"missing": missing})
    if missing:
        return

    new_d = load_npz(Path(analytic["d"]))
    old_d = load_npz(Path(legacy["d"]))
    old_idx = subset_indices(old_d["cell_id"], selected)
    ids = np.asarray(new_d["cell_id"], dtype=np.int64).tolist()
    check(f"{kind}_stage_d_44_cells", ids == selected, ids)
    unchanged = [
        "r_opt_deg", "counts_map", "background_map", "surface_coefficients", "annulus_surface_scale",
        "annulus_inner_deg", "annulus_outer_deg", "annulus_counts", "annulus_pixels", "training_mask",
    ]
    equality = {key: exact_array_equal(np.asarray(new_d[key]), np.asarray(old_d[key])[old_idx]) for key in unchanged}
    check(f"{kind}_stage_d_surface_reused", all(equality.values()), equality)
    check(
        f"{kind}_legacy_b_on_reproduced",
        exact_array_equal(np.asarray(new_d["B_on_pixel_center"]), np.asarray(old_d["B_on"])[old_idx]),
        "B_on_pixel_center exactly equals legacy B_on",
    )
    step = float(np.asarray(new_d["on_aperture_grid_step_deg"], dtype=np.float64)[0])
    coefficients = np.asarray(new_d["surface_coefficients"], dtype=np.float64)
    radii = np.asarray(new_d["r_opt_deg"], dtype=np.float64)
    scales = np.asarray(new_d["annulus_surface_scale"], dtype=np.float64)
    expected = scales / step**2 * (
        math.pi * radii**2 * coefficients[:, 0]
        + 0.25 * math.pi * radii**4 * (coefficients[:, 3] + coefficients[:, 5])
    )
    active = np.asarray(new_d["B_on"], dtype=np.float64)
    check(
        f"{kind}_analytic_formula",
        bool(np.allclose(active, expected, rtol=5.0e-12, atol=1.0e-8)),
        {"max_abs_diff": float(np.max(np.abs(active - expected)))},
    )
    quadrature = np.asarray(
        [gauss_disk_integral(coefficients[i], radii[i], step, scales[i]) for i in range(len(radii))],
        dtype=np.float64,
    )
    check(
        f"{kind}_independent_gauss_integral",
        bool(np.allclose(active, quadrature, rtol=1.0e-10, atol=1.0e-8)),
        {"max_relative_diff": float(np.max(np.abs(active - quadrature) / active))},
    )
    minima = np.asarray(new_d["on_aperture_surface_min"], dtype=np.float64)
    check(f"{kind}_positive_on_aperture", bool(np.all(minima > 0.0)), {"minimum": float(np.min(minima))})
    d_meta = load_json(Path(analytic["d_meta"]))
    model = d_meta.get("background_model") or {}
    check(
        f"{kind}_analytic_metadata_contract",
        model.get("on_aperture_integration") == "analytic-quadratic"
        and model.get("analytic_positive_surface_required") is True,
        model,
    )

    new_e = load_npz(Path(analytic["e"]))
    old_e = load_npz(Path(legacy["e"]))
    old_e_idx = subset_indices(old_e["cell_id"], selected)
    check(
        f"{kind}_stage_e_n_on_unchanged",
        exact_array_equal(np.asarray(new_e["N_on"]), np.asarray(old_e["N_on"])[old_e_idx]),
        "event-level N_on arrays exactly equal",
    )
    check(
        f"{kind}_stage_e_active_b_on",
        exact_array_equal(np.asarray(new_e["B_on"], dtype=np.float64), active),
        "Stage E B_on exactly equals Stage D analytic B_on",
    )
    check(
        f"{kind}_stage_e_containment_one",
        bool(np.array_equal(np.asarray(new_e["containment_r_opt"]), np.ones(len(selected), dtype=new_e["containment_r_opt"].dtype))),
        "containment_r_opt is exactly one",
    )

    new_f = load_npz(Path(analytic["f"]))
    old_f = load_npz(Path(legacy["f"]))
    f_meta = load_json(Path(analytic["f_meta"]))
    f_inputs = f_meta.get("inputs") or {}
    check(f"{kind}_stage_f_fixed_response", same_path(f_inputs.get("response_npz"), FIXED_RESPONSE), f_inputs)
    check(f"{kind}_stage_f_selector", np.asarray(new_f["cell_id"], dtype=np.int64).tolist() == selected, selected)
    check(
        f"{kind}_stage_f_exposure_unchanged",
        exact_array_equal(np.asarray(new_f["theta_exposure_sec"]), np.asarray(old_f["theta_exposure_sec"])),
        "theta_exposure_sec arrays exactly equal",
    )
    g_meta = load_json(Path(analytic["g_meta"]))
    old_g_meta = load_json(Path(legacy["g_meta"]))
    check(f"{kind}_stage_g_groupings_unchanged", g_meta.get("groupings") == old_g_meta.get("groupings"), g_meta.get("groupings"))


def branch_cell_data(signal_path: Path, fit_path: Path) -> dict[int, dict[str, float]]:
    signal = load_npz(signal_path)
    fit = load_npz(fit_path)
    fit_index = {int(cell_id): idx for idx, cell_id in enumerate(np.asarray(fit["cell_id"], dtype=np.int64))}
    out: dict[int, dict[str, float]] = {}
    for idx, cell_id in enumerate(np.asarray(signal["cell_id"], dtype=np.int64)):
        cid = int(cell_id)
        if cid not in fit_index:
            continue
        fidx = fit_index[cid]
        out[cid] = {
            "N_on": float(signal["N_on"][idx]),
            "B_on": float(signal["B_on"][idx]),
            "excess": float(signal["excess"][idx]),
            "pull": float(fit["logpar_conservative_pull"][fidx]),
        }
    return out


def build_comparison(selected: list[int]) -> dict[str, Any]:
    specs = [
        ("Scheme R 1R legacy", "1R", False),
        ("Scheme R 1R analytic B_on", "1R", True),
        ("Scheme R 2R legacy", "2R", False),
        ("Scheme R 2R analytic B_on", "2R", True),
    ]
    records: list[dict[str, Any]] = []
    cells_by_branch: dict[str, dict[int, dict[str, float]]] = {}
    for label, kind, analytic in specs:
        paths = branch_paths(kind, analytic=analytic)
        branch_id = str(paths["branch_id"])
        record, _ = branch_record(
            label,
            branch_id,
            FIXED_RESPONSE,
            Path(paths["f"]),
            Path(paths["f_meta"]),
            f"{kind} aperture; Aeff_R={TARGET_CONTAINMENT}*Aeff_nominal; "
            + ("analytic quadratic B_on" if analytic else "pixel-center B_on"),
        )
        record["background_integration"] = "analytic-quadratic" if analytic else "pixel-center"
        record["signal_path"] = str(paths["e"])
        records.append(record)
        cells_by_branch[branch_id] = branch_cell_data(Path(paths["e"]), Path(paths["f"]))
    cell_rows = []
    for cell_id in selected:
        row: dict[str, Any] = {"cell_id": cell_id, "branches": {}}
        pulls = []
        for record in records:
            branch_id = record["branch_id"]
            values = cells_by_branch[branch_id][cell_id]
            row["branches"][branch_id] = values
            pulls.append(abs(values["pull"]))
        row["max_abs_pull"] = max(pulls)
        cell_rows.append(row)
    return {
        "experiment_id": EXPERIMENT2_ID,
        "control_experiment_id": EXPERIMENT1_ID,
        "target_containment": TARGET_CONTAINMENT,
        "selector": str(SELECTOR),
        "selected_cells": selected,
        "comparison": records,
        "cell_rows": cell_rows,
        "large_pull_threshold": 5.0,
        "large_pull_cells": [row for row in cell_rows if row["max_abs_pull"] >= 5.0],
        "input_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail})

    shared_required = [SELECTOR, NOMINAL_RESPONSE, FIXED_RESPONSE, FIXED_RESPONSE_META, STAGE_C / "obs_events_metadata.json"]
    for kind in ("1R", "2R"):
        analytic = branch_paths(kind, analytic=True)
        legacy = branch_paths(kind, analytic=False)
        shared_required.extend([Path(analytic["psf"]), Path(analytic["psf_meta"]), Path(legacy["d"]), Path(legacy["e"]), Path(legacy["f"]), Path(legacy["f_meta"]), Path(legacy["g_meta"])])
    missing = [str(path) for path in shared_required if not path.exists() or path.stat().st_size == 0]
    check("required_preflight_artifacts", not missing, {"missing": missing})
    if missing:
        raise SystemExit(json.dumps(checks, indent=2))
    selected = selected_cells()
    check("selector_sha256", sha256(SELECTOR) == SELECTOR_SHA256, sha256(SELECTOR))
    check("selector_44_cells", len(selected) == 44, selected)
    nominal = load_npz(NOMINAL_RESPONSE)
    fixed = load_npz(FIXED_RESPONSE)
    non_aeff_equal = set(nominal) == set(fixed) and all(exact_array_equal(nominal[key], fixed[key]) for key in nominal if key != "a_eff")
    mask = np.isfinite(nominal["a_eff"]) & (nominal["a_eff"] != 0)
    ratio = fixed["a_eff"][mask] / nominal["a_eff"][mask]
    check("fixed_response_contract", non_aeff_equal and bool(np.allclose(ratio, TARGET_CONTAINMENT, rtol=2.0e-7, atol=0.0)), {"ratio_min": float(np.min(ratio)), "ratio_max": float(np.max(ratio))})
    psf2_meta = load_json(Path(branch_paths("2R", analytic=True)["psf_meta"]))
    target = (psf2_meta.get("psf_comparison") or {}).get("target_containment")
    check("double_rayleigh_target", math.isclose(float(target), TARGET_CONTAINMENT, rel_tol=0.0, abs_tol=1.0e-15), target)

    if args.preflight_only:
        failed = [row for row in checks if row["status"] != "passed"]
        print(json.dumps({"status": "failed" if failed else "passed", "checks": checks}, indent=2))
        if failed:
            raise SystemExit(1)
        return

    kinds = [args.branch] if args.branch else ["1R", "2R"]
    for kind in kinds:
        assert kind is not None
        check_branch(kind, selected, check)
    comparison = None if args.branch else build_comparison(selected)

    if args.require_report:
        images = image_validation(REPORT) if REPORT.exists() else {"image_refs": [], "missing_image_refs": [str(REPORT)], "image_ref_count": 0}
        text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
        check(
            "final_report_sections",
            REPORT.exists() and all(token in text for token in ["Analytic B_on Contract", "Four-Branch Controlled Comparison", "Cell 1 Diagnostic", "Large-Pull Migration"]),
            str(REPORT),
        )
        check("final_report_images", images["image_ref_count"] >= 24 and not images["missing_image_refs"], images)
        report_validation = load_json(REPORT_VALIDATION_JSON) if REPORT_VALIDATION_JSON.exists() else {}
        contamination = report_validation.get("metadata_contamination") or {}
        check("metadata_contamination", contamination.get("status") == "passed" and not contamination.get("offenders"), contamination)
        if comparison is not None:
            comparison["report_path"] = str(REPORT)
            comparison["report_sha256"] = sha256(REPORT)

    failed = [row for row in checks if row["status"] != "passed"]
    status = "failed" if failed else "passed"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    output = ASSET_DIR / f"branch_{args.branch}_validation.json" if args.branch else VALIDATION_JSON
    output.write_text(json.dumps({"status": status, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    if comparison is not None:
        COMPARISON_JSON.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Validation {status}: {output}")
    if failed:
        raise SystemExit(f"Analytic B_on validation failed: {[row['name'] for row in failed]}")


if __name__ == "__main__":
    main()
