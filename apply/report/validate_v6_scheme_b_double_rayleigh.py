#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
RUN_ID = f"{BASE_RUN_ID}_double_rayleigh"
SOURCE_RUN_ID = "v6_64748_nhit100_highEplus1_split56"
TARGET_CONTAINMENT = 0.7129790300890827
SELECTOR_SHA256 = "c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the isolated v6 Scheme B double-Rayleigh Stage A-G chain.")
    parser.add_argument("--require-report", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    selector = REPO_ROOT / "apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv"
    stage_b = REPO_ROOT / f"apply/output/stage_b_{RUN_ID}/runs/{RUN_ID}_stage_b_psf"
    old_stage_b = REPO_ROOT / f"apply/output/stage_b_{BASE_RUN_ID}/runs/{BASE_RUN_ID}_stage_b_psf"
    response_dir = REPO_ROOT / f"apply/output/stage_a_{RUN_ID}_aperture_conditioned"
    stage_c = REPO_ROOT / f"apply/output/stage_c_{SOURCE_RUN_ID}/runs/{SOURCE_RUN_ID}_stage_c_halfyear"
    stage_d = REPO_ROOT / f"apply/output/stage_d_{RUN_ID}_annnorm/runs/{RUN_ID}_stage_d_annnorm"
    stage_e = REPO_ROOT / f"apply/output/stage_e_{RUN_ID}_containment1_annnorm/runs/{RUN_ID}_stage_e_containment1_annnorm"
    stage_f = REPO_ROOT / f"apply/output/stage_f_{RUN_ID}/runs/{RUN_ID}_stage_f"
    stage_g = REPO_ROOT / f"apply/output/stage_g_{RUN_ID}/runs/{RUN_ID}_stage_g"
    old_stage_d = REPO_ROOT / f"apply/output/stage_d_{BASE_RUN_ID}_annnorm/runs/{BASE_RUN_ID}_stage_d_annnorm"
    old_stage_f = REPO_ROOT / f"apply/output/stage_f_{BASE_RUN_ID}/runs/{BASE_RUN_ID}_stage_f"
    old_stage_g = REPO_ROOT / f"apply/output/stage_g_{BASE_RUN_ID}/runs/{BASE_RUN_ID}_stage_g"

    psf_npz = stage_b / f"psf_{RUN_ID}.npz"
    psf_meta_path = stage_b / f"psf_{RUN_ID}_metadata.json"
    response_npz = response_dir / f"response_2d_{RUN_ID}_aperture_conditioned.npz"
    response_meta_path = response_dir / f"response_2d_{RUN_ID}_aperture_conditioned_metadata.json"
    d_npz = stage_d / f"background_{RUN_ID}_annnorm.npz"
    d_meta_path = stage_d / f"background_{RUN_ID}_annnorm_metadata.json"
    e_npz = stage_e / f"signal_{RUN_ID}_containment1_annnorm.npz"
    e_meta_path = stage_e / f"signal_{RUN_ID}_containment1_annnorm_metadata.json"
    f_npz = stage_f / f"fit_{RUN_ID}.npz"
    f_meta_path = stage_f / f"fit_{RUN_ID}_metadata.json"
    g_npz = stage_g / f"sed_points_{RUN_ID}.npz"
    g_meta_path = stage_g / f"sed_points_{RUN_ID}_metadata.json"
    g_summary_json = stage_g / f"sed_points_{RUN_ID}_summary.json"

    required = [
        selector,
        psf_npz,
        psf_meta_path,
        stage_b / f"psf_{RUN_ID}_summary.csv",
        stage_b / f"psf_{RUN_ID}_summary.md",
        response_npz,
        response_meta_path,
        stage_c / "obs_events_metadata.json",
        d_npz,
        d_meta_path,
        stage_d / "roi_counts_grid.png",
        stage_d / "roi_excess_grid.png",
        e_npz,
        e_meta_path,
        stage_e / "formal_sigma_grid.png",
        f_npz,
        f_meta_path,
        stage_f / "pull_grid_logpar.png",
        g_npz,
        g_meta_path,
        g_summary_json,
        stage_g / "sed_points_ratio.png",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail})

    check("required_artifacts", not missing, {"missing": missing, "count": len(required)})
    if missing:
        write_and_fail(checks)

    check("selector_sha256", sha256(selector) == SELECTOR_SHA256, sha256(selector))
    with selector.open(newline="", encoding="utf-8") as handle:
        selected = sorted(int(float(row["cell_id"])) for row in csv.DictReader(handle) if row.get("include") == "1")
    check("selector_44_cells", len(selected) == 44, selected)

    psf = load_npz(psf_npz)
    old_psf = load_npz(old_stage_b / f"psf_{BASE_RUN_ID}.npz")
    same_profile = (
        np.array_equal(psf["cell_id"], old_psf["cell_id"])
        and np.array_equal(psf["profile_edges_deg"], old_psf["profile_edges_deg"])
        and np.array_equal(psf["profile_density"], old_psf["profile_density"])
        and np.array_equal(psf["crab_theta_probability"], old_psf["crab_theta_probability"])
    )
    check("same_crab_theta_weighted_profiles", same_profile, "cell IDs, profile bins/densities, and Crab theta weights are byte-equal arrays")
    method = np.asarray(psf["psf_method"], dtype=str)
    check("psf_method", bool(np.all(method == "double_rayleigh_mixture")), sorted(set(method.tolist())))

    a = np.asarray(psf["double_rayleigh_A"], dtype=np.float64)
    s1 = np.asarray(psf["double_rayleigh_sigma1_deg"], dtype=np.float64)
    s2 = np.asarray(psf["double_rayleigh_sigma2_deg"], dtype=np.float64)
    r_opt = np.asarray(psf["double_rayleigh_r_opt_deg"], dtype=np.float64)
    quality = np.asarray(psf["double_rayleigh_fit_quality"], dtype=str)
    reasons = np.asarray(psf["double_rayleigh_fallback_reason"], dtype=str)
    ok = quality == "ok"
    fallback = np.char.startswith(quality, "fallback:")
    constraints = bool(np.all((a[ok] > 0.0) & (a[ok] < 1.0) & (s1[ok] > 0.0) & (s1[ok] < s2[ok])))
    profile_max = float(np.asarray(psf["profile_edges_deg"], dtype=np.float64)[-1])
    stable_support = bool(np.all((r_opt[ok] > 0.0) & (r_opt[ok] <= profile_max)))
    fallback_reasons = bool(np.all(np.char.str_len(reasons[fallback]) > 0))
    empirical = np.asarray(psf["double_rayleigh_containment_r_opt"], dtype=np.float64)
    empirical_stable = bool(np.all(np.abs(empirical[ok] - TARGET_CONTAINMENT) <= 0.12))
    check("double_rayleigh_constraints", constraints, {"fit_cells": int(np.count_nonzero(ok))})
    check("double_rayleigh_stability_support", stable_support, {"profile_max_deg": profile_max})
    check("double_rayleigh_empirical_containment", empirical_stable, {"tolerance": 0.12})
    check("fallback_reasons_recorded", fallback_reasons and bool(np.all(ok | fallback)), {"fallback_cells": int(np.count_nonzero(fallback))})
    psf_meta = load_json(psf_meta_path)
    target = ((psf_meta.get("psf_comparison") or {}).get("target_containment"))
    check("target_containment", math.isclose(float(target), TARGET_CONTAINMENT, rel_tol=0.0, abs_tol=1.0e-15), target)
    observed_profile = ((psf_meta.get("psf_comparison") or {}).get("observed_data_profile") or {})
    check(
        "observed_psf_not_used",
        observed_profile.get("status") == "not_used_by_double_rayleigh_mixture",
        observed_profile,
    )

    response_meta = load_json(response_meta_path)
    aperture_path = str(
        (response_meta.get("response_aperture_conditioning") or {}).get("psf_npz")
        or response_meta.get("aperture_psf_npz")
        or ""
    )
    check("aperture_conditioned_response", response_meta.get("response_type") == "primary_thrown_aperture_conditioned_response", response_meta.get("response_type"))
    check("response_uses_new_psf", RUN_ID in json.dumps(response_meta) and RUN_ID in aperture_path, aperture_path)

    d_meta = load_json(d_meta_path)
    old_d_meta = load_json(old_stage_d / f"background_{BASE_RUN_ID}_annnorm_metadata.json")
    d_contract_keys = ["background_mode", "method", "background_form", "surface_order", "annulus_normalize_surface", "annulus_default_inner_deg", "annulus_width_deg", "annulus_max_inner_deg"]
    new_contract = d_meta.get("background_model") or {}
    old_contract = old_d_meta.get("background_model") or {}
    contract_same = all(new_contract.get(key) == old_contract.get(key) for key in d_contract_keys)
    check("stage_d_background_contract_unchanged", contract_same, {key: new_contract.get(key) for key in d_contract_keys})
    check("stage_c_reused", str(stage_c.resolve()) == str(Path((d_meta.get("inputs") or {}).get("stage_c_dir", "")).resolve()), (d_meta.get("inputs") or {}).get("stage_c_dir"))

    e = load_npz(e_npz)
    e_meta = load_json(e_meta_path)
    containment = np.asarray(e["containment_r_opt"], dtype=np.float64)
    check("stage_e_containment_one", bool(np.allclose(containment, 1.0, rtol=0.0, atol=0.0)), e_meta.get("containment"))
    check("stage_e_quality", (e_meta.get("quality_gate") or {}).get("status") == "passed", e_meta.get("quality_gate"))

    fit = load_npz(f_npz)
    old_fit = load_npz(old_stage_f / f"fit_{BASE_RUN_ID}.npz")
    fit_ids = np.asarray(fit["cell_id"], dtype=np.int64).tolist()
    check("stage_f_selector_unchanged", fit_ids == selected, fit_ids)
    exposure_same = np.array_equal(np.asarray(fit["theta_exposure_sec"]), np.asarray(old_fit["theta_exposure_sec"]))
    check("stage_f_exposure_unchanged", exposure_same, "theta_exposure_sec arrays are exactly equal")
    f_meta = load_json(f_meta_path)
    check("stage_f_fit", (f_meta.get("quality") or {}).get("fit_status") == "passed", f_meta.get("preferred_fit"))

    g_meta = load_json(g_meta_path)
    old_g_meta = load_json(old_stage_g / f"sed_points_{BASE_RUN_ID}_metadata.json")
    check("stage_g_groupings_unchanged", g_meta.get("groupings") == old_g_meta.get("groupings"), g_meta.get("groupings"))
    check("stage_g_quality", (g_meta.get("quality") or {}).get("status") == "passed", g_meta.get("quality"))

    if args.require_report:
        report = REPO_ROOT / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_B_report.html"
        report_text = report.read_text(encoding="utf-8") if report.exists() else ""
        check(
            "final_report",
            report.exists()
            and "Double-Rayleigh PSF Contract" in report_text
            and "Old Rayleigh vs New Double-Rayleigh Scheme B" in report_text
            and "Original large-pull cells" in report_text,
            str(report),
        )

    write_and_fail(checks)


def write_and_fail(checks: list[dict[str, Any]]) -> None:
    output = REPO_ROOT / f"apply/report/assets/{RUN_ID.replace('_', '-')}/scheme_B_double_rayleigh_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    status = "passed" if checks and all(row["status"] == "passed" for row in checks) else "failed"
    output.write_text(json.dumps({"status": status, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(f"Validation {status}: {output}")
    if status != "passed":
        failed = [row["name"] for row in checks if row["status"] != "passed"]
        raise SystemExit(f"Double-Rayleigh Scheme B validation failed: {failed}")


if __name__ == "__main__":
    main()
