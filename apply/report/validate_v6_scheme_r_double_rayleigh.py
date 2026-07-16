#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


REPO_ROOT = Path(os.environ.get("V6_VALIDATION_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
PSF_RUN_ID = f"{BASE_RUN_ID}_double_rayleigh"
EXPERIMENT_ID = f"{PSF_RUN_ID}_scheme_R_fixed712979"
SOURCE_RUN_ID = "v6_64748_nhit100_highEplus1_split56"
TARGET_CONTAINMENT = 0.7129790300890827
SELECTOR_SHA256 = "c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6"
M2_TO_CM2 = 1.0e4

SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{BASE_RUN_ID}_fit.csv"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
ASSET_DIR = Path(
    os.environ.get(
        "V6_VALIDATION_ASSET_DIR",
        REPO_ROOT / "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R",
    )
).resolve()
VALIDATION_JSON = ASSET_DIR / "scheme_R_double_rayleigh_validation.json"
COMPARISON_JSON = ASSET_DIR / "scheme_R_double_rayleigh_comparison.json"
REPORT_VALIDATION_JSON = ASSET_DIR / "report_validation.json"
REPORT = REPO_ROOT / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_report.html"

NOMINAL_RESPONSE = REPO_ROOT / f"apply/output/stage_a_{SOURCE_RUN_ID}/response_2d_{SOURCE_RUN_ID}.npz"
NOMINAL_RESPONSE_META = NOMINAL_RESPONSE.with_name(f"response_2d_{SOURCE_RUN_ID}_metadata.json")
FIXED_RESPONSE_ROOT = REPO_ROOT / f"apply/output/stage_e_{BASE_RUN_ID}_fixed712979_rayleigh_annnorm"
FIXED_RESPONSE = FIXED_RESPONSE_ROOT / f"response_{BASE_RUN_ID}_fixed712979_rayleigh.npz"
FIXED_RESPONSE_META = FIXED_RESPONSE_ROOT / f"response_{BASE_RUN_ID}_fixed712979_rayleigh_metadata.json"

STAGE_B = REPO_ROOT / f"apply/output/stage_b_{PSF_RUN_ID}/runs/{PSF_RUN_ID}_stage_b_psf"
PSF_NPZ = STAGE_B / f"psf_{PSF_RUN_ID}.npz"
PSF_META = STAGE_B / f"psf_{PSF_RUN_ID}_metadata.json"
STAGE_C = REPO_ROOT / f"apply/output/stage_c_{SOURCE_RUN_ID}/runs/{SOURCE_RUN_ID}_stage_c_halfyear"
STAGE_D = REPO_ROOT / f"apply/output/stage_d_{PSF_RUN_ID}_annnorm/runs/{PSF_RUN_ID}_stage_d_annnorm"
STAGE_D_NPZ = STAGE_D / f"background_{PSF_RUN_ID}_annnorm.npz"
STAGE_D_META = STAGE_D / f"background_{PSF_RUN_ID}_annnorm_metadata.json"
STAGE_E = REPO_ROOT / f"apply/output/stage_e_{PSF_RUN_ID}_containment1_annnorm/runs/{PSF_RUN_ID}_stage_e_containment1_annnorm"
STAGE_E_NPZ = STAGE_E / f"signal_{PSF_RUN_ID}_containment1_annnorm.npz"
STAGE_E_META = STAGE_E / f"signal_{PSF_RUN_ID}_containment1_annnorm_metadata.json"

STAGE_F = REPO_ROOT / f"apply/output/stage_f_{EXPERIMENT_ID}/runs/{EXPERIMENT_ID}_stage_f"
STAGE_F_NPZ = STAGE_F / f"fit_{EXPERIMENT_ID}.npz"
STAGE_F_META = STAGE_F / f"fit_{EXPERIMENT_ID}_metadata.json"
STAGE_G = REPO_ROOT / f"apply/output/stage_g_{EXPERIMENT_ID}/runs/{EXPERIMENT_ID}_stage_g"
STAGE_G_NPZ = STAGE_G / f"sed_points_{EXPERIMENT_ID}.npz"
STAGE_G_META = STAGE_G / f"sed_points_{EXPERIMENT_ID}_metadata.json"
STAGE_G_SUMMARY = STAGE_G / f"sed_points_{EXPERIMENT_ID}_summary.json"

R1_FIT_ROOT = REPO_ROOT / f"apply/output/stage_f_{BASE_RUN_ID}_scheme_R_fixed712979/runs/{BASE_RUN_ID}_stage_f_scheme_R_fixed712979"
R1_FIT_NPZ = R1_FIT_ROOT / f"fit_{BASE_RUN_ID}_scheme_R_fixed712979.npz"
R1_FIT_META = R1_FIT_ROOT / f"fit_{BASE_RUN_ID}_scheme_R_fixed712979_metadata.json"
R1_G_ROOT = REPO_ROOT / f"apply/output/stage_g_{BASE_RUN_ID}_scheme_R_fixed712979/runs/{BASE_RUN_ID}_stage_g_scheme_R_fixed712979"
R1_G_META = R1_G_ROOT / f"sed_points_{BASE_RUN_ID}_scheme_R_fixed712979_metadata.json"

B2_RESPONSE = REPO_ROOT / f"apply/output/stage_a_{PSF_RUN_ID}_aperture_conditioned/response_2d_{PSF_RUN_ID}_aperture_conditioned.npz"
B2_FIT_ROOT = REPO_ROOT / f"apply/output/stage_f_{PSF_RUN_ID}/runs/{PSF_RUN_ID}_stage_f"
B2_FIT_NPZ = B2_FIT_ROOT / f"fit_{PSF_RUN_ID}.npz"
B2_FIT_META = B2_FIT_ROOT / f"fit_{PSF_RUN_ID}_metadata.json"
B2_G_ROOT = REPO_ROOT / f"apply/output/stage_g_{PSF_RUN_ID}/runs/{PSF_RUN_ID}_stage_g"
B2_G_META = B2_G_ROOT / f"sed_points_{PSF_RUN_ID}_metadata.json"


class ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        source = dict(attrs).get("src")
        if source:
            self.sources.append(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the isolated v6 Scheme R double-Rayleigh experiment.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-report", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def exact_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def same_path(value: Any, expected: Path) -> bool:
    if not value:
        return False
    return Path(str(value)).resolve() == expected.resolve()


def image_validation(report: Path) -> dict[str, Any]:
    parser = ImageParser()
    parser.feed(report.read_text(encoding="utf-8"))
    missing = [source for source in parser.sources if not (report.parent / source).resolve().exists()]
    return {"image_refs": parser.sources, "missing_image_refs": missing, "image_ref_count": len(parser.sources)}


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
    for idx, (low, high) in enumerate(zip(loge_edges[:-1], loge_edges[1:])):
        loge = 0.5 * (high - low) * nodes + 0.5 * (high + low)
        energy_tev = np.power(10.0, loge) / 1000.0
        integrand = loglog_interp(energy_tev, energy, flux) * math.log(10.0) * energy_tev
        out[idx] = 0.5 * (high - low) * float(np.sum(weights * integrand))
    return out


def official_expected_counts(response: dict[str, np.ndarray], fit: dict[str, np.ndarray]) -> np.ndarray:
    flux_integral = integrate_pass5(np.asarray(response["logE_true_edges"], dtype=np.float64))
    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    fit_ids = np.asarray(fit["cell_id"], dtype=np.int64)
    index_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(response_ids)}
    indices = np.asarray([index_by_cell[int(cell_id)] for cell_id in fit_ids], dtype=np.int64)
    return M2_TO_CM2 * np.einsum(
        "bet,e,t->b",
        np.asarray(response["a_eff"], dtype=np.float64)[indices],
        flux_integral,
        np.asarray(fit["theta_exposure_sec"], dtype=np.float64),
    )


def branch_record(
    label: str,
    branch_id: str,
    response_path: Path,
    fit_path: Path,
    fit_meta_path: Path,
    response_contract: str,
) -> tuple[dict[str, Any], dict[int, float]]:
    response = load_npz(response_path)
    fit = load_npz(fit_path)
    metadata = load_json(fit_meta_path)
    logpar = (metadata.get("fits") or {}).get("logpar_conservative") or {}
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
        "branch_id": branch_id,
        "response_contract": response_contract,
        "response_path": str(response_path),
        "response_sha256": sha256(response_path),
        "fit_path": str(fit_path),
        "fit_sha256": sha256(fit_path),
        "fit_metadata_sha256": sha256(fit_meta_path),
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


def build_comparison(selected: list[int]) -> dict[str, Any]:
    specs = [
        (
            "Scheme R single-Rayleigh",
            f"{BASE_RUN_ID}_scheme_R_fixed712979",
            FIXED_RESPONSE,
            R1_FIT_NPZ,
            R1_FIT_META,
            "single-Rayleigh aperture; Aeff_R=0.7129790300890827*Aeff_nominal",
        ),
        (
            "Scheme B double-Rayleigh",
            PSF_RUN_ID,
            B2_RESPONSE,
            B2_FIT_NPZ,
            B2_FIT_META,
            "double-Rayleigh aperture-conditioned response",
        ),
        (
            "Scheme R double-Rayleigh",
            EXPERIMENT_ID,
            FIXED_RESPONSE,
            STAGE_F_NPZ,
            STAGE_F_META,
            "double-Rayleigh aperture; Aeff_R=0.7129790300890827*Aeff_nominal",
        ),
    ]
    records: list[dict[str, Any]] = []
    pulls_by_branch: dict[str, dict[int, float]] = {}
    for label, branch_id, response, fit, metadata, contract in specs:
        record, pulls = branch_record(label, branch_id, response, fit, metadata, contract)
        records.append(record)
        pulls_by_branch[branch_id] = pulls

    cell_pulls = []
    for cell_id in selected:
        row = {"cell_id": cell_id}
        values = []
        for record in records:
            value = pulls_by_branch[record["branch_id"]].get(cell_id)
            row[record["branch_id"]] = value
            if value is not None:
                values.append(abs(value))
        row["max_abs_pull"] = max(values) if values else None
        cell_pulls.append(row)
    large_pull_cells = sorted(
        (row for row in cell_pulls if finite(row.get("max_abs_pull")) is not None and float(row["max_abs_pull"]) >= 5.0),
        key=lambda row: float(row["max_abs_pull"]),
        reverse=True,
    )
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "target_containment": TARGET_CONTAINMENT,
        "selector": str(SELECTOR),
        "selected_cells": selected,
        "shared_artifacts": {
            "stage_b_psf": {"path": str(PSF_NPZ), "sha256": sha256(PSF_NPZ)},
            "stage_d_background": {"path": str(STAGE_D_NPZ), "sha256": sha256(STAGE_D_NPZ)},
            "stage_e_signal": {"path": str(STAGE_E_NPZ), "sha256": sha256(STAGE_E_NPZ)},
            "fixed_response": {"path": str(FIXED_RESPONSE), "sha256": sha256(FIXED_RESPONSE)},
        },
        "comparison": records,
        "cell_pulls": cell_pulls,
        "large_pull_threshold": 5.0,
        "large_pull_cells": large_pull_cells,
        "input_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }
    if REPORT.exists():
        payload["report_path"] = str(REPORT)
        payload["report_sha256"] = sha256(REPORT)
    return payload


def write_outputs(checks: list[dict[str, Any]], comparison: dict[str, Any] | None, phase: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    status = "passed" if checks and all(row["status"] == "passed" for row in checks) else "failed"
    VALIDATION_JSON.write_text(
        json.dumps({"status": status, "phase": phase, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )
    if comparison is not None:
        COMPARISON_JSON.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Validation {status}: {VALIDATION_JSON}")
    if status != "passed":
        failed = [row["name"] for row in checks if row["status"] != "passed"]
        raise SystemExit(f"Scheme R double-Rayleigh validation failed: {failed}")


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail})

    preflight_required = [
        SELECTOR,
        PASS5_CSV,
        NOMINAL_RESPONSE,
        NOMINAL_RESPONSE_META,
        FIXED_RESPONSE,
        FIXED_RESPONSE_META,
        PSF_NPZ,
        PSF_META,
        STAGE_C / "obs_events_metadata.json",
        STAGE_D_NPZ,
        STAGE_D_META,
        STAGE_E_NPZ,
        STAGE_E_META,
        R1_FIT_NPZ,
        R1_FIT_META,
        R1_G_META,
        B2_RESPONSE,
        B2_FIT_NPZ,
        B2_FIT_META,
        B2_G_META,
    ]
    missing = [str(path) for path in preflight_required if not path.exists() or path.stat().st_size == 0]
    check("required_preflight_artifacts", not missing, {"missing": missing, "count": len(preflight_required)})
    if missing:
        write_outputs(checks, None, "preflight")

    selector_sha = sha256(SELECTOR)
    check("selector_sha256", selector_sha == SELECTOR_SHA256, selector_sha)
    with SELECTOR.open(newline="", encoding="utf-8") as handle:
        selected = sorted(int(float(row["cell_id"])) for row in csv.DictReader(handle) if row.get("include") == "1")
    check("selector_44_cells", len(selected) == 44, selected)

    psf = load_npz(PSF_NPZ)
    psf_meta = load_json(PSF_META)
    method = np.asarray(psf["psf_method"], dtype=str)
    quality = np.asarray(psf["double_rayleigh_fit_quality"], dtype=str)
    a = np.asarray(psf["double_rayleigh_A"], dtype=np.float64)
    sigma1 = np.asarray(psf["double_rayleigh_sigma1_deg"], dtype=np.float64)
    sigma2 = np.asarray(psf["double_rayleigh_sigma2_deg"], dtype=np.float64)
    ok = quality == "ok"
    check("psf_method_double_rayleigh", bool(np.all(method == "double_rayleigh_mixture")), sorted(set(method.tolist())))
    check(
        "double_rayleigh_constraints",
        bool(np.all((a[ok] > 0.0) & (a[ok] < 1.0) & (sigma1[ok] > 0.0) & (sigma1[ok] < sigma2[ok]))),
        {"fit_cells": int(np.count_nonzero(ok)), "fallback_cells": int(np.count_nonzero(~ok))},
    )
    psf_target = ((psf_meta.get("psf_comparison") or {}).get("target_containment"))
    check("target_containment", math.isclose(float(psf_target), TARGET_CONTAINMENT, rel_tol=0.0, abs_tol=1.0e-15), psf_target)

    nominal = load_npz(NOMINAL_RESPONSE)
    fixed = load_npz(FIXED_RESPONSE)
    same_keys = set(nominal) == set(fixed)
    check("fixed_response_keys", same_keys, {"nominal": sorted(nominal), "fixed": sorted(fixed)})
    if same_keys:
        non_aeff_equal = all(exact_array_equal(nominal[key], fixed[key]) for key in nominal if key != "a_eff")
        nominal_aeff = np.asarray(nominal["a_eff"], dtype=np.float64)
        fixed_aeff = np.asarray(fixed["a_eff"], dtype=np.float64)
        mask = np.isfinite(nominal_aeff) & (nominal_aeff != 0.0)
        ratio = fixed_aeff[mask] / nominal_aeff[mask]
        ratio_ok = bool(np.allclose(ratio, TARGET_CONTAINMENT, rtol=2.0e-7, atol=0.0))
        check("fixed_response_non_aeff_unchanged", non_aeff_equal, "all non-a_eff arrays are exactly equal")
        check(
            "fixed_response_aeff_ratio",
            ratio_ok,
            {"target": TARGET_CONTAINMENT, "min": float(np.min(ratio)), "max": float(np.max(ratio))},
        )
    fixed_meta = load_json(FIXED_RESPONSE_META)
    contract = fixed_meta.get("single_application_contract") or {}
    check(
        "fixed_response_single_application",
        contract.get("application_count") == 1
        and math.isclose(float((fixed_meta.get("derived_response") or {}).get("containment")), TARGET_CONTAINMENT, rel_tol=0.0, abs_tol=1.0e-15)
        and contract.get("downstream_containment") == 1.0
        and contract.get("aperture_conditioned_response_used") is False,
        contract,
    )

    d_meta = load_json(STAGE_D_META)
    d_inputs = d_meta.get("inputs") or {}
    check("stage_d_uses_double_rayleigh_psf", same_path(d_inputs.get("psf_npz"), PSF_NPZ), d_inputs.get("psf_npz"))
    check("stage_d_reuses_stage_c", same_path(d_inputs.get("stage_c_dir"), STAGE_C), d_inputs.get("stage_c_dir"))
    e = load_npz(STAGE_E_NPZ)
    e_meta = load_json(STAGE_E_META)
    e_inputs = e_meta.get("inputs") or {}
    check("stage_e_uses_shared_background", same_path(e_inputs.get("background_npz"), STAGE_D_NPZ), e_inputs.get("background_npz"))
    check(
        "stage_e_containment_one",
        bool(np.allclose(np.asarray(e["containment_r_opt"], dtype=np.float64), 1.0, rtol=0.0, atol=0.0)),
        e_meta.get("containment"),
    )

    if args.preflight_only:
        write_outputs(checks, None, "preflight")
        return

    output_required = [
        STAGE_F_NPZ,
        STAGE_F_META,
        STAGE_F / "model_counts_vs_excess.png",
        STAGE_F / "pull_grid_logpar.png",
        STAGE_G_NPZ,
        STAGE_G_META,
        STAGE_G_SUMMARY,
        STAGE_G / "sed_points_ratio.png",
    ]
    output_missing = [str(path) for path in output_required if not path.exists() or path.stat().st_size == 0]
    check("required_experiment_outputs", not output_missing, {"missing": output_missing, "count": len(output_required)})
    if output_missing:
        write_outputs(checks, None, "outputs")

    fit = load_npz(STAGE_F_NPZ)
    f_meta = load_json(STAGE_F_META)
    f_inputs = f_meta.get("inputs") or {}
    check("stage_f_uses_fixed_response", same_path(f_inputs.get("response_npz"), FIXED_RESPONSE), f_inputs.get("response_npz"))
    check("stage_f_reuses_double_rayleigh_signal", same_path(f_inputs.get("signal_npz"), STAGE_E_NPZ), f_inputs.get("signal_npz"))
    check("stage_f_no_aperture_response", "aperture_conditioned" not in str(f_inputs.get("response_npz", "")), f_inputs.get("response_npz"))
    fit_ids = np.asarray(fit["cell_id"], dtype=np.int64).tolist()
    check("stage_f_selector_unchanged", fit_ids == selected, fit_ids)
    r1_fit = load_npz(R1_FIT_NPZ)
    b2_fit = load_npz(B2_FIT_NPZ)
    exposure = np.asarray(fit["theta_exposure_sec"])
    check(
        "stage_f_exposure_unchanged",
        exact_array_equal(exposure, np.asarray(r1_fit["theta_exposure_sec"]))
        and exact_array_equal(exposure, np.asarray(b2_fit["theta_exposure_sec"])),
        "theta_exposure_sec arrays exactly equal R-1R and B-2R",
    )
    check(
        "stage_f_logpar_fit",
        (f_meta.get("quality") or {}).get("fit_status") == "passed"
        and (f_meta.get("preferred_fit") or {}).get("model") == "logpar",
        f_meta.get("preferred_fit"),
    )

    g_meta = load_json(STAGE_G_META)
    g_inputs = g_meta.get("inputs") or {}
    check("stage_g_uses_fixed_response", same_path(g_inputs.get("response_npz"), FIXED_RESPONSE), g_inputs.get("response_npz"))
    check("stage_g_reuses_double_rayleigh_signal", same_path(g_inputs.get("signal_npz"), STAGE_E_NPZ), g_inputs.get("signal_npz"))
    r1_g_meta = load_json(R1_G_META)
    b2_g_meta = load_json(B2_G_META)
    check(
        "stage_g_groupings_unchanged",
        g_meta.get("groupings") == r1_g_meta.get("groupings") == b2_g_meta.get("groupings"),
        g_meta.get("groupings"),
    )
    check("stage_g_quality", (g_meta.get("quality") or {}).get("status") == "passed", g_meta.get("quality"))

    comparison = build_comparison(selected)
    if args.require_report:
        report_text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
        images = image_validation(REPORT) if REPORT.exists() else {"image_refs": [], "missing_image_refs": [str(REPORT)], "image_ref_count": 0}
        check(
            "final_report_sections",
            REPORT.exists()
            and "Scheme R Double-Rayleigh Contract" in report_text
            and "Three-Branch Controlled Comparison" in report_text
            and "Large-Pull Migration" in report_text,
            str(REPORT),
        )
        check("final_report_images", images["image_ref_count"] == 24 and not images["missing_image_refs"], images)
        report_validation = load_json(REPORT_VALIDATION_JSON) if REPORT_VALIDATION_JSON.exists() else {}
        contamination = report_validation.get("metadata_contamination") or {}
        check("metadata_contamination", contamination.get("status") == "passed" and not contamination.get("offenders"), contamination)
        report_validation.update({"experiment_id": EXPERIMENT_ID, "report_path": str(REPORT), "html_image_validation": images})
        REPORT_VALIDATION_JSON.write_text(json.dumps(report_validation, indent=2) + "\n", encoding="utf-8")
        comparison["report_path"] = str(REPORT)
        comparison["report_sha256"] = sha256(REPORT)

    write_outputs(checks, comparison, "report" if args.require_report else "outputs")


if __name__ == "__main__":
    main()
