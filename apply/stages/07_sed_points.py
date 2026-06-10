#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RESPONSE_NPZ = "apply/output/stage_a/response_2d.npz"
DEFAULT_RESPONSE_METADATA = "apply/output/stage_a/response_2d_metadata.json"
DEFAULT_SIGNAL_NPZ = "apply/output/stage_e/current/signal_v1.npz"
DEFAULT_SIGNAL_METADATA = "apply/output/stage_e/current/signal_v1_metadata.json"
DEFAULT_STAGE_F_NPZ = "apply/output/stage_f/current/fit_v1.npz"
DEFAULT_STAGE_F_METADATA = "apply/output/stage_f/current/fit_v1_metadata.json"
DEFAULT_OUTPUT_DIR = "apply/output/stage_g"
DEFAULT_REPORT_HTML = "apply/report/stage_g_report.html"
DEFAULT_REFERENCE_PHI0 = 2.114e-12
DEFAULT_REFERENCE_GAMMA = 2.69
DEFAULT_PIVOT_TEV = 3.0
DEFAULT_EXPECTED_STAGE_F_RUN_ID = "codex_stage_f_cells8to18_diag"
DEFAULT_EXPECTED_PHI0 = 2.599910e-12
DEFAULT_EXPECTED_GAMMA = 2.84847
DEFAULT_EXPECTED_CHI2 = 17.831
DEFAULT_EXPECTED_NDOF = 9
M2_TO_CM2 = 1.0e4
REQUIRED_CELL_IDS = list(range(8, 19))
EXCLUDED_CELL_IDS = list(range(1, 8))

WCDA1_POOL1_TABLE1_SOURCE = {
    "name": "LHAASO-WCDA-1 / Pool-1 Crab Table 1 SED points",
    "paper": "Performance of LHAASO-WCDA and Observation of Crab Nebula as a Standard Candle",
    "doi": "10.1088/1674-1137/ac041b",
    "url": "https://cpc.ihep.ac.cn/article/doi/10.1088/1674-1137/ac041b",
    "note": (
        "Final CPC HTML Table 1 values are used. The arXiv v1 PDF/text has exponent "
        "mismatches for at least the 60-100 and 300-400 Nhit rows, so it is not used "
        "as the numeric authority here."
    ),
}

WCDA1_POOL1_TABLE1_POINTS = [
    {
        "label": "60-100",
        "nhit": "60-100",
        "emed_tev": 0.58,
        "excess": 1438.2,
        "background": 24885.8,
        "significance_sigma": 9.1,
        "dnde": 1.66e-10,
        "dnde_err": 0.20e-10,
    },
    {
        "label": "100-200",
        "nhit": "100-200",
        "emed_tev": 1.1,
        "excess": 1082.7,
        "background": 5202.3,
        "significance_sigma": 15.0,
        "dnde": 2.89e-11,
        "dnde_err": 0.23e-11,
    },
    {
        "label": "200-300",
        "nhit": "200-300",
        "emed_tev": 2.4,
        "excess": 456.2,
        "background": 1376.8,
        "significance_sigma": 12.3,
        "dnde": 4.74e-12,
        "dnde_err": 0.48e-12,
    },
    {
        "label": "300-400",
        "nhit": "300-400",
        "emed_tev": 3.9,
        "excess": 161.2,
        "background": 335.8,
        "significance_sigma": 8.8,
        "dnde": 1.12e-12,
        "dnde_err": 0.17e-12,
    },
    {
        "label": "400-500",
        "nhit": "400-500",
        "emed_tev": 5.9,
        "excess": 60.3,
        "background": 77.7,
        "significance_sigma": 6.8,
        "dnde": 3.54e-13,
        "dnde_err": 0.74e-13,
    },
    {
        "label": "500-800",
        "nhit": "500-800",
        "emed_tev": 12.1,
        "excess": 82.7,
        "background": 45.3,
        "significance_sigma": 12.3,
        "dnde": 6.91e-14,
        "dnde_err": 1.0e-14,
    },
]


@dataclass(frozen=True)
class SedPoint:
    grouping: str
    group_label: str
    cell_ids: List[int]
    n_cells: int
    is_single_cell_point: bool
    n_valid_cells: int
    n0: float
    n0_err: float
    effective_energy_tev: float
    true_energy_p16_tev: float
    true_energy_p50_tev: float
    true_energy_p84_tev: float
    e2_dnde: float
    e2_dnde_err: float
    chi2: float
    ndof: int
    chi2_over_ndof: Optional[float]
    ratio_to_stage_f_pl: float
    ratio_to_stage_f_pl_err: float
    pull_vs_stage_f_pl: Optional[float]
    ratio_to_wcda1_ref: float
    ratio_to_wcda1_ref_err: float
    pull_vs_wcda1_ref: Optional[float]
    observed_excess_total: float
    model_counts_total: float
    expected_stage_f_counts_total: float
    error_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage G diagnostic SED points for a Stage F Crab cell baseline.")
    parser.add_argument("--response-npz", type=str, default=DEFAULT_RESPONSE_NPZ)
    parser.add_argument("--response-metadata", type=str, default=DEFAULT_RESPONSE_METADATA)
    parser.add_argument("--signal-npz", type=str, default=DEFAULT_SIGNAL_NPZ)
    parser.add_argument("--signal-metadata", type=str, default=DEFAULT_SIGNAL_METADATA)
    parser.add_argument("--stage-f-npz", type=str, default=DEFAULT_STAGE_F_NPZ)
    parser.add_argument("--stage-f-metadata", type=str, default=DEFAULT_STAGE_F_METADATA)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--report-html", type=str, default=DEFAULT_REPORT_HTML)

    parser.add_argument("--pivot-tev", type=float, default=DEFAULT_PIVOT_TEV)
    parser.add_argument("--reference-phi0", type=float, default=DEFAULT_REFERENCE_PHI0)
    parser.add_argument("--reference-gamma", type=float, default=DEFAULT_REFERENCE_GAMMA)
    parser.add_argument("--energy-quadrature-points", type=int, default=64)
    parser.add_argument("--expected-stage-f-run-id", type=str, default=DEFAULT_EXPECTED_STAGE_F_RUN_ID)
    parser.add_argument("--expected-stage-f-phi0", type=float, default=DEFAULT_EXPECTED_PHI0)
    parser.add_argument("--expected-stage-f-gamma", type=float, default=DEFAULT_EXPECTED_GAMMA)
    parser.add_argument("--expected-stage-f-chi2", type=float, default=DEFAULT_EXPECTED_CHI2)
    parser.add_argument("--expected-stage-f-ndof", type=int, default=DEFAULT_EXPECTED_NDOF)
    parser.add_argument(
        "--skip-expected-stage-f-validation",
        action="store_true",
        default=False,
        help="Do not compare Stage F run id or fitted PL values to frozen expected constants.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="cells8to18_stageg_diag",
        help="Name recorded in Stage G metadata/report for the Stage F cell baseline.",
    )
    parser.add_argument(
        "--required-cell-ids",
        type=str,
        default=",".join(str(v) for v in REQUIRED_CELL_IDS),
        help="Comma-separated Stage F cell ids that must be present, in order. Empty disables exact-cell validation.",
    )
    parser.add_argument(
        "--excluded-cell-ids",
        type=str,
        default=",".join(str(v) for v in EXCLUDED_CELL_IDS),
        help="Comma-separated cell ids that must not appear in Stage F or any Stage G point. Empty disables this check.",
    )
    parser.add_argument("--validation-rtol", type=float, default=1.0e-4)
    parser.add_argument("--model-counts-rtol", type=float, default=1.0e-8)

    parser.add_argument("--npz-name", type=str, default="sed_points_v1.npz")
    parser.add_argument("--metadata-name", type=str, default="sed_points_v1_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="sed_points_v1_summary.csv")
    parser.add_argument("--summary-json-name", type=str, default="sed_points_v1_summary.json")
    parser.add_argument("--summary-md-name", type=str, default="sed_points_v1_summary.md")
    return parser.parse_args()


def parse_cell_ids(value: str) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def make_default_run_id() -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"slurm_{slurm_job_id}"
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not value:
        raise ValueError("--run-id cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("--run-id may only contain letters, digits, dots, underscores, and hyphens")
    if value in {".", ".."}:
        raise ValueError(f"Invalid --run-id: {value!r}")
    return value


def prepare_run_output_dir(output_root: Path, run_id: str, *, overwrite_run_dir: bool) -> Path:
    run_dir = output_root / "runs" / run_id
    if run_dir.exists():
        if overwrite_run_dir:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"Stage G run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def replace_path_atomic(target: Path, replacement: Path) -> None:
    backup = target.with_name(f".{target.name}.old")
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    if target.exists() or target.is_symlink():
        target.replace(backup)
    replacement.replace(target)
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def copytree_atomic(source: Path, target: Path) -> None:
    tmp = target.with_name(f".{target.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        if tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()
    shutil.copytree(source, tmp)
    replace_path_atomic(target, tmp)


def symlink_atomic(link_path: Path, target: Path) -> None:
    tmp = link_path.with_name(f".{link_path.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    replace_path_atomic(link_path, tmp)


def promote_successful_run(output_root: Path, run_dir: Path) -> None:
    current = output_root / "current"
    latest = output_root / "latest"
    try:
        current_tmp = output_root / ".current.tmp"
        if current_tmp.exists() or current_tmp.is_symlink():
            if current_tmp.is_dir() and not current_tmp.is_symlink():
                shutil.rmtree(current_tmp)
            else:
                current_tmp.unlink()
        current_tmp.symlink_to(run_dir)
        replace_path_atomic(current, current_tmp)
    except OSError:
        copytree_atomic(run_dir, current)
    try:
        symlink_atomic(latest, run_dir)
    except OSError:
        latest.write_text(str(run_dir) + "\n", encoding="utf-8")


def parse_interval(label: str) -> Tuple[Optional[float], Optional[float]]:
    label = label.strip()
    if label.startswith("[") and label.endswith(")"):
        low, high = label[1:-1].split(",", 1)
        return float(low), float(high)
    if label.startswith("<"):
        return None, float(label[1:])
    if label.startswith(">="):
        return float(label[2:]), None
    raise ValueError(f"Unsupported interval label: {label}")


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{label} NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def require_arrays(values: Dict[str, np.ndarray], required: Sequence[str], label: str) -> None:
    missing = set(required) - set(values)
    if missing:
        raise ValueError(f"{label} is missing arrays: {sorted(missing)}")


def cell_tuples(values: Dict[str, np.ndarray]) -> List[Tuple[int, str, str]]:
    return [
        (int(cell_id), str(nhit), str(pred))
        for cell_id, nhit, pred in zip(values["cell_id"], values["nhit_bin"], values["predE_bin"])
    ]


def filter_by_indices(values: Dict[str, np.ndarray], indices: Sequence[int], n_cell: int) -> Dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    filtered: Dict[str, np.ndarray] = {}
    for key, value in values.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n_cell:
            filtered[key] = arr[idx].copy()
        else:
            filtered[key] = arr.copy()
    return filtered


def align_to_stage_f_cells(
    response: Dict[str, np.ndarray],
    signal: Dict[str, np.ndarray],
    stage_f: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    stage_f_cells = cell_tuples(stage_f)
    response_cells = cell_tuples(response)
    signal_cells = cell_tuples(signal)
    response_index = {cell[0]: idx for idx, cell in enumerate(response_cells)}
    signal_index = {cell[0]: idx for idx, cell in enumerate(signal_cells)}

    response_indices: List[int] = []
    signal_indices: List[int] = []
    for cell in stage_f_cells:
        cell_id = cell[0]
        if cell_id not in response_index:
            raise ValueError(f"Stage F cell {cell_id} is missing from Stage A response")
        if cell_id not in signal_index:
            raise ValueError(f"Stage F cell {cell_id} is missing from Stage E signal")
        if response_cells[response_index[cell_id]] != cell:
            raise ValueError(f"Stage A label mismatch for Stage F cell {cell_id}")
        if signal_cells[signal_index[cell_id]] != cell:
            raise ValueError(f"Stage E label mismatch for Stage F cell {cell_id}")
        response_indices.append(response_index[cell_id])
        signal_indices.append(signal_index[cell_id])

    return (
        filter_by_indices(response, response_indices, len(response_cells)),
        filter_by_indices(signal, signal_indices, len(signal_cells)),
    )


def close_enough(actual: float, expected: float, *, rtol: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=float(rtol), abs_tol=abs(float(expected)) * float(rtol))


def validate_inputs(
    *,
    args: argparse.Namespace,
    response: Dict[str, np.ndarray],
    signal: Dict[str, np.ndarray],
    stage_f: Dict[str, np.ndarray],
    response_metadata: Dict[str, object],
    signal_metadata: Dict[str, object],
    stage_f_metadata: Dict[str, object],
    stage_f_model_counts_recomputed: np.ndarray,
) -> Dict[str, object]:
    required_cell_ids = parse_cell_ids(str(args.required_cell_ids))
    excluded_cell_ids = parse_cell_ids(str(args.excluded_cell_ids))
    require_arrays(
        response,
        ["a_eff", "logE_true_edges", "theta_true_edges_deg", "cell_id", "nhit_bin", "predE_bin"],
        "Stage A response",
    )
    require_arrays(
        signal,
        [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "containment_r_opt",
            "N_on",
            "B_on",
            "excess",
            "excess_err_stat",
            "excess_err_conservative",
        ],
        "Stage E signal",
    )
    require_arrays(
        stage_f,
        [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "N_on",
            "B_on",
            "excess",
            "excess_err_conservative",
            "containment_r_opt",
            "theta_exposure_sec",
            "pl_conservative_model_counts",
        ],
        "Stage F fit",
    )

    response_expected = {
        "response_type": "primary_thrown_response",
        "absolute_effective_area_status": "available",
        "weighting": "mc_weight_baseline",
    }
    response_mismatches = [
        f"{key}={response_metadata.get(key)!r}, expected {value!r}"
        for key, value in response_expected.items()
        if response_metadata.get(key) != value
    ]
    if response_mismatches:
        raise ValueError("Stage A metadata is not the current production response: " + "; ".join(response_mismatches))

    stage_f_run_id = str(stage_f_metadata.get("run_id") or "")
    enforce_expected = not bool(args.skip_expected_stage_f_validation)
    if enforce_expected and str(args.expected_stage_f_run_id) and stage_f_run_id != str(args.expected_stage_f_run_id):
        raise ValueError(
            f"Stage F run id is {stage_f_run_id!r}; expected {str(args.expected_stage_f_run_id)!r} for Stage G baseline"
        )

    cell_ids = [int(v) for v in stage_f["cell_id"]]
    if required_cell_ids and cell_ids != required_cell_ids:
        raise ValueError(f"Stage G baseline must use cells {required_cell_ids}; got {cell_ids}")
    forbidden = sorted(set(cell_ids) & set(excluded_cell_ids))
    if forbidden:
        raise ValueError(f"Stage G baseline must not include excluded cells {excluded_cell_ids}; got {forbidden}")

    validation = stage_f_metadata.get("validation") if isinstance(stage_f_metadata, dict) else None
    subset = validation.get("cell_subset") if isinstance(validation, dict) else None
    included_from_meta = subset.get("included_cell_ids") if isinstance(subset, dict) else None
    excluded_from_meta = subset.get("excluded_cell_ids") if isinstance(subset, dict) else None
    if required_cell_ids and included_from_meta is not None and [int(v) for v in included_from_meta] != required_cell_ids:
        raise ValueError(f"Stage F metadata included cells are not {required_cell_ids}: {included_from_meta}")
    if excluded_cell_ids and excluded_from_meta is not None:
        missing_excluded = sorted(set(excluded_cell_ids) - {int(v) for v in excluded_from_meta})
        if missing_excluded:
            raise ValueError(f"Stage F metadata does not explicitly exclude expected cells: {missing_excluded}")

    preferred = stage_f_metadata.get("preferred_fit") if isinstance(stage_f_metadata, dict) else None
    if not isinstance(preferred, dict) or preferred.get("model") != "pl" or preferred.get("error_mode") != "conservative":
        raise ValueError(f"Stage G requires Stage F preferred conservative PL fit; got {preferred!r}")
    fits = stage_f_metadata.get("fits") if isinstance(stage_f_metadata, dict) else None
    pl = fits.get("pl_conservative") if isinstance(fits, dict) else None
    if not isinstance(pl, dict):
        raise ValueError("Stage F metadata is missing fits.pl_conservative")
    params = pl.get("parameters") if isinstance(pl.get("parameters"), dict) else {}
    phi0 = finite_float(params.get("phi0"))
    gamma = finite_float(params.get("gamma"))
    chi2 = finite_float(pl.get("chi2"))
    ndof = int(pl.get("ndof")) if pl.get("ndof") is not None else None
    if phi0 is None:
        raise ValueError("Stage F PL phi0 is missing")
    if gamma is None:
        raise ValueError("Stage F PL gamma is missing")
    if chi2 is None:
        raise ValueError("Stage F PL chi2 is missing")
    if ndof is None:
        raise ValueError("Stage F PL ndof is missing")
    if enforce_expected:
        if not close_enough(phi0, float(args.expected_stage_f_phi0), rtol=float(args.validation_rtol)):
            raise ValueError(f"Stage F PL phi0 {phi0!r} does not match expected {args.expected_stage_f_phi0:.6e}")
        if not close_enough(gamma, float(args.expected_stage_f_gamma), rtol=float(args.validation_rtol)):
            raise ValueError(f"Stage F PL gamma {gamma!r} does not match expected {args.expected_stage_f_gamma:.6g}")
        if not close_enough(chi2, float(args.expected_stage_f_chi2), rtol=float(args.validation_rtol)):
            raise ValueError(f"Stage F PL chi2 {chi2!r} does not match expected {args.expected_stage_f_chi2:.6g}")
        if ndof != int(args.expected_stage_f_ndof):
            raise ValueError(f"Stage F PL ndof {ndof!r} does not match expected {args.expected_stage_f_ndof}")

    for name in ["N_on", "B_on", "excess", "excess_err_conservative", "containment_r_opt"]:
        lhs = np.asarray(signal[name])
        rhs = np.asarray(stage_f[name])
        if not np.allclose(lhs, rhs, rtol=1.0e-10, atol=1.0e-8, equal_nan=True):
            raise ValueError(f"Stage E aligned {name} does not match Stage F {name}")

    stage_f_counts = np.asarray(stage_f["pl_conservative_model_counts"], dtype=np.float64)
    if not np.allclose(stage_f_model_counts_recomputed, stage_f_counts, rtol=float(args.model_counts_rtol), atol=1.0e-8):
        max_abs = float(np.nanmax(np.abs(stage_f_model_counts_recomputed - stage_f_counts)))
        max_rel = float(np.nanmax(np.abs((stage_f_model_counts_recomputed - stage_f_counts) / stage_f_counts)))
        raise ValueError(f"Recomputed Stage F PL model counts do not match NPZ: max_abs={max_abs}, max_rel={max_rel}")

    signal_quality = signal_metadata.get("quality_gate") if isinstance(signal_metadata, dict) else None
    contract = signal_metadata.get("stage_d_contract") if isinstance(signal_metadata, dict) else None
    return {
        "baseline": str(args.baseline_name),
        "required_cell_ids": required_cell_ids,
        "excluded_cell_ids": excluded_cell_ids,
        "stage_f_run_id": stage_f_run_id,
        "stage_f_subset_included": included_from_meta,
        "stage_f_subset_excluded": excluded_from_meta,
        "expected_stage_f_validation": {
            "enabled": enforce_expected,
            "expected_run_id": str(args.expected_stage_f_run_id) if enforce_expected else None,
            "rtol": float(args.validation_rtol),
        },
        "stage_f_preferred_fit": preferred,
        "stage_f_pl_parameters_validated": {
            "phi0": phi0,
            "gamma": gamma,
            "chi2": chi2,
            "ndof": ndof,
            "expected_phi0": float(args.expected_stage_f_phi0) if enforce_expected else None,
            "expected_gamma": float(args.expected_stage_f_gamma) if enforce_expected else None,
            "expected_chi2": float(args.expected_stage_f_chi2) if enforce_expected else None,
            "expected_ndof": int(args.expected_stage_f_ndof) if enforce_expected else None,
            "rtol": float(args.validation_rtol) if enforce_expected else None,
        },
        "stage_a_response_type": response_metadata.get("response_type"),
        "stage_a_absolute_effective_area_status": response_metadata.get("absolute_effective_area_status"),
        "stage_a_weighting": response_metadata.get("weighting"),
        "stage_e_quality_status": signal_quality.get("status") if isinstance(signal_quality, dict) else None,
        "stage_e_quality_promotable": signal_quality.get("promotable") if isinstance(signal_quality, dict) else None,
        "background_mode": contract.get("background_mode") if isinstance(contract, dict) else None,
        "background_form": contract.get("background_form") if isinstance(contract, dict) else None,
        "stage_f_model_counts_recomputed_max_abs_diff": float(
            np.nanmax(np.abs(stage_f_model_counts_recomputed - stage_f_counts))
        ),
    }


def pl_flux_tev(E_tev: np.ndarray | float, *, phi0: float, gamma: float, pivot_tev: float) -> np.ndarray:
    ratio = np.asarray(E_tev, dtype=np.float64) / float(pivot_tev)
    return float(phi0) * np.power(ratio, -float(gamma))


def integrate_pl_flux_bins(
    loge_edges: np.ndarray,
    *,
    phi0: float,
    gamma: float,
    pivot_tev: float,
    quadrature_points: int,
) -> np.ndarray:
    if quadrature_points <= 1:
        raise ValueError("--energy-quadrature-points must be greater than 1")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    out = np.zeros(loge_edges.size - 1, dtype=np.float64)
    for idx, (lo, hi) in enumerate(zip(loge_edges[:-1], loge_edges[1:])):
        xs = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        E_tev = np.power(10.0, xs) / 1000.0
        flux = pl_flux_tev(E_tev, phi0=phi0, gamma=gamma, pivot_tev=pivot_tev)
        integrand = flux * math.log(10.0) * E_tev
        out[idx] = 0.5 * (hi - lo) * float(np.sum(weights * integrand))
    return out


def model_counts_from_flux_integral(
    *,
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    flux_integral: np.ndarray,
) -> np.ndarray:
    counts = M2_TO_CM2 * np.einsum("bet,e,t->b", a_eff_m2, flux_integral, theta_exposure_sec)
    return np.asarray(containment, dtype=np.float64) * np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_quantiles_from_loge_bins(
    loge_edges: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> List[float]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return [float("nan") for _ in quantiles]
    cumulative = np.cumsum(weights)
    out: List[float] = []
    for q in quantiles:
        target = float(q) * total
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), weights.size - 1)
        previous = float(cumulative[idx - 1]) if idx > 0 else 0.0
        width = float(loge_edges[idx + 1] - loge_edges[idx])
        if weights[idx] > 0.0:
            frac = min(1.0, max(0.0, (target - previous) / float(weights[idx])))
        else:
            frac = 0.5
        loge = float(loge_edges[idx] + frac * width)
        out.append(float(np.power(10.0, loge) / 1000.0))
    return out


def true_energy_weights_by_bin(
    *,
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    flux_integral: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        raise ValueError("Cannot compute true-energy weights for an empty cell group")
    contribution = (
        np.asarray(containment, dtype=np.float64)[selected, None]
        * M2_TO_CM2
        * np.einsum("bet,t->be", np.asarray(a_eff_m2, dtype=np.float64)[selected], theta_exposure_sec)
        * flux_integral[None, :]
    )
    return np.sum(np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0), axis=0)


def point_to_dict(point: SedPoint) -> Dict[str, object]:
    return {
        "grouping": point.grouping,
        "group_label": point.group_label,
        "cell_ids": point.cell_ids,
        "n_cells": point.n_cells,
        "is_single_cell_point": point.is_single_cell_point,
        "n_valid_cells": point.n_valid_cells,
        "N0": point.n0,
        "N0_err": point.n0_err,
        "effective_energy_tev": point.effective_energy_tev,
        "true_energy_p16_tev": point.true_energy_p16_tev,
        "true_energy_p50_tev": point.true_energy_p50_tev,
        "true_energy_p84_tev": point.true_energy_p84_tev,
        "E2_dnde": point.e2_dnde,
        "E2_dnde_err": point.e2_dnde_err,
        "chi2": point.chi2,
        "ndof": point.ndof,
        "chi2_over_ndof": point.chi2_over_ndof,
        "ratio_to_stage_f_pl": point.ratio_to_stage_f_pl,
        "ratio_to_stage_f_pl_err": point.ratio_to_stage_f_pl_err,
        "pull_vs_stage_f_pl": point.pull_vs_stage_f_pl,
        "ratio_to_full_array_pl_ref": point.ratio_to_wcda1_ref,
        "ratio_to_full_array_pl_ref_err": point.ratio_to_wcda1_ref_err,
        "pull_vs_full_array_pl_ref": point.pull_vs_wcda1_ref,
        "ratio_to_wcda1_ref": point.ratio_to_wcda1_ref,
        "ratio_to_wcda1_ref_err": point.ratio_to_wcda1_ref_err,
        "pull_vs_wcda1_ref": point.pull_vs_wcda1_ref,
        "observed_excess_total": point.observed_excess_total,
        "model_counts_total": point.model_counts_total,
        "expected_stage_f_counts_total": point.expected_stage_f_counts_total,
        "error_mode": point.error_mode,
    }


def pool1_reference_points() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for point in WCDA1_POOL1_TABLE1_POINTS:
        energy = float(point["emed_tev"])
        dnde = float(point["dnde"])
        dnde_err = float(point["dnde_err"])
        e2 = energy * energy
        out.append(
            {
                **point,
                "E2_dnde": e2 * dnde,
                "E2_dnde_err": e2 * dnde_err,
            }
        )
    return out


def build_group_specs(stage_f: Dict[str, np.ndarray]) -> List[Tuple[str, str, np.ndarray]]:
    specs: List[Tuple[str, str, np.ndarray]] = []
    nhit_labels = sorted({str(v) for v in stage_f["nhit_bin"]}, key=interval_key)
    pred_labels = sorted({str(v) for v in stage_f["predE_bin"]}, key=interval_key)
    nhit = np.asarray([str(v) for v in stage_f["nhit_bin"]])
    pred = np.asarray([str(v) for v in stage_f["predE_bin"]])
    for label in nhit_labels:
        specs.append(("nhit", label, nhit == label))
    for label in pred_labels:
        specs.append(("predE", label, pred == label))
    return specs


def fit_sed_points(
    *,
    excluded_cell_ids: Sequence[int],
    stage_f: Dict[str, np.ndarray],
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    loge_edges: np.ndarray,
    frozen_phi0: float,
    frozen_gamma: float,
    pivot_tev: float,
    reference_phi0: float,
    reference_gamma: float,
    quadrature_points: int,
) -> Tuple[List[SedPoint], np.ndarray, np.ndarray, np.ndarray]:
    unit_flux_integral = integrate_pl_flux_bins(
        loge_edges,
        phi0=1.0,
        gamma=frozen_gamma,
        pivot_tev=pivot_tev,
        quadrature_points=quadrature_points,
    )
    frozen_flux_integral = unit_flux_integral * float(frozen_phi0)
    unit_counts = model_counts_from_flux_integral(
        a_eff_m2=a_eff_m2,
        containment=containment,
        theta_exposure_sec=theta_exposure_sec,
        flux_integral=unit_flux_integral,
    )
    frozen_counts = unit_counts * float(frozen_phi0)
    observed = np.asarray(stage_f["excess"], dtype=np.float64)
    errors = np.asarray(stage_f["excess_err_conservative"], dtype=np.float64)
    cell_ids = np.asarray(stage_f["cell_id"], dtype=np.int64)

    points: List[SedPoint] = []
    for grouping, group_label, mask in build_group_specs(stage_f):
        mask = np.asarray(mask, dtype=bool)
        valid = mask & np.isfinite(observed) & np.isfinite(errors) & (errors > 0.0) & np.isfinite(unit_counts) & (unit_counts > 0.0)
        n_valid = int(np.count_nonzero(valid))
        if n_valid <= 0:
            raise ValueError(f"No valid cells for Stage G point {grouping}={group_label}")
        weights = 1.0 / (errors[valid] * errors[valid])
        denominator = float(np.sum(unit_counts[valid] * unit_counts[valid] * weights))
        numerator = float(np.sum(observed[valid] * unit_counts[valid] * weights))
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError(f"Invalid fit denominator for Stage G point {grouping}={group_label}")
        n0 = numerator / denominator
        n0_err = 1.0 / math.sqrt(denominator)
        model = n0 * unit_counts[valid]
        chi2 = float(np.sum(np.square((observed[valid] - model) / errors[valid])))
        ndof = n_valid - 1
        chi2_over_ndof = chi2 / ndof if ndof > 0 else None

        energy_weights = true_energy_weights_by_bin(
            a_eff_m2=a_eff_m2,
            containment=containment,
            theta_exposure_sec=theta_exposure_sec,
            flux_integral=frozen_flux_integral,
            mask=mask,
        )
        e16, e50, e84 = weighted_quantiles_from_loge_bins(loge_edges, energy_weights, [0.16, 0.50, 0.84])
        effective_energy = e50

        stage_f_flux = float(pl_flux_tev(effective_energy, phi0=frozen_phi0, gamma=frozen_gamma, pivot_tev=pivot_tev))
        point_flux = float(pl_flux_tev(effective_energy, phi0=n0, gamma=frozen_gamma, pivot_tev=pivot_tev))
        point_flux_err = float(pl_flux_tev(effective_energy, phi0=n0_err, gamma=frozen_gamma, pivot_tev=pivot_tev))
        reference_flux = float(pl_flux_tev(effective_energy, phi0=reference_phi0, gamma=reference_gamma, pivot_tev=pivot_tev))
        e2 = effective_energy * effective_energy
        e2_dnde = e2 * point_flux
        e2_dnde_err = e2 * point_flux_err
        stage_f_e2 = e2 * stage_f_flux
        reference_e2 = e2 * reference_flux
        ratio_stage_f = e2_dnde / stage_f_e2 if stage_f_e2 > 0.0 else float("nan")
        ratio_stage_f_err = e2_dnde_err / stage_f_e2 if stage_f_e2 > 0.0 else float("nan")
        ratio_reference = e2_dnde / reference_e2 if reference_e2 > 0.0 else float("nan")
        ratio_reference_err = e2_dnde_err / reference_e2 if reference_e2 > 0.0 else float("nan")
        pull_stage_f = (e2_dnde - stage_f_e2) / e2_dnde_err if e2_dnde_err > 0.0 else None
        pull_reference = (e2_dnde - reference_e2) / e2_dnde_err if e2_dnde_err > 0.0 else None

        selected_cell_ids = [int(v) for v in cell_ids[mask]]
        forbidden = sorted(set(selected_cell_ids) & {int(v) for v in excluded_cell_ids})
        if forbidden:
            raise ValueError(f"Stage G point {grouping}={group_label} includes excluded cells {forbidden}")
        points.append(
            SedPoint(
                grouping=grouping,
                group_label=group_label,
                cell_ids=selected_cell_ids,
                n_cells=len(selected_cell_ids),
                is_single_cell_point=len(selected_cell_ids) == 1,
                n_valid_cells=n_valid,
                n0=n0,
                n0_err=n0_err,
                effective_energy_tev=effective_energy,
                true_energy_p16_tev=e16,
                true_energy_p50_tev=e50,
                true_energy_p84_tev=e84,
                e2_dnde=e2_dnde,
                e2_dnde_err=e2_dnde_err,
                chi2=chi2,
                ndof=ndof,
                chi2_over_ndof=chi2_over_ndof,
                ratio_to_stage_f_pl=ratio_stage_f,
                ratio_to_stage_f_pl_err=ratio_stage_f_err,
                pull_vs_stage_f_pl=pull_stage_f,
                ratio_to_wcda1_ref=ratio_reference,
                ratio_to_wcda1_ref_err=ratio_reference_err,
                pull_vs_wcda1_ref=pull_reference,
                observed_excess_total=float(np.sum(observed[mask])),
                model_counts_total=float(np.sum(n0 * unit_counts[mask])),
                expected_stage_f_counts_total=float(np.sum(frozen_counts[mask])),
                error_mode="conservative",
            )
        )

    return points, unit_counts, frozen_counts, frozen_flux_integral


def format_float(value: object, digits: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def format_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


def format_cell_list(values: Sequence[object]) -> str:
    ids = [int(v) for v in values]
    if not ids:
        return "n/a"
    return ",".join(str(v) for v in ids)


def write_summary_csv(path: Path, points: Sequence[SedPoint]) -> None:
    fieldnames = [
        "grouping",
        "group_label",
        "cell_ids",
        "n_cells",
        "is_single_cell_point",
        "N0",
        "N0_err",
        "effective_energy_tev",
        "true_energy_p16_tev",
        "true_energy_p84_tev",
        "E2_dnde",
        "E2_dnde_err",
        "chi2",
        "ndof",
        "chi2_over_ndof",
        "ratio_to_stage_f_pl",
        "ratio_to_stage_f_pl_err",
        "ratio_to_full_array_pl_ref",
        "ratio_to_full_array_pl_ref_err",
        "observed_excess_total",
        "model_counts_total",
        "expected_stage_f_counts_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row = point_to_dict(point)
            row["cell_ids"] = ";".join(str(v) for v in point.cell_ids)
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_pool1_reference_csv(path: Path) -> None:
    fieldnames = [
        "label",
        "nhit",
        "emed_tev",
        "excess",
        "background",
        "significance_sigma",
        "dnde",
        "dnde_err",
        "E2_dnde",
        "E2_dnde_err",
        "source_doi",
        "source_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in pool1_reference_points():
            row = dict(point)
            row["source_doi"] = WCDA1_POOL1_TABLE1_SOURCE["doi"]
            row["source_url"] = WCDA1_POOL1_TABLE1_SOURCE["url"]
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary_md(path: Path, metadata: Dict[str, object], points: Sequence[SedPoint]) -> None:
    frozen = metadata["frozen_spectrum"]  # type: ignore[index]
    validation = metadata["validation"]  # type: ignore[index]
    reference = metadata["reference_spectrum"]  # type: ignore[index]
    pool1_reference = metadata.get("wcda1_pool1_reference", {})
    baseline_name = validation.get("baseline", "diagnostic") if isinstance(validation, dict) else "diagnostic"
    required_ids = validation.get("required_cell_ids", []) if isinstance(validation, dict) else []
    excluded_ids = validation.get("excluded_cell_ids", []) if isinstance(validation, dict) else []
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage G Diagnostic SED Points Summary\n\n")
        f.write(f"- Run id: `{metadata['run_id']}`\n")
        f.write(f"- Scope: diagnostic `{baseline_name}` SED points; not a publication baseline.\n")
        f.write(f"- Stage F run: `{validation['stage_f_run_id']}`\n")
        f.write(f"- Included cells: `{format_cell_list(required_ids)}`\n")
        f.write(
            f"- Frozen PL: phi0={format_float(frozen['phi0'], 6)}, "
            f"gamma={format_float(frozen['gamma'], 6)}, pivot={format_float(frozen['pivot_tev'], 4)} TeV\n"
        )
        f.write(
            f"- Full-array PL reference: {reference['name']}, "
            f"phi0={format_float(reference['phi0'], 6)}, gamma={format_float(reference['gamma'], 4)}\n"
        )
        if isinstance(pool1_reference, dict):
            f.write(
                f"- Pool-1 reference points: {pool1_reference.get('name', 'WCDA-1 Pool-1 Table 1')}; "
                f"DOI `{pool1_reference.get('doi', 'n/a')}`\n"
            )
        f.write(f"- Excluded cells: `{format_cell_list(excluded_ids)}`\n\n")
        for grouping in ["nhit", "predE"]:
            f.write(f"## {grouping} points\n\n")
            f.write(
                "| group | cells | E_eff [TeV] | E2 dN/dE | err | N0 | N0 err | chi2/ndof | "
                "ratio StageF | ratio full-array PL | single cell |\n"
            )
            f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
            for point in points:
                if point.grouping != grouping:
                    continue
                chi = f"{format_float(point.chi2, 4)}/{point.ndof}" if point.ndof > 0 else f"{format_float(point.chi2, 4)}/0"
                f.write(
                    f"| {point.group_label} | {','.join(str(v) for v in point.cell_ids)} | "
                    f"{format_float(point.effective_energy_tev, 5)} | {format_float(point.e2_dnde, 6)} | "
                    f"{format_float(point.e2_dnde_err, 4)} | {format_float(point.n0, 6)} | "
                    f"{format_float(point.n0_err, 4)} | {chi} | {format_float(point.ratio_to_stage_f_pl, 5)} | "
                    f"{format_float(point.ratio_to_wcda1_ref, 5)} | {point.is_single_cell_point} |\n"
                )
            f.write("\n")
        f.write("## WCDA-1 Pool-1 Table 1 reference points\n\n")
        f.write("| Nhit | Emed [TeV] | dN/dE | err | E2 dN/dE | E2 err | significance |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for point in pool1_reference_points():
            f.write(
                f"| {point['nhit']} | {format_float(point['emed_tev'], 5)} | "
                f"{format_float(point['dnde'], 6)} | {format_float(point['dnde_err'], 4)} | "
                f"{format_float(point['E2_dnde'], 6)} | {format_float(point['E2_dnde_err'], 4)} | "
                f"{format_float(point['significance_sigma'], 4)} |\n"
            )
        f.write("\n")


def write_npz(path: Path, points: Sequence[SedPoint], stage_f: Dict[str, np.ndarray], frozen_counts: np.ndarray) -> None:
    rows = [point_to_dict(point) for point in points]
    payload: Dict[str, np.ndarray] = {
        "grouping": np.asarray([row["grouping"] for row in rows], dtype="U16"),
        "group_label": np.asarray([row["group_label"] for row in rows], dtype="U32"),
        "cell_ids": np.asarray([";".join(str(v) for v in row["cell_ids"]) for row in rows], dtype="U64"),
        "n_cells": np.asarray([row["n_cells"] for row in rows], dtype=np.int32),
        "is_single_cell_point": np.asarray([row["is_single_cell_point"] for row in rows], dtype=bool),
        "N0": np.asarray([row["N0"] for row in rows], dtype=np.float64),
        "N0_err": np.asarray([row["N0_err"] for row in rows], dtype=np.float64),
        "effective_energy_tev": np.asarray([row["effective_energy_tev"] for row in rows], dtype=np.float64),
        "true_energy_p16_tev": np.asarray([row["true_energy_p16_tev"] for row in rows], dtype=np.float64),
        "true_energy_p50_tev": np.asarray([row["true_energy_p50_tev"] for row in rows], dtype=np.float64),
        "true_energy_p84_tev": np.asarray([row["true_energy_p84_tev"] for row in rows], dtype=np.float64),
        "E2_dnde": np.asarray([row["E2_dnde"] for row in rows], dtype=np.float64),
        "E2_dnde_err": np.asarray([row["E2_dnde_err"] for row in rows], dtype=np.float64),
        "chi2": np.asarray([row["chi2"] for row in rows], dtype=np.float64),
        "ndof": np.asarray([row["ndof"] for row in rows], dtype=np.int32),
        "chi2_over_ndof": np.asarray(
            [row["chi2_over_ndof"] if row["chi2_over_ndof"] is not None else np.nan for row in rows],
            dtype=np.float64,
        ),
        "ratio_to_stage_f_pl": np.asarray([row["ratio_to_stage_f_pl"] for row in rows], dtype=np.float64),
        "ratio_to_stage_f_pl_err": np.asarray([row["ratio_to_stage_f_pl_err"] for row in rows], dtype=np.float64),
        "ratio_to_full_array_pl_ref": np.asarray(
            [row["ratio_to_full_array_pl_ref"] for row in rows],
            dtype=np.float64,
        ),
        "ratio_to_full_array_pl_ref_err": np.asarray(
            [row["ratio_to_full_array_pl_ref_err"] for row in rows],
            dtype=np.float64,
        ),
        "ratio_to_wcda1_ref": np.asarray([row["ratio_to_wcda1_ref"] for row in rows], dtype=np.float64),
        "ratio_to_wcda1_ref_err": np.asarray([row["ratio_to_wcda1_ref_err"] for row in rows], dtype=np.float64),
        "pool1_reference_label": np.asarray([str(row["label"]) for row in pool1_reference_points()], dtype="U32"),
        "pool1_reference_emed_tev": np.asarray([row["emed_tev"] for row in pool1_reference_points()], dtype=np.float64),
        "pool1_reference_dnde": np.asarray([row["dnde"] for row in pool1_reference_points()], dtype=np.float64),
        "pool1_reference_dnde_err": np.asarray([row["dnde_err"] for row in pool1_reference_points()], dtype=np.float64),
        "pool1_reference_E2_dnde": np.asarray([row["E2_dnde"] for row in pool1_reference_points()], dtype=np.float64),
        "pool1_reference_E2_dnde_err": np.asarray(
            [row["E2_dnde_err"] for row in pool1_reference_points()],
            dtype=np.float64,
        ),
        "stage_f_cell_id": np.asarray(stage_f["cell_id"], dtype=np.int32),
        "stage_f_pl_model_counts": np.asarray(frozen_counts, dtype=np.float64),
    }
    np.savez_compressed(path, **payload)


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def sed_curve(E_tev: np.ndarray, *, phi0: float, gamma: float, pivot_tev: float) -> np.ndarray:
    return E_tev * E_tev * pl_flux_tev(E_tev, phi0=phi0, gamma=gamma, pivot_tev=pivot_tev)


def interpolate_pool1_e2_sed(E_tev: np.ndarray | float) -> np.ndarray:
    points = pool1_reference_points()
    x = np.asarray([float(p["emed_tev"]) for p in points], dtype=np.float64)
    y = np.asarray([float(p["E2_dnde"]) for p in points], dtype=np.float64)
    target = np.asarray(E_tev, dtype=np.float64)
    valid = (x > 0.0) & (y > 0.0)
    out = np.full(target.shape, np.nan, dtype=np.float64)
    if np.count_nonzero(valid) < 2:
        return out
    lx = np.log10(x[valid])
    ly = np.log10(y[valid])
    lt = np.log10(target)
    inside = (target >= np.nanmin(x[valid])) & (target <= np.nanmax(x[valid])) & np.isfinite(lt)
    out[inside] = np.power(10.0, np.interp(lt[inside], lx, ly))
    return out


def plot_sed_points(
    points: Sequence[SedPoint],
    path: Path,
    *,
    baseline_name: str,
    frozen_phi0: float,
    frozen_gamma: float,
    reference_phi0: float,
    reference_gamma: float,
    pivot_tev: float,
) -> None:
    plt = setup_matplotlib()
    energies = np.asarray([p.effective_energy_tev for p in points if p.e2_dnde > 0.0], dtype=np.float64)
    if energies.size:
        emin = max(0.2, float(np.nanmin(energies)) / 1.8)
        emax = min(200.0, float(np.nanmax(energies)) * 1.8)
    else:
        emin, emax = 0.3, 80.0
    x = np.geomspace(emin, emax, 240)
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.plot(x, sed_curve(x, phi0=frozen_phi0, gamma=frozen_gamma, pivot_tev=pivot_tev), color="#1f77b4", lw=2.0, label="Stage F frozen PL")
    ax.plot(
        x,
        sed_curve(x, phi0=reference_phi0, gamma=reference_gamma, pivot_tev=pivot_tev),
        color="#555555",
        lw=1.8,
        ls="--",
        label="1LHAASO WCDA full-array PL",
    )
    pool1_points = pool1_reference_points()
    ax.errorbar(
        [float(p["emed_tev"]) for p in pool1_points],
        [float(p["E2_dnde"]) for p in pool1_points],
        yerr=[float(p["E2_dnde_err"]) for p in pool1_points],
        fmt="^",
        color="#7f3fbf",
        ecolor="#7f3fbf",
        capsize=3,
        ms=5,
        lw=0.9,
        label="WCDA-1 Pool-1 Table 1",
    )
    styles = {
        "nhit": {"fmt": "o", "color": "#d62728", "label": "Nhit grouped"},
        "predE": {"fmt": "s", "color": "#2ca02c", "label": "predE grouped"},
    }
    for grouping in ["nhit", "predE"]:
        selected = [p for p in points if p.grouping == grouping and p.e2_dnde > 0.0 and p.effective_energy_tev > 0.0]
        if not selected:
            continue
        ax.errorbar(
            [p.effective_energy_tev for p in selected],
            [p.e2_dnde for p in selected],
            yerr=[p.e2_dnde_err for p in selected],
            capsize=3,
            ms=5,
            lw=0.9,
            **styles[grouping],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective true energy [TeV]")
    ax.set_ylabel(r"$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    ax.set_title(f"Stage G diagnostic SED points, {baseline_name}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ratio_points(points: Sequence[SedPoint], path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.4), sharex=True, constrained_layout=True)
    styles = {
        "nhit": {"fmt": "o", "color": "#d62728", "label": "Nhit grouped"},
        "predE": {"fmt": "s", "color": "#2ca02c", "label": "predE grouped"},
    }
    for grouping in ["nhit", "predE"]:
        selected = [p for p in points if p.grouping == grouping and p.effective_energy_tev > 0.0]
        if not selected:
            continue
        axes[0].errorbar(
            [p.effective_energy_tev for p in selected],
            [p.ratio_to_stage_f_pl for p in selected],
            yerr=[p.ratio_to_stage_f_pl_err for p in selected],
            capsize=3,
            ms=5,
            lw=0.9,
            **styles[grouping],
        )
        axes[1].errorbar(
            [p.effective_energy_tev for p in selected],
            [p.ratio_to_wcda1_ref for p in selected],
            yerr=[p.ratio_to_wcda1_ref_err for p in selected],
            capsize=3,
            ms=5,
            lw=0.9,
            **styles[grouping],
        )
        pool1_interp = interpolate_pool1_e2_sed(np.asarray([p.effective_energy_tev for p in selected], dtype=np.float64))
        ratio_pool1: List[float] = []
        ratio_pool1_err: List[float] = []
        for p, ref in zip(selected, pool1_interp):
            if ref > 0.0 and math.isfinite(float(ref)):
                ratio_pool1.append(p.e2_dnde / float(ref))
                ratio_pool1_err.append(p.e2_dnde_err / float(ref))
            else:
                ratio_pool1.append(float("nan"))
                ratio_pool1_err.append(float("nan"))
        axes[2].errorbar(
            [p.effective_energy_tev for p in selected],
            ratio_pool1,
            yerr=ratio_pool1_err,
            capsize=3,
            ms=5,
            lw=0.9,
            **styles[grouping],
        )
    for ax, ylabel in zip(
        axes,
        ["Point / Stage F PL", "Point / 1LHAASO full-array PL", "Point / WCDA-1 Pool-1 points"],
    ):
        ax.axhline(1.0, color="#333333", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    axes[2].set_xlabel("Effective true energy [TeV]")
    axes[0].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_point_cell_counts(points: Sequence[SedPoint], path: Path, *, baseline_name: str) -> None:
    plt = setup_matplotlib()
    labels = [f"{p.grouping}:{p.group_label}" for p in points]
    counts = [p.n_cells for p in points]
    colors = ["#e9b44c" if p.is_single_cell_point else "#1f77b4" for p in points]
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    x = np.arange(len(points), dtype=np.float64)
    ax.bar(x, counts, color=colors)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Stage G cells used by each diagnostic SED point, {baseline_name}")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, point in enumerate(points):
        ax.text(idx, point.n_cells + 0.05, str(point.n_cells), ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def relative_path_for_html(target: str, report_path: Path) -> str:
    target_path = Path(target)
    if target_path.is_absolute():
        return os.path.relpath(target_path.resolve(), start=report_path.resolve().parent)
    return target


def write_report_html(path: Path, metadata: Dict[str, object], points: Sequence[SedPoint]) -> None:
    outputs = metadata.get("outputs", {})
    validation = metadata["validation"]  # type: ignore[index]
    frozen = metadata["frozen_spectrum"]  # type: ignore[index]
    reference = metadata["reference_spectrum"]  # type: ignore[index]
    pool1_reference = metadata.get("wcda1_pool1_reference", {})
    baseline_name = str(validation.get("baseline", "diagnostic")) if isinstance(validation, dict) else "diagnostic"
    included_cells = validation.get("required_cell_ids", []) if isinstance(validation, dict) else []
    if not included_cells:
        included_cells = sorted({cell_id for point in points for cell_id in point.cell_ids})
    excluded_cells = validation.get("excluded_cell_ids", []) if isinstance(validation, dict) else []
    background_mode = validation.get("background_mode") if isinstance(validation, dict) else None
    background_form = validation.get("background_form") if isinstance(validation, dict) else None

    def img(key: str, label: str) -> str:
        target = outputs.get(key) if isinstance(outputs, dict) else None
        if not target:
            return ""
        rel = relative_path_for_html(str(target), path)
        return f'<figure><img src="{html.escape(rel)}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'

    table_rows: List[str] = []
    for point in points:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(point.grouping)}</td>"
            f"<td>{html.escape(point.group_label)}</td>"
            f"<td>{html.escape(','.join(str(v) for v in point.cell_ids))}</td>"
            f"<td class=\"num\">{format_float(point.effective_energy_tev, 5)}</td>"
            f"<td class=\"num\">{format_float(point.e2_dnde, 6)}</td>"
            f"<td class=\"num\">{format_float(point.e2_dnde_err, 4)}</td>"
            f"<td class=\"num\">{format_float(point.n0, 6)}</td>"
            f"<td class=\"num\">{format_float(point.chi2, 4)}/{point.ndof}</td>"
            f"<td class=\"num\">{format_float(point.ratio_to_stage_f_pl, 5)}</td>"
            f"<td class=\"num\">{format_float(point.ratio_to_wcda1_ref, 5)}</td>"
            f"<td>{'yes' if point.is_single_cell_point else 'no'}</td>"
            "</tr>"
        )

    pool1_rows: List[str] = []
    for point in pool1_reference_points():
        pool1_rows.append(
            "<tr>"
            f"<td>{html.escape(str(point['nhit']))}</td>"
            f"<td class=\"num\">{format_float(point['emed_tev'], 5)}</td>"
            f"<td class=\"num\">{format_float(point['dnde'], 6)}</td>"
            f"<td class=\"num\">{format_float(point['dnde_err'], 4)}</td>"
            f"<td class=\"num\">{format_float(point['E2_dnde'], 6)}</td>"
            f"<td class=\"num\">{format_float(point['E2_dnde_err'], 4)}</td>"
            f"<td class=\"num\">{format_float(point['significance_sigma'], 4)}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage G Diagnostic SED 报告</title>
<style>
:root {{ --bg:#101418; --fg:#eef3f5; --muted:#a8b2b8; --panel:#171d22; --panel2:#1d242a; --border:#2e3942; --accent:#5fb3b3; --warn:#e9b44c; --code:#0b0f12; }}
@media (prefers-color-scheme: light) {{ :root {{ --bg:#f7f8f9; --fg:#1b2329; --muted:#56636b; --panel:#fff; --panel2:#f1f4f6; --border:#d7dee3; --code:#eef2f4; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans CJK SC","Microsoft YaHei",sans-serif; line-height:1.62; }}
main {{ max-width:1180px; margin:0 auto; padding:42px 20px 64px; }}
header {{ border-bottom:1px solid var(--border); margin-bottom:32px; padding-bottom:24px; }}
.eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; font-weight:700; }}
h1 {{ margin:8px 0 10px; font-size:clamp(30px,5vw,44px); line-height:1.15; }}
h2 {{ margin:38px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--border); font-size:24px; }}
p {{ margin:10px 0; }}
code {{ padding:2px 5px; border-radius:4px; background:var(--code); font-size:13px; }}
.lead {{ max-width:940px; color:var(--muted); font-size:17px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
.metric {{ min-height:104px; padding:16px; border:1px solid var(--border); border-radius:8px; background:var(--panel); }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.07em; }}
.value {{ margin-top:8px; font-size:24px; font-weight:700; overflow-wrap:anywhere; }}
.note {{ margin-top:7px; color:var(--muted); font-size:13px; }}
.callout {{ margin:18px 0; padding:16px 18px; border:1px solid var(--border); border-left:4px solid var(--warn); border-radius:8px; background:var(--panel); }}
.table-wrap {{ width:100%; overflow-x:auto; margin:18px 0; border:1px solid var(--border); border-radius:8px; background:var(--panel); }}
table {{ width:100%; min-width:1120px; border-collapse:collapse; font-size:14px; }}
table.pool1 {{ min-width:860px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
th {{ background:var(--panel); font-weight:700; white-space:nowrap; }}
tbody tr:nth-child(odd) td {{ background:var(--panel2); }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; margin-top:18px; }}
figure {{ margin:0; padding:12px; border:1px solid var(--border); border-radius:8px; background:var(--panel); }}
figure img {{ display:block; width:100%; height:auto; border-radius:4px; background:#fff; }}
figcaption {{ margin-top:9px; color:var(--muted); font-size:13px; }}
footer {{ margin-top:54px; padding-top:18px; border-top:1px solid var(--border); color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
@media (max-width:900px) {{ .grid,.figure-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">LHAASO-WCDA · Crab SED diagnostic</div>
    <h1>Stage G Diagnostic SED 报告</h1>
    <p class="lead">本页是 <code>{html.escape(baseline_name)}</code> 的 diagnostic SED points。它用于检查 Stage F 固定谱形下不同能段的归一化是否自洽，不作为正式发表版。</p>
  </header>
  <section>
    <h2>结论摘要</h2>
    <p>Stage G 固定 Stage F conservative PL 谱形，只在每个分组里重新拟合归一化 <code>N0_bin</code>，再转换为 <code>E^2 dN/dE</code>。能量位置使用冻结 PL 和响应权重下的 true-energy 加权中位数。</p>
    <div class="grid">
      <div class="metric"><div class="label">Run</div><div class="value">{html.escape(str(metadata['run_id']))}</div><div class="note">Stage G diagnostic</div></div>
      <div class="metric"><div class="label">Cells</div><div class="value">{len(included_cells)}</div><div class="note"><code>{html.escape(format_cell_list(included_cells))}</code></div></div>
      <div class="metric"><div class="label">Nhit points</div><div class="value">{sum(1 for p in points if p.grouping == 'nhit')}</div><div class="note">grouped by Nhit</div></div>
      <div class="metric"><div class="label">predE points</div><div class="value">{sum(1 for p in points if p.grouping == 'predE')}</div><div class="note">grouped by predicted energy bin</div></div>
    </div>
    <div class="callout">
      <strong>Diagnostic only：</strong>本页继承 Stage E 的背景统计口径：<code>{html.escape(str(background_mode))}</code> / <code>{html.escape(str(background_form))}</code>。若为 direct-expectation background，Li-Ma 不适用。显式排除 cells：<code>{html.escape(format_cell_list(excluded_cells))}</code>。
    </div>
  </section>
  <section>
    <h2>Stage G 做了什么</h2>
    <p>输入来自 Stage A response、Stage E signal 和 Stage F current fit。脚本验证 Stage F run 为 <code>{html.escape(str(validation['stage_f_run_id']))}</code>，并确认 included cells 为 <code>{html.escape(format_cell_list(included_cells))}</code>。</p>
    <p>冻结谱形为 <code>phi0={format_float(frozen['phi0'], 6)}</code>、<code>gamma={format_float(frozen['gamma'], 6)}</code>、pivot <code>{format_float(frozen['pivot_tev'], 4)} TeV</code>。PL 曲线参考为 <code>{html.escape(str(reference['name']))}</code>：<code>phi0={format_float(reference['phi0'], 6)}</code>、<code>gamma={format_float(reference['gamma'], 4)}</code>。</p>
    <p>本版新增 <code>{html.escape(str(pool1_reference.get('name', 'WCDA-1 Pool-1 Table 1 SED points')) if isinstance(pool1_reference, dict) else 'WCDA-1 Pool-1 Table 1 SED points')}</code> 作为逐点参考。它来自 2021 年 WCDA-1 Crab 标准烛光论文 Table 1，代表一号水池/Pool-1 结果；它不是胡 2023 图 6-32 的全阵列紫色逐点数据。</p>
  </section>
  <section>
    <h2>SED 点结果</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>grouping</th><th>group</th><th>cells</th><th class="num">E_eff [TeV]</th><th class="num">E2 dN/dE</th><th class="num">err</th><th class="num">N0</th><th class="num">chi2/ndof</th><th class="num">ratio StageF</th><th class="num">ratio full-array PL</th><th>single</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>WCDA-1 Pool-1 参考点</h2>
    <p>下表为正式 CPC HTML Table 1 数值，并已转换出 <code>E^2 dN/dE</code>。该参考点用于和一号水池结果做视觉对照；Stage G 拟合本身没有使用这些点。</p>
    <div class="table-wrap">
      <table class="pool1">
        <thead><tr><th>Nhit</th><th class="num">Emed [TeV]</th><th class="num">dN/dE</th><th class="num">err</th><th class="num">E2 dN/dE</th><th class="num">E2 err</th><th class="num">sigma</th></tr></thead>
        <tbody>{''.join(pool1_rows)}</tbody>
      </table>
    </div>
    <p>来源：<code>{html.escape(str(pool1_reference.get('paper', '') if isinstance(pool1_reference, dict) else ''))}</code>，DOI <code>{html.escape(str(pool1_reference.get('doi', '') if isinstance(pool1_reference, dict) else ''))}</code>。</p>
  </section>
  <section>
    <h2>对比图</h2>
    <div class="figure-grid">
      {img('sed_png', 'E^2 dN/dE SED points + Stage F PL + 1LHAASO full-array PL + WCDA-1 Pool-1 points')}
      {img('ratio_png', '每点相对 Stage F PL、1LHAASO full-array PL 和 WCDA-1 Pool-1 points 的 ratio')}
      {img('cell_counts_png', '每个 SED 点使用的 cell 数量')}
    </div>
  </section>
  <footer>
    Generated from <code>{html.escape(str(outputs.get('metadata_json', '')) if isinstance(outputs, dict) else '')}</code>.
  </footer>
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def make_summary_json(metadata: Dict[str, object], points: Sequence[SedPoint]) -> Dict[str, object]:
    return {
        "run_id": metadata["run_id"],
        "description": f"Stage G diagnostic SED points for {metadata['validation'].get('baseline', 'diagnostic')}.",
        "diagnostic_only": True,
        "frozen_spectrum": metadata["frozen_spectrum"],
        "reference_spectrum": metadata["reference_spectrum"],
        "wcda1_pool1_reference": metadata["wcda1_pool1_reference"],
        "validation": metadata["validation"],
        "points": [point_to_dict(point) for point in points],
        "points_by_grouping": {
            "nhit": [point_to_dict(point) for point in points if point.grouping == "nhit"],
            "predE": [point_to_dict(point) for point in points if point.grouping == "predE"],
        },
    }


def make_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    output_root: Path,
    response_npz: Path,
    response_metadata_path: Path,
    signal_npz: Path,
    signal_metadata_path: Path,
    stage_f_npz: Path,
    stage_f_metadata_path: Path,
    response_metadata: Dict[str, object],
    signal_metadata: Dict[str, object],
    stage_f_metadata: Dict[str, object],
    validation: Dict[str, object],
    frozen_phi0: float,
    frozen_gamma: float,
    points: Sequence[SedPoint],
    outputs: Dict[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    baseline_name = str(validation.get("baseline", args.baseline_name))
    required_ids = validation.get("required_cell_ids", [])
    excluded_ids = validation.get("excluded_cell_ids", [])
    return {
        "description": f"Stage G diagnostic SED points for the {baseline_name} Crab baseline.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "diagnostic_only": True,
        "inputs": {
            "response_npz": str(response_npz),
            "response_metadata_json": str(response_metadata_path),
            "signal_npz": str(signal_npz),
            "signal_metadata_json": str(signal_metadata_path),
            "stage_f_npz": str(stage_f_npz),
            "stage_f_metadata_json": str(stage_f_metadata_path),
            "stage_a_run_dir": response_metadata.get("run_dir") if isinstance(response_metadata, dict) else None,
            "stage_e_run_id": signal_metadata.get("run_id") if isinstance(signal_metadata, dict) else None,
            "stage_f_run_id": stage_f_metadata.get("run_id") if isinstance(stage_f_metadata, dict) else None,
        },
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "validation": validation,
        "method": {
            "summary": "Fixed Stage F global PL shape; refit only N0_bin for each diagnostic energy grouping.",
            "error_mode": "conservative sqrt(N_on + B_on)",
            "normalization_fit": "N0 = sum(excess_b * M_unit_b / sigma_b^2) / sum(M_unit_b^2 / sigma_b^2)",
            "normalization_error": "N0_err = 1 / sqrt(sum(M_unit_b^2 / sigma_b^2))",
            "energy_position": "true-energy weighted median under frozen Stage F PL and response weights",
            "e2_flux": "E_eff^2 * N0_bin * (E_eff / pivot)^(-gamma)",
            "energy_quadrature_points": int(args.energy_quadrature_points),
        },
        "frozen_spectrum": {
            "model": "pl",
            "phi0": float(frozen_phi0),
            "gamma": float(frozen_gamma),
            "pivot_tev": float(args.pivot_tev),
            "source": "Stage F pl_conservative fit",
            "stage_f_chi2": validation["stage_f_pl_parameters_validated"]["chi2"],
            "stage_f_ndof": validation["stage_f_pl_parameters_validated"]["ndof"],
        },
        "reference_spectrum": {
            "name": "Hu 2023 / 1LHAASO WCDA full-array Crab PL reference",
            "model": "pl",
            "phi0": float(args.reference_phi0),
            "gamma": float(args.reference_gamma),
            "pivot_tev": float(args.pivot_tev),
            "source": "Hu 2023 / 1LHAASO WCDA full-array catalog PL value used in roadmap",
            "notes": [
                "This is an analytic PL curve from the full-array catalog fit, not digitized Fig. 6-32 SED points.",
                "It should not be labeled WCDA-1/Pool-1.",
            ],
        },
        "wcda1_pool1_reference": {
            **WCDA1_POOL1_TABLE1_SOURCE,
            "points": pool1_reference_points(),
        },
        "groupings": {
            "nhit": {
                "n_points": int(sum(1 for point in points if point.grouping == "nhit")),
                "labels": [point.group_label for point in points if point.grouping == "nhit"],
            },
            "predE": {
                "n_points": int(sum(1 for point in points if point.grouping == "predE")),
                "labels": [point.group_label for point in points if point.grouping == "predE"],
            },
        },
        "points": [point_to_dict(point) for point in points],
        "quality": {
            "status": "passed",
            "diagnostic_only": True,
            "stage_g_publication_ready": False,
            "reason": (
                f"Stage G uses diagnostic baseline {baseline_name}; "
                f"included cells={format_cell_list(required_ids)}, excluded cells={format_cell_list(excluded_ids)}."
            ),
        },
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": outputs,
        "elapsed_seconds": float(elapsed_seconds),
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.pivot_tev <= 0.0:
        raise ValueError("--pivot-tev must be positive")
    if args.reference_phi0 <= 0.0:
        raise ValueError("--reference-phi0 must be positive")
    if args.energy_quadrature_points <= 1:
        raise ValueError("--energy-quadrature-points must be greater than 1")

    response_npz = Path(args.response_npz).resolve()
    response_metadata_path = Path(args.response_metadata).resolve()
    signal_npz = Path(args.signal_npz).resolve()
    signal_metadata_path = Path(args.signal_metadata).resolve()
    stage_f_npz = Path(args.stage_f_npz).resolve()
    stage_f_metadata_path = Path(args.stage_f_metadata).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(args.run_id or make_default_run_id())
    run_dir = prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))

    response_metadata = load_json(response_metadata_path)
    signal_metadata = load_json(signal_metadata_path)
    stage_f_metadata = load_json(stage_f_metadata_path)
    response_full = load_npz(response_npz, "Stage A response")
    signal_full = load_npz(signal_npz, "Stage E signal")
    stage_f = load_npz(stage_f_npz, "Stage F fit")
    response, signal = align_to_stage_f_cells(response_full, signal_full, stage_f)

    fits = stage_f_metadata.get("fits") if isinstance(stage_f_metadata, dict) else None
    pl = fits.get("pl_conservative") if isinstance(fits, dict) else None
    if not isinstance(pl, dict):
        raise ValueError("Stage F metadata is missing fits.pl_conservative")
    params = pl.get("parameters") if isinstance(pl.get("parameters"), dict) else {}
    frozen_phi0 = float(params["phi0"])
    frozen_gamma = float(params["gamma"])

    a_eff = np.asarray(response["a_eff"], dtype=np.float64)
    containment = np.asarray(signal["containment_r_opt"], dtype=np.float64)
    theta_exposure = np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64)
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    stage_f_flux_integral = integrate_pl_flux_bins(
        loge_edges,
        phi0=frozen_phi0,
        gamma=frozen_gamma,
        pivot_tev=float(args.pivot_tev),
        quadrature_points=int(args.energy_quadrature_points),
    )
    stage_f_model_counts_recomputed = model_counts_from_flux_integral(
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        flux_integral=stage_f_flux_integral,
    )
    validation = validate_inputs(
        args=args,
        response=response,
        signal=signal,
        stage_f=stage_f,
        response_metadata=response_metadata,
        signal_metadata=signal_metadata,
        stage_f_metadata=stage_f_metadata,
        stage_f_model_counts_recomputed=stage_f_model_counts_recomputed,
    )
    baseline_name = str(validation.get("baseline", args.baseline_name))
    excluded_cell_ids = [int(v) for v in validation.get("excluded_cell_ids", [])]
    points, unit_counts, frozen_counts, frozen_flux_integral = fit_sed_points(
        excluded_cell_ids=excluded_cell_ids,
        stage_f=stage_f,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        frozen_phi0=frozen_phi0,
        frozen_gamma=frozen_gamma,
        pivot_tev=float(args.pivot_tev),
        reference_phi0=float(args.reference_phi0),
        reference_gamma=float(args.reference_gamma),
        quadrature_points=int(args.energy_quadrature_points),
    )

    npz_path = run_dir / args.npz_name
    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_json_path = run_dir / args.summary_json_name
    summary_md_path = run_dir / args.summary_md_name
    pool1_reference_csv_path = run_dir / "wcda1_pool1_table1_reference.csv"
    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = {
            "sed_png": str(run_dir / "sed_points_stage_f_fullarray_pool1.png"),
            "ratio_png": str(run_dir / "sed_points_ratio.png"),
            "cell_counts_png": str(run_dir / "sed_point_cell_counts.png"),
        }
        plot_sed_points(
            points,
            Path(plot_outputs["sed_png"]),
            baseline_name=baseline_name,
            frozen_phi0=frozen_phi0,
            frozen_gamma=frozen_gamma,
            reference_phi0=float(args.reference_phi0),
            reference_gamma=float(args.reference_gamma),
            pivot_tev=float(args.pivot_tev),
        )
        plot_ratio_points(points, Path(plot_outputs["ratio_png"]))
        plot_point_cell_counts(points, Path(plot_outputs["cell_counts_png"]), baseline_name=baseline_name)

    outputs: Dict[str, object] = {
        "npz": str(npz_path),
        "metadata_json": str(metadata_path),
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "pool1_reference_csv": str(pool1_reference_csv_path),
        "report_html": str(Path(args.report_html).resolve()) if args.report_html else None,
        **plot_outputs,
    }
    metadata = make_metadata(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        output_root=output_root,
        response_npz=response_npz,
        response_metadata_path=response_metadata_path,
        signal_npz=signal_npz,
        signal_metadata_path=signal_metadata_path,
        stage_f_npz=stage_f_npz,
        stage_f_metadata_path=stage_f_metadata_path,
        response_metadata=response_metadata,
        signal_metadata=signal_metadata,
        stage_f_metadata=stage_f_metadata,
        validation=validation,
        frozen_phi0=frozen_phi0,
        frozen_gamma=frozen_gamma,
        points=points,
        outputs=outputs,
        elapsed_seconds=time.perf_counter() - start,
    )
    summary_json = make_summary_json(metadata, points)

    write_npz(npz_path, points, stage_f, frozen_counts)
    write_summary_csv(summary_csv_path, points)
    write_pool1_reference_csv(pool1_reference_csv_path)
    write_summary_md(summary_md_path, metadata, points)
    write_json(summary_json_path, summary_json)
    write_json(metadata_path, metadata)

    if not bool(args.no_promote_current):
        promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted_diagnostic"
        metadata["promotion"]["current_dir"] = str(output_root / "current")
        metadata["promotion"]["latest"] = str(output_root / "latest")
    else:
        metadata["promotion"]["status"] = "skipped_no_promote_current"
    write_json(metadata_path, metadata)
    write_summary_md(summary_md_path, metadata, points)
    write_json(summary_json_path, make_summary_json(metadata, points))
    if args.report_html:
        write_report_html(Path(args.report_html).resolve(), metadata, points)

    print(f"Loaded Stage A response: {response_npz}", flush=True)
    print(f"Loaded Stage E signal: {signal_npz}", flush=True)
    print(f"Loaded Stage F fit: {stage_f_npz}", flush=True)
    print(
        f"Validated {baseline_name}; frozen PL phi0={frozen_phi0:.6e} gamma={frozen_gamma:.6g}",
        flush=True,
    )
    print(
        f"Wrote {sum(1 for p in points if p.grouping == 'nhit')} Nhit points and "
        f"{sum(1 for p in points if p.grouping == 'predE')} predE points",
        flush=True,
    )
    print(f"Wrote {npz_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)
    print(f"Wrote {pool1_reference_csv_path}", flush=True)
    print(f"Wrote {summary_json_path}", flush=True)
    print(f"Wrote {summary_md_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    if args.report_html:
        print(f"Wrote report {Path(args.report_html).resolve()}", flush=True)
    if not bool(args.no_promote_current):
        print(f"Promoted current Stage G output to {output_root / 'current'}", flush=True)


if __name__ == "__main__":
    main()
