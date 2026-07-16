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
from iminuit import Minuit


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RESPONSE_NPZ = "apply/output/stage_a/response_2d.npz"
DEFAULT_RESPONSE_METADATA = "apply/output/stage_a/response_2d_metadata.json"
DEFAULT_SIGNAL_NPZ = "apply/output/stage_e/current/signal_v1.npz"
DEFAULT_SIGNAL_METADATA = "apply/output/stage_e/current/signal_v1_metadata.json"
DEFAULT_STAGE_C_DIR = "apply/output/stage_c/current"
DEFAULT_CELL_SUBSET_CSV = ""
DEFAULT_OUTPUT_DIR = "apply/output/stage_f"
DEFAULT_REPORT_HTML = "apply/report/stage_f_report.html"
DEFAULT_SOURCE_RA_DEG = 83.63
DEFAULT_SOURCE_DEC_DEG = 22.01
DEFAULT_LHAASO_LAT_DEG = 29.45
DEFAULT_LHAASO_LON_DEG = 100.14
DEFAULT_REFERENCE_PHI0 = 2.114e-12
DEFAULT_REFERENCE_GAMMA = 2.69
DEFAULT_PIVOT_TEV = 3.0
M2_TO_CM2 = 1.0e4


@dataclass(frozen=True)
class FitResult:
    model_name: str
    error_mode: str
    valid: bool
    parameters: Dict[str, float]
    errors: Dict[str, Optional[float]]
    fit_parameters: Dict[str, float]
    fit_parameter_errors: Dict[str, Optional[float]]
    fit_parameter_names: List[str]
    covariance_parameterization: str
    covariance: Optional[List[List[float]]]
    chi2: float
    ndof: int
    p_value: Optional[float]
    model_counts: np.ndarray
    residual: np.ndarray
    pull: np.ndarray
    whitened_residual: np.ndarray
    minuit_status: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage F forward-folding chi2 fit for Crab SED cells.")
    parser.add_argument("--response-npz", type=str, default=DEFAULT_RESPONSE_NPZ)
    parser.add_argument("--response-metadata", type=str, default=DEFAULT_RESPONSE_METADATA)
    parser.add_argument("--signal-npz", type=str, default=DEFAULT_SIGNAL_NPZ)
    parser.add_argument("--signal-metadata", type=str, default=DEFAULT_SIGNAL_METADATA)
    parser.add_argument(
        "--excess-covariance-npz",
        type=str,
        default=None,
        help=(
            "Optional bootstrap covariance artifact. Its cell_id must exactly match the selected "
            "Stage E cells and excess_covariance must be symmetric positive definite."
        ),
    )
    parser.add_argument("--stage-c-dir", type=str, default=DEFAULT_STAGE_C_DIR)
    parser.add_argument("--source-files-csv", type=str, default=None)
    parser.add_argument(
        "--cell-subset-csv",
        type=str,
        default=DEFAULT_CELL_SUBSET_CSV,
        help="Optional CSV with cell_id/include columns. Used for diagnostic subset fits.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--report-html", type=str, default=DEFAULT_REPORT_HTML)

    parser.add_argument("--source-ra-deg", type=float, default=DEFAULT_SOURCE_RA_DEG)
    parser.add_argument("--source-dec-deg", type=float, default=DEFAULT_SOURCE_DEC_DEG)
    parser.add_argument("--lhaaso-lat-deg", type=float, default=DEFAULT_LHAASO_LAT_DEG)
    parser.add_argument("--lhaaso-lon-deg", type=float, default=DEFAULT_LHAASO_LON_DEG)
    parser.add_argument("--exposure-sample-step-sec", type=float, default=60.0)
    parser.add_argument("--pivot-tev", type=float, default=DEFAULT_PIVOT_TEV)
    parser.add_argument("--reference-phi0", type=float, default=DEFAULT_REFERENCE_PHI0)
    parser.add_argument("--reference-gamma", type=float, default=DEFAULT_REFERENCE_GAMMA)
    parser.add_argument("--reference-ratio-min", type=float, default=0.3)
    parser.add_argument("--reference-ratio-max", type=float, default=3.0)
    parser.add_argument("--energy-quadrature-points", type=int, default=64)

    parser.add_argument("--npz-name", type=str, default="fit_v1.npz")
    parser.add_argument("--metadata-name", type=str, default="fit_v1_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="fit_v1_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="fit_v1_summary.md")
    return parser.parse_args()


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
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


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
            raise FileExistsError(f"Stage F run directory already exists: {run_dir}")
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
    if label.lower() in {"all", "*"}:
        return None, None
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
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def gmst_deg_from_mjd(mjd: np.ndarray | float) -> np.ndarray:
    jd = np.asarray(mjd, dtype=np.float64) + 2400000.5
    t = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return np.mod(gmst, 360.0)


def local_sidereal_deg(mjd: np.ndarray | float, longitude_east_deg: float) -> np.ndarray:
    return np.mod(gmst_deg_from_mjd(mjd) + float(longitude_east_deg), 360.0)


def wrap_angle_deg(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + 180.0) % 360.0 - 180.0


def source_theta_deg(
    mjd: np.ndarray,
    *,
    source_ra_deg: float,
    source_dec_deg: float,
    latitude_deg: float,
    longitude_east_deg: float,
) -> np.ndarray:
    lst = local_sidereal_deg(mjd, longitude_east_deg)
    hour_angle = np.radians(wrap_angle_deg(lst - float(source_ra_deg)))
    lat = math.radians(float(latitude_deg))
    dec = math.radians(float(source_dec_deg))
    cos_theta = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(hour_angle)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def compute_theta_exposure(
    source_files_csv: Path,
    theta_edges_deg: np.ndarray,
    *,
    source_ra_deg: float,
    source_dec_deg: float,
    latitude_deg: float,
    longitude_east_deg: float,
    sample_step_sec: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not source_files_csv.exists():
        raise FileNotFoundError(f"source_files.csv does not exist: {source_files_csv}")
    if sample_step_sec <= 0.0:
        raise ValueError("--exposure-sample-step-sec must be positive")

    exposure = np.zeros(theta_edges_deg.size - 1, dtype=np.float64)
    processed_files = 0
    skipped_files = 0
    total_live_seconds = 0.0
    total_span_seconds = 0.0
    total_samples = 0
    mjd_min: Optional[float] = None
    mjd_max: Optional[float] = None

    with source_files_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = str(row.get("status") or "")
            if status and status != "processed":
                skipped_files += 1
                continue
            start = finite_float(row.get("matched_mjd_min"))
            stop = finite_float(row.get("matched_mjd_max"))
            live = finite_float(row.get("rough_live_time_seconds"))
            if start is None or stop is None or live is None or stop <= start or live <= 0.0:
                skipped_files += 1
                continue

            span_sec = (stop - start) * 86400.0
            n_samples = max(1, int(math.ceil(span_sec / float(sample_step_sec))))
            step_days = (stop - start) / float(n_samples)
            mids = start + (np.arange(n_samples, dtype=np.float64) + 0.5) * step_days
            theta = source_theta_deg(
                mids,
                source_ra_deg=source_ra_deg,
                source_dec_deg=source_dec_deg,
                latitude_deg=latitude_deg,
                longitude_east_deg=longitude_east_deg,
            )
            hist, _ = np.histogram(theta, bins=theta_edges_deg, weights=np.full(n_samples, live / n_samples))
            exposure += hist.astype(np.float64, copy=False)

            processed_files += 1
            total_live_seconds += live
            total_span_seconds += span_sec
            total_samples += n_samples
            mjd_min = start if mjd_min is None else min(mjd_min, start)
            mjd_max = stop if mjd_max is None else max(mjd_max, stop)

    visible_live_seconds = float(np.sum(exposure))
    meta = {
        "source_files_csv": str(source_files_csv),
        "processed_files": int(processed_files),
        "skipped_files": int(skipped_files),
        "sample_step_seconds": float(sample_step_sec),
        "total_samples": int(total_samples),
        "total_live_seconds": float(total_live_seconds),
        "total_live_days": float(total_live_seconds / 86400.0),
        "source_visible_live_seconds": visible_live_seconds,
        "source_visible_live_days": float(visible_live_seconds / 86400.0),
        "source_visible_fraction_of_live": float(visible_live_seconds / total_live_seconds)
        if total_live_seconds > 0.0
        else None,
        "mjd_min": mjd_min,
        "mjd_max": mjd_max,
        "theta_edges_deg": [float(x) for x in theta_edges_deg],
        "theta_exposure_sec": [float(x) for x in exposure],
        "method": "per-file midpoint integration of Stage C rough live time into source theta bins",
    }
    if processed_files <= 0 or total_live_seconds <= 0.0:
        raise ValueError(f"No usable live-time rows found in {source_files_csv}")
    if visible_live_seconds <= 0.0:
        raise ValueError("Source has zero exposure inside Stage A theta bins")
    return exposure, meta


def pl_flux_tev(E_tev: np.ndarray, *, phi0: float, gamma: float, pivot_tev: float) -> np.ndarray:
    ratio = np.asarray(E_tev, dtype=np.float64) / float(pivot_tev)
    return float(phi0) * np.power(ratio, -float(gamma))


def logpar_flux_tev(
    E_tev: np.ndarray,
    *,
    phi0: float,
    alpha: float,
    beta: float,
    pivot_tev: float,
) -> np.ndarray:
    ratio = np.asarray(E_tev, dtype=np.float64) / float(pivot_tev)
    log_ratio = np.log(ratio)
    return float(phi0) * np.exp((-float(alpha) - float(beta) * log_ratio) * log_ratio)


def integrate_flux_bins(
    loge_edges: np.ndarray,
    *,
    model_name: str,
    params: Dict[str, float],
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
        if model_name == "pl":
            flux = pl_flux_tev(E_tev, phi0=params["phi0"], gamma=params["gamma"], pivot_tev=pivot_tev)
        elif model_name == "logpar":
            flux = logpar_flux_tev(
                E_tev,
                phi0=params["phi0"],
                alpha=params["alpha"],
                beta=params["beta"],
                pivot_tev=pivot_tev,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        # logE is log10(E/GeV); E_TeV = 10**logE / 1000, so dE_TeV/dlogE = ln(10) * E_TeV.
        integrand = flux * math.log(10.0) * E_tev
        out[idx] = 0.5 * (hi - lo) * float(np.sum(weights * integrand))
    return out


def model_counts(
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    loge_edges: np.ndarray,
    *,
    model_name: str,
    params: Dict[str, float],
    pivot_tev: float,
    quadrature_points: int,
) -> np.ndarray:
    flux_integral = integrate_flux_bins(
        loge_edges,
        model_name=model_name,
        params=params,
        pivot_tev=pivot_tev,
        quadrature_points=quadrature_points,
    )
    counts = M2_TO_CM2 * np.einsum("bet,e,t->b", a_eff_m2, flux_integral, theta_exposure_sec)
    return np.asarray(containment, dtype=np.float64) * np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)


def chi2_p_value(chi2_value: float, ndof: int) -> Optional[float]:
    if ndof <= 0 or not math.isfinite(chi2_value):
        return None
    try:
        from scipy.stats import chi2 as chi2_dist

        return float(chi2_dist.sf(float(chi2_value), int(ndof)))
    except Exception:
        return None


def covariance_to_list(minuit: Minuit, names: Sequence[str]) -> Optional[List[List[float]]]:
    if minuit.covariance is None:
        return None
    rows: List[List[float]] = []
    for row_name in names:
        row: List[float] = []
        for col_name in names:
            try:
                row.append(float(minuit.covariance[row_name, col_name]))
            except Exception:
                row.append(float("nan"))
        rows.append(row)
    return rows


def covariance_cholesky(covariance: np.ndarray, *, name: str = "covariance") -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-12):
        asymmetry = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(f"{name} must be symmetric; maximum asymmetry is {asymmetry:.6g}")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
        raise ValueError(
            f"{name} must be positive definite; minimum eigenvalue is {minimum_eigenvalue:.6g}"
        ) from exc


def generalized_chi2(residual: np.ndarray, covariance: np.ndarray) -> float:
    vector = np.asarray(residual, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"residual must be one-dimensional, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual contains non-finite values")
    cholesky = covariance_cholesky(covariance)
    if cholesky.shape[0] != vector.size:
        raise ValueError(
            f"covariance dimension {cholesky.shape[0]} does not match residual length {vector.size}"
        )
    whitened = np.linalg.solve(cholesky, vector)
    return float(whitened @ whitened)


def load_excess_covariance(path: Path, expected_cell_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    expected = np.asarray(expected_cell_ids, dtype=np.int64)
    if expected.ndim != 1:
        raise ValueError(f"Expected cell ids must be one-dimensional, got shape {expected.shape}")
    if expected.size != 44:
        raise ValueError(
            f"Background covariance diagnostic requires exactly 44 selected cells, got {expected.size}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Excess covariance NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"cell_id", "excess_covariance"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        actual = np.asarray(data["cell_id"], dtype=np.int64)
        covariance = np.asarray(data["excess_covariance"], dtype=np.float64)
    if actual.ndim != 1 or actual.size != 44:
        raise ValueError(f"Covariance artifact cell_id must contain exactly 44 entries, got shape {actual.shape}")
    if not np.array_equal(actual, expected):
        raise ValueError(
            "Covariance artifact cell_id order does not exactly match selected Stage E cells: "
            f"artifact={actual.tolist()} selected={expected.tolist()}"
        )
    if covariance.shape != (44, 44):
        raise ValueError(f"excess_covariance must have shape (44, 44), got {covariance.shape}")
    cholesky = covariance_cholesky(covariance, name="excess_covariance")
    eigenvalues = np.linalg.eigvalsh(covariance)
    return covariance, {
        "path": str(path),
        "n_cells": 44,
        "cell_id": actual.tolist(),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(np.linalg.cond(covariance)),
        "cholesky_diagonal_minimum": float(np.min(np.diag(cholesky))),
    }


def fit_model(
    *,
    model_name: str,
    error_mode: str,
    observed: np.ndarray,
    errors: np.ndarray,
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    loge_edges: np.ndarray,
    pivot_tev: float,
    quadrature_points: int,
    start_gamma: float,
    start_phi0: float,
    full_covariance: Optional[np.ndarray] = None,
) -> FitResult:
    observed = np.asarray(observed, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    valid_mask = np.isfinite(observed) & np.isfinite(errors) & (errors > 0.0)
    fit_cholesky: Optional[np.ndarray] = None
    if full_covariance is not None:
        if not np.all(valid_mask):
            raise ValueError(
                f"Full covariance fit {model_name}/{error_mode} requires every selected cell to have finite positive errors"
            )
        matrix = np.asarray(full_covariance, dtype=np.float64)
        if matrix.shape != (observed.size, observed.size):
            raise ValueError(
                f"Full covariance shape {matrix.shape} does not match {observed.size} observed cells"
            )
        fit_cholesky = covariance_cholesky(matrix, name="full excess covariance")
    if not np.any(valid_mask):
        raise ValueError(f"No valid cells for {model_name}/{error_mode} fit")

    observed_fit = observed[valid_mask]
    errors_fit = errors[valid_mask]

    def counts_for_params(params: Dict[str, float]) -> np.ndarray:
        return model_counts(
            a_eff_m2,
            containment,
            theta_exposure_sec,
            loge_edges,
            model_name=model_name,
            params=params,
            pivot_tev=pivot_tev,
            quadrature_points=quadrature_points,
        )

    if model_name == "pl":
        unit_counts = counts_for_params({"phi0": 1.0, "gamma": start_gamma})
        positive_signal = float(np.sum(np.maximum(observed_fit, 0.0)))
        positive_model = float(np.sum(np.maximum(unit_counts[valid_mask], 0.0)))
        phi0_start = start_phi0
        if positive_model > 0.0 and positive_signal > 0.0:
            phi0_start = positive_signal / positive_model
        log10_phi0_start = float(np.clip(math.log10(max(phi0_start, 1.0e-30)), -30.0, 0.0))

        def objective(log10_phi0: float, gamma: float) -> float:
            phi0 = math.pow(10.0, float(log10_phi0))
            model = counts_for_params({"phi0": phi0, "gamma": float(gamma)})
            residual = observed_fit - model[valid_mask]
            if fit_cholesky is not None:
                whitened = np.linalg.solve(fit_cholesky, residual)
                return float(whitened @ whitened)
            scaled = residual / errors_fit
            return float(scaled @ scaled)

        minuit = Minuit(objective, log10_phi0=log10_phi0_start, gamma=float(start_gamma))
        minuit.limits["log10_phi0"] = (-30.0, 0.0)
        minuit.limits["gamma"] = (0.5, 6.0)
        minuit.errors["log10_phi0"] = 0.05
        minuit.errors["gamma"] = 0.05
        param_names = ["log10_phi0", "gamma"]
    elif model_name == "logpar":
        unit_counts = counts_for_params({"phi0": 1.0, "alpha": start_gamma, "beta": 0.0})
        positive_signal = float(np.sum(np.maximum(observed_fit, 0.0)))
        positive_model = float(np.sum(np.maximum(unit_counts[valid_mask], 0.0)))
        phi0_start = start_phi0
        if positive_model > 0.0 and positive_signal > 0.0:
            phi0_start = positive_signal / positive_model
        log10_phi0_start = float(np.clip(math.log10(max(phi0_start, 1.0e-30)), -30.0, 0.0))

        def objective(log10_phi0: float, alpha: float, beta: float) -> float:
            phi0 = math.pow(10.0, float(log10_phi0))
            model = counts_for_params({"phi0": phi0, "alpha": float(alpha), "beta": float(beta)})
            residual = observed_fit - model[valid_mask]
            if fit_cholesky is not None:
                whitened = np.linalg.solve(fit_cholesky, residual)
                return float(whitened @ whitened)
            scaled = residual / errors_fit
            return float(scaled @ scaled)

        minuit = Minuit(objective, log10_phi0=log10_phi0_start, alpha=float(start_gamma), beta=0.0)
        minuit.limits["log10_phi0"] = (-30.0, 0.0)
        minuit.limits["alpha"] = (0.5, 6.0)
        minuit.limits["beta"] = (-2.0, 2.0)
        minuit.errors["log10_phi0"] = 0.05
        minuit.errors["alpha"] = 0.05
        minuit.errors["beta"] = 0.02
        param_names = ["log10_phi0", "alpha", "beta"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    minuit.errordef = Minuit.LEAST_SQUARES
    minuit.migrad()
    if minuit.valid:
        try:
            minuit.hesse()
        except Exception:
            pass

    values = {name: float(minuit.values[name]) for name in param_names}
    fit_errors_out = {
        name: (float(minuit.errors[name]) if np.isfinite(float(minuit.errors[name])) else None)
        for name in param_names
    }
    phi0 = math.pow(10.0, values["log10_phi0"])
    log10_phi0_err = fit_errors_out.get("log10_phi0")
    phi0_err = None
    if log10_phi0_err is not None:
        phi0_err = math.log(10.0) * phi0 * float(log10_phi0_err)
    if model_name == "pl":
        physical_params = {"phi0": phi0, "gamma": values["gamma"]}
        physical_errors: Dict[str, Optional[float]] = {"phi0": phi0_err, "gamma": fit_errors_out.get("gamma")}
    else:
        physical_params = {"phi0": phi0, "alpha": values["alpha"], "beta": values["beta"]}
        physical_errors = {
            "phi0": phi0_err,
            "alpha": fit_errors_out.get("alpha"),
            "beta": fit_errors_out.get("beta"),
        }
    model = counts_for_params(physical_params)
    residual = observed - model
    if full_covariance is not None:
        marginal_errors = np.sqrt(np.diag(np.asarray(full_covariance, dtype=np.float64)))
        pull = residual / marginal_errors
        whitened_residual = np.linalg.solve(fit_cholesky, residual)
    else:
        pull = np.divide(residual, errors, out=np.full_like(residual, np.nan, dtype=np.float64), where=errors > 0.0)
        whitened_residual = pull.copy()
    chi2_value = float(minuit.fval) if minuit.fval is not None else float("nan")
    ndof = int(np.count_nonzero(valid_mask) - len(param_names))
    minuit_status = {
        "valid": bool(minuit.valid),
        "accurate": bool(minuit.accurate),
        "fmin_is_valid": bool(minuit.fmin.is_valid) if minuit.fmin is not None else False,
        "has_covariance": bool(minuit.fmin.has_covariance) if minuit.fmin is not None else False,
        "has_accurate_covar": bool(minuit.fmin.has_accurate_covar) if minuit.fmin is not None else False,
        "edm": float(minuit.fmin.edm) if minuit.fmin is not None and minuit.fmin.edm is not None else None,
    }
    return FitResult(
        model_name=model_name,
        error_mode=error_mode,
        valid=bool(minuit.valid and math.isfinite(chi2_value)),
        parameters=physical_params,
        errors=physical_errors,
        fit_parameters=values,
        fit_parameter_errors=fit_errors_out,
        fit_parameter_names=list(param_names),
        covariance_parameterization="minuit_fit_space: log10_phi0 plus shape parameters",
        covariance=covariance_to_list(minuit, param_names),
        chi2=chi2_value,
        ndof=ndof,
        p_value=chi2_p_value(chi2_value, ndof),
        model_counts=model,
        residual=residual,
        pull=pull,
        whitened_residual=whitened_residual,
        minuit_status=minuit_status,
    )


def result_to_metadata(result: FitResult) -> Dict[str, object]:
    return {
        "model_name": result.model_name,
        "error_mode": result.error_mode,
        "valid": result.valid,
        "parameters": result.parameters,
        "errors": result.errors,
        "fit_parameters": result.fit_parameters,
        "fit_parameter_errors": result.fit_parameter_errors,
        "fit_parameter_names": result.fit_parameter_names,
        "covariance_parameterization": result.covariance_parameterization,
        "covariance": result.covariance,
        "chi2": result.chi2,
        "ndof": result.ndof,
        "chi2_over_ndof": result.chi2 / result.ndof if result.ndof > 0 else None,
        "p_value": result.p_value,
        "minuit_status": result.minuit_status,
    }


def load_arrays(response_npz: Path, signal_npz: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not response_npz.exists():
        raise FileNotFoundError(f"Stage A response NPZ does not exist: {response_npz}")
    if not signal_npz.exists():
        raise FileNotFoundError(f"Stage E signal NPZ does not exist: {signal_npz}")
    with np.load(response_npz, allow_pickle=False) as data:
        response = {name: data[name].copy() for name in data.files}
    with np.load(signal_npz, allow_pickle=False) as data:
        signal = {name: data[name].copy() for name in data.files}
    return response, signal


def load_cell_subset(path: Optional[Path], available_cell_ids: np.ndarray) -> Dict[str, object]:
    if path is None:
        return {
            "enabled": False,
            "path": None,
            "included_cell_ids": [int(v) for v in available_cell_ids],
            "excluded_cell_ids": [],
            "subset_version": None,
            "reasons": {},
        }
    if not path.exists():
        raise FileNotFoundError(f"Cell subset CSV does not exist: {path}")
    include_by_id: Dict[int, bool] = {}
    reasons: Dict[int, str] = {}
    versions: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"cell_id", "include"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            cell_id = int(row["cell_id"])
            value = str(row["include"]).strip().lower()
            include = value in {"1", "true", "yes", "y", "include"}
            include_by_id[cell_id] = include
            reason = str(row.get("subset_reason") or "").strip()
            if reason:
                reasons[cell_id] = reason
            version = str(row.get("subset_version") or "").strip()
            if version:
                versions.append(version)

    available = [int(v) for v in available_cell_ids]
    available_set = set(available)
    unknown_included = sorted(
        cell_id for cell_id, include in include_by_id.items() if include and cell_id not in available_set
    )
    if unknown_included:
        raise ValueError(f"{path} includes cell ids not present in inputs: {unknown_included}")
    included = [cell_id for cell_id in available if include_by_id.get(cell_id, True)]
    excluded = [cell_id for cell_id, include in include_by_id.items() if not include]
    if len(included) < 4:
        raise ValueError(f"Cell subset leaves too few cells for PL/LogPar diagnostics: {included}")
    return {
        "enabled": True,
        "path": str(path),
        "included_cell_ids": included,
        "excluded_cell_ids": excluded,
        "subset_version": sorted(set(versions))[0] if len(set(versions)) == 1 else sorted(set(versions)),
        "reasons": {str(k): v for k, v in reasons.items()},
    }


def apply_cell_subset(
    response: Dict[str, np.ndarray],
    signal: Dict[str, np.ndarray],
    subset: Dict[str, object],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    included_ids = {int(v) for v in subset["included_cell_ids"]}  # type: ignore[index]
    signal_cell_ids = np.asarray(signal["cell_id"], dtype=np.int64)
    response_cell_ids = np.asarray(response["cell_id"], dtype=np.int64)
    if len(set(int(value) for value in response_cell_ids)) != response_cell_ids.size:
        raise ValueError("Stage A response has duplicate cell_id values")
    mask = np.asarray([int(cell_id) in included_ids for cell_id in signal_cell_ids], dtype=bool)
    if not np.any(mask):
        raise ValueError("Cell subset selected no cells")

    selected_cell_ids = signal_cell_ids[mask]
    response_index = {int(cell_id): index for index, cell_id in enumerate(response_cell_ids)}
    missing_response = [int(cell_id) for cell_id in selected_cell_ids if int(cell_id) not in response_index]
    if missing_response:
        raise ValueError(f"Stage A response is missing selected signal cells: {missing_response}")
    response_order = np.asarray([response_index[int(cell_id)] for cell_id in selected_cell_ids], dtype=np.int64)

    def filter_dict(
        values: Dict[str, np.ndarray],
        *,
        input_cell_count: int,
        selection: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        filtered: Dict[str, np.ndarray] = {}
        for key, value in values.items():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == input_cell_count:
                filtered[key] = arr[selection].copy()
            else:
                filtered[key] = arr.copy()
        return filtered

    subset_out = dict(subset)
    subset_out["mask"] = mask.tolist()
    subset_out["n_input_cells"] = int(signal_cell_ids.size)
    subset_out["n_input_response_cells"] = int(response_cell_ids.size)
    subset_out["n_included_cells"] = int(np.count_nonzero(mask))
    subset_out["n_excluded_input_cells"] = int(signal_cell_ids.size - np.count_nonzero(mask))
    subset_out["n_excluded_cells"] = int(len(subset_out.get("excluded_cell_ids", [])))
    return (
        filter_dict(response, input_cell_count=int(response_cell_ids.size), selection=response_order),
        filter_dict(signal, input_cell_count=int(signal_cell_ids.size), selection=mask),
        subset_out,
    )


def validate_inputs(
    response: Dict[str, np.ndarray],
    signal: Dict[str, np.ndarray],
    response_metadata: Dict[str, object],
    signal_metadata: Dict[str, object],
) -> Dict[str, object]:
    required_response = {"a_eff", "logE_true_edges", "theta_true_edges_deg", "cell_id", "nhit_bin", "predE_bin"}
    missing_response = required_response - set(response)
    if missing_response:
        raise ValueError(f"Stage A response is missing arrays: {sorted(missing_response)}")
    required_signal = {
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "containment_r_opt",
        "N_on",
        "B_on",
        "excess",
        "excess_err_stat",
        "excess_err_conservative",
    }
    missing_signal = required_signal - set(signal)
    if missing_signal:
        raise ValueError(f"Stage E signal is missing arrays: {sorted(missing_signal)}")

    response_cells = [
        (int(cid), str(nhit), str(pred))
        for cid, nhit, pred in zip(response["cell_id"], response["nhit_bin"], response["predE_bin"])
    ]
    signal_cells = [
        (int(cid), str(nhit), str(pred))
        for cid, nhit, pred in zip(signal["cell_id"], signal["nhit_bin"], signal["predE_bin"])
    ]
    if response_cells != signal_cells:
        raise ValueError("Stage A response cells do not match Stage E signal cells in order and labels")

    response_type = response_metadata.get("response_type") if isinstance(response_metadata, dict) else None
    if response_metadata:
        expected = {
            "absolute_effective_area_status": "available",
            "weighting": "mc_weight_baseline",
        }
        mismatches = [
            f"{key}={response_metadata.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if response_metadata.get(key) != value
        ]
        if response_type not in {"primary_thrown_response", "primary_thrown_aperture_conditioned_response"}:
            mismatches.append(
                f"response_type={response_type!r}, expected 'primary_thrown_response' "
                "or 'primary_thrown_aperture_conditioned_response'"
            )
        if response_type == "primary_thrown_aperture_conditioned_response":
            conditioning = response_metadata.get("response_aperture_conditioning")
            mode = conditioning.get("mode") if isinstance(conditioning, dict) else None
            if mode != "mc_dangle_le_r_opt":
                mismatches.append(f"response_aperture_conditioning.mode={mode!r}, expected 'mc_dangle_le_r_opt'")
            containment = np.asarray(signal["containment_r_opt"], dtype=np.float64)
            if not np.all(np.isfinite(containment)) or not np.allclose(containment, 1.0, rtol=0.0, atol=1.0e-10):
                mismatches.append("aperture-conditioned Stage A response requires Stage E containment_r_opt == 1")
        if mismatches:
            raise ValueError("Stage A metadata is not an accepted response contract: " + "; ".join(mismatches))

    quality = signal_metadata.get("quality_gate") if isinstance(signal_metadata, dict) else None
    signal_promotable = bool(isinstance(quality, dict) and quality.get("promotable"))
    contract = signal_metadata.get("stage_d_contract") if isinstance(signal_metadata, dict) else None
    background_form = contract.get("background_form") if isinstance(contract, dict) else None
    background_mode = contract.get("background_mode") if isinstance(contract, dict) else None
    cell_versions = []
    for cell in response_metadata.get("cells", []) if isinstance(response_metadata, dict) else []:
        if isinstance(cell, dict):
            version = str(cell.get("selection_version") or "")
            if version:
                cell_versions.append(version)
    return {
        "n_cells": int(len(response_cells)),
        "cells_match": True,
        "signal_quality_promotable": signal_promotable,
        "signal_quality_status": quality.get("status") if isinstance(quality, dict) else None,
        "background_mode": background_mode,
        "background_form": background_form,
        "stage_a_response_type": response_type,
        "cell_selection_versions": sorted(set(cell_versions)),
        "cell_selection_version": sorted(set(cell_versions))[0] if len(set(cell_versions)) == 1 else None,
    }


def resolve_source_files_csv(args: argparse.Namespace, signal_metadata: Dict[str, object]) -> Path:
    if args.source_files_csv:
        return Path(args.source_files_csv).resolve()
    stage_c_dir: Optional[str] = None
    inputs = signal_metadata.get("inputs") if isinstance(signal_metadata, dict) else None
    if isinstance(inputs, dict):
        value = inputs.get("stage_c_dir")
        if isinstance(value, str) and value:
            stage_c_dir = value
    if stage_c_dir is None:
        stage_c_dir = args.stage_c_dir
    return (Path(stage_c_dir).resolve() / "source_files.csv").resolve()


def choose_preferred_fit(fits: Dict[str, FitResult], error_mode: str = "conservative") -> Dict[str, object]:
    pl = fits[f"pl_{error_mode}"]
    logpar = fits[f"logpar_{error_mode}"]
    delta = pl.chi2 - logpar.chi2 if pl.valid and logpar.valid else float("nan")
    if logpar.valid and math.isfinite(delta) and delta >= 4.0:
        preferred = "logpar"
        reason = f"logpar improves chi2 by {delta:.6g} >= 4"
    else:
        preferred = "pl"
        reason = "PL retained because LogPar improvement is below delta_chi2 >= 4 or fit is invalid"
    return {
        "error_mode": error_mode,
        "model": preferred,
        "reason": reason,
        "delta_chi2_pl_minus_logpar": delta,
    }


def reference_preflight(
    *,
    args: argparse.Namespace,
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    loge_edges: np.ndarray,
    observed_excess: np.ndarray,
    quadrature_points: int,
) -> Dict[str, object]:
    counts = model_counts(
        a_eff_m2,
        containment,
        theta_exposure_sec,
        loge_edges,
        model_name="pl",
        params={"phi0": float(args.reference_phi0), "gamma": float(args.reference_gamma)},
        pivot_tev=float(args.pivot_tev),
        quadrature_points=quadrature_points,
    )
    model_total = float(np.sum(counts))
    excess_total = float(np.sum(observed_excess))
    ratio = excess_total / model_total if model_total > 0.0 else float("inf")
    passed = bool(
        math.isfinite(ratio)
        and float(args.reference_ratio_min) <= ratio <= float(args.reference_ratio_max)
    )
    return {
        "reference_model": "PL",
        "reference_phi0_tev_cm2_s": float(args.reference_phi0),
        "reference_gamma": float(args.reference_gamma),
        "pivot_tev": float(args.pivot_tev),
        "reference_source": "Hu 2023 Crab catalog value corrected to 2.114e-12 TeV^-1 cm^-2 s^-1 at 3 TeV",
        "expected_counts_by_cell": counts.tolist(),
        "expected_counts_total": model_total,
        "observed_excess_total": excess_total,
        "observed_over_expected": ratio,
        "allowed_ratio_range": [float(args.reference_ratio_min), float(args.reference_ratio_max)],
        "status": "passed" if passed else "failed_reference_count_preflight",
        "note": (
            "Stage F still writes a diagnostic fit when this fails; Stage G physical SED should wait "
            "until Stage A absolute response normalization is understood."
        ),
    }


def fit_quality(fits: Dict[str, FitResult], reference: Dict[str, object]) -> Dict[str, object]:
    main_valid = bool(fits["pl_conservative"].valid and fits["logpar_conservative"].valid)
    physical_status = str(reference.get("status") or "")
    return {
        "fit_status": "passed" if main_valid else "failed_fit",
        "stage_f_current_promotable": main_valid,
        "stage_g_physical_promotable": bool(main_valid and physical_status == "passed"),
        "physical_flux_status": "ok" if physical_status == "passed" else physical_status,
        "reason": (
            "fits are finite; physical normalization preflight controls Stage G promotion"
            if main_valid
            else "one or more main fits are invalid"
        ),
    }


def build_rows(
    signal: Dict[str, np.ndarray],
    fits: Dict[str, FitResult],
    preferred: Dict[str, object],
) -> List[Dict[str, object]]:
    preferred_model = str(preferred["model"])
    pref = fits[f"{preferred_model}_conservative"]
    pl = fits["pl_conservative"]
    logpar = fits["logpar_conservative"]
    rows: List[Dict[str, object]] = []
    for idx, cell_id in enumerate(signal["cell_id"]):
        rows.append(
            {
                "cell_id": int(cell_id),
                "nhit_bin": str(signal["nhit_bin"][idx]),
                "predE_bin": str(signal["predE_bin"][idx]),
                "N_on": int(signal["N_on"][idx]),
                "B_on": float(signal["B_on"][idx]),
                "excess": float(signal["excess"][idx]),
                "error_conservative": float(signal["excess_err_conservative"][idx]),
                "error_sqrt_n": float(signal["excess_err_stat"][idx]),
                "pl_model": float(pl.model_counts[idx]),
                "pl_pull": float(pl.pull[idx]),
                "logpar_model": float(logpar.model_counts[idx]),
                "logpar_pull": float(logpar.pull[idx]),
                "preferred_model": preferred_model,
                "preferred_counts": float(pref.model_counts[idx]),
                "preferred_pull": float(pref.pull[idx]),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "N_on",
        "B_on",
        "excess",
        "error_conservative",
        "error_sqrt_n",
        "pl_model",
        "pl_pull",
        "logpar_model",
        "logpar_pull",
        "preferred_model",
        "preferred_counts",
        "preferred_pull",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def write_summary_md(path: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    fits = metadata["fits"]  # type: ignore[index]
    preferred = metadata["preferred_fit"]  # type: ignore[index]
    quality = metadata["quality"]  # type: ignore[index]
    reference = metadata["reference_count_preflight"]  # type: ignore[index]
    exposure = metadata["exposure"]  # type: ignore[index]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage F Forward-Folding Fit Summary\n\n")
        f.write(f"- Run id: `{metadata['run_id']}`\n")
        f.write(f"- Preferred model: `{preferred['model']}` ({preferred['reason']})\n")
        f.write(f"- Physical flux status: `{quality['physical_flux_status']}`\n")
        f.write(f"- Source visible exposure: {format_float(exposure['source_visible_live_days'], 6)} days\n")
        f.write(f"- Reference observed/expected counts ratio: {format_float(reference['observed_over_expected'], 6)}\n\n")
        for key in ["pl_conservative", "logpar_conservative", "pl_sqrt_n", "logpar_sqrt_n"]:
            result = fits[key]
            params = ", ".join(f"{name}={format_float(value, 6)}" for name, value in result["parameters"].items())
            f.write(
                f"- `{key}`: valid={result['valid']} chi2/ndof="
                f"{format_float(result['chi2'], 6)}/{result['ndof']} p={format_float(result['p_value'], 4)}; {params}\n"
            )
        f.write("\n| cell | Nhit bin | predE bin | excess | err | PL model | PL pull | LogPar model | LogPar pull |\n")
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{format_float(row['excess'], 6)} | {format_float(row['error_conservative'], 6)} | "
                f"{format_float(row['pl_model'], 6)} | {format_float(row['pl_pull'], 6)} | "
                f"{format_float(row['logpar_model'], 6)} | {format_float(row['logpar_pull'], 6)} |\n"
            )


def heatmap_matrix(values: np.ndarray, signal: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], List[str]]:
    nhit_labels = sorted({str(x) for x in signal["nhit_bin"]}, key=interval_key)
    pred_labels = sorted({str(x) for x in signal["predE_bin"]}, key=interval_key)
    y_index = {label: idx for idx, label in enumerate(nhit_labels)}
    x_index = {label: idx for idx, label in enumerate(pred_labels)}
    matrix = np.full((len(nhit_labels), len(pred_labels)), np.nan, dtype=np.float64)
    for idx, value in enumerate(values):
        y = y_index[str(signal["nhit_bin"][idx])]
        x = x_index[str(signal["predE_bin"][idx])]
        matrix[y, x] = float(value)
    return matrix, nhit_labels, pred_labels


def plot_heatmap(values: np.ndarray, signal: Dict[str, np.ndarray], path: Path, *, title: str, colorbar_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix, nhit_labels, pred_labels = heatmap_matrix(values, signal)
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("log10 E_pred bin")
    ax.set_ylabel("Nhit bin")
    ax.set_xticks(np.arange(len(pred_labels)), pred_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(nhit_labels)), nhit_labels)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y, x]:.2g}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_theta_exposure(theta_edges: np.ndarray, exposure: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.step(centers, exposure / 86400.0, where="mid", color="#1f4e79")
    ax.set_xlabel("Crab theta [deg]")
    ax.set_ylabel("source exposure [days]")
    ax.set_title("Stage F Crab theta exposure from Stage C live-time rows")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_counts(
    signal: Dict[str, np.ndarray],
    rows: Sequence[Dict[str, object]],
    preferred: Dict[str, object],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(rows), dtype=np.float64) + 1.0
    excess = np.asarray([row["excess"] for row in rows], dtype=np.float64)
    err = np.asarray([row["error_conservative"] for row in rows], dtype=np.float64)
    pl = np.asarray([row["pl_model"] for row in rows], dtype=np.float64)
    logpar = np.asarray([row["logpar_model"] for row in rows], dtype=np.float64)
    labels = [str(v) for v in signal["cell_id"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.errorbar(x, excess, yerr=err, fmt="o", color="#222222", markersize=4, label="Stage E excess")
    ax.plot(x, pl, "-o", color="#1f77b4", markersize=3, label="PL model")
    ax.plot(x, logpar, "-o", color="#d62728", markersize=3, label="LogPar model")
    ax.set_xticks(x, labels)
    ax.set_xlabel("cell_id")
    ax.set_ylabel("counts")
    ax.set_title(f"Stage F model counts vs excess; preferred={preferred['model']}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def relative_path_for_html(target: str, report_path: Path) -> str:
    target_path = Path(target)
    if target_path.is_absolute():
        return os.path.relpath(target_path.resolve(), start=report_path.resolve().parent)
    return target


def write_report_html(path: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    outputs = metadata.get("outputs", {})
    fits = metadata["fits"]  # type: ignore[index]
    preferred = metadata["preferred_fit"]  # type: ignore[index]
    quality = metadata["quality"]  # type: ignore[index]
    reference = metadata["reference_count_preflight"]  # type: ignore[index]
    exposure = metadata["exposure"]  # type: ignore[index]

    def img(key: str, label: str) -> str:
        target = outputs.get(key) if isinstance(outputs, dict) else None
        if not target:
            return ""
        rel = relative_path_for_html(str(target), path)
        return f'<figure><img src="{html.escape(rel)}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['cell_id']}</td>"
            f"<td>{html.escape(str(row['nhit_bin']))}</td>"
            f"<td>{html.escape(str(row['predE_bin']))}</td>"
            f"<td class=\"num\">{format_float(row['excess'], 6)}</td>"
            f"<td class=\"num\">{format_float(row['error_conservative'], 5)}</td>"
            f"<td class=\"num\">{format_float(row['pl_model'], 6)}</td>"
            f"<td class=\"num\">{format_float(row['pl_pull'], 5)}</td>"
            f"<td class=\"num\">{format_float(row['logpar_model'], 6)}</td>"
            f"<td class=\"num\">{format_float(row['logpar_pull'], 5)}</td>"
            "</tr>"
        )

    pl = fits["pl_conservative"]
    logpar = fits["logpar_conservative"]
    callout_class = "good" if quality["stage_g_physical_promotable"] else "warn"
    validation = metadata.get("validation", {})
    inputs = metadata.get("inputs", {})
    subset = {}
    if isinstance(validation, dict):
        subset = validation.get("cell_subset", {}) if isinstance(validation.get("cell_subset", {}), dict) else {}
    included_cells = subset.get("included_cell_ids", []) if isinstance(subset, dict) else []
    excluded_cells = subset.get("excluded_cell_ids", []) if isinstance(subset, dict) else []
    included_text = ", ".join(str(v) for v in included_cells) if included_cells else "n/a"
    excluded_text = ", ".join(str(v) for v in excluded_cells) if excluded_cells else "无"
    subset_version = subset.get("subset_version", "n/a") if isinstance(subset, dict) else "n/a"
    subset_label = subset_version if subset_version else "all-cells"
    subset_path = subset.get("path") if isinstance(subset, dict) else None
    n_cells = validation.get("n_cells", len(rows)) if isinstance(validation, dict) else len(rows)
    stage_g_status = "可作为 Stage G diagnostic baseline" if quality["stage_g_physical_promotable"] else "仅保留为诊断结果"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage F 前向折叠拟合报告</title>
<style>
:root {{ --bg:#101418; --fg:#eef3f5; --muted:#a8b2b8; --panel:#171d22; --panel2:#1d242a; --border:#2e3942; --accent:#5fb3b3; --good:#52b788; --warn:#e9b44c; --code:#0b0f12; }}
@media (prefers-color-scheme: light) {{ :root {{ --bg:#f7f8f9; --fg:#1b2329; --muted:#56636b; --panel:#fff; --panel2:#f1f4f6; --border:#d7dee3; --code:#eef2f4; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans CJK SC","Microsoft YaHei",sans-serif; line-height:1.62; }}
main {{ max-width:1180px; margin:0 auto; padding:42px 20px 64px; }}
header {{ border-bottom:1px solid var(--border); margin-bottom:32px; padding-bottom:24px; }}
.eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; font-weight:700; }}
h1 {{ margin:8px 0 10px; font-size:clamp(30px,5vw,44px); line-height:1.15; }}
h2 {{ margin:38px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--border); font-size:24px; }}
h3 {{ margin:24px 0 8px; color:var(--accent); font-size:18px; }}
p {{ margin:10px 0; }}
code {{ padding:2px 5px; border-radius:4px; background:var(--code); font-size:13px; }}
.lead {{ max-width:940px; color:var(--muted); font-size:17px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
.metric {{ min-height:104px; padding:16px; border:1px solid var(--border); border-radius:8px; background:var(--panel); }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.07em; }}
.value {{ margin-top:8px; font-size:24px; font-weight:700; overflow-wrap:anywhere; }}
.note {{ margin-top:7px; color:var(--muted); font-size:13px; }}
.callout {{ margin:18px 0; padding:16px 18px; border:1px solid var(--border); border-left:4px solid var(--accent); border-radius:8px; background:var(--panel); }}
.callout.good {{ border-left-color:var(--good); }}
.callout.warn {{ border-left-color:var(--warn); }}
.table-wrap {{ width:100%; overflow-x:auto; margin:18px 0; border:1px solid var(--border); border-radius:8px; background:var(--panel); }}
table {{ width:100%; min-width:1040px; border-collapse:collapse; font-size:14px; }}
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
    <h1>Stage F 前向折叠拟合报告</h1>
    <p class="lead">本页记录当前 Stage F 的目的、输入、方法和结果：把假设的 Crab 能谱通过 Stage A 二维响应折叠到观测 cell，与 Stage E 的逐 cell excess 做拟合对比。当前报告使用 selector <code>{html.escape(str(subset_label))}</code>，作为进入 Stage G 的 diagnostic baseline。</p>
  </header>
  <section>
    <h2>结论摘要</h2>
    <p>保守误差口径下，当前首选模型是 <code>{html.escape(str(preferred['model']))}</code>。Reference-count preflight 状态为 <code>{html.escape(str(reference['status']))}</code>，该 run 的状态是：<strong>{html.escape(stage_g_status)}</strong>。</p>
    <div class="grid">
      <div class="metric"><div class="label">Run</div><div class="value">{html.escape(str(metadata['run_id']))}</div><div class="note">Stage F fit</div></div>
      <div class="metric"><div class="label">使用 cells</div><div class="value">{format_int(n_cells)}</div><div class="note">subset: <code>{html.escape(str(subset_version))}</code></div></div>
      <div class="metric"><div class="label">可见曝光</div><div class="value">{format_float(exposure['source_visible_live_days'], 5)} d</div><div class="note">Crab theta inside response grid</div></div>
      <div class="metric"><div class="label">PL chi2/ndof</div><div class="value">{format_float(pl['chi2'], 5)}/{pl['ndof']}</div><div class="note">conservative error</div></div>
    </div>
    <div class="callout {callout_class}">
      <strong>Reference-count preflight：</strong>
      使用 WCDA-1 Crab 参考谱时，Stage A/E 预计信号数为 <code>{format_float(reference['expected_counts_total'], 6)}</code>，Stage E excess 为 <code>{format_float(reference['observed_excess_total'], 6)}</code>，observed/expected = <code>{format_float(reference['observed_over_expected'], 6)}</code>。
    </div>
  </section>
  <section>
    <h2>为什么做 Stage F</h2>
    <p>Stage A 给出 MC 响应 <code>A_eff(E_true, theta)</code>，Stage E 给出观测侧每个 cell 的 <code>N_on</code>、背景期望 <code>B_on</code> 和 excess。Stage F 的作用是把二者接起来：不从 <code>E_pred</code> 反推真实能量，而是把一个物理能谱假设从真实能量空间前向折叠到观测 cell 空间，再和 Stage E excess 比较。</p>
    <p>这一步有两个检查目标：第一，确认 Stage A 的绝对响应归一化和 Stage E 的信号量级在同一数量级；第二，为 Stage G 的 SED 点提供一个冻结谱形的全局拟合基准。</p>
  </section>
  <section>
    <h2>Stage F 做了什么</h2>
    <p>本 run 读取 Stage A 的二维响应和 Stage E 的逐 cell 信号表，按 Crab 在观测窗口内的天顶角曝光分布计算每个 cell 的预期信号数。拟合统计量是保守误差口径下的 chi-square：</p>
    <p><code>chi2 = sum_b ((excess_b - N_exp,b) / sigma_b)^2</code></p>
    <p>谱模型包括幂律 PL 和 LogPar。LogPar 只有在相对 PL 的 chi2 改善达到阈值时才作为首选；本 run 的改善量为 <code>{format_float(preferred['delta_chi2_pl_minus_logpar'], 6)}</code>，因此保留 PL。</p>
    <ul>
      <li>Stage A response: <code>{html.escape(str(inputs.get('response_npz', 'n/a')) if isinstance(inputs, dict) else 'n/a')}</code></li>
      <li>Stage E signal: <code>{html.escape(str(inputs.get('signal_npz', 'n/a')) if isinstance(inputs, dict) else 'n/a')}</code></li>
      <li>Cell subset: <code>{html.escape(str(subset_path or 'n/a'))}</code></li>
    </ul>
  </section>
  <section>
    <h2>Cell subset 说明</h2>
    <p>Stage F 不在拟合中按 residual 自动删 cell；include/exclude 完全来自传入的 selector CSV。selector 中的排除原因会记录在 metadata，用于区分 frozen baseline、transition probe 和 diagnostics。</p>
    <ul>
      <li>Included cells: <code>{html.escape(included_text)}</code></li>
      <li>Excluded cells: <code>{html.escape(excluded_text)}</code></li>
    </ul>
  </section>
  <section>
    <h2>拟合参数</h2>
    <ul>
      <li>PL: <code>phi0={format_float(pl['parameters']['phi0'], 6)}</code>, <code>gamma={format_float(pl['parameters']['gamma'], 6)}</code>, p-value <code>{format_float(pl['p_value'], 4)}</code>，chi2/ndof = <code>{format_float(pl['chi2'], 5)}/{pl['ndof']}</code>。</li>
      <li>LogPar: <code>phi0={format_float(logpar['parameters']['phi0'], 6)}</code>, <code>alpha={format_float(logpar['parameters']['alpha'], 6)}</code>, <code>beta={format_float(logpar['parameters']['beta'], 6)}</code>, p-value <code>{format_float(logpar['p_value'], 4)}</code>，chi2/ndof = <code>{format_float(logpar['chi2'], 5)}/{logpar['ndof']}</code>。</li>
    </ul>
  </section>
  <section>
    <h2>逐 cell 残差</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>cell</th><th>Nhit bin</th><th>log10 E_pred bin</th><th class="num">excess</th><th class="num">err</th><th class="num">PL model</th><th class="num">PL pull</th><th class="num">LogPar model</th><th class="num">LogPar pull</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>结果图</h2>
    <div class="figure-grid">
      {img('theta_exposure_png', 'Crab 天顶角曝光分布')}
      {img('model_counts_png', 'Stage E excess 与前向折叠模型计数')}
      {img('pull_grid_pl_png', 'PL pull 网格')}
      {img('pull_grid_logpar_png', 'LogPar pull 网格')}
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


def write_npz(path: Path, signal: Dict[str, np.ndarray], fits: Dict[str, FitResult], theta_exposure_sec: np.ndarray) -> None:
    payload: Dict[str, np.ndarray] = {
        "cell_id": signal["cell_id"].astype(np.int32),
        "nhit_bin": signal["nhit_bin"].astype("U32"),
        "predE_bin": signal["predE_bin"].astype("U32"),
        "N_on": signal["N_on"].astype(np.int64),
        "B_on": signal["B_on"].astype(np.float64),
        "excess": signal["excess"].astype(np.float64),
        "excess_err_stat": signal["excess_err_stat"].astype(np.float64),
        "excess_err_conservative": signal["excess_err_conservative"].astype(np.float64),
        "containment_r_opt": signal["containment_r_opt"].astype(np.float64),
        "theta_exposure_sec": theta_exposure_sec.astype(np.float64),
    }
    for key, result in fits.items():
        payload[f"{key}_model_counts"] = result.model_counts.astype(np.float64)
        payload[f"{key}_residual"] = result.residual.astype(np.float64)
        payload[f"{key}_pull"] = result.pull.astype(np.float64)
        payload[f"{key}_whitened_residual"] = result.whitened_residual.astype(np.float64)
    np.savez_compressed(path, **payload)


def make_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    output_root: Path,
    response_npz: Path,
    response_metadata_path: Path,
    response_metadata: Dict[str, object],
    signal_npz: Path,
    signal_metadata_path: Path,
    signal_metadata: Dict[str, object],
    source_files_csv: Path,
    validation: Dict[str, object],
    exposure_meta: Dict[str, object],
    fits: Dict[str, FitResult],
    preferred: Dict[str, object],
    reference: Dict[str, object],
    quality: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    outputs: Dict[str, object],
    elapsed_seconds: float,
    excess_covariance_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "description": "Stage F forward-folding chi2 fit for Crab SED cells.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {
            "response_npz": str(response_npz),
            "response_metadata_json": str(response_metadata_path),
            "signal_npz": str(signal_npz),
            "signal_metadata_json": str(signal_metadata_path),
            "source_files_csv": str(source_files_csv),
            "cell_subset_csv": str(Path(args.cell_subset_csv).resolve()) if str(args.cell_subset_csv or "").strip() else None,
            "excess_covariance_npz": (
                str(Path(args.excess_covariance_npz).resolve()) if args.excess_covariance_npz else None
            ),
            "stage_a_run_dir": response_metadata.get("run_dir") if isinstance(response_metadata, dict) else None,
            "stage_e_run_id": signal_metadata.get("run_id") if isinstance(signal_metadata, dict) else None,
        },
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "validation": validation,
        "source": {
            "name": "Crab",
            "ra_deg": float(args.source_ra_deg),
            "dec_deg": float(args.source_dec_deg),
        },
        "site": {
            "latitude_deg": float(args.lhaaso_lat_deg),
            "longitude_east_deg": float(args.lhaaso_lon_deg),
            "sidereal_time_formula": "IAU-style GMST polynomial from MJD plus east longitude; no astropy/IERS download.",
        },
        "exposure": exposure_meta,
        "forward_folding": {
            "counts_formula": "S_b = containment_b * 1e4 * sum_E,theta(A_eff_b,E,theta * theta_exposure_theta * integral_flux_E)",
            "a_eff_units": "m^2",
            "flux_units": "TeV^-1 cm^-2 s^-1",
            "energy_grid": "Stage A log10(E_true/GeV) edges",
            "theta_grid": "Stage A theta_true_edges_deg",
            "pivot_tev": float(args.pivot_tev),
            "logpar_log_base": "natural_log",
            "energy_quadrature_points": int(args.energy_quadrature_points),
        },
        "statistic": {
            "baseline": "chi2 on Stage E excess",
            "main_error_mode": "conservative sqrt(N_on + B_on)",
            "comparison_error_mode": "sqrt(N_on)",
            "li_ma_status": "not_applicable_for_direct_expectation_background",
            "background_covariance_diagnostic": excess_covariance_info,
            "background_covariance_objective": (
                "r.T @ C^-1 @ r evaluated by Cholesky solve; no explicit inverse"
                if excess_covariance_info is not None
                else None
            ),
            "background_covariance_pull": (
                "marginal residual / sqrt(diag(C)); whitened residual stored separately"
                if excess_covariance_info is not None
                else None
            ),
        },
        "reference_count_preflight": reference,
        "fits": {key: result_to_metadata(result) for key, result in fits.items()},
        "preferred_fit": preferred,
        "quality": quality,
        "cells": list(rows),
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
    if args.reference_ratio_min <= 0.0 or args.reference_ratio_max <= args.reference_ratio_min:
        raise ValueError("Invalid reference ratio bounds")

    response_npz = Path(args.response_npz).resolve()
    response_metadata_path = Path(args.response_metadata).resolve()
    signal_npz = Path(args.signal_npz).resolve()
    signal_metadata_path = Path(args.signal_metadata).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(args.run_id or make_default_run_id())
    run_dir = prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))

    response_metadata = load_json(response_metadata_path) if response_metadata_path.exists() else {}
    signal_metadata = load_json(signal_metadata_path) if signal_metadata_path.exists() else {}
    source_files_csv = resolve_source_files_csv(args, signal_metadata)
    response, signal = load_arrays(response_npz, signal_npz)
    subset_path = Path(args.cell_subset_csv).resolve() if str(args.cell_subset_csv or "").strip() else None
    cell_subset = load_cell_subset(subset_path, np.asarray(signal["cell_id"], dtype=np.int64))
    response, signal, cell_subset = apply_cell_subset(response, signal, cell_subset)
    validation = validate_inputs(response, signal, response_metadata, signal_metadata)
    validation["cell_subset"] = {key: value for key, value in cell_subset.items() if key != "mask"}

    excess_covariance: Optional[np.ndarray] = None
    excess_covariance_info: Optional[Dict[str, object]] = None
    if args.excess_covariance_npz:
        excess_covariance_path = Path(args.excess_covariance_npz).resolve()
        excess_covariance, excess_covariance_info = load_excess_covariance(
            excess_covariance_path,
            np.asarray(signal["cell_id"], dtype=np.int64),
        )

    a_eff = np.asarray(response["a_eff"], dtype=np.float64)
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    theta_edges = np.asarray(response["theta_true_edges_deg"], dtype=np.float64)
    containment = np.asarray(signal["containment_r_opt"], dtype=np.float64)
    observed = np.asarray(signal["excess"], dtype=np.float64)
    err_conservative = np.asarray(signal["excess_err_conservative"], dtype=np.float64)
    err_sqrt_n = np.asarray(signal["excess_err_stat"], dtype=np.float64)

    theta_exposure, exposure_meta = compute_theta_exposure(
        source_files_csv,
        theta_edges,
        source_ra_deg=float(args.source_ra_deg),
        source_dec_deg=float(args.source_dec_deg),
        latitude_deg=float(args.lhaaso_lat_deg),
        longitude_east_deg=float(args.lhaaso_lon_deg),
        sample_step_sec=float(args.exposure_sample_step_sec),
    )
    reference = reference_preflight(
        args=args,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        observed_excess=observed,
        quadrature_points=int(args.energy_quadrature_points),
    )

    print(f"Loaded Stage A response: {response_npz}", flush=True)
    print(f"Loaded Stage E signal: {signal_npz}", flush=True)
    print(f"Source files: {source_files_csv}", flush=True)
    print(
        "Exposure: "
        f"total_live={exposure_meta['total_live_days']:.6g} d "
        f"source_visible={exposure_meta['source_visible_live_days']:.6g} d",
        flush=True,
    )
    print(
        "Reference preflight: "
        f"expected={reference['expected_counts_total']:.6g} "
        f"observed={reference['observed_excess_total']:.6g} "
        f"ratio={reference['observed_over_expected']:.6g} "
        f"status={reference['status']}",
        flush=True,
    )

    fits: Dict[str, FitResult] = {}
    fits["pl_conservative"] = fit_model(
        model_name="pl",
        error_mode="conservative",
        observed=observed,
        errors=err_conservative,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        pivot_tev=float(args.pivot_tev),
        quadrature_points=int(args.energy_quadrature_points),
        start_gamma=float(args.reference_gamma),
        start_phi0=float(args.reference_phi0),
    )
    fits["logpar_conservative"] = fit_model(
        model_name="logpar",
        error_mode="conservative",
        observed=observed,
        errors=err_conservative,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        pivot_tev=float(args.pivot_tev),
        quadrature_points=int(args.energy_quadrature_points),
        start_gamma=float(fits["pl_conservative"].parameters.get("gamma", args.reference_gamma)),
        start_phi0=float(fits["pl_conservative"].parameters.get("phi0", args.reference_phi0)),
    )
    fits["pl_sqrt_n"] = fit_model(
        model_name="pl",
        error_mode="sqrt_n",
        observed=observed,
        errors=err_sqrt_n,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        pivot_tev=float(args.pivot_tev),
        quadrature_points=int(args.energy_quadrature_points),
        start_gamma=float(args.reference_gamma),
        start_phi0=float(fits["pl_conservative"].parameters.get("phi0", args.reference_phi0)),
    )
    fits["logpar_sqrt_n"] = fit_model(
        model_name="logpar",
        error_mode="sqrt_n",
        observed=observed,
        errors=err_sqrt_n,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        pivot_tev=float(args.pivot_tev),
        quadrature_points=int(args.energy_quadrature_points),
        start_gamma=float(fits["pl_sqrt_n"].parameters.get("gamma", args.reference_gamma)),
        start_phi0=float(fits["pl_sqrt_n"].parameters.get("phi0", args.reference_phi0)),
    )
    if excess_covariance is not None:
        covariance_errors = np.sqrt(np.diag(excess_covariance))
        fits["pl_background_covariance"] = fit_model(
            model_name="pl",
            error_mode="background_covariance",
            observed=observed,
            errors=covariance_errors,
            full_covariance=excess_covariance,
            a_eff_m2=a_eff,
            containment=containment,
            theta_exposure_sec=theta_exposure,
            loge_edges=loge_edges,
            pivot_tev=float(args.pivot_tev),
            quadrature_points=int(args.energy_quadrature_points),
            start_gamma=float(fits["pl_conservative"].parameters.get("gamma", args.reference_gamma)),
            start_phi0=float(fits["pl_conservative"].parameters.get("phi0", args.reference_phi0)),
        )
        fits["logpar_background_covariance"] = fit_model(
            model_name="logpar",
            error_mode="background_covariance",
            observed=observed,
            errors=covariance_errors,
            full_covariance=excess_covariance,
            a_eff_m2=a_eff,
            containment=containment,
            theta_exposure_sec=theta_exposure,
            loge_edges=loge_edges,
            pivot_tev=float(args.pivot_tev),
            quadrature_points=int(args.energy_quadrature_points),
            start_gamma=float(
                fits["pl_background_covariance"].parameters.get("gamma", args.reference_gamma)
            ),
            start_phi0=float(
                fits["pl_background_covariance"].parameters.get("phi0", args.reference_phi0)
            ),
        )
    preferred = choose_preferred_fit(fits, error_mode="conservative")
    quality = fit_quality(fits, reference)
    rows = build_rows(signal, fits, preferred)

    npz_path = run_dir / args.npz_name
    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = {
            "theta_exposure_png": str(run_dir / "theta_exposure.png"),
            "model_counts_png": str(run_dir / "model_counts_vs_excess.png"),
            "pull_grid_pl_png": str(run_dir / "pull_grid_pl.png"),
            "pull_grid_logpar_png": str(run_dir / "pull_grid_logpar.png"),
        }
        plot_theta_exposure(theta_edges, theta_exposure, Path(plot_outputs["theta_exposure_png"]))
        plot_model_counts(signal, rows, preferred, Path(plot_outputs["model_counts_png"]))
        plot_heatmap(
            fits["pl_conservative"].pull,
            signal,
            Path(plot_outputs["pull_grid_pl_png"]),
            title="Stage F PL conservative-error pulls",
            colorbar_label="pull",
        )
        plot_heatmap(
            fits["logpar_conservative"].pull,
            signal,
            Path(plot_outputs["pull_grid_logpar_png"]),
            title="Stage F LogPar conservative-error pulls",
            colorbar_label="pull",
        )

    outputs: Dict[str, object] = {
        "npz": str(npz_path),
        "metadata_json": str(metadata_path),
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
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
        response_metadata=response_metadata,
        signal_npz=signal_npz,
        signal_metadata_path=signal_metadata_path,
        signal_metadata=signal_metadata,
        source_files_csv=source_files_csv,
        validation=validation,
        exposure_meta=exposure_meta,
        fits=fits,
        preferred=preferred,
        reference=reference,
        quality=quality,
        rows=rows,
        outputs=outputs,
        elapsed_seconds=time.perf_counter() - start,
        excess_covariance_info=excess_covariance_info,
    )

    write_npz(npz_path, signal, fits, theta_exposure)
    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, metadata, rows)
    write_json(metadata_path, metadata)

    promotable = bool(quality.get("stage_f_current_promotable")) and not bool(args.no_promote_current)
    if promotable:
        promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = (
            "promoted_physical" if quality.get("stage_g_physical_promotable") else "promoted_diagnostic"
        )
        metadata["promotion"]["current_dir"] = str(output_root / "current")
        metadata["promotion"]["latest"] = str(output_root / "latest")
    elif args.no_promote_current:
        metadata["promotion"]["status"] = "skipped_no_promote_current"
    else:
        metadata["promotion"]["status"] = "blocked_failed_fit"

    write_json(metadata_path, metadata)
    write_summary_md(summary_md_path, metadata, rows)
    if args.report_html:
        write_report_html(Path(args.report_html).resolve(), metadata, rows)

    for key, result in fits.items():
        params = ", ".join(f"{name}={value:.6g}" for name, value in result.parameters.items())
        print(
            f"{key}: valid={result.valid} chi2={result.chi2:.6g} ndof={result.ndof} "
            f"p={format_float(result.p_value, 4)} {params}",
            flush=True,
        )
    print(f"Preferred fit: {preferred['model']} ({preferred['reason']})", flush=True)
    print(f"Quality: {quality['fit_status']} physical_flux_status={quality['physical_flux_status']}", flush=True)
    print(f"Wrote {npz_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)
    print(f"Wrote {summary_md_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    if args.report_html:
        print(f"Wrote report {Path(args.report_html).resolve()}", flush=True)
    if promotable:
        print(f"Promoted current Stage F output to {output_root / 'current'}", flush=True)

    if not quality.get("stage_f_current_promotable"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
