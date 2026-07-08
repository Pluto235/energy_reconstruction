#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_stage02():
    module_path = REPO_ROOT / "apply/stages/02_build_psf.py"
    spec = importlib.util.spec_from_file_location("stage02_build_psf", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage02_build_psf"] = module
    spec.loader.exec_module(module)
    return module


stage02 = load_stage02()

DEFAULT_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate"
DEFAULT_CELL_LEDGER = "apply/config/cell_ledger_v3_candidate.csv"
DEFAULT_STAGE_A_METADATA = "apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json"
DEFAULT_OUTPUT_DIR = "apply/output/stage_b_v5_psf_compare"
DEFAULT_TREE_NAME = "t_eventout"
DEFAULT_LHAASO_LAT_DEG = 29.45
DEFAULT_SOURCE_DEC_DEG = 22.01
DEFAULT_THETA_MAX_DEG = 50.0

PSF_METHODS = (
    "rayleigh_baseline",
    "two_1d_gaussian",
    "mc_quantile_715",
    "observed_data",
    "double_rayleigh_mixture",
)
PSF_METHOD_CHOICES = (*PSF_METHODS, "all")
RAYLEIGH_OPT_RADIUS_FACTOR = stage02.RAYLEIGH_OPT_RADIUS_FACTOR
TARGET_CONTAINMENT = stage02.RAYLEIGH_OPT_CONTAINMENT
DEFAULT_OBSERVED_PROFILE_NPZ = "apply/report/assets/v4-empirical-psf/empirical_psf_profiles.npz"
DEFAULT_OBSERVED_PROFILE_SUMMARY_CSV = "apply/report/assets/v4-empirical-psf/empirical_psf_cell_summary.csv"

V3_BORROW_SPECS = {
    39: {
        "target_bin": ("[500,800)", "[3,3.25)"),
        "source_bins": [("[500,800)", "[3.25,3.5)")],
        "sources": [40],
        "weights": [1.0],
        "method": "nearest_neighbor_borrow",
    },
    52: {
        "target_bin": ("[800,1100)", "[3.25,3.5)"),
        "source_bins": [("[800,1100)", "[3.5,3.75)"), ("[800,1100)", "[3.75,4.0)")],
        "sources": [53, 54],
        "weights": [2.0 / 3.0, 1.0 / 3.0],
        "method": "nearest_neighbor_weighted_interpolation",
    },
    65: {
        "target_bin": ("[1100,2000)", "[3.5,3.75)"),
        "source_bins": [("[1100,2000)", "[3.75,4.0)"), ("[1100,2000)", "[4.0,4.25)")],
        "sources": [66, 67],
        "weights": [2.0 / 3.0, 1.0 / 3.0],
        "method": "nearest_neighbor_weighted_interpolation",
    },
}

BORROW_NUMERIC_KEYS = [
    "sigma_rad",
    "sigma_deg",
    "sigma_mc_weight_deg",
    "sigma_unweighted_deg",
    "sigma_full_rayleigh_rad",
    "sigma_full_rayleigh_deg",
    "sigma_full_mc_weight_deg",
    "sigma_full_unweighted_deg",
    "r_opt_rad",
    "r_opt_deg",
    "containment_r_opt",
    "containment_r_opt_core_fit_full_distribution",
    "containment_minus_expected",
    "r68_deg",
    "r90_deg",
    "r95_deg",
    "core_r68_deg",
    "core_r90_deg",
    "core_r95_deg",
    "r715_deg",
    "sigma_eff_deg",
    "sigma_x_deg",
    "sigma_y_deg",
    "sigma_x_over_y",
    "mu_x_deg",
    "mu_y_deg",
    "mc_quantile_r715_deg",
    "observed_data_r715_deg",
    "observed_data_containment_r_opt",
    "observed_data_raw_r715_deg",
    "observed_data_positive_total",
    "observed_data_raw_positive_total",
    "observed_data_raw_total",
    "observed_data_pedestal_per_deg2",
    "observed_data_r_opt_over_rayleigh",
    "observed_data_r_opt_over_mc_quantile",
    "rayleigh_baseline_r715_deg",
    "rayleigh_baseline_containment_r_opt",
    "two_1d_gaussian_r715_deg",
    "two_1d_gaussian_containment_r_opt",
    "mc_quantile_containment_r_opt",
    "double_rayleigh_A",
    "double_rayleigh_sigma1_deg",
    "double_rayleigh_sigma2_deg",
    "double_rayleigh_sigma_eq_deg",
    "double_rayleigh_r_opt_deg",
    "double_rayleigh_containment_r_opt",
    "double_rayleigh_model_containment_r_opt",
    "double_rayleigh_chi2",
    "double_rayleigh_ndof",
    "double_rayleigh_chi2_ndof",
]


class ExtendedCellEvents:
    def __init__(
        self,
        *,
        dangle_rad: np.ndarray,
        reco_theta_rad: np.ndarray,
        reco_phi_rad: np.ndarray,
        mc_theta_rad: np.ndarray,
        mc_phi_rad: np.ndarray,
        mc_theta_deg: np.ndarray,
        mc_weight: np.ndarray,
        loge_true: np.ndarray,
        input_files: int,
        angle_check_absdiff_rad: np.ndarray,
    ) -> None:
        self.dangle_rad = dangle_rad
        self.reco_theta_rad = reco_theta_rad
        self.reco_phi_rad = reco_phi_rad
        self.mc_theta_rad = mc_theta_rad
        self.mc_phi_rad = mc_phi_rad
        self.mc_theta_deg = mc_theta_deg
        self.mc_weight = mc_weight
        self.loge_true = loge_true
        self.input_files = input_files
        self.angle_check_absdiff_rad = angle_check_absdiff_rad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a v5 Stage B PSF artifact for one PSF aperture-comparison method."
    )
    parser.add_argument("--psf-method", choices=PSF_METHOD_CHOICES, required=True)
    parser.add_argument("--binned-root", type=str, default=DEFAULT_BINNED_ROOT)
    parser.add_argument("--cell-selection-csv", type=str, default=DEFAULT_CELL_LEDGER)
    parser.add_argument("--stage-a-metadata", type=str, default=DEFAULT_STAGE_A_METADATA)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--tree-name", type=str, default=DEFAULT_TREE_NAME)
    parser.add_argument("--max-files-per-cell", type=int, default=None)
    parser.add_argument("--allow-missing-stage-a-metadata", action="store_true", default=False)
    parser.add_argument("--allow-missing-cell-dirs", action="store_true", default=False)
    parser.add_argument("--weight-branch", type=str, default="mc_weight")
    parser.add_argument("--allow-missing-weight", action="store_true", default=False)
    parser.add_argument("--logE-min", type=float, default=None)
    parser.add_argument("--logE-max", type=float, default=None)
    parser.add_argument("--lhaaso-lat-deg", type=float, default=DEFAULT_LHAASO_LAT_DEG)
    parser.add_argument("--source-dec-deg", type=float, default=DEFAULT_SOURCE_DEC_DEG)
    parser.add_argument("--theta-min-deg", type=float, default=0.0)
    parser.add_argument("--theta-max-deg", type=float, default=DEFAULT_THETA_MAX_DEG)
    parser.add_argument("--theta-step-deg", type=float, default=1.0)
    parser.add_argument("--hour-angle-samples", type=int, default=200000)
    parser.add_argument("--allow-incomplete-theta-support", action="store_true", default=False)
    parser.add_argument("--min-events-per-cell", type=int, default=1000)
    parser.add_argument("--min-effective-events", type=float, default=200.0)
    parser.add_argument("--allow-low-stat-psf-fallback", action="store_true", default=False)
    parser.add_argument("--core-fit-max-deg", type=float, default=3.0)
    parser.add_argument("--theta-missing-mass-fail-threshold", type=float, default=0.10)
    parser.add_argument("--containment-warning-tolerance", type=float, default=0.12)
    parser.add_argument("--angle-check-max-events", type=int, default=20000)
    parser.add_argument("--angle-check-warn-rad", type=float, default=1.0e-4)
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile-max-deg", type=float, default=5.0)
    parser.add_argument("--profile-bin-width-deg", type=float, default=0.05)
    parser.add_argument("--two1d-radial-quadrature", type=int, default=96)
    parser.add_argument("--two1d-angle-samples", type=int, default=512)
    parser.add_argument("--observed-profile-npz", type=str, default=DEFAULT_OBSERVED_PROFILE_NPZ)
    parser.add_argument("--observed-profile-summary-csv", type=str, default=DEFAULT_OBSERVED_PROFILE_SUMMARY_CSV)
    parser.add_argument("--observed-stage-b-source-npz", type=str, default=None)
    parser.add_argument("--observed-stage-b-source-metadata", type=str, default=None)
    parser.add_argument("--double-rayleigh-stage-b-source-npz", type=str, default=None)
    parser.add_argument("--double-rayleigh-stage-b-source-metadata", type=str, default=None)
    parser.add_argument("--observed-pedestal-min-deg", type=float, default=2.5)
    parser.add_argument("--observed-max-r-opt-over-rayleigh", type=float, default=2.5)
    parser.add_argument("--observed-max-r-opt-over-mc-quantile", type=float, default=2.0)
    parser.add_argument("--observed-max-r-opt-deg", type=float, default=2.0)
    parser.add_argument("--observed-min-positive-total", type=float, default=100.0)
    parser.add_argument("--observed-require-reliable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no-borrow-v3-fallback-psf",
        action="store_true",
        default=False,
        help="Disable the v3/v4 PSF-borrow policy for ridge-left bins matching legacy cells 39, 52, and 65.",
    )
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--npz-name", type=str, default="psf_v5_compare.npz")
    parser.add_argument("--metadata-name", type=str, default="psf_v5_compare_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="psf_v5_compare_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="psf_v5_compare_summary.md")
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help="Run the two-1D Gaussian containment isotropic sanity check and exit.",
    )
    return parser.parse_args()


def path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


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
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_extended_cell_events(
    files: Sequence[Path],
    *,
    tree_name: str,
    weight_branch: str,
    allow_missing_weight: bool,
    angle_check_max_events: int,
    file_progress_every: int,
    progress_label: str,
) -> ExtendedCellEvents:
    dangle_chunks: List[np.ndarray] = []
    reco_theta_chunks: List[np.ndarray] = []
    reco_phi_chunks: List[np.ndarray] = []
    mc_theta_chunks: List[np.ndarray] = []
    mc_phi_chunks: List[np.ndarray] = []
    mc_theta_deg_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []
    loge_chunks: List[np.ndarray] = []
    angle_check_chunks: List[np.ndarray] = []
    angle_check_remaining = max(0, int(angle_check_max_events))

    required_base = ["mc_dangle", "mc_theta", "mc_phi", "theta", "phi", "mc_energy"]
    optional_branches: List[str] = []
    if allow_missing_weight:
        optional_branches.append(weight_branch)
    else:
        required_base.append(weight_branch)

    file_progress_every = max(0, int(file_progress_every))
    for file_idx, file_path in enumerate(files, start=1):
        arrays = stage02.arrays_for_tree(
            file_path,
            tree_name,
            required_base,
            optional_branches=optional_branches,
        )
        dangle = np.asarray(arrays["mc_dangle"], dtype=np.float64)
        mc_theta = np.asarray(arrays["mc_theta"], dtype=np.float64)
        mc_phi = np.asarray(arrays["mc_phi"], dtype=np.float64)
        reco_theta = np.asarray(arrays["theta"], dtype=np.float64)
        reco_phi = np.asarray(arrays["phi"], dtype=np.float64)
        weight = stage02.load_weight(
            arrays,
            weight_branch=weight_branch,
            allow_missing_weight=allow_missing_weight,
        )
        mc_energy = np.asarray(arrays["mc_energy"], dtype=np.float64)

        dangle_chunks.append(dangle)
        reco_theta_chunks.append(reco_theta)
        reco_phi_chunks.append(reco_phi)
        mc_theta_chunks.append(mc_theta)
        mc_phi_chunks.append(mc_phi)
        mc_theta_deg_chunks.append(np.degrees(mc_theta))
        weight_chunks.append(weight)
        loge_chunks.append(np.log10(mc_energy, where=mc_energy > 0, out=np.full_like(mc_energy, np.nan, dtype=np.float64)))

        if angle_check_remaining > 0 and dangle.size > 0:
            take = min(angle_check_remaining, dangle.size)
            sep = stage02.spherical_separation_rad(
                reco_theta[:take],
                reco_phi[:take],
                mc_theta[:take],
                mc_phi[:take],
            )
            angle_check_chunks.append(np.abs(sep - dangle[:take]))
            angle_check_remaining -= take

        if file_progress_every > 0 and (file_idx % file_progress_every == 0 or file_idx == len(files)):
            print(
                f"[{progress_label}] read {file_idx}/{len(files)} files | events={sum(len(chunk) for chunk in dangle_chunks)}",
                flush=True,
            )

    return ExtendedCellEvents(
        dangle_rad=stage02.append_concat(dangle_chunks, np.float64),
        reco_theta_rad=stage02.append_concat(reco_theta_chunks, np.float64),
        reco_phi_rad=stage02.append_concat(reco_phi_chunks, np.float64),
        mc_theta_rad=stage02.append_concat(mc_theta_chunks, np.float64),
        mc_phi_rad=stage02.append_concat(mc_phi_chunks, np.float64),
        mc_theta_deg=stage02.append_concat(mc_theta_deg_chunks, np.float64),
        mc_weight=stage02.append_concat(weight_chunks, np.float64),
        loge_true=stage02.append_concat(loge_chunks, np.float64),
        input_files=len(files),
        angle_check_absdiff_rad=stage02.append_concat(angle_check_chunks, np.float64),
    )


def tangent_offsets_deg(
    reco_theta: np.ndarray,
    reco_phi: np.ndarray,
    true_theta: np.ndarray,
    true_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sin_t = np.sin(reco_theta)
    reco_x = sin_t * np.cos(reco_phi)
    reco_y = sin_t * np.sin(reco_phi)
    reco_z = np.cos(reco_theta)

    sin_t0 = np.sin(true_theta)
    cos_t0 = np.cos(true_theta)
    cos_p0 = np.cos(true_phi)
    sin_p0 = np.sin(true_phi)
    true_x = sin_t0 * cos_p0
    true_y = sin_t0 * sin_p0
    true_z = cos_t0

    east_x = -sin_p0
    east_y = cos_p0
    east_z = np.zeros_like(east_x)
    south_x = cos_t0 * cos_p0
    south_y = cos_t0 * sin_p0
    south_z = -sin_t0

    dot_center = reco_x * true_x + reco_y * true_y + reco_z * true_z
    dot_center = np.where(np.abs(dot_center) > 1.0e-12, dot_center, np.nan)
    x_rad = (reco_x * east_x + reco_y * east_y + reco_z * east_z) / dot_center
    y_rad = (reco_x * south_x + reco_y * south_y + reco_z * south_z) / dot_center
    return np.degrees(x_rad), np.degrees(y_rad)


def weighted_mean_sigma(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan"), float("nan")
    x = values[valid]
    w = weights[valid]
    sumw = float(np.sum(w))
    if sumw <= 0.0:
        return float("nan"), float("nan")
    mean = float(np.sum(w * x) / sumw)
    var = float(np.sum(w * (x - mean) ** 2) / sumw)
    return mean, math.sqrt(max(var, 0.0))


def gaussian_circular_containment(
    radius_deg: float,
    *,
    mu_x_deg: float,
    mu_y_deg: float,
    sigma_x_deg: float,
    sigma_y_deg: float,
    radial_quadrature: int,
    angle_samples: int,
) -> float:
    radius = float(radius_deg)
    if radius <= 0.0:
        return 0.0
    sx = max(float(sigma_x_deg), 1.0e-9)
    sy = max(float(sigma_y_deg), 1.0e-9)
    nodes, weights = np.polynomial.legendre.leggauss(max(16, int(radial_quadrature)))
    rho = 0.5 * radius * (nodes + 1.0)
    wr = 0.5 * radius * weights
    phi = np.linspace(0.0, 2.0 * math.pi, max(64, int(angle_samples)), endpoint=False, dtype=np.float64)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    norm = 1.0 / (2.0 * math.pi * sx * sy)
    total = 0.0
    for r, w in zip(rho, wr):
        x = r * cos_phi
        y = r * sin_phi
        exponent = -0.5 * (((x - mu_x_deg) / sx) ** 2 + ((y - mu_y_deg) / sy) ** 2)
        angular_integral = float(np.mean(np.exp(exponent)) * (2.0 * math.pi))
        total += float(w * r * norm * angular_integral)
    return float(min(max(total, 0.0), 1.0))


def gaussian_radius_for_containment(
    target: float,
    *,
    mu_x_deg: float,
    mu_y_deg: float,
    sigma_x_deg: float,
    sigma_y_deg: float,
    radial_quadrature: int,
    angle_samples: int,
) -> float:
    sx = max(float(sigma_x_deg), 1.0e-9)
    sy = max(float(sigma_y_deg), 1.0e-9)
    mu = math.hypot(float(mu_x_deg), float(mu_y_deg))
    high = max(RAYLEIGH_OPT_RADIUS_FACTOR * math.sqrt(0.5 * (sx * sx + sy * sy)), mu + 8.0 * max(sx, sy), 1.0e-6)
    for _ in range(30):
        cdf = gaussian_circular_containment(
            high,
            mu_x_deg=mu_x_deg,
            mu_y_deg=mu_y_deg,
            sigma_x_deg=sigma_x_deg,
            sigma_y_deg=sigma_y_deg,
            radial_quadrature=radial_quadrature,
            angle_samples=angle_samples,
        )
        if cdf >= target:
            break
        high *= 1.5
    low = 0.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        cdf = gaussian_circular_containment(
            mid,
            mu_x_deg=mu_x_deg,
            mu_y_deg=mu_y_deg,
            sigma_x_deg=sigma_x_deg,
            sigma_y_deg=sigma_y_deg,
            radial_quadrature=radial_quadrature,
            angle_samples=angle_samples,
        )
        if cdf < target:
            low = mid
        else:
            high = mid
    return float(high)


def empirical_containment(r_rad: np.ndarray, weight: np.ndarray, radius_rad: float) -> float:
    valid = np.isfinite(r_rad) & (r_rad >= 0.0) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return float("nan")
    denom = float(np.sum(weight[valid]))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weight[valid & (r_rad <= radius_rad)]) / denom)


def shell_quantile_radius(edges_deg: np.ndarray, shell_values: np.ndarray, quantile: float) -> Tuple[float, float]:
    edges = np.asarray(edges_deg, dtype=np.float64)
    values = np.asarray(shell_values, dtype=np.float64)
    values = np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0 or edges.size != values.size + 1:
        return float("nan"), total
    target = float(quantile) * total
    cumulative = np.cumsum(values)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx < 0 or idx >= values.size:
        return float("nan"), total
    previous = float(cumulative[idx - 1]) if idx > 0 else 0.0
    shell_value = float(values[idx])
    fraction = 0.0 if shell_value <= 0.0 else (target - previous) / shell_value
    fraction = min(max(float(fraction), 0.0), 1.0)
    radius = float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))
    return radius, total


def stable_logit(value: float) -> float:
    clipped = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(clipped / (1.0 - clipped))


def stable_sigmoid(value: float) -> float:
    x = float(value)
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def double_rayleigh_pdf_deg(r_deg: np.ndarray, a_core: float, sigma1_deg: float, sigma2_deg: float) -> np.ndarray:
    r = np.asarray(r_deg, dtype=np.float64)
    a = float(a_core)
    s1 = float(sigma1_deg)
    s2 = float(sigma2_deg)
    if not (0.0 < a < 1.0 and 0.0 < s1 < s2):
        return np.full(r.shape, np.nan, dtype=np.float64)
    r_pos = np.clip(r, 0.0, None)
    core = a * r_pos / (s1 * s1) * np.exp(-0.5 * (r_pos / s1) ** 2)
    tail = (1.0 - a) * r_pos / (s2 * s2) * np.exp(-0.5 * (r_pos / s2) ** 2)
    return core + tail


def double_rayleigh_cdf_deg(radius_deg: float, a_core: float, sigma1_deg: float, sigma2_deg: float) -> float:
    r = float(radius_deg)
    a = float(a_core)
    s1 = float(sigma1_deg)
    s2 = float(sigma2_deg)
    if not (r >= 0.0 and 0.0 < a < 1.0 and 0.0 < s1 < s2):
        return float("nan")
    return float(1.0 - a * math.exp(-0.5 * (r / s1) ** 2) - (1.0 - a) * math.exp(-0.5 * (r / s2) ** 2))


def double_rayleigh_radius_for_containment(
    target: float,
    *,
    a_core: float,
    sigma1_deg: float,
    sigma2_deg: float,
) -> float:
    target_value = float(target)
    if not (0.0 < target_value < 1.0 and 0.0 < a_core < 1.0 and 0.0 < sigma1_deg < sigma2_deg):
        return float("nan")
    high = max(RAYLEIGH_OPT_RADIUS_FACTOR * float(sigma2_deg), RAYLEIGH_OPT_RADIUS_FACTOR * float(sigma1_deg), 1.0e-6)
    for _ in range(40):
        cdf = double_rayleigh_cdf_deg(high, a_core, sigma1_deg, sigma2_deg)
        if np.isfinite(cdf) and cdf >= target_value:
            break
        high *= 1.8
    else:
        return float("nan")
    low = 0.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        cdf = double_rayleigh_cdf_deg(mid, a_core, sigma1_deg, sigma2_deg)
        if not np.isfinite(cdf):
            return float("nan")
        if cdf < target_value:
            low = mid
        else:
            high = mid
    return float(high)


def weighted_profile_for_fit(
    r_deg: np.ndarray,
    weight: np.ndarray,
    profile_edges_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    valid = np.isfinite(r_deg) & (r_deg >= 0.0) & np.isfinite(weight) & (weight > 0.0)
    edges = np.asarray(profile_edges_deg, dtype=np.float64)
    if not np.any(valid):
        zeros = np.zeros(edges.size - 1, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:]), zeros, zeros, zeros, 0.0, 0.0
    hist, _ = np.histogram(r_deg[valid], bins=edges, weights=weight[valid])
    hist_w2, _ = np.histogram(r_deg[valid], bins=edges, weights=weight[valid] ** 2)
    total = float(np.sum(hist))
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if total <= 0.0:
        zeros = np.zeros(edges.size - 1, dtype=np.float64)
        return centers, zeros, zeros, hist, 0.0, 0.0
    density = np.divide(hist, total * widths, out=np.zeros_like(hist, dtype=np.float64), where=widths > 0.0)
    sigma_density = np.divide(
        np.sqrt(np.clip(hist_w2, 0.0, None)),
        total * widths,
        out=np.zeros_like(hist_w2, dtype=np.float64),
        where=widths > 0.0,
    )
    sumw2 = float(np.sum(hist_w2))
    neff = (total * total / sumw2) if sumw2 > 0.0 else 0.0
    return centers, density, sigma_density, hist, total, float(neff)


def fit_double_rayleigh_mixture(
    r_deg: np.ndarray,
    weight: np.ndarray,
    profile_edges_deg: np.ndarray,
    *,
    rayleigh_sigma_deg: float,
) -> Dict[str, object]:
    centers, density, sigma_density, hist, total, profile_neff = weighted_profile_for_fit(r_deg, weight, profile_edges_deg)
    positive_bins = (hist > 0.0) & np.isfinite(density)
    if total <= 0.0:
        return {"status": "fallback", "reason": "double_rayleigh_empty_weighted_profile"}
    if int(np.count_nonzero(positive_bins)) < 6:
        return {
            "status": "fallback",
            "reason": f"double_rayleigh_positive_profile_bins_below_min:{int(np.count_nonzero(positive_bins))}<6",
            "profile_effective_events": profile_neff,
        }

    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - depends on runtime environment.
        return {"status": "fallback", "reason": f"double_rayleigh_scipy_unavailable:{type(exc).__name__}"}

    edges = np.asarray(profile_edges_deg, dtype=np.float64)
    fit_max_deg = float(edges[-1])
    fit_mask = positive_bins & (centers > 0.0)
    if int(np.count_nonzero(fit_mask)) < 6:
        return {
            "status": "fallback",
            "reason": f"double_rayleigh_fit_bins_below_min:{int(np.count_nonzero(fit_mask))}<6",
            "profile_effective_events": profile_neff,
        }

    x_fit = centers[fit_mask]
    y_fit = density[fit_mask]
    err_fit = sigma_density[fit_mask]
    positive_err = err_fit[np.isfinite(err_fit) & (err_fit > 0.0)]
    err_floor = float(np.nanmedian(positive_err) * 0.5) if positive_err.size else float("nan")
    if not np.isfinite(err_floor) or err_floor <= 0.0:
        err_floor = max(float(np.nanmedian(y_fit[y_fit > 0.0])) * 0.05 if np.any(y_fit > 0.0) else 1.0, 1.0e-9)
    err_fit = np.maximum(np.where(np.isfinite(err_fit), err_fit, err_floor), err_floor)

    sigma_min = max(float(np.nanmin(np.diff(edges))) * 0.05, 1.0e-4)
    sigma_seed = float(rayleigh_sigma_deg) if np.isfinite(rayleigh_sigma_deg) and rayleigh_sigma_deg > 0.0 else float("nan")
    if not np.isfinite(sigma_seed) or sigma_seed <= 0.0:
        q68 = stage02.weighted_quantile(r_deg, [0.68], weight)[0]
        sigma_seed = float(q68 / math.sqrt(-2.0 * math.log(1.0 - 0.68))) if np.isfinite(q68) else 0.25
    sigma_seed = max(float(sigma_seed), sigma_min * 2.0)
    sigma_max = max(fit_max_deg * 4.0, sigma_seed * 20.0, 5.0)
    delta_min = sigma_min * 0.2
    delta_max = sigma_max
    lower = np.asarray([-8.0, math.log(sigma_min), math.log(delta_min)], dtype=np.float64)
    upper = np.asarray([8.0, math.log(sigma_max), math.log(delta_max)], dtype=np.float64)

    q_rad = stage02.weighted_quantile(np.radians(r_deg), [0.50, 0.68, 0.90, 0.95], weight)
    q_deg = np.degrees(q_rad)
    q68_sigma = (
        float(q_deg[1]) / math.sqrt(-2.0 * math.log(1.0 - 0.68))
        if q_deg.size > 1 and np.isfinite(q_deg[1]) and q_deg[1] > 0.0
        else sigma_seed
    )
    q90_sigma = (
        float(q_deg[2]) / math.sqrt(-2.0 * math.log(1.0 - 0.90))
        if q_deg.size > 2 and np.isfinite(q_deg[2]) and q_deg[2] > 0.0
        else sigma_seed * 1.8
    )
    q95_sigma = (
        float(q_deg[3]) / math.sqrt(-2.0 * math.log(1.0 - 0.95))
        if q_deg.size > 3 and np.isfinite(q_deg[3]) and q_deg[3] > 0.0
        else sigma_seed * 2.2
    )

    def normalize_start(a_core: float, sigma1: float, sigma2: float) -> Tuple[float, float, float]:
        a = min(max(float(a_core), 0.02), 0.98)
        s1 = min(max(float(sigma1), sigma_min), sigma_max * 0.95)
        s2 = min(max(float(sigma2), s1 + delta_min), sigma_max)
        if s2 <= s1:
            s2 = min(sigma_max, s1 + max(delta_min, 0.25 * s1))
        return a, s1, s2

    starts = [
        normalize_start(0.85, 0.75 * sigma_seed, 1.80 * sigma_seed),
        normalize_start(0.70, 0.60 * sigma_seed, 2.50 * sigma_seed),
        normalize_start(0.55, 0.50 * sigma_seed, 3.50 * sigma_seed),
        normalize_start(0.90, min(q68_sigma, sigma_seed), max(q90_sigma, 1.60 * sigma_seed)),
        normalize_start(0.65, min(q68_sigma, 0.80 * sigma_seed), max(q95_sigma, 2.20 * sigma_seed)),
    ]

    def pack(a_core: float, sigma1: float, sigma2: float) -> np.ndarray:
        delta = max(float(sigma2) - float(sigma1), delta_min)
        return np.asarray([stable_logit(a_core), math.log(float(sigma1)), math.log(delta)], dtype=np.float64)

    def unpack(params: np.ndarray) -> Tuple[float, float, float]:
        a = stable_sigmoid(float(params[0]))
        s1 = math.exp(float(params[1]))
        s2 = s1 + math.exp(float(params[2]))
        return a, s1, s2

    def residual(params: np.ndarray) -> np.ndarray:
        a, s1, s2 = unpack(params)
        cdf_max = double_rayleigh_cdf_deg(fit_max_deg, a, s1, s2)
        if not np.isfinite(cdf_max) or cdf_max <= 1.0e-6:
            return np.full(x_fit.shape, 1.0e6, dtype=np.float64)
        model = double_rayleigh_pdf_deg(x_fit, a, s1, s2) / cdf_max
        out = (model - y_fit) / err_fit
        return np.where(np.isfinite(out), out, 1.0e6)

    best = None
    for start in starts:
        x0 = np.clip(pack(*start), lower + 1.0e-9, upper - 1.0e-9)
        try:
            result = least_squares(
                residual,
                x0,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=2500,
            )
        except Exception:
            continue
        raw_residual = residual(result.x)
        chi2 = float(np.sum(raw_residual * raw_residual))
        candidate = (chi2, result)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return {"status": "fallback", "reason": "double_rayleigh_optimizer_failed", "profile_effective_events": profile_neff}

    chi2, result = best
    a_fit, sigma1_fit, sigma2_fit = unpack(result.x)
    r_opt_deg = double_rayleigh_radius_for_containment(
        TARGET_CONTAINMENT,
        a_core=a_fit,
        sigma1_deg=sigma1_fit,
        sigma2_deg=sigma2_fit,
    )
    if not (
        result.success
        and np.isfinite(r_opt_deg)
        and r_opt_deg > 0.0
        and np.isfinite(a_fit)
        and 0.0 < a_fit < 1.0
        and np.isfinite(sigma1_fit)
        and np.isfinite(sigma2_fit)
        and 0.0 < sigma1_fit < sigma2_fit
    ):
        return {
            "status": "fallback",
            "reason": f"double_rayleigh_invalid_fit:{result.message}",
            "profile_effective_events": profile_neff,
        }

    ndof = max(0, int(np.count_nonzero(fit_mask)) - 3)
    return {
        "status": "ok",
        "reason": "double_rayleigh_profile_fit",
        "A": float(a_fit),
        "sigma1_deg": float(sigma1_fit),
        "sigma2_deg": float(sigma2_fit),
        "sigma_eq_deg": float(r_opt_deg / RAYLEIGH_OPT_RADIUS_FACTOR),
        "r_opt_deg": float(r_opt_deg),
        "model_containment_r_opt": float(TARGET_CONTAINMENT),
        "chi2": float(chi2),
        "ndof": int(ndof),
        "chi2_ndof": float(chi2 / ndof) if ndof > 0 else None,
        "positive_profile_bins": int(np.count_nonzero(positive_bins)),
        "fit_profile_bins": int(np.count_nonzero(fit_mask)),
        "profile_effective_events": float(profile_neff),
        "optimizer_nfev": int(result.nfev),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
    }


def fit_double_rayleigh_mixture_from_profile_density(
    profile_density: np.ndarray,
    profile_edges_deg: np.ndarray,
    *,
    rayleigh_sigma_deg: float,
) -> Dict[str, object]:
    edges = np.asarray(profile_edges_deg, dtype=np.float64)
    density = np.asarray(profile_density, dtype=np.float64)
    if edges.size != density.size + 1:
        return {"status": "fallback", "reason": "double_rayleigh_source_profile_shape_mismatch"}
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pseudo_weight = np.clip(np.where(np.isfinite(density), density, 0.0), 0.0, None) * widths
    if float(np.sum(pseudo_weight)) <= 0.0:
        return {"status": "fallback", "reason": "double_rayleigh_source_profile_empty"}
    return fit_double_rayleigh_mixture(
        centers,
        pseudo_weight,
        edges,
        rayleigh_sigma_deg=rayleigh_sigma_deg,
    )


def profile_density_containment(profile_density: np.ndarray, profile_edges_deg: np.ndarray, radius_deg: float) -> float:
    edges = np.asarray(profile_edges_deg, dtype=np.float64)
    density = np.asarray(profile_density, dtype=np.float64)
    if edges.size != density.size + 1:
        return float("nan")
    r = float(radius_deg)
    if not np.isfinite(r) or r < 0.0:
        return float("nan")
    widths = np.diff(edges)
    mass = np.clip(np.where(np.isfinite(density), density, 0.0), 0.0, None) * widths
    total = float(np.sum(mass))
    if total <= 0.0:
        return float("nan")
    full = edges[1:] <= r
    contained = float(np.sum(mass[full]))
    partial = np.nonzero((edges[:-1] < r) & (r < edges[1:]))[0]
    if partial.size:
        idx = int(partial[0])
        width = float(widths[idx])
        if width > 0.0:
            contained += float(mass[idx]) * (r - float(edges[idx])) / width
    return float(min(max(contained / total, 0.0), 1.0))


def load_observed_profile_summary(summary_csv: Path) -> Dict[int, Dict[str, str]]:
    if not summary_csv.exists():
        return {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["cell_id"]): dict(row) for row in csv.DictReader(handle) if row.get("cell_id")}


def load_observed_profiles(npz_path: Path, summary_csv: Path) -> Dict[str, object]:
    if not npz_path.exists():
        return {
            "status": "missing",
            "path": str(npz_path),
            "summary_csv": str(summary_csv),
            "by_cell": {},
        }
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"profile_edges_deg", "cell_id", "excess_profile"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{npz_path} is missing observed profile arrays: {sorted(missing)}")
        edges = np.asarray(data["profile_edges_deg"], dtype=np.float64)
        cell_ids = np.asarray(data["cell_id"], dtype=np.int64)
        excess = np.asarray(data["excess_profile"], dtype=np.float64)
    summary_by_cell = load_observed_profile_summary(summary_csv)
    by_cell = {
        int(cell_id): {
            "index": idx,
            "excess_profile": excess[idx],
            "summary": summary_by_cell.get(int(cell_id), {}),
        }
        for idx, cell_id in enumerate(cell_ids)
    }
    return {
        "status": "loaded",
        "path": str(npz_path),
        "summary_csv": str(summary_csv),
        "profile_edges_deg": edges,
        "by_cell": by_cell,
    }


def observed_profile_aperture(
    *,
    cell_id: int,
    observed_profiles: Dict[str, object],
    rayleigh_r_opt_deg: float,
    mc_quantile_r715_deg: float,
    target_containment: float,
    pedestal_min_deg: float,
    max_r_opt_over_rayleigh: float,
    max_r_opt_over_mc_quantile: float,
    max_r_opt_deg: float,
    min_positive_total: float,
    require_reliable: bool,
) -> Dict[str, object]:
    by_cell = observed_profiles.get("by_cell")
    edges = observed_profiles.get("profile_edges_deg")
    if not isinstance(by_cell, dict) or not isinstance(edges, np.ndarray):
        return {"status": "fallback", "reason": "observed_profile_input_missing"}
    payload = by_cell.get(int(cell_id))
    if not isinstance(payload, dict):
        return {"status": "fallback", "reason": "observed_profile_missing_for_cell"}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if require_reliable and str(summary.get("fit_reliable", "0")).strip() not in {"1", "true", "True"}:
        reason = str(summary.get("unreliable_reason") or summary.get("fit_reason") or "empirical_profile_not_reliable")
        return {"status": "fallback", "reason": f"observed_profile_not_reliable:{reason}"}

    profile = np.asarray(payload["excess_profile"], dtype=np.float64)
    edges_arr = np.asarray(edges, dtype=np.float64)
    if profile.size + 1 != edges_arr.size:
        return {"status": "fallback", "reason": "observed_profile_shape_mismatch"}
    raw_r715, raw_positive_total = shell_quantile_radius(edges_arr, profile, target_containment)
    raw_total = float(np.nansum(profile))
    centers = 0.5 * (edges_arr[:-1] + edges_arr[1:])
    shell_area = math.pi * (edges_arr[1:] ** 2 - edges_arr[:-1] ** 2)
    density = np.divide(profile, shell_area, out=np.full_like(profile, np.nan, dtype=np.float64), where=shell_area > 0.0)
    outer = np.isfinite(density) & (centers >= float(pedestal_min_deg))
    if not np.any(outer):
        return {"status": "fallback", "reason": "observed_profile_no_pedestal_region"}
    pedestal_per_deg2 = float(np.nanmedian(density[outer]))
    corrected = profile - pedestal_per_deg2 * shell_area
    r715, positive_total = shell_quantile_radius(edges_arr, corrected, target_containment)
    if not np.isfinite(r715) or r715 <= 0.0:
        return {
            "status": "fallback",
            "reason": "observed_profile_invalid_quantile",
            "raw_r715_deg": raw_r715,
            "raw_positive_total": raw_positive_total,
            "raw_total": raw_total,
            "pedestal_per_deg2": pedestal_per_deg2,
            "positive_total": positive_total,
        }
    if positive_total < float(min_positive_total):
        return {
            "status": "fallback",
            "reason": f"observed_profile_positive_total_below_min:{positive_total:.6g}<{min_positive_total}",
            "raw_r715_deg": raw_r715,
            "raw_positive_total": raw_positive_total,
            "raw_total": raw_total,
            "pedestal_per_deg2": pedestal_per_deg2,
            "positive_total": positive_total,
            "r715_deg": r715,
        }
    ratio_rayleigh = float(r715 / rayleigh_r_opt_deg) if rayleigh_r_opt_deg > 0.0 else float("nan")
    ratio_mc = float(r715 / mc_quantile_r715_deg) if np.isfinite(mc_quantile_r715_deg) and mc_quantile_r715_deg > 0.0 else float("nan")
    if np.isfinite(ratio_rayleigh) and ratio_rayleigh > float(max_r_opt_over_rayleigh):
        return {
            "status": "fallback",
            "reason": f"observed_profile_r_opt_over_rayleigh:{ratio_rayleigh:.6g}>{max_r_opt_over_rayleigh}",
            "raw_r715_deg": raw_r715,
            "raw_positive_total": raw_positive_total,
            "raw_total": raw_total,
            "pedestal_per_deg2": pedestal_per_deg2,
            "positive_total": positive_total,
            "r715_deg": r715,
            "r_opt_over_rayleigh": ratio_rayleigh,
            "r_opt_over_mc_quantile": ratio_mc,
        }
    if np.isfinite(ratio_mc) and np.isfinite(max_r_opt_over_mc_quantile) and ratio_mc > float(max_r_opt_over_mc_quantile):
        return {
            "status": "fallback",
            "reason": f"observed_profile_r_opt_over_mc_quantile:{ratio_mc:.6g}>{max_r_opt_over_mc_quantile}",
            "raw_r715_deg": raw_r715,
            "raw_positive_total": raw_positive_total,
            "raw_total": raw_total,
            "pedestal_per_deg2": pedestal_per_deg2,
            "positive_total": positive_total,
            "r715_deg": r715,
            "r_opt_over_rayleigh": ratio_rayleigh,
            "r_opt_over_mc_quantile": ratio_mc,
        }
    if r715 > float(max_r_opt_deg):
        return {
            "status": "fallback",
            "reason": f"observed_profile_r_opt_deg:{r715:.6g}>{max_r_opt_deg}",
            "raw_r715_deg": raw_r715,
            "raw_positive_total": raw_positive_total,
            "raw_total": raw_total,
            "pedestal_per_deg2": pedestal_per_deg2,
            "positive_total": positive_total,
            "r715_deg": r715,
            "r_opt_over_rayleigh": ratio_rayleigh,
            "r_opt_over_mc_quantile": ratio_mc,
        }
    return {
        "status": "ok",
        "reason": "observed_profile_quantile",
        "r715_deg": r715,
        "containment": float(target_containment),
        "raw_r715_deg": raw_r715,
        "raw_positive_total": raw_positive_total,
        "raw_total": raw_total,
        "pedestal_per_deg2": pedestal_per_deg2,
        "positive_total": positive_total,
        "r_opt_over_rayleigh": ratio_rayleigh,
        "r_opt_over_mc_quantile": ratio_mc,
        "fit_reliable": summary.get("fit_reliable", ""),
        "significance": summary.get("significance", ""),
    }


def corrected_observed_profile_density(
    *,
    cell_id: int,
    observed_profiles: Dict[str, object],
    target_edges_deg: np.ndarray,
    pedestal_min_deg: float,
) -> Optional[np.ndarray]:
    by_cell = observed_profiles.get("by_cell")
    source_edges = observed_profiles.get("profile_edges_deg")
    if not isinstance(by_cell, dict) or not isinstance(source_edges, np.ndarray):
        return None
    payload = by_cell.get(int(cell_id))
    if not isinstance(payload, dict):
        return None
    profile = np.asarray(payload.get("excess_profile"), dtype=np.float64)
    source_edges_arr = np.asarray(source_edges, dtype=np.float64)
    target_edges_arr = np.asarray(target_edges_deg, dtype=np.float64)
    if profile.size + 1 != source_edges_arr.size or target_edges_arr.size < 2:
        return None

    source_centers = 0.5 * (source_edges_arr[:-1] + source_edges_arr[1:])
    source_area = math.pi * (source_edges_arr[1:] ** 2 - source_edges_arr[:-1] ** 2)
    source_density_area = np.divide(
        profile,
        source_area,
        out=np.full_like(profile, np.nan, dtype=np.float64),
        where=source_area > 0.0,
    )
    outer = np.isfinite(source_density_area) & (source_centers >= float(pedestal_min_deg))
    if not np.any(outer):
        return None
    pedestal_per_deg2 = float(np.nanmedian(source_density_area[outer]))
    corrected_mass = np.clip(np.where(np.isfinite(profile), profile, 0.0) - pedestal_per_deg2 * source_area, 0.0, None)
    total = float(np.sum(corrected_mass))
    if total <= 0.0:
        return None

    target_mass = np.zeros(target_edges_arr.size - 1, dtype=np.float64)
    source_width = np.diff(source_edges_arr)
    valid_source = source_width > 0.0
    for src_idx, ok in enumerate(valid_source):
        if not ok or corrected_mass[src_idx] <= 0.0:
            continue
        lo = float(source_edges_arr[src_idx])
        hi = float(source_edges_arr[src_idx + 1])
        per_deg = float(corrected_mass[src_idx] / source_width[src_idx])
        overlap = np.minimum(target_edges_arr[1:], hi) - np.maximum(target_edges_arr[:-1], lo)
        overlap = np.clip(overlap, 0.0, None)
        target_mass += per_deg * overlap

    target_width = np.diff(target_edges_arr)
    density = np.divide(target_mass, total * target_width, out=np.zeros_like(target_mass), where=target_width > 0.0)
    return density


def replace_observed_profile_density(
    rows: Sequence[Dict[str, object]],
    profile_density: np.ndarray,
    *,
    observed_profiles: Dict[str, object],
    profile_edges_deg: np.ndarray,
    pedestal_min_deg: float,
) -> None:
    for idx, row in enumerate(rows):
        if bool(row.get("observed_data_fallback")):
            row["observed_data_profile_source"] = "fallback_mc_profile"
            continue
        density = corrected_observed_profile_density(
            cell_id=int(row["cell_id"]),
            observed_profiles=observed_profiles,
            target_edges_deg=profile_edges_deg,
            pedestal_min_deg=pedestal_min_deg,
        )
        if density is None:
            row["observed_data_profile_source"] = "fallback_mc_profile_no_observed_density"
            continue
        profile_density[idx] = density.astype(profile_density.dtype)
        row["observed_data_profile_source"] = "pedestal_subtracted_observed_excess"


def add_method_fields(
    row: Dict[str, object],
    *,
    psf_method: str,
    r715_deg: float,
    sigma_eff_deg: float,
    sigma_x_deg: float,
    sigma_y_deg: float,
    mu_x_deg: float,
    mu_y_deg: float,
    mc_quantile_r715_deg: float,
    fit_quality: str,
    containment_r_opt: float,
) -> None:
    row["psf_method"] = psf_method
    row["target_containment"] = float(TARGET_CONTAINMENT)
    row["r715_deg"] = float(r715_deg)
    row["sigma_eff_deg"] = float(sigma_eff_deg)
    row["sigma_x_deg"] = float(sigma_x_deg)
    row["sigma_y_deg"] = float(sigma_y_deg)
    row["sigma_x_over_y"] = float(sigma_x_deg / sigma_y_deg) if sigma_y_deg > 0.0 else None
    row["mu_x_deg"] = float(mu_x_deg)
    row["mu_y_deg"] = float(mu_y_deg)
    row["mc_quantile_r715_deg"] = float(mc_quantile_r715_deg)
    row["fit_quality"] = fit_quality
    row["containment_r_opt"] = float(containment_r_opt)
    row["containment_r_opt_core_fit_full_distribution"] = float(containment_r_opt)
    row["containment_minus_expected"] = float(containment_r_opt - TARGET_CONTAINMENT)


def method_choice(method: str, row: Dict[str, object]) -> Tuple[float, float]:
    if method == "rayleigh_baseline":
        r_opt = finite_float(row.get("rayleigh_baseline_r715_deg"))
        containment = finite_float(row.get("rayleigh_baseline_containment_r_opt"))
    elif method == "two_1d_gaussian":
        r_opt = finite_float(row.get("two_1d_gaussian_r715_deg"))
        containment = finite_float(row.get("two_1d_gaussian_containment_r_opt"))
    elif method == "mc_quantile_715":
        r_opt = finite_float(row.get("mc_quantile_r715_deg"))
        containment = finite_float(row.get("mc_quantile_containment_r_opt"))
    elif method == "observed_data":
        r_opt = finite_float(row.get("observed_data_r715_deg"))
        containment = finite_float(row.get("observed_data_containment_r_opt"))
    elif method == "double_rayleigh_mixture":
        r_opt = finite_float(row.get("double_rayleigh_r_opt_deg"))
        containment = finite_float(row.get("double_rayleigh_containment_r_opt"))
    else:
        raise ValueError(f"Unsupported PSF method: {method}")
    if r_opt is None or r_opt <= 0.0:
        raise ValueError(f"Row for cell {row.get('cell_id')} has invalid {method} radius: {r_opt}")
    if containment is None or containment <= 0.0:
        containment = TARGET_CONTAINMENT
    return float(r_opt), float(containment)


def apply_method_to_row(row: Dict[str, object], method: str, *, containment_warning_tolerance: float) -> Dict[str, object]:
    out = dict(row)
    r_opt_deg, containment = method_choice(method, out)
    out["psf_method"] = method
    out["r_opt_deg"] = float(r_opt_deg)
    out["r_opt_rad"] = math.radians(float(r_opt_deg))
    out["r715_deg"] = float(r_opt_deg)
    out["containment_r_opt"] = float(containment)
    out["containment_r_opt_core_fit_full_distribution"] = float(containment)
    out["containment_minus_expected"] = float(containment - TARGET_CONTAINMENT)
    out["containment_warning"] = abs(float(containment - TARGET_CONTAINMENT)) > float(containment_warning_tolerance)
    if method == "two_1d_gaussian":
        out["fit_quality"] = row.get("two_1d_gaussian_fit_quality", row.get("fit_quality", "ok"))
    elif method == "mc_quantile_715":
        out["fit_quality"] = row.get("mc_quantile_fit_quality", row.get("fit_quality", "ok"))
    elif method == "observed_data":
        out["fit_quality"] = row.get("observed_data_fit_quality", row.get("fit_quality", "ok"))
    elif method == "double_rayleigh_mixture":
        out["fit_quality"] = row.get("double_rayleigh_fit_quality", row.get("fit_quality", "ok"))
        sigma_eq = finite_float(row.get("double_rayleigh_sigma_eq_deg"))
        if sigma_eq is not None:
            out["sigma_eff_deg"] = float(sigma_eq)
    else:
        out["fit_quality"] = row.get("rayleigh_fit_quality", row.get("fit_quality", "ok"))
    if out.get("psf_borrowed"):
        out["fit_quality"] = "borrowed_neighbor_psf"
        out["psf_quality_flag"] = "borrowed"
    else:
        out["psf_quality_flag"] = "warning" if (out.get("containment_warning") or out.get("angle_check_warning") or out.get("fit_quality") != "ok") else "ok"
    return out


def fallback_row(
    cell,
    *,
    cell_dir: Path,
    input_files: int,
    events: int,
    reason: str,
    profile_edges_deg: np.ndarray,
    psf_method: str,
) -> Tuple[Dict[str, object], np.ndarray]:
    row, profile = stage02.fallback_psf_row(
        cell,
        cell_dir=cell_dir,
        input_files=input_files,
        events=events,
        reason=reason,
        profile_edges_deg=profile_edges_deg,
    )
    sigma_deg = finite_float(row.get("sigma_deg")) or 1.0
    r_opt_deg = finite_float(row.get("r_opt_deg")) or (RAYLEIGH_OPT_RADIUS_FACTOR * sigma_deg)
    row["rayleigh_baseline_r715_deg"] = float(r_opt_deg)
    row["rayleigh_baseline_containment_r_opt"] = float(TARGET_CONTAINMENT)
    row["two_1d_gaussian_r715_deg"] = float(r_opt_deg)
    row["two_1d_gaussian_containment_r_opt"] = float(TARGET_CONTAINMENT)
    row["mc_quantile_containment_r_opt"] = float(TARGET_CONTAINMENT)
    row["observed_data_r715_deg"] = float(r_opt_deg)
    row["observed_data_containment_r_opt"] = float(TARGET_CONTAINMENT)
    row["observed_data_fit_quality"] = f"fallback:{reason}"
    row["observed_data_fallback"] = True
    row["observed_data_fallback_reason"] = reason
    row["observed_data_profile_source"] = "fallback_low_stat"
    row["double_rayleigh_A"] = None
    row["double_rayleigh_sigma1_deg"] = float(sigma_deg)
    row["double_rayleigh_sigma2_deg"] = None
    row["double_rayleigh_sigma_eq_deg"] = float(r_opt_deg / RAYLEIGH_OPT_RADIUS_FACTOR)
    row["double_rayleigh_r_opt_deg"] = float(r_opt_deg)
    row["double_rayleigh_containment_r_opt"] = float(TARGET_CONTAINMENT)
    row["double_rayleigh_model_containment_r_opt"] = float(TARGET_CONTAINMENT)
    row["double_rayleigh_fit_quality"] = f"fallback:{reason}"
    row["double_rayleigh_fallback_reason"] = reason
    row["double_rayleigh_chi2"] = None
    row["double_rayleigh_ndof"] = None
    row["double_rayleigh_chi2_ndof"] = None
    row["double_rayleigh_positive_profile_bins"] = None
    row["double_rayleigh_fit_profile_bins"] = None
    row["double_rayleigh_profile_effective_events"] = None
    row["rayleigh_fit_quality"] = f"fallback:{reason}"
    row["two_1d_gaussian_fit_quality"] = f"fallback:{reason}"
    row["mc_quantile_fit_quality"] = f"fallback:{reason}"
    add_method_fields(
        row,
        psf_method=psf_method if psf_method in PSF_METHODS else "rayleigh_baseline",
        r715_deg=r_opt_deg,
        sigma_eff_deg=sigma_deg,
        sigma_x_deg=sigma_deg,
        sigma_y_deg=sigma_deg,
        mu_x_deg=0.0,
        mu_y_deg=0.0,
        mc_quantile_r715_deg=r_opt_deg,
        fit_quality=f"fallback:{reason}",
        containment_r_opt=TARGET_CONTAINMENT,
    )
    row["r_opt_deg"] = float(r_opt_deg)
    row["r_opt_rad"] = math.radians(float(r_opt_deg))
    row["psf_quality_flag"] = "fallback_low_stat"
    return row, profile


def process_cell_v5(cell, kwargs: Dict[str, object]) -> Tuple[int, Dict[str, object], np.ndarray]:
    row, profile = _process_cell_v5(cell, **kwargs)
    return int(cell.index), row, profile


def _process_cell_v5(
    cell,
    *,
    psf_method: str,
    binned_root: Path,
    tree_name: str,
    weight_branch: str,
    allow_missing_weight: bool,
    max_files_per_cell: Optional[int],
    allow_missing_cell_dirs: bool,
    theta_edges_deg: np.ndarray,
    crab_prob: np.ndarray,
    loge_min: float,
    loge_max: float,
    allow_incomplete_theta_support: bool,
    min_events_per_cell: int,
    min_effective_events: float,
    allow_low_stat_psf_fallback: bool,
    core_fit_max_deg: float,
    theta_missing_mass_fail_threshold: float,
    containment_warning_tolerance: float,
    angle_check_max_events: int,
    angle_check_warn_rad: float,
    file_progress_every: int,
    profile_edges_deg: np.ndarray,
    two1d_radial_quadrature: int,
    two1d_angle_samples: int,
    observed_profiles: Dict[str, object],
    observed_pedestal_min_deg: float,
    observed_max_r_opt_over_rayleigh: float,
    observed_max_r_opt_over_mc_quantile: float,
    observed_max_r_opt_deg: float,
    observed_min_positive_total: float,
    observed_require_reliable: bool,
) -> Tuple[Dict[str, object], np.ndarray]:
    cell_dir = stage02.binned_cell_dir(binned_root, cell)
    files = stage02.discover_cell_files(
        cell_dir,
        max_files_per_cell,
        allow_missing_cell_dirs=allow_missing_cell_dirs,
    )
    if not files and allow_low_stat_psf_fallback:
        return fallback_row(
            cell,
            cell_dir=cell_dir,
            input_files=0,
            events=0,
            reason="no_input_files_for_cell",
            profile_edges_deg=profile_edges_deg,
            psf_method=psf_method,
        )

    events = read_extended_cell_events(
        files,
        tree_name=tree_name,
        weight_branch=weight_branch,
        allow_missing_weight=allow_missing_weight,
        angle_check_max_events=angle_check_max_events,
        file_progress_every=file_progress_every,
        progress_label=f"cell {cell.cell_id}",
    )

    n_events = int(events.dangle_rad.size)
    if n_events < int(min_events_per_cell):
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"events_below_min_events_per_cell:{n_events}<{min_events_per_cell}",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(f"Cell {cell.cell_id} has {n_events} events, below --min-events-per-cell={min_events_per_cell}")

    loge_valid = np.isfinite(events.loge_true) & (events.loge_true >= float(loge_min)) & (events.loge_true < float(loge_max))
    ratio_support = (
        loge_valid
        & np.isfinite(events.dangle_rad)
        & (events.dangle_rad >= 0.0)
        & np.isfinite(events.mc_weight)
        & (events.mc_weight > 0.0)
    )
    theta_ratio, theta_meta = stage02.theta_reweight_ratio(
        events.mc_theta_deg,
        events.mc_weight,
        theta_edges_deg,
        crab_prob,
        support_mask=ratio_support,
        allow_incomplete_theta_support=allow_incomplete_theta_support,
    )
    missing_crab_mass = float(theta_meta.get("missing_crab_probability_mass") or 0.0)
    if missing_crab_mass > float(theta_missing_mass_fail_threshold):
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"theta_missing_crab_probability_mass:{missing_crab_mass:.6g}>{theta_missing_mass_fail_threshold}",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(
            f"Cell {cell.cell_id} is missing Crab theta support mass {missing_crab_mass:.4g}, "
            f"above --theta-missing-mass-fail-threshold={theta_missing_mass_fail_threshold}"
        )

    theta_idx, theta_valid = stage02.theta_bin_indices(events.mc_theta_deg, theta_edges_deg)
    base_valid = theta_valid & ratio_support
    full_weight = np.zeros(n_events, dtype=np.float64)
    full_weight[base_valid] = events.mc_weight[base_valid] * theta_ratio[theta_idx[base_valid]]
    mc_weight_only = np.where(base_valid, events.mc_weight, 0.0)
    unweighted = np.where(base_valid, 1.0, 0.0)
    positive_full = full_weight > 0.0
    sumw_full = float(np.sum(full_weight[positive_full]))
    if sumw_full <= 0.0:
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason="no_positive_baseline_weight_after_theta_reweighting",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(f"Cell {cell.cell_id} has no positive baseline weight after Crab theta reweighting.")

    neff = stage02.effective_event_count(full_weight)
    if neff < float(min_effective_events):
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"effective_events_below_min:{neff:.6g}<{min_effective_events}",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(f"Cell {cell.cell_id} has effective events {neff:.3g}, below --min-effective-events={min_effective_events}")

    core_fit_max_rad = math.radians(float(core_fit_max_deg))
    core_mask = positive_full & np.isfinite(events.dangle_rad) & (events.dangle_rad <= core_fit_max_rad)
    core_weight = np.where(core_mask, full_weight, 0.0)
    mc_weight_core = np.where(core_mask, events.mc_weight, 0.0)
    unweighted_core = np.where(core_mask, 1.0, 0.0)
    core_sumw = float(np.sum(core_weight[core_weight > 0.0]))
    core_neff = stage02.effective_event_count(core_weight)
    if core_sumw <= 0.0:
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason="no_positive_baseline_weight_inside_core_fit_range",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(f"Cell {cell.cell_id} has no positive baseline weight inside core fit range.")
    if core_neff < float(min_effective_events):
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"core_effective_events_below_min:{core_neff:.6g}<{min_effective_events}",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(f"Cell {cell.cell_id} has core effective events {core_neff:.3g}, below --min-effective-events={min_effective_events}")

    sigma_rad = stage02.rayleigh_sigma_mle(events.dangle_rad, core_weight)
    sigma_mc_weight_rad = stage02.rayleigh_sigma_mle(events.dangle_rad, mc_weight_core)
    sigma_unweighted_rad = stage02.rayleigh_sigma_mle(events.dangle_rad, unweighted_core)
    sigma_full_rad = stage02.rayleigh_sigma_mle(events.dangle_rad, full_weight)
    sigma_full_mc_weight_rad = stage02.rayleigh_sigma_mle(events.dangle_rad, mc_weight_only)
    sigma_full_unweighted_rad = stage02.rayleigh_sigma_mle(events.dangle_rad, unweighted)
    if not np.isfinite(sigma_rad) or sigma_rad <= 0.0:
        if allow_low_stat_psf_fallback:
            return fallback_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"invalid_baseline_sigma:{sigma_rad}",
                profile_edges_deg=profile_edges_deg,
                psf_method=psf_method,
            )
        raise ValueError(f"Cell {cell.cell_id} has invalid baseline sigma: {sigma_rad}")

    x_deg, y_deg = tangent_offsets_deg(
        events.reco_theta_rad,
        events.reco_phi_rad,
        events.mc_theta_rad,
        events.mc_phi_rad,
    )
    mu_x_deg, sigma_x_deg = weighted_mean_sigma(x_deg, core_weight)
    mu_y_deg, sigma_y_deg = weighted_mean_sigma(y_deg, core_weight)
    if not (np.isfinite(sigma_x_deg) and sigma_x_deg > 0.0 and np.isfinite(sigma_y_deg) and sigma_y_deg > 0.0):
        mu_x_deg = 0.0
        mu_y_deg = 0.0
        sigma_x_deg = math.degrees(sigma_rad)
        sigma_y_deg = math.degrees(sigma_rad)
        fit_quality = "warning_invalid_two1d_fit_fell_back_to_rayleigh_sigma"
    else:
        fit_quality = "ok"
    sigma_eff_deg = math.sqrt(0.5 * (sigma_x_deg * sigma_x_deg + sigma_y_deg * sigma_y_deg))
    mc_quantile_rad = stage02.weighted_quantile(events.dangle_rad, [TARGET_CONTAINMENT], full_weight)[0]
    mc_quantile_r715_deg = float(math.degrees(mc_quantile_rad)) if np.isfinite(mc_quantile_rad) else float("nan")

    rayleigh_r_opt_deg = float(math.degrees(RAYLEIGH_OPT_RADIUS_FACTOR * sigma_rad))
    two1d_r715_deg = gaussian_radius_for_containment(
        TARGET_CONTAINMENT,
        mu_x_deg=float(mu_x_deg),
        mu_y_deg=float(mu_y_deg),
        sigma_x_deg=float(sigma_x_deg),
        sigma_y_deg=float(sigma_y_deg),
        radial_quadrature=two1d_radial_quadrature,
        angle_samples=two1d_angle_samples,
    )
    double_fit = fit_double_rayleigh_mixture(
        np.degrees(events.dangle_rad),
        full_weight,
        profile_edges_deg,
        rayleigh_sigma_deg=float(math.degrees(sigma_rad)),
    )
    double_fit_ok = double_fit.get("status") == "ok"
    double_rayleigh_r_opt_deg = (
        float(double_fit["r_opt_deg"])
        if double_fit_ok and finite_float(double_fit.get("r_opt_deg")) is not None
        else float(rayleigh_r_opt_deg)
    )
    double_rayleigh_fit_quality = (
        "ok"
        if double_fit_ok
        else f"fallback:{double_fit.get('reason', 'double_rayleigh_fit_failed')}"
    )

    rayleigh_containment = empirical_containment(events.dangle_rad, full_weight, math.radians(rayleigh_r_opt_deg))
    two1d_containment = empirical_containment(events.dangle_rad, full_weight, math.radians(two1d_r715_deg))
    mc_quantile_containment = (
        empirical_containment(events.dangle_rad, full_weight, math.radians(mc_quantile_r715_deg))
        if np.isfinite(mc_quantile_r715_deg) and mc_quantile_r715_deg > 0.0
        else float("nan")
    )
    double_rayleigh_containment = empirical_containment(events.dangle_rad, full_weight, math.radians(double_rayleigh_r_opt_deg))
    observed_aperture = observed_profile_aperture(
        cell_id=int(cell.cell_id),
        observed_profiles=observed_profiles,
        rayleigh_r_opt_deg=float(rayleigh_r_opt_deg),
        mc_quantile_r715_deg=float(mc_quantile_r715_deg),
        target_containment=float(TARGET_CONTAINMENT),
        pedestal_min_deg=float(observed_pedestal_min_deg),
        max_r_opt_over_rayleigh=float(observed_max_r_opt_over_rayleigh),
        max_r_opt_over_mc_quantile=float(observed_max_r_opt_over_mc_quantile),
        max_r_opt_deg=float(observed_max_r_opt_deg),
        min_positive_total=float(observed_min_positive_total),
        require_reliable=bool(observed_require_reliable),
    )
    observed_r715_deg = finite_float(observed_aperture.get("r715_deg"))
    observed_data_r715_deg = float(observed_r715_deg) if observed_aperture.get("status") == "ok" and observed_r715_deg else float(rayleigh_r_opt_deg)
    observed_data_containment = float(observed_aperture.get("containment") or TARGET_CONTAINMENT)

    active_method = psf_method if psf_method in PSF_METHODS else "rayleigh_baseline"
    if active_method == "rayleigh_baseline":
        r_opt_deg = rayleigh_r_opt_deg
        r715_deg = rayleigh_r_opt_deg
    elif active_method == "two_1d_gaussian":
        r_opt_deg = two1d_r715_deg
        r715_deg = two1d_r715_deg
    elif active_method == "mc_quantile_715":
        if not np.isfinite(mc_quantile_r715_deg) or mc_quantile_r715_deg <= 0.0:
            if allow_low_stat_psf_fallback:
                return fallback_row(
                    cell,
                    cell_dir=cell_dir,
                    input_files=events.input_files,
                    events=n_events,
                    reason="invalid_mc_quantile_r715",
                    profile_edges_deg=profile_edges_deg,
                    psf_method=active_method,
                )
            raise ValueError(f"Cell {cell.cell_id} has invalid MC quantile r715: {mc_quantile_r715_deg}")
        r_opt_deg = mc_quantile_r715_deg
        r715_deg = mc_quantile_r715_deg
    elif active_method == "observed_data":
        if observed_aperture.get("status") != "ok":
            if allow_low_stat_psf_fallback:
                r_opt_deg = rayleigh_r_opt_deg
                r715_deg = rayleigh_r_opt_deg
            else:
                raise ValueError(f"Cell {cell.cell_id} has invalid observed-data aperture: {observed_aperture}")
        else:
            r_opt_deg = observed_data_r715_deg
            r715_deg = observed_data_r715_deg
    elif active_method == "double_rayleigh_mixture":
        r_opt_deg = double_rayleigh_r_opt_deg
        r715_deg = double_rayleigh_r_opt_deg
    else:
        raise ValueError(f"Unsupported psf_method: {psf_method}")

    r_opt_rad = math.radians(float(r_opt_deg))
    containment = empirical_containment(events.dangle_rad, full_weight, r_opt_rad)
    if not np.isfinite(containment) or containment <= 0.0 or containment > 1.0:
        raise ValueError(f"Cell {cell.cell_id} has invalid r_opt containment: {containment}")

    containment_warning = abs(containment - TARGET_CONTAINMENT) > float(containment_warning_tolerance)
    quantiles_rad = stage02.weighted_quantile(events.dangle_rad, [0.68, 0.90, 0.95], full_weight)
    core_quantiles_rad = stage02.weighted_quantile(events.dangle_rad, [0.68, 0.90, 0.95], core_weight)
    r_deg = np.degrees(events.dangle_rad)
    profile_density = stage02.profile_histogram(r_deg, full_weight, profile_edges_deg)
    angle_check_percentiles = stage02.finite_percentiles(events.angle_check_absdiff_rad, [50.0, 90.0, 99.0, 100.0])
    angle_check_warning = angle_check_percentiles[0] is not None and float(angle_check_percentiles[0]) > float(angle_check_warn_rad)
    loge_percentiles = stage02.finite_percentiles(events.loge_true[loge_valid], [5.0, 50.0, 95.0])

    row: Dict[str, object] = {
        "cell_index": int(cell.index),
        "cell_id": int(cell.cell_id),
        "nhit_bin": cell.nhit_bin,
        "predE_bin": cell.predE_bin,
        "input_dir": str(cell_dir),
        "input_files": int(events.input_files),
        "events": n_events,
        "logE_range_events": int(np.count_nonzero(loge_valid)),
        "valid_events": int(np.count_nonzero(base_valid)),
        "positive_baseline_weight_events": int(np.count_nonzero(positive_full)),
        "sumw_baseline": sumw_full,
        "sumw_mc_weight": float(np.sum(mc_weight_only[mc_weight_only > 0])),
        "effective_events": float(neff),
        "core_fit_max_deg": float(core_fit_max_deg),
        "core_fit_events": int(np.count_nonzero(core_mask)),
        "core_fit_sumw": core_sumw,
        "core_fit_effective_events": float(core_neff),
        "core_fit_weight_fraction": float(core_sumw / sumw_full),
        "tail_weight_fraction_above_core_fit": float(max(0.0, 1.0 - core_sumw / sumw_full)),
        "sigma_rad": float(sigma_rad),
        "sigma_deg": float(math.degrees(sigma_rad)),
        "sigma_mc_weight_deg": float(math.degrees(sigma_mc_weight_rad)) if np.isfinite(sigma_mc_weight_rad) else None,
        "sigma_unweighted_deg": float(math.degrees(sigma_unweighted_rad)) if np.isfinite(sigma_unweighted_rad) else None,
        "sigma_full_rayleigh_rad": float(sigma_full_rad) if np.isfinite(sigma_full_rad) else None,
        "sigma_full_rayleigh_deg": float(math.degrees(sigma_full_rad)) if np.isfinite(sigma_full_rad) else None,
        "sigma_full_mc_weight_deg": float(math.degrees(sigma_full_mc_weight_rad)) if np.isfinite(sigma_full_mc_weight_rad) else None,
        "sigma_full_unweighted_deg": float(math.degrees(sigma_full_unweighted_rad)) if np.isfinite(sigma_full_unweighted_rad) else None,
        "r_opt_rad": float(r_opt_rad),
        "r_opt_deg": float(r_opt_deg),
        "r_opt_factor": float(RAYLEIGH_OPT_RADIUS_FACTOR),
        "containment_r_opt": containment,
        "containment_r_opt_core_fit_full_distribution": containment,
        "rayleigh_expected_containment_r_opt": float(TARGET_CONTAINMENT),
        "containment_minus_expected": float(containment - TARGET_CONTAINMENT),
        "containment_warning": bool(containment_warning),
        "r68_deg": float(math.degrees(quantiles_rad[0])) if np.isfinite(quantiles_rad[0]) else None,
        "r90_deg": float(math.degrees(quantiles_rad[1])) if np.isfinite(quantiles_rad[1]) else None,
        "r95_deg": float(math.degrees(quantiles_rad[2])) if np.isfinite(quantiles_rad[2]) else None,
        "core_r68_deg": float(math.degrees(core_quantiles_rad[0])) if np.isfinite(core_quantiles_rad[0]) else None,
        "core_r90_deg": float(math.degrees(core_quantiles_rad[1])) if np.isfinite(core_quantiles_rad[1]) else None,
        "core_r95_deg": float(math.degrees(core_quantiles_rad[2])) if np.isfinite(core_quantiles_rad[2]) else None,
        "mc_logE_true_p05": loge_percentiles[0],
        "mc_logE_true_p50": loge_percentiles[1],
        "mc_logE_true_p95": loge_percentiles[2],
        "theta_missing_crab_probability_mass": missing_crab_mass,
        "theta_reweight": theta_meta,
        "angle_check_absdiff_rad_p50": angle_check_percentiles[0],
        "angle_check_absdiff_rad_p90": angle_check_percentiles[1],
        "angle_check_absdiff_rad_p99": angle_check_percentiles[2],
        "angle_check_absdiff_rad_max": angle_check_percentiles[3],
        "angle_check_warning": bool(angle_check_warning),
        "psf_quality_flag": "warning" if (containment_warning or angle_check_warning or fit_quality != "ok") else "ok",
        "warnings": [],
        "rayleigh_baseline_r715_deg": float(rayleigh_r_opt_deg),
        "rayleigh_baseline_containment_r_opt": float(rayleigh_containment),
        "two_1d_gaussian_r715_deg": float(two1d_r715_deg),
        "two_1d_gaussian_containment_r_opt": float(two1d_containment),
        "mc_quantile_containment_r_opt": float(mc_quantile_containment),
        "observed_data_r715_deg": float(observed_data_r715_deg),
        "observed_data_containment_r_opt": float(observed_data_containment),
        "observed_data_fit_quality": (
            "ok" if observed_aperture.get("status") == "ok" else f"fallback:{observed_aperture.get('reason', 'observed_profile_invalid')}"
        ),
        "observed_data_fallback": bool(observed_aperture.get("status") != "ok"),
        "observed_data_fallback_reason": "" if observed_aperture.get("status") == "ok" else str(observed_aperture.get("reason", "")),
        "observed_data_raw_r715_deg": finite_float(observed_aperture.get("raw_r715_deg")),
        "observed_data_positive_total": finite_float(observed_aperture.get("positive_total")),
        "observed_data_raw_positive_total": finite_float(observed_aperture.get("raw_positive_total")),
        "observed_data_raw_total": finite_float(observed_aperture.get("raw_total")),
        "observed_data_pedestal_per_deg2": finite_float(observed_aperture.get("pedestal_per_deg2")),
        "observed_data_r_opt_over_rayleigh": finite_float(observed_aperture.get("r_opt_over_rayleigh")),
        "observed_data_r_opt_over_mc_quantile": finite_float(observed_aperture.get("r_opt_over_mc_quantile")),
        "double_rayleigh_A": finite_float(double_fit.get("A")),
        "double_rayleigh_sigma1_deg": (
            finite_float(double_fit.get("sigma1_deg"))
            if double_fit_ok
            else float(math.degrees(sigma_rad))
        ),
        "double_rayleigh_sigma2_deg": finite_float(double_fit.get("sigma2_deg")),
        "double_rayleigh_sigma_eq_deg": (
            finite_float(double_fit.get("sigma_eq_deg"))
            if double_fit_ok
            else float(double_rayleigh_r_opt_deg / RAYLEIGH_OPT_RADIUS_FACTOR)
        ),
        "double_rayleigh_r_opt_deg": float(double_rayleigh_r_opt_deg),
        "double_rayleigh_containment_r_opt": float(double_rayleigh_containment),
        "double_rayleigh_model_containment_r_opt": finite_float(double_fit.get("model_containment_r_opt")) or float(TARGET_CONTAINMENT),
        "double_rayleigh_fit_quality": double_rayleigh_fit_quality,
        "double_rayleigh_fallback_reason": "" if double_fit_ok else str(double_fit.get("reason", "double_rayleigh_fit_failed")),
        "double_rayleigh_chi2": finite_float(double_fit.get("chi2")),
        "double_rayleigh_ndof": finite_float(double_fit.get("ndof")),
        "double_rayleigh_chi2_ndof": finite_float(double_fit.get("chi2_ndof")),
        "double_rayleigh_positive_profile_bins": finite_float(double_fit.get("positive_profile_bins")),
        "double_rayleigh_fit_profile_bins": finite_float(double_fit.get("fit_profile_bins")),
        "double_rayleigh_profile_effective_events": finite_float(double_fit.get("profile_effective_events")),
        "rayleigh_fit_quality": "ok",
        "two_1d_gaussian_fit_quality": fit_quality,
        "mc_quantile_fit_quality": "ok" if np.isfinite(mc_quantile_r715_deg) and mc_quantile_r715_deg > 0.0 else "invalid_mc_quantile_r715",
    }
    method_sigma_eff_deg = sigma_eff_deg
    method_fit_quality = fit_quality
    if active_method == "double_rayleigh_mixture":
        method_sigma_eff_deg = float(row["double_rayleigh_sigma_eq_deg"])
        method_fit_quality = double_rayleigh_fit_quality
    add_method_fields(
        row,
        psf_method=active_method,
        r715_deg=r715_deg,
        sigma_eff_deg=method_sigma_eff_deg,
        sigma_x_deg=sigma_x_deg,
        sigma_y_deg=sigma_y_deg,
        mu_x_deg=mu_x_deg,
        mu_y_deg=mu_y_deg,
        mc_quantile_r715_deg=mc_quantile_r715_deg,
        fit_quality=method_fit_quality,
        containment_r_opt=containment,
    )
    return row, profile_density


def metric_array(rows: Sequence[Dict[str, object]], key: str, *, default: float = np.nan) -> np.ndarray:
    out = []
    for row in rows:
        value = finite_float(row.get(key))
        out.append(default if value is None else value)
    return np.asarray(out, dtype=np.float64)


def string_array(rows: Sequence[Dict[str, object]], key: str, *, width: int = 96) -> np.ndarray:
    return np.asarray([str(row.get(key, "")) for row in rows], dtype=f"U{width}")


def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def weighted_borrow_value(rows_by_cell: Dict[int, Dict[str, object]], sources: Sequence[int], weights: Sequence[float], key: str) -> Optional[float]:
    values = [finite_float(rows_by_cell[int(source)].get(key)) for source in sources]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) * float(weight) for value, weight in zip(values, weights)))


def resolve_borrow_spec(
    spec: Dict[str, object],
    rows_by_cell: Dict[int, Dict[str, object]],
    rows_by_bin: Dict[Tuple[str, str], Dict[str, object]],
) -> Optional[Tuple[int, List[int], List[float]]]:
    target_row: Optional[Dict[str, object]] = None
    target_id = spec.get("target")
    if target_id is not None and int(target_id) in rows_by_cell:
        target_row = rows_by_cell[int(target_id)]
    else:
        target_bin = spec.get("target_bin")
        if isinstance(target_bin, tuple) and len(target_bin) == 2:
            target_row = rows_by_bin.get((str(target_bin[0]), str(target_bin[1])))
    if target_row is None:
        return None

    source_rows: List[Dict[str, object]] = []
    source_ids = spec.get("sources")
    if isinstance(source_ids, list) and all(int(source) in rows_by_cell for source in source_ids):
        source_rows = [rows_by_cell[int(source)] for source in source_ids]
    else:
        source_bins = spec.get("source_bins")
        if isinstance(source_bins, list):
            for source_bin in source_bins:
                if not isinstance(source_bin, tuple) or len(source_bin) != 2:
                    return None
                row = rows_by_bin.get((str(source_bin[0]), str(source_bin[1])))
                if row is None:
                    return None
                source_rows.append(row)
    if not source_rows:
        return None

    weights = [float(weight) for weight in spec.get("weights", [])]  # type: ignore[arg-type]
    if len(weights) != len(source_rows):
        return None
    wsum = sum(weights)
    if wsum <= 0.0:
        return None
    return (
        int(target_row["cell_id"]),
        [int(row["cell_id"]) for row in source_rows],
        [weight / wsum for weight in weights],
    )


def apply_borrow_policy(
    rows: List[Dict[str, object]],
    profile_density: np.ndarray,
    *,
    enabled: bool,
) -> List[Dict[str, object]]:
    if not enabled:
        for row in rows:
            row.setdefault("psf_borrowed", False)
            row.setdefault("borrowed_from", "")
            row.setdefault("borrow_method", "")
            row.setdefault("borrow_weights", "")
        return []

    rows_by_cell = {int(row["cell_id"]): row for row in rows}
    rows_by_bin = {(str(row["nhit_bin"]), str(row["predE_bin"])): row for row in rows}
    row_index_by_cell = {int(row["cell_id"]): idx for idx, row in enumerate(rows)}
    records: List[Dict[str, object]] = []
    for legacy_target, spec in V3_BORROW_SPECS.items():
        resolved = resolve_borrow_spec(spec, rows_by_cell, rows_by_bin)
        if resolved is None:
            continue
        target, sources, weights = resolved
        target_row = rows_by_cell[target]
        original = {key: target_row.get(key) for key in BORROW_NUMERIC_KEYS if key in target_row}
        original.update(
            {
                "effective_events": target_row.get("effective_events"),
                "core_fit_effective_events": target_row.get("core_fit_effective_events"),
                "theta_missing_crab_probability_mass": target_row.get("theta_missing_crab_probability_mass"),
            }
        )
        for key in BORROW_NUMERIC_KEYS:
            value = weighted_borrow_value(rows_by_cell, sources, weights, key)
            if value is not None:
                target_row[key] = value
        if target_row.get("r_opt_deg") is not None:
            target_row["r_opt_rad"] = math.radians(float(target_row["r_opt_deg"]))
        if target_row.get("containment_r_opt") is not None:
            target_row["containment_minus_expected"] = float(target_row["containment_r_opt"]) - TARGET_CONTAINMENT
        target_row["psf_borrowed"] = True
        target_row["borrowed_from"] = ",".join(str(src) for src in sources)
        target_row["borrow_method"] = str(spec["method"])
        target_row["borrow_weights"] = ",".join(f"{weight:.8g}" for weight in weights)
        target_row["fit_quality"] = "borrowed_neighbor_psf"
        if "observed_data_profile_source" in target_row:
            target_row["observed_data_profile_source"] = f"borrowed_profile_from:{target_row['borrowed_from']}"
        target_row["psf_quality_flag"] = "borrowed"
        warnings = target_row.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
        warnings.append(f"psf_borrowed_from:{target_row['borrowed_from']}")
        target_row["warnings"] = warnings
        target_idx = row_index_by_cell[target]
        borrowed_profile = np.zeros_like(profile_density[target_idx], dtype=np.float64)
        for src, weight in zip(sources, weights):
            borrowed_profile += profile_density[row_index_by_cell[src]].astype(np.float64) * weight
        profile_density[target_idx] = borrowed_profile.astype(profile_density.dtype)
        records.append(
            {
                "target_cell_id": target,
                "legacy_target_cell_id": legacy_target,
                "target_bin": [target_row.get("nhit_bin"), target_row.get("predE_bin")],
                "borrowed_from": sources,
                "borrowed_from_bins": [
                    [rows_by_cell[src].get("nhit_bin"), rows_by_cell[src].get("predE_bin")]
                    for src in sources
                ],
                "weights": {str(src): weight for src, weight in zip(sources, weights)},
                "method": str(spec["method"]),
                "original": original,
                "borrowed": {key: target_row.get(key) for key in BORROW_NUMERIC_KEYS if key in target_row},
            }
        )
    for row in rows:
        row.setdefault("psf_borrowed", False)
        row.setdefault("borrowed_from", "")
        row.setdefault("borrow_method", "")
        row.setdefault("borrow_weights", "")
    return records


def write_summary_csv(path_: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "psf_method",
        "input_files",
        "events",
        "valid_events",
        "effective_events",
        "core_fit_effective_events",
        "sigma_deg",
        "sigma_eff_deg",
        "sigma_x_deg",
        "sigma_y_deg",
        "sigma_x_over_y",
        "mu_x_deg",
        "mu_y_deg",
        "r_opt_deg",
        "r715_deg",
        "mc_quantile_r715_deg",
        "observed_data_r715_deg",
        "observed_data_raw_r715_deg",
        "observed_data_containment_r_opt",
        "observed_data_r_opt_over_rayleigh",
        "observed_data_r_opt_over_mc_quantile",
        "observed_data_positive_total",
        "observed_data_raw_total",
        "observed_data_pedestal_per_deg2",
        "observed_data_fallback",
        "observed_data_fallback_reason",
        "observed_data_profile_source",
        "double_rayleigh_A",
        "double_rayleigh_sigma1_deg",
        "double_rayleigh_sigma2_deg",
        "double_rayleigh_sigma_eq_deg",
        "double_rayleigh_r_opt_deg",
        "double_rayleigh_containment_r_opt",
        "double_rayleigh_model_containment_r_opt",
        "double_rayleigh_fit_quality",
        "double_rayleigh_fallback_reason",
        "double_rayleigh_chi2",
        "double_rayleigh_ndof",
        "double_rayleigh_chi2_ndof",
        "double_rayleigh_positive_profile_bins",
        "double_rayleigh_fit_profile_bins",
        "double_rayleigh_profile_effective_events",
        "target_containment",
        "containment_r_opt",
        "r68_deg",
        "r90_deg",
        "r95_deg",
        "fit_quality",
        "psf_quality_flag",
        "psf_borrowed",
        "borrowed_from",
        "borrow_method",
        "borrow_weights",
        "theta_missing_crab_probability_mass",
        "containment_warning",
        "angle_check_warning",
    ]
    with path_.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary_md(path_: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    with path_.open("w", encoding="utf-8") as handle:
        handle.write("# Stage B v5 PSF Comparison Summary\n\n")
        handle.write(f"- Method: `{metadata['psf_comparison']['method']}`\n")
        handle.write(f"- Target containment: {metadata['psf_comparison']['target_containment']:.8g}\n")
        handle.write(f"- Cells: {metadata['n_cells']}\n")
        handle.write(f"- Output NPZ: `{metadata['outputs']['npz']}`\n")
        handle.write(f"- Borrow policy: `{metadata['psf_comparison']['borrow_policy']['status']}`\n\n")
        handle.write("| cell | Nhit | predE | r_opt deg | ray sigma | sigma_x | sigma_y | MC q | observed q | double q | containment | quality | borrowed |\n")
        handle.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{float(row['r_opt_deg']):.6g} | {float(row['sigma_deg']):.6g} | "
                f"{float(row['sigma_x_deg']):.6g} | {float(row['sigma_y_deg']):.6g} | "
                f"{float(row['mc_quantile_r715_deg']):.6g} | {float(row['observed_data_r715_deg']):.6g} | "
                f"{float(row['double_rayleigh_r_opt_deg']):.6g} | "
                f"{float(row['containment_r_opt']):.6g} | "
                f"{row.get('fit_quality', '')} | {row.get('borrowed_from', '')} |\n"
            )


def save_npz(path_: Path, rows: Sequence[Dict[str, object]], cells: Sequence[object], profile_density: np.ndarray, theta_edges_deg: np.ndarray, crab_prob: np.ndarray, profile_edges_deg: np.ndarray) -> None:
    np.savez_compressed(
        path_,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        sigma_rad=metric_array(rows, "sigma_rad").astype(np.float32),
        sigma_deg=metric_array(rows, "sigma_deg").astype(np.float32),
        sigma_mc_weight_deg=metric_array(rows, "sigma_mc_weight_deg").astype(np.float32),
        sigma_unweighted_deg=metric_array(rows, "sigma_unweighted_deg").astype(np.float32),
        sigma_full_rayleigh_rad=metric_array(rows, "sigma_full_rayleigh_rad").astype(np.float32),
        sigma_full_rayleigh_deg=metric_array(rows, "sigma_full_rayleigh_deg").astype(np.float32),
        sigma_full_mc_weight_deg=metric_array(rows, "sigma_full_mc_weight_deg").astype(np.float32),
        sigma_full_unweighted_deg=metric_array(rows, "sigma_full_unweighted_deg").astype(np.float32),
        r_opt_rad=metric_array(rows, "r_opt_rad").astype(np.float32),
        r_opt_deg=metric_array(rows, "r_opt_deg").astype(np.float32),
        containment_r_opt=metric_array(rows, "containment_r_opt").astype(np.float32),
        r68_deg=metric_array(rows, "r68_deg").astype(np.float32),
        r90_deg=metric_array(rows, "r90_deg").astype(np.float32),
        r95_deg=metric_array(rows, "r95_deg").astype(np.float32),
        core_r68_deg=metric_array(rows, "core_r68_deg").astype(np.float32),
        core_r90_deg=metric_array(rows, "core_r90_deg").astype(np.float32),
        core_r95_deg=metric_array(rows, "core_r95_deg").astype(np.float32),
        effective_events=metric_array(rows, "effective_events").astype(np.float32),
        core_fit_effective_events=metric_array(rows, "core_fit_effective_events").astype(np.float32),
        core_fit_weight_fraction=metric_array(rows, "core_fit_weight_fraction").astype(np.float32),
        tail_weight_fraction_above_core_fit=metric_array(rows, "tail_weight_fraction_above_core_fit").astype(np.float32),
        theta_missing_crab_probability_mass=metric_array(rows, "theta_missing_crab_probability_mass").astype(np.float32),
        events=metric_array(rows, "events", default=0.0).astype(np.int64),
        sumw_baseline=metric_array(rows, "sumw_baseline").astype(np.float64),
        psf_method=string_array(rows, "psf_method", width=32),
        target_containment=metric_array(rows, "target_containment").astype(np.float32),
        r715_deg=metric_array(rows, "r715_deg").astype(np.float32),
        sigma_eff_deg=metric_array(rows, "sigma_eff_deg").astype(np.float32),
        sigma_x_deg=metric_array(rows, "sigma_x_deg").astype(np.float32),
        sigma_y_deg=metric_array(rows, "sigma_y_deg").astype(np.float32),
        sigma_x_over_y=metric_array(rows, "sigma_x_over_y").astype(np.float32),
        mu_x_deg=metric_array(rows, "mu_x_deg").astype(np.float32),
        mu_y_deg=metric_array(rows, "mu_y_deg").astype(np.float32),
        mc_quantile_r715_deg=metric_array(rows, "mc_quantile_r715_deg").astype(np.float32),
        observed_data_r715_deg=metric_array(rows, "observed_data_r715_deg").astype(np.float32),
        observed_data_containment_r_opt=metric_array(rows, "observed_data_containment_r_opt").astype(np.float32),
        observed_data_raw_r715_deg=metric_array(rows, "observed_data_raw_r715_deg").astype(np.float32),
        observed_data_positive_total=metric_array(rows, "observed_data_positive_total").astype(np.float32),
        observed_data_raw_positive_total=metric_array(rows, "observed_data_raw_positive_total").astype(np.float32),
        observed_data_raw_total=metric_array(rows, "observed_data_raw_total").astype(np.float32),
        observed_data_pedestal_per_deg2=metric_array(rows, "observed_data_pedestal_per_deg2").astype(np.float32),
        observed_data_r_opt_over_rayleigh=metric_array(rows, "observed_data_r_opt_over_rayleigh").astype(np.float32),
        observed_data_r_opt_over_mc_quantile=metric_array(rows, "observed_data_r_opt_over_mc_quantile").astype(np.float32),
        observed_data_fallback=np.asarray([bool(row.get("observed_data_fallback", False)) for row in rows], dtype=bool),
        observed_data_fallback_reason=string_array(rows, "observed_data_fallback_reason", width=128),
        observed_data_profile_source=string_array(rows, "observed_data_profile_source", width=96),
        double_rayleigh_A=metric_array(rows, "double_rayleigh_A").astype(np.float32),
        double_rayleigh_sigma1_deg=metric_array(rows, "double_rayleigh_sigma1_deg").astype(np.float32),
        double_rayleigh_sigma2_deg=metric_array(rows, "double_rayleigh_sigma2_deg").astype(np.float32),
        double_rayleigh_sigma_eq_deg=metric_array(rows, "double_rayleigh_sigma_eq_deg").astype(np.float32),
        double_rayleigh_r_opt_deg=metric_array(rows, "double_rayleigh_r_opt_deg").astype(np.float32),
        double_rayleigh_containment_r_opt=metric_array(rows, "double_rayleigh_containment_r_opt").astype(np.float32),
        double_rayleigh_model_containment_r_opt=metric_array(rows, "double_rayleigh_model_containment_r_opt").astype(np.float32),
        double_rayleigh_fit_quality=string_array(rows, "double_rayleigh_fit_quality", width=128),
        double_rayleigh_fallback_reason=string_array(rows, "double_rayleigh_fallback_reason", width=160),
        double_rayleigh_chi2=metric_array(rows, "double_rayleigh_chi2").astype(np.float32),
        double_rayleigh_ndof=metric_array(rows, "double_rayleigh_ndof").astype(np.float32),
        double_rayleigh_chi2_ndof=metric_array(rows, "double_rayleigh_chi2_ndof").astype(np.float32),
        double_rayleigh_positive_profile_bins=metric_array(rows, "double_rayleigh_positive_profile_bins").astype(np.float32),
        double_rayleigh_fit_profile_bins=metric_array(rows, "double_rayleigh_fit_profile_bins").astype(np.float32),
        double_rayleigh_profile_effective_events=metric_array(rows, "double_rayleigh_profile_effective_events").astype(np.float32),
        fit_quality=string_array(rows, "fit_quality", width=96),
        psf_borrowed=np.asarray([bool(row.get("psf_borrowed", False)) for row in rows], dtype=bool),
        borrowed_from=string_array(rows, "borrowed_from", width=32),
        borrow_method=string_array(rows, "borrow_method", width=64),
        theta_edges_deg=theta_edges_deg.astype(np.float32),
        crab_theta_probability=crab_prob.astype(np.float32),
        profile_edges_deg=profile_edges_deg.astype(np.float32),
        profile_density=profile_density.astype(np.float32),
    )


def run_self_test(args: argparse.Namespace) -> None:
    sigma = 0.42
    expected = RAYLEIGH_OPT_RADIUS_FACTOR * sigma
    observed = gaussian_radius_for_containment(
        TARGET_CONTAINMENT,
        mu_x_deg=0.0,
        mu_y_deg=0.0,
        sigma_x_deg=sigma,
        sigma_y_deg=sigma,
        radial_quadrature=int(args.two1d_radial_quadrature),
        angle_samples=int(args.two1d_angle_samples),
    )
    rel = abs(observed / expected - 1.0)
    print(f"isotropic two1d radius={observed:.10g} expected={expected:.10g} rel_diff={rel:.3g}")
    if rel > 5.0e-4:
        raise SystemExit("two1d isotropic self-test failed")


def make_default_run_id(psf_method: str) -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"v5_psf_{psf_method}_slurm_{slurm_job_id}"
    return f"v5_psf_{psf_method}_{time.strftime('%Y%m%d_%H%M%S')}"


def output_name_for_method(template: str, method: str, *, all_mode: bool) -> str:
    if not all_mode:
        return template
    if "all" in template:
        return template.replace("all", method)
    stem = Path(template).stem
    suffix = "".join(Path(template).suffixes)
    return f"{stem}_{method}{suffix}"


def run_id_for_method(base_run_id: str, method: str, *, all_mode: bool) -> str:
    if not all_mode:
        return base_run_id
    if "all" in base_run_id:
        return stage02.sanitize_run_id(base_run_id.replace("all", method))
    return stage02.sanitize_run_id(f"{base_run_id}_{method}")


def build_metadata(
    *,
    args: argparse.Namespace,
    method: str,
    run_id: str,
    run_dir: Path,
    output_root: Path,
    binned_root: Path,
    selection_csv: Path,
    stage_a_metadata_path: Path,
    stage_a_metadata: Dict[str, object],
    theta_edges_deg: np.ndarray,
    crab_prob: np.ndarray,
    loge_min: float,
    loge_max: float,
    loge_range_source: str,
    cells: Sequence[object],
    rows: Sequence[Dict[str, object]],
    borrow_records: Sequence[Dict[str, object]],
    warning_rows: Sequence[Dict[str, object]],
    npz_path: Path,
    metadata_path: Path,
    summary_csv_path: Path,
    summary_md_path: Path,
    plot_outputs: Dict[str, str],
    elapsed_seconds: float,
) -> Dict[str, object]:
    return {
        "description": "Stage B v5 PSF aperture-comparison table for configured Crab SED cells.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "binned_root": str(binned_root),
        "cell_selection_csv": str(selection_csv),
        "stage_a_metadata": str(stage_a_metadata_path),
        "stage_a_snapshot": stage02.compact_stage_a_snapshot(stage_a_metadata),
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "tree_name": args.tree_name,
        "n_cells": len(cells),
        "cells": list(rows),
        "weighting": {
            "baseline": f"{args.weight_branch} * crab_declination_theta_reweight",
            "diagnostics": ["unweighted", args.weight_branch],
            "allow_missing_weight": bool(args.allow_missing_weight),
        },
        "logE_true_filter": {
            "min_inclusive": float(loge_min),
            "max_exclusive": float(loge_max),
            "source": loge_range_source,
        },
        "crab_track": {
            "lhaaso_lat_deg": float(args.lhaaso_lat_deg),
            "source_dec_deg": float(args.source_dec_deg),
            "theta_max_deg": float(args.theta_max_deg),
            "hour_angle_samples": int(args.hour_angle_samples),
            "interpretation": "uniform hour-angle samples conditioned on theta < theta_max_deg",
        },
        "theta_edges_deg": theta_edges_deg.tolist(),
        "crab_theta_probability": crab_prob.tolist(),
        "psf_comparison": {
            "method": method,
            "target_containment": float(TARGET_CONTAINMENT),
            "target_containment_definition": "1 - exp(-0.5 * 1.58^2), matching the v4 Rayleigh aperture contract",
            "methods": {
                "rayleigh_baseline": "Core Rayleigh sigma=sqrt(<r^2>/2), r_opt=1.58*sigma.",
                "two_1d_gaussian": "Weighted x/y tangent-plane Gaussian core fit; r_opt solves circular 2D Gaussian containment at target_containment.",
                "mc_quantile_715": "Crab-theta-reweighted empirical mc_dangle quantile at target_containment.",
                "observed_data": "Pedestal-subtracted observed Crab excess radial-profile quantile at target_containment; fallback to Rayleigh for unreliable or divergent profiles.",
                "double_rayleigh_mixture": "Two-component circular 2D-Gaussian / double-Rayleigh radial mixture fit to the Crab-theta-weighted MC radial profile; r_opt solves the mixture CDF at target_containment.",
            },
            "radial_residual_branch": "mc_dangle",
            "observed_data_profile": {
                "npz": str(path(args.observed_profile_npz)),
                "summary_csv": str(path(args.observed_profile_summary_csv)),
                "pedestal": "median outer-shell excess density per deg^2 subtracted before positive-shell integration",
                "pedestal_min_deg": float(args.observed_pedestal_min_deg),
                "max_r_opt_over_rayleigh": float(args.observed_max_r_opt_over_rayleigh),
                "max_r_opt_over_mc_quantile": float(args.observed_max_r_opt_over_mc_quantile),
                "max_r_opt_deg": float(args.observed_max_r_opt_deg),
                "min_positive_total": float(args.observed_min_positive_total),
                "require_reliable": bool(args.observed_require_reliable),
                "coverage": "Observed profiles currently cover v4/drop4 fit cells; missing or rejected cells use fallback Rayleigh before psfborrow.",
            },
            "signed_offset_projection": "gnomonic tangent-plane offsets around the MC direction, in degrees",
            "core_fit_max_deg": float(args.core_fit_max_deg),
            "two1d_radial_quadrature": int(args.two1d_radial_quadrature),
            "two1d_angle_samples": int(args.two1d_angle_samples),
            "borrow_policy": {
                "status": "enabled_v3_v4_fallback_cells" if not bool(args.no_borrow_v3_fallback_psf) else "disabled",
                "target_cells": sorted(V3_BORROW_SPECS) if not bool(args.no_borrow_v3_fallback_psf) else [],
                "records": borrow_records,
                "note": "Enabled by default to match the v3/v4 psfborrow baseline for cells 39, 52, and 65.",
            },
        },
        "quality_thresholds": {
            "min_events_per_cell": int(args.min_events_per_cell),
            "min_effective_events": float(args.min_effective_events),
            "core_fit_max_deg": float(args.core_fit_max_deg),
            "containment_warning_tolerance": float(args.containment_warning_tolerance),
            "allow_incomplete_theta_support": bool(args.allow_incomplete_theta_support),
            "theta_missing_mass_fail_threshold": float(args.theta_missing_mass_fail_threshold),
            "angle_check_max_events": int(args.angle_check_max_events),
            "angle_check_warn_rad": float(args.angle_check_warn_rad),
            "file_progress_every": int(args.file_progress_every),
        },
        "warning_rows": list(warning_rows),
        "elapsed_seconds": float(elapsed_seconds),
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": {
            "npz": str(npz_path),
            "metadata_json": str(metadata_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            **plot_outputs,
        },
    }


def write_method_output(
    *,
    args: argparse.Namespace,
    method: str,
    base_run_id: str,
    all_mode: bool,
    output_root: Path,
    binned_root: Path,
    selection_csv: Path,
    stage_a_metadata_path: Path,
    stage_a_metadata: Dict[str, object],
    theta_edges_deg: np.ndarray,
    crab_prob: np.ndarray,
    profile_edges_deg: np.ndarray,
    loge_min: float,
    loge_max: float,
    loge_range_source: str,
    cells: Sequence[object],
    base_rows: Sequence[Dict[str, object]],
    base_profile_density: np.ndarray,
    start_time: float,
) -> None:
    run_id = run_id_for_method(base_run_id, method, all_mode=all_mode)
    run_dir = stage02.prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))
    rows = [
        apply_method_to_row(dict(row), method, containment_warning_tolerance=float(args.containment_warning_tolerance))
        for row in base_rows
    ]
    profile_density = np.asarray(base_profile_density, dtype=np.float32).copy()
    if method == "observed_data":
        observed_profiles = load_observed_profiles(path(args.observed_profile_npz), path(args.observed_profile_summary_csv))
        replace_observed_profile_density(
            rows,
            profile_density,
            observed_profiles=observed_profiles,
            profile_edges_deg=profile_edges_deg,
            pedestal_min_deg=float(args.observed_pedestal_min_deg),
        )
    borrow_records = apply_borrow_policy(rows, profile_density, enabled=not bool(args.no_borrow_v3_fallback_psf))
    rows = [
        apply_method_to_row(dict(row), method, containment_warning_tolerance=float(args.containment_warning_tolerance))
        for row in rows
    ]

    npz_path = run_dir / output_name_for_method(args.npz_name, method, all_mode=all_mode)
    metadata_path = run_dir / output_name_for_method(args.metadata_name, method, all_mode=all_mode)
    summary_csv_path = run_dir / output_name_for_method(args.summary_csv_name, method, all_mode=all_mode)
    summary_md_path = run_dir / output_name_for_method(args.summary_md_name, method, all_mode=all_mode)
    save_npz(npz_path, rows, cells, profile_density, theta_edges_deg, crab_prob, profile_edges_deg)

    if not args.no_plots:
        plot_outputs = stage02.write_plots(
            run_dir,
            rows=rows,
            cells=cells,
            profile_density=profile_density,
            profile_edges_deg=profile_edges_deg,
        )
    else:
        plot_outputs = {}

    warning_rows = [
        {
            "cell_id": row["cell_id"],
            "containment_warning": row.get("containment_warning"),
            "angle_check_warning": row.get("angle_check_warning"),
            "fit_quality": row.get("fit_quality"),
            "psf_borrowed": row.get("psf_borrowed"),
            "missing_crab_probability_mass": (
                row.get("theta_reweight", {}).get("missing_crab_probability_mass")
                if isinstance(row.get("theta_reweight"), dict)
                else row.get("theta_missing_crab_probability_mass")
            ),
        }
        for row in rows
        if row.get("containment_warning") or row.get("angle_check_warning") or row.get("fit_quality") != "ok" or row.get("psf_borrowed")
    ]

    metadata = build_metadata(
        args=args,
        method=method,
        run_id=run_id,
        run_dir=run_dir,
        output_root=output_root,
        binned_root=binned_root,
        selection_csv=selection_csv,
        stage_a_metadata_path=stage_a_metadata_path,
        stage_a_metadata=stage_a_metadata,
        theta_edges_deg=theta_edges_deg,
        crab_prob=crab_prob,
        loge_min=loge_min,
        loge_max=loge_max,
        loge_range_source=loge_range_source,
        cells=cells,
        rows=rows,
        borrow_records=borrow_records,
        warning_rows=warning_rows,
        npz_path=npz_path,
        metadata_path=metadata_path,
        summary_csv_path=summary_csv_path,
        summary_md_path=summary_md_path,
        plot_outputs=plot_outputs,
        elapsed_seconds=time.perf_counter() - start_time,
    )

    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, metadata, rows)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, indent=2)

    if not args.no_promote_current:
        stage02.promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"
        metadata["promotion"]["current_dir"] = str(output_root / "current")
        metadata["promotion"]["latest"] = str(output_root / "latest")
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(json_ready(metadata), handle, indent=2)
    else:
        metadata["promotion"]["status"] = "skipped"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(json_ready(metadata), handle, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {metadata_path}")
    if warning_rows:
        print(f"Warnings recorded for {len(warning_rows)} cells in {method}; inspect metadata warning_rows.")


def load_npz_arrays(path_: Path) -> Dict[str, np.ndarray]:
    with np.load(path_, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def rows_from_stage_b_source(
    arrays: Dict[str, np.ndarray],
    source_metadata: Dict[str, object],
) -> Tuple[List[object], List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_ids = np.asarray(arrays["cell_id"], dtype=np.int64)
    nhit_bins = np.asarray(arrays["nhit_bin"], dtype=str)
    pred_bins = np.asarray(arrays["predE_bin"], dtype=str)
    profile_density = np.asarray(arrays["profile_density"], dtype=np.float32).copy()
    profile_edges_deg = np.asarray(arrays["profile_edges_deg"], dtype=np.float64)
    theta_edges_deg = np.asarray(arrays["theta_edges_deg"], dtype=np.float64)
    crab_prob = np.asarray(arrays["crab_theta_probability"], dtype=np.float64)

    metadata_cells = source_metadata.get("cells") if isinstance(source_metadata.get("cells"), list) else []
    meta_by_cell = {
        int(row["cell_id"]): dict(row)
        for row in metadata_cells
        if isinstance(row, dict) and finite_float(row.get("cell_id")) is not None
    }
    cells: List[object] = []
    rows: List[Dict[str, object]] = []
    for idx, cell_id_raw in enumerate(cell_ids):
        cell_id = int(cell_id_raw)
        nhit = str(nhit_bins[idx])
        pred = str(pred_bins[idx])
        cells.append(SimpleNamespace(index=idx, cell_id=cell_id, nhit_bin=nhit, predE_bin=pred))
        row = meta_by_cell.get(cell_id, {}).copy()
        row.update(
            {
                "cell_index": idx,
                "cell_id": cell_id,
                "nhit_bin": nhit,
                "predE_bin": pred,
                "psf_method": "rayleigh_baseline",
                "target_containment": float(TARGET_CONTAINMENT),
            }
        )
        for key in [
            "sigma_rad",
            "sigma_deg",
            "sigma_mc_weight_deg",
            "sigma_unweighted_deg",
            "sigma_full_rayleigh_rad",
            "sigma_full_rayleigh_deg",
            "sigma_full_mc_weight_deg",
            "sigma_full_unweighted_deg",
            "r_opt_rad",
            "r_opt_deg",
            "containment_r_opt",
            "r68_deg",
            "r90_deg",
            "r95_deg",
            "core_r68_deg",
            "core_r90_deg",
            "core_r95_deg",
            "effective_events",
            "core_fit_effective_events",
            "core_fit_weight_fraction",
            "tail_weight_fraction_above_core_fit",
            "theta_missing_crab_probability_mass",
            "events",
            "sumw_baseline",
            "r715_deg",
            "sigma_eff_deg",
            "sigma_x_deg",
            "sigma_y_deg",
            "sigma_x_over_y",
            "mu_x_deg",
            "mu_y_deg",
            "mc_quantile_r715_deg",
        ]:
            if key in arrays and idx < np.asarray(arrays[key]).shape[0]:
                row[key] = float(np.asarray(arrays[key], dtype=np.float64)[idx])
        for key in ["psf_method", "fit_quality", "borrowed_from", "borrow_method"]:
            if key in arrays and idx < np.asarray(arrays[key]).shape[0]:
                row[key] = str(np.asarray(arrays[key], dtype=str)[idx])
        if "psf_borrowed" in arrays and idx < np.asarray(arrays["psf_borrowed"]).shape[0]:
            row["psf_borrowed"] = bool(np.asarray(arrays["psf_borrowed"], dtype=bool)[idx])
        row["rayleigh_baseline_r715_deg"] = finite_float(row.get("rayleigh_baseline_r715_deg")) or finite_float(row.get("r_opt_deg")) or float("nan")
        row["rayleigh_baseline_containment_r_opt"] = finite_float(row.get("rayleigh_baseline_containment_r_opt")) or finite_float(row.get("containment_r_opt")) or TARGET_CONTAINMENT
        row["two_1d_gaussian_r715_deg"] = finite_float(row.get("two_1d_gaussian_r715_deg")) or finite_float(row.get("r_opt_deg")) or float("nan")
        row["two_1d_gaussian_containment_r_opt"] = finite_float(row.get("two_1d_gaussian_containment_r_opt")) or finite_float(row.get("containment_r_opt")) or TARGET_CONTAINMENT
        row["mc_quantile_containment_r_opt"] = finite_float(row.get("mc_quantile_containment_r_opt")) or TARGET_CONTAINMENT
        row["observed_data_r715_deg"] = finite_float(row.get("observed_data_r715_deg")) or finite_float(row.get("r_opt_deg")) or float("nan")
        row["observed_data_containment_r_opt"] = finite_float(row.get("observed_data_containment_r_opt")) or finite_float(row.get("containment_r_opt")) or TARGET_CONTAINMENT
        row["observed_data_fit_quality"] = row.get("observed_data_fit_quality", "not_recomputed_from_source_stage_b")
        row["observed_data_fallback"] = bool(row.get("observed_data_fallback", False))
        row["observed_data_fallback_reason"] = row.get("observed_data_fallback_reason", "")
        row["observed_data_profile_source"] = row.get("observed_data_profile_source", "source_stage_b_profile")
        row["rayleigh_fit_quality"] = row.get("rayleigh_fit_quality", row.get("fit_quality", "ok"))
        row["two_1d_gaussian_fit_quality"] = row.get("two_1d_gaussian_fit_quality", "not_recomputed_from_source_stage_b")
        row["mc_quantile_fit_quality"] = row.get("mc_quantile_fit_quality", "ok")
        row.setdefault("input_files", "")
        row.setdefault("valid_events", "")
        row.setdefault("angle_check_warning", False)
        row.setdefault("containment_warning", False)
        row.setdefault("warnings", [])
        rows.append(row)

    return cells, rows, profile_density, profile_edges_deg, theta_edges_deg, crab_prob


def write_observed_data_from_stage_b_source(
    *,
    args: argparse.Namespace,
    start_time: float,
) -> None:
    source_npz = path(args.observed_stage_b_source_npz)
    source_meta_path = path(args.observed_stage_b_source_metadata) if args.observed_stage_b_source_metadata else source_npz.with_name(source_npz.stem + "_metadata.json")
    if not source_npz.exists():
        raise FileNotFoundError(f"Missing observed-data source Stage B NPZ: {source_npz}")
    if not source_meta_path.exists():
        raise FileNotFoundError(f"Missing observed-data source Stage B metadata: {source_meta_path}")

    arrays = load_npz_arrays(source_npz)
    source_metadata = json.loads(source_meta_path.read_text(encoding="utf-8"))
    cells, rows, profile_density, profile_edges_deg, theta_edges_deg, crab_prob = rows_from_stage_b_source(arrays, source_metadata)
    observed_profiles = load_observed_profiles(path(args.observed_profile_npz), path(args.observed_profile_summary_csv))

    for row in rows:
        rayleigh_r = float(finite_float(row.get("rayleigh_baseline_r715_deg")) or finite_float(row.get("r_opt_deg")) or 0.0)
        mc_r = float(finite_float(row.get("mc_quantile_r715_deg")) or float("nan"))
        aperture = observed_profile_aperture(
            cell_id=int(row["cell_id"]),
            observed_profiles=observed_profiles,
            rayleigh_r_opt_deg=rayleigh_r,
            mc_quantile_r715_deg=mc_r,
            target_containment=float(TARGET_CONTAINMENT),
            pedestal_min_deg=float(args.observed_pedestal_min_deg),
            max_r_opt_over_rayleigh=float(args.observed_max_r_opt_over_rayleigh),
            max_r_opt_over_mc_quantile=float(args.observed_max_r_opt_over_mc_quantile),
            max_r_opt_deg=float(args.observed_max_r_opt_deg),
            min_positive_total=float(args.observed_min_positive_total),
            require_reliable=bool(args.observed_require_reliable),
        )
        observed_r = finite_float(aperture.get("r715_deg"))
        if aperture.get("status") == "ok" and observed_r is not None:
            row["observed_data_r715_deg"] = float(observed_r)
            row["observed_data_containment_r_opt"] = float(aperture.get("containment") or TARGET_CONTAINMENT)
            row["observed_data_fit_quality"] = "ok"
            row["observed_data_fallback"] = False
            row["observed_data_fallback_reason"] = ""
        else:
            row["observed_data_r715_deg"] = rayleigh_r
            row["observed_data_containment_r_opt"] = float(TARGET_CONTAINMENT)
            row["observed_data_fit_quality"] = f"fallback:{aperture.get('reason', 'observed_profile_invalid')}"
            row["observed_data_fallback"] = True
            row["observed_data_fallback_reason"] = str(aperture.get("reason", ""))
        row["observed_data_raw_r715_deg"] = finite_float(aperture.get("raw_r715_deg"))
        row["observed_data_positive_total"] = finite_float(aperture.get("positive_total"))
        row["observed_data_raw_positive_total"] = finite_float(aperture.get("raw_positive_total"))
        row["observed_data_raw_total"] = finite_float(aperture.get("raw_total"))
        row["observed_data_pedestal_per_deg2"] = finite_float(aperture.get("pedestal_per_deg2"))
        row["observed_data_r_opt_over_rayleigh"] = finite_float(aperture.get("r_opt_over_rayleigh"))
        row["observed_data_r_opt_over_mc_quantile"] = finite_float(aperture.get("r_opt_over_mc_quantile"))

    method = "observed_data"
    output_root = path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = stage02.sanitize_run_id(args.run_id or "v5_psf_observed_data_drop4")
    run_dir = stage02.prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))
    rows = [apply_method_to_row(row, method, containment_warning_tolerance=float(args.containment_warning_tolerance)) for row in rows]
    replace_observed_profile_density(
        rows,
        profile_density,
        observed_profiles=observed_profiles,
        profile_edges_deg=profile_edges_deg,
        pedestal_min_deg=float(args.observed_pedestal_min_deg),
    )
    borrow_records = apply_borrow_policy(rows, profile_density, enabled=not bool(args.no_borrow_v3_fallback_psf))
    rows = [apply_method_to_row(row, method, containment_warning_tolerance=float(args.containment_warning_tolerance)) for row in rows]

    npz_path = run_dir / args.npz_name
    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    save_npz(npz_path, rows, cells, profile_density, theta_edges_deg, crab_prob, profile_edges_deg)
    plot_outputs = (
        stage02.write_plots(run_dir, rows=rows, cells=cells, profile_density=profile_density, profile_edges_deg=profile_edges_deg)
        if not args.no_plots
        else {}
    )
    warning_rows = [
        {
            "cell_id": row["cell_id"],
            "containment_warning": row.get("containment_warning"),
            "angle_check_warning": row.get("angle_check_warning"),
            "fit_quality": row.get("fit_quality"),
            "observed_data_fallback": row.get("observed_data_fallback"),
            "observed_data_fallback_reason": row.get("observed_data_fallback_reason"),
            "psf_borrowed": row.get("psf_borrowed"),
        }
        for row in rows
        if row.get("containment_warning") or row.get("angle_check_warning") or row.get("fit_quality") != "ok" or row.get("psf_borrowed") or row.get("observed_data_fallback")
    ]
    source_snapshot = source_metadata.get("stage_a_snapshot") if isinstance(source_metadata, dict) else {}
    metadata = build_metadata(
        args=args,
        method=method,
        run_id=run_id,
        run_dir=run_dir,
        output_root=output_root,
        binned_root=path(args.binned_root),
        selection_csv=path(args.cell_selection_csv),
        stage_a_metadata_path=path(args.stage_a_metadata),
        stage_a_metadata={"response_snapshot_source": str(source_meta_path), **(source_snapshot if isinstance(source_snapshot, dict) else {})},
        theta_edges_deg=theta_edges_deg,
        crab_prob=crab_prob,
        loge_min=float(source_metadata.get("logE_true_filter", {}).get("min_inclusive", float("nan"))) if isinstance(source_metadata.get("logE_true_filter"), dict) else float("nan"),
        loge_max=float(source_metadata.get("logE_true_filter", {}).get("max_exclusive", float("nan"))) if isinstance(source_metadata.get("logE_true_filter"), dict) else float("nan"),
        loge_range_source=f"derived_from:{source_meta_path}",
        cells=cells,
        rows=rows,
        borrow_records=borrow_records,
        warning_rows=warning_rows,
        npz_path=npz_path,
        metadata_path=metadata_path,
        summary_csv_path=summary_csv_path,
        summary_md_path=summary_md_path,
        plot_outputs=plot_outputs,
        elapsed_seconds=time.perf_counter() - start_time,
    )
    metadata["psf_comparison"]["observed_data_profile"]["source_stage_b_npz"] = str(source_npz)  # type: ignore[index]
    metadata["psf_comparison"]["observed_data_profile"]["source_stage_b_metadata"] = str(source_meta_path)  # type: ignore[index]
    metadata["psf_comparison"]["observed_data_profile"]["derivation_mode"] = "derived_from_existing_stage_b_base_fields_without_rerunning_mc_event_loop"  # type: ignore[index]

    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, metadata, rows)
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    if not args.no_promote_current:
        stage02.promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"
        metadata["promotion"]["current_dir"] = str(output_root / "current")
        metadata["promotion"]["latest"] = str(output_root / "latest")
    else:
        metadata["promotion"]["status"] = "skipped"
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {metadata_path}")
    if warning_rows:
        print(f"Warnings recorded for {len(warning_rows)} cells in {method}; inspect metadata warning_rows.")


def write_double_rayleigh_from_stage_b_source(
    *,
    args: argparse.Namespace,
    start_time: float,
) -> None:
    source_npz = path(args.double_rayleigh_stage_b_source_npz)
    source_meta_path = (
        path(args.double_rayleigh_stage_b_source_metadata)
        if args.double_rayleigh_stage_b_source_metadata
        else source_npz.with_name(source_npz.stem + "_metadata.json")
    )
    if not source_npz.exists():
        raise FileNotFoundError(f"Missing double-Rayleigh source Stage B NPZ: {source_npz}")
    if not source_meta_path.exists():
        raise FileNotFoundError(f"Missing double-Rayleigh source Stage B metadata: {source_meta_path}")

    arrays = load_npz_arrays(source_npz)
    source_metadata = json.loads(source_meta_path.read_text(encoding="utf-8"))
    cells, rows, profile_density, profile_edges_deg, theta_edges_deg, crab_prob = rows_from_stage_b_source(arrays, source_metadata)

    for idx, row in enumerate(rows):
        sigma_deg = finite_float(row.get("sigma_deg"))
        rayleigh_r = finite_float(row.get("rayleigh_baseline_r715_deg")) or finite_float(row.get("r_opt_deg"))
        if sigma_deg is None or sigma_deg <= 0.0:
            sigma_deg = (float(rayleigh_r) / RAYLEIGH_OPT_RADIUS_FACTOR) if rayleigh_r is not None and rayleigh_r > 0.0 else 1.0
        if rayleigh_r is None or rayleigh_r <= 0.0:
            rayleigh_r = RAYLEIGH_OPT_RADIUS_FACTOR * float(sigma_deg)

        fit = fit_double_rayleigh_mixture_from_profile_density(
            profile_density[idx],
            profile_edges_deg,
            rayleigh_sigma_deg=float(sigma_deg),
        )
        fit_ok = fit.get("status") == "ok"
        r_opt_deg = float(fit["r_opt_deg"]) if fit_ok and finite_float(fit.get("r_opt_deg")) is not None else float(rayleigh_r)
        containment = profile_density_containment(profile_density[idx], profile_edges_deg, r_opt_deg)
        if not np.isfinite(containment) or containment <= 0.0:
            containment = float(TARGET_CONTAINMENT)

        row["double_rayleigh_A"] = finite_float(fit.get("A"))
        row["double_rayleigh_sigma1_deg"] = finite_float(fit.get("sigma1_deg")) if fit_ok else float(sigma_deg)
        row["double_rayleigh_sigma2_deg"] = finite_float(fit.get("sigma2_deg"))
        row["double_rayleigh_sigma_eq_deg"] = finite_float(fit.get("sigma_eq_deg")) if fit_ok else float(r_opt_deg / RAYLEIGH_OPT_RADIUS_FACTOR)
        row["double_rayleigh_r_opt_deg"] = float(r_opt_deg)
        row["double_rayleigh_containment_r_opt"] = float(containment)
        row["double_rayleigh_model_containment_r_opt"] = finite_float(fit.get("model_containment_r_opt")) or float(TARGET_CONTAINMENT)
        row["double_rayleigh_fit_quality"] = "ok" if fit_ok else f"fallback:{fit.get('reason', 'double_rayleigh_fit_failed')}"
        row["double_rayleigh_fallback_reason"] = "" if fit_ok else str(fit.get("reason", "double_rayleigh_fit_failed"))
        row["double_rayleigh_chi2"] = finite_float(fit.get("chi2"))
        row["double_rayleigh_ndof"] = finite_float(fit.get("ndof"))
        row["double_rayleigh_chi2_ndof"] = finite_float(fit.get("chi2_ndof"))
        row["double_rayleigh_positive_profile_bins"] = finite_float(fit.get("positive_profile_bins"))
        row["double_rayleigh_fit_profile_bins"] = finite_float(fit.get("fit_profile_bins"))
        row["double_rayleigh_profile_effective_events"] = finite_float(fit.get("profile_effective_events"))

    method = "double_rayleigh_mixture"
    output_root = path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = stage02.sanitize_run_id(args.run_id or "v5_psf_double_rayleigh_mixture_drop4")
    run_dir = stage02.prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))
    rows = [apply_method_to_row(row, method, containment_warning_tolerance=float(args.containment_warning_tolerance)) for row in rows]
    borrow_records = apply_borrow_policy(rows, profile_density, enabled=not bool(args.no_borrow_v3_fallback_psf))
    rows = [apply_method_to_row(row, method, containment_warning_tolerance=float(args.containment_warning_tolerance)) for row in rows]

    npz_path = run_dir / args.npz_name
    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    save_npz(npz_path, rows, cells, profile_density, theta_edges_deg, crab_prob, profile_edges_deg)
    plot_outputs = (
        stage02.write_plots(run_dir, rows=rows, cells=cells, profile_density=profile_density, profile_edges_deg=profile_edges_deg)
        if not args.no_plots
        else {}
    )
    warning_rows = [
        {
            "cell_id": row["cell_id"],
            "containment_warning": row.get("containment_warning"),
            "angle_check_warning": row.get("angle_check_warning"),
            "fit_quality": row.get("fit_quality"),
            "double_rayleigh_fallback_reason": row.get("double_rayleigh_fallback_reason"),
            "psf_borrowed": row.get("psf_borrowed"),
        }
        for row in rows
        if row.get("containment_warning")
        or row.get("angle_check_warning")
        or row.get("fit_quality") != "ok"
        or row.get("double_rayleigh_fallback_reason")
        or row.get("psf_borrowed")
    ]
    source_snapshot = source_metadata.get("stage_a_snapshot") if isinstance(source_metadata, dict) else {}
    metadata = build_metadata(
        args=args,
        method=method,
        run_id=run_id,
        run_dir=run_dir,
        output_root=output_root,
        binned_root=path(args.binned_root),
        selection_csv=path(args.cell_selection_csv),
        stage_a_metadata_path=path(args.stage_a_metadata),
        stage_a_metadata={"response_snapshot_source": str(source_meta_path), **(source_snapshot if isinstance(source_snapshot, dict) else {})},
        theta_edges_deg=theta_edges_deg,
        crab_prob=crab_prob,
        loge_min=float(source_metadata.get("logE_true_filter", {}).get("min_inclusive", float("nan"))) if isinstance(source_metadata.get("logE_true_filter"), dict) else float("nan"),
        loge_max=float(source_metadata.get("logE_true_filter", {}).get("max_exclusive", float("nan"))) if isinstance(source_metadata.get("logE_true_filter"), dict) else float("nan"),
        loge_range_source=f"derived_from:{source_meta_path}",
        cells=cells,
        rows=rows,
        borrow_records=borrow_records,
        warning_rows=warning_rows,
        npz_path=npz_path,
        metadata_path=metadata_path,
        summary_csv_path=summary_csv_path,
        summary_md_path=summary_md_path,
        plot_outputs=plot_outputs,
        elapsed_seconds=time.perf_counter() - start_time,
    )
    metadata["psf_comparison"]["double_rayleigh_mixture_profile"] = {  # type: ignore[index]
        "source_stage_b_npz": str(source_npz),
        "source_stage_b_metadata": str(source_meta_path),
        "derivation_mode": "fit_double_rayleigh_mixture_to_existing_stage_b_profile_density_without_rerunning_mc_event_loop",
        "aperture_definition": "r_opt solves F(r)=target_containment for the fitted two-component Rayleigh CDF; sigma_eq is diagnostic only",
    }

    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, metadata, rows)
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    if not args.no_promote_current:
        stage02.promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"
        metadata["promotion"]["current_dir"] = str(output_root / "current")
        metadata["promotion"]["latest"] = str(output_root / "latest")
    else:
        metadata["promotion"]["status"] = "skipped"
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {metadata_path}")
    if warning_rows:
        print(f"Warnings recorded for {len(warning_rows)} cells in {method}; inspect metadata warning_rows.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args)
        return

    start = time.perf_counter()
    if args.psf_method == "observed_data" and args.observed_stage_b_source_npz:
        write_observed_data_from_stage_b_source(args=args, start_time=start)
        return
    if args.psf_method == "double_rayleigh_mixture" and args.double_rayleigh_stage_b_source_npz:
        write_double_rayleigh_from_stage_b_source(args=args, start_time=start)
        return

    binned_root = path(args.binned_root)
    selection_csv = path(args.cell_selection_csv)
    stage_a_metadata_path = path(args.stage_a_metadata)
    output_root = path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = stage02.sanitize_run_id(args.run_id or make_default_run_id(args.psf_method))

    cells = stage02.load_cells(selection_csv)
    stage_a_metadata = stage02.load_stage_a_metadata(stage_a_metadata_path, bool(args.allow_missing_stage_a_metadata))
    stage02.validate_stage_a_metadata_for_production(stage_a_metadata)
    stage02.validate_stage_a_cells(cells, stage_a_metadata)
    loge_min, loge_max, loge_range_source = stage02.resolve_loge_range(args, stage_a_metadata)
    theta_edges_deg = stage02.make_edges(float(args.theta_min_deg), float(args.theta_max_deg), float(args.theta_step_deg))
    profile_edges_deg = stage02.make_edges(0.0, float(args.profile_max_deg), float(args.profile_bin_width_deg))
    crab_prob = stage02.crab_theta_probability(
        theta_edges_deg,
        latitude_deg=float(args.lhaaso_lat_deg),
        declination_deg=float(args.source_dec_deg),
        theta_max_deg=float(args.theta_max_deg),
        hour_angle_samples=int(args.hour_angle_samples),
    )
    observed_profiles = load_observed_profiles(path(args.observed_profile_npz), path(args.observed_profile_summary_csv))

    cell_kwargs: Dict[str, object] = {
        "psf_method": args.psf_method,
        "binned_root": binned_root,
        "tree_name": args.tree_name,
        "weight_branch": args.weight_branch,
        "allow_missing_weight": bool(args.allow_missing_weight),
        "max_files_per_cell": args.max_files_per_cell,
        "allow_missing_cell_dirs": bool(args.allow_missing_cell_dirs),
        "theta_edges_deg": theta_edges_deg,
        "crab_prob": crab_prob,
        "loge_min": loge_min,
        "loge_max": loge_max,
        "allow_incomplete_theta_support": bool(args.allow_incomplete_theta_support),
        "min_events_per_cell": int(args.min_events_per_cell),
        "min_effective_events": float(args.min_effective_events),
        "allow_low_stat_psf_fallback": bool(args.allow_low_stat_psf_fallback),
        "core_fit_max_deg": float(args.core_fit_max_deg),
        "theta_missing_mass_fail_threshold": float(args.theta_missing_mass_fail_threshold),
        "containment_warning_tolerance": float(args.containment_warning_tolerance),
        "angle_check_max_events": int(args.angle_check_max_events),
        "angle_check_warn_rad": float(args.angle_check_warn_rad),
        "file_progress_every": int(args.file_progress_every),
        "profile_edges_deg": profile_edges_deg,
        "two1d_radial_quadrature": int(args.two1d_radial_quadrature),
        "two1d_angle_samples": int(args.two1d_angle_samples),
        "observed_profiles": observed_profiles,
        "observed_pedestal_min_deg": float(args.observed_pedestal_min_deg),
        "observed_max_r_opt_over_rayleigh": float(args.observed_max_r_opt_over_rayleigh),
        "observed_max_r_opt_over_mc_quantile": float(args.observed_max_r_opt_over_mc_quantile),
        "observed_max_r_opt_deg": float(args.observed_max_r_opt_deg),
        "observed_min_positive_total": float(args.observed_min_positive_total),
        "observed_require_reliable": bool(args.observed_require_reliable),
    }
    rows_by_index: Dict[int, Dict[str, object]] = {}
    profiles_by_index: Dict[int, np.ndarray] = {}
    workers = max(1, int(args.workers))
    if workers == 1:
        for done_count, cell in enumerate(cells, start=1):
            row, profile_density = _process_cell_v5(cell, **cell_kwargs)
            rows_by_index[int(cell.index)] = row
            profiles_by_index[int(cell.index)] = profile_density
            print(
                f"[{done_count}/{len(cells)}] {args.psf_method} cell={cell.cell_id} "
                f"events={row['events']} Neff={row['effective_events']:.1f} "
                f"r_opt={row['r_opt_deg']:.4g} deg",
                flush=True,
            )
    else:
        print(f"Processing {len(cells)} cells with {workers} workers.", flush=True)
        tasks = [(cell, cell_kwargs) for cell in cells]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_by_cell = {executor.submit(process_cell_v5, cell, kwargs): cell for cell, kwargs in tasks}
            for done_count, future in enumerate(as_completed(future_by_cell), start=1):
                cell = future_by_cell[future]
                cell_index, row, profile_density = future.result()
                rows_by_index[int(cell_index)] = row
                profiles_by_index[int(cell_index)] = profile_density
                print(
                    f"[{done_count}/{len(cells)}] {args.psf_method} cell={cell.cell_id} "
                    f"events={row['events']} Neff={row['effective_events']:.1f} "
                    f"r_opt={row['r_opt_deg']:.4g} deg",
                    flush=True,
                )

    rows = [rows_by_index[int(cell.index)] for cell in cells]
    profile_density = np.vstack([profiles_by_index[int(cell.index)] for cell in cells]).astype(np.float32)

    methods_to_write = list(PSF_METHODS) if args.psf_method == "all" else [args.psf_method]
    for method in methods_to_write:
        write_method_output(
            args=args,
            method=method,
            base_run_id=run_id,
            all_mode=(args.psf_method == "all"),
            output_root=output_root,
            binned_root=binned_root,
            selection_csv=selection_csv,
            stage_a_metadata_path=stage_a_metadata_path,
            stage_a_metadata=stage_a_metadata,
            theta_edges_deg=theta_edges_deg,
            crab_prob=crab_prob,
            profile_edges_deg=profile_edges_deg,
            loge_min=loge_min,
            loge_max=loge_max,
            loge_range_source=loge_range_source,
            cells=cells,
            base_rows=rows,
            base_profile_density=profile_density,
            start_time=start,
        )


if __name__ == "__main__":
    main()
