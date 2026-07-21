#!/usr/bin/env python3
"""Fit Pass5-like WCDA PSFs from the official ETO simulation sample.

The raw sample is streamed into compact Nhit x theta x dangle histograms.  The
histograms are then fitted with the same mathematical family used by XSQ: a
mixture of two circular 2D Gaussians, equivalently a double-Rayleigh radial
distribution.  No ROOT files are modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


NHIT_EDGES = np.asarray([30, 60, 100, 200, 300, 500, 800, 2000], dtype=np.int64)
NHIT_LABELS = tuple(f"[{low},{high})" for low, high in zip(NHIT_EDGES[:-1], NHIT_EDGES[1:]))
PASS5_PINC_MAX = np.asarray([1.12, 1.02, 0.90, 0.88, 0.88, 0.84, 0.84], dtype=np.float64)
VARIANTS = ("unweighted", "mc_weight", "crab_theta")

# Values read from data/pass5/z50/WCDA/irfs.root. XSQ's fraction multiplies
# sigma1 and does not require sigma1 < sigma2.
XSQ_RAW = (
    (0.657720, 0.437695, 1.020580, 0.89, 3.49209),
    (0.515064, 0.311013, 0.598829, 0.66, 2.11346),
    (0.412479, 0.450880, 0.259153, 0.50, 1.57464),
    (0.803910, 0.231834, 0.450269, 0.40, 1.47496),
    (0.779136, 0.194469, 0.336419, 0.34, 1.11655),
    (0.115909, 0.318292, 0.161700, 0.27, 0.997343),
    (0.0443794, 0.0570923, 0.131437, 0.20, 0.504671),
)

FILE_INDEX_RE = re.compile(r"Egr(\d+)_")
SCALAR_BRANCHES = (
    "pincness",
    "fitstat",
    "theta",
    "rmds",
    "dcedge",
    "mc_theta",
    "mc_weight",
    "mc_dangle",
)


@dataclass(frozen=True)
class FitResult:
    success: bool
    a_core: float
    sigma_core_deg: float
    sigma_tail_deg: float
    objective: float
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Pass5 seven-bin double-Rayleigh PSFs from /mnt/mydisk/WCDA_simulation."
    )
    parser.add_argument("--input-root", default="/mnt/mydisk/WCDA_simulation")
    parser.add_argument("--input-glob", default="*_eventout.root")
    parser.add_argument("--tree-name", default="t_eventout")
    parser.add_argument("--output-dir", default="apply/output/pass5_psf_mc_comparison")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--file-index-min", type=int, default=None)
    parser.add_argument("--file-index-max", type=int, default=None)
    parser.add_argument("--flush-files", type=int, default=100)
    parser.add_argument("--strict-files", action="store_true")
    parser.add_argument(
        "--nhit-definition",
        choices=("pass5_nq05t30", "nv", "nfit", "nfitb", "nrange"),
        default="pass5_nq05t30",
        help="Pass5 uses nq05t30 reconstructed from the selected charge branch and vt; scalar branches are diagnostic alternatives.",
    )
    parser.add_argument("--hit-charge-threshold", type=float, default=0.5)
    parser.add_argument(
        "--hit-charge-branch",
        choices=("vq", "vqsamp"),
        default="vq",
        help="nq05 is defined from normalized charge vq; vqsamp is retained as a diagnostic alternative.",
    )
    parser.add_argument("--hit-time-window-ns", type=float, default=30.0)
    parser.add_argument(
        "--pincness-calibration",
        type=float,
        default=1.07,
        help="Apply pincness/calibration < Pass5 threshold. Crab-dec response uses 1.07.",
    )
    parser.add_argument("--rmds-max", type=float, default=20.0)
    parser.add_argument("--dcedge-min", type=float, default=None)
    parser.add_argument("--fitstat-equals", type=int, default=0)
    parser.add_argument("--theta-max-deg", type=float, default=50.0)
    parser.add_argument("--dangle-max-deg", type=float, default=10.0)
    parser.add_argument("--dangle-bin-width-deg", type=float, default=0.01)
    parser.add_argument("--fit-max-deg", type=float, default=6.0)
    parser.add_argument("--theta-bin-width-deg", type=float, default=1.0)
    parser.add_argument("--latitude-deg", type=float, default=29.45)
    parser.add_argument("--source-dec-deg", type=float, default=22.01)
    parser.add_argument("--hour-angle-samples", type=int, default=200000)
    parser.add_argument("--primary-variant", choices=VARIANTS, default="crab_theta")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def ordered_xsq(bin_index: int) -> Tuple[float, float, float, float, float]:
    frac, sigma1, sigma2, r68, r999 = XSQ_RAW[bin_index]
    if sigma1 <= sigma2:
        return frac, sigma1, sigma2, r68, r999
    return 1.0 - frac, sigma2, sigma1, r68, r999


def assign_pass5_bins(
    nhit: np.ndarray,
    pincness: np.ndarray,
    fitstat: np.ndarray,
    reco_theta_rad: np.ndarray,
    dangle_rad: np.ndarray,
    rmds: np.ndarray,
    dcedge: np.ndarray,
    *,
    fitstat_equals: int,
    theta_max_deg: float,
    pincness_calibration: float,
    rmds_max: Optional[float],
    dcedge_min: Optional[float],
) -> np.ndarray:
    """Return Pass5 bin indices, with -1 for rejected events."""
    nhit = np.asarray(nhit)
    pincness = np.asarray(pincness, dtype=np.float64)
    fitstat = np.asarray(fitstat)
    reco_theta_rad = np.asarray(reco_theta_rad, dtype=np.float64)
    dangle_rad = np.asarray(dangle_rad, dtype=np.float64)
    rmds = np.asarray(rmds, dtype=np.float64)
    dcedge = np.asarray(dcedge, dtype=np.float64)
    idx = np.searchsorted(NHIT_EDGES, nhit, side="right") - 1
    valid_bin = (idx >= 0) & (idx < len(NHIT_LABELS)) & (nhit < NHIT_EDGES[-1])
    out = np.full(nhit.shape, -1, dtype=np.int16)
    base = (
        valid_bin
        & np.isfinite(pincness)
        & np.isfinite(reco_theta_rad)
        & (reco_theta_rad >= 0.0)
        & (reco_theta_rad < math.radians(float(theta_max_deg)))
        & np.isfinite(dangle_rad)
        & (dangle_rad >= 0.0)
        & (fitstat == int(fitstat_equals))
    )
    if rmds_max is not None:
        base &= np.isfinite(rmds) & (rmds <= float(rmds_max))
    if dcedge_min is not None:
        base &= np.isfinite(dcedge) & (dcedge > float(dcedge_min))
    if np.any(base):
        selected = np.flatnonzero(base)
        selected_bins = idx[selected]
        keep = pincness[selected] / float(pincness_calibration) < PASS5_PINC_MAX[selected_bins]
        out[selected[keep]] = selected_bins[keep].astype(np.int16)
    return out


def crab_theta_probability(
    theta_edges_deg: np.ndarray,
    *,
    latitude_deg: float,
    declination_deg: float,
    theta_max_deg: float,
    hour_angle_samples: int,
) -> np.ndarray:
    lat = math.radians(float(latitude_deg))
    dec = math.radians(float(declination_deg))
    hour_angle = np.linspace(-math.pi, math.pi, int(hour_angle_samples), endpoint=False)
    cos_theta = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(hour_angle)
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    hist, _ = np.histogram(theta_deg[theta_deg < float(theta_max_deg)], bins=theta_edges_deg)
    total = float(hist.sum())
    if total <= 0.0:
        raise ValueError("Source has no visible hour-angle samples inside theta range")
    return hist.astype(np.float64) / total


def _empty_accumulator(n_r: int, n_theta: int) -> Dict[str, object]:
    return {
        "events_total": 0,
        "events_selected": np.zeros(7, dtype=np.int64),
        "events_dangle_overflow": np.zeros(7, dtype=np.int64),
        "unweighted": np.zeros((7, n_r), dtype=np.float64),
        "mc_weight": np.zeros((7, n_r), dtype=np.float64),
        "mc_weight2": np.zeros((7, n_r), dtype=np.float64),
        "theta_weight": np.zeros((7, n_theta, n_r), dtype=np.float64),
        "theta_weight2": np.zeros((7, n_theta, n_r), dtype=np.float64),
        "files_read": 0,
        "files_failed": 0,
        "errors": [],
    }


def _flush_buffers(
    accumulator: Dict[str, object],
    buffers: Dict[str, List[np.ndarray]],
    *,
    n_r: int,
    n_theta: int,
) -> None:
    if not buffers["unweighted_bin"] and not buffers["bin"]:
        return
    shape_1d = (7, n_r)
    if buffers["unweighted_bin"]:
        unweighted_bin = np.concatenate(buffers["unweighted_bin"])
        unweighted_r = np.concatenate(buffers["unweighted_r"])
        flat_unweighted = unweighted_bin * n_r + unweighted_r
        accumulator["unweighted"] += np.bincount(flat_unweighted, minlength=7 * n_r).reshape(shape_1d)
    if buffers["bin"]:
        bin_idx = np.concatenate(buffers["bin"])
        r_idx = np.concatenate(buffers["r"])
        weight = np.concatenate(buffers["weight"])
        flat = bin_idx * n_r + r_idx
        accumulator["mc_weight"] += np.bincount(flat, weights=weight, minlength=7 * n_r).reshape(shape_1d)
        accumulator["mc_weight2"] += np.bincount(flat, weights=weight * weight, minlength=7 * n_r).reshape(shape_1d)

    if buffers["theta"]:
        theta_bin = np.concatenate(buffers["theta"])
        theta_event_bin = np.concatenate(buffers["theta_event_bin"])
        theta_r = np.concatenate(buffers["theta_r"])
        theta_weight = np.concatenate(buffers["theta_event_weight"])
        flat3 = (theta_event_bin * n_theta + theta_bin) * n_r + theta_r
        shape_3d = (7, n_theta, n_r)
        accumulator["theta_weight"] += np.bincount(
            flat3, weights=theta_weight, minlength=7 * n_theta * n_r
        ).reshape(shape_3d)
        accumulator["theta_weight2"] += np.bincount(
            flat3, weights=theta_weight * theta_weight, minlength=7 * n_theta * n_r
        ).reshape(shape_3d)
    for chunks in buffers.values():
        chunks.clear()


def scan_worker(
    files: Sequence[str],
    *,
    tree_name: str,
    fitstat_equals: int,
    theta_max_deg: float,
    dangle_max_deg: float,
    dangle_bin_width_deg: float,
    theta_bin_width_deg: float,
    flush_files: int,
    strict_files: bool,
    nhit_definition: str,
    hit_charge_branch: str,
    hit_charge_threshold: float,
    hit_time_window_ns: float,
    pincness_calibration: float,
    rmds_max: Optional[float],
    dcedge_min: Optional[float],
) -> Dict[str, object]:
    import awkward as ak
    import uproot

    n_r = int(round(dangle_max_deg / dangle_bin_width_deg))
    n_theta = int(round(theta_max_deg / theta_bin_width_deg))
    accumulator = _empty_accumulator(n_r, n_theta)
    buffers: Dict[str, List[np.ndarray]] = {
        "unweighted_bin": [],
        "unweighted_r": [],
        "bin": [],
        "r": [],
        "weight": [],
        "theta": [],
        "theta_event_bin": [],
        "theta_r": [],
        "theta_event_weight": [],
    }

    for file_number, file_name in enumerate(files, start=1):
        try:
            with uproot.open(file_name) as root_file:
                tree = root_file[tree_name]
                if nhit_definition == "pass5_nq05t30":
                    branches = (*SCALAR_BRANCHES, hit_charge_branch, "vt")
                else:
                    branches = (*SCALAR_BRANCHES, nhit_definition)
                arrays = tree.arrays(branches, library="ak")
            scalars = {name: ak.to_numpy(arrays[name]) for name in SCALAR_BRANCHES}
            if nhit_definition == "pass5_nq05t30":
                nhit = ak.to_numpy(
                    ak.sum(
                        (arrays[hit_charge_branch] >= float(hit_charge_threshold))
                        & (abs(arrays["vt"]) <= float(hit_time_window_ns)),
                        axis=1,
                    )
                )
            else:
                nhit = ak.to_numpy(arrays[nhit_definition])
            accumulator["events_total"] += int(len(nhit))
            bins = assign_pass5_bins(
                nhit,
                scalars["pincness"],
                scalars["fitstat"],
                scalars["theta"],
                scalars["mc_dangle"],
                scalars["rmds"],
                scalars["dcedge"],
                fitstat_equals=fitstat_equals,
                theta_max_deg=theta_max_deg,
                pincness_calibration=pincness_calibration,
                rmds_max=rmds_max,
                dcedge_min=dcedge_min,
            )
            selected = bins >= 0
            if not np.any(selected):
                accumulator["files_read"] += 1
                continue
            selected_bins = bins[selected].astype(np.int64)
            np.add.at(accumulator["events_selected"], selected_bins, 1)
            dangle_deg = np.degrees(np.asarray(scalars["mc_dangle"], dtype=np.float64)[selected])
            weight = np.asarray(scalars["mc_weight"], dtype=np.float64)[selected]
            valid_weight = np.isfinite(weight) & (weight > 0.0)
            r_idx = np.floor(dangle_deg / dangle_bin_width_deg).astype(np.int64)
            in_profile = (r_idx >= 0) & (r_idx < n_r)
            if np.any(~in_profile):
                np.add.at(accumulator["events_dangle_overflow"], selected_bins[~in_profile], 1)

            if np.any(in_profile):
                buffers["unweighted_bin"].append(selected_bins[in_profile])
                buffers["unweighted_r"].append(r_idx[in_profile])
            weighted_profile_valid = in_profile & valid_weight
            if np.any(weighted_profile_valid):
                buffers["bin"].append(selected_bins[weighted_profile_valid])
                buffers["r"].append(r_idx[weighted_profile_valid])
                buffers["weight"].append(weight[weighted_profile_valid])

                mc_theta_deg = np.degrees(np.asarray(scalars["mc_theta"], dtype=np.float64)[selected])
                theta_idx = np.floor(mc_theta_deg / theta_bin_width_deg).astype(np.int64)
                theta_valid = weighted_profile_valid & np.isfinite(mc_theta_deg) & (theta_idx >= 0) & (theta_idx < n_theta)
                if np.any(theta_valid):
                    buffers["theta"].append(theta_idx[theta_valid])
                    buffers["theta_event_bin"].append(selected_bins[theta_valid])
                    buffers["theta_r"].append(r_idx[theta_valid])
                    buffers["theta_event_weight"].append(weight[theta_valid])
            accumulator["files_read"] += 1
        except Exception as exc:
            accumulator["files_failed"] += 1
            if len(accumulator["errors"]) < 20:
                accumulator["errors"].append(f"{file_name}: {type(exc).__name__}: {exc}")
            if strict_files:
                raise
        if file_number % max(1, int(flush_files)) == 0:
            _flush_buffers(accumulator, buffers, n_r=n_r, n_theta=n_theta)
    _flush_buffers(accumulator, buffers, n_r=n_r, n_theta=n_theta)
    return accumulator


def merge_accumulators(target: Dict[str, object], source: Mapping[str, object]) -> None:
    for key in (
        "events_total",
        "files_read",
        "files_failed",
    ):
        target[key] += int(source[key])
    for key in (
        "events_selected",
        "events_dangle_overflow",
        "unweighted",
        "mc_weight",
        "mc_weight2",
        "theta_weight",
        "theta_weight2",
    ):
        target[key] += np.asarray(source[key])
    remaining = max(0, 20 - len(target["errors"]))
    target["errors"].extend(list(source["errors"])[:remaining])


def split_files(files: Sequence[Path], workers: int) -> List[List[str]]:
    workers = max(1, min(int(workers), len(files)))
    chunks: List[List[str]] = [[] for _ in range(workers)]
    for index, file_name in enumerate(files):
        chunks[index % workers].append(str(file_name))
    return [chunk for chunk in chunks if chunk]


def effective_events(sumw: float, sumw2: float) -> float:
    return (sumw * sumw / sumw2) if sumw > 0.0 and sumw2 > 0.0 else 0.0


def fold_crab_profile(
    theta_weight: np.ndarray,
    theta_weight2: np.ndarray,
    target_probability: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    theta_sumw = np.sum(theta_weight, axis=1)
    total = float(np.sum(theta_sumw))
    mc_probability = theta_sumw / total if total > 0.0 else np.zeros_like(theta_sumw)
    ratio = np.zeros_like(mc_probability)
    supported = (target_probability > 0.0) & (mc_probability > 0.0)
    ratio[supported] = target_probability[supported] / mc_probability[supported]
    profile = np.sum(theta_weight * ratio[:, None], axis=0)
    profile_sumw2 = np.sum(theta_weight2 * ratio[:, None] ** 2, axis=0)
    missing_mass = float(np.sum(target_probability[(target_probability > 0.0) & (mc_probability <= 0.0)]))
    return profile, profile_sumw2, {
        "mc_theta_probability": mc_probability,
        "theta_ratio": ratio,
        "missing_target_probability_mass": missing_mass,
    }


def model_shell_mass(edges_deg: np.ndarray, a_core: float, sigma_core: float, sigma_tail: float) -> np.ndarray:
    low = np.asarray(edges_deg[:-1], dtype=np.float64)
    high = np.asarray(edges_deg[1:], dtype=np.float64)
    core = np.exp(-0.5 * (low / sigma_core) ** 2) - np.exp(-0.5 * (high / sigma_core) ** 2)
    tail = np.exp(-0.5 * (low / sigma_tail) ** 2) - np.exp(-0.5 * (high / sigma_tail) ** 2)
    return a_core * core + (1.0 - a_core) * tail


def mixture_cdf(radius_deg: np.ndarray | float, a_core: float, sigma_core: float, sigma_tail: float):
    radius = np.asarray(radius_deg, dtype=np.float64)
    return 1.0 - a_core * np.exp(-0.5 * (radius / sigma_core) ** 2) - (1.0 - a_core) * np.exp(
        -0.5 * (radius / sigma_tail) ** 2
    )


def mixture_quantile(probability: float, a_core: float, sigma_core: float, sigma_tail: float) -> float:
    low = 0.0
    high = max(1.0, 12.0 * sigma_tail)
    for _ in range(80):
        mid = 0.5 * (low + high)
        if float(mixture_cdf(mid, a_core, sigma_core, sigma_tail)) < probability:
            low = mid
        else:
            high = mid
    return high


def histogram_quantile(edges_deg: np.ndarray, shell_weight: np.ndarray, probability: float) -> float:
    values = np.clip(np.asarray(shell_weight, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return float("nan")
    target = probability * total
    cumulative = np.cumsum(values)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(values) - 1)
    before = float(cumulative[index - 1]) if index else 0.0
    fraction = (target - before) / values[index] if values[index] > 0.0 else 0.0
    return float(edges_deg[index] + np.clip(fraction, 0.0, 1.0) * (edges_deg[index + 1] - edges_deg[index]))


def histogram_quantile_with_right_overflow(
    edges_deg: np.ndarray,
    shell_weight: np.ndarray,
    overflow_weight: float,
    probability: float,
) -> float:
    """Return an unconditioned quantile when all overflow is above the last edge."""
    values = np.clip(np.asarray(shell_weight, dtype=np.float64), 0.0, None)
    overflow = max(float(overflow_weight), 0.0)
    target = float(probability) * (float(values.sum()) + overflow)
    if target > float(values.sum()):
        return float("nan")
    cumulative = np.cumsum(values)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(values) - 1)
    before = float(cumulative[index - 1]) if index else 0.0
    fraction = (target - before) / values[index] if values[index] > 0.0 else 0.0
    return float(edges_deg[index] + np.clip(fraction, 0.0, 1.0) * (edges_deg[index + 1] - edges_deg[index]))


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _unpack_fit(parameters: np.ndarray) -> Tuple[float, float, float]:
    x0 = float(parameters[0])
    a_core = 1.0 / (1.0 + math.exp(-x0))
    sigma_core = math.exp(float(parameters[1]))
    sigma_tail = sigma_core + math.exp(float(parameters[2]))
    return a_core, sigma_core, sigma_tail


def fit_double_rayleigh(
    edges_deg: np.ndarray,
    shell_weight: np.ndarray,
    *,
    reference: Optional[Tuple[float, float, float]] = None,
) -> FitResult:
    from scipy.optimize import minimize

    observed = np.clip(np.asarray(shell_weight, dtype=np.float64), 0.0, None)
    total = float(observed.sum())
    if total <= 0.0:
        return FitResult(False, float("nan"), float("nan"), float("nan"), float("nan"), "empty profile")
    q = observed / total
    centers = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    second_moment = float(np.sum(q * centers * centers))
    sigma_single = max(math.sqrt(max(second_moment / 2.0, 1.0e-8)), 0.01)
    r68 = histogram_quantile(edges_deg, observed, 0.68)
    starts: List[Tuple[float, float, float]] = [
        (0.8, max(0.5 * sigma_single, 0.01), max(1.8 * sigma_single, 0.03)),
        (0.5, max(0.4 * r68, 0.01), max(r68, 0.03)),
        (0.9, max(0.55 * sigma_single, 0.01), max(2.5 * sigma_single, 0.03)),
    ]
    if reference is not None:
        starts.insert(0, reference)

    def objective(parameters: np.ndarray) -> float:
        a_core, sigma_core, sigma_tail = _unpack_fit(parameters)
        mass = model_shell_mass(edges_deg, a_core, sigma_core, sigma_tail)
        norm = float(mass.sum())
        if not math.isfinite(norm) or norm <= 0.0:
            return 1.0e30
        probability = np.clip(mass / norm, 1.0e-300, None)
        return float(-np.sum(q * np.log(probability)))

    best = None
    bounds = [(-9.0, 9.0), (math.log(0.003), math.log(5.0)), (math.log(0.003), math.log(10.0))]
    for a0, s10, s20 in starts:
        s10 = max(float(s10), 0.0031)
        s20 = max(float(s20), s10 + 0.0031)
        initial = np.asarray([_logit(a0), math.log(s10), math.log(s20 - s10)], dtype=np.float64)
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    a_core, sigma_core, sigma_tail = _unpack_fit(best.x)
    return FitResult(
        bool(best.success),
        a_core,
        sigma_core,
        sigma_tail,
        float(best.fun),
        str(best.message),
    )


def profile_metrics(
    edges_deg: np.ndarray,
    shell_weight: np.ndarray,
    fit: FitResult,
    xsq: Tuple[float, float, float, float, float],
) -> Dict[str, float | str | bool]:
    observed = np.clip(np.asarray(shell_weight, dtype=np.float64), 0.0, None)
    empirical_probability = observed / observed.sum()
    empirical_cdf = np.cumsum(empirical_probability)
    a_xsq, s1_xsq, s2_xsq, xsq_saved_r68, xsq_saved_r999 = xsq

    def comparison(a: float, s1: float, s2: float) -> Tuple[float, float]:
        mass = model_shell_mass(edges_deg, a, s1, s2)
        probability = np.clip(mass / mass.sum(), 1.0e-300, None)
        positive = empirical_probability > 0.0
        kl = float(np.sum(empirical_probability[positive] * np.log(empirical_probability[positive] / probability[positive])))
        ks = float(np.max(np.abs(empirical_cdf - np.cumsum(probability))))
        return kl, ks

    fit_kl, fit_ks = comparison(fit.a_core, fit.sigma_core_deg, fit.sigma_tail_deg)
    xsq_kl, xsq_ks = comparison(a_xsq, s1_xsq, s2_xsq)
    empirical_r68 = histogram_quantile(edges_deg, observed, 0.68)
    empirical_r999 = histogram_quantile(edges_deg, observed, 0.999)
    fit_r68 = mixture_quantile(0.68, fit.a_core, fit.sigma_core_deg, fit.sigma_tail_deg)
    fit_r999 = mixture_quantile(0.999, fit.a_core, fit.sigma_core_deg, fit.sigma_tail_deg)
    xsq_model_r68 = mixture_quantile(0.68, a_xsq, s1_xsq, s2_xsq)
    xsq_model_r999 = mixture_quantile(0.999, a_xsq, s1_xsq, s2_xsq)
    r68_relative = empirical_r68 / xsq_saved_r68 - 1.0
    if xsq_ks <= 0.03 and abs(r68_relative) <= 0.10:
        agreement = "strong"
    elif xsq_ks <= 0.08 and abs(r68_relative) <= 0.20:
        agreement = "compatible"
    else:
        agreement = "different"
    return {
        "fit_success": fit.success,
        "fit_A_core": fit.a_core,
        "fit_sigma_core_deg": fit.sigma_core_deg,
        "fit_sigma_tail_deg": fit.sigma_tail_deg,
        "fit_objective": fit.objective,
        "fit_message": fit.message,
        "fit_kl": fit_kl,
        "fit_ks": fit_ks,
        "fit_r68_deg": fit_r68,
        "fit_r999_deg": fit_r999,
        "empirical_r68_deg": empirical_r68,
        "empirical_r999_deg": empirical_r999,
        "xsq_kl": xsq_kl,
        "xsq_ks": xsq_ks,
        "xsq_model_r68_deg": xsq_model_r68,
        "xsq_model_r999_deg": xsq_model_r999,
        "xsq_saved_r68_deg": xsq_saved_r68,
        "xsq_saved_r999_deg": xsq_saved_r999,
        "empirical_r68_minus_xsq_fraction": r68_relative,
        "agreement": agreement,
    }


def discover_files(args: argparse.Namespace) -> List[Path]:
    files = sorted(Path(args.input_root).glob(args.input_glob))
    selected: List[Path] = []
    for file_name in files:
        match = FILE_INDEX_RE.search(file_name.name)
        file_index = int(match.group(1)) if match else None
        if args.file_index_min is not None and (file_index is None or file_index < args.file_index_min):
            continue
        if args.file_index_max is not None and (file_index is None or file_index >= args.file_index_max):
            continue
        selected.append(file_name)
    if args.max_files is not None:
        selected = selected[: max(0, int(args.max_files))]
    if not selected:
        raise FileNotFoundError(f"No input files matched {args.input_root}/{args.input_glob}")
    return selected


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]], primary_variant: str, metadata: Mapping[str, object]) -> None:
    primary = [row for row in rows if row["variant"] == primary_variant]
    unweighted = {int(row["bin_index"]): row for row in rows if row["variant"] == "unweighted"}
    lines = [
        "# Pass5 PSF comparison from official WCDA MC",
        "",
        f"Primary profile: `{primary_variant}`. Files read: {metadata['files_read']}; failed: {metadata['files_failed']}.",
        "",
        "| Nhit | events | dangle > 10 deg | Neff | A core fit/XSQ | sigma core fit/XSQ [deg] | sigma tail fit/XSQ [deg] | r68 central/XSQ [deg] | raw-MC full r68 [deg] | KS vs XSQ | agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in primary:
        raw_row = unweighted[int(row["bin_index"])]
        raw_full_r68 = raw_row["full_sample_r68_deg"]
        raw_full_r68_text = f"{float(raw_full_r68):.4f}" if raw_full_r68 is not None else f"> {metadata['histogram']['dangle_edges_deg'][-1]:.1f}"
        lines.append(
            (
                "| {nhit_bin} | {selected_events} | {dangle_overflow_fraction:.2%} | {effective_events:.1f} | {fit_A_core:.4f}/{xsq_A_core:.4f} | "
                "{fit_sigma_core_deg:.4f}/{xsq_sigma_core_deg:.4f} | {fit_sigma_tail_deg:.4f}/{xsq_sigma_tail_deg:.4f} | "
                "{empirical_r68_deg:.4f}/{xsq_saved_r68_deg:.4f} | "
                + raw_full_r68_text
                + " | {xsq_ks:.4f} | {agreement} |"
            ).format(**row)
        )
    lines.extend(
        [
            "",
            "The fitted PSF, central r68, and KS values are conditional on the configured dangle histogram range. The raw-MC full r68 uses unit-weight events and includes right overflow; weighted overflow is not available from the compact profile.",
            "",
            "`agreement` is a diagnostic label, not a statistical hypothesis test. Parameter comparison uses ordered core/tail labels; XSQ's raw `fraction` always multiplies raw `sigma1`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(
    output_dir: Path,
    edges_deg: np.ndarray,
    profiles: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, object]],
    primary_variant: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_lookup = {(str(row["variant"]), int(row["bin_index"])): row for row in rows}
    centers = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    width = np.diff(edges_deg)
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.8), constrained_layout=True)
    for bin_index, axis in enumerate(axes.flat):
        if bin_index >= 7:
            axis.axis("off")
            continue
        row = row_lookup[(primary_variant, bin_index)]
        shell = np.asarray(profiles[primary_variant][bin_index], dtype=np.float64)
        density = shell / shell.sum() / width
        fit_mass = model_shell_mass(
            edges_deg,
            float(row["fit_A_core"]),
            float(row["fit_sigma_core_deg"]),
            float(row["fit_sigma_tail_deg"]),
        )
        xsq_mass = model_shell_mass(
            edges_deg,
            float(row["xsq_A_core"]),
            float(row["xsq_sigma_core_deg"]),
            float(row["xsq_sigma_tail_deg"]),
        )
        axis.step(centers, density, where="mid", color="#111827", lw=1.0, label="MC")
        axis.plot(centers, fit_mass / fit_mass.sum() / width, color="#0072B2", lw=1.8, label="MC 2R fit")
        axis.plot(centers, xsq_mass / xsq_mass.sum() / width, color="#D55E00", lw=1.6, ls="--", label="XSQ")
        axis.axvline(float(row["xsq_saved_r68_deg"]), color="#D55E00", lw=0.9, ls=":")
        x_max = min(float(edges_deg[-1]), max(1.0, 4.0 * float(row["xsq_saved_r68_deg"])))
        axis.set_xlim(0.0, x_max)
        positive = density[density > 0.0]
        if positive.size:
            axis.set_ylim(max(float(np.max(density)) * 1.0e-5, float(np.min(positive)) * 0.5), float(np.max(density)) * 1.8)
        axis.set_yscale("log")
        axis.set_title(f"{NHIT_LABELS[bin_index]}  KS={float(row['xsq_ks']):.3f}")
        axis.set_xlabel("dangle [deg]")
        axis.set_ylabel("radial density [deg$^{-1}$]")
        axis.grid(alpha=0.22)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "pass5_psf_mc_profile_comparison.png", dpi=180)
    fig.savefig(output_dir / "pass5_psf_mc_profile_comparison.pdf")
    plt.close(fig)

    x = np.arange(7)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    styles = {
        "unweighted": ("o", "#6B7280"),
        "mc_weight": ("s", "#009E73"),
        "crab_theta": ("^", "#0072B2"),
    }
    metrics = (
        ("fit_A_core", "core fraction"),
        ("fit_sigma_core_deg", "core sigma [deg]"),
        ("fit_sigma_tail_deg", "tail sigma [deg]"),
    )
    for variant in VARIANTS:
        variant_rows = [row_lookup[(variant, index)] for index in range(7)]
        marker, color = styles[variant]
        for axis, (key, _) in zip(axes, metrics):
            axis.plot(x, [float(row[key]) for row in variant_rows], marker=marker, color=color, label=variant)
    xsq_rows = [row_lookup[(primary_variant, index)] for index in range(7)]
    for axis, (key, label) in zip(axes, metrics):
        xsq_key = key.replace("fit_", "xsq_")
        axis.plot(x, [float(row[xsq_key]) for row in xsq_rows], "D--", color="#D55E00", label="XSQ")
        axis.set_xticks(x, [str(i + 1) for i in x])
        axis.set_xlabel("Pass5 Nhit bin")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "pass5_psf_mc_parameter_comparison.png", dpi=180)
    fig.savefig(output_dir / "pass5_psf_mc_parameter_comparison.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    started = time.time()
    files = discover_files(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    n_r = int(round(args.dangle_max_deg / args.dangle_bin_width_deg))
    n_theta = int(round(args.theta_max_deg / args.theta_bin_width_deg))
    dangle_edges = np.linspace(0.0, args.dangle_max_deg, n_r + 1)
    theta_edges = np.linspace(0.0, args.theta_max_deg, n_theta + 1)
    chunks = split_files(files, args.workers)
    accumulator = _empty_accumulator(n_r, n_theta)
    worker_kwargs = {
        "tree_name": args.tree_name,
        "fitstat_equals": args.fitstat_equals,
        "theta_max_deg": args.theta_max_deg,
        "dangle_max_deg": args.dangle_max_deg,
        "dangle_bin_width_deg": args.dangle_bin_width_deg,
        "theta_bin_width_deg": args.theta_bin_width_deg,
        "flush_files": args.flush_files,
        "strict_files": args.strict_files,
        "nhit_definition": args.nhit_definition,
        "hit_charge_branch": args.hit_charge_branch,
        "hit_charge_threshold": args.hit_charge_threshold,
        "hit_time_window_ns": args.hit_time_window_ns,
        "pincness_calibration": args.pincness_calibration,
        "rmds_max": args.rmds_max,
        "dcedge_min": args.dcedge_min,
    }
    print(f"Scanning {len(files)} files with {len(chunks)} workers", flush=True)
    if len(chunks) == 1:
        merge_accumulators(accumulator, scan_worker(chunks[0], **worker_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(scan_worker, chunk, **worker_kwargs) for chunk in chunks]
            for completed, future in enumerate(as_completed(futures), start=1):
                merge_accumulators(accumulator, future.result())
                print(
                    f"Completed worker {completed}/{len(futures)}; files_read={accumulator['files_read']} "
                    f"failed={accumulator['files_failed']}",
                    flush=True,
                )

    target_theta = crab_theta_probability(
        theta_edges,
        latitude_deg=args.latitude_deg,
        declination_deg=args.source_dec_deg,
        theta_max_deg=args.theta_max_deg,
        hour_angle_samples=args.hour_angle_samples,
    )
    profiles = {
        "unweighted": np.asarray(accumulator["unweighted"]),
        "mc_weight": np.asarray(accumulator["mc_weight"]),
        "crab_theta": np.zeros((7, n_r), dtype=np.float64),
    }
    profile_sumw2 = {
        "unweighted": np.asarray(accumulator["unweighted"]),
        "mc_weight": np.asarray(accumulator["mc_weight2"]),
        "crab_theta": np.zeros((7, n_r), dtype=np.float64),
    }
    theta_metadata: List[Dict[str, object]] = []
    for bin_index in range(7):
        profile, sumw2, meta = fold_crab_profile(
            np.asarray(accumulator["theta_weight"])[bin_index],
            np.asarray(accumulator["theta_weight2"])[bin_index],
            target_theta,
        )
        profiles["crab_theta"][bin_index] = profile
        profile_sumw2["crab_theta"][bin_index] = sumw2
        theta_metadata.append(meta)

    fit_stop = int(np.searchsorted(dangle_edges, args.fit_max_deg, side="left"))
    fit_edges = dangle_edges[: fit_stop + 1]
    rows: List[Dict[str, object]] = []
    for variant in VARIANTS:
        for bin_index in range(7):
            a_xsq, s1_xsq, s2_xsq, r68_xsq, r999_xsq = ordered_xsq(bin_index)
            shell = profiles[variant][bin_index, :fit_stop]
            fit = fit_double_rayleigh(fit_edges, shell, reference=(a_xsq, s1_xsq, s2_xsq))
            metrics = profile_metrics(
                dangle_edges,
                profiles[variant][bin_index],
                fit,
                (a_xsq, s1_xsq, s2_xsq, r68_xsq, r999_xsq),
            )
            sumw = float(np.sum(profiles[variant][bin_index]))
            sumw2 = float(np.sum(profile_sumw2[variant][bin_index]))
            selected_events = int(np.asarray(accumulator["events_selected"])[bin_index])
            overflow_events = int(np.asarray(accumulator["events_dangle_overflow"])[bin_index])
            full_sample_r68 = None
            if variant == "unweighted":
                value = histogram_quantile_with_right_overflow(
                    dangle_edges,
                    profiles[variant][bin_index],
                    overflow_events,
                    0.68,
                )
                full_sample_r68 = value if math.isfinite(value) else None
            raw_frac, raw_s1, raw_s2, _, _ = XSQ_RAW[bin_index]
            rows.append(
                {
                    "variant": variant,
                    "bin_index": bin_index,
                    "nhit_bin": NHIT_LABELS[bin_index],
                    "pincness_max": float(PASS5_PINC_MAX[bin_index]),
                    "selected_events": selected_events,
                    "dangle_overflow_events": overflow_events,
                    "dangle_overflow_fraction": overflow_events / selected_events if selected_events else float("nan"),
                    "full_sample_r68_deg": full_sample_r68,
                    "sumw": sumw,
                    "effective_events": effective_events(sumw, sumw2),
                    "xsq_raw_fraction": raw_frac,
                    "xsq_raw_sigma1_deg": raw_s1,
                    "xsq_raw_sigma2_deg": raw_s2,
                    "xsq_A_core": a_xsq,
                    "xsq_sigma_core_deg": s1_xsq,
                    "xsq_sigma_tail_deg": s2_xsq,
                    "theta_missing_target_mass": float(theta_metadata[bin_index]["missing_target_probability_mass"]),
                    **metrics,
                }
            )

    metadata: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "input_root": str(Path(args.input_root).resolve()),
        "input_glob": args.input_glob,
        "files_requested": len(files),
        "files_read": int(accumulator["files_read"]),
        "files_failed": int(accumulator["files_failed"]),
        "file_errors": list(accumulator["errors"]),
        "events_total": int(accumulator["events_total"]),
        "selection": {
            "nhit_edges": NHIT_EDGES,
            "pincness_max_by_bin": PASS5_PINC_MAX,
            "pincness_calibration_divisor": args.pincness_calibration,
            "fitstat_equals": args.fitstat_equals,
            "reconstructed_theta_max_deg": args.theta_max_deg,
            "rmds_max": args.rmds_max,
            "dcedge_min": args.dcedge_min,
            "nhit_definition": args.nhit_definition,
            "hit_charge_branch": args.hit_charge_branch,
            "hit_charge_threshold": args.hit_charge_threshold,
            "hit_time_window_ns": args.hit_time_window_ns,
            "nhit_upper_bound_exclusive": int(NHIT_EDGES[-1]),
        },
        "histogram": {
            "dangle_edges_deg": dangle_edges,
            "theta_edges_deg": theta_edges,
            "fit_max_deg": args.fit_max_deg,
            "central_metrics_condition": f"0 <= dangle < {args.dangle_max_deg:g} deg",
            "unweighted_full_sample_r68_includes_right_overflow": True,
        },
        "crab_theta_weighting": {
            "latitude_deg": args.latitude_deg,
            "source_declination_deg": args.source_dec_deg,
            "hour_angle_samples": args.hour_angle_samples,
            "target_probability": target_theta,
            "per_nhit": theta_metadata,
        },
        "xsq_reference": {
            "source": "/home/lhaaso/xishaoqiang/lhaaso/data/pass5/z50/WCDA/irfs.root",
            "raw_columns": ["fraction", "sigma1_deg", "sigma2_deg", "saved_r68_deg", "saved_r999_deg"],
            "raw_values": XSQ_RAW,
            "note": "Ordered core/tail values swap fraction when raw sigma1 > raw sigma2.",
        },
        "variants": {
            "unweighted": "Selected MC events with unit weight.",
            "mc_weight": "Selected MC events weighted by the primary-particle mc_weight branch.",
            "crab_theta": "mc_weight profile folded to the uniform-hour-angle theta exposure of a source at Crab declination.",
        },
        "primary_variant": args.primary_variant,
    }
    write_csv(output_dir / "pass5_psf_mc_comparison_summary.csv", rows)
    (output_dir / "pass5_psf_mc_comparison.json").write_text(
        json.dumps(json_ready({"metadata": metadata, "rows": rows}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "pass5_psf_mc_profiles.npz",
        dangle_edges_deg=dangle_edges,
        theta_edges_deg=theta_edges,
        target_theta_probability=target_theta,
        unweighted_profile=profiles["unweighted"],
        mc_weight_profile=profiles["mc_weight"],
        crab_theta_profile=profiles["crab_theta"],
        unweighted_sumw2=profile_sumw2["unweighted"],
        mc_weight_sumw2=profile_sumw2["mc_weight"],
        crab_theta_sumw2=profile_sumw2["crab_theta"],
        selected_events=np.asarray(accumulator["events_selected"]),
    )
    write_markdown(
        output_dir / "pass5_psf_mc_comparison.md",
        rows,
        args.primary_variant,
        metadata,
    )
    if not args.no_plots:
        make_plots(output_dir, fit_edges, {key: value[:, :fit_stop] for key, value in profiles.items()}, rows, args.primary_variant)
    print(f"Wrote {output_dir}", flush=True)
    return 0 if int(accumulator["files_failed"]) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
