#!/usr/bin/env python
"""Check 2B: low-threshold effective-area audit for the v6 Stage A response.

This is a *read-only* diagnostic. It never touches a pipeline stage, a frozen
output, or the remote cluster. It loads the Stage A response NPZ written by
``stages/01_build_response.py`` (keys ``eta``, ``a_eff`` with shape
``(n_cells, n_logE_true, n_theta)``, plus ``logE_true_edges``,
``theta_true_edges_deg``, ``cell_id``, ``nhit_bin``, ``predE_bin``, ``s0_m2``)
and audits the *lowest-Nhit* and *lowest-predE* cells for the sanity properties
that a physically well-behaved effective area must satisfy:

  * ``a_eff`` finite and > 0 where the response has support;
  * detection efficiency ``eta <= 1`` everywhere (a probability);
  * ``s0_m2`` consistency: 4e6 m^2 == 4e10 cm^2 (the injection area);
  * ``a_eff`` monotone increasing with true energy, per cell (the low-energy
    turn-on should not un-turn-on toward high energy);
  * a magnitude/shape comparison of ``a_eff(E_true)`` between the two lowest
    Nhit bins ``[100,200)`` and ``[200,300)`` (a higher-Nhit bin should reach a
    larger effective area at its most efficient energies).

The effective-area convention matches Stage A:
``a_eff = s0_m2 * cos(theta_true_bin_center) * eta`` in m^2.

Outputs a JSON of pass/fail flags and a PNG plotting ``a_eff(E_true)`` curves
for the audited low cells. ``--self-test`` fabricates a tiny synthetic response
so the numeric core runs without any cluster data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESPONSE_NPZ = "apply/output/stage_a/response_2d.npz"
DEFAULT_RESPONSE_METADATA = "apply/output/stage_a/response_2d_metadata.json"

# Injection / thrown-area constants, mirrored from stages/01_build_response.py.
EXPECTED_S0_M2 = 4.0e6
EXPECTED_S0_CM2 = 4.0e10
CM2_PER_M2 = 1.0e4

# Two lowest Nhit bins compared head-to-head (check 2B).
LOWEST_NHIT_LABELS = ("[100,200)", "[200,300)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check 2B: audit the low-threshold effective area of the v6 Stage A response."
    )
    parser.add_argument("--response-npz", type=str, default=DEFAULT_RESPONSE_NPZ)
    parser.add_argument("--response-metadata", type=str, default=DEFAULT_RESPONSE_METADATA)
    parser.add_argument("--output-json", type=str, default="apply/output/diagnostics/aeff_threshold_audit.json")
    parser.add_argument("--output-png", type=str, default="apply/output/diagnostics/aeff_threshold_audit.png")
    parser.add_argument(
        "--monotone-atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance (m^2) allowed on non-monotone a_eff steps before flagging.",
    )
    parser.add_argument(
        "--eta-atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance allowed above 1.0 for eta before flagging eta>1.",
    )
    parser.add_argument(
        "--s0-rtol",
        type=float,
        default=1.0e-6,
        help="Relative tolerance for the s0_m2 = 4e6 m^2 = 4e10 cm^2 consistency check.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help="Fabricate a tiny synthetic response and run the audit without cluster data.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Interval-label parsing (v6 nhit_bin / predE_bin are STRING labels like        #
# "[100,200)" or "[2,2.5)"). Reused sort key mirrors stages/07_sed_points.py.    #
# --------------------------------------------------------------------------- #
def parse_interval(label: str) -> Tuple[Optional[float], Optional[float]]:
    text = str(label).strip()
    if text.lower() in {"all", "*"}:
        return None, None
    if text.startswith("[") and text.endswith(")"):
        low, high = text[1:-1].split(",", 1)
        return float(low), float(high)
    if text.startswith("<"):
        return None, float(text[1:])
    if text.startswith(">="):
        return float(text[2:]), None
    if text.startswith(">"):
        return float(text[1:]), None
    raise ValueError(f"Unsupported interval label: {label!r}")


def interval_lower(label: str) -> float:
    """Lower bound used to sort cells by threshold; -inf/+inf for open ends."""
    low, high = parse_interval(label)
    if low is None and high is None:
        return math.inf
    if low is None:
        return -math.inf
    return float(low)


# --------------------------------------------------------------------------- #
# Response loading                                                             #
# --------------------------------------------------------------------------- #
def load_response(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Response NPZ does not exist: {npz_path}")
    required = ["eta", "a_eff", "logE_true_edges", "theta_true_edges_deg", "cell_id", "nhit_bin", "predE_bin"]
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required arrays: {missing}")
        out: Dict[str, np.ndarray] = {name: data[name].copy() for name in data.files}
    return out


def response_s0_m2(response: Dict[str, np.ndarray], metadata: Optional[Dict[str, object]]) -> float:
    if "s0_m2" in response:
        return float(np.asarray(response["s0_m2"]).reshape(-1)[0])
    if isinstance(metadata, dict) and metadata.get("s0_m2") is not None:
        return float(metadata["s0_m2"])  # type: ignore[arg-type]
    return float("nan")


# --------------------------------------------------------------------------- #
# Per-cell audit                                                              #
# --------------------------------------------------------------------------- #
def logE_true_centers(loge_edges: np.ndarray) -> np.ndarray:
    return 0.5 * (loge_edges[:-1] + loge_edges[1:])


def aeff_theta_summed(a_eff: np.ndarray) -> np.ndarray:
    """Collapse the theta axis by summing, giving a_eff(cell, logE) proxy shape.

    A theta-summed curve is the most robust 1D proxy of the effective-area
    turn-on: the per-theta cos(theta) weighting only rescales it and does not
    change the monotone-in-energy question we audit.
    """
    return np.nansum(np.asarray(a_eff, dtype=np.float64), axis=2)


def audit_cell(
    *,
    cell_index: int,
    cell_id: int,
    nhit_bin: str,
    predE_bin: str,
    eta_cell: np.ndarray,
    a_eff_cell: np.ndarray,
    loge_centers: np.ndarray,
    monotone_atol: float,
    eta_atol: float,
) -> Dict[str, object]:
    a_eff_e = np.nansum(np.asarray(a_eff_cell, dtype=np.float64), axis=1)  # sum over theta -> (n_logE,)
    eta_flat = np.asarray(eta_cell, dtype=np.float64).reshape(-1)

    support = a_eff_e > 0.0
    finite_where_support = bool(np.all(np.isfinite(a_eff_e[support]))) if np.any(support) else True
    positive_where_support = bool(np.all(a_eff_e[support] > 0.0)) if np.any(support) else True
    a_eff_finite = bool(np.all(np.isfinite(np.asarray(a_eff_cell, dtype=np.float64))))

    finite_eta = eta_flat[np.isfinite(eta_flat)]
    eta_max = float(np.max(finite_eta)) if finite_eta.size else float("nan")
    eta_le_one = bool(finite_eta.size == 0 or eta_max <= 1.0 + float(eta_atol))

    # Monotone-increasing check on the support region only. Below the turn-on
    # the effective area is legitimately zero, so we test the contiguous
    # nonzero span from first support bin to the last.
    monotone = True
    max_backstep = 0.0
    if np.any(support):
        first = int(np.argmax(support))
        seg = a_eff_e[first:]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2:
            diffs = np.diff(seg)
            backsteps = -diffs[diffs < -float(monotone_atol)]
            if backsteps.size:
                monotone = False
                max_backstep = float(np.max(backsteps))

    peak_idx = int(np.nanargmax(a_eff_e)) if np.any(np.isfinite(a_eff_e)) else -1
    return {
        "cell_index": int(cell_index),
        "cell_id": int(cell_id),
        "nhit_bin": str(nhit_bin),
        "predE_bin": str(predE_bin),
        "nhit_lower": interval_lower(str(nhit_bin)),
        "predE_lower": interval_lower(str(predE_bin)),
        "a_eff_finite": a_eff_finite,
        "a_eff_finite_on_support": finite_where_support,
        "a_eff_positive_on_support": positive_where_support,
        "a_eff_pass": bool(a_eff_finite and finite_where_support and positive_where_support),
        "eta_max": eta_max,
        "eta_le_one_pass": eta_le_one,
        "a_eff_monotone_increasing_pass": bool(monotone),
        "max_monotone_backstep_m2": max_backstep,
        "n_support_bins": int(np.count_nonzero(support)),
        "a_eff_max_m2": float(np.nanmax(a_eff_e)) if np.any(np.isfinite(a_eff_e)) else None,
        "a_eff_peak_logE_true": float(loge_centers[peak_idx]) if peak_idx >= 0 else None,
        "a_eff_theta_summed_curve": [float(v) for v in a_eff_e],
    }


def compare_two_lowest_nhit(
    *,
    a_eff: np.ndarray,
    nhit_bins: Sequence[str],
    loge_centers: np.ndarray,
) -> Dict[str, object]:
    """Compare a_eff(E_true) magnitude/shape across the two lowest Nhit bins."""
    a_eff_ce = aeff_theta_summed(a_eff)  # (n_cells, n_logE)
    result: Dict[str, object] = {
        "labels": list(LOWEST_NHIT_LABELS),
        "found": {label: False for label in LOWEST_NHIT_LABELS},
    }
    curves: Dict[str, np.ndarray] = {}
    for label in LOWEST_NHIT_LABELS:
        idx = [i for i, nb in enumerate(nhit_bins) if str(nb) == label]
        if not idx:
            continue
        result["found"][label] = True
        curve = np.nansum(a_eff_ce[idx], axis=0)  # pool over cells sharing this Nhit bin
        curves[label] = curve
        result[f"peak_m2_{label}"] = float(np.nanmax(curve)) if np.any(np.isfinite(curve)) else None
        peak_idx = int(np.nanargmax(curve)) if np.any(np.isfinite(curve)) else -1
        result[f"peak_logE_true_{label}"] = float(loge_centers[peak_idx]) if peak_idx >= 0 else None

    lo_label, hi_label = LOWEST_NHIT_LABELS
    if lo_label in curves and hi_label in curves:
        lo_peak = float(np.nanmax(curves[lo_label]))
        hi_peak = float(np.nanmax(curves[hi_label]))
        result["higher_nhit_reaches_larger_peak"] = bool(hi_peak >= lo_peak)
        result["peak_ratio_hi_over_lo"] = float(hi_peak / lo_peak) if lo_peak > 0.0 else None
        lo_pk = int(np.nanargmax(curves[lo_label]))
        hi_pk = int(np.nanargmax(curves[hi_label]))
        result["higher_nhit_peaks_at_higher_energy"] = bool(loge_centers[hi_pk] >= loge_centers[lo_pk])
        result["comparison_available"] = True
    else:
        result["comparison_available"] = False
    return result


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def setup_matplotlib():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_audit(
    path: Path,
    *,
    audited_cells: Sequence[Dict[str, object]],
    loge_centers: np.ndarray,
    nhit_comparison: Dict[str, object],
    a_eff: np.ndarray,
    nhit_bins: Sequence[str],
) -> None:
    plt = setup_matplotlib()
    e_tev = np.power(10.0, np.asarray(loge_centers, dtype=np.float64)) / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    ax = axes[0]
    for cell in audited_cells:
        curve = np.asarray(cell["a_eff_theta_summed_curve"], dtype=np.float64)
        label = f"cell {cell['cell_id']} {cell['nhit_bin']} {cell['predE_bin']}"
        ax.plot(e_tev, curve, lw=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True energy [TeV]")
    ax.set_ylabel(r"$A_{\mathrm{eff}}$ (theta-summed) [m$^2$]")
    ax.set_title("Lowest-Nhit / lowest-predE cells")
    ax.grid(True, which="both", alpha=0.25)
    if audited_cells:
        ax.legend(fontsize=6, ncol=1)

    ax2 = axes[1]
    a_eff_ce = aeff_theta_summed(a_eff)
    for label in LOWEST_NHIT_LABELS:
        idx = [i for i, nb in enumerate(nhit_bins) if str(nb) == label]
        if not idx:
            continue
        curve = np.nansum(a_eff_ce[idx], axis=0)
        ax2.plot(e_tev, curve, lw=1.6, marker="o", ms=3, label=f"Nhit {label}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("True energy [TeV]")
    ax2.set_ylabel(r"$A_{\mathrm{eff}}$ (theta+cell summed) [m$^2$]")
    ratio = nhit_comparison.get("peak_ratio_hi_over_lo")
    ratio_txt = f"peak ratio hi/lo = {ratio:.3g}" if isinstance(ratio, (int, float)) else "peak ratio n/a"
    ax2.set_title(f"Two lowest Nhit bins ({ratio_txt})")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Synthetic response for --self-test                                          #
# --------------------------------------------------------------------------- #
def synthetic_response() -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Build a tiny physically-plausible response (monotone turn-on, eta<=1)."""
    loge_edges = np.arange(2.0, 6.01, 0.5)  # log10(E/GeV)
    theta_edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    n_logE = loge_edges.size - 1
    n_theta = theta_edges.size - 1
    loge_centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    cos_theta = np.cos(np.radians(theta_centers))

    nhit_bins = ["[100,200)", "[200,300)", "[300,500)"]
    # Each higher-Nhit bin turns on at higher energy and reaches a larger eta.
    turn_on = {"[100,200)": 2.8, "[200,300)": 3.4, "[300,500)": 4.0}
    eta_cap = {"[100,200)": 0.55, "[200,300)": 0.72, "[300,500)": 0.88}
    n_cells = len(nhit_bins)

    eta = np.zeros((n_cells, n_logE, n_theta), dtype=np.float64)
    for c, nb in enumerate(nhit_bins):
        # smooth logistic turn-on in logE, saturating below the eta cap
        curve = eta_cap[nb] / (1.0 + np.exp(-(loge_centers - turn_on[nb]) / 0.35))
        for t in range(n_theta):
            eta[c, :, t] = curve  # theta-independent efficiency for the toy
    a_eff = EXPECTED_S0_M2 * eta * cos_theta[None, None, :]

    response = {
        "eta": eta.astype(np.float32),
        "a_eff": a_eff.astype(np.float32),
        "logE_true_edges": loge_edges,
        "theta_true_edges_deg": theta_edges,
        "cell_id": np.asarray([10 + c for c in range(n_cells)], dtype=np.int32),
        "nhit_bin": np.asarray(nhit_bins, dtype="U32"),
        "predE_bin": np.asarray(["[2,2.5)", "[2.5,3)", "[3,3.5)"], dtype="U32"),
        "s0_m2": np.asarray(EXPECTED_S0_M2, dtype=np.float32),
    }
    metadata = {"s0_m2": EXPECTED_S0_M2, "response_type": "synthetic_self_test"}
    return response, metadata


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def select_low_cells(response: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    nhit_bins = [str(v) for v in response["nhit_bin"]]
    pred_bins = [str(v) for v in response["predE_bin"]]
    nhit_lo = [interval_lower(v) for v in nhit_bins]
    pred_lo = [interval_lower(v) for v in pred_bins]
    min_nhit = min(nhit_lo)
    min_pred = min(pred_lo)
    lowest_nhit_cells = [i for i, v in enumerate(nhit_lo) if v == min_nhit]
    lowest_pred_cells = [i for i, v in enumerate(pred_lo) if v == min_pred]
    audited = sorted(set(lowest_nhit_cells) | set(lowest_pred_cells))
    return {
        "lowest_nhit_cells": lowest_nhit_cells,
        "lowest_predE_cells": lowest_pred_cells,
        "audited_cells": audited,
    }


def run_audit(
    response: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    eta = np.asarray(response["eta"], dtype=np.float64)
    a_eff = np.asarray(response["a_eff"], dtype=np.float64)
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    loge_centers = logE_true_centers(loge_edges)
    cell_id = np.asarray(response["cell_id"]).reshape(-1)
    nhit_bins = [str(v) for v in response["nhit_bin"]]
    pred_bins = [str(v) for v in response["predE_bin"]]
    n_cells = a_eff.shape[0]

    s0_m2 = response_s0_m2(response, metadata)
    s0_ok = (
        math.isfinite(s0_m2)
        and math.isclose(s0_m2, EXPECTED_S0_M2, rel_tol=float(args.s0_rtol))
        and math.isclose(s0_m2 * CM2_PER_M2, EXPECTED_S0_CM2, rel_tol=float(args.s0_rtol))
    )

    selection = select_low_cells(response)
    audited_indices = selection["audited_cells"]
    audited: List[Dict[str, object]] = []
    for idx in audited_indices:
        audited.append(
            audit_cell(
                cell_index=idx,
                cell_id=int(cell_id[idx]),
                nhit_bin=nhit_bins[idx],
                predE_bin=pred_bins[idx],
                eta_cell=eta[idx],
                a_eff_cell=a_eff[idx],
                loge_centers=loge_centers,
                monotone_atol=float(args.monotone_atol),
                eta_atol=float(args.eta_atol),
            )
        )

    nhit_comparison = compare_two_lowest_nhit(a_eff=a_eff, nhit_bins=nhit_bins, loge_centers=loge_centers)

    aeff_pass = all(bool(c["a_eff_pass"]) for c in audited) if audited else False
    eta_pass = all(bool(c["eta_le_one_pass"]) for c in audited) if audited else False
    monotone_pass = all(bool(c["a_eff_monotone_increasing_pass"]) for c in audited) if audited else False
    nhit_cmp_pass = bool(nhit_comparison.get("higher_nhit_reaches_larger_peak", False)) if nhit_comparison.get(
        "comparison_available"
    ) else None

    overall = bool(aeff_pass and eta_pass and monotone_pass and s0_ok and (nhit_cmp_pass in (True, None)))

    return {
        "tool": "aeff_threshold_audit",
        "check": "2B",
        "response_npz": str(getattr(args, "_response_npz_used", "")),
        "response_metadata": str(getattr(args, "_response_metadata_used", "")),
        "n_cells": int(n_cells),
        "s0_m2": s0_m2,
        "s0_cm2": s0_m2 * CM2_PER_M2 if math.isfinite(s0_m2) else None,
        "expected_s0_m2": EXPECTED_S0_M2,
        "expected_s0_cm2": EXPECTED_S0_CM2,
        "flags": {
            "s0_consistency_pass": bool(s0_ok),
            "a_eff_finite_positive_pass": bool(aeff_pass),
            "eta_le_one_pass": bool(eta_pass),
            "a_eff_monotone_increasing_pass": bool(monotone_pass),
            "two_lowest_nhit_comparison_pass": nhit_cmp_pass,
            "overall_pass": overall,
        },
        "selection": {
            "lowest_nhit_cell_ids": [int(cell_id[i]) for i in selection["lowest_nhit_cells"]],
            "lowest_predE_cell_ids": [int(cell_id[i]) for i in selection["lowest_predE_cells"]],
            "audited_cell_ids": [int(cell_id[i]) for i in audited_indices],
        },
        "audited_cells": audited,
        "two_lowest_nhit_comparison": nhit_comparison,
        "logE_true_centers": [float(v) for v in loge_centers],
        "energy_true_tev_centers": [float(10.0**v / 1000.0) for v in loge_centers],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        response, metadata = synthetic_response()
        args._response_npz_used = "<synthetic>"
        args._response_metadata_used = "<synthetic>"
    else:
        npz_path = Path(args.response_npz)
        if not npz_path.is_absolute():
            npz_path = (REPO_ROOT / npz_path).resolve()
        response = load_response(npz_path)
        metadata = None
        meta_path = Path(args.response_metadata)
        if not meta_path.is_absolute():
            meta_path = (REPO_ROOT / meta_path).resolve()
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        args._response_npz_used = str(npz_path)
        args._response_metadata_used = str(meta_path)

    result = run_audit(response, metadata, args)

    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = (REPO_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    out_png = Path(args.output_png)
    if not out_png.is_absolute():
        out_png = (REPO_ROOT / out_png).resolve()
    loge_centers = np.asarray(result["logE_true_centers"], dtype=np.float64)
    plot_audit(
        out_png,
        audited_cells=result["audited_cells"],
        loge_centers=loge_centers,
        nhit_comparison=result["two_lowest_nhit_comparison"],
        a_eff=np.asarray(response["a_eff"], dtype=np.float64),
        nhit_bins=[str(v) for v in response["nhit_bin"]],
    )

    flags = result["flags"]
    print(f"[aeff_threshold_audit] wrote {out_json}")
    print(f"[aeff_threshold_audit] wrote {out_png}")
    print(f"[aeff_threshold_audit] overall_pass={flags['overall_pass']} flags={json.dumps(flags)}")
    return 0 if flags["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
