#!/usr/bin/env python
"""Check 2C: forward-fold a spectrum through v6's response into Pass5 Nhit bins.

This is a *read-only* diagnostic. It never touches a pipeline stage, a frozen
output, or the remote cluster.

What it does
------------
1. Loads v6's Stage A response NPZ (``a_eff`` shape ``(n_cells, n_logE_true,
   n_theta)`` in m^2, plus ``logE_true_edges``, ``theta_true_edges_deg``,
   ``cell_id``, ``nhit_bin``, ``predE_bin``) written by
   ``stages/01_build_response.py``.
2. Forward-folds a chosen spectrum (LogPar or PowerLaw) through that response
   using the *exact same math* as ``stages/06_fit.py::model_counts`` /
   ``integrate_flux_bins`` and the Stage F theta-exposure array:
       S_b = containment_b * 1e4 * sum_{E,theta} A_eff[b,E,theta]
                                    * flux_integral[E] * theta_exposure[theta]
   producing predicted counts per v6 cell.
3. RE-GROUPS the per-cell predicted counts into the *coarser* Nhit-only bins of
   the official Pass5 SED CSV (each v6 ``nhit_bin`` label like ``[100,200)`` is
   nested into a Pass5 bin like ``100-200`` / ``>800``).
4. Converts each regrouped Nhit bin to a model flux (dN/dE at the bin's
   effective true energy) and compares it against the 7 observed Pass5 SED
   points as a predicted/observed ratio.

IMPORTANT / APPROXIMATE
-----------------------
Pass5 uses a different analysis: Nhit range roughly 30-2000 with different cuts,
its own PSF/containment, exposure and background model. v6's response only
covers Nhit >= 100, uses aperture-conditioned cells, and a different livetime.
The two Pass5 bins ``30-60`` and ``60-100`` have *no v6 coverage at all*.
Therefore this comparison is an APPROXIMATE cross-check of shape/normalization,
NOT an apples-to-apples flux measurement. The ratio should be read as
"does v6's forward-folded model land in the right ballpark of Pass5", not as a
calibrated agreement. This caveat is repeated in the JSON output.

Spectrum spec
-------------
By default the spectrum is the Pass5 LogPar returned by
``stages/07_sed_points.py::official_pass5_point_fit`` (a quadratic-in-log fit of
the 7 Pass5 points), so the default run folds Pass5's own spectrum back through
v6's response. Override with ``--model {logpar,powerlaw}`` plus ``--phi0``,
``--alpha``/``--gamma``, ``--beta``, ``--pivot-tev``.

``--self-test`` fabricates a tiny synthetic response + a synthetic Pass5 CSV so
the numeric core runs without any cluster data.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESPONSE_NPZ = "apply/output/stage_a/response_2d.npz"
DEFAULT_RESPONSE_METADATA = "apply/output/stage_a/response_2d_metadata.json"
DEFAULT_PASS5_CSV = "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
DEFAULT_PIVOT_TEV = 3.0
DEFAULT_QUADRATURE_POINTS = 64
DEFAULT_EXPOSURE_SEC = 1484.36 * 86400.0  # Pass5 livetime_days from the CSV, used only as a fallback.

M2_TO_CM2 = 1.0e4


# --------------------------------------------------------------------------- #
# Load the two helper functions from the (numerically-named) stage modules,     #
# so the forward-fold math is literally the pipeline's own code.                #
# --------------------------------------------------------------------------- #
def _load_module(path: Path, name: str):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass introspection reads sys.modules[__module__].
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_stage_helpers() -> Tuple[object, object]:
    stages_dir = REPO_ROOT / "apply" / "stages"
    fit_mod = _load_module(stages_dir / "06_fit.py", "stage06_fit")
    sed_mod = _load_module(stages_dir / "07_sed_points.py", "stage07_sed_points")
    return fit_mod, sed_mod


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check 2C: forward-fold a spectrum through v6's response and re-group into Pass5 Nhit bins."
    )
    parser.add_argument("--response-npz", type=str, default=DEFAULT_RESPONSE_NPZ)
    parser.add_argument("--response-metadata", type=str, default=DEFAULT_RESPONSE_METADATA)
    parser.add_argument("--pass5-csv", type=str, default=DEFAULT_PASS5_CSV)
    parser.add_argument("--output-json", type=str, default="apply/output/diagnostics/pass5_nhit_forward_fold.json")
    parser.add_argument("--output-png", type=str, default="apply/output/diagnostics/pass5_nhit_forward_fold.png")

    parser.add_argument("--model", type=str, choices=["logpar", "powerlaw"], default=None,
                        help="Spectrum model. Default: Pass5 LogPar from official_pass5_point_fit().")
    parser.add_argument("--phi0", type=float, default=None, help="Flux normalization at pivot [TeV^-1 cm^-2 s^-1].")
    parser.add_argument("--alpha", type=float, default=None, help="LogPar alpha (spectral index at pivot).")
    parser.add_argument("--gamma", type=float, default=None, help="PowerLaw spectral index gamma.")
    parser.add_argument("--beta", type=float, default=None, help="LogPar curvature beta.")
    parser.add_argument("--pivot-tev", type=float, default=DEFAULT_PIVOT_TEV)
    parser.add_argument("--quadrature-points", type=int, default=DEFAULT_QUADRATURE_POINTS)

    parser.add_argument(
        "--theta-exposure-json",
        type=str,
        default=None,
        help=(
            "Optional JSON with a 'theta_exposure_sec' list matching theta bins "
            "(e.g. a Stage F fit metadata). If omitted, a flat exposure over all "
            "theta bins summing to livetime_days*86400 (from the Pass5 CSV, or "
            "the fallback constant) is used. Flat exposure only affects absolute "
            "normalization, which the APPROXIMATE caveat already covers."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help="Fabricate a synthetic response + Pass5 CSV and run without cluster data.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Interval parsing for v6 nhit_bin labels and Pass5 nhit bins                   #
# --------------------------------------------------------------------------- #
def parse_v6_interval(label: str) -> Tuple[float, float]:
    """Parse a v6 nhit_bin label like '[100,200)' into (lo, hi)."""
    text = str(label).strip()
    if text.startswith("[") and text.endswith(")"):
        lo, hi = text[1:-1].split(",", 1)
        return float(lo), float(hi)
    if text.startswith(">="):
        return float(text[2:]), math.inf
    if text.startswith(">"):
        return float(text[1:]), math.inf
    if text.startswith("<"):
        return -math.inf, float(text[1:])
    raise ValueError(f"Unsupported v6 nhit interval label: {label!r}")


def parse_pass5_nhit(label: str) -> Tuple[float, float]:
    """Parse a Pass5 nhit bin like '100-200', '>800', '<60' into (lo, hi)."""
    text = str(label).strip()
    if text.startswith(">="):
        return float(text[2:]), math.inf
    if text.startswith(">"):
        return float(text[1:]), math.inf
    if text.startswith("<="):
        return -math.inf, float(text[2:])
    if text.startswith("<"):
        return -math.inf, float(text[1:])
    if "-" in text:
        lo, hi = text.split("-", 1)
        return float(lo), float(hi)
    raise ValueError(f"Unsupported Pass5 nhit bin label: {label!r}")


def map_v6_to_pass5_nhit(v6_label: str, pass5_bins: Sequence[str]) -> Optional[str]:
    """Return the Pass5 bin whose [plo, phi) contains the v6 [lo, hi) interval.

    v6 fine bins are nested inside the coarser Pass5 bins:
      [100,200)->100-200, [200,300)->200-300, [300,500)->300-500,
      [500,800)->500-800, [800,1100)/[1100,2000)/[2000,3000)->>800.
    A v6 bin maps to the first Pass5 bin that fully contains it (v6 lower >=
    plo and v6 upper <= phi). Falls back to a midpoint-membership test if no
    bin fully contains it (never happens for the current schema).
    """
    v_lo, v_hi = parse_v6_interval(v6_label)
    parsed = [(b, *parse_pass5_nhit(b)) for b in pass5_bins]
    for b, p_lo, p_hi in parsed:
        if v_lo >= p_lo - 1e-9 and v_hi <= p_hi + 1e-9:
            return b
    mid = 0.5 * (v_lo + (v_hi if math.isfinite(v_hi) else v_lo * 2.0))
    for b, p_lo, p_hi in parsed:
        if p_lo - 1e-9 <= mid < p_hi:
            return b
    return None


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_response(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Response NPZ does not exist: {npz_path}")
    required = ["a_eff", "logE_true_edges", "theta_true_edges_deg", "cell_id", "nhit_bin", "predE_bin"]
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required arrays: {missing}")
        return {name: data[name].copy() for name in data.files}


def load_pass5_points(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Pass5 CSV does not exist: {csv_path}")
    points: List[Dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                e = float(row["energy_tev"])
                dnde = float(row["flux_per_tev_cm2_s"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append(
                {
                    "energy_tev": e,
                    "flux_per_tev_cm2_s": dnde,
                    "nhit_bin": str(row.get("nhit_bin") or ""),
                    "ts": _maybe_float(row.get("ts")),
                    "is_upper_limit": str(row.get("is_upper_limit") or "").strip().lower() == "true",
                    "livetime_days": _maybe_float(row.get("livetime_days")),
                }
            )
    if not points:
        raise ValueError(f"No usable Pass5 SED points found in {csv_path}")
    return points


def _maybe_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Spectrum resolution                                                         #
# --------------------------------------------------------------------------- #
def resolve_spectrum(args: argparse.Namespace, sed_mod: object) -> Dict[str, object]:
    """Return {model_name, params, pivot_tev, source} for the forward fold."""
    model = args.model
    if model is None:
        # Default: Pass5 LogPar from official_pass5_point_fit (reads its own CSV path).
        fit = sed_mod.official_pass5_point_fit(pivot_tev=float(args.pivot_tev))  # type: ignore[attr-defined]
        return {
            "model_name": "logpar",
            "params": {"phi0": float(fit["phi0"]), "alpha": float(fit["alpha"]), "beta": float(fit["beta"])},
            "pivot_tev": float(fit["pivot_tev"]),
            "source": "official_pass5_point_fit (default)",
            "pass5_fit": {k: float(v) for k, v in fit.items()},
        }
    if model == "logpar":
        if args.phi0 is None or args.alpha is None or args.beta is None:
            raise ValueError("--model logpar requires --phi0, --alpha, --beta")
        return {
            "model_name": "logpar",
            "params": {"phi0": float(args.phi0), "alpha": float(args.alpha), "beta": float(args.beta)},
            "pivot_tev": float(args.pivot_tev),
            "source": "CLI logpar",
        }
    # powerlaw -> pipeline model name "pl"
    if args.phi0 is None or args.gamma is None:
        raise ValueError("--model powerlaw requires --phi0 and --gamma")
    return {
        "model_name": "pl",
        "params": {"phi0": float(args.phi0), "gamma": float(args.gamma)},
        "pivot_tev": float(args.pivot_tev),
        "source": "CLI powerlaw",
    }


# --------------------------------------------------------------------------- #
# Forward fold + regroup                                                      #
# --------------------------------------------------------------------------- #
def build_theta_exposure(
    args: argparse.Namespace,
    n_theta: int,
    pass5_points: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, Dict[str, object]]:
    if args.theta_exposure_json:
        path = Path(args.theta_exposure_json)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        exp = payload.get("theta_exposure_sec")
        if exp is None and isinstance(payload.get("theta_exposure"), dict):
            exp = payload["theta_exposure"].get("theta_exposure_sec")
        if exp is None:
            raise ValueError(f"{path} has no 'theta_exposure_sec' array")
        exposure = np.asarray(exp, dtype=np.float64)
        if exposure.size != n_theta:
            raise ValueError(f"theta_exposure_sec length {exposure.size} != n_theta {n_theta}")
        return exposure, {"mode": "json", "source": str(path), "total_sec": float(np.sum(exposure))}
    # Flat exposure fallback: total livetime spread evenly over theta bins.
    livetimes = [p.get("livetime_days") for p in pass5_points if p.get("livetime_days")]
    total_sec = float(livetimes[0]) * 86400.0 if livetimes else DEFAULT_EXPOSURE_SEC
    exposure = np.full(n_theta, total_sec / float(n_theta), dtype=np.float64)
    return exposure, {
        "mode": "flat_fallback",
        "total_sec": total_sec,
        "note": "Flat theta exposure; only rescales absolute normalization (APPROXIMATE caveat applies).",
    }


def forward_fold_and_regroup(
    *,
    response: Dict[str, np.ndarray],
    spectrum: Dict[str, object],
    theta_exposure_sec: np.ndarray,
    pass5_points: Sequence[Dict[str, object]],
    quadrature_points: int,
    fit_mod: object,
    sed_mod: object,
) -> Dict[str, object]:
    a_eff = np.asarray(response["a_eff"], dtype=np.float64)
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    n_cells = a_eff.shape[0]
    containment = np.ones(n_cells, dtype=np.float64)
    v6_nhit = [str(v) for v in response["nhit_bin"]]
    cell_id = np.asarray(response["cell_id"]).reshape(-1)

    model_name = str(spectrum["model_name"])
    params = dict(spectrum["params"])  # type: ignore[arg-type]
    pivot_tev = float(spectrum["pivot_tev"])

    # Per-cell predicted counts via the pipeline's own model_counts().
    per_cell_counts = fit_mod.model_counts(  # type: ignore[attr-defined]
        a_eff,
        containment,
        theta_exposure_sec,
        loge_edges,
        model_name=model_name,
        params=params,
        pivot_tev=pivot_tev,
        quadrature_points=int(quadrature_points),
    )
    per_cell_counts = np.asarray(per_cell_counts, dtype=np.float64)

    # True-energy weights per cell (for effective energy of each Nhit group),
    # reusing the Stage G helpers so the median-energy definition matches.
    unit_flux_integral = sed_mod.integrate_flux_bins(  # type: ignore[attr-defined]
        loge_edges,
        model_name=model_name,
        params=params,
        pivot_tev=pivot_tev,
        quadrature_points=int(quadrature_points),
    )

    pass5_bins = [str(p["nhit_bin"]) for p in pass5_points]
    # Map each v6 cell to a Pass5 Nhit bin.
    cell_to_pass5: List[Optional[str]] = [map_v6_to_pass5_nhit(nb, pass5_bins) for nb in v6_nhit]

    # Aggregate predicted counts + energy weights per Pass5 bin.
    grouped: Dict[str, Dict[str, object]] = {}
    for b in pass5_bins:
        grouped.setdefault(
            b,
            {
                "pass5_nhit_bin": b,
                "v6_cell_ids": [],
                "v6_nhit_bins": [],
                "predicted_counts": 0.0,
                "energy_weights": np.zeros(loge_edges.size - 1, dtype=np.float64),
            },
        )
    for idx in range(n_cells):
        b = cell_to_pass5[idx]
        if b is None:
            continue
        g = grouped[b]
        g["predicted_counts"] = float(g["predicted_counts"]) + float(per_cell_counts[idx])  # type: ignore[index]
        g["v6_cell_ids"].append(int(cell_id[idx]))  # type: ignore[union-attr]
        g["v6_nhit_bins"].append(v6_nhit[idx])  # type: ignore[union-attr]
        # per-cell energy weight = containment * 1e4 * (A_eff summed over theta) * flux_integral
        contribution = M2_TO_CM2 * np.einsum("et,t->e", a_eff[idx], theta_exposure_sec) * unit_flux_integral
        g["energy_weights"] = np.asarray(g["energy_weights"]) + np.nan_to_num(contribution)  # type: ignore[index]

    # Build comparison rows against the observed Pass5 points.
    rows: List[Dict[str, object]] = []
    for p in pass5_points:
        b = str(p["nhit_bin"])
        g = grouped[b]
        weights = np.asarray(g["energy_weights"], dtype=np.float64)
        e16, e50, e84 = sed_mod.weighted_quantiles_from_loge_bins(  # type: ignore[attr-defined]
            loge_edges, weights, [0.16, 0.50, 0.84]
        )
        eff_e = e50 if (isinstance(e50, float) and math.isfinite(e50) and e50 > 0.0) else float(p["energy_tev"])
        # Model flux (dN/dE) at the effective true energy of this group.
        model_flux = float(
            sed_mod.flux_tev(eff_e, model_name=model_name, params=params, pivot_tev=pivot_tev)  # type: ignore[attr-defined]
        )
        obs_flux = float(p["flux_per_tev_cm2_s"])
        has_v6 = len(g["v6_cell_ids"]) > 0  # type: ignore[arg-type]
        ratio = (model_flux / obs_flux) if (obs_flux > 0.0 and has_v6) else None
        rows.append(
            {
                "pass5_nhit_bin": b,
                "pass5_energy_tev": float(p["energy_tev"]),
                "pass5_flux_per_tev_cm2_s": obs_flux,
                "pass5_ts": p.get("ts"),
                "pass5_is_upper_limit": bool(p.get("is_upper_limit")),
                "v6_covered": has_v6,
                "v6_cell_ids": list(g["v6_cell_ids"]),
                "v6_nhit_bins": list(dict.fromkeys(g["v6_nhit_bins"])),  # unique, order preserved
                "predicted_counts": float(g["predicted_counts"]),
                "effective_true_energy_tev": float(eff_e),
                "true_energy_p16_tev": float(e16) if isinstance(e16, float) else None,
                "true_energy_p50_tev": float(e50) if isinstance(e50, float) else None,
                "true_energy_p84_tev": float(e84) if isinstance(e84, float) else None,
                "model_flux_per_tev_cm2_s": model_flux,
                "predicted_over_pass5_ratio": ratio,
            }
        )

    covered = [r for r in rows if r["v6_covered"] and r["predicted_over_pass5_ratio"] is not None]
    ratios = [float(r["predicted_over_pass5_ratio"]) for r in covered]  # type: ignore[arg-type]
    return {
        "rows": rows,
        "n_pass5_points": len(rows),
        "n_v6_covered_points": len(covered),
        "n_uncovered_points": len(rows) - len(covered),
        "ratio_median": float(np.median(ratios)) if ratios else None,
        "ratio_min": float(np.min(ratios)) if ratios else None,
        "ratio_max": float(np.max(ratios)) if ratios else None,
        "per_cell_predicted_counts_total": float(np.sum(per_cell_counts)),
    }


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


def plot_forward_fold(path: Path, *, rows: Sequence[Dict[str, object]], spectrum: Dict[str, object]) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True, constrained_layout=True)

    ax = axes[0]
    obs_e = [float(r["pass5_energy_tev"]) for r in rows]
    obs_e2f = [float(r["pass5_energy_tev"]) ** 2 * float(r["pass5_flux_per_tev_cm2_s"]) for r in rows]
    ax.plot(obs_e, obs_e2f, "ks", ms=6, label="Pass5 observed (E^2 dN/dE)")
    cov = [r for r in rows if r["v6_covered"]]
    if cov:
        mod_e = [float(r["effective_true_energy_tev"]) for r in cov]
        mod_e2f = [
            float(r["effective_true_energy_tev"]) ** 2 * float(r["model_flux_per_tev_cm2_s"]) for r in cov
        ]
        ax.plot(mod_e, mod_e2f, "ro", ms=5, label="v6 forward-fold model (E^2 dN/dE)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    ax.set_title("APPROXIMATE: v6 forward-fold vs Pass5 (Nhit 30-2000, different cuts)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    if cov:
        rat_e = [float(r["pass5_energy_tev"]) for r in cov]
        rat = [float(r["predicted_over_pass5_ratio"]) for r in cov]
        ax2.plot(rat_e, rat, "ro-", ms=5, label="predicted / Pass5")
    ax2.axhline(1.0, color="#888888", ls="--", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xlabel("Pass5 energy [TeV]")
    ax2.set_ylabel("predicted / Pass5")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Synthetic data for --self-test                                              #
# --------------------------------------------------------------------------- #
def synthetic_response_and_pass5(tmp_dir: Path) -> Tuple[Dict[str, np.ndarray], Path]:
    loge_edges = np.arange(2.0, 6.01, 0.5)
    theta_edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    n_logE = loge_edges.size - 1
    n_theta = theta_edges.size - 1
    loge_centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    cos_theta = np.cos(np.radians(theta_centers))

    v6_bins = ["[100,200)", "[200,300)", "[300,500)", "[500,800)", "[800,1100)", "[1100,2000)", "[2000,3000)"]
    turn_on = {"[100,200)": 2.8, "[200,300)": 3.1, "[300,500)": 3.4, "[500,800)": 3.8,
               "[800,1100)": 4.1, "[1100,2000)": 4.4, "[2000,3000)": 4.7}
    eta_cap = {"[100,200)": 0.5, "[200,300)": 0.6, "[300,500)": 0.7, "[500,800)": 0.8,
               "[800,1100)": 0.85, "[1100,2000)": 0.9, "[2000,3000)": 0.92}
    n_cells = len(v6_bins)
    eta = np.zeros((n_cells, n_logE, n_theta), dtype=np.float64)
    for c, nb in enumerate(v6_bins):
        curve = eta_cap[nb] / (1.0 + np.exp(-(loge_centers - turn_on[nb]) / 0.35))
        for t in range(n_theta):
            eta[c, :, t] = curve
    a_eff = 4.0e6 * eta * cos_theta[None, None, :]

    response = {
        "a_eff": a_eff.astype(np.float32),
        "eta": eta.astype(np.float32),
        "logE_true_edges": loge_edges,
        "theta_true_edges_deg": theta_edges,
        "cell_id": np.asarray([10 + c for c in range(n_cells)], dtype=np.int32),
        "nhit_bin": np.asarray(v6_bins, dtype="U32"),
        "predE_bin": np.asarray([f"[{2.0 + 0.4 * c:.1f},{2.4 + 0.4 * c:.1f})" for c in range(n_cells)], dtype="U32"),
        "s0_m2": np.asarray(4.0e6, dtype=np.float32),
    }

    # Synthetic Pass5 CSV with the real 7-bin Nhit schema.
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "synthetic_pass5.csv"
    header = "energy_tev,flux_per_tev_cm2_s,ts,nhit_bin,error_status,is_upper_limit,stderr_empty,livetime_days,source_note"
    rows = [
        "0.56,1.399e-10,12297.1,30-60,3,false,true,1484.36,synthetic",
        "0.89,5.094e-11,38689.6,60-100,3,false,true,1484.36,synthetic",
        "1.41,1.648e-11,108612.8,100-200,3,false,true,1484.36,synthetic",
        "2.51,3.671e-12,77285.8,200-300,3,false,true,1484.36,synthetic",
        "3.98,1.033e-12,68602.6,300-500,3,false,true,1484.36,synthetic",
        "7.94,1.471e-13,28980.6,500-800,3,false,true,1484.36,synthetic",
        "15.85,1.944e-14,11297.9,>800,3,false,true,1484.36,synthetic",
    ]
    csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return response, csv_path


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
APPROXIMATE_CAVEAT = (
    "APPROXIMATE cross-check only. Pass5 uses Nhit ~30-2000 with different cuts, "
    "PSF/containment, exposure and background than v6. v6's response covers "
    "Nhit>=100 only; Pass5 bins 30-60 and 60-100 have no v6 coverage. Read the "
    "ratio as a shape/normalization ballpark check, not a calibrated agreement."
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fit_mod, sed_mod = load_stage_helpers()

    if args.self_test:
        tmp_dir = Path("/tmp/pass5_nhit_forward_fold_selftest")
        response, pass5_csv = synthetic_response_and_pass5(tmp_dir)
        response_npz_used = "<synthetic>"
    else:
        npz_path = Path(args.response_npz)
        if not npz_path.is_absolute():
            npz_path = (REPO_ROOT / npz_path).resolve()
        response = load_response(npz_path)
        pass5_csv = Path(args.pass5_csv)
        if not pass5_csv.is_absolute():
            pass5_csv = (REPO_ROOT / pass5_csv).resolve()
        response_npz_used = str(npz_path)

    pass5_points = load_pass5_points(pass5_csv)
    spectrum = resolve_spectrum(args, sed_mod)

    n_theta = int(np.asarray(response["a_eff"]).shape[2])
    theta_exposure_sec, exposure_meta = build_theta_exposure(args, n_theta, pass5_points)

    fold = forward_fold_and_regroup(
        response=response,
        spectrum=spectrum,
        theta_exposure_sec=theta_exposure_sec,
        pass5_points=pass5_points,
        quadrature_points=int(args.quadrature_points),
        fit_mod=fit_mod,
        sed_mod=sed_mod,
    )

    # Document the v6 -> Pass5 mapping used.
    pass5_bins = [str(p["nhit_bin"]) for p in pass5_points]
    v6_nhit = [str(v) for v in response["nhit_bin"]]
    nhit_mapping = {nb: map_v6_to_pass5_nhit(nb, pass5_bins) for nb in dict.fromkeys(v6_nhit)}

    result = {
        "tool": "pass5_nhit_forward_fold",
        "check": "2C",
        "caveat": APPROXIMATE_CAVEAT,
        "response_npz": response_npz_used,
        "pass5_csv": str(pass5_csv),
        "spectrum": {
            "model_name": spectrum["model_name"],
            "params": spectrum["params"],
            "pivot_tev": spectrum["pivot_tev"],
            "source": spectrum["source"],
            **({"pass5_fit": spectrum["pass5_fit"]} if "pass5_fit" in spectrum else {}),
        },
        "quadrature_points": int(args.quadrature_points),
        "theta_exposure": exposure_meta,
        "v6_nhit_to_pass5_nhit_mapping": nhit_mapping,
        "pass5_nhit_bins": pass5_bins,
        "summary": {
            "n_pass5_points": fold["n_pass5_points"],
            "n_v6_covered_points": fold["n_v6_covered_points"],
            "n_uncovered_points": fold["n_uncovered_points"],
            "ratio_median": fold["ratio_median"],
            "ratio_min": fold["ratio_min"],
            "ratio_max": fold["ratio_max"],
            "per_cell_predicted_counts_total": fold["per_cell_predicted_counts_total"],
        },
        "points": fold["rows"],
    }

    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = (REPO_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    out_png = Path(args.output_png)
    if not out_png.is_absolute():
        out_png = (REPO_ROOT / out_png).resolve()
    plot_forward_fold(out_png, rows=fold["rows"], spectrum=spectrum)

    print(f"[pass5_nhit_forward_fold] wrote {out_json}")
    print(f"[pass5_nhit_forward_fold] wrote {out_png}")
    print(
        f"[pass5_nhit_forward_fold] covered {fold['n_v6_covered_points']}/{fold['n_pass5_points']} "
        f"Pass5 points; ratio median={fold['ratio_median']}"
    )
    print(f"[pass5_nhit_forward_fold] NOTE: {APPROXIMATE_CAVEAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
