#!/usr/bin/env python
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_HTML = REPO_ROOT / "apply/report/crab_sed_v5_psf_comparison_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v5-psf-comparison"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"

METHODS: Dict[str, Dict[str, str]] = {
    "rayleigh_baseline": {
        "label": "Rayleigh baseline",
        "run_id": "v5_psf_rayleigh_baseline_drop4",
        "role": "v4-compatible core Rayleigh r_opt = 1.58 sigma",
        "color": "#111827",
        "marker": "o",
    },
    "two_1d_gaussian": {
        "label": "Two 1D Gaussian",
        "run_id": "v5_psf_two_1d_gaussian_drop4",
        "role": "signed x/y tangent-plane Gaussian with circular containment radius",
        "color": "#2563eb",
        "marker": "s",
    },
    "mc_quantile_715": {
        "label": "MC quantile",
        "run_id": "v5_psf_mc_quantile_715_drop4",
        "role": "Crab-theta-reweighted empirical mc_dangle quantile",
        "color": "#dc2626",
        "marker": "^",
    },
    "observed_data": {
        "label": "Observed data",
        "run_id": "v5_psf_observed_data_drop4",
        "role": "data-driven observed-Crab radial-profile aperture after annnorm-background and pedestal subtraction",
        "color": "#059669",
        "marker": "D",
    },
    "double_rayleigh_mixture": {
        "label": "Double Rayleigh mixture",
        "run_id": "v5_psf_double_rayleigh_mixture_drop4",
        "role": "two-component circular 2D-Gaussian / double-Rayleigh radial mixture aperture",
        "color": "#7e22ce",
        "marker": "P",
    },
}

STAGE_B_RUNS = REPO_ROOT / "apply/output/stage_b_v5_psf_compare/runs"
STAGE_A_DIR = REPO_ROOT / "apply/output/stage_a_v5_psf_compare"
STAGE_D_RUNS = REPO_ROOT / "apply/output/stage_d_v5_psf_compare/runs"
STAGE_E_RUNS = REPO_ROOT / "apply/output/stage_e_v5_psf_compare/runs"
STAGE_F_RUNS = REPO_ROOT / "apply/output/stage_f_v5_psf_compare/runs"
STAGE_G_RUNS = REPO_ROOT / "apply/output/stage_g_v5_psf_compare/runs"

V4_STAGE_B_NPZ = REPO_ROOT / "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate.npz"
V4_STAGE_F_NPZ = REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz"
V4_STAGE_F_META = REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_metadata.json"
V4_STAGE_G_SUMMARY = REPO_ROOT / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4/sed_points_v4_aperture_conditioned_drop4_summary.csv"
V4_STAGE_G_META = REPO_ROOT / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4/sed_points_v4_aperture_conditioned_drop4_metadata.json"
V4_EMPIRICAL_PSF_SUMMARY = REPO_ROOT / "apply/report/assets/v4-empirical-psf/empirical_psf_cell_summary.csv"
V4_EMPIRICAL_PSF_PROFILES = REPO_ROOT / "apply/report/assets/v4-empirical-psf/empirical_psf_profiles.npz"

M2_TO_CM2 = 1.0e4
TARGET_CONTAINMENT = 1.0 - math.exp(-0.5 * 1.58 * 1.58)
FOCUS_CELLS = (15, 27, 43, 55, 65)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def finite_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def rel(path: Path) -> str:
    return os.path.relpath(path, REPORT_HTML.parent).replace(os.sep, "/")


def fmt(value: object, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def fmt_int(value: object) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number:,.0f}"


def interval_key(label: object) -> float:
    text = str(label or "").strip()
    if text.startswith("[") and "," in text:
        try:
            return float(text[1:].split(",", 1)[0])
        except ValueError:
            return 1.0e30
    if text.startswith(">="):
        try:
            return float(text[2:])
        except ValueError:
            return 1.0e30
    return 1.0e30


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fit_params_from_meta(meta: Dict[str, object], model: str, error_mode: str = "conservative") -> Dict[str, object]:
    fits = meta.get("fits") if isinstance(meta.get("fits"), dict) else {}
    direct = fits.get(f"{model}_{error_mode}") if isinstance(fits, dict) else None
    if isinstance(direct, dict):
        return direct
    if isinstance(fits, dict):
        for value in fits.values():
            if not isinstance(value, dict):
                continue
            if str(value.get("model_name")).lower() == model and str(value.get("error_mode")).lower() == error_mode:
                return value
    return {}


def fit_parameters(meta: Dict[str, object], model: str) -> Dict[str, float]:
    fit = fit_params_from_meta(meta, model)
    params = fit.get("parameters") if isinstance(fit.get("parameters"), dict) else {}
    return {str(key): float(value) for key, value in params.items() if finite_float(value) is not None}


def chi2_over_ndof(fit: Dict[str, object]) -> Optional[float]:
    value = finite_float(fit.get("chi2_over_ndof"))
    if value is not None:
        return value
    chi2 = finite_float(fit.get("chi2"))
    ndof = finite_float(fit.get("ndof"))
    if chi2 is None or ndof is None or ndof <= 0:
        return None
    return chi2 / ndof


def max_abs_pull(fit_npz: Dict[str, np.ndarray], key: str = "logpar_conservative_pull") -> Optional[float]:
    if key not in fit_npz:
        return None
    values = np.asarray(fit_npz[key], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.nanmax(np.abs(finite))) if finite.size else None


def pl_flux(energy_tev: np.ndarray, params: Dict[str, float], pivot_tev: float = 3.0) -> np.ndarray:
    return params["phi0"] * np.power(np.asarray(energy_tev, dtype=np.float64) / pivot_tev, -params["gamma"])


def logpar_flux(energy_tev: np.ndarray, params: Dict[str, float], pivot_tev: float = 3.0) -> np.ndarray:
    ratio = np.asarray(energy_tev, dtype=np.float64) / pivot_tev
    log_ratio = np.log(ratio)
    return params["phi0"] * np.exp((-params["alpha"] - params["beta"] * log_ratio) * log_ratio)


def e2_curve(energy_tev: np.ndarray, params: Dict[str, float], model: str) -> np.ndarray:
    flux = logpar_flux(energy_tev, params) if model == "logpar" else pl_flux(energy_tev, params)
    return np.square(energy_tev) * flux


def pass5_dnde_points() -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(PASS5_CSV)
    energy = []
    flux = []
    for row in rows:
        e = finite_float(row.get("energy_tev"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        if e is not None and dnde is not None and e > 0.0 and dnde > 0.0:
            energy.append(e)
            flux.append(dnde)
    return np.asarray(energy, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def pass5_e2_points() -> Tuple[np.ndarray, np.ndarray]:
    e, dnde = pass5_dnde_points()
    return e, e * e * dnde


def loglog_powerlaw_interp(energy_tev: np.ndarray, knots_energy: np.ndarray, knots_dnde: np.ndarray) -> np.ndarray:
    x = np.log(np.asarray(energy_tev, dtype=np.float64))
    xk = np.log(np.asarray(knots_energy, dtype=np.float64))
    yk = np.log(np.asarray(knots_dnde, dtype=np.float64))
    slopes = np.diff(yk) / np.diff(xk)
    y = np.interp(x, xk, yk)
    lo = x < xk[0]
    hi = x > xk[-1]
    if np.any(lo):
        y[lo] = yk[0] + slopes[0] * (x[lo] - xk[0])
    if np.any(hi):
        y[hi] = yk[-1] + slopes[-1] * (x[hi] - xk[-1])
    return np.exp(y)


def integrate_spectrum_bins(
    loge_edges: np.ndarray,
    spectrum_energy: np.ndarray,
    spectrum_dnde: np.ndarray,
    quadrature_points: int = 96,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    out = np.zeros(loge_edges.size - 1, dtype=np.float64)
    for idx, (lo, hi) in enumerate(zip(loge_edges[:-1], loge_edges[1:])):
        xs = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        energy_tev = np.power(10.0, xs) / 1000.0
        flux = loglog_powerlaw_interp(energy_tev, spectrum_energy, spectrum_dnde)
        integrand = flux * math.log(10.0) * energy_tev
        out[idx] = 0.5 * (hi - lo) * float(np.sum(weights * integrand))
    return out


def official_expected_counts(response: Dict[str, np.ndarray], stage_f: Dict[str, np.ndarray]) -> np.ndarray:
    pass5_energy, pass5_dnde = pass5_dnde_points()
    flux_integral = integrate_spectrum_bins(
        np.asarray(response["logE_true_edges"], dtype=np.float64),
        pass5_energy,
        pass5_dnde,
        96,
    )
    response_cell_id = np.asarray(response["cell_id"], dtype=np.int64)
    fit_cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)
    response_index = {int(cid): idx for idx, cid in enumerate(response_cell_id)}
    missing = [int(cid) for cid in fit_cell_id if int(cid) not in response_index]
    if missing:
        raise ValueError(f"Response missing Stage F cells: {missing}")
    index = np.asarray([response_index[int(cid)] for cid in fit_cell_id], dtype=np.int64)
    theta_exposure = np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64)
    return M2_TO_CM2 * np.einsum(
        "bet,e,t->b",
        np.asarray(response["a_eff"], dtype=np.float64)[index],
        flux_integral,
        theta_exposure,
    )


def sum_rows_for_mask(
    method: str,
    mask: np.ndarray,
    stage_f: Dict[str, np.ndarray],
    official_expected: np.ndarray,
    nhit_label: str,
) -> Dict[str, object]:
    n_on = np.asarray(stage_f["N_on"], dtype=np.float64)
    b_on = np.asarray(stage_f["B_on"], dtype=np.float64)
    excess = np.asarray(stage_f["excess"], dtype=np.float64)
    expected = np.asarray(official_expected, dtype=np.float64)
    total_excess = float(np.nansum(excess[mask]))
    total_expected = float(np.nansum(expected[mask]))
    return {
        "method": method,
        "spectrum": "official_pass5",
        "nhit_bin": nhit_label,
        "cells": int(np.count_nonzero(mask)),
        "N_on": float(np.nansum(n_on[mask])),
        "B_on": float(np.nansum(b_on[mask])),
        "excess": total_excess,
        "official_expected_counts": total_expected,
        "observed_over_expected": total_excess / total_expected if total_expected > 0.0 else "",
        "N_on_over_B_on": float(np.nansum(n_on[mask])) / float(np.nansum(b_on[mask]))
        if float(np.nansum(b_on[mask])) > 0.0
        else "",
    }


def build_forward_fold_tables(runs: Dict[str, Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    nhit_rows: List[Dict[str, object]] = []
    cell_rows: List[Dict[str, object]] = []
    for method, payload in runs.items():
        stage_f = payload.get("fit_npz")
        response = payload.get("response_npz")
        if not stage_f or not response:
            continue
        expected = official_expected_counts(response, stage_f)  # type: ignore[arg-type]
        cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)  # type: ignore[index]
        nhit = np.asarray(stage_f["nhit_bin"]).astype(str)  # type: ignore[index]
        pred = np.asarray(stage_f["predE_bin"]).astype(str)  # type: ignore[index]
        excess = np.asarray(stage_f["excess"], dtype=np.float64)  # type: ignore[index]
        err = np.asarray(stage_f["excess_err_conservative"], dtype=np.float64)  # type: ignore[index]
        mask = np.ones(cell_id.shape, dtype=bool)
        summary_rows.append(sum_rows_for_mask(method, mask, stage_f, expected, "all"))  # type: ignore[arg-type]
        for label in sorted(set(nhit.tolist()), key=interval_key):
            label_mask = mask & (nhit == label)
            if np.any(label_mask):
                nhit_rows.append(sum_rows_for_mask(method, label_mask, stage_f, expected, label))  # type: ignore[arg-type]
        for idx, cid in enumerate(cell_id):
            pull = (excess[idx] - expected[idx]) / err[idx] if err[idx] > 0.0 else float("nan")
            cell_rows.append(
                {
                    "method": method,
                    "spectrum": "official_pass5",
                    "cell_id": int(cid),
                    "nhit_bin": str(nhit[idx]),
                    "predE_bin": str(pred[idx]),
                    "excess": float(excess[idx]),
                    "official_expected_counts": float(expected[idx]),
                    "observed_over_expected": float(excess[idx] / expected[idx]) if expected[idx] > 0.0 else "",
                    "excess_minus_expected": float(excess[idx] - expected[idx]),
                    "pull_conservative": float(pull) if math.isfinite(pull) else "",
                }
            )

    write_csv(
        ASSET_DIR / "official_pass5_forward_fold_summary.csv",
        summary_rows,
        ["method", "spectrum", "nhit_bin", "cells", "N_on", "B_on", "excess", "official_expected_counts", "observed_over_expected", "N_on_over_B_on"],
    )
    write_csv(
        ASSET_DIR / "official_pass5_forward_fold_nhit_summary.csv",
        nhit_rows,
        ["method", "spectrum", "nhit_bin", "cells", "N_on", "B_on", "excess", "official_expected_counts", "observed_over_expected", "N_on_over_B_on"],
    )
    write_csv(
        ASSET_DIR / "official_pass5_forward_fold_cell_counts.csv",
        cell_rows,
        ["method", "spectrum", "cell_id", "nhit_bin", "predE_bin", "excess", "official_expected_counts", "observed_over_expected", "excess_minus_expected", "pull_conservative"],
    )
    return summary_rows, nhit_rows, cell_rows


def run_paths(run_id: str) -> Dict[str, Path]:
    return {
        "psf_npz": STAGE_B_RUNS / run_id / f"psf_{run_id}.npz",
        "psf_meta": STAGE_B_RUNS / run_id / f"psf_{run_id}_metadata.json",
        "response_npz": STAGE_A_DIR / method_from_run_id(run_id) / f"response_2d_{run_id}.npz",
        "response_meta": STAGE_A_DIR / method_from_run_id(run_id) / f"response_2d_{run_id}_metadata.json",
        "stage_d_npz": STAGE_D_RUNS / run_id / f"background_{run_id}.npz",
        "stage_d_meta": STAGE_D_RUNS / run_id / f"background_{run_id}_metadata.json",
        "signal_npz": STAGE_E_RUNS / run_id / f"signal_{run_id}.npz",
        "signal_meta": STAGE_E_RUNS / run_id / f"signal_{run_id}_metadata.json",
        "fit_npz": STAGE_F_RUNS / run_id / f"fit_{run_id}.npz",
        "fit_meta": STAGE_F_RUNS / run_id / f"fit_{run_id}_metadata.json",
        "stage_g_summary": STAGE_G_RUNS / run_id / f"sed_points_{run_id}_summary.csv",
        "stage_g_meta": STAGE_G_RUNS / run_id / f"sed_points_{run_id}_metadata.json",
    }


def method_from_run_id(run_id: str) -> str:
    prefix = "v5_psf_"
    suffix = "_drop4"
    if run_id.startswith(prefix) and run_id.endswith(suffix):
        return run_id[len(prefix) : -len(suffix)]
    return run_id


def load_runs() -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for method, config in METHODS.items():
        run_id = config["run_id"]
        paths = run_paths(run_id)
        fit_meta = read_json(paths["fit_meta"])
        fit_npz = load_npz(paths["fit_npz"])
        runs[method] = {
            "method": method,
            "label": config["label"],
            "role": config["role"],
            "run_id": run_id,
            "paths": paths,
            "psf_npz": load_npz(paths["psf_npz"]),
            "psf_meta": read_json(paths["psf_meta"]),
            "response_npz": load_npz(paths["response_npz"]),
            "response_meta": read_json(paths["response_meta"]),
            "stage_d_npz": load_npz(paths["stage_d_npz"]),
            "stage_d_meta": read_json(paths["stage_d_meta"]),
            "signal_npz": load_npz(paths["signal_npz"]),
            "signal_meta": read_json(paths["signal_meta"]),
            "fit_npz": fit_npz,
            "fit_meta": fit_meta,
            "stage_g_rows": read_csv_rows(paths["stage_g_summary"]),
            "stage_g_meta": read_json(paths["stage_g_meta"]),
            "logpar": fit_params_from_meta(fit_meta, "logpar"),
            "pl": fit_params_from_meta(fit_meta, "pl"),
            "max_abs_pull": max_abs_pull(fit_npz),
            "status": "complete" if paths["fit_meta"].exists() and paths["stage_g_summary"].exists() else "pending",
        }
    return runs


def group_stage_g_rows(rows: Sequence[Dict[str, str]], grouping: str) -> List[Dict[str, str]]:
    return sorted([row for row in rows if row.get("grouping") == grouping], key=lambda row: interval_key(row.get("group_label")))


def cell_grid_labels(psf_npz: Dict[str, np.ndarray]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], int]]:
    nhit = np.asarray(psf_npz.get("nhit_bin", []), dtype=str)
    pred = np.asarray(psf_npz.get("predE_bin", []), dtype=str)
    cid = np.asarray(psf_npz.get("cell_id", []), dtype=np.int64)
    nhit_labels = sorted(set(nhit.tolist()), key=interval_key)
    pred_labels = sorted(set(pred.tolist()), key=interval_key)
    by_pair = {(str(n), str(p)): int(c) for n, p, c in zip(nhit, pred, cid)}
    return nhit_labels, pred_labels, by_pair


def plot_r_opt_ratio_heatmap(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    base = runs["rayleigh_baseline"]["psf_npz"]  # type: ignore[assignment]
    nhit_labels, pred_labels, by_pair = cell_grid_labels(base)
    base_r = {
        int(cid): float(r)
        for cid, r in zip(np.asarray(base["cell_id"], dtype=np.int64), np.asarray(base["r_opt_deg"], dtype=np.float64))
    }
    n_methods = len(METHODS)
    fig = plt.figure(figsize=(4.8 * n_methods + 1.0, 5.2), dpi=160, constrained_layout=True)
    grid = fig.add_gridspec(1, n_methods + 1, width_ratios=[1.0] * n_methods + [0.045], wspace=0.08)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(n_methods)]
    cax = fig.add_subplot(grid[0, n_methods])
    for ax, method in zip(axes, METHODS):
        psf = runs[method]["psf_npz"]  # type: ignore[assignment]
        method_r = {
            int(cid): float(r)
            for cid, r in zip(np.asarray(psf.get("cell_id", []), dtype=np.int64), np.asarray(psf.get("r_opt_deg", []), dtype=np.float64))
        }
        matrix = np.full((len(nhit_labels), len(pred_labels)), np.nan, dtype=np.float64)
        for i, nhit in enumerate(nhit_labels):
            for j, pred in enumerate(pred_labels):
                cid = by_pair.get((nhit, pred))
                if cid is not None and cid in method_r and cid in base_r and base_r[cid] > 0.0:
                    matrix[i, j] = method_r[cid] / base_r[cid]
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0.65, vmax=1.35)
        ax.set_title(METHODS[method]["label"])
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(nhit_labels)))
        ax.set_yticklabels(nhit_labels, fontsize=8)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#111827")
        ax.set_xlabel("predicted-energy bin")
    axes[0].set_ylabel("Nhit bin")
    fig.colorbar(im, cax=cax, label="r_opt / Rayleigh baseline")
    fig.suptitle("v5 PSF aperture radius ratio heatmap")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_observed_data_radius_comparison(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    observed = runs.get("observed_data", {}).get("psf_npz")
    rayleigh = runs.get("rayleigh_baseline", {}).get("psf_npz")
    mcq = runs.get("mc_quantile_715", {}).get("psf_npz")
    if not isinstance(observed, dict) or not observed or not isinstance(rayleigh, dict) or not rayleigh:
        return
    nhit_labels, pred_labels, by_pair = cell_grid_labels(rayleigh)
    obs_r = {
        int(cid): float(r)
        for cid, r in zip(np.asarray(observed.get("cell_id", []), dtype=np.int64), np.asarray(observed.get("r_opt_deg", []), dtype=np.float64))
    }
    obs_fallback = {
        int(cid): bool(flag)
        for cid, flag in zip(
            np.asarray(observed.get("cell_id", []), dtype=np.int64),
            np.asarray(observed.get("observed_data_fallback", np.zeros(len(obs_r), dtype=bool)), dtype=bool),
        )
    }
    ray_r = {
        int(cid): float(r)
        for cid, r in zip(np.asarray(rayleigh.get("cell_id", []), dtype=np.int64), np.asarray(rayleigh.get("r_opt_deg", []), dtype=np.float64))
    }
    mc_r = {}
    if isinstance(mcq, dict) and mcq:
        mc_r = {
            int(cid): float(r)
            for cid, r in zip(np.asarray(mcq.get("cell_id", []), dtype=np.int64), np.asarray(mcq.get("r_opt_deg", []), dtype=np.float64))
        }
    panels = [
        ("observed_data / rayleigh", ray_r),
        ("observed_data / MC quantile", mc_r),
    ]
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(13.8, 5.2), dpi=160, constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.08)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    last_im = None
    for ax, (title, denominator) in zip(axes, panels):
        matrix = np.full((len(nhit_labels), len(pred_labels)), np.nan, dtype=np.float64)
        fallback_matrix = np.zeros_like(matrix, dtype=bool)
        for i, nhit in enumerate(nhit_labels):
            for j, pred in enumerate(pred_labels):
                cid = by_pair.get((nhit, pred))
                if cid is not None and cid in obs_r and cid in denominator and denominator[cid] > 0.0:
                    matrix[i, j] = obs_r[cid] / denominator[cid]
                    fallback_matrix[i, j] = obs_fallback.get(cid, False)
        last_im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0.5, vmax=2.5)
        ax.set_title(title)
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(nhit_labels)))
        ax.set_yticklabels(nhit_labels, fontsize=8)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    suffix = "*" if fallback_matrix[i, j] else ""
                    ax.text(j, i, f"{matrix[i, j]:.2f}{suffix}", ha="center", va="center", fontsize=6.5, color="#111827")
        ax.set_xlabel("predicted-energy bin")
    axes[0].set_ylabel("Nhit bin")
    if last_im is not None:
        fig.colorbar(last_im, cax=cax, label="radius ratio")
    fig.suptitle("Observed-data aperture radius ratios (* = fallback cell)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_shape_scatter(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
    for method, config in METHODS.items():
        psf = runs[method]["psf_npz"]  # type: ignore[assignment]
        if not psf:
            continue
        x = np.asarray(psf.get("sigma_eff_deg", psf.get("sigma_deg", [])), dtype=np.float64)
        y = np.asarray(psf.get("r_opt_deg", []), dtype=np.float64)
        ax.scatter(x, y, s=28, alpha=0.78, color=config["color"], marker=config["marker"], label=config["label"])
    ax.set_xlabel("sigma_eff or Rayleigh sigma (deg)")
    ax.set_ylabel("r_opt (deg)")
    ax.set_title("PSF radius versus shape scale")
    ax.grid(alpha=0.24)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def selected_fit_cell_ids(runs: Dict[str, Dict[str, object]]) -> List[int]:
    for payload in runs.values():
        fit_npz = payload.get("fit_npz")
        if isinstance(fit_npz, dict) and "cell_id" in fit_npz:
            return [int(value) for value in np.asarray(fit_npz["cell_id"], dtype=np.int64)]
    return []


def fit_cell_ids(payload: Dict[str, object]) -> set[int]:
    fit_npz = payload.get("fit_npz")
    if isinstance(fit_npz, dict) and "cell_id" in fit_npz:
        return {int(value) for value in np.asarray(fit_npz["cell_id"], dtype=np.int64)}
    return set()


def plot_weighted_psf_profile_overlay(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    cell_ids = selected_fit_cell_ids(runs)
    if not cell_ids:
        return

    ncols = 4
    nrows = int(math.ceil(len(cell_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, max(3.0, 2.45 * nrows)), dpi=160, sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, cell_id in zip(axes_arr, cell_ids):
        ax.set_facecolor("#ecfdf5")
        for spine in ax.spines.values():
            spine.set_color("#059669")
            spine.set_linewidth(1.25)
        ax.text(
            0.97,
            0.94,
            "fit",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color="#047857",
            fontweight="bold",
        )
        title_bits = [str(cell_id)]
        for method, config in METHODS.items():
            psf = runs[method]["psf_npz"]  # type: ignore[assignment]
            if not psf or "profile_density" not in psf or "profile_edges_deg" not in psf:
                continue
            psf_cell_id = np.asarray(psf.get("cell_id", []), dtype=np.int64)
            match = np.nonzero(psf_cell_id == int(cell_id))[0]
            if match.size == 0:
                continue
            idx = int(match[0])
            edges = np.asarray(psf["profile_edges_deg"], dtype=np.float64)
            centers = 0.5 * (edges[:-1] + edges[1:])
            profile = np.asarray(psf["profile_density"], dtype=np.float64)[idx]
            finite = np.isfinite(profile)
            if not np.any(finite):
                continue
            peak = float(np.nanmax(profile[finite]))
            y = profile / peak if peak > 0.0 else profile
            ax.plot(centers, y, color=config["color"], lw=1.35, alpha=0.9, label=config["label"])
            r_opt = finite_float(np.asarray(psf.get("r_opt_deg", []), dtype=np.float64)[idx])
            if r_opt is not None:
                ax.axvline(r_opt, color=config["color"], lw=0.85, ls="--", alpha=0.65)
            if method == "rayleigh_baseline":
                nhit = str(np.asarray(psf.get("nhit_bin", []), dtype=str)[idx])
                pred = str(np.asarray(psf.get("predE_bin", []), dtype=str)[idx])
                title_bits = [str(cell_id), nhit, pred]
        ax.set_title(" ".join(title_bits), fontsize=8.5)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.2)

    for ax in axes_arr[len(cell_ids):]:
        ax.axis("off")
    for ax in axes_arr[-ncols:]:
        ax.set_xlabel("dangle [deg]")
    for row in range(nrows):
        axes_arr[row * ncols].set_ylabel("weighted profile / peak")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(METHODS), 4), frameon=False, bbox_to_anchor=(0.5, 0.998))
    fig.suptitle("Weighted/display PSF radial profiles for drop4 fit cells", y=1.012)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def rayleigh_pdf_deg(r_deg: np.ndarray, sigma_rad: float) -> np.ndarray:
    r_rad = np.radians(r_deg)
    pdf_per_rad = (r_rad / (sigma_rad * sigma_rad)) * np.exp(-0.5 * (r_rad / sigma_rad) ** 2)
    return pdf_per_rad * (math.pi / 180.0)


def double_rayleigh_pdf_deg(r_deg: np.ndarray, a_core: float, sigma1_deg: float, sigma2_deg: float) -> np.ndarray:
    r = np.asarray(r_deg, dtype=np.float64)
    a = float(a_core)
    s1 = float(sigma1_deg)
    s2 = float(sigma2_deg)
    if not (0.0 < a < 1.0 and 0.0 < s1 < s2):
        return np.full(r.shape, np.nan, dtype=np.float64)
    r_pos = np.clip(r, 0.0, None)
    return (
        a * r_pos / (s1 * s1) * np.exp(-0.5 * (r_pos / s1) ** 2)
        + (1.0 - a) * r_pos / (s2 * s2) * np.exp(-0.5 * (r_pos / s2) ** 2)
    )


def two1d_radial_pdf_deg(
    r_deg: np.ndarray,
    *,
    mu_x_deg: float,
    mu_y_deg: float,
    sigma_x_deg: float,
    sigma_y_deg: float,
    angle_samples: int = 256,
) -> np.ndarray:
    radius = np.asarray(r_deg, dtype=np.float64)
    sx = max(float(sigma_x_deg), 1.0e-9)
    sy = max(float(sigma_y_deg), 1.0e-9)
    phi = np.linspace(0.0, 2.0 * math.pi, max(64, int(angle_samples)), endpoint=False, dtype=np.float64)
    x = radius[:, None] * np.cos(phi)[None, :]
    y = radius[:, None] * np.sin(phi)[None, :]
    exponent = -0.5 * (((x - float(mu_x_deg)) / sx) ** 2 + ((y - float(mu_y_deg)) / sy) ** 2)
    angular_integral = np.mean(np.exp(exponent), axis=1) * (2.0 * math.pi)
    return radius * angular_integral / (2.0 * math.pi * sx * sy)


def profile_grid_caption(method: str) -> str:
    common = (
        "Blue steps are the same Rayleigh-baseline Stage B Crab-theta-weighted MC radial profile in all five grids. "
        "Each cell uses the same Rayleigh-baseline 0-5 deg x range and Rayleigh-reference y scale, so model overlays cannot rescale the MC profile. "
    )
    captions = {
        "rayleigh_baseline": (
            common + "The orange curve is the Rayleigh radial PDF fit/reference from the same cell's Rayleigh sigma. "
            "The grey dashed line is Rayleigh r_opt = 1.58 sigma. "
            "Green shaded panels are cells included in the final SED fit."
        ),
        "two_1d_gaussian": (
            common + "The purple curve is the radial PDF induced by the fitted independent x/y Gaussian model; "
            "this is not Rayleigh except in the circular sigma_x = sigma_y limit. "
            "The grey dashed line is the two-1D circular containment radius solved from the fitted x/y Gaussian sigmas. "
            "Green shaded panels are cells included in the final SED fit."
        ),
        "mc_quantile_715": (
            common + "No parametric fit curve is drawn because this branch defines the aperture directly from the weighted empirical quantile. "
            "The grey dashed line is the empirical Crab-theta-reweighted mc_dangle quantile radius. "
            "Green shaded panels are cells included in the final SED fit."
        ),
        "observed_data": (
            common + "Teal steps additionally show the pedestal-subtracted observed Crab excess radial profile for cells that pass the data-PSF quality gates; "
            "they are omitted for fallback cells. "
            "The orange curve is the Rayleigh radial PDF reference from the same cell's Rayleigh sigma; it is shown only as a comparison curve, "
            "not as the fit used to define the observed-data aperture. "
            "The grey dashed line is the observed-Crab radial-profile aperture after annnorm-background and flat-pedestal subtraction, "
            "or the documented fallback radius when the observed profile fails quality gates. "
            "Green shaded panels are cells included in the final SED fit."
        ),
        "double_rayleigh_mixture": (
            common + "The purple curve is the fitted two-component Rayleigh radial PDF; "
            "the grey dashed line is r_opt from the fitted mixture CDF at the Rayleigh-contract target containment. "
            "Fallback or psfborrow cells are listed in the diagnostics tables. "
            "Green shaded panels are cells included in the final SED fit."
        ),
    }
    return captions[method]


def plot_fit_shaded_psf_profile_grid(
    method: str,
    payload: Dict[str, object],
    path: Path,
    *,
    rayleigh_reference: Optional[Dict[str, object]] = None,
) -> bool:
    psf = payload.get("psf_npz")
    if not isinstance(psf, dict):
        return False
    required = {"cell_id", "nhit_bin", "predE_bin", "profile_edges_deg", "profile_density"}
    if any(key not in psf for key in required):
        return False
    reference = rayleigh_reference if isinstance(rayleigh_reference, dict) else psf
    if any(key not in reference for key in required):
        return False

    cell_ids = np.asarray(psf["cell_id"], dtype=np.int64)
    nhit_bins = np.asarray(psf["nhit_bin"], dtype=str)
    pred_bins = np.asarray(psf["predE_bin"], dtype=str)
    profile_edges_deg = np.asarray(psf["profile_edges_deg"], dtype=np.float64)
    profile_density = np.asarray(psf["profile_density"], dtype=np.float64)
    r_opt = np.asarray(psf.get("r_opt_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    sigma_rad = np.asarray(psf.get("sigma_rad", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    sigma_x_deg = np.asarray(psf.get("sigma_x_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    sigma_y_deg = np.asarray(psf.get("sigma_y_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    mu_x_deg = np.asarray(psf.get("mu_x_deg", np.zeros(cell_ids.shape)), dtype=np.float64)
    mu_y_deg = np.asarray(psf.get("mu_y_deg", np.zeros(cell_ids.shape)), dtype=np.float64)
    double_a = np.asarray(psf.get("double_rayleigh_A", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    double_s1 = np.asarray(psf.get("double_rayleigh_sigma1_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    double_s2 = np.asarray(psf.get("double_rayleigh_sigma2_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    observed_fallback = np.asarray(psf.get("observed_data_fallback", np.ones(cell_ids.shape, dtype=bool)), dtype=bool)
    if cell_ids.size == 0 or profile_density.size == 0:
        return False

    reference_cell_ids = np.asarray(reference["cell_id"], dtype=np.int64)
    reference_edges_deg = np.asarray(reference["profile_edges_deg"], dtype=np.float64)
    reference_density = np.asarray(reference["profile_density"], dtype=np.float64)
    reference_sigma_rad = np.asarray(
        reference.get("sigma_rad", np.full(reference_cell_ids.shape, np.nan)),
        dtype=np.float64,
    )
    if (
        reference_cell_ids.size == 0
        or reference_density.size == 0
        or profile_edges_deg.shape != reference_edges_deg.shape
        or not np.allclose(profile_edges_deg, reference_edges_deg, rtol=0.0, atol=1.0e-12)
    ):
        return False

    fit_ids = fit_cell_ids(payload)
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    reference_index_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(reference_cell_ids)}
    centers = 0.5 * (reference_edges_deg[:-1] + reference_edges_deg[1:])
    x_min = float(reference_edges_deg[0])
    x_max = float(reference_edges_deg[-1])

    plt = setup_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(2.0 * len(ordered_pred), 1.55 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for i, nhit in enumerate(ordered_nhit):
        for j, pred in enumerate(ordered_pred):
            ax = axes[i, j]
            idx = index_by_key.get((nhit, pred))
            if idx is None:
                ax.set_axis_off()
                continue

            cell_id = int(cell_ids[idx])
            reference_idx = reference_index_by_cell.get(cell_id)
            if reference_idx is None:
                ax.set_axis_off()
                continue
            if cell_id in fit_ids:
                ax.set_facecolor("#ecfdf5")
                for spine in ax.spines.values():
                    spine.set_color("#059669")
                    spine.set_linewidth(1.25)
                ax.text(
                    0.97,
                    0.94,
                    "fit",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.8,
                    color="#047857",
                    fontweight="bold",
                )

            density = np.asarray(reference_density[reference_idx], dtype=np.float64)
            method_density = np.asarray(profile_density[idx], dtype=np.float64)
            has_profile = bool(np.isfinite(density).any() and np.nansum(density) > 0.0)
            rayleigh_curve = np.full(centers.shape, np.nan, dtype=np.float64)
            if (
                has_profile
                and reference_idx < reference_sigma_rad.size
                and np.isfinite(reference_sigma_rad[reference_idx])
                and reference_sigma_rad[reference_idx] > 0.0
            ):
                rayleigh_curve = rayleigh_pdf_deg(centers, float(reference_sigma_rad[reference_idx]))

            if method == "observed_data" and idx < observed_fallback.size and not observed_fallback[idx]:
                ax.step(
                    centers,
                    method_density,
                    where="mid",
                    color="#0f766e",
                    linewidth=0.9,
                    alpha=0.9,
                    zorder=2.5,
                )
            if (
                method in {"rayleigh_baseline", "observed_data"}
                and has_profile
                and np.isfinite(rayleigh_curve).any()
            ):
                ax.plot(centers, rayleigh_curve, color="#c9501a", linewidth=1.0, alpha=0.95, zorder=3)
            if (
                method == "two_1d_gaussian"
                and has_profile
                and idx < sigma_x_deg.size
                and idx < sigma_y_deg.size
                and np.isfinite(sigma_x_deg[idx])
                and np.isfinite(sigma_y_deg[idx])
                and sigma_x_deg[idx] > 0.0
                and sigma_y_deg[idx] > 0.0
            ):
                mx = float(mu_x_deg[idx]) if idx < mu_x_deg.size and np.isfinite(mu_x_deg[idx]) else 0.0
                my = float(mu_y_deg[idx]) if idx < mu_y_deg.size and np.isfinite(mu_y_deg[idx]) else 0.0
                ax.plot(
                    centers,
                    two1d_radial_pdf_deg(
                        centers,
                        mu_x_deg=mx,
                        mu_y_deg=my,
                        sigma_x_deg=float(sigma_x_deg[idx]),
                        sigma_y_deg=float(sigma_y_deg[idx]),
                    ),
                    color="#7c3aed",
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=3,
                )
            if (
                method == "double_rayleigh_mixture"
                and has_profile
                and idx < double_a.size
                and idx < double_s1.size
                and idx < double_s2.size
                and np.isfinite(double_a[idx])
                and np.isfinite(double_s1[idx])
                and np.isfinite(double_s2[idx])
                and 0.0 < double_a[idx] < 1.0
                and 0.0 < double_s1[idx] < double_s2[idx]
            ):
                ax.plot(
                    centers,
                    double_rayleigh_pdf_deg(
                        centers,
                        float(double_a[idx]),
                        float(double_s1[idx]),
                        float(double_s2[idx]),
                    ),
                    color="#7e22ce",
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=3,
                )
            if idx < r_opt.size and np.isfinite(r_opt[idx]):
                radius = float(r_opt[idx])
                if x_min <= radius <= x_max:
                    ax.axvline(radius, color="#444444", linewidth=0.7, linestyle="--", zorder=2)
                else:
                    ax.text(
                        0.98,
                        0.06,
                        f"r_opt={radius:.2g} outside",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=5.2,
                        color="#4b5563",
                    )
            ax.step(centers, density, where="mid", color="#1f4e79", linewidth=0.9, zorder=4)

            reference_peak = max(
                float(np.nanmax(density)) if np.isfinite(density).any() else 0.0,
                float(np.nanmax(rayleigh_curve)) if np.isfinite(rayleigh_curve).any() else 0.0,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.0, max(1.0e-6, 1.25 * reference_peak))
            ax.set_title(f"cell {cell_id}: {pred}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            ax.grid(alpha=0.22, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(nhit, fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("dangle [deg]", fontsize=6.7)

    handles = [
        Line2D([0], [0], color="#1f4e79", linewidth=0.9, label="common Rayleigh-baseline weighted MC profile"),
    ]
    if method == "observed_data":
        handles.append(Line2D([0], [0], color="#0f766e", linewidth=0.9, label="accepted observed excess profile"))
    if method in {"rayleigh_baseline", "observed_data"}:
        rayleigh_label = "Rayleigh radial PDF reference" if method == "observed_data" else "Rayleigh radial PDF"
        handles.append(Line2D([0], [0], color="#c9501a", linewidth=0.9, label=rayleigh_label))
    if method == "two_1d_gaussian":
        handles.append(Line2D([0], [0], color="#7c3aed", linewidth=0.9, label="two-1D induced radial PDF"))
    if method == "double_rayleigh_mixture":
        handles.append(Line2D([0], [0], color="#7e22ce", linewidth=0.9, label="double-Rayleigh mixture PDF"))
    handles.extend(
        [
            Line2D([0], [0], color="#444444", linewidth=0.8, linestyle="--", label=f"{method} r_opt"),
            Patch(facecolor="#ecfdf5", edgecolor="#059669", label="included in fit"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.988))
    title_tail = {
        "rayleigh_baseline": "MC profile, Rayleigh radial PDF, method r_opt",
        "two_1d_gaussian": "MC profile, two-1D induced radial PDF, method r_opt",
        "mc_quantile_715": "MC profile and empirical-quantile method r_opt",
        "observed_data": "common MC profile, observed profile, Rayleigh reference, method r_opt",
        "double_rayleigh_mixture": "MC profile, fitted double-Rayleigh mixture PDF, method r_opt",
    }[method]
    fig.suptitle(
        f"{METHODS[method]['label']} Stage B weighted radial PSF profiles: {title_tail}",
        fontsize=11,
        y=0.999,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def psf_profile_grid_figures(runs: Dict[str, Dict[str, object]]) -> str:
    items = []
    for method, payload in runs.items():
        grid = ASSET_DIR / f"v5_psf_{method}_weighted_profiles_fit_shaded.png"
        items.append(
            figure(
                grid,
                f"{METHODS[method]['label']} fit-cell shaded weighted PSF profile grid",
                profile_grid_caption(method),
            )
        )
    return "\n".join(items)


def plot_pull_grid(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    n_methods = len(METHODS)
    fig = plt.figure(figsize=(5.1 * n_methods + 1.5, 4.8), dpi=160, constrained_layout=True)
    grid = fig.add_gridspec(1, n_methods + 1, width_ratios=[1.0] * n_methods + [0.045], wspace=0.08)
    axes_list = []
    share_ax = None
    for idx in range(n_methods):
        ax = fig.add_subplot(grid[0, idx], sharex=share_ax, sharey=share_ax)
        if share_ax is None:
            share_ax = ax
        axes_list.append(ax)
    axes_arr = np.asarray(axes_list, dtype=object)
    cax = fig.add_subplot(grid[0, n_methods])
    labels_nhit = sorted(
        {
            str(value)
            for payload in runs.values()
            for value in np.asarray(payload.get("fit_npz", {}).get("nhit_bin", []), dtype=str)  # type: ignore[union-attr]
        },
        key=interval_key,
    )
    labels_pred = sorted(
        {
            str(value)
            for payload in runs.values()
            for value in np.asarray(payload.get("fit_npz", {}).get("predE_bin", []), dtype=str)  # type: ignore[union-attr]
        },
        key=interval_key,
    )
    pred_index = {label: idx for idx, label in enumerate(labels_pred)}
    sc = None
    for ax, method in zip(axes_arr, METHODS):
        fit_npz = runs[method]["fit_npz"]  # type: ignore[assignment]
        cell_ids = np.asarray(fit_npz.get("cell_id", []), dtype=np.int64)
        nhit = np.asarray(fit_npz.get("nhit_bin", []), dtype=str)
        pred = np.asarray(fit_npz.get("predE_bin", []), dtype=str)
        pulls = np.asarray(fit_npz.get("logpar_conservative_pull", []), dtype=np.float64)
        if cell_ids.size:
            x = np.asarray([pred_index[str(v)] for v in pred], dtype=np.float64)
            y = np.asarray([labels_nhit.index(str(v)) for v in nhit], dtype=np.float64)
            sc = ax.scatter(x, y, c=pulls, cmap="coolwarm", vmin=-6, vmax=6, s=185, edgecolor="#111827", lw=0.45)
            for xi, yi, cid in zip(x, y, cell_ids):
                ax.text(xi, yi, str(int(cid)), ha="center", va="center", fontsize=7.5, color="#111827")
        ax.set_title(METHODS[method]["label"])
        ax.set_xticks(range(len(labels_pred)))
        ax.set_xticklabels(labels_pred, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels_nhit)))
        ax.set_yticklabels(labels_nhit, fontsize=8)
        ax.set_xlim(-0.6, len(labels_pred) - 0.4)
        ax.set_ylim(len(labels_nhit) - 0.4, -0.6)
        ax.grid(alpha=0.2)
    axes_arr[0].set_ylabel("Nhit bin")
    for ax in axes_arr:
        ax.set_xlabel("predicted-energy bin")
    if sc is not None:
        fig.colorbar(sc, cax=cax, label="(excess - model) / conservative err")
    fig.suptitle("Stage F LogPar cell pulls")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_sed_overlay(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=160)
    e_pass5, y_pass5 = pass5_e2_points()
    if e_pass5.size:
        ax.plot(e_pass5, y_pass5, "o", color="#111827", ms=5.5, label="official pass5 WCDA")

    external_path = next(
        (
            payload["paths"]["stage_g_meta"].parent / "external_crab_sed_references.csv"  # type: ignore[index]
            for payload in runs.values()
            if Path(payload["paths"]["stage_g_meta"]).parent.joinpath("external_crab_sed_references.csv").exists()  # type: ignore[index]
        ),
        None,
    )
    if external_path is None:
        fallback = V4_STAGE_G_META.parent / "external_crab_sed_references.csv"
        external_path = fallback if fallback.exists() else None
    if external_path is not None:
        external_rows = read_csv_rows(Path(external_path))
        for dataset, marker, color in [
            ("magic_joint_crab", ".", "#7c3aed"),
            ("hess_2024_stereo", ".", "#059669"),
            ("hawc_2019_nn", ".", "#b45309"),
        ]:
            selected = [row for row in external_rows if row.get("dataset") == dataset and str(row.get("is_upper_limit")).lower() != "true"]
            good = []
            for row in selected:
                e = finite_float(row.get("energy_tev"))
                y = finite_float(row.get("e2_dnde"))
                if e is not None and y is not None and e > 0.0 and y > 0.0:
                    good.append((e, y))
            if good:
                ax.scatter([a for a, _ in good], [b for _, b in good], s=13, marker=marker, color=color, alpha=0.38, label=dataset)

    v4_rows = group_stage_g_rows(read_csv_rows(V4_STAGE_G_SUMMARY), "nhit")
    if v4_rows:
        x = [finite_float(row.get("effective_energy_tev")) for row in v4_rows]
        y = [finite_float(row.get("E2_dnde")) for row in v4_rows]
        dy = [finite_float(row.get("E2_dnde_err")) or 0.0 for row in v4_rows]
        good = [i for i, (a, b) in enumerate(zip(x, y)) if a is not None and b is not None and a > 0.0 and b > 0.0]
        ax.errorbar([x[i] for i in good], [y[i] for i in good], yerr=[dy[i] for i in good], fmt="D", ms=4.5, lw=0.9, color="#6b7280", alpha=0.72, label="v4 aperture baseline")

    for method, payload in runs.items():
        config = METHODS[method]
        rows = group_stage_g_rows(payload["stage_g_rows"], "nhit")  # type: ignore[arg-type]
        x = [finite_float(row.get("effective_energy_tev")) for row in rows]
        y = [finite_float(row.get("E2_dnde")) for row in rows]
        dy = [finite_float(row.get("E2_dnde_err")) or 0.0 for row in rows]
        good = [i for i, (a, b) in enumerate(zip(x, y)) if a is not None and b is not None and a > 0.0 and b > 0.0]
        if good:
            ax.errorbar(
                [x[i] for i in good],
                [y[i] for i in good],
                yerr=[dy[i] for i in good],
                fmt=config["marker"],
                ms=5.2,
                lw=1.0,
                color=config["color"],
                ecolor=config["color"],
                alpha=0.92,
                capsize=2.2,
                label=config["label"],
            )
        params = fit_parameters(payload["fit_meta"], "logpar")  # type: ignore[arg-type]
        if {"phi0", "alpha", "beta"} <= set(params):
            x_curve = np.logspace(np.log10(0.3), np.log10(90.0), 220)
            ax.plot(x_curve, e2_curve(x_curve, params, "logpar"), color=config["color"], lw=1.35, alpha=0.82)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Crab SED: v5 PSF aperture comparison")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.1, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_sed_flux_ratio_to_v4(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    v4_rows = group_stage_g_rows(read_csv_rows(V4_STAGE_G_SUMMARY), "nhit")
    v4_by_label = {
        str(row.get("group_label")): (
            finite_float(row.get("effective_energy_tev")),
            finite_float(row.get("E2_dnde")),
        )
        for row in v4_rows
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.1), dpi=160)
    for method, payload in runs.items():
        config = METHODS[method]
        rows = group_stage_g_rows(payload["stage_g_rows"], "nhit")  # type: ignore[arg-type]
        x = []
        ratio = []
        for row in rows:
            label = str(row.get("group_label"))
            e = finite_float(row.get("effective_energy_tev"))
            y = finite_float(row.get("E2_dnde"))
            v4_e, v4_y = v4_by_label.get(label, (None, None))
            if e is not None and y is not None and v4_y is not None and v4_y > 0.0:
                x.append(e if v4_e is None else v4_e)
                ratio.append(y / v4_y)
        if x:
            ax.plot(x, ratio, marker=config["marker"], color=config["color"], lw=1.4, ms=5, label=config["label"])
    ax.axhline(1.0, color="#6b7280", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("v4 Stage G Nhit effective energy (TeV)")
    ax.set_ylabel("v5 Stage G E2 flux / v4 aperture baseline")
    ax.set_title("Stage G SED flux ratio relative to v4 baseline")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_forward_fold_ratios(nhit_rows: Sequence[Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    labels = sorted({str(row["nhit_bin"]) for row in nhit_rows}, key=interval_key)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    for method, config in METHODS.items():
        values = []
        for label in labels:
            row = next((r for r in nhit_rows if r["method"] == method and r["nhit_bin"] == label), None)
            values.append(finite_float(row.get("observed_over_expected")) if row else float("nan"))
        ax.plot(labels, values, marker=config["marker"], lw=1.5, color=config["color"], label=config["label"])
    ax.axhline(1.0, color="#6b7280", lw=1.0, ls="--")
    ax.set_ylabel("Stage E excess / official pass5 expected")
    ax.set_xlabel("Nhit bin")
    ax.set_title("Official pass5 forward-fold obs/exp by Nhit")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def shell_quantile_radius(edges: np.ndarray, shell_values: np.ndarray, quantile: float) -> Optional[float]:
    values = np.asarray(shell_values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    weights = np.clip(values, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0 or edges.size != weights.size + 1:
        return None
    target = float(quantile) * total
    cumulative = np.cumsum(weights)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), weights.size - 1)
    previous = float(cumulative[idx - 1]) if idx > 0 else 0.0
    shell_weight = float(weights[idx])
    fraction = 0.0 if shell_weight <= 0.0 else (target - previous) / shell_weight
    fraction = min(max(fraction, 0.0), 1.0)
    return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def observed_excess_r715_rows(runs: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows = read_csv_rows(V4_EMPIRICAL_PSF_SUMMARY)
    if not summary_rows or not V4_EMPIRICAL_PSF_PROFILES.exists():
        return []
    summary_by_cell = {int(row["cell_id"]): row for row in summary_rows if finite_float(row.get("cell_id")) is not None}
    rayleigh_psf = runs["rayleigh_baseline"]["psf_npz"]  # type: ignore[assignment]
    psf_cell_id = np.asarray(rayleigh_psf.get("cell_id", []), dtype=np.int64)
    mc_r715_by_cell = {
        int(cid): float(value)
        for cid, value in zip(psf_cell_id, np.asarray(rayleigh_psf.get("mc_quantile_r715_deg", []), dtype=np.float64))
        if math.isfinite(float(value))
    }
    rayleigh_r_by_cell = {
        int(cid): float(value)
        for cid, value in zip(psf_cell_id, np.asarray(rayleigh_psf.get("r_opt_deg", []), dtype=np.float64))
        if math.isfinite(float(value))
    }

    with np.load(V4_EMPIRICAL_PSF_PROFILES, allow_pickle=False) as data:
        edges = np.asarray(data["profile_edges_deg"], dtype=np.float64)
        cell_ids = np.asarray(data["cell_id"], dtype=np.int64)
        excess_profiles = np.asarray(data["excess_profile"], dtype=np.float64)

    rows: List[Dict[str, object]] = []
    for idx, cell_id_raw in enumerate(cell_ids):
        cell_id = int(cell_id_raw)
        meta = summary_by_cell.get(cell_id, {})
        if str(meta.get("fit_reliable")) != "1":
            continue
        profile = excess_profiles[idx]
        r_obs = shell_quantile_radius(edges, profile, TARGET_CONTAINMENT)
        positive_total = float(np.sum(np.clip(np.where(np.isfinite(profile), profile, 0.0), 0.0, None)))
        raw_total = float(np.nansum(profile))
        mc_r715 = mc_r715_by_cell.get(cell_id)
        rows.append(
            {
                "cell_id": cell_id,
                "nhit_bin": meta.get("nhit_bin", ""),
                "predE_bin": meta.get("predE_bin", ""),
                "target_containment": TARGET_CONTAINMENT,
                "observed_excess_r715_deg": r_obs if r_obs is not None else "",
                "mc_quantile_r715_deg": mc_r715 if mc_r715 is not None else "",
                "observed_over_mc_quantile_r715": (r_obs / mc_r715) if r_obs is not None and mc_r715 and mc_r715 > 0.0 else "",
                "rayleigh_r_opt_deg": rayleigh_r_by_cell.get(cell_id, ""),
                "positive_shell_excess": positive_total,
                "raw_shell_excess": raw_total,
                "fit_reliable": meta.get("fit_reliable", ""),
                "significance": meta.get("significance", ""),
            }
        )
    rows.sort(key=lambda row: (interval_key(row.get("nhit_bin")), interval_key(row.get("predE_bin")), int(row.get("cell_id", 0))))
    return rows


def write_observed_excess_r715_csv(rows: Sequence[Dict[str, object]]) -> None:
    write_csv(
        ASSET_DIR / "observed_excess_r715_diagnostic.csv",
        rows,
        [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "target_containment",
            "observed_excess_r715_deg",
            "mc_quantile_r715_deg",
            "observed_over_mc_quantile_r715",
            "rayleigh_r_opt_deg",
            "positive_shell_excess",
            "raw_shell_excess",
            "fit_reliable",
            "significance",
        ],
    )


def plot_observed_excess_r715(rows: Sequence[Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    x = []
    y = []
    labels = []
    for row in rows:
        mc = finite_float(row.get("mc_quantile_r715_deg"))
        obs = finite_float(row.get("observed_excess_r715_deg"))
        if mc is not None and obs is not None and mc > 0.0 and obs > 0.0:
            x.append(mc)
            y.append(obs)
            labels.append(str(row.get("cell_id")))
    fig, ax = plt.subplots(figsize=(6.3, 5.2), dpi=160)
    if x:
        ax.scatter(x, y, s=42, color="#2563eb", edgecolor="#111827", lw=0.45, alpha=0.88)
        lo = min(min(x), min(y)) * 0.85
        hi = max(max(x), max(y)) * 1.15
        ax.plot([lo, hi], [lo, hi], color="#6b7280", ls="--", lw=1.0)
        for xv, yv, label in zip(x, y, labels):
            ax.text(xv, yv, label, fontsize=7, ha="left", va="bottom", color="#374151")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("MC Crab-theta quantile r715 (deg)")
    ax.set_ylabel("observed excess r715 diagnostic (deg)")
    ax.set_title("Observed Crab excess r715 diagnostic")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def psf_shape_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    rows = []
    for method, payload in runs.items():
        psf = payload["psf_npz"]  # type: ignore[assignment]
        if not psf:
            rows.append([method, "pending", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        r = np.asarray(psf["r_opt_deg"], dtype=np.float64)
        sx = np.asarray(psf.get("sigma_x_deg", np.full_like(r, np.nan)), dtype=np.float64)
        sy = np.asarray(psf.get("sigma_y_deg", np.full_like(r, np.nan)), dtype=np.float64)
        q = np.asarray(psf.get("mc_quantile_r715_deg", np.full_like(r, np.nan)), dtype=np.float64)
        borrowed = np.asarray(psf.get("psf_borrowed", np.zeros(r.shape, dtype=bool)), dtype=bool)
        ratio = np.divide(sx, sy, out=np.full_like(sx, np.nan), where=sy > 0.0)
        rows.append(
            [
                method,
                str(r.size),
                fmt(np.nanmedian(r), 4),
                fmt(np.nanmin(r), 4),
                fmt(np.nanmax(r), 4),
                fmt(np.nanmedian(ratio), 4),
                fmt(np.nanmedian(q), 4),
                fmt_int(np.count_nonzero(borrowed)),
            ]
        )
    return rows


def stage_f_table_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    rows = []
    v4_meta = read_json(V4_STAGE_F_META)
    all_items = [("v4 aperture baseline", {"fit_meta": v4_meta, "fit_npz": load_npz(V4_STAGE_F_NPZ), "max_abs_pull": max_abs_pull(load_npz(V4_STAGE_F_NPZ))})]
    all_items.extend((method, payload) for method, payload in runs.items())
    for name, payload in all_items:
        meta = payload["fit_meta"]  # type: ignore[index]
        logpar = fit_params_from_meta(meta, "logpar") if isinstance(meta, dict) else {}
        pl = fit_params_from_meta(meta, "pl") if isinstance(meta, dict) else {}
        lp_params = logpar.get("parameters") if isinstance(logpar.get("parameters"), dict) else {}
        pl_params = pl.get("parameters") if isinstance(pl.get("parameters"), dict) else {}
        rows.append(
            [
                name,
                fmt(lp_params.get("phi0"), 5),
                fmt(lp_params.get("alpha"), 5),
                fmt(lp_params.get("beta"), 5),
                f"{fmt(logpar.get('chi2'), 4)} / {fmt_int(logpar.get('ndof'))}",
                fmt(chi2_over_ndof(logpar), 4),
                fmt(payload.get("max_abs_pull"), 4),  # type: ignore[union-attr]
                fmt(pl_params.get("phi0"), 5),
                fmt(pl_params.get("gamma"), 5),
            ]
        )
    return rows


def two1d_radius_diagnostic(runs: Dict[str, Dict[str, object]]) -> str:
    rayleigh = runs["rayleigh_baseline"]["psf_npz"]  # type: ignore[assignment]
    two1d = runs["two_1d_gaussian"]["psf_npz"]  # type: ignore[assignment]
    if not rayleigh or not two1d:
        return ""
    rayleigh_r = {
        int(cid): float(r)
        for cid, r in zip(np.asarray(rayleigh.get("cell_id", []), dtype=np.int64), np.asarray(rayleigh.get("r_opt_deg", []), dtype=np.float64))
        if np.isfinite(r) and float(r) > 0.0
    }
    ratios = []
    for cid, r in zip(np.asarray(two1d.get("cell_id", []), dtype=np.int64), np.asarray(two1d.get("r_opt_deg", []), dtype=np.float64)):
        cell_id = int(cid)
        if cell_id in rayleigh_r and np.isfinite(r):
            ratios.append(float(r) / rayleigh_r[cell_id])
    if not ratios:
        return ""
    ratio_arr = np.asarray(ratios, dtype=np.float64)
    sigma_ratio = np.asarray(two1d.get("sigma_x_over_y", []), dtype=np.float64)
    finite_sigma_ratio = sigma_ratio[np.isfinite(sigma_ratio)]
    sigma_text = ""
    if finite_sigma_ratio.size:
        sigma_text = (
            f" The fitted anisotropy is modest: sigma_x/sigma_y spans "
            f"{fmt(float(np.nanmin(finite_sigma_ratio)), 4)}-{fmt(float(np.nanmax(finite_sigma_ratio)), 4)} "
            f"with median {fmt(float(np.nanmedian(finite_sigma_ratio)), 4)}."
        )
    return (
        "<p>"
        "For this v5 run the two-1D Gaussian aperture radius is numerically very close to the Rayleigh baseline: "
        f"r_opt(two-1D) / r_opt(Rayleigh) ranges from {fmt(float(np.nanmin(ratio_arr)), 5)} "
        f"to {fmt(float(np.nanmax(ratio_arr)), 5)}, with median {fmt(float(np.nanmedian(ratio_arr)), 5)}."
        f"{sigma_text} This is why the grey r_opt markers are visually almost indistinguishable from the Rayleigh baseline."
        "</p>"
    )


def psf_index_by_cell(psf: Dict[str, np.ndarray]) -> Dict[int, int]:
    cell_ids = np.asarray(psf.get("cell_id", []), dtype=np.int64)
    return {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}


def npz_value(psf: Dict[str, np.ndarray], key: str, idx: int) -> object:
    if key not in psf:
        return None
    values = np.asarray(psf[key])
    if idx >= values.shape[0]:
        return None
    value = values[idx]
    if isinstance(value, np.generic):
        return value.item()
    return value


def double_rayleigh_focus_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    double_psf = runs.get("double_rayleigh_mixture", {}).get("psf_npz")
    rayleigh_psf = runs.get("rayleigh_baseline", {}).get("psf_npz")
    if not isinstance(double_psf, dict) or not double_psf or not isinstance(rayleigh_psf, dict) or not rayleigh_psf:
        return [["double_rayleigh_mixture not available", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]
    double_idx = psf_index_by_cell(double_psf)
    rayleigh_idx = psf_index_by_cell(rayleigh_psf)
    rows: List[List[str]] = []
    for cell_id in FOCUS_CELLS:
        idx = double_idx.get(int(cell_id))
        if idx is None:
            rows.append([str(cell_id), "missing", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        ray_idx = rayleigh_idx.get(int(cell_id))
        r_double = finite_float(npz_value(double_psf, "r_opt_deg", idx))
        r_ray = finite_float(npz_value(rayleigh_psf, "r_opt_deg", ray_idx)) if ray_idx is not None else None
        ratio = r_double / r_ray if r_double is not None and r_ray not in {None, 0.0} else None
        borrowed = bool(npz_value(double_psf, "psf_borrowed", idx) or False)
        reason = str(npz_value(double_psf, "double_rayleigh_fallback_reason", idx) or "")
        borrowed_from = str(npz_value(double_psf, "borrowed_from", idx) or "")
        if borrowed:
            reason = "; ".join(bit for bit in [reason, f"psfborrow_from:{borrowed_from}"] if bit)
        rows.append(
            [
                f"<strong>{int(cell_id)}</strong>",
                esc(npz_value(double_psf, "nhit_bin", idx) or ""),
                esc(npz_value(double_psf, "predE_bin", idx) or ""),
                fmt(npz_value(double_psf, "double_rayleigh_A", idx), 4),
                fmt(npz_value(double_psf, "double_rayleigh_sigma1_deg", idx), 4),
                fmt(npz_value(double_psf, "double_rayleigh_sigma2_deg", idx), 4),
                fmt(npz_value(double_psf, "double_rayleigh_sigma_eq_deg", idx), 4),
                fmt(r_double, 4),
                fmt(ratio, 4),
                fmt(npz_value(double_psf, "double_rayleigh_containment_r_opt", idx), 4),
                fmt(npz_value(double_psf, "double_rayleigh_chi2_ndof", idx), 4),
                esc(str(npz_value(double_psf, "fit_quality", idx) or "")),
                esc(reason),
            ]
        )
    return rows


def double_rayleigh_summary(runs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    psf = runs.get("double_rayleigh_mixture", {}).get("psf_npz")
    rayleigh = runs.get("rayleigh_baseline", {}).get("psf_npz")
    if not isinstance(psf, dict) or not psf:
        return {"status": "missing"}
    a = np.asarray(psf.get("double_rayleigh_A", []), dtype=np.float64)
    s1 = np.asarray(psf.get("double_rayleigh_sigma1_deg", []), dtype=np.float64)
    s2 = np.asarray(psf.get("double_rayleigh_sigma2_deg", []), dtype=np.float64)
    seq = np.asarray(psf.get("double_rayleigh_sigma_eq_deg", []), dtype=np.float64)
    r = np.asarray(psf.get("r_opt_deg", []), dtype=np.float64)
    borrowed = np.asarray(psf.get("psf_borrowed", np.zeros(r.shape, dtype=bool)), dtype=bool)
    ratio_values: List[float] = []
    if isinstance(rayleigh, dict) and rayleigh:
        ray_r = {
            int(cid): float(value)
            for cid, value in zip(np.asarray(rayleigh.get("cell_id", []), dtype=np.int64), np.asarray(rayleigh.get("r_opt_deg", []), dtype=np.float64))
            if np.isfinite(value) and float(value) > 0.0
        }
        for cid, value in zip(np.asarray(psf.get("cell_id", []), dtype=np.int64), r):
            if int(cid) in ray_r and np.isfinite(value):
                ratio_values.append(float(value) / ray_r[int(cid)])
    return {
        "status": "available",
        "median_A": float(np.nanmedian(a)) if a.size and np.isfinite(a).any() else None,
        "median_sigma1_deg": float(np.nanmedian(s1)) if s1.size and np.isfinite(s1).any() else None,
        "median_sigma2_deg": float(np.nanmedian(s2)) if s2.size and np.isfinite(s2).any() else None,
        "median_sigma_eq_deg": float(np.nanmedian(seq)) if seq.size and np.isfinite(seq).any() else None,
        "median_r_opt_over_rayleigh": float(np.nanmedian(ratio_values)) if ratio_values else None,
        "min_r_opt_over_rayleigh": float(np.nanmin(ratio_values)) if ratio_values else None,
        "max_r_opt_over_rayleigh": float(np.nanmax(ratio_values)) if ratio_values else None,
        "borrowed_cells": int(np.count_nonzero(borrowed)),
    }


def double_rayleigh_focus_records(runs: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    double_psf = runs.get("double_rayleigh_mixture", {}).get("psf_npz")
    rayleigh_psf = runs.get("rayleigh_baseline", {}).get("psf_npz")
    if not isinstance(double_psf, dict) or not double_psf or not isinstance(rayleigh_psf, dict) or not rayleigh_psf:
        return []
    double_idx = psf_index_by_cell(double_psf)
    rayleigh_idx = psf_index_by_cell(rayleigh_psf)
    records: List[Dict[str, object]] = []
    for cell_id in FOCUS_CELLS:
        idx = double_idx.get(int(cell_id))
        if idx is None:
            records.append({"cell_id": int(cell_id), "status": "missing"})
            continue
        ray_idx = rayleigh_idx.get(int(cell_id))
        r_double = finite_float(npz_value(double_psf, "r_opt_deg", idx))
        r_ray = finite_float(npz_value(rayleigh_psf, "r_opt_deg", ray_idx)) if ray_idx is not None else None
        records.append(
            {
                "cell_id": int(cell_id),
                "nhit_bin": str(npz_value(double_psf, "nhit_bin", idx) or ""),
                "predE_bin": str(npz_value(double_psf, "predE_bin", idx) or ""),
                "A": finite_float(npz_value(double_psf, "double_rayleigh_A", idx)),
                "sigma1_deg": finite_float(npz_value(double_psf, "double_rayleigh_sigma1_deg", idx)),
                "sigma2_deg": finite_float(npz_value(double_psf, "double_rayleigh_sigma2_deg", idx)),
                "sigma_eq_deg": finite_float(npz_value(double_psf, "double_rayleigh_sigma_eq_deg", idx)),
                "r_opt_deg": r_double,
                "r_opt_over_rayleigh": r_double / r_ray if r_double is not None and r_ray not in {None, 0.0} else None,
                "containment_r_opt": finite_float(npz_value(double_psf, "double_rayleigh_containment_r_opt", idx)),
                "chi2_ndof": finite_float(npz_value(double_psf, "double_rayleigh_chi2_ndof", idx)),
                "fit_quality": str(npz_value(double_psf, "fit_quality", idx) or ""),
                "fallback_reason": str(npz_value(double_psf, "double_rayleigh_fallback_reason", idx) or ""),
                "psf_borrowed": bool(npz_value(double_psf, "psf_borrowed", idx) or False),
                "borrowed_from": str(npz_value(double_psf, "borrowed_from", idx) or ""),
            }
        )
    return records


def fallback_quality_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for method, payload in runs.items():
        psf = payload.get("psf_npz")
        if not isinstance(psf, dict) or not psf:
            continue
        cell_ids = np.asarray(psf.get("cell_id", []), dtype=np.int64)
        fit_quality = np.asarray(psf.get("fit_quality", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        borrowed = np.asarray(psf.get("psf_borrowed", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
        borrowed_from = np.asarray(psf.get("borrowed_from", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        observed_fallback = np.asarray(psf.get("observed_data_fallback", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
        observed_reason = np.asarray(psf.get("observed_data_fallback_reason", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        double_reason = np.asarray(psf.get("double_rayleigh_fallback_reason", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        nhit = np.asarray(psf.get("nhit_bin", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        pred = np.asarray(psf.get("predE_bin", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        for idx, cell_id in enumerate(cell_ids):
            reasons = []
            quality = str(fit_quality[idx]) if idx < fit_quality.size else ""
            if quality and quality != "ok":
                reasons.append(quality)
            if method == "observed_data" and idx < observed_fallback.size and bool(observed_fallback[idx]):
                reasons.append(str(observed_reason[idx] if idx < observed_reason.size else "observed_profile_fallback"))
            if method == "double_rayleigh_mixture":
                reason = str(double_reason[idx] if idx < double_reason.size else "")
                if reason:
                    reasons.append(reason)
            if idx < borrowed.size and bool(borrowed[idx]):
                reasons.append(f"psfborrow_from:{borrowed_from[idx] if idx < borrowed_from.size else ''}")
            deduped = list(dict.fromkeys(bit for bit in reasons if bit))
            if deduped:
                rows.append(
                    [
                        method,
                        int(cell_id),
                        esc(nhit[idx] if idx < nhit.size else ""),
                        esc(pred[idx] if idx < pred.size else ""),
                        esc("; ".join(deduped)),
                    ]
                )
    if not rows:
        return [["none", "n/a", "n/a", "n/a", "n/a"]]
    return rows


def focus_cell_pull_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    method_pull: Dict[str, Dict[int, float]] = {}
    method_meta: Dict[int, Tuple[str, str]] = {}
    for method, payload in runs.items():
        fit_npz = payload.get("fit_npz")
        if not isinstance(fit_npz, dict) or not fit_npz:
            continue
        cell_ids = np.asarray(fit_npz.get("cell_id", []), dtype=np.int64)
        pulls = np.asarray(fit_npz.get("logpar_conservative_pull", []), dtype=np.float64)
        nhit = np.asarray(fit_npz.get("nhit_bin", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        pred = np.asarray(fit_npz.get("predE_bin", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
        method_pull[method] = {
            int(cell_id): float(pull)
            for cell_id, pull in zip(cell_ids, pulls)
            if np.isfinite(pull)
        }
        for idx, cell_id in enumerate(cell_ids):
            method_meta.setdefault(int(cell_id), (str(nhit[idx]) if idx < nhit.size else "", str(pred[idx]) if idx < pred.size else ""))
    rows: List[List[str]] = []
    for cell_id in FOCUS_CELLS:
        nhit, pred = method_meta.get(int(cell_id), ("", ""))
        row = [f"<strong>{int(cell_id)}</strong>", esc(nhit), esc(pred)]
        for method in METHODS:
            row.append(fmt(method_pull.get(method, {}).get(int(cell_id)), 4))
        rows.append(row)
    return rows


def focus_cell_pull_headers() -> List[str]:
    return ["cell", "Nhit", "predE"] + [METHODS[method]["label"] for method in METHODS]


def main_comparison_rows(runs: Dict[str, Dict[str, object]], summary_rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    base_params = fit_parameters(read_json(V4_STAGE_F_META), "logpar")
    rows = []
    for method, payload in runs.items():
        logpar = payload["logpar"]  # type: ignore[assignment]
        params = logpar.get("parameters") if isinstance(logpar.get("parameters"), dict) else {}
        total = next((row for row in summary_rows if row["method"] == method and row["nhit_bin"] == "all"), {})
        phi_shift = (
            finite_float(params.get("phi0")) / base_params.get("phi0") - 1.0
            if finite_float(params.get("phi0")) is not None and base_params.get("phi0") not in {None, 0.0}
            else None
        )
        rows.append(
            [
                method,
                payload["status"],
                fmt(params.get("phi0"), 5),
                fmt(phi_shift, 5),
                fmt(params.get("alpha"), 5),
                fmt(params.get("beta"), 5),
                fmt(chi2_over_ndof(logpar), 4),
                fmt(payload.get("max_abs_pull"), 4),
                fmt(total.get("observed_over_expected"), 4),
            ]
        )
    return rows


def main_comparison_records(runs: Dict[str, Dict[str, object]], summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    base_params = fit_parameters(read_json(V4_STAGE_F_META), "logpar")
    records: List[Dict[str, object]] = []
    for method, payload in runs.items():
        logpar = payload["logpar"]  # type: ignore[assignment]
        params = logpar.get("parameters") if isinstance(logpar.get("parameters"), dict) else {}
        total = next((row for row in summary_rows if row["method"] == method and row["nhit_bin"] == "all"), {})
        phi0 = finite_float(params.get("phi0"))
        base_phi0 = base_params.get("phi0")
        records.append(
            {
                "method": method,
                "status": payload["status"],
                "phi0": phi0,
                "phi0_shift_vs_v4": phi0 / base_phi0 - 1.0 if phi0 is not None and base_phi0 not in {None, 0.0} else None,
                "alpha": finite_float(params.get("alpha")),
                "beta": finite_float(params.get("beta")),
                "chi2": finite_float(logpar.get("chi2")) if isinstance(logpar, dict) else None,
                "ndof": finite_float(logpar.get("ndof")) if isinstance(logpar, dict) else None,
                "chi2_over_ndof": chi2_over_ndof(logpar) if isinstance(logpar, dict) else None,
                "max_abs_pull": finite_float(payload.get("max_abs_pull")),
                "total_obs_over_pass5": finite_float(total.get("observed_over_expected")),
            }
        )
    return records


def low_nhit_rows(nhit_rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    focus = ["[125,200)", "[200,300)", "[300,500)"]
    rows = []
    for label in focus:
        item = [label]
        for method in METHODS:
            row = next((r for r in nhit_rows if r["method"] == method and r["nhit_bin"] == label), None)
            item.extend(
                [
                    fmt(row.get("observed_over_expected") if row else None, 4),
                    fmt(row.get("excess") if row else None, 5),
                    fmt(row.get("official_expected_counts") if row else None, 5),
                ]
            )
        rows.append(item)
    return rows


def low_nhit_headers() -> List[str]:
    headers = ["Nhit"]
    for method, config in METHODS.items():
        short = config["label"]
        headers.extend([f"{short} obs/pass5", f"{short} excess", f"{short} pass5"])
    return headers


def observed_data_fallback_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    psf = runs.get("observed_data", {}).get("psf_npz")
    if not isinstance(psf, dict) or not psf:
        return [["observed_data not available", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]
    cell_ids = np.asarray(psf.get("cell_id", []), dtype=np.int64)
    fallback = np.asarray(psf.get("observed_data_fallback", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
    reason = np.asarray(psf.get("observed_data_fallback_reason", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
    borrowed = np.asarray(psf.get("psf_borrowed", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
    borrowed_from = np.asarray(psf.get("borrowed_from", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
    r_obs = np.asarray(psf.get("observed_data_r715_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    r_current = np.asarray(psf.get("r_opt_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    containment = np.asarray(psf.get("containment_r_opt", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    data_containment = np.asarray(psf.get("observed_data_containment_r_opt", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    rows: List[List[str]] = []
    for idx, cell_id in enumerate(cell_ids):
        is_observed_fallback = idx < fallback.size and bool(fallback[idx])
        is_psfborrow = idx < borrowed.size and bool(borrowed[idx])
        if is_observed_fallback or is_psfborrow:
            reason_bits = []
            if is_observed_fallback:
                reason_bits.append(str(reason[idx] if idx < reason.size else "observed_profile_fallback"))
            if is_psfborrow:
                reason_bits.append(f"psfborrow_from:{borrowed_from[idx] if idx < borrowed_from.size else ''}")
            rows.append(
                [
                    int(cell_id),
                    esc(np.asarray(psf.get("nhit_bin", []), dtype=str)[idx] if "nhit_bin" in psf else ""),
                    esc(np.asarray(psf.get("predE_bin", []), dtype=str)[idx] if "predE_bin" in psf else ""),
                    fmt(r_obs[idx] if idx < r_obs.size else None, 4),
                    fmt(r_current[idx] if idx < r_current.size else None, 4),
                    fmt(data_containment[idx] if idx < data_containment.size else None, 4),
                    fmt(containment[idx] if idx < containment.size else None, 4),
                    esc("; ".join(bit for bit in reason_bits if bit)),
                ]
            )
    if not rows:
        return [["no fallback cells", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]
    return rows


def observed_data_used_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    psf = runs.get("observed_data", {}).get("psf_npz")
    if not isinstance(psf, dict) or not psf:
        return [["observed_data not available", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]
    cell_ids = np.asarray(psf.get("cell_id", []), dtype=np.int64)
    fallback = np.asarray(psf.get("observed_data_fallback", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
    borrowed = np.asarray(psf.get("psf_borrowed", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
    r_obs = np.asarray(psf.get("observed_data_r715_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    containment = np.asarray(psf.get("observed_data_containment_r_opt", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    ratio_ray = np.asarray(psf.get("observed_data_r_opt_over_rayleigh", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    ratio_mc = np.asarray(psf.get("observed_data_r_opt_over_mc_quantile", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    positive_total = np.asarray(psf.get("observed_data_positive_total", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    rows: List[List[str]] = []
    for idx, cell_id in enumerate(cell_ids):
        is_fallback = idx < fallback.size and bool(fallback[idx])
        is_borrowed = idx < borrowed.size and bool(borrowed[idx])
        if not is_fallback and not is_borrowed and idx < r_obs.size and np.isfinite(r_obs[idx]):
            rows.append(
                [
                    int(cell_id),
                    esc(np.asarray(psf.get("nhit_bin", []), dtype=str)[idx] if "nhit_bin" in psf else ""),
                    esc(np.asarray(psf.get("predE_bin", []), dtype=str)[idx] if "predE_bin" in psf else ""),
                    fmt(r_obs[idx], 4),
                    fmt(containment[idx] if idx < containment.size else None, 4),
                    fmt(ratio_ray[idx] if idx < ratio_ray.size else None, 4),
                    fmt(ratio_mc[idx] if idx < ratio_mc.size else None, 4),
                    fmt(positive_total[idx] if idx < positive_total.size else None, 5),
                ]
            )
    if not rows:
        return [["no accepted observed-data cells", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]
    return rows


def observed_data_aperture_summary(runs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    psf = runs.get("observed_data", {}).get("psf_npz")
    if not isinstance(psf, dict) or not psf:
        return {"status": "missing"}
    cell_ids = np.asarray(psf.get("cell_id", []), dtype=np.int64)
    nhit = np.asarray(psf.get("nhit_bin", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
    pred = np.asarray(psf.get("predE_bin", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
    fallback = np.asarray(psf.get("observed_data_fallback", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
    borrowed = np.asarray(psf.get("psf_borrowed", np.zeros(cell_ids.shape, dtype=bool)), dtype=bool)
    reason = np.asarray(psf.get("observed_data_fallback_reason", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
    borrowed_from = np.asarray(psf.get("borrowed_from", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)
    r_opt = np.asarray(psf.get("r_opt_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    obs_r = np.asarray(psf.get("observed_data_r715_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    raw_obs_r = np.asarray(psf.get("observed_data_raw_r715_deg", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    containment = np.asarray(psf.get("containment_r_opt", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    obs_containment = np.asarray(psf.get("observed_data_containment_r_opt", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    ratio_ray = np.asarray(psf.get("observed_data_r_opt_over_rayleigh", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    ratio_mc = np.asarray(psf.get("observed_data_r_opt_over_mc_quantile", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    positive_total = np.asarray(psf.get("observed_data_positive_total", np.full(cell_ids.shape, np.nan)), dtype=np.float64)
    profile_source = np.asarray(psf.get("observed_data_profile_source", np.full(cell_ids.shape, "", dtype="U1")), dtype=str)

    accepted: List[Dict[str, object]] = []
    fallback_rows: List[Dict[str, object]] = []
    for idx, cell_id in enumerate(cell_ids):
        row = {
            "cell_id": int(cell_id),
            "nhit_bin": str(nhit[idx]) if idx < nhit.size else "",
            "predE_bin": str(pred[idx]) if idx < pred.size else "",
            "r_opt_deg": float(r_opt[idx]) if idx < r_opt.size and np.isfinite(r_opt[idx]) else None,
            "observed_data_r715_deg": float(obs_r[idx]) if idx < obs_r.size and np.isfinite(obs_r[idx]) else None,
            "observed_data_raw_r715_deg": float(raw_obs_r[idx]) if idx < raw_obs_r.size and np.isfinite(raw_obs_r[idx]) else None,
            "containment_r_opt": float(containment[idx]) if idx < containment.size and np.isfinite(containment[idx]) else None,
            "observed_data_containment_r_opt": float(obs_containment[idx]) if idx < obs_containment.size and np.isfinite(obs_containment[idx]) else None,
            "observed_data_r_opt_over_rayleigh": float(ratio_ray[idx]) if idx < ratio_ray.size and np.isfinite(ratio_ray[idx]) else None,
            "observed_data_r_opt_over_mc_quantile": float(ratio_mc[idx]) if idx < ratio_mc.size and np.isfinite(ratio_mc[idx]) else None,
            "observed_data_positive_total": float(positive_total[idx]) if idx < positive_total.size and np.isfinite(positive_total[idx]) else None,
            "observed_data_profile_source": str(profile_source[idx]) if idx < profile_source.size else "",
        }
        is_fallback = idx < fallback.size and bool(fallback[idx])
        is_borrowed = idx < borrowed.size and bool(borrowed[idx])
        if is_fallback or is_borrowed:
            reasons = []
            if is_fallback:
                reasons.append(str(reason[idx] if idx < reason.size else "observed_profile_fallback"))
            if is_borrowed:
                reasons.append(f"psfborrow_from:{borrowed_from[idx] if idx < borrowed_from.size else ''}")
            row["fallback_reason"] = "; ".join(bit for bit in reasons if bit)
            fallback_rows.append(row)
        else:
            accepted.append(row)

    return {
        "status": "available",
        "target_containment": TARGET_CONTAINMENT,
        "accepted_data_psf_cells": accepted,
        "fallback_cells": fallback_rows,
        "accepted_count": len(accepted),
        "fallback_count": len(fallback_rows),
        "note": "fallback_cells includes both observed-profile quality-gate fallback and the existing v3/v4 psfborrow cells.",
    }


def observed_excess_r715_table_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    if not rows:
        return [["not available", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]
    out = []
    for row in rows:
        out.append(
            [
                esc(row.get("cell_id")),
                esc(row.get("nhit_bin")),
                esc(row.get("predE_bin")),
                fmt(row.get("observed_excess_r715_deg"), 4),
                fmt(row.get("mc_quantile_r715_deg"), 4),
                fmt(row.get("observed_over_mc_quantile_r715"), 4),
                fmt(row.get("rayleigh_r_opt_deg"), 4),
            ]
        )
    return out[:12] if out else [["available but no reliable cells", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], *, classes: str = "") -> str:
    cls = f' class="{classes}"' if classes else ""
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    return f"<table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure(path: Path, title: str, caption: str) -> str:
    if not path.exists():
        return f"<p class=\"warn\">Missing figure: <code>{esc(path)}</code></p>"
    return (
        "<figure>"
        f"<img src=\"{esc(rel(path))}\" alt=\"{esc(title)}\">"
        f"<figcaption><strong>{esc(title)}</strong> {esc(caption)}</figcaption>"
        "</figure>"
    )


def method_definition_table() -> str:
    rows = []
    for method, config in METHODS.items():
        rows.append([method, esc(config["role"]), f"<code>{esc(config['run_id'])}</code>"])
    return table(["method", "role", "run id"], rows)


def metadata_validation_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    rows = []
    for method, payload in runs.items():
        response_meta = payload["response_meta"]  # type: ignore[assignment]
        signal_npz = payload["signal_npz"]  # type: ignore[assignment]
        conditioning = response_meta.get("response_aperture_conditioning") if isinstance(response_meta, dict) else {}
        containment = np.asarray(signal_npz.get("containment_r_opt", []), dtype=np.float64) if signal_npz else np.asarray([])
        rows.append(
            [
                method,
                esc(response_meta.get("response_type") if isinstance(response_meta, dict) else ""),
                esc(conditioning.get("mode") if isinstance(conditioning, dict) else ""),
                "yes" if containment.size and np.allclose(containment, 1.0, rtol=0.0, atol=1.0e-10) else "no",
            ]
        )
    return rows


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    summary_rows, nhit_rows, cell_rows = build_forward_fold_tables(runs)
    observed_r715_rows = observed_excess_r715_rows(runs)

    plot_r_opt_ratio_heatmap(runs, ASSET_DIR / "v5_psf_r_opt_ratio_heatmap.png")
    plot_observed_data_radius_comparison(runs, ASSET_DIR / "v5_psf_observed_data_radius_comparison.png")
    plot_shape_scatter(runs, ASSET_DIR / "v5_psf_radius_shape_scatter.png")
    plot_weighted_psf_profile_overlay(runs, ASSET_DIR / "v5_psf_weighted_profile_overlay.png")
    rayleigh_reference = runs.get("rayleigh_baseline", {}).get("psf_npz")
    for method, payload in runs.items():
        plot_fit_shaded_psf_profile_grid(
            method,
            payload,
            ASSET_DIR / f"v5_psf_{method}_weighted_profiles_fit_shaded.png",
            rayleigh_reference=rayleigh_reference if isinstance(rayleigh_reference, dict) else None,
        )
    plot_pull_grid(runs, ASSET_DIR / "v5_psf_cell_pull_grid.png")
    plot_sed_overlay(runs, ASSET_DIR / "v5_psf_sed_overlay.png")
    plot_sed_flux_ratio_to_v4(runs, ASSET_DIR / "v5_psf_sed_flux_ratio_to_v4.png")
    plot_forward_fold_ratios(nhit_rows, ASSET_DIR / "v5_psf_official_forward_fold_nhit.png")
    write_observed_excess_r715_csv(observed_r715_rows)
    plot_observed_excess_r715(observed_r715_rows, ASSET_DIR / "v5_observed_excess_r715_diagnostic.png")

    write_json(
        ASSET_DIR / "v5_psf_comparison_summary.json",
        {
            "methods": {
                method: {
                    "run_id": payload["run_id"],
                    "status": payload["status"],
                    "paths": {name: str(path) for name, path in payload["paths"].items()},  # type: ignore[union-attr]
                    "logpar": payload["logpar"],
                    "pl": payload["pl"],
                    "max_abs_pull": payload["max_abs_pull"],
                }
                for method, payload in runs.items()
            },
            "official_forward_fold": {
                "summary_csv": str(ASSET_DIR / "official_pass5_forward_fold_summary.csv"),
                "nhit_summary_csv": str(ASSET_DIR / "official_pass5_forward_fold_nhit_summary.csv"),
                "cell_counts_csv": str(ASSET_DIR / "official_pass5_forward_fold_cell_counts.csv"),
                "method": "official pass5 dN/dE log-log piecewise interpolation folded through each v5 aperture-conditioned Stage A response and Stage E containment=1 exposure",
            },
            "main_result": main_comparison_records(runs, summary_rows),
            "v4_reference": {
                "stage_b_npz": str(V4_STAGE_B_NPZ),
                "stage_f_npz": str(V4_STAGE_F_NPZ),
                "stage_g_summary_csv": str(V4_STAGE_G_SUMMARY),
            },
            "weighted_psf_profiles": {
                "overlay_figure": str(ASSET_DIR / "v5_psf_weighted_profile_overlay.png"),
                "profile_source": (
                    "All five fit-shaded grids use the exact Rayleigh-baseline Stage B profile_density as the common blue MC reference, "
                    "with identical 0-5 deg x limits and per-cell Rayleigh-reference y limits. "
                    "The observed_data grid overlays accepted pedestal-subtracted observed excess profiles in teal, "
                    "and double_rayleigh_mixture overlays the fitted mixture PDF."
                ),
                "fit_shaded_grids": {
                    method: str(ASSET_DIR / f"v5_psf_{method}_weighted_profiles_fit_shaded.png")
                    for method, payload in runs.items()
                },
            },
            "double_rayleigh_mixture_diagnostics": {
                **double_rayleigh_summary(runs),
                "focus_cells": list(FOCUS_CELLS),
                "focus_cell_records": double_rayleigh_focus_records(runs),
            },
            "fallback_cells": {
                "note": "Rows include non-ok fit_quality, observed-data quality-gate fallback, double-Rayleigh fit fallback, and v3/v4 psfborrow cells.",
                "rows": fallback_quality_rows(runs),
            },
            "observed_data_aperture": {
                **observed_data_aperture_summary(runs),
                "radius_comparison_figure": str(ASSET_DIR / "v5_psf_observed_data_radius_comparison.png"),
            },
            "observed_excess_r715_diagnostic": {
                "target_containment": TARGET_CONTAINMENT,
                "csv": str(ASSET_DIR / "observed_excess_r715_diagnostic.csv"),
                "figure": str(ASSET_DIR / "v5_observed_excess_r715_diagnostic.png"),
                "note": "legacy diagnostic computed from v4 empirical radial excess profiles using positive shell excess only; observed_data branch uses the pedestal-subtracted version as its aperture input",
            },
        },
    )

    css = """
    :root { color-scheme: light; --ink:#111827; --muted:#6b7280; --line:#d1d5db; --bg:#ffffff; --soft:#f3f4f6; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }
    h1 { font-size: 30px; line-height:1.15; margin: 0 0 10px; letter-spacing:0; }
    h2 { font-size: 22px; margin: 34px 0 12px; border-top:1px solid var(--line); padding-top:24px; letter-spacing:0; }
    h3 { font-size: 16px; margin: 24px 0 10px; letter-spacing:0; }
    p { line-height:1.58; color:#374151; }
    code { background:var(--soft); padding:1px 4px; border-radius:4px; }
    table { width:100%; border-collapse: collapse; margin: 12px 0 22px; font-size: 13px; }
    th, td { border:1px solid var(--line); padding:7px 8px; vertical-align: top; text-align:left; }
    th { background:#f9fafb; font-weight:650; }
    td:nth-child(n+2), th:nth-child(n+2) { text-align:right; }
    td:first-child, th:first-child { text-align:left; }
    img { width:100%; display:block; border:1px solid var(--line); background:#fff; }
    figure { margin: 18px 0 26px; }
    figcaption { font-size: 13px; color:#4b5563; margin-top:7px; line-height:1.45; }
    .lede { font-size: 15px; max-width: 930px; }
    .grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 16px 0 20px; }
    .metric { border:1px solid var(--line); padding:12px; background:#f9fafb; }
    .metric .label { color:var(--muted); font-size:12px; }
    .metric .value { font-size:22px; font-weight:720; margin-top:2px; }
    .warn { color:#991b1b; }
    @media (max-width: 820px) { .grid { grid-template-columns:1fr; } table { font-size:12px; } main { padding:24px 14px 42px; } }
    """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crab SED v5 PSF Aperture Comparison</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Crab SED v5 PSF Aperture Comparison</h1>
  <p class="lede">This report compares {len(METHODS)} PSF aperture definitions under the same v4 response contract: each branch has its own Stage B PSF artifact, aperture-conditioned Stage A response, annulus-normalized Stage D background, Stage E signal with <code>containment_r_opt=1</code>, Stage F fit, and Stage G SED points. The target containment is the existing Rayleigh contract <code>1 - exp(-0.5 * 1.58^2)</code>, not a newly rounded 0.715 scalar.</p>

  <div class="grid">
    <div class="metric"><div class="label">methods complete</div><div class="value">{sum(1 for p in runs.values() if p["status"] == "complete")}/{len(METHODS)}</div></div>
    <div class="metric"><div class="label">main contract</div><div class="value">aperture A</div></div>
    <div class="metric"><div class="label">reference</div><div class="value">v4 drop4</div></div>
  </div>

  <h2>Method Definitions</h2>
  {method_definition_table()}

  <h2>Main Result</h2>
  {table(["method", "status", "LogPar phi0", "phi0 shift vs v4", "alpha", "beta", "chi2/ndof", "max abs pull", "total obs/pass5"], main_comparison_rows(runs, summary_rows))}

  <h2>PSF Radius And Shape</h2>
  {table(["method", "cells", "median r_opt", "min r_opt", "max r_opt", "median sigma_x/sigma_y", "median MC quantile r715", "borrowed cells"], psf_shape_rows(runs))}
  {figure(ASSET_DIR / "v5_psf_r_opt_ratio_heatmap.png", "r_opt ratio heatmap", "Each panel is one PSF method on the same Nhit versus predicted-energy cell grid. The number and color in each cell are r_opt(method) / r_opt(rayleigh_baseline): 1.00 means the same aperture radius as the Rayleigh baseline, values above 1 use a larger aperture, and values below 1 use a smaller aperture. Borrowed PSF cells follow the v3/v4 psfborrow policy by default.")}
  {figure(ASSET_DIR / "v5_psf_observed_data_radius_comparison.png", "Observed-data radius comparison", "The two panels compare the observed-data aperture radius against the Rayleigh baseline and the MC-quantile branch on the same cell grid. A star marks cells where the observed profile failed the data-PSF quality gates and therefore used the fallback radius before the normal psfborrow policy.")}
  <h3>Double-Rayleigh Mixture Diagnostics</h3>
  <p>The double-Rayleigh branch fits the Crab-theta-weighted MC radial profile with a two-component circular Gaussian radial mixture and defines <code>r_opt</code> by solving the fitted CDF at <code>{TARGET_CONTAINMENT:.15f}</code>. The equivalent sigma below is only <code>r_opt / 1.58</code> for comparison; it is not used to define the aperture.</p>
  {table(["cell", "Nhit", "predE", "A", "sigma1", "sigma2", "sigma_eq", "r_opt", "r/r_rayleigh", "empirical containment", "chi2/ndof", "quality", "fallback/borrow reason"], double_rayleigh_focus_rows(runs))}
  {table(["method", "cell", "Nhit", "predE", "fallback or warning reason"], fallback_quality_rows(runs))}
  <h3>Observed-Data Aperture Cells</h3>
  <p>The observed-data branch uses the pedestal-subtracted observed Crab excess profile only when the profile passes the reliability, positive-total, and radius-divergence gates. Cells without an accepted data profile, and cells handled by the existing psfborrow policy, are explicitly listed as fallback below.</p>
  {table(["cell", "Nhit", "predE", "r_opt_obs", "containment", "r_obs/r_rayleigh", "r_obs/r_mcq", "positive total"], observed_data_used_rows(runs))}
  {table(["cell", "Nhit", "predE", "r_opt_obs", "final r_opt", "data containment", "final containment", "fallback reason"], observed_data_fallback_rows(runs))}
  {two1d_radius_diagnostic(runs)}
  {figure(ASSET_DIR / "v5_psf_radius_shape_scatter.png", "PSF radius versus shape scale", "The two-1D branch uses sigma_eff = sqrt((sigma_x^2 + sigma_y^2)/2).")}

  <h2>Weighted PSF Profiles</h2>
  <p>The overlay below uses the Stage B <code>profile_density</code> arrays for the drop4 fit cells. In the five per-method grids, the blue steps are always the exact Rayleigh-baseline Crab-theta-weighted MC profile. Every matching cell uses the same 0-5 deg x range and the same Rayleigh-reference y scale, and the blue steps are drawn above the model curves so they remain visually identical. The observed_data grid adds accepted pedestal-subtracted observed excess profiles in teal; dashed vertical lines mark each branch's <code>r_opt</code>, with out-of-range radii annotated instead of expanding the axes.</p>
  {figure(ASSET_DIR / "v5_psf_weighted_profile_overlay.png", "Weighted PSF profile overlay", f"Each small panel is one drop4 fit cell, with {len(METHODS)} PSF aperture branches overlaid.")}
  {psf_profile_grid_figures(runs)}

  <h2>Stage F Fit And Pulls</h2>
  {table(["run", "LogPar phi0", "alpha", "beta", "chi2 / ndof", "chi2/ndof", "max abs pull", "PL phi0", "PL gamma"], stage_f_table_rows(runs))}
  {table(focus_cell_pull_headers(), focus_cell_pull_rows(runs))}
  {figure(ASSET_DIR / "v5_psf_cell_pull_grid.png", "Cell pull comparison", f"Stage F LogPar conservative pulls for the {len(METHODS)} v5 PSF aperture branches.")}

  <h2>Stage G SED</h2>
  {figure(ASSET_DIR / "v5_psf_sed_overlay.png", "SED overlay", f"{len(METHODS)} v5 Stage G Nhit points and LogPar curves, plus the v4 aperture baseline, official pass5 WCDA points, and external references when available.")}
  {figure(ASSET_DIR / "v5_psf_sed_flux_ratio_to_v4.png", "SED flux ratio relative to v4", "The ratio uses matching Nhit group labels and the v4 aperture-conditioned Stage G result as denominator.")}

  <h2>Official Pass5 Forward Fold</h2>
  <p>The official pass5 spectrum is folded through each branch's own aperture-conditioned Stage A response and Stage F exposure. Full tables are written to <code>{esc(rel(ASSET_DIR / "official_pass5_forward_fold_summary.csv"))}</code>, <code>{esc(rel(ASSET_DIR / "official_pass5_forward_fold_nhit_summary.csv"))}</code>, and <code>{esc(rel(ASSET_DIR / "official_pass5_forward_fold_cell_counts.csv"))}</code>.</p>
  {table(low_nhit_headers(), low_nhit_rows(nhit_rows))}
  {figure(ASSET_DIR / "v5_psf_official_forward_fold_nhit.png", "Official pass5 obs/exp by Nhit", "This is the requested official pass5 forward-fold observed/expected comparison.")}

  <h2>Observed Excess PSF Diagnostic</h2>
  <p>This legacy diagnostic uses the same unrounded Rayleigh-contract target containment, <code>{TARGET_CONTAINMENT:.6f}</code>, and computes an observed-excess <code>r715</code> from positive radial excess shells in the existing v4 empirical-PSF profiles. The observed_data branch above uses the stricter pedestal-subtracted version with fallback gates as its Stage B aperture input.</p>
  {table(["cell", "Nhit", "predE", "observed r715", "MC quantile r715", "obs/MC r715", "Rayleigh r_opt"], observed_excess_r715_table_rows(observed_r715_rows))}
  {figure(ASSET_DIR / "v5_observed_excess_r715_diagnostic.png", "Observed excess r715 diagnostic", "Legacy observed Crab excess r715 cross-check before the stricter observed_data branch gates are applied.")}

  <h2>Contract Validation</h2>
  {table(["method", "Stage A response type", "aperture mode", "Stage E containment=1"], metadata_validation_rows(runs))}

  <h2>Artifacts</h2>
  <p>Stage B/A/D/E/F/G outputs live under <code>apply/output/stage_*_v5_psf_compare</code>. This report summary JSON is <code>{esc(rel(ASSET_DIR / "v5_psf_comparison_summary.json"))}</code>.</p>
</main>
</body>
</html>
"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
