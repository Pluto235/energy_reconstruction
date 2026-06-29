#!/usr/bin/env python
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_HTML = REPO_ROOT / "apply/report/crab_sed_v5_cell_selection_comparison_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v5-cell-selection"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
RESPONSE_NPZ = REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz"
SIGNAL_NPZ = (
    REPO_ROOT
    / "apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz"
)
STAGE_F_OUT = REPO_ROOT / "apply/output/stage_f_v5_cell_selection/runs"
STAGE_G_OUT = REPO_ROOT / "apply/output/stage_g_v5_cell_selection/runs"
M2_TO_CM2 = 1.0e4

SELECTORS: Dict[str, Dict[str, Any]] = {
    "strict20": {
        "label": "strict20",
        "role": "conservative quality bracket",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_cellsel_strict20.csv",
        "run_id": "v5_cellsel_strict20",
    },
    "baseline26": {
        "label": "baseline26",
        "role": "nominal reference, current v4 baseline",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_cellsel_baseline26.csv",
        "run_id": "v5_cellsel_baseline26",
    },
    "loose36": {
        "label": "loose36",
        "role": "stress test / systematic bracket",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_cellsel_loose36.csv",
        "run_id": "v5_cellsel_loose36",
    },
}


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
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def finite_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


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


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def selector_rows(path: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    if rows and len(rows) != 84:
        raise ValueError(f"{path} must contain 84 data rows, got {len(rows)}")
    return rows


def included_cell_ids(rows: Sequence[Dict[str, str]]) -> List[int]:
    return [int(row["cell_id"]) for row in rows if truthy(row.get("include"))]


def fit_path(run_id: str, suffix: str) -> Path:
    return STAGE_F_OUT / run_id / suffix


def stage_g_path(run_id: str, suffix: str) -> Path:
    return STAGE_G_OUT / run_id / suffix


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


def pass5_points() -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(PASS5_CSV)
    energy = []
    flux = []
    for row in rows:
        e = finite_float(row.get("energy_tev"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        if e is not None and dnde is not None and e > 0 and dnde > 0:
            energy.append(e)
            flux.append(e * e * dnde)
    return np.asarray(energy, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def pass5_dnde_points() -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(PASS5_CSV)
    energy = []
    flux = []
    for row in rows:
        e = finite_float(row.get("energy_tev"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        if e is not None and dnde is not None and e > 0 and dnde > 0:
            energy.append(e)
            flux.append(dnde)
    return np.asarray(energy, dtype=np.float64), np.asarray(flux, dtype=np.float64)


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
    selector: str,
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
        "selector": selector,
        "spectrum": "official_pass5",
        "nhit_bin": nhit_label,
        "cells": int(np.count_nonzero(mask)),
        "N_on": float(np.nansum(n_on[mask])),
        "B_on": float(np.nansum(b_on[mask])),
        "excess": total_excess,
        "official_expected_counts": total_expected,
        "observed_over_expected": total_excess / total_expected if total_expected > 0 else "",
        "N_on_over_B_on": float(np.nansum(n_on[mask])) / float(np.nansum(b_on[mask]))
        if float(np.nansum(b_on[mask])) > 0
        else "",
    }


def build_forward_fold_tables(runs: Dict[str, Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    response = load_npz(RESPONSE_NPZ)

    summary_rows: List[Dict[str, object]] = []
    nhit_rows: List[Dict[str, object]] = []
    cell_rows: List[Dict[str, object]] = []
    for name, payload in runs.items():
        stage_f = payload["fit_npz"]  # type: ignore[assignment]
        if not stage_f:
            continue
        expected = official_expected_counts(response, stage_f)  # type: ignore[arg-type]
        cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)
        nhit = np.asarray(stage_f["nhit_bin"]).astype(str)
        pred = np.asarray(stage_f["predE_bin"]).astype(str)
        excess = np.asarray(stage_f["excess"], dtype=np.float64)
        err = np.asarray(stage_f["excess_err_conservative"], dtype=np.float64)
        mask = np.ones(cell_id.shape, dtype=bool)
        summary_rows.append(sum_rows_for_mask(name, mask, stage_f, expected, "all"))  # type: ignore[arg-type]
        for label in sorted(set(nhit.tolist()), key=interval_key):
            label_mask = mask & (nhit == label)
            if np.any(label_mask):
                nhit_rows.append(sum_rows_for_mask(name, label_mask, stage_f, expected, label))  # type: ignore[arg-type]
        for idx, cid in enumerate(cell_id):
            pull = (excess[idx] - expected[idx]) / err[idx] if err[idx] > 0 else float("nan")
            cell_rows.append(
                {
                    "selector": name,
                    "spectrum": "official_pass5",
                    "cell_id": int(cid),
                    "nhit_bin": str(nhit[idx]),
                    "predE_bin": str(pred[idx]),
                    "excess": float(excess[idx]),
                    "official_expected_counts": float(expected[idx]),
                    "observed_over_expected": float(excess[idx] / expected[idx]) if expected[idx] > 0 else "",
                    "excess_minus_expected": float(excess[idx] - expected[idx]),
                    "pull_conservative": float(pull) if math.isfinite(pull) else "",
                }
            )

    write_csv(
        ASSET_DIR / "official_pass5_forward_fold_summary.csv",
        summary_rows,
        [
            "selector",
            "spectrum",
            "nhit_bin",
            "cells",
            "N_on",
            "B_on",
            "excess",
            "official_expected_counts",
            "observed_over_expected",
            "N_on_over_B_on",
        ],
    )
    write_csv(
        ASSET_DIR / "official_pass5_forward_fold_nhit_summary.csv",
        nhit_rows,
        [
            "selector",
            "spectrum",
            "nhit_bin",
            "cells",
            "N_on",
            "B_on",
            "excess",
            "official_expected_counts",
            "observed_over_expected",
            "N_on_over_B_on",
        ],
    )
    write_csv(
        ASSET_DIR / "official_pass5_forward_fold_cell_counts.csv",
        cell_rows,
        [
            "selector",
            "spectrum",
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "excess",
            "official_expected_counts",
            "observed_over_expected",
            "excess_minus_expected",
            "pull_conservative",
        ],
    )
    return summary_rows, nhit_rows, cell_rows


def load_selector_payloads() -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for name, config in SELECTORS.items():
        rows = selector_rows(Path(config["selector"]))
        included = included_cell_ids(rows)
        out[name] = {
            "name": name,
            "label": config["label"],
            "role": config["role"],
            "selector": Path(config["selector"]),
            "run_id": config["run_id"],
            "rows": rows,
            "included_cell_ids": included,
            "n_rows": len(rows),
            "n_included": len(included),
        }
    return out


def load_run_payload(name: str, selector_payload: Dict[str, object]) -> Dict[str, object]:
    run_id = str(selector_payload["run_id"])
    fit_npz_path = fit_path(run_id, f"fit_{run_id}.npz")
    fit_meta_path = fit_path(run_id, f"fit_{run_id}_metadata.json")
    fit_summary_path = fit_path(run_id, f"fit_{run_id}_summary.csv")
    stage_g_summary_path = stage_g_path(run_id, f"sed_points_{run_id}_summary.csv")
    stage_g_meta_path = stage_g_path(run_id, f"sed_points_{run_id}_metadata.json")
    fit_meta = read_json(fit_meta_path)
    fit_npz = load_npz(fit_npz_path)
    stage_g_rows = read_csv_rows(stage_g_summary_path)
    stage_g_meta = read_json(stage_g_meta_path)
    logpar = fit_params_from_meta(fit_meta, "logpar")
    pl = fit_params_from_meta(fit_meta, "pl")
    return {
        **selector_payload,
        "fit_npz_path": fit_npz_path,
        "fit_meta_path": fit_meta_path,
        "fit_summary_path": fit_summary_path,
        "stage_g_summary_path": stage_g_summary_path,
        "stage_g_meta_path": stage_g_meta_path,
        "fit_meta": fit_meta,
        "fit_npz": fit_npz,
        "stage_g_rows": stage_g_rows,
        "stage_g_meta": stage_g_meta,
        "logpar": logpar,
        "pl": pl,
        "max_abs_pull": max_abs_pull(fit_npz),
        "status": "complete" if fit_meta_path.exists() and stage_g_summary_path.exists() else "pending",
    }


def group_stage_g_rows(rows: Sequence[Dict[str, str]], grouping: str) -> List[Dict[str, str]]:
    return sorted([row for row in rows if row.get("grouping") == grouping], key=lambda row: interval_key(row.get("group_label")))


def plot_cell_grid(selector_payloads: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    all_rows = next(iter(selector_payloads.values()))["rows"]  # type: ignore[index]
    nhit_labels = sorted({row["nhit_bin"] for row in all_rows}, key=interval_key)
    pred_labels = sorted({row["predE_bin"] for row in all_rows}, key=interval_key)
    pred_index = {label: i for i, label in enumerate(pred_labels)}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), dpi=160, sharex=True, sharey=True)
    colors = {"strict20": "#2563eb", "baseline26": "#111827", "loose36": "#dc2626"}
    for ax, (name, payload) in zip(axes, selector_payloads.items()):
        included = set(payload["included_cell_ids"])  # type: ignore[arg-type]
        for row in payload["rows"]:  # type: ignore[index]
            x = pred_index[row["predE_bin"]]
            y = nhit_labels.index(row["nhit_bin"])
            cid = int(row["cell_id"])
            face = colors[name] if cid in included else "#f3f4f6"
            edge = "#374151" if cid in included else "#d1d5db"
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor=face, edgecolor=edge, lw=0.8))
            if cid in included:
                ax.text(x, y, str(cid), ha="center", va="center", fontsize=7.5, color="white")
            else:
                ax.text(x, y, str(cid), ha="center", va="center", fontsize=6.5, color="#6b7280")
        ax.set_title(f"{name} ({payload['n_included']} cells)")
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(nhit_labels)))
        ax.set_yticklabels(nhit_labels, fontsize=8)
        ax.set_xlim(-0.6, len(pred_labels) - 0.4)
        ax.set_ylim(len(nhit_labels) - 0.4, -0.6)
        ax.grid(False)
    axes[0].set_ylabel("Nhit bin")
    for ax in axes:
        ax.set_xlabel("predicted-energy bin")
    fig.suptitle("v5 Cell Selection Selector Grid")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_sed_overlay(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=160)
    e_pass5, y_pass5 = pass5_points()
    if e_pass5.size:
        ax.plot(e_pass5, y_pass5, "o", color="#111827", ms=5.5, label="official pass5 WCDA")

    external_path = next(
        (
            payload["stage_g_meta_path"].parent / "external_crab_sed_references.csv"  # type: ignore[index,union-attr]
            for payload in runs.values()
            if Path(payload["stage_g_meta_path"]).parent.joinpath("external_crab_sed_references.csv").exists()
        ),
        None,
    )
    if external_path is not None:
        external_rows = read_csv_rows(Path(external_path))
        for dataset, marker, color in [
            ("magic_joint_crab", ".", "#7c3aed"),
            ("hess_2024_stereo", ".", "#059669"),
            ("hawc_2019_nn", ".", "#b45309"),
        ]:
            selected = [row for row in external_rows if row.get("dataset") == dataset and str(row.get("is_upper_limit")).lower() != "true"]
            x = [finite_float(row.get("energy_tev")) for row in selected]
            y = [finite_float(row.get("e2_dnde")) for row in selected]
            good = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and a > 0 and b > 0]
            if good:
                ax.scatter([a for a, _ in good], [b for _, b in good], s=13, marker=marker, color=color, alpha=0.42, label=dataset)

    styles = {
        "strict20": ("#2563eb", "s"),
        "baseline26": ("#111827", "o"),
        "loose36": ("#dc2626", "^"),
    }
    for name, payload in runs.items():
        color, marker = styles[name]
        for grouping, alpha, fill in [("nhit", 0.95, color), ("predE", 0.62, "none")]:
            rows = group_stage_g_rows(payload["stage_g_rows"], grouping)  # type: ignore[arg-type]
            x = [finite_float(row.get("effective_energy_tev")) for row in rows]
            y = [finite_float(row.get("E2_dnde")) for row in rows]
            dy = [finite_float(row.get("E2_dnde_err")) or 0.0 for row in rows]
            good_idx = [i for i, (a, b) in enumerate(zip(x, y)) if a is not None and b is not None and a > 0 and b > 0]
            if good_idx:
                ax.errorbar(
                    [x[i] for i in good_idx],
                    [y[i] for i in good_idx],
                    yerr=[dy[i] for i in good_idx],
                    fmt=marker,
                    ms=5.2,
                    lw=1.0,
                    color=color,
                    ecolor=color,
                    alpha=alpha,
                    markerfacecolor=fill,
                    capsize=2.2,
                    label=f"{name} {grouping}",
                )
        params = fit_parameters(payload["fit_meta"], "logpar")  # type: ignore[arg-type]
        if {"phi0", "alpha", "beta"} <= set(params):
            x_curve = np.logspace(np.log10(0.3), np.log10(90.0), 220)
            ax.plot(x_curve, e2_curve(x_curve, params, "logpar"), color=color, lw=1.4, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Crab SED: v5 cell selection comparison")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.2, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_pull_grid(runs: Dict[str, Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8), dpi=160, sharex=True, sharey=True)
    all_nhit = sorted(
        {
            str(value)
            for payload in runs.values()
            for value in np.asarray(payload.get("fit_npz", {}).get("nhit_bin", []), dtype=str)  # type: ignore[union-attr]
        },
        key=interval_key,
    )
    all_pred = sorted(
        {
            str(value)
            for payload in runs.values()
            for value in np.asarray(payload.get("fit_npz", {}).get("predE_bin", []), dtype=str)  # type: ignore[union-attr]
        },
        key=interval_key,
    )
    pred_index = {label: idx for idx, label in enumerate(all_pred)}
    for ax, (name, payload) in zip(axes, runs.items()):
        fit_npz = payload["fit_npz"]  # type: ignore[assignment]
        cell_ids = np.asarray(fit_npz.get("cell_id", []), dtype=np.int64)
        nhit = np.asarray(fit_npz.get("nhit_bin", []), dtype=str)
        pred = np.asarray(fit_npz.get("predE_bin", []), dtype=str)
        pulls = np.asarray(fit_npz.get("logpar_conservative_pull", []), dtype=np.float64)
        sc = None
        if cell_ids.size:
            x = np.asarray([pred_index[str(v)] for v in pred], dtype=np.float64)
            y = np.asarray([all_nhit.index(str(v)) for v in nhit], dtype=np.float64)
            sc = ax.scatter(x, y, c=pulls, cmap="coolwarm", vmin=-6, vmax=6, s=185, edgecolor="#111827", lw=0.45)
            for xi, yi, cid in zip(x, y, cell_ids):
                ax.text(xi, yi, str(int(cid)), ha="center", va="center", fontsize=7.5, color="#111827")
        ax.set_title(f"{name} LogPar pulls")
        ax.set_xticks(range(len(all_pred)))
        ax.set_xticklabels(all_pred, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(all_nhit)))
        ax.set_yticklabels(all_nhit, fontsize=8)
        ax.set_xlim(-0.6, len(all_pred) - 0.4)
        ax.set_ylim(len(all_nhit) - 0.4, -0.6)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Nhit bin")
    for ax in axes:
        ax.set_xlabel("predicted-energy bin")
    if sc is not None:
        cax = fig.add_axes([0.915, 0.25, 0.014, 0.54])
        fig.colorbar(sc, cax=cax, label="(excess - model) / conservative err")
    fig.suptitle("Stage F residual / pull grid")
    fig.subplots_adjust(left=0.055, right=0.89, bottom=0.27, top=0.82, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_forward_fold_ratios(nhit_rows: Sequence[Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    labels = sorted({str(row["nhit_bin"]) for row in nhit_rows}, key=interval_key)
    selectors = ["strict20", "baseline26", "loose36"]
    colors = {"strict20": "#2563eb", "baseline26": "#111827", "loose36": "#dc2626"}
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    for selector in selectors:
        values = []
        for label in labels:
            row = next((r for r in nhit_rows if r["selector"] == selector and r["nhit_bin"] == label), None)
            values.append(finite_float(row.get("observed_over_expected")) if row else float("nan"))
        ax.plot(labels, values, marker="o", lw=1.5, color=colors[selector], label=selector)
    ax.axhline(1.0, color="#6b7280", lw=1.0, ls="--")
    ax.set_ylabel("Stage E excess / official pass5 expected")
    ax.set_xlabel("Nhit bin")
    ax.set_title("Official pass5 forward-fold ratios by Nhit")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def stage_f_table_rows(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    rows = []
    for name, payload in runs.items():
        logpar = payload["logpar"]  # type: ignore[assignment]
        pl = payload["pl"]  # type: ignore[assignment]
        lp_params = logpar.get("parameters") if isinstance(logpar.get("parameters"), dict) else {}
        pl_params = pl.get("parameters") if isinstance(pl.get("parameters"), dict) else {}
        rows.append(
            [
                name,
                str(payload["n_included"]),
                fmt(lp_params.get("phi0"), 5),
                fmt(lp_params.get("alpha"), 5),
                fmt(lp_params.get("beta"), 5),
                f"{fmt(logpar.get('chi2'), 4)} / {fmt_int(logpar.get('ndof'))}",
                fmt(chi2_over_ndof(logpar), 4),
                fmt(payload.get("max_abs_pull"), 4),
                fmt(pl_params.get("phi0"), 5),
                fmt(pl_params.get("gamma"), 5),
            ]
        )
    return rows


def low_nhit_rows(nhit_rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    focus = ["[125,200)", "[200,300)", "[300,500)"]
    rows = []
    for label in focus:
        item = [label]
        for selector in ["strict20", "baseline26", "loose36"]:
            row = next((r for r in nhit_rows if r["selector"] == selector and r["nhit_bin"] == label), None)
            item.extend(
                [
                    fmt(row.get("observed_over_expected") if row else None, 4),
                    fmt(row.get("excess") if row else None, 5),
                    fmt(row.get("official_expected_counts") if row else None, 5),
                ]
            )
        rows.append(item)
    return rows


def systematic_shift_rows(runs: Dict[str, Dict[str, object]], nhit_rows: Sequence[Dict[str, object]], summary_rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    baseline = runs["baseline26"]
    base_params = fit_parameters(baseline["fit_meta"], "logpar")  # type: ignore[arg-type]
    base_total = next(row for row in summary_rows if row["selector"] == "baseline26")
    base_total_ratio = finite_float(base_total.get("observed_over_expected"))
    base_low = {
        row["nhit_bin"]: finite_float(row.get("observed_over_expected"))
        for row in nhit_rows
        if row["selector"] == "baseline26" and row["nhit_bin"] in {"[125,200)", "[200,300)", "[300,500)"}
    }
    rows = []
    for name in ["strict20", "loose36"]:
        params = fit_parameters(runs[name]["fit_meta"], "logpar")  # type: ignore[arg-type]
        total = next(row for row in summary_rows if row["selector"] == name)
        total_ratio = finite_float(total.get("observed_over_expected"))
        low_parts = []
        for label in ["[125,200)", "[200,300)", "[300,500)"]:
            row = next((r for r in nhit_rows if r["selector"] == name and r["nhit_bin"] == label), None)
            ratio = finite_float(row.get("observed_over_expected")) if row else None
            base_ratio = base_low.get(label)
            shift = None if ratio is None or base_ratio in {None, 0.0} else ratio / base_ratio - 1.0
            low_parts.append(f"{label}: {fmt(shift, 4)}")
        phi_shift = (
            params.get("phi0") / base_params.get("phi0") - 1.0
            if params.get("phi0") is not None and base_params.get("phi0") not in {None, 0.0}
            else None
        )
        rows.append(
            [
                name,
                fmt(phi_shift, 5),
                fmt(params.get("alpha") - base_params.get("alpha") if "alpha" in params and "alpha" in base_params else None, 5),
                fmt(params.get("beta") - base_params.get("beta") if "beta" in params and "beta" in base_params else None, 5),
                fmt(total_ratio / base_total_ratio - 1.0 if total_ratio is not None and base_total_ratio not in {None, 0.0} else None, 5),
                "<br>".join(esc(part) for part in low_parts),
            ]
        )
    return rows


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], *, classes: str = "") -> str:
    cls = f' class="{classes}"' if classes else ""
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
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


def selector_definition_table(payloads: Dict[str, Dict[str, object]]) -> str:
    rows = []
    for name, payload in payloads.items():
        cells = ",".join(str(v) for v in payload["included_cell_ids"])  # type: ignore[index]
        rows.append([name, esc(payload["role"]), str(payload["n_included"]), f"<code>{esc(cells)}</code>"])
    return table(["selector", "role", "cells", "included cell ids"], rows)


def stage_g_metadata_validation(runs: Dict[str, Dict[str, object]]) -> List[List[str]]:
    rows = []
    for name, payload in runs.items():
        meta = payload["stage_g_meta"]  # type: ignore[assignment]
        validation = meta.get("validation") if isinstance(meta.get("validation"), dict) else {}
        inputs = meta.get("inputs") if isinstance(meta.get("inputs"), dict) else {}
        expected_stage_f = str(payload["fit_meta_path"])  # type: ignore[arg-type]
        observed_stage_f = str(inputs.get("stage_f_metadata_json", ""))
        rows.append(
            [
                name,
                esc(validation.get("stage_f_run_id")),
                "yes" if observed_stage_f == expected_stage_f else f"mismatch: {esc(observed_stage_f)}",
                str(len(validation.get("required_cell_ids", [])) if isinstance(validation.get("required_cell_ids"), list) else "n/a"),
            ]
        )
    return rows


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    selectors = load_selector_payloads()
    runs = {name: load_run_payload(name, payload) for name, payload in selectors.items()}
    summary_rows, nhit_forward_rows, cell_forward_rows = build_forward_fold_tables(runs)

    plot_cell_grid(selectors, ASSET_DIR / "v5_cell_selection_grid.png")
    plot_sed_overlay(runs, ASSET_DIR / "v5_cell_selection_sed_overlay.png")
    plot_pull_grid(runs, ASSET_DIR / "v5_cell_selection_pull_grid.png")
    plot_forward_fold_ratios(nhit_forward_rows, ASSET_DIR / "v5_cell_selection_official_forward_fold_nhit.png")

    write_json(
        ASSET_DIR / "v5_cell_selection_comparison_summary.json",
        {
            "selectors": {
                name: {
                    "role": payload["role"],
                    "selector_csv": str(payload["selector"]),
                    "included_cell_ids": payload["included_cell_ids"],
                    "n_included": payload["n_included"],
                }
                for name, payload in selectors.items()
            },
            "runs": {
                name: {
                    "run_id": payload["run_id"],
                    "status": payload["status"],
                    "fit_metadata": str(payload["fit_meta_path"]),
                    "stage_g_metadata": str(payload["stage_g_meta_path"]),
                    "logpar": payload["logpar"],
                    "pl": payload["pl"],
                    "max_abs_pull": payload["max_abs_pull"],
                }
                for name, payload in runs.items()
            },
            "official_forward_fold": {
                "summary_csv": str(ASSET_DIR / "official_pass5_forward_fold_summary.csv"),
                "nhit_summary_csv": str(ASSET_DIR / "official_pass5_forward_fold_nhit_summary.csv"),
                "cell_counts_csv": str(ASSET_DIR / "official_pass5_forward_fold_cell_counts.csv"),
                "method": "official pass5 dN/dE log-log piecewise interpolation folded through v4 aperture-conditioned Stage A and Stage E containment=1 signal exposure",
            },
        },
    )

    total_forward = {
        row["selector"]: row for row in summary_rows if row["nhit_bin"] == "all" and row["spectrum"] == "official_pass5"
    }
    main_rows = []
    for name, payload in runs.items():
        logpar = payload["logpar"]  # type: ignore[assignment]
        params = logpar.get("parameters") if isinstance(logpar.get("parameters"), dict) else {}
        total = total_forward.get(name, {})
        low_125 = next((r for r in nhit_forward_rows if r["selector"] == name and r["nhit_bin"] == "[125,200)"), {})
        main_rows.append(
            [
                name + (" (nominal)" if name == "baseline26" else ""),
                str(payload["n_included"]),
                fmt(params.get("phi0"), 5),
                fmt(params.get("alpha"), 5),
                fmt(params.get("beta"), 5),
                fmt(chi2_over_ndof(logpar), 4),
                fmt(payload.get("max_abs_pull"), 4),
                fmt(total.get("observed_over_expected"), 4),
                fmt(low_125.get("observed_over_expected"), 4),
            ]
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
    .lede { font-size: 15px; max-width: 900px; }
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
  <title>Crab SED v5 Cell Selection Comparison</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Crab SED v5 Cell Selection Comparison</h1>
  <p class="lede">This report holds the v4 physics contract fixed: aperture-conditioned Stage A response, Stage E annulus-normalized containment=1 signal, and identical Stage F/G fitting logic. Only the fit-cell selector changes. <code>baseline26</code> is the nominal reference; <code>strict20</code> and <code>loose36</code> are selector-sensitivity brackets.</p>

  <div class="grid">
    <div class="metric"><div class="label">nominal selector</div><div class="value">baseline26</div></div>
    <div class="metric"><div class="label">Stage F/G runs</div><div class="value">{sum(1 for p in runs.values() if p["status"] == "complete")}/3</div></div>
    <div class="metric"><div class="label">official fold</div><div class="value">pass5</div></div>
  </div>

  <h2>Selector Definitions</h2>
  {selector_definition_table(selectors)}
  {figure(ASSET_DIR / "v5_cell_selection_grid.png", "Selector cell grid", "Included cells are colored and annotated by cell id. The grid uses the same 84-row cell contract as the current v4 baseline selector.")}

  <h2>Main Comparison</h2>
  {table(["selector", "cells", "LogPar phi0", "alpha", "beta", "chi2/ndof", "max abs pull", "total obs/pass5", "[125,200) obs/pass5"], main_rows)}

  <h2>Stage F Fits</h2>
  {table(["selector", "cells", "LogPar phi0", "alpha", "beta", "chi2 / ndof", "chi2/ndof", "max abs pull", "PL phi0", "PL gamma"], stage_f_table_rows(runs))}
  {figure(ASSET_DIR / "v5_cell_selection_pull_grid.png", "Stage F residual / pull grid", "Cell-level LogPar conservative pulls for each selector. Loose36 is a stress-test bracket, not a promoted baseline candidate.")}

  <h2>Official Pass5 Forward Fold</h2>
  <p>The official pass5 spectrum is folded through the aperture-conditioned v4 Stage A response and Stage E containment=1 exposure, then summed over each selector. The total, Nhit, and cell-level tables are written to <code>{esc(rel(ASSET_DIR / "official_pass5_forward_fold_summary.csv"))}</code>, <code>{esc(rel(ASSET_DIR / "official_pass5_forward_fold_nhit_summary.csv"))}</code>, and <code>{esc(rel(ASSET_DIR / "official_pass5_forward_fold_cell_counts.csv"))}</code>.</p>
  {table(["Nhit", "strict20 obs/pass5", "strict20 excess", "strict20 pass5", "baseline26 obs/pass5", "baseline26 excess", "baseline26 pass5", "loose36 obs/pass5", "loose36 excess", "loose36 pass5"], low_nhit_rows(nhit_forward_rows))}
  {figure(ASSET_DIR / "v5_cell_selection_official_forward_fold_nhit.png", "Official pass5 forward-fold ratios by Nhit", "Low-Nhit stability is read directly from the obs/pass5 ratios for [125,200), [200,300), and [300,500).")}

  <h2>Stage G SED Overlay</h2>
  {figure(ASSET_DIR / "v5_cell_selection_sed_overlay.png", "SED overlay", "Three selector Stage G SED points and LogPar curves are shown with official pass5 and external Crab references exported by Stage G.")}

  <h2>Systematic Shifts</h2>
  {table(["selector", "flux normalization shift", "alpha shift", "beta shift", "total obs/pass5 shift", "low-Nhit obs/pass5 shifts"], systematic_shift_rows(runs, nhit_forward_rows, summary_rows))}

  <h2>Stage G Metadata Check</h2>
  {table(["selector", "Stage F run id in Stage G", "Stage F metadata path matches", "required cells"], stage_g_metadata_validation(runs))}

  <h2>Artifacts</h2>
  <p>Stage F outputs live under <code>apply/output/stage_f_v5_cell_selection/runs/v5_cellsel_*</code>. Stage G outputs live under <code>apply/output/stage_g_v5_cell_selection/runs/v5_cellsel_*</code>. This report summary JSON is <code>{esc(rel(ASSET_DIR / "v5_cell_selection_comparison_summary.json"))}</code>.</p>
</main>
</body>
</html>
"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
