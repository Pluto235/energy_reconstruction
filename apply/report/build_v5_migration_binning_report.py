#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "apply/report/crab_sed_v5_migration_binned_comparison_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v5-migration-binning"

RESPONSE_NPZ = (
    REPO_ROOT
    / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz"
)
SIGNAL_NPZ = (
    REPO_ROOT
    / "apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz"
)
V4_REBIN_GROUPS_CSV = REPO_ROOT / "apply/report/assets/v4-rebin-diagnostics/candidate_rebin_groups.csv"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"

PIVOT_TEV = 3.0
REFERENCE_PHI0 = 2.114e-12
REFERENCE_GAMMA = 2.69
QUADRATURE_POINTS = 64
M2_TO_CM2 = 1.0e4

V5_GROUPS = [
    ("v5_01", "1+2+3", [1, 2, 3]),
    ("v5_02", "4", [4]),
    ("v5_03", "14+15+16", [14, 15, 16]),
    ("v5_04", "17", [17]),
    ("v5_05", "26+27+28+29", [26, 27, 28, 29]),
    ("v5_06", "30", [30]),
    ("v5_07", "39+40+41+42", [39, 40, 41, 42]),
    ("v5_08", "43", [43]),
    ("v5_09", "52+53+54+55", [52, 53, 54, 55]),
    ("v5_10", "65+66+67", [65, 66, 67]),
    ("v5_11", "68+69", [68, 69]),
    ("v5_12", "81+82+83", [81, 82, 83]),
]

CONSERVATIVE_GROUPS = [
    ("c7_01", "1+2+3", [1, 2, 3]),
    ("c7_02", "14+15+16", [14, 15, 16]),
    ("c7_03", "26+27+28+29+30", [26, 27, 28, 29, 30]),
    ("c7_04", "40+41+42", [40, 41, 42]),
    ("c7_05", "52+53+54+55", [52, 53, 54, 55]),
    ("c7_06", "65+66+67+68+69", [65, 66, 67, 68, 69]),
    ("c7_07", "81+82+83", [81, 82, 83]),
]

FINE30_IDS = sorted({cid for _, _, ids in V5_GROUPS for cid in ids})
RESTORED_IDS = {4, 17, 39, 43}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def report_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPORT_PATH.parent))
    except ValueError:
        return str(path)


def format_float(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def parse_interval(label: str) -> tuple[float | None, float | None]:
    text = str(label).strip()
    if text.startswith("[") and text.endswith(")") and "," in text:
        low, high = text[1:-1].split(",", 1)
        return float(low), float(high)
    if text.startswith(">="):
        return float(text[2:]), None
    return None, None


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    return low


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_stage06():
    module_path = REPO_ROOT / "apply/stages/06_fit.py"
    spec = importlib.util.spec_from_file_location("stage06_fit_for_v5_migration_binning", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def pl_flux_tev(energy_tev: np.ndarray, *, phi0: float, gamma: float) -> np.ndarray:
    return float(phi0) * np.power(np.asarray(energy_tev, dtype=np.float64) / PIVOT_TEV, -float(gamma))


def logpar_flux_tev(energy_tev: np.ndarray, *, phi0: float, alpha: float, beta: float) -> np.ndarray:
    ratio = np.asarray(energy_tev, dtype=np.float64) / PIVOT_TEV
    log_ratio = np.log(ratio)
    return float(phi0) * np.exp((-float(alpha) - float(beta) * log_ratio) * log_ratio)


def flux_for_model(energy_tev: np.ndarray, model: str, params: dict[str, float]) -> np.ndarray:
    if model == "pl":
        return pl_flux_tev(energy_tev, phi0=params["phi0"], gamma=params["gamma"])
    if model == "logpar":
        return logpar_flux_tev(energy_tev, phi0=params["phi0"], alpha=params["alpha"], beta=params["beta"])
    raise ValueError(f"Unsupported model: {model}")


def integrate_flux_bins(
    loge_edges: np.ndarray,
    *,
    model: str,
    params: dict[str, float],
    quadrature_points: int = QUADRATURE_POINTS,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    out = np.zeros(loge_edges.size - 1, dtype=np.float64)
    for idx, (lo, hi) in enumerate(zip(loge_edges[:-1], loge_edges[1:])):
        xs = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        energy_tev = np.power(10.0, xs) / 1000.0
        flux = flux_for_model(energy_tev, model, params)
        integrand = flux * math.log(10.0) * energy_tev
        out[idx] = 0.5 * (hi - lo) * float(np.sum(weights * integrand))
    return out


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    x = x[mask]
    w = w[mask]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return float(np.interp(float(q), cdf, x))


def energy_quantile_row(loge_edges: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    p16_log = weighted_quantile(centers, weights, 0.16)
    p50_log = weighted_quantile(centers, weights, 0.50)
    p84_log = weighted_quantile(centers, weights, 0.84)
    sigma68 = 0.5 * (p84_log - p16_log) if math.isfinite(p16_log + p84_log) else float("nan")
    return {
        "logE_p16": p16_log,
        "logE_p50": p50_log,
        "logE_p84": p84_log,
        "sigma68_dex": sigma68,
        "E_p16_TeV": math.pow(10.0, p16_log) / 1000.0 if math.isfinite(p16_log) else float("nan"),
        "E_p50_TeV": math.pow(10.0, p50_log) / 1000.0 if math.isfinite(p50_log) else float("nan"),
        "E_p84_TeV": math.pow(10.0, p84_log) / 1000.0 if math.isfinite(p84_log) else float("nan"),
    }


def true_energy_weights(context: dict[str, Any], a_eff: np.ndarray, *, model: str, params: dict[str, float]) -> np.ndarray:
    flux_integral = integrate_flux_bins(context["loge_edges"], model=model, params=params)
    exposure = np.asarray(context["theta_exposure_sec"], dtype=np.float64)
    return M2_TO_CM2 * flux_integral[None, :] * np.einsum("bet,t->be", a_eff, exposure)


def load_context() -> dict[str, Any]:
    response = load_npz(RESPONSE_NPZ)
    signal = load_npz(SIGNAL_NPZ)
    response_ids = [int(v) for v in response["cell_id"]]
    signal_ids = [int(v) for v in signal["cell_id"]]
    response_index = [response_ids.index(cid) for cid in FINE30_IDS]
    signal_index = [signal_ids.index(cid) for cid in FINE30_IDS]
    return {
        "response": response,
        "signal": signal,
        "response_cell_id": np.asarray(response["cell_id"], dtype=np.int64),
        "cell_id": np.asarray(FINE30_IDS, dtype=np.int64),
        "nhit_bin": np.asarray(response["nhit_bin"])[response_index].astype(str),
        "predE_bin": np.asarray(response["predE_bin"])[response_index].astype(str),
        "a_eff": np.asarray(response["a_eff"], dtype=np.float64)[response_index],
        "loge_edges": np.asarray(response["logE_true_edges"], dtype=np.float64),
        "theta_edges": np.asarray(response["theta_true_edges_deg"], dtype=np.float64),
        "theta_exposure_sec": estimate_theta_exposure_template(signal),
        "containment": np.ones(len(FINE30_IDS), dtype=np.float64),
        "N_on": np.asarray(signal["N_on"], dtype=np.float64)[signal_index],
        "B_on": np.asarray(signal["B_on"], dtype=np.float64)[signal_index],
        "observed": np.asarray(signal["excess"], dtype=np.float64)[signal_index],
        "errors": np.asarray(signal["excess_err_conservative"], dtype=np.float64)[signal_index],
    }


def estimate_theta_exposure_template(signal: dict[str, np.ndarray]) -> np.ndarray:
    # Stage E signal does not carry exposure. Use the latest Stage F output if present.
    candidates = [
        REPO_ROOT
        / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz",
        REPO_ROOT / "apply/output/stage_f_v4_supercell/runs/v4_supercell_stage_f/fit_v4_supercell.npz",
    ]
    for path in candidates:
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if "theta_exposure_sec" in data.files:
                    return np.asarray(data["theta_exposure_sec"], dtype=np.float64).copy()
    raise FileNotFoundError("No Stage F theta_exposure_sec template found")


def fit_models(context: dict[str, Any], stage06: Any, *, a_eff: np.ndarray, observed: np.ndarray, errors: np.ndarray, label: str) -> dict[str, Any]:
    pl = stage06.fit_model(
        model_name="pl",
        error_mode=f"{label}_conservative",
        observed=observed,
        errors=errors,
        a_eff_m2=a_eff,
        containment=np.ones(observed.size, dtype=np.float64),
        theta_exposure_sec=context["theta_exposure_sec"],
        loge_edges=context["loge_edges"],
        pivot_tev=PIVOT_TEV,
        quadrature_points=QUADRATURE_POINTS,
        start_gamma=REFERENCE_GAMMA,
        start_phi0=REFERENCE_PHI0,
    )
    logpar = stage06.fit_model(
        model_name="logpar",
        error_mode=f"{label}_conservative",
        observed=observed,
        errors=errors,
        a_eff_m2=a_eff,
        containment=np.ones(observed.size, dtype=np.float64),
        theta_exposure_sec=context["theta_exposure_sec"],
        loge_edges=context["loge_edges"],
        pivot_tev=PIVOT_TEV,
        quadrature_points=QUADRATURE_POINTS,
        start_gamma=float(pl.parameters.get("gamma", REFERENCE_GAMMA)),
        start_phi0=float(pl.parameters.get("phi0", REFERENCE_PHI0)),
    )
    return {"pl": pl, "logpar": logpar, "preferred": logpar}


def result_payload(result: Any) -> dict[str, Any]:
    return {
        "model": result.model_name,
        "valid": bool(result.valid),
        "parameters": result.parameters,
        "errors": result.errors,
        "chi2": float(result.chi2),
        "ndof": int(result.ndof),
        "chi2_over_ndof": float(result.chi2 / result.ndof) if int(result.ndof) > 0 else None,
        "p_value": result.p_value,
        "max_abs_pull": float(np.nanmax(np.abs(result.pull))),
    }


def ids_to_indices(context: dict[str, Any], ids: list[int]) -> list[int]:
    by_id = {int(cid): idx for idx, cid in enumerate(context["cell_id"])}
    return [by_id[int(cid)] for cid in ids]


def label_span(labels: np.ndarray, indices: list[int]) -> str:
    ordered = sorted({str(labels[idx]) for idx in indices}, key=interval_key)
    if len(ordered) == 1:
        return ordered[0]
    return f"{ordered[0]}..{ordered[-1]}"


def aggregate_groups(context: dict[str, Any], specs: list[tuple[str, str, list[int]]], label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    a_eff_rows: list[np.ndarray] = []
    observed: list[float] = []
    errors: list[float] = []
    n_on: list[float] = []
    b_on: list[float] = []
    for group_id, group_label, ids in specs:
        idx = ids_to_indices(context, ids)
        a_eff_rows.append(np.sum(context["a_eff"][idx], axis=0))
        obs = float(np.sum(context["observed"][idx]))
        err = float(np.sqrt(np.sum(np.square(context["errors"][idx]))))
        rows.append(
            {
                "analysis": label,
                "group_id": group_id,
                "group_label": group_label,
                "cell_ids": ";".join(str(cid) for cid in ids),
                "n_cells": len(ids),
                "restored_cells": ";".join(str(cid) for cid in ids if cid in RESTORED_IDS),
                "nhit_span": label_span(context["nhit_bin"], idx),
                "predE_span": label_span(context["predE_bin"], idx),
                "N_on": float(np.sum(context["N_on"][idx])),
                "B_on": float(np.sum(context["B_on"][idx])),
                "excess": obs,
                "error": err,
            }
        )
        observed.append(obs)
        errors.append(err)
        n_on.append(float(np.sum(context["N_on"][idx])))
        b_on.append(float(np.sum(context["B_on"][idx])))
    return {
        "label": label,
        "rows": rows,
        "a_eff": np.asarray(a_eff_rows, dtype=np.float64),
        "observed": np.asarray(observed, dtype=np.float64),
        "errors": np.asarray(errors, dtype=np.float64),
        "N_on": np.asarray(n_on, dtype=np.float64),
        "B_on": np.asarray(b_on, dtype=np.float64),
    }


def enrich_rows_with_fit(context: dict[str, Any], groups: dict[str, Any], preferred: Any) -> list[dict[str, Any]]:
    weights = true_energy_weights(context, groups["a_eff"], model=preferred.model_name, params=preferred.parameters)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(groups["rows"]):
        q = energy_quantile_row(context["loge_edges"], weights[idx])
        model_counts = float(preferred.model_counts[idx])
        ratio = float(groups["observed"][idx] / model_counts) if model_counts > 0 else float("nan")
        ratio_err = float(groups["errors"][idx] / model_counts) if model_counts > 0 else float("nan")
        e_eff = q["E_p50_TeV"]
        params = dict(preferred.parameters)
        params["phi0"] = float(params["phi0"]) * ratio
        e2 = e_eff * e_eff * float(flux_for_model(np.asarray([e_eff]), preferred.model_name, params)[0])
        e2_err = abs(e2 * ratio_err / ratio) if ratio > 0 else float("nan")
        rows.append(
            {
                **row,
                "E_p16_TeV": q["E_p16_TeV"],
                "E_p50_TeV": q["E_p50_TeV"],
                "E_p84_TeV": q["E_p84_TeV"],
                "sigma68_dex": q["sigma68_dex"],
                "model_counts": model_counts,
                "pull": float(preferred.pull[idx]),
                "N0_ratio": ratio,
                "N0_ratio_err": ratio_err,
                "E2_dnde": e2,
                "E2_dnde_err": e2_err,
            }
        )
    return rows


def fine30_fit(context: dict[str, Any], stage06: Any) -> dict[str, Any]:
    fits = fit_models(
        context,
        stage06,
        a_eff=context["a_eff"],
        observed=context["observed"],
        errors=context["errors"],
        label="fine30",
    )
    return {"fits": fits, "rows": fine30_rows(context, fits["preferred"])}


def fine30_rows(context: dict[str, Any], preferred: Any) -> list[dict[str, Any]]:
    weights = true_energy_weights(context, context["a_eff"], model=preferred.model_name, params=preferred.parameters)
    rows = []
    for idx, cid in enumerate(context["cell_id"]):
        q = energy_quantile_row(context["loge_edges"], weights[idx])
        rows.append(
            {
                "cell_id": int(cid),
                "nhit_bin": str(context["nhit_bin"][idx]),
                "predE_bin": str(context["predE_bin"][idx]),
                "E_p50_TeV": q["E_p50_TeV"],
                "excess": float(context["observed"][idx]),
                "error": float(context["errors"][idx]),
                "model_counts": float(preferred.model_counts[idx]),
                "pull": float(preferred.pull[idx]),
                "restored_cell": int(cid) in RESTORED_IDS,
            }
        )
    return rows


def pass5_points() -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(PASS5_CSV)
    x: list[float] = []
    y: list[float] = []
    for row in rows:
        e = finite_float(row.get("energy_tev"))
        flux = finite_float(row.get("flux_per_tev_cm2_s"))
        if e is not None and flux is not None:
            x.append(e)
            y.append(e * e * flux)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def fit_ratio_to_pass5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    e_pass5, y_pass5 = pass5_points()
    if e_pass5.size == 0:
        return []
    out = []
    for row in rows:
        e = float(row["E_p50_TeV"])
        y = float(row["E2_dnde"])
        ref = float(np.interp(np.log10(e), np.log10(e_pass5), np.log10(y_pass5)))
        ref_y = math.pow(10.0, ref)
        out.append({**row, "pass5_interp_E2": ref_y, "ratio_to_pass5": y / ref_y if ref_y > 0 else float("nan")})
    return out


def group_norm(weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights)
    return weights / total if total > 0 else weights


def overlap_value(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.minimum(group_norm(left), group_norm(right))))


def kernel_rows(context: dict[str, Any], *, params: dict[str, float], model: str = "logpar") -> list[dict[str, Any]]:
    weights = true_energy_weights(context, context["a_eff"], model=model, params=params)
    rows = []
    by_id = {int(cid): idx for idx, cid in enumerate(context["cell_id"])}
    for specs_label, specs in [("v5_migration_12bin", V5_GROUPS), ("conservative_7bin", CONSERVATIVE_GROUPS)]:
        for (left_id, left_label, left_cells), (right_id, right_label, right_cells) in zip(specs[:-1], specs[1:]):
            left_idx = [by_id[cid] for cid in left_cells]
            right_idx = [by_id[cid] for cid in right_cells]
            left_w = np.sum(weights[left_idx], axis=0)
            right_w = np.sum(weights[right_idx], axis=0)
            rows.append(
                {
                    "analysis": specs_label,
                    "left_group": left_label,
                    "right_group": right_label,
                    "left_cells": ";".join(str(cid) for cid in left_cells),
                    "right_cells": ";".join(str(cid) for cid in right_cells),
                    "overlap": overlap_value(left_w, right_w),
                }
            )
    return rows


def html_table(headers: list[str], rows: list[list[Any]], *, cls: str = "compact") -> str:
    out = [f'<table class="{html.escape(cls)}"><thead><tr>']
    out.extend(f"<th>{html.escape(str(header))}</th>" for header in headers)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{html.escape(str(item))}</td>" for item in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def figure(path: Path, title: str, caption: str) -> str:
    return (
        f'<figure class="figure"><img src="{html.escape(report_rel(path))}" alt="{html.escape(title)}">'
        f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</figcaption></figure>"
    )


def plot_sed(v5_rows: list[dict[str, Any]], c7_rows: list[dict[str, Any]], v5_fit: Any, c7_fit: Any, path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=170)
    e_pass5, y_pass5 = pass5_points()
    if e_pass5.size:
        ax.plot(e_pass5, y_pass5, "o", color="#111827", ms=5.2, label="official pass5 WCDA")
    for rows, fit, style in [
        (v5_rows, v5_fit, {"fmt": "o", "color": "#059669", "label": "v5 migration 12-bin points"}),
        (c7_rows, c7_fit, {"fmt": "s", "color": "#2563eb", "label": "7-bin conservative points"}),
    ]:
        x = np.asarray([float(row["E_p50_TeV"]) for row in rows])
        y = np.asarray([float(row["E2_dnde"]) for row in rows])
        yerr = np.asarray([float(row["E2_dnde_err"]) for row in rows])
        ax.errorbar(x, y, yerr=yerr, ms=5.4, lw=1.1, capsize=2.5, **style)
    grid = np.logspace(math.log10(0.25), math.log10(150.0), 500)
    ax.plot(
        grid,
        grid * grid * flux_for_model(grid, v5_fit.model_name, v5_fit.parameters),
        color="#059669",
        lw=1.8,
        label="v5 migration 12-bin LogPar fit",
    )
    ax.plot(
        grid,
        grid * grid * flux_for_model(grid, c7_fit.model_name, c7_fit.parameters),
        color="#2563eb",
        lw=1.5,
        ls="--",
        label="7-bin conservative LogPar fit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2 dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Crab SED: v5 migration-binned vs official pass5")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pulls(v5_rows: list[dict[str, Any]], c7_rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), dpi=160, sharey=True)
    for ax, rows, title, color in [
        (axes[0], v5_rows, "v5 migration 12-bin pulls", "#059669"),
        (axes[1], c7_rows, "7-bin conservative pulls", "#2563eb"),
    ]:
        x = np.arange(len(rows))
        ax.bar(x, [float(row["pull"]) for row in rows], color=color, alpha=0.82)
        ax.axhline(0.0, color="#111827", lw=0.8)
        for y in [-3.0, 3.0]:
            ax.axhline(y, color="#9ca3af", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["group_label"]) for row in rows], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("pull")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_kernels(context: dict[str, Any], v5_rows: list[dict[str, Any]], c7_rows: list[dict[str, Any]], params: dict[str, float], path: Path) -> None:
    plt = setup_matplotlib()
    weights = true_energy_weights(context, context["a_eff"], model="logpar", params=params)
    centers = np.power(10.0, 0.5 * (context["loge_edges"][:-1] + context["loge_edges"][1:])) / 1000.0
    by_id = {int(cid): idx for idx, cid in enumerate(context["cell_id"])}
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), dpi=160, sharex=True)
    for ax, rows, title in [
        (axes[0], v5_rows, "v5 migration 12-bin normalized true-energy kernels"),
        (axes[1], c7_rows, "7-bin conservative normalized true-energy kernels"),
    ]:
        for row in rows:
            ids = [int(v) for v in str(row["cell_ids"]).split(";")]
            idx = [by_id[cid] for cid in ids]
            w = np.sum(weights[idx], axis=0)
            norm = group_norm(w)
            ax.plot(centers, norm, lw=1.2, label=str(row["group_label"]))
        ax.set_xscale("log")
        ax.set_ylabel("normalized weight")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6.4, ncol=4)
    axes[-1].set_xlabel("True energy (TeV)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def add_pass5_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["analysis"], row["group_id"]): row for row in fit_ratio_to_pass5(rows)}
    out = []
    for row in rows:
        enriched = by_key.get((row["analysis"], row["group_id"]), row)
        out.append(enriched)
    return out


def build_report(
    *,
    summary: dict[str, Any],
    v5_rows: list[dict[str, Any]],
    c7_rows: list[dict[str, Any]],
    fine_rows: list[dict[str, Any]],
    kernel_overlap_rows: list[dict[str, Any]],
) -> None:
    v5 = summary["fits"]["v5_migration_12bin"]["logpar"]
    c7 = summary["fits"]["conservative_7bin"]["logpar"]
    fine = summary["fits"]["fine30"]["logpar"]
    cards = f"""
    <div class="cards">
      <div class="card"><div class="k">fine 30 cells</div><div class="v">{format_float(fine['chi2'])}/{fine['ndof']}</div><p>LogPar chi2/ndof {format_float(fine['chi2_over_ndof'])}</p></div>
      <div class="card"><div class="k">v5 migration 12-bin</div><div class="v">{format_float(v5['chi2'])}/{v5['ndof']}</div><p>LogPar chi2/ndof {format_float(v5['chi2_over_ndof'])}; restored 4/17/39/43</p></div>
      <div class="card"><div class="k">7-bin conservative</div><div class="v">{format_float(c7['chi2'])}/{c7['ndof']}</div><p>LogPar chi2/ndof {format_float(c7['chi2_over_ndof'])}</p></div>
      <div class="card"><div class="k">contract</div><div class="v">containment=1</div><p>aperture-conditioned Stage A response</p></div>
    </div>
    """
    fit_rows = [
        [
            "fine 30 cells",
            format_float(fine["chi2"], 5),
            fine["ndof"],
            format_float(fine["chi2_over_ndof"], 4),
            format_float(fine["max_abs_pull"], 4),
            format_float(fine["parameters"]["phi0"], 5),
            format_float(fine["parameters"]["alpha"], 5),
            format_float(fine["parameters"]["beta"], 5),
        ],
        [
            "v5 migration 12-bin",
            format_float(v5["chi2"], 5),
            v5["ndof"],
            format_float(v5["chi2_over_ndof"], 4),
            format_float(v5["max_abs_pull"], 4),
            format_float(v5["parameters"]["phi0"], 5),
            format_float(v5["parameters"]["alpha"], 5),
            format_float(v5["parameters"]["beta"], 5),
        ],
        [
            "7-bin conservative",
            format_float(c7["chi2"], 5),
            c7["ndof"],
            format_float(c7["chi2_over_ndof"], 4),
            format_float(c7["max_abs_pull"], 4),
            format_float(c7["parameters"]["phi0"], 5),
            format_float(c7["parameters"]["alpha"], 5),
            format_float(c7["parameters"]["beta"], 5),
        ],
    ]
    group_headers = [
        "group",
        "cells",
        "restored",
        "Nhit span",
        "predE span",
        "E50 TeV",
        "E16-E84 TeV",
        "sigma68",
        "excess",
        "model",
        "pull",
        "ratio/pass5",
    ]

    def group_table(rows: list[dict[str, Any]]) -> str:
        return html_table(
            group_headers,
            [
                [
                    row["group_label"],
                    row["cell_ids"],
                    row["restored_cells"] or "",
                    row["nhit_span"],
                    row["predE_span"],
                    format_float(row["E_p50_TeV"], 4),
                    f"{format_float(row['E_p16_TeV'], 4)}-{format_float(row['E_p84_TeV'], 4)}",
                    format_float(row["sigma68_dex"], 4),
                    format_float(row["excess"], 5),
                    format_float(row["model_counts"], 5),
                    format_float(row["pull"], 4),
                    format_float(row.get("ratio_to_pass5"), 4),
                ]
                for row in rows
            ],
        )

    overlap_table = html_table(
        ["analysis", "left", "right", "overlap"],
        [
            [row["analysis"], row["left_group"], row["right_group"], format_float(row["overlap"], 4)]
            for row in sorted(kernel_overlap_rows, key=lambda r: (r["analysis"], -float(r["overlap"])))
        ],
    )
    fine_table = html_table(
        ["cell", "Nhit", "predE", "E50 TeV", "excess", "model", "pull", "restored"],
        [
            [
                row["cell_id"],
                row["nhit_bin"],
                row["predE_bin"],
                format_float(row["E_p50_TeV"], 4),
                format_float(row["excess"], 5),
                format_float(row["model_counts"], 5),
                format_float(row["pull"], 4),
                "yes" if row["restored_cell"] else "",
            ]
            for row in fine_rows
        ],
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Crab SED v5 MC-Migration Binning Comparison</title>
  <style>
    body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#111827; background:#f8fafc; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 24px 48px; }}
    h1 {{ font-size:30px; margin:0 0 8px; }}
    h2 {{ font-size:20px; margin:28px 0 10px; }}
    h3 {{ font-size:16px; margin:20px 0 8px; }}
    p {{ line-height:1.55; }}
    code {{ background:#eef2ff; padding:1px 4px; border-radius:4px; }}
    .section {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:20px; margin:18px 0; box-shadow:0 1px 2px rgba(15,23,42,.05); }}
    .cards {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .card {{ border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:#ffffff; }}
    .card .k {{ font-size:12px; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; }}
    .card .v {{ font-size:22px; font-weight:700; margin-top:4px; }}
    .card p {{ margin:4px 0 0; color:#4b5563; font-size:13px; }}
    .note {{ background:#f0fdf4; border-left:4px solid #059669; padding:10px 12px; border-radius:6px; margin:12px 0; }}
    .warn {{ background:#fff7ed; border-left:4px solid #ea580c; padding:10px 12px; border-radius:6px; margin:12px 0; }}
    table.compact {{ border-collapse:collapse; width:100%; font-size:12px; margin:10px 0 16px; }}
    table.compact th, table.compact td {{ border-bottom:1px solid #e5e7eb; padding:6px 7px; vertical-align:top; text-align:left; }}
    table.compact th {{ background:#f9fafb; font-weight:700; }}
    .figure {{ margin:14px 0 22px; }}
    .figure img {{ max-width:100%; border:1px solid #e5e7eb; border-radius:8px; background:#fff; }}
    .figure figcaption {{ font-size:13px; color:#4b5563; line-height:1.45; margin-top:6px; }}
    .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }}
    @media (max-width:900px) {{ .cards,.grid2 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <h1>Crab SED v5 MC-Migration Binning Comparison</h1>
  <p>This report is independent of previous v4/v5 reports. It uses the current annnorm background and aperture-conditioned Stage A response, then changes only the final SED binning layer.</p>
  {cards}
  <section class="section">
    <h2>1. Binning Rationale</h2>
    <p>The binning diagnostic uses the response-weighted true-energy kernel <code>K_cell(E_true) = A_eff(cell, E_true, theta) x Crab theta exposure x assumed Crab spectrum</code>. Adjacent fine cells are merged when their normalized kernels overlap strongly.</p>
    <p>The overlap diagnostic is <code>overlap(i,j) = sum_E min(K_i_norm(E), K_j_norm(E))</code>. A large value means two fine cells are not independent measurements of distinct true-energy ranges; treating both as separate final SED points can create unstable flux allocation and large pulls.</p>
    <div class="note">v5 main result restores cells <strong>4, 17, 39, 43</strong> and absorbs them into migration-informed coarse bins. This is different from the v4 drop4 control, which removed those cells before fitting.</div>
    {html_table(["fit", "chi2", "ndof", "chi2/ndof", "max |pull|", "phi0", "alpha", "beta"], fit_rows)}
  </section>
  <section class="section">
    <h2>2. Final Crab SED Comparison</h2>
    {figure(ASSET_DIR / "v5_sed_pass5_comparison.png", "Crab SED: v5 migration-binned and 7-bin conservative versus pass5", "Green circles are the v5 12-bin migration-binned SED points and solid green line is its independent LogPar fit. Blue squares and dashed blue line are the independent 7-bin conservative cross-check. Black points are official pass5 WCDA.")}
  </section>
  <section class="section">
    <h2>3. Binning Tables</h2>
    <h3>v5 migration 12-bin main result</h3>
    {group_table(v5_rows)}
    <h3>7-bin conservative cross-check</h3>
    {group_table(c7_rows)}
    <h3>fine 30-cell residual reference</h3>
    <div class="warn">This table is diagnostic only. The fine 30 cells are not the final v5 SED binning because their true-energy kernels overlap strongly.</div>
    {fine_table}
  </section>
  <section class="section">
    <h2>4. Minimal Diagnostics</h2>
    <div class="grid2">
      {figure(ASSET_DIR / "v5_fit_pull_comparison.png", "Pull comparison", "Stage F pulls for the v5 12-bin main result and 7-bin conservative cross-check.")}
      {figure(ASSET_DIR / "v5_true_energy_overlap_or_kernels.png", "True-energy kernels", "Normalized true-energy kernels used to motivate the final bin merging.")}
    </div>
    <h3>Adjacent final-bin true-energy overlaps</h3>
    {overlap_table}
  </section>
  <section class="section">
    <h2>5. Outputs</h2>
    <p>Machine-readable outputs are in <code>{html.escape(str(ASSET_DIR.relative_to(REPO_ROOT)))}</code>: <code>v5_migration_binning_summary.json</code>, <code>v5_migration_binning_groups.csv</code>, and the three PNG figures.</p>
  </section>
</main></body></html>
"""
    REPORT_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    stage06 = load_stage06()
    context = load_context()

    fine = fine30_fit(context, stage06)
    v5_groups = aggregate_groups(context, V5_GROUPS, "v5_migration_12bin")
    c7_groups = aggregate_groups(context, CONSERVATIVE_GROUPS, "conservative_7bin")

    v5_fits = fit_models(context, stage06, a_eff=v5_groups["a_eff"], observed=v5_groups["observed"], errors=v5_groups["errors"], label="v5_migration_12bin")
    c7_fits = fit_models(context, stage06, a_eff=c7_groups["a_eff"], observed=c7_groups["observed"], errors=c7_groups["errors"], label="conservative_7bin")

    v5_rows = add_pass5_ratios(enrich_rows_with_fit(context, v5_groups, v5_fits["preferred"]))
    c7_rows = add_pass5_ratios(enrich_rows_with_fit(context, c7_groups, c7_fits["preferred"]))
    fine_rows = fine["rows"]
    kernel_overlap_rows = kernel_rows(context, params=v5_fits["preferred"].parameters)

    plot_sed(v5_rows, c7_rows, v5_fits["preferred"], c7_fits["preferred"], ASSET_DIR / "v5_sed_pass5_comparison.png")
    plot_pulls(v5_rows, c7_rows, ASSET_DIR / "v5_fit_pull_comparison.png")
    plot_kernels(context, v5_rows, c7_rows, v5_fits["preferred"].parameters, ASSET_DIR / "v5_true_energy_overlap_or_kernels.png")

    all_group_rows = v5_rows + c7_rows
    write_csv(
        ASSET_DIR / "v5_migration_binning_groups.csv",
        all_group_rows,
        [
            "analysis",
            "group_id",
            "group_label",
            "cell_ids",
            "n_cells",
            "restored_cells",
            "nhit_span",
            "predE_span",
            "N_on",
            "B_on",
            "excess",
            "error",
            "E_p16_TeV",
            "E_p50_TeV",
            "E_p84_TeV",
            "sigma68_dex",
            "model_counts",
            "pull",
            "N0_ratio",
            "N0_ratio_err",
            "E2_dnde",
            "E2_dnde_err",
            "pass5_interp_E2",
            "ratio_to_pass5",
        ],
    )
    write_csv(
        ASSET_DIR / "v5_true_energy_overlap.csv",
        kernel_overlap_rows,
        ["analysis", "left_group", "right_group", "left_cells", "right_cells", "overlap"],
    )
    summary = {
        "inputs": {
            "response_npz": str(RESPONSE_NPZ.relative_to(REPO_ROOT)),
            "signal_npz": str(SIGNAL_NPZ.relative_to(REPO_ROOT)),
            "pass5_csv": str(PASS5_CSV.relative_to(REPO_ROOT)),
            "v4_rebin_groups_csv": str(V4_REBIN_GROUPS_CSV.relative_to(REPO_ROOT)),
        },
        "contract": {
            "background": "current annnorm Stage E signal",
            "response": "v4 aperture-conditioned Stage A response",
            "containment": 1.0,
            "stage_scope": "final SED binning layer only; no Stage A-E recomputation",
        },
        "groups": {
            "v5_migration_12bin": V5_GROUPS,
            "conservative_7bin": CONSERVATIVE_GROUPS,
            "restored_cells": sorted(RESTORED_IDS),
        },
        "fits": {
            "fine30": {name: result_payload(fit) for name, fit in fine["fits"].items()},
            "v5_migration_12bin": {name: result_payload(fit) for name, fit in v5_fits.items()},
            "conservative_7bin": {name: result_payload(fit) for name, fit in c7_fits.items()},
        },
        "validation": {
            "n_v5_rows": len(v5_rows),
            "n_conservative_rows": len(c7_rows),
            "restored_cells_present": sorted({cid for row in v5_rows for cid in [int(v) for v in row["cell_ids"].split(";")] if cid in RESTORED_IDS}),
            "html_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
        },
    }
    write_json(ASSET_DIR / "v5_migration_binning_summary.json", summary)
    build_report(summary=summary, v5_rows=v5_rows, c7_rows=c7_rows, fine_rows=fine_rows, kernel_overlap_rows=kernel_overlap_rows)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {ASSET_DIR}")


if __name__ == "__main__":
    main()
