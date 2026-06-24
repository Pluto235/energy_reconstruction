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
REPORT_PATH = REPO_ROOT / "apply/report/crab_sed_v4_supercell_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v4-supercell"
STAGE_F_DIR = REPO_ROOT / "apply/output/stage_f_v4_supercell/runs/v4_supercell_stage_f"
STAGE_G_DIR = REPO_ROOT / "apply/output/stage_g_v4_supercell/runs/v4_supercell_stage_g"

RESPONSE_NPZ = REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz"
SIGNAL_NPZ = (
    REPO_ROOT
    / "apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz"
)
STAGE_F_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz"
)
STAGE_F_METADATA = (
    REPO_ROOT
    / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_metadata.json"
)
STAGE_G_SUMMARY = (
    REPO_ROOT
    / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4/sed_points_v4_aperture_conditioned_drop4_summary.csv"
)
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
V099_CSV = REPO_ROOT / "apply/report/assets/official-v099/wcda_crab_sed_v099_20250731_20260616_123624.csv"

PIVOT_TEV = 3.0
REFERENCE_PHI0 = 2.114e-12
REFERENCE_GAMMA = 2.69
QUADRATURE_POINTS = 64
M2_TO_CM2 = 1.0e4

SUPER_GROUPS = [
    ("nhit_125_200", "[125,200)", [1, 2, 3]),
    ("nhit_200_300", "[200,300)", [14, 15, 16]),
    ("nhit_300_500", "[300,500)", [26, 27, 28, 29, 30]),
    ("nhit_500_800", "[500,800)", [40, 41, 42]),
    ("nhit_800_1100", "[800,1100)", [52, 53, 54, 55]),
    ("nhit_1100_2000", "[1100,2000)", [65, 66, 67, 68, 69]),
    ("nhit_2000_3000", "[2000,3000)", [81, 82, 83]),
]


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def report_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPORT_PATH.parent))
    except ValueError:
        return rel(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_stage06():
    module_path = REPO_ROOT / "apply/stages/06_fit.py"
    spec = importlib.util.spec_from_file_location("stage06_fit_for_v4_supercell", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path, *, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=allow_pickle) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def parse_interval(label: str) -> tuple[float | None, float | None]:
    text = str(label).strip()
    if text.lower() in {"all", "*"}:
        return None, None
    if text.startswith("[") and text.endswith(")") and "," in text:
        low, high = text[1:-1].split(",", 1)
        return float(low), float(high)
    if text.startswith(">="):
        return float(text[2:]), None
    if text.startswith("<"):
        return None, float(text[1:])
    return None, None


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    return low


def format_float(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    abs_number = abs(number)
    if abs_number < 1.0e-3 or abs_number >= 1.0e4:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


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


def load_context(stage06: Any) -> dict[str, Any]:
    response = load_npz(RESPONSE_NPZ)
    signal = load_npz(SIGNAL_NPZ)
    stage_f = load_npz(STAGE_F_NPZ)

    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    selected_ids = np.asarray(stage_f["cell_id"], dtype=np.int64)
    selected_index = np.asarray([int(np.where(response_ids == cid)[0][0]) for cid in selected_ids], dtype=np.int64)

    return {
        "stage06": stage06,
        "response": response,
        "signal": signal,
        "stage_f": stage_f,
        "response_cell_id": response_ids,
        "response_nhit_bin": np.asarray(response["nhit_bin"]).astype(str),
        "response_predE_bin": np.asarray(response["predE_bin"]).astype(str),
        "all_a_eff": np.asarray(response["a_eff"], dtype=np.float64),
        "cell_id": selected_ids,
        "nhit_bin": np.asarray(stage_f["nhit_bin"]).astype(str),
        "predE_bin": np.asarray(stage_f["predE_bin"]).astype(str),
        "a_eff": np.asarray(response["a_eff"], dtype=np.float64)[selected_index],
        "selected_response_index": selected_index,
        "loge_edges": np.asarray(response["logE_true_edges"], dtype=np.float64),
        "theta_edges": np.asarray(response["theta_true_edges_deg"], dtype=np.float64),
        "theta_exposure_sec": np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64),
        "containment": np.asarray(stage_f["containment_r_opt"], dtype=np.float64),
        "N_on": np.asarray(stage_f["N_on"], dtype=np.float64),
        "B_on": np.asarray(stage_f["B_on"], dtype=np.float64),
        "observed": np.asarray(stage_f["excess"], dtype=np.float64),
        "errors": np.asarray(stage_f["excess_err_conservative"], dtype=np.float64),
    }


def fit_models(
    context: dict[str, Any],
    *,
    a_eff: np.ndarray,
    containment: np.ndarray,
    observed: np.ndarray,
    errors: np.ndarray,
    label: str,
) -> dict[str, Any]:
    stage06 = context["stage06"]
    pl = stage06.fit_model(
        model_name="pl",
        error_mode=f"{label}_conservative",
        observed=observed,
        errors=errors,
        a_eff_m2=a_eff,
        containment=containment,
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
        containment=containment,
        theta_exposure_sec=context["theta_exposure_sec"],
        loge_edges=context["loge_edges"],
        pivot_tev=PIVOT_TEV,
        quadrature_points=QUADRATURE_POINTS,
        start_gamma=float(pl.parameters.get("gamma", REFERENCE_GAMMA)),
        start_phi0=float(pl.parameters.get("phi0", REFERENCE_PHI0)),
    )
    return {"pl": pl, "logpar": logpar}


def result_payload(result: Any) -> dict[str, Any]:
    return {
        "model": result.model_name,
        "valid": bool(result.valid),
        "parameters": result.parameters,
        "errors": result.errors,
        "fit_parameters": result.fit_parameters,
        "fit_parameter_errors": result.fit_parameter_errors,
        "chi2": float(result.chi2),
        "ndof": int(result.ndof),
        "chi2_over_ndof": float(result.chi2 / result.ndof) if int(result.ndof) > 0 else None,
        "p_value": result.p_value,
        "max_abs_pull": float(np.nanmax(np.abs(result.pull))),
    }


def aggregate_supercells(context: dict[str, Any]) -> dict[str, Any]:
    cell_ids = np.asarray(context["cell_id"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    a_eff_rows: list[np.ndarray] = []
    observed: list[float] = []
    errors: list[float] = []
    n_on: list[float] = []
    b_on: list[float] = []
    for group_id, group_label, ids in SUPER_GROUPS:
        mask = np.isin(cell_ids, ids)
        if not np.any(mask):
            continue
        weighted_a_eff = np.asarray(context["a_eff"], dtype=np.float64)[mask] * np.asarray(
            context["containment"], dtype=np.float64
        )[mask, None, None]
        a_eff_rows.append(np.sum(weighted_a_eff, axis=0))
        obs = float(np.sum(context["observed"][mask]))
        err = float(np.sqrt(np.sum(np.square(context["errors"][mask]))))
        rows.append(
            {
                "group_id": group_id,
                "group_label": group_label,
                "cell_ids": ";".join(str(cid) for cid in ids),
                "n_cells": int(np.count_nonzero(mask)),
                "predE_span": f"{min(context['predE_bin'][mask], key=interval_key)}..{max(context['predE_bin'][mask], key=interval_key)}",
                "N_on": float(np.sum(context["N_on"][mask])),
                "B_on": float(np.sum(context["B_on"][mask])),
                "excess": obs,
                "error": err,
            }
        )
        observed.append(obs)
        errors.append(err)
        n_on.append(float(np.sum(context["N_on"][mask])))
        b_on.append(float(np.sum(context["B_on"][mask])))
    return {
        "rows": rows,
        "group_id": np.asarray([row["group_id"] for row in rows], dtype="U64"),
        "group_label": np.asarray([row["group_label"] for row in rows], dtype="U64"),
        "cell_ids": np.asarray([row["cell_ids"] for row in rows], dtype="U128"),
        "a_eff": np.asarray(a_eff_rows, dtype=np.float64),
        "containment": np.ones(len(rows), dtype=np.float64),
        "observed": np.asarray(observed, dtype=np.float64),
        "errors": np.asarray(errors, dtype=np.float64),
        "N_on": np.asarray(n_on, dtype=np.float64),
        "B_on": np.asarray(b_on, dtype=np.float64),
    }


def neighbor_index_map(context: dict[str, Any]) -> dict[int, dict[str, int | None]]:
    mapping: dict[int, dict[str, int | None]] = {}
    by_nhit: dict[str, list[int]] = {}
    for idx, nhit in enumerate(context["response_nhit_bin"]):
        by_nhit.setdefault(str(nhit), []).append(idx)
    for indices in by_nhit.values():
        ordered = sorted(indices, key=lambda idx: interval_key(str(context["response_predE_bin"][idx])))
        for pos, idx in enumerate(ordered):
            mapping[int(context["response_cell_id"][idx])] = {
                "left": int(ordered[pos - 1]) if pos > 0 else None,
                "right": int(ordered[pos + 1]) if pos + 1 < len(ordered) else None,
            }
    return mapping


def mixed_response(context: dict[str, Any], *, mode: str, fraction: float) -> np.ndarray:
    if fraction <= 0.0 or mode == "baseline":
        return np.asarray(context["a_eff"], dtype=np.float64).copy()
    out = np.asarray(context["a_eff"], dtype=np.float64).copy()
    all_a_eff = np.asarray(context["all_a_eff"], dtype=np.float64)
    neighbors = neighbor_index_map(context)
    for out_idx, cid in enumerate(np.asarray(context["cell_id"], dtype=np.int64)):
        info = neighbors.get(int(cid), {})
        if mode == "left":
            picks = [info.get("left")]
        elif mode == "right":
            picks = [info.get("right")]
        elif mode == "symmetric":
            picks = [info.get("left"), info.get("right")]
        else:
            raise ValueError(f"Unknown response mixing mode: {mode}")
        valid = [int(pick) for pick in picks if pick is not None]
        if not valid:
            continue
        borrowed = np.mean(all_a_eff[valid], axis=0)
        out[out_idx] = (1.0 - float(fraction)) * out[out_idx] + float(fraction) * borrowed
    return out


def response_morph_profile(context: dict[str, Any], *, prior_sigma: float = 0.10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ["left", "right", "symmetric"]:
        for fraction in np.linspace(0.0, 0.30, 31):
            a_eff = mixed_response(context, mode=mode, fraction=float(fraction))
            fit = fit_models(
                context,
                a_eff=a_eff,
                containment=context["containment"],
                observed=context["observed"],
                errors=context["errors"],
                label=f"morph_{mode}_{fraction:.2f}",
            )["logpar"]
            penalty = float((fraction / prior_sigma) ** 2) if prior_sigma > 0 else 0.0
            row = {
                "mode": mode,
                "fraction": float(fraction),
                "prior_sigma": prior_sigma,
                "chi2": float(fit.chi2),
                "ndof": int(fit.ndof),
                "chi2_over_ndof": float(fit.chi2 / fit.ndof) if int(fit.ndof) > 0 else "",
                "penalty": penalty,
                "profile_chi2": float(fit.chi2 + penalty),
                "profile_chi2_over_ndof": float((fit.chi2 + penalty) / fit.ndof) if int(fit.ndof) > 0 else "",
                "phi0": float(fit.parameters["phi0"]),
                "alpha": float(fit.parameters["alpha"]),
                "beta": float(fit.parameters["beta"]),
                "max_abs_pull": float(np.nanmax(np.abs(fit.pull))),
            }
            rows.append(row)
    rows.sort(key=lambda row: (str(row["mode"]), float(row["fraction"])))
    return rows


def true_energy_weights(
    context: dict[str, Any],
    a_eff: np.ndarray,
    *,
    model: str,
    params: dict[str, float],
) -> np.ndarray:
    flux_integral = integrate_flux_bins(context["loge_edges"], model=model, params=params)
    exposure = np.asarray(context["theta_exposure_sec"], dtype=np.float64)
    return M2_TO_CM2 * flux_integral[None, :] * np.einsum("bet,t->be", a_eff, exposure)


def energy_quantile_row(loge_edges: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    p16_log = weighted_quantile(centers, weights, 0.16)
    p50_log = weighted_quantile(centers, weights, 0.50)
    p84_log = weighted_quantile(centers, weights, 0.84)
    return {
        "logE_p16": p16_log,
        "logE_p50": p50_log,
        "logE_p84": p84_log,
        "sigma68_dex": 0.5 * (p84_log - p16_log) if math.isfinite(p16_log + p84_log) else float("nan"),
        "E_p16_TeV": math.pow(10.0, p16_log) / 1000.0 if math.isfinite(p16_log) else float("nan"),
        "E_p50_TeV": math.pow(10.0, p50_log) / 1000.0 if math.isfinite(p50_log) else float("nan"),
        "E_p84_TeV": math.pow(10.0, p84_log) / 1000.0 if math.isfinite(p84_log) else float("nan"),
    }


def binning_resolution_tables(context: dict[str, Any], preferred: Any, supercell: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_weights = true_energy_weights(context, context["a_eff"], model=preferred.model_name, params=preferred.parameters)
    rows: list[dict[str, Any]] = []
    for idx, cid in enumerate(context["cell_id"]):
        q = energy_quantile_row(context["loge_edges"], cell_weights[idx])
        low, high = parse_interval(str(context["predE_bin"][idx]))
        width = (high - low) if low is not None and high is not None else float("nan")
        sigma = q["sigma68_dex"]
        rows.append(
            {
                "cell_id": int(cid),
                "nhit_bin": str(context["nhit_bin"][idx]),
                "predE_bin": str(context["predE_bin"][idx]),
                "predE_width_dex": width,
                **q,
                "predE_width_over_sigma68": width / sigma if sigma > 0 else "",
                "too_fine_flag": bool(width < sigma) if math.isfinite(width) and math.isfinite(sigma) else "",
            }
        )

    norm = np.divide(cell_weights, np.sum(cell_weights, axis=1, keepdims=True), out=np.zeros_like(cell_weights), where=np.sum(cell_weights, axis=1, keepdims=True) > 0)
    overlap_rows: list[dict[str, Any]] = []
    by_nhit: dict[str, list[int]] = {}
    for idx, nhit in enumerate(context["nhit_bin"]):
        by_nhit.setdefault(str(nhit), []).append(idx)
    for nhit, indices in by_nhit.items():
        ordered = sorted(indices, key=lambda idx: interval_key(str(context["predE_bin"][idx])))
        for left, right in zip(ordered[:-1], ordered[1:]):
            overlap = float(np.sum(np.minimum(norm[left], norm[right])))
            overlap_rows.append(
                {
                    "nhit_bin": nhit,
                    "left_cell": int(context["cell_id"][left]),
                    "left_predE": str(context["predE_bin"][left]),
                    "right_cell": int(context["cell_id"][right]),
                    "right_predE": str(context["predE_bin"][right]),
                    "true_energy_overlap": overlap,
                    "high_overlap_flag": bool(overlap >= 0.50),
                }
            )
    overlap_rows.sort(key=lambda row: float(row["true_energy_overlap"]), reverse=True)

    group_weights = true_energy_weights(context, supercell["a_eff"], model=preferred.model_name, params=preferred.parameters)
    for idx, row in enumerate(supercell["rows"]):
        row.update({f"true_energy_{key}": val for key, val in energy_quantile_row(context["loge_edges"], group_weights[idx]).items()})
    return rows, overlap_rows


def sed_points_from_supercell(context: dict[str, Any], supercell: dict[str, Any], preferred: Any) -> list[dict[str, Any]]:
    weights = true_energy_weights(context, supercell["a_eff"], model=preferred.model_name, params=preferred.parameters)
    model_counts = np.asarray(preferred.model_counts, dtype=np.float64)
    points: list[dict[str, Any]] = []
    for idx, row in enumerate(supercell["rows"]):
        obs = float(supercell["observed"][idx])
        err = float(supercell["errors"][idx])
        model = float(model_counts[idx])
        ratio = obs / model if model > 0 else float("nan")
        ratio_err = err / model if model > 0 else float("nan")
        q = energy_quantile_row(context["loge_edges"], weights[idx])
        e_eff = q["E_p50_TeV"]
        params = dict(preferred.parameters)
        params["phi0"] = float(params["phi0"]) * ratio
        e2 = e_eff * e_eff * float(flux_for_model(np.asarray([e_eff]), preferred.model_name, params)[0])
        e2_err = abs(e2 * ratio_err / ratio) if ratio > 0 else float("nan")
        points.append(
            {
                "grouping": "nhit_supercell",
                "group_label": row["group_label"],
                "cell_ids": row["cell_ids"],
                "n_cells": row["n_cells"],
                "N0_ratio": ratio,
                "N0_ratio_err": ratio_err,
                "effective_energy_tev": e_eff,
                "true_energy_p16_tev": q["E_p16_TeV"],
                "true_energy_p84_tev": q["E_p84_TeV"],
                "E2_dnde": e2,
                "E2_dnde_err": e2_err,
                "observed_excess_total": obs,
                "model_counts_total": model,
                "pull": float(preferred.pull[idx]),
            }
        )
    return points


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
    return np.asarray(x), np.asarray(y)


def v099_points() -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(V099_CSV)
    x: list[float] = []
    y: list[float] = []
    for row in rows:
        e = finite_float(row.get("energy_tev"))
        e2_scaled = finite_float(row.get("e2_flux_scaled_1e14_tev_cm2_s"))
        if e is not None and e2_scaled is not None:
            x.append(e)
            y.append(e2_scaled * 1.0e-14)
    return np.asarray(x), np.asarray(y)


def original_v4_nhit_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(STAGE_G_SUMMARY) if row.get("grouping") == "nhit"]
    x: list[float] = []
    y: list[float] = []
    yerr: list[float] = []
    for row in rows:
        e = finite_float(row.get("effective_energy_tev"))
        e2 = finite_float(row.get("E2_dnde"))
        err = finite_float(row.get("E2_dnde_err"))
        if e is not None and e2 is not None:
            x.append(e)
            y.append(e2)
            yerr.append(err if err is not None else float("nan"))
    return np.asarray(x), np.asarray(y), np.asarray(yerr)


def plot_supercell_fit(supercell: dict[str, Any], preferred: Any, path: Path) -> None:
    plt = setup_matplotlib()
    labels = [row["group_label"] for row in supercell["rows"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)
    ax.errorbar(x, supercell["observed"], yerr=supercell["errors"], fmt="o", color="#111827", label="super-cell excess")
    ax.plot(x, preferred.model_counts, "s-", color="#047857", label=f"{preferred.model_name} model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("counts")
    ax.set_title("v4_supercell Stage F model counts vs excess")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_supercell_pulls(supercell: dict[str, Any], preferred: Any, path: Path) -> None:
    plt = setup_matplotlib()
    labels = [row["group_label"] for row in supercell["rows"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160)
    colors = ["#dc2626" if val > 0 else "#2563eb" for val in preferred.pull]
    ax.bar(x, preferred.pull, color=colors, alpha=0.82)
    for y in [-3.0, 0.0, 3.0]:
        ax.axhline(y, color="#111827" if y == 0.0 else "#9ca3af", lw=0.8, ls="-" if y == 0.0 else "--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("pull")
    ax.set_title("v4_supercell Stage F pulls")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_response_morph(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=160)
    colors = {"left": "#d55e00", "right": "#0072b2", "symmetric": "#009e73"}
    for mode in ["left", "right", "symmetric"]:
        selected = [row for row in rows if row["mode"] == mode]
        ax.plot(
            [float(row["fraction"]) for row in selected],
            [float(row["chi2_over_ndof"]) for row in selected],
            marker="o",
            ms=3.5,
            lw=1.3,
            color=colors[mode],
            label=f"{mode} free chi2",
        )
        ax.plot(
            [float(row["fraction"]) for row in selected],
            [float(row["profile_chi2_over_ndof"]) for row in selected],
            lw=1.0,
            ls="--",
            color=colors[mode],
            alpha=0.8,
            label=f"{mode} + prior",
        )
    ax.set_xlabel("adjacent predE response-mixing fraction")
    ax.set_ylabel("LogPar chi2 / ndof")
    ax.set_title("Response-morph nuisance profile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_energy_resolution(cell_rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    ordered = sorted(cell_rows, key=lambda row: (interval_key(row["nhit_bin"]), interval_key(row["predE_bin"])))
    labels = [str(row["cell_id"]) for row in ordered]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=160)
    ax.bar(x - 0.18, [float(row["predE_width_dex"]) for row in ordered], width=0.36, color="#9ca3af", label="predE bin width")
    ax.bar(x + 0.18, [float(row["sigma68_dex"]) for row in ordered], width=0.36, color="#2563eb", label="MC true-energy sigma68")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("cell id, ordered by Nhit/predE")
    ax.set_ylabel("dex")
    ax.set_title("PredE cell width versus response-weighted true-energy spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_adjacent_overlap(overlap_rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    ordered = sorted(overlap_rows, key=lambda row: float(row["true_energy_overlap"]), reverse=True)
    labels = [f"{row['left_cell']}-{row['right_cell']}" for row in ordered]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=160)
    ax.bar(x, [float(row["true_energy_overlap"]) for row in ordered], color="#7c3aed", alpha=0.8)
    ax.axhline(0.5, color="#111827", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("sum min(P_i(E), P_j(E))")
    ax.set_title("Adjacent-cell true-energy overlap")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sed(points: list[dict[str, Any]], preferred: Any, morph_best: dict[str, Any] | None, path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=170)
    e_pass5, y_pass5 = pass5_points()
    if e_pass5.size:
        ax.plot(e_pass5, y_pass5, "o", color="#111827", ms=5, label="official pass5 WCDA")
    e_v099, y_v099 = v099_points()
    if e_v099.size:
        ax.plot(e_v099, y_v099, "s", color="#9a3412", ms=4.5, mfc="none", label="tutorial v0.99 WCDA")
    e_old, y_old, yerr_old = original_v4_nhit_points()
    if e_old.size:
        ax.errorbar(e_old, y_old, yerr=yerr_old, fmt="o", color="#64748b", mfc="none", ms=5, label="original v4 Nhit points")
    x = np.asarray([float(row["effective_energy_tev"]) for row in points])
    y = np.asarray([float(row["E2_dnde"]) for row in points])
    yerr = np.asarray([float(row["E2_dnde_err"]) for row in points])
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#059669", ms=6, label="v4_supercell Nhit points")
    grid = np.logspace(math.log10(0.25), math.log10(150.0), 400)
    ax.plot(
        grid,
        grid * grid * flux_for_model(grid, preferred.model_name, preferred.parameters),
        color="#059669",
        lw=1.8,
        label="v4_supercell fit",
    )
    if morph_best:
        params = {
            "phi0": float(morph_best["phi0"]),
            "alpha": float(morph_best["alpha"]),
            "beta": float(morph_best["beta"]),
        }
        ax.plot(
            grid,
            grid * grid * flux_for_model(grid, "logpar", params),
            color="#d55e00",
            lw=1.3,
            ls="--",
            label=f"response-morph prior best ({morph_best['mode']} {float(morph_best['fraction']):.2f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2 dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("v4_supercell SED comparison")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_supercell_outputs(context: dict[str, Any], supercell: dict[str, Any], fits: dict[str, Any], points: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    STAGE_F_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_G_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "group_id": supercell["group_id"],
        "group_label": supercell["group_label"],
        "cell_ids": supercell["cell_ids"],
        "N_on": supercell["N_on"],
        "B_on": supercell["B_on"],
        "excess": supercell["observed"],
        "excess_err_conservative": supercell["errors"],
        "containment_r_opt": supercell["containment"],
        "theta_exposure_sec": context["theta_exposure_sec"],
    }
    for name, fit in fits.items():
        payload[f"{name}_model_counts"] = fit.model_counts
        payload[f"{name}_residual"] = fit.residual
        payload[f"{name}_pull"] = fit.pull
    np.savez(STAGE_F_DIR / "fit_v4_supercell.npz", **payload)
    write_json(STAGE_F_DIR / "fit_v4_supercell_metadata.json", metadata)
    write_csv(
        STAGE_F_DIR / "fit_v4_supercell_summary.csv",
        [
            {
                **row,
                "pl_model": float(fits["pl"].model_counts[idx]),
                "pl_pull": float(fits["pl"].pull[idx]),
                "logpar_model": float(fits["logpar"].model_counts[idx]),
                "logpar_pull": float(fits["logpar"].pull[idx]),
            }
            for idx, row in enumerate(supercell["rows"])
        ],
        [
            "group_id",
            "group_label",
            "cell_ids",
            "n_cells",
            "predE_span",
            "N_on",
            "B_on",
            "excess",
            "error",
            "pl_model",
            "pl_pull",
            "logpar_model",
            "logpar_pull",
        ],
    )
    write_csv(
        STAGE_G_DIR / "sed_points_v4_supercell_summary.csv",
        points,
        [
            "grouping",
            "group_label",
            "cell_ids",
            "n_cells",
            "N0_ratio",
            "N0_ratio_err",
            "effective_energy_tev",
            "true_energy_p16_tev",
            "true_energy_p84_tev",
            "E2_dnde",
            "E2_dnde_err",
            "observed_excess_total",
            "model_counts_total",
            "pull",
        ],
    )
    np.savez(
        STAGE_G_DIR / "sed_points_v4_supercell.npz",
        group_label=np.asarray([row["group_label"] for row in points], dtype="U64"),
        cell_ids=np.asarray([row["cell_ids"] for row in points], dtype="U128"),
        effective_energy_tev=np.asarray([row["effective_energy_tev"] for row in points], dtype=np.float64),
        e2_dnde=np.asarray([row["E2_dnde"] for row in points], dtype=np.float64),
        e2_dnde_err=np.asarray([row["E2_dnde_err"] for row in points], dtype=np.float64),
        observed_excess_total=np.asarray([row["observed_excess_total"] for row in points], dtype=np.float64),
        model_counts_total=np.asarray([row["model_counts_total"] for row in points], dtype=np.float64),
    )


def build_report(
    *,
    summary: dict[str, Any],
    supercell: dict[str, Any],
    points: list[dict[str, Any]],
    morph_rows: list[dict[str, Any]],
    cell_resolution_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> None:
    baseline = summary["baseline_original_v4"]
    super_fit = summary["supercell_fit"]["preferred"]
    best_free = summary["response_morph"]["best_free"]
    best_prior = summary["response_morph"]["best_with_prior"]

    cards = f"""
    <div class="cards">
      <div class="card"><div class="k">original v4</div><div class="v">{format_float(baseline['chi2'])}/{baseline['ndof']}</div><p>LogPar chi2/ndof {format_float(baseline['chi2_over_ndof'])}</p></div>
      <div class="card"><div class="k">v4_supercell</div><div class="v">{format_float(super_fit['chi2'])}/{super_fit['ndof']}</div><p>LogPar chi2/ndof {format_float(super_fit['chi2_over_ndof'])}; 7 Nhit super-cells</p></div>
      <div class="card"><div class="k">response morph</div><div class="v">{html.escape(str(best_prior['mode']))} {format_float(best_prior['fraction'], 3)}</div><p>prior-profile chi2/ndof {format_float(best_prior['profile_chi2_over_ndof'])}</p></div>
      <div class="card"><div class="k">binning check</div><div class="v">{summary['binning']['too_fine_cells']}</div><p>cells with predE width below MC sigma68</p></div>
    </div>
    """

    super_rows = [
        [
            row["group_label"],
            row["cell_ids"],
            row["predE_span"],
            format_float(row["excess"], 5),
            format_float(row["error"], 5),
            format_float(row.get("logpar_model"), 5),
            format_float(row.get("logpar_pull"), 4),
            format_float(row.get("true_energy_E_p50_TeV"), 4),
            format_float(row.get("true_energy_sigma68_dex"), 4),
        ]
        for row in supercell["rows"]
    ]
    point_rows = [
        [
            row["group_label"],
            row["cell_ids"],
            format_float(row["effective_energy_tev"], 4),
            format_float(row["E2_dnde"], 5),
            format_float(row["E2_dnde_err"], 4),
            format_float(row["N0_ratio"], 4),
            format_float(row["pull"], 4),
        ]
        for row in points
    ]
    morph_keep = sorted(morph_rows, key=lambda row: float(row["profile_chi2"]))[:8]
    morph_table = [
        [
            row["mode"],
            format_float(row["fraction"], 3),
            format_float(row["chi2"], 5),
            format_float(row["chi2_over_ndof"], 4),
            format_float(row["penalty"], 4),
            format_float(row["profile_chi2"], 5),
            format_float(row["profile_chi2_over_ndof"], 4),
            format_float(row["alpha"], 4),
            format_float(row["beta"], 4),
        ]
        for row in morph_keep
    ]
    resolution_table = [
        [
            row["cell_id"],
            row["nhit_bin"],
            row["predE_bin"],
            format_float(row["predE_width_dex"], 4),
            format_float(row["sigma68_dex"], 4),
            format_float(row["predE_width_over_sigma68"], 4),
            row["too_fine_flag"],
        ]
        for row in cell_resolution_rows[:14]
    ]
    overlap_table = [
        [
            row["nhit_bin"],
            f"{row['left_cell']} {row['left_predE']}",
            f"{row['right_cell']} {row['right_predE']}",
            format_float(row["true_energy_overlap"], 4),
            row["high_overlap_flag"],
        ]
        for row in overlap_rows[:12]
    ]

    body = (
        "<section class=\"section\"><h2>Repair Summary</h2>"
        "<p>This report is a new v4_supercell branch. It keeps the current annnorm Stage D/E background and the aperture-conditioned Stage A response, and does not overwrite the original v4 report.</p>"
        '<div class="note">Working conclusion: the immediate stable repair is to promote Nhit-row super-cells for the current short-data SED. The response-morph profile remains a systematic diagnostic, not a hard-coded correction. The MC binning check shows why the original fine predE cells should stay diagnostic until response migration is modeled explicitly.</div>'
        + cards
        + "</section>"
        "<section class=\"section\"><h2>1. Super-Cell Stage F/G</h2>"
        "<p>Adjacent predE cells in the same Nhit row are summed before the forward-folding fit. Model response is summed with the Stage E containment already fixed to one.</p>"
        + html_table(["Nhit", "cells", "predE span", "excess", "err", "LogPar model", "pull", "E50 TeV", "sigma68 dex"], super_rows)
        + '<div class="grid2">'
        + figure(ASSET_DIR / "v4_supercell_fit_counts.png", "Super-cell model counts vs excess", "The fit is done on seven coarse Nhit-row super-cells.")
        + figure(ASSET_DIR / "v4_supercell_pulls.png", "Super-cell pull grid", "Residuals after replacing fine predE cells with Nhit-row super-cells.")
        + figure(ASSET_DIR / "v4_supercell_sed_comparison.png", "v4_supercell SED comparison", "Super-cell SED points, original v4 Nhit points, official pass5, v0.99, and fitted curves.")
        + "</div>"
        + "<h3>Stage G-style super-cell points</h3>"
        + html_table(["Nhit", "cells", "Eeff TeV", "E2 dN/dE", "err", "N0 ratio", "pull"], point_rows)
        + "</section>"
        "<section class=\"section\"><h2>2. Response-Morph Nuisance</h2>"
        "<p>The original 26-cell fit is profiled over adjacent predE response mixing. A Gaussian prior with sigma=0.10 is included as a systematic-nuisance diagnostic.</p>"
        + '<div class="note">'
        + f"Best free morph: {html.escape(str(best_free['mode']))} f={format_float(best_free['fraction'], 3)}, chi2/ndof={format_float(best_free['chi2_over_ndof'])}. "
        + f"Best prior-profile morph: {html.escape(str(best_prior['mode']))} f={format_float(best_prior['fraction'], 3)}, profile chi2/ndof={format_float(best_prior['profile_chi2_over_ndof'])}. "
        + "This supports response / migration sensitivity, but it should be reported as a systematic envelope unless the response model is re-derived."
        + "</div>"
        + figure(ASSET_DIR / "v4_response_morph_profile.png", "Response-morph nuisance profile", "Solid curves are free chi2; dashed curves include the Gaussian prior penalty.")
        + html_table(["mode", "fraction", "chi2", "chi2/ndof", "prior", "profile chi2", "profile/ndof", "alpha", "beta"], morph_table)
        + "</section>"
        "<section class=\"section\"><h2>3. MC Energy-Resolution / Binning Check</h2>"
        "<p>True-energy support is computed from the aperture-conditioned response weighted by the v4_supercell preferred spectrum and the Crab theta exposure.</p>"
        + '<div class="note">'
        + f"{summary['binning']['too_fine_cells']} of {summary['binning']['n_cells']} original fit cells have predE bin width smaller than their response-weighted true-energy sigma68, so the simple width/sigma test alone does not flag the grid as too narrow. "
        + f"However, {summary['binning']['high_overlap_pairs']} adjacent pairs have true-energy overlap >= 0.50. That overlap is the stronger warning: adjacent fine cells are not independent enough for a final fit unless response migration is profiled."
        + "</div>"
        + '<div class="grid2">'
        + figure(ASSET_DIR / "v4_energy_resolution_by_cell.png", "PredE width versus MC true-energy spread", "Cells where the blue bar exceeds the gray bar are too fine for the current response resolution.")
        + figure(ASSET_DIR / "v4_adjacent_true_energy_overlap.png", "Adjacent true-energy overlap", "Large overlap means adjacent predE cells cannot be treated as independent spectral bins without a migration nuisance.")
        + "</div>"
        + "<h3>Most problematic width / resolution rows</h3>"
        + html_table(["cell", "Nhit", "predE", "width", "sigma68", "width/sigma", "too fine"], resolution_table)
        + "<h3>Highest adjacent overlaps</h3>"
        + html_table(["Nhit", "left cell", "right cell", "overlap", "high"], overlap_table)
        + "</section>"
        "<section class=\"section\"><h2>4. Recommended Use</h2>"
        "<p>For the current short-data analysis, use <strong>v4_supercell</strong> as the robust diagnostic SED branch and keep the original fine-cell v4 result as a residual-localization diagnostic. The next full analysis should rebuild v5 binning around the observed true-energy overlap, or profile response migration as a real nuisance, before treating adjacent predE cells as independent final fit bins.</p>"
        f"<p>Machine-readable outputs: <code>{html.escape(rel(STAGE_F_DIR))}</code>, <code>{html.escape(rel(STAGE_G_DIR))}</code>, <code>{html.escape(rel(ASSET_DIR))}</code>.</p>"
        + "</section>"
    )

    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Crab SED v4_supercell Repair Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;}}
header{{background:#0f172a;color:white;padding:28px 40px;}}
main{{max-width:1180px;margin:0 auto;padding:24px;}}
h1{{margin:0 0 6px 0;font-size:28px;}} h2{{margin:0 0 14px 0;font-size:22px;}}
.section{{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:20px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04);}}
.note{{background:#ecfdf5;border-left:4px solid #059669;padding:12px 14px;margin:12px 0;border-radius:4px;line-height:1.45;}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:14px 0;}}
.card{{border:1px solid #e5e7eb;border-radius:8px;padding:12px;background:#f9fafb;}}
.card .k{{font-size:12px;text-transform:uppercase;color:#64748b;letter-spacing:.03em;}}
.card .v{{font-size:24px;font-weight:700;margin:4px 0;}}
.card p{{margin:0;color:#475569;font-size:13px;line-height:1.35;}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;align-items:start;}}
.figure{{margin:0;border:1px solid #e5e7eb;border-radius:8px;padding:10px;background:#fff;}}
.figure img{{width:100%;height:auto;display:block;}}
figcaption{{font-size:12px;color:#475569;line-height:1.35;margin-top:8px;}}
table.compact{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0;}}
table.compact th,table.compact td{{border:1px solid #e5e7eb;padding:5px 7px;text-align:right;}}
table.compact th:first-child,table.compact td:first-child{{text-align:left;}}
table.compact th{{background:#f1f5f9;color:#334155;}}
code{{background:#f1f5f9;padding:1px 4px;border-radius:4px;}}
</style></head>
<body><header><h1>Crab SED v4_supercell Repair Report</h1><p>Independent repair branch for super-cell binning and response-migration diagnostics.</p></header><main>{body}</main></body></html>
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    stage06 = load_stage06()
    context = load_context(stage06)

    baseline_original = fit_models(
        context,
        a_eff=context["a_eff"],
        containment=context["containment"],
        observed=context["observed"],
        errors=context["errors"],
        label="original_v4_recomputed",
    )["logpar"]

    supercell = aggregate_supercells(context)
    super_fits = fit_models(
        context,
        a_eff=supercell["a_eff"],
        containment=supercell["containment"],
        observed=supercell["observed"],
        errors=supercell["errors"],
        label="v4_supercell",
    )
    preferred = super_fits["logpar"]
    for idx, row in enumerate(supercell["rows"]):
        row["pl_model"] = float(super_fits["pl"].model_counts[idx])
        row["pl_pull"] = float(super_fits["pl"].pull[idx])
        row["logpar_model"] = float(preferred.model_counts[idx])
        row["logpar_pull"] = float(preferred.pull[idx])

    points = sed_points_from_supercell(context, supercell, preferred)
    morph_rows = response_morph_profile(context, prior_sigma=0.10)
    best_free = min(morph_rows, key=lambda row: float(row["chi2"]))
    best_prior = min(morph_rows, key=lambda row: float(row["profile_chi2"]))

    cell_resolution_rows, overlap_rows = binning_resolution_tables(context, preferred, supercell)
    cell_resolution_rows.sort(
        key=lambda row: (finite_float(row.get("predE_width_over_sigma68")) or 1.0e9, interval_key(row["nhit_bin"]))
    )
    too_fine = [row for row in cell_resolution_rows if row.get("too_fine_flag") is True]
    high_overlap = [row for row in overlap_rows if row.get("high_overlap_flag") is True]

    summary = {
        "inputs": {
            "response_npz": rel(RESPONSE_NPZ),
            "signal_npz": rel(SIGNAL_NPZ),
            "original_stage_f_npz": rel(STAGE_F_NPZ),
            "original_stage_f_metadata": rel(STAGE_F_METADATA),
        },
        "baseline_original_v4": result_payload(baseline_original),
        "supercell_fit": {
            "grouping": "Nhit-row super-cells; adjacent predE cells summed before Stage F",
            "n_supercells": int(len(supercell["rows"])),
            "pl": result_payload(super_fits["pl"]),
            "logpar": result_payload(super_fits["logpar"]),
            "preferred": result_payload(preferred),
        },
        "response_morph": {
            "prior_sigma": 0.10,
            "best_free": best_free,
            "best_with_prior": best_prior,
        },
        "binning": {
            "n_cells": int(len(cell_resolution_rows)),
            "too_fine_cells": int(len(too_fine)),
            "high_overlap_pairs": int(len(high_overlap)),
            "too_fine_rule": "predE_width_dex < response-weighted true-energy sigma68_dex",
            "high_overlap_rule": "adjacent true-energy overlap >= 0.50",
        },
        "recommendation": (
            "Use v4_supercell as the current robust diagnostic SED branch; keep original fine cells for residual localization; "
            "rebuild v5 binning and response-migration nuisance before treating fine predE cells as final fit bins."
        ),
    }

    write_supercell_outputs(context, supercell, super_fits, points, summary)
    write_json(ASSET_DIR / "v4_supercell_summary.json", summary)
    write_csv(
        ASSET_DIR / "response_morph_profile.csv",
        morph_rows,
        [
            "mode",
            "fraction",
            "prior_sigma",
            "chi2",
            "ndof",
            "chi2_over_ndof",
            "penalty",
            "profile_chi2",
            "profile_chi2_over_ndof",
            "phi0",
            "alpha",
            "beta",
            "max_abs_pull",
        ],
    )
    write_csv(
        ASSET_DIR / "energy_resolution_by_cell.csv",
        cell_resolution_rows,
        [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "predE_width_dex",
            "logE_p16",
            "logE_p50",
            "logE_p84",
            "sigma68_dex",
            "E_p16_TeV",
            "E_p50_TeV",
            "E_p84_TeV",
            "predE_width_over_sigma68",
            "too_fine_flag",
        ],
    )
    write_csv(
        ASSET_DIR / "adjacent_true_energy_overlap.csv",
        overlap_rows,
        ["nhit_bin", "left_cell", "left_predE", "right_cell", "right_predE", "true_energy_overlap", "high_overlap_flag"],
    )

    plot_supercell_fit(supercell, preferred, ASSET_DIR / "v4_supercell_fit_counts.png")
    plot_supercell_pulls(supercell, preferred, ASSET_DIR / "v4_supercell_pulls.png")
    plot_response_morph(morph_rows, ASSET_DIR / "v4_response_morph_profile.png")
    plot_energy_resolution(cell_resolution_rows, ASSET_DIR / "v4_energy_resolution_by_cell.png")
    plot_adjacent_overlap(overlap_rows, ASSET_DIR / "v4_adjacent_true_energy_overlap.png")
    plot_sed(points, preferred, best_prior, ASSET_DIR / "v4_supercell_sed_comparison.png")

    build_report(
        summary=summary,
        supercell=supercell,
        points=points,
        morph_rows=morph_rows,
        cell_resolution_rows=cell_resolution_rows,
        overlap_rows=overlap_rows,
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {ASSET_DIR}")
    print(f"Wrote {STAGE_F_DIR}")
    print(f"Wrote {STAGE_G_DIR}")


if __name__ == "__main__":
    main()
