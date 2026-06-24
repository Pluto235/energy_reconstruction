#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "apply/report/assets/v4-residual-ablation"

RESPONSE_NPZ = REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz"
SIGNAL_NPZ = (
    REPO_ROOT
    / "apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz"
)
STAGE_F_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz"
)
STAGE_F_SUMMARY_CSV = (
    REPO_ROOT
    / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_summary.csv"
)
RESPONSE_AUDIT_CELL_CSV = (
    REPO_ROOT / "apply/report/assets/v4-response-audit/official_pass5_containment_ablation_by_cell.csv"
)
STAGE_D_SUMMARY_CSV = (
    REPO_ROOT
    / "apply/output/stage_d_v3_candidate_annnorm/runs/v3_stage_d_annnorm_from_psfborrow/background_v3_candidate_annnorm_summary.csv"
)
CELL_CROSSMATCH_CSV = (
    REPO_ROOT / "apply/report/assets/v4-root-cause-diagnostics/v4_drop4_cell_root_cause_crossmatch.csv"
)

PIVOT_TEV = 3.0
REFERENCE_PHI0 = 2.114e-12
REFERENCE_GAMMA = 2.69
QUADRATURE_POINTS = 64
TARGET_CELLS = [15, 16, 29, 55, 65, 69]


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_interval(label: str) -> tuple[float | None, float | None]:
    text = str(label).strip()
    if text.lower() in {"all", "*"}:
        return None, None
    if text.startswith("[") and text.endswith(")") and "," in text:
        lo, hi = text[1:-1].split(",", 1)
        return float(lo), float(hi)
    if text.startswith(">="):
        return float(text[2:]), None
    if text.startswith("<"):
        return None, float(text[1:])
    return None, None


def interval_key(label: str) -> float:
    lo, hi = parse_interval(label)
    if lo is None and hi is None:
        return 1.0e30
    if lo is None:
        return -1.0e30
    return lo


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    return value


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_stage06():
    module_path = REPO_ROOT / "apply/stages/06_fit.py"
    spec = importlib.util.spec_from_file_location("stage06_fit_for_v4_residual_ablation", module_path)
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


def load_context(stage06: Any) -> dict[str, Any]:
    response = load_npz(RESPONSE_NPZ)
    signal = load_npz(SIGNAL_NPZ)
    stage_f = load_npz(STAGE_F_NPZ)

    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    selected_ids = np.asarray(stage_f["cell_id"], dtype=np.int64)
    selected_index = np.asarray([int(np.where(response_ids == cid)[0][0]) for cid in selected_ids], dtype=np.int64)
    signal_ids = np.asarray(signal["cell_id"], dtype=np.int64)
    signal_index = np.asarray([int(np.where(signal_ids == cid)[0][0]) for cid in selected_ids], dtype=np.int64)

    # Use the already-promoted Stage F arrays for observed quantities and exposure;
    # this makes the baseline reproducibility check exact.
    context = {
        "stage06": stage06,
        "response": response,
        "signal": signal,
        "stage_f": stage_f,
        "cell_id": selected_ids,
        "nhit_bin": np.asarray(stage_f["nhit_bin"]).astype(str),
        "predE_bin": np.asarray(stage_f["predE_bin"]).astype(str),
        "a_eff": np.asarray(response["a_eff"], dtype=np.float64)[selected_index],
        "all_a_eff": np.asarray(response["a_eff"], dtype=np.float64),
        "response_cell_id": response_ids,
        "response_nhit_bin": np.asarray(response["nhit_bin"]).astype(str),
        "response_predE_bin": np.asarray(response["predE_bin"]).astype(str),
        "selected_response_index": selected_index,
        "signal_index": signal_index,
        "loge_edges": np.asarray(response["logE_true_edges"], dtype=np.float64),
        "theta_exposure_sec": np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64),
        "containment": np.asarray(stage_f["containment_r_opt"], dtype=np.float64),
        "N_on": np.asarray(stage_f["N_on"], dtype=np.float64),
        "B_on": np.asarray(stage_f["B_on"], dtype=np.float64),
        "observed": np.asarray(stage_f["excess"], dtype=np.float64),
        "errors": np.asarray(stage_f["excess_err_conservative"], dtype=np.float64),
        "official_by_cell": {
            int(float(row["cell_id"])): row
            for row in read_csv(RESPONSE_AUDIT_CELL_CSV)
            if row.get("cell_id") and row.get("official_expected_aperture_response")
        },
    }
    return context


def fit_logpar(context: dict[str, Any], a_eff: np.ndarray, observed: np.ndarray, label: str) -> dict[str, Any]:
    stage06 = context["stage06"]
    pl = stage06.fit_model(
        model_name="pl",
        error_mode=f"{label}_conservative",
        observed=observed,
        errors=context["errors"],
        a_eff_m2=a_eff,
        containment=context["containment"],
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
        errors=context["errors"],
        a_eff_m2=a_eff,
        containment=context["containment"],
        theta_exposure_sec=context["theta_exposure_sec"],
        loge_edges=context["loge_edges"],
        pivot_tev=PIVOT_TEV,
        quadrature_points=QUADRATURE_POINTS,
        start_gamma=float(pl.parameters.get("gamma", REFERENCE_GAMMA)),
        start_phi0=float(pl.parameters.get("phi0", REFERENCE_PHI0)),
    )
    return {"pl": pl, "logpar": logpar}


def fit_summary_row(
    *,
    context: dict[str, Any],
    label: str,
    family: str,
    parameter: str,
    value: float,
    fit: Any,
    baseline_chi2: float,
    observed: np.ndarray,
) -> dict[str, Any]:
    cell_ids = np.asarray(context["cell_id"], dtype=np.int64)
    low_mid = np.isin(context["nhit_bin"], ["[125,200)", "[200,300)", "[300,500)"])
    target = np.isin(cell_ids, TARGET_CELLS)
    row: dict[str, Any] = {
        "label": label,
        "family": family,
        "parameter": parameter,
        "value": value,
        "chi2": float(fit.chi2),
        "ndof": int(fit.ndof),
        "chi2_over_ndof": float(fit.chi2 / fit.ndof) if fit.ndof > 0 else "",
        "delta_chi2_vs_baseline": float(fit.chi2 - baseline_chi2),
        "max_abs_pull": float(np.nanmax(np.abs(fit.pull))),
        "low_mid_chi2": float(np.nansum(np.square(fit.pull[low_mid]))),
        "target_chi2": float(np.nansum(np.square(fit.pull[target]))),
        "total_excess": float(np.sum(observed)),
        "total_model": float(np.sum(fit.model_counts)),
        "total_obs_over_model": float(np.sum(observed) / np.sum(fit.model_counts))
        if float(np.sum(fit.model_counts)) > 0
        else "",
    }
    for cid in TARGET_CELLS:
        idx = np.where(cell_ids == cid)[0]
        if idx.size:
            row[f"pull_cell_{cid}"] = float(fit.pull[int(idx[0])])
            row[f"model_cell_{cid}"] = float(fit.model_counts[int(idx[0])])
    return row


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
    all_a_eff = np.asarray(context["all_a_eff"], dtype=np.float64)
    out = np.asarray(context["a_eff"], dtype=np.float64).copy()
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


def response_migration_ablation(context: dict[str, Any], baseline_fit: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, float]] = [("baseline", "baseline", 0.0)]
    for mode in ["left", "right", "symmetric"]:
        for fraction in [0.05, 0.10, 0.20, 0.30]:
            specs.append((f"{mode}_{fraction:.2f}", mode, fraction))
    for label, mode, fraction in specs:
        a_eff = mixed_response(context, mode=mode, fraction=fraction)
        fit = fit_logpar(context, a_eff, context["observed"], label)["logpar"]
        rows.append(
            fit_summary_row(
                context=context,
                label=label,
                family="response_migration",
                parameter=mode,
                value=fraction,
                fit=fit,
                baseline_chi2=float(baseline_fit.chi2),
                observed=context["observed"],
            )
        )
        for idx, cid in enumerate(context["cell_id"]):
            cell_rows.append(
                {
                    "label": label,
                    "mode": mode,
                    "fraction": fraction,
                    "cell_id": int(cid),
                    "nhit_bin": str(context["nhit_bin"][idx]),
                    "predE_bin": str(context["predE_bin"][idx]),
                    "excess": float(context["observed"][idx]),
                    "model": float(fit.model_counts[idx]),
                    "pull": float(fit.pull[idx]),
                }
            )
    return rows, cell_rows


def background_scaled_observed(context: dict[str, Any], cells: set[int] | None, nhit: str | None, scale: float) -> np.ndarray:
    mask = np.ones(context["cell_id"].shape, dtype=bool)
    if cells is not None:
        mask &= np.isin(context["cell_id"], list(cells))
    if nhit is not None:
        mask &= context["nhit_bin"] == nhit
    b_new = np.asarray(context["B_on"], dtype=np.float64).copy()
    b_new[mask] *= 1.0 + float(scale)
    return np.asarray(context["N_on"], dtype=np.float64) - b_new


def background_scale_ablation(context: dict[str, Any], baseline_fit: Any) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, set[int] | None, str | None, float]] = [("baseline", "none", None, None, 0.0)]
    for scale in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        specs.append((f"all_bkg_plus_{scale:.2f}", "all_cells", None, None, scale))
    for nhit in ["[125,200)", "[200,300)", "[300,500)", "[500,800)", "[800,1100)", "[1100,2000)"]:
        for scale in [0.02, 0.05, 0.10, 0.15, 0.20]:
            specs.append((f"nhit_{nhit}_bkg_plus_{scale:.2f}", "nhit_row", None, nhit, scale))
    groups = {
        "bad_low_mid": {15, 16, 29},
        "bad_positive": {15, 16, 29, 55, 69},
        "row200_core": {14, 15, 16},
        "row300_core": {27, 28, 29, 30},
    }
    for group, cells in groups.items():
        for scale in [0.05, 0.10, 0.15, 0.20, 0.30]:
            specs.append((f"{group}_bkg_plus_{scale:.2f}", group, cells, None, scale))

    rows: list[dict[str, Any]] = []
    for label, family, cells, nhit, scale in specs:
        observed = background_scaled_observed(context, cells, nhit, scale)
        fit = fit_logpar(context, context["a_eff"], observed, label)["logpar"]
        rows.append(
            fit_summary_row(
                context=context,
                label=label,
                family=f"background_{family}",
                parameter=nhit or (",".join(str(x) for x in sorted(cells)) if cells else "all"),
                value=scale,
                fit=fit,
                baseline_chi2=float(baseline_fit.chi2),
                observed=observed,
            )
        )
    return rows


def required_background_shift_rows(context: dict[str, Any], baseline_fit: Any) -> list[dict[str, Any]]:
    cross = {
        int(float(row["cell_id"])): row
        for row in read_csv(CELL_CROSSMATCH_CSV)
        if row.get("cell_id")
    }
    stage_d = {
        int(float(row["cell_id"])): row
        for row in read_csv(STAGE_D_SUMMARY_CSV)
        if row.get("cell_id")
    }
    rows: list[dict[str, Any]] = []
    for idx, cid_raw in enumerate(context["cell_id"]):
        cid = int(cid_raw)
        official = context["official_by_cell"].get(cid, {})
        drow = stage_d.get(cid, {})
        crow = cross.get(cid, {})
        excess = float(context["observed"][idx])
        model = float(baseline_fit.model_counts[idx])
        b_on = float(context["B_on"][idx])
        official_expected = finite_float(official.get("official_expected_aperture_response"))
        rows.append(
            {
                "cell_id": cid,
                "nhit_bin": str(context["nhit_bin"][idx]),
                "predE_bin": str(context["predE_bin"][idx]),
                "N_on": float(context["N_on"][idx]),
                "B_on": b_on,
                "excess": excess,
                "logpar_model": model,
                "logpar_pull": float(baseline_fit.pull[idx]),
                "required_delta_b_to_logpar": excess - model,
                "required_delta_b_over_b_to_logpar": (excess - model) / b_on if b_on > 0 else "",
                "official_expected_aperture_response": official_expected if official_expected is not None else "",
                "required_delta_b_to_official": (excess - official_expected) if official_expected is not None else "",
                "required_delta_b_over_b_to_official": (excess - official_expected) / b_on
                if official_expected is not None and b_on > 0
                else "",
                "annulus_residual_rms": drow.get("annulus_residual_rms", ""),
                "annulus_surface_scale": drow.get("annulus_surface_scale", ""),
                "surface_fit_chi2_over_ndof": (
                    (finite_float(drow.get("surface_fit_chi2")) or 0.0)
                    / (finite_float(drow.get("surface_fit_ndof")) or float("nan"))
                    if finite_float(drow.get("surface_fit_ndof"))
                    else ""
                ),
                "source_masked_fraction": drow.get("source_masked_fraction", ""),
                "offsource_mean_sigma": crow.get("offsource_mean_sigma", ""),
                "sigma_obs_over_mc": crow.get("sigma_obs_over_mc", ""),
            }
        )
    rows.sort(key=lambda row: abs(finite_float(row.get("logpar_pull")) or 0.0), reverse=True)
    return rows


def supercell_rows(context: dict[str, Any], baseline_fit: Any) -> list[dict[str, Any]]:
    groups = [
        ("low_125_200_cells_1_2_3", [1, 2, 3]),
        ("low_125_200_cells_2_3", [2, 3]),
        ("nhit_200_300_cells_14_15_16", [14, 15, 16]),
        ("nhit_200_300_cells_15_16", [15, 16]),
        ("nhit_300_500_cells_26_to_30", [26, 27, 28, 29, 30]),
        ("nhit_300_500_cells_27_28_29", [27, 28, 29]),
        ("nhit_300_500_cells_28_29_30", [28, 29, 30]),
        ("nhit_500_800_cells_40_41_42", [40, 41, 42]),
        ("nhit_800_1100_cells_52_to_55", [52, 53, 54, 55]),
        ("nhit_1100_2000_cells_65_to_69", [65, 66, 67, 68, 69]),
        ("nhit_2000_3000_cells_81_to_83", [81, 82, 83]),
    ]
    rows: list[dict[str, Any]] = []
    cell_ids = np.asarray(context["cell_id"], dtype=np.int64)
    for label, ids in groups:
        mask = np.isin(cell_ids, ids)
        if not np.any(mask):
            continue
        excess = float(np.sum(context["observed"][mask]))
        model = float(np.sum(baseline_fit.model_counts[mask]))
        err = float(np.sqrt(np.sum(np.square(context["errors"][mask]))))
        b_on = float(np.sum(context["B_on"][mask]))
        individual_chi2 = float(np.nansum(np.square(baseline_fit.pull[mask])))
        group_pull = (excess - model) / err if err > 0 else float("nan")
        official_expected = 0.0
        official_cells = 0
        for cid in ids:
            row = context["official_by_cell"].get(int(cid), {})
            value = finite_float(row.get("official_expected_aperture_response"))
            if value is not None:
                official_expected += value
                official_cells += 1
        rows.append(
            {
                "group": label,
                "cell_ids": ",".join(str(cid) for cid in ids),
                "cells_present": int(np.count_nonzero(mask)),
                "nhit_bin": ",".join(sorted({str(x) for x in context["nhit_bin"][mask]}, key=interval_key)),
                "predE_span": f"{min(str(x) for x in context['predE_bin'][mask])}..{max(str(x) for x in context['predE_bin'][mask])}",
                "excess": excess,
                "logpar_model": model,
                "error_quadrature": err,
                "group_pull": group_pull,
                "individual_chi2_sum": individual_chi2,
                "group_chi2": group_pull * group_pull,
                "chi2_relief": individual_chi2 - group_pull * group_pull,
                "obs_over_logpar": excess / model if model > 0 else "",
                "official_expected_aperture_response": official_expected if official_cells else "",
                "obs_over_official": excess / official_expected if official_expected > 0 else "",
                "required_delta_b_over_b_to_logpar": (excess - model) / b_on if b_on > 0 else "",
                "required_delta_b_over_b_to_official": (excess - official_expected) / b_on
                if official_expected > 0 and b_on > 0
                else "",
            }
        )
    rows.sort(key=lambda row: float(row["chi2_relief"]), reverse=True)
    return rows


def plot_response_chi2(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    for mode, color in [("left", "#d55e00"), ("right", "#0072b2"), ("symmetric", "#009e73")]:
        selected = sorted(
            [row for row in rows if row["family"] == "response_migration" and row["parameter"] == mode],
            key=lambda row: float(row["value"]),
        )
        if selected:
            ax.plot(
                [float(row["value"]) for row in selected],
                [float(row["chi2_over_ndof"]) for row in selected],
                marker="o",
                lw=1.6,
                color=color,
                label=mode,
            )
    base = next((row for row in rows if row["label"] == "baseline"), None)
    if base:
        ax.axhline(float(base["chi2_over_ndof"]), color="#111827", lw=1.0, ls="--", label="baseline")
    ax.set_xlabel("Adjacent predE response mixing fraction")
    ax.set_ylabel("LogPar chi2 / ndof")
    ax.set_title("Response / energy-migration ablation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_response_target_pulls(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=160, sharey=True)
    for ax, mode in zip(axes, ["left", "right", "symmetric"]):
        selected = sorted(
            [row for row in rows if row["family"] == "response_migration" and row["parameter"] == mode],
            key=lambda row: float(row["value"]),
        )
        for cid in [15, 29, 55, 65, 69]:
            vals = [finite_float(row.get(f"pull_cell_{cid}")) for row in selected]
            if any(value is not None for value in vals):
                ax.plot(
                    [float(row["value"]) for row in selected],
                    [float(value) if value is not None else np.nan for value in vals],
                    marker="o",
                    lw=1.2,
                    label=f"cell {cid}",
                )
        ax.axhline(0.0, color="#111827", lw=0.8)
        for y in [-3.0, 3.0]:
            ax.axhline(y, color="#9ca3af", lw=0.7, ls="--")
        ax.set_title(mode)
        ax.set_xlabel("mixing fraction")
        ax.grid(alpha=0.23)
    axes[0].set_ylabel("LogPar pull")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Target-cell pulls under adjacent-response mixing")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_background_chi2(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    keep_families = [
        "background_all_cells",
        "background_bad_low_mid",
        "background_bad_positive",
        "background_row200_core",
        "background_row300_core",
    ]
    colors = ["#111827", "#d55e00", "#cc79a7", "#0072b2", "#009e73"]
    for family, color in zip(keep_families, colors):
        selected = sorted([row for row in rows if row["family"] == family], key=lambda row: float(row["value"]))
        if selected:
            ax.plot(
                [100.0 * float(row["value"]) for row in selected],
                [float(row["chi2_over_ndof"]) for row in selected],
                marker="o",
                lw=1.5,
                color=color,
                label=family.replace("background_", ""),
            )
    base = next((row for row in rows if row["label"] == "baseline"), None)
    if base:
        ax.axhline(float(base["chi2_over_ndof"]), color="#6b7280", lw=1.0, ls="--", label="baseline")
    ax.set_xlabel("Injected B_on scale increase (%)")
    ax.set_ylabel("LogPar chi2 / ndof")
    ax.set_title("Background-scale sensitivity ablation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_supercell_relief(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    ordered = sorted(rows, key=lambda row: float(row["chi2_relief"]), reverse=True)
    labels = [str(row["group"]).replace("nhit_", "").replace("_cells_", "\n") for row in ordered]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=160)
    ax.bar(x, [float(row["individual_chi2_sum"]) for row in ordered], color="#d1d5db", label="sum individual chi2")
    ax.bar(x, [float(row["group_chi2"]) for row in ordered], color="#2563eb", label="grouped chi2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.2)
    ax.set_ylabel("chi2 contribution")
    ax.set_title("Does coarse grouping absorb fine-cell residuals?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_required_b_shift(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    ordered = rows[:16]
    labels = [str(row["cell_id"]) for row in ordered]
    values = [100.0 * (finite_float(row.get("required_delta_b_over_b_to_logpar")) or 0.0) for row in ordered]
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in values]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)
    ax.bar(np.arange(len(ordered)), values, color=colors, alpha=0.8)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("cell id, sorted by |Stage F pull|")
    ax.set_ylabel("Required delta B_on / B_on to match LogPar (%)")
    ax.set_title("Local background shift required by current v4 LogPar residuals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage06 = load_stage06()
    context = load_context(stage06)

    baseline = fit_logpar(context, context["a_eff"], context["observed"], "baseline")["logpar"]
    response_rows, response_cell_rows = response_migration_ablation(context, baseline)
    background_rows = background_scale_ablation(context, baseline)
    super_rows = supercell_rows(context, baseline)
    required_rows = required_background_shift_rows(context, baseline)

    write_csv(
        OUT_DIR / "response_migration_ablation_summary.csv",
        response_rows,
        [
            "label",
            "family",
            "parameter",
            "value",
            "chi2",
            "ndof",
            "chi2_over_ndof",
            "delta_chi2_vs_baseline",
            "max_abs_pull",
            "low_mid_chi2",
            "target_chi2",
            "total_excess",
            "total_model",
            "total_obs_over_model",
            *[f"pull_cell_{cid}" for cid in TARGET_CELLS],
            *[f"model_cell_{cid}" for cid in TARGET_CELLS],
        ],
    )
    write_csv(
        OUT_DIR / "response_migration_cell_pulls.csv",
        response_cell_rows,
        ["label", "mode", "fraction", "cell_id", "nhit_bin", "predE_bin", "excess", "model", "pull"],
    )
    write_csv(
        OUT_DIR / "background_scale_ablation_summary.csv",
        background_rows,
        [
            "label",
            "family",
            "parameter",
            "value",
            "chi2",
            "ndof",
            "chi2_over_ndof",
            "delta_chi2_vs_baseline",
            "max_abs_pull",
            "low_mid_chi2",
            "target_chi2",
            "total_excess",
            "total_model",
            "total_obs_over_model",
            *[f"pull_cell_{cid}" for cid in TARGET_CELLS],
            *[f"model_cell_{cid}" for cid in TARGET_CELLS],
        ],
    )
    write_csv(
        OUT_DIR / "supercell_residual_summary.csv",
        super_rows,
        [
            "group",
            "cell_ids",
            "cells_present",
            "nhit_bin",
            "predE_span",
            "excess",
            "logpar_model",
            "error_quadrature",
            "group_pull",
            "individual_chi2_sum",
            "group_chi2",
            "chi2_relief",
            "obs_over_logpar",
            "official_expected_aperture_response",
            "obs_over_official",
            "required_delta_b_over_b_to_logpar",
            "required_delta_b_over_b_to_official",
        ],
    )
    write_csv(
        OUT_DIR / "required_background_shift_to_logpar_by_cell.csv",
        required_rows,
        [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "N_on",
            "B_on",
            "excess",
            "logpar_model",
            "logpar_pull",
            "required_delta_b_to_logpar",
            "required_delta_b_over_b_to_logpar",
            "official_expected_aperture_response",
            "required_delta_b_to_official",
            "required_delta_b_over_b_to_official",
            "annulus_residual_rms",
            "annulus_surface_scale",
            "surface_fit_chi2_over_ndof",
            "source_masked_fraction",
            "offsource_mean_sigma",
            "sigma_obs_over_mc",
        ],
    )

    plot_response_chi2(response_rows, OUT_DIR / "response_migration_chi2.png")
    plot_response_target_pulls(response_rows, OUT_DIR / "response_migration_target_pulls.png")
    plot_background_chi2(background_rows, OUT_DIR / "background_scale_chi2.png")
    plot_supercell_relief(super_rows, OUT_DIR / "supercell_chi2_relief.png")
    plot_required_b_shift(required_rows, OUT_DIR / "required_background_shift_to_logpar.png")

    best_response = min(response_rows, key=lambda row: float(row["chi2"]))
    best_background = min(background_rows, key=lambda row: float(row["chi2"]))
    best_supercell = max(super_rows, key=lambda row: float(row["chi2_relief"])) if super_rows else {}
    summary = {
        "inputs": {
            "response_npz": rel(RESPONSE_NPZ),
            "signal_npz": rel(SIGNAL_NPZ),
            "stage_f_npz": rel(STAGE_F_NPZ),
            "stage_f_summary_csv": rel(STAGE_F_SUMMARY_CSV),
        },
        "baseline": {
            "chi2": float(baseline.chi2),
            "ndof": int(baseline.ndof),
            "chi2_over_ndof": float(baseline.chi2 / baseline.ndof),
            "max_abs_pull": float(np.nanmax(np.abs(baseline.pull))),
            "n_cells": int(len(context["cell_id"])),
        },
        "best_response_migration": best_response,
        "best_background_scale": best_background,
        "best_supercell_relief": best_supercell,
        "interpretation": {
            "response_migration": (
                "If adjacent-predE response mixing materially lowers chi2 without targeted B_on changes, "
                "the residual is consistent with response/energy-migration mismatch across fine 2D cells."
            ),
            "background": (
                "If only targeted B_on scaling lowers chi2, compare the required delta B/B with annulus residuals; "
                "large required shifts without annulus/off-source support are not a clean background explanation."
            ),
            "supercell": (
                "Large chi2 relief after grouping adjacent predE cells means the total row-like flux is less problematic "
                "than the fine-cell distribution, pointing to migration/selection/response shape rather than pure statistics."
            ),
        },
    }
    write_json(OUT_DIR / "v4_residual_ablation_summary.json", summary)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
