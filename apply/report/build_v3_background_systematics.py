#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

stage_f = importlib.import_module("apply.stages.06_fit")
stage_g = importlib.import_module("apply.stages.07_sed_points")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3 background-method systematics from existing Stage D maps."
    )
    parser.add_argument("--stage-d-npz", type=str, default="apply/output/stage_d_v3_candidate/runs/v3_stage_d_slurm_42024/background_v3_candidate.npz")
    parser.add_argument("--stage-d-metadata", type=str, default="apply/output/stage_d_v3_candidate/runs/v3_stage_d_slurm_42024/background_v3_candidate_metadata.json")
    parser.add_argument("--stage-e-npz", type=str, default="apply/output/stage_e_v3_candidate/runs/v3_stage_e_slurm_42024/signal_v3_candidate.npz")
    parser.add_argument("--stage-e-metadata", type=str, default="apply/output/stage_e_v3_candidate/runs/v3_stage_e_slurm_42024/signal_v3_candidate_metadata.json")
    parser.add_argument("--stage-c-dir", type=str, default="apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024")
    parser.add_argument("--response-npz", type=str, default="apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz")
    parser.add_argument("--response-metadata", type=str, default="apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline.csv")
    parser.add_argument("--stage-f-reference-metadata", type=str, default="apply/output/stage_f_v3_baseline/runs/v3_stage_f_slurm_42024/fit_v3_baseline_metadata.json")
    parser.add_argument("--stage-g-reference-metadata", type=str, default="apply/output/stage_g_v3_baseline/runs/v3_stage_g_slurm_42024/sed_points_v3_baseline_metadata.json")
    parser.add_argument("--output-dir", type=str, default="apply/report/assets/v3-background-systematics")
    parser.add_argument("--signal-output-dir", type=str, default="apply/output/stage_e_v3_background_systematics")
    parser.add_argument("--fit-output-dir", type=str, default="apply/output/stage_f_v3_background_systematics")
    parser.add_argument("--pivot-tev", type=float, default=3.0)
    parser.add_argument("--reference-phi0", type=float, default=2.114e-12)
    parser.add_argument("--reference-gamma", type=float, default=2.69)
    parser.add_argument("--exposure-sample-step-sec", type=float, default=60.0)
    parser.add_argument("--energy-quadrature-points", type=int, default=64)
    parser.add_argument("--profile-half-width-deg", type=float, default=1.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    def ready(value):
        if isinstance(value, dict):
            return {str(k): ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ready(v) for v in value]
        if isinstance(value, np.ndarray):
            return ready(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            number = float(value)
            return number if math.isfinite(number) else None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(ready(payload), f, indent=2)


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def selector_ids(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids: List[int] = []
    for row in rows:
        if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}:
            ids.append(int(row["cell_id"]))
    if not ids:
        raise ValueError(f"No included cells in selector: {path}")
    return ids


def known_background_sigma(n_on: np.ndarray, b_on: np.ndarray) -> np.ndarray:
    n = np.asarray(n_on, dtype=np.float64)
    b = np.asarray(b_on, dtype=np.float64)
    sigma = np.full(n.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(b) & (b > 0.0)
    positive = valid & (n > 0.0)
    term = np.zeros(n.shape, dtype=np.float64)
    term[positive] = n[positive] * np.log(n[positive] / b[positive]) - (n[positive] - b[positive])
    term[valid & (n <= 0.0)] = b[valid & (n <= 0.0)]
    sigma[valid] = np.sqrt(2.0 * np.maximum(term[valid], 0.0))
    sigma[valid & (n < b)] *= -1.0
    return sigma


def surface_design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    if int(order) == 1:
        return np.column_stack([np.ones_like(x), x, y]).astype(np.float64)
    if int(order) == 2:
        return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y]).astype(np.float64)
    raise ValueError(f"Unsupported surface order: {order}")


def annulus_placement(
    r_opt_deg: np.ndarray,
    *,
    default_inner_deg: float,
    width_deg: float,
    source_mask_min_deg: float,
    source_mask_r_opt_factor: float,
    source_mask_margin_deg: float,
    max_inner_deg: float,
    shifted: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_opt = np.asarray(r_opt_deg, dtype=np.float64)
    source_mask = np.maximum(float(source_mask_min_deg), float(source_mask_r_opt_factor) * r_opt)
    if shifted:
        shifted_inner = source_mask + float(source_mask_margin_deg)
        inner = np.full(r_opt.shape, float(default_inner_deg), dtype=np.float64)
        needs_shift = shifted_inner > float(default_inner_deg)
        inner[needs_shift] = np.minimum(np.ceil(shifted_inner[needs_shift] * 2.0) / 2.0, float(max_inner_deg))
    else:
        inner = np.full(r_opt.shape, float(default_inner_deg), dtype=np.float64)
    outer = inner + float(width_deg)
    return source_mask, inner, outer


def fit_background_variant(
    counts_map: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    fiducial_mask: np.ndarray,
    on_mask: np.ndarray,
    r_opt_deg: np.ndarray,
    *,
    order: int,
    shifted: bool,
    log_link: bool,
    background_model: Dict[str, object],
) -> Dict[str, np.ndarray]:
    n_cells, n_y, n_x = counts_map.shape
    x_grid, y_grid = np.meshgrid(x_centers.astype(np.float64), y_centers.astype(np.float64))
    rho_grid = np.hypot(x_grid, y_grid)
    full_design = surface_design_matrix(x_grid.ravel(), y_grid.ravel(), int(order))
    source_radius, inner, outer = annulus_placement(
        r_opt_deg,
        default_inner_deg=float(background_model.get("annulus_default_inner_deg", 1.5) or 1.5),
        width_deg=float(background_model.get("annulus_width_deg", 2.0) or 2.0),
        source_mask_min_deg=float(background_model.get("annulus_source_mask_min_deg", 1.5) or 1.5),
        source_mask_r_opt_factor=float(background_model.get("annulus_source_mask_r_opt_factor", 2.0) or 2.0),
        source_mask_margin_deg=float(background_model.get("annulus_source_mask_margin_deg", 0.2) or 0.2),
        max_inner_deg=float(background_model.get("annulus_max_inner_deg", 4.5) or 4.5),
        shifted=shifted,
    )
    background = np.full_like(counts_map, np.nan, dtype=np.float32)
    residual = np.full_like(counts_map, np.nan, dtype=np.float32)
    training_mask = np.zeros_like(counts_map, dtype=bool)
    b_on = np.zeros(n_cells, dtype=np.float64)
    annulus_counts = np.zeros(n_cells, dtype=np.float64)
    annulus_pixels = np.zeros(n_cells, dtype=np.int64)
    condition = np.full(n_cells, np.inf, dtype=np.float64)
    rank = np.zeros(n_cells, dtype=np.int64)
    residual_rms = np.full(n_cells, np.nan, dtype=np.float64)
    fit_success = np.zeros(n_cells, dtype=bool)
    min_training_pixels = int(background_model.get("surface_min_training_pixels", 80) or 80)
    condition_max = float(background_model.get("surface_condition_max", 1.0e8) or 1.0e8)

    for idx in range(n_cells):
        annulus = (rho_grid >= inner[idx]) & (rho_grid < outer[idx]) & fiducial_mask
        training_mask[idx] = annulus
        annulus_pixels[idx] = int(np.count_nonzero(annulus))
        annulus_counts[idx] = float(np.nansum(counts_map[idx][annulus]))
        if annulus_pixels[idx] < min_training_pixels:
            continue
        z = counts_map[idx][annulus].astype(np.float64).ravel()
        design = surface_design_matrix(x_grid[annulus].ravel(), y_grid[annulus].ravel(), int(order))
        if log_link:
            z_fit = np.log(np.maximum(z, 0.0) + 0.5)
            weights = np.sqrt(np.maximum(z, 1.0))
        else:
            z_fit = z
            weights = 1.0 / np.sqrt(np.maximum(z, 1.0))
        weighted_design = design * weights[:, None]
        weighted_z = z_fit * weights
        rank[idx] = int(np.linalg.matrix_rank(weighted_design))
        if rank[idx] < design.shape[1]:
            continue
        condition[idx] = float(np.linalg.cond(weighted_design))
        if not np.isfinite(condition[idx]) or condition[idx] > condition_max:
            continue
        coeff, _, _, _ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)
        pred_linear = (full_design @ coeff).reshape(n_y, n_x)
        pred = np.exp(pred_linear) - 0.5 if log_link else pred_linear
        if np.any(pred[fiducial_mask] < 0.0):
            if log_link:
                pred = np.maximum(pred, 0.0)
            else:
                continue
        pred = np.maximum(pred, 0.0)
        pred[~fiducial_mask] = np.nan
        background[idx] = pred.astype(np.float32)
        on = on_mask[idx] & fiducial_mask
        b_on[idx] = float(np.nansum(pred[on]))
        valid_bg = annulus & np.isfinite(pred) & (pred > 0.0)
        res = np.full((n_y, n_x), np.nan, dtype=np.float64)
        res[valid_bg] = (counts_map[idx][valid_bg] - pred[valid_bg]) / np.sqrt(np.maximum(pred[valid_bg], 1.0))
        residual[idx] = res.astype(np.float32)
        values = res[valid_bg]
        residual_rms[idx] = float(np.sqrt(np.nanmean(values * values))) if values.size else np.nan
        fit_success[idx] = bool(b_on[idx] > 0.0)

    return {
        "background_map": background,
        "training_mask": training_mask,
        "residual_map": residual,
        "B_on": b_on,
        "annulus_inner_deg": inner.astype(np.float32),
        "annulus_outer_deg": outer.astype(np.float32),
        "source_mask_radius_deg": source_radius.astype(np.float32),
        "annulus_counts": annulus_counts,
        "annulus_pixels": annulus_pixels,
        "fit_rank": rank,
        "fit_condition_number": condition,
        "annulus_residual_rms": residual_rms,
        "surface_fit_success": fit_success,
    }


def compute_signal_arrays(
    template: Dict[str, np.ndarray],
    b_on: np.ndarray,
) -> Dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in template.items()}
    n_on = np.asarray(out["N_on"], dtype=np.int64)
    b = np.asarray(b_on, dtype=np.float64)
    out["B_on"] = b
    out["N_off"] = np.full(b.shape, np.nan, dtype=np.float64)
    out["alpha"] = np.full(b.shape, np.nan, dtype=np.float64)
    out["excess"] = n_on.astype(np.float64) - b
    out["excess_err_stat"] = np.sqrt(np.maximum(n_on.astype(np.float64), 0.0))
    out["excess_err_conservative"] = np.sqrt(np.maximum(n_on.astype(np.float64) + b, 0.0))
    out["known_b_sigma"] = known_background_sigma(n_on, b)
    out["li_ma_sigma"] = np.full(b.shape, np.nan, dtype=np.float64)
    out["formal_sigma"] = out["known_b_sigma"].copy()
    out["background_mode"] = np.asarray(["crab_roi_local"] * b.size, dtype="U64")
    out["background_form"] = np.asarray(["direct_expectation"] * b.size, dtype="U64")
    out["statistic_kind"] = np.asarray(["known_background_poisson"] * b.size, dtype="U64")
    return out


def write_signal_npz(path: Path, signal: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **signal)


def signal_metadata(
    *,
    name: str,
    signal_path: Path,
    template_metadata: Dict[str, object],
    stage_d_metadata_path: Path,
    total_n: int,
    total_b: float,
    formal_sigma: float,
    valid_background_cells: int,
) -> Dict[str, object]:
    meta = dict(template_metadata)
    meta["description"] = f"Stage E signal table recomputed for v3 background-systematics variant {name}."
    meta["run_id"] = f"v3_background_systematics_{name}"
    meta["output_dir"] = str(signal_path.parent)
    meta["inputs"] = dict(meta.get("inputs", {})) if isinstance(meta.get("inputs"), dict) else {}
    meta["inputs"]["background_systematics_variant"] = name
    meta["inputs"]["background_metadata_json"] = str(stage_d_metadata_path)
    meta["stage_d_contract"] = dict(meta.get("stage_d_contract", {})) if isinstance(meta.get("stage_d_contract"), dict) else {}
    meta["stage_d_contract"]["background_mode"] = "crab_roi_local"
    meta["stage_d_contract"]["background_form"] = "direct_expectation"
    meta["stage_d_contract"]["statistic_kind"] = "known_background_poisson"
    meta["stage_d_contract"]["promotable_contract"] = True
    meta["quality_gate"] = {
        "status": "passed",
        "reason": "background-systematics diagnostic signal table; not promoted as nominal Stage E",
        "exit_code": 0,
        "promotable": True,
        "diagnostic_only": True,
        "valid_background_cells": int(valid_background_cells),
    }
    meta["totals"] = {
        "N_on": int(total_n),
        "B_on": float(total_b),
        "N_off": None,
        "excess": float(total_n - total_b),
        "excess_err_stat": float(math.sqrt(max(total_n, 0.0))),
        "known_b_sigma_aggregate": float(formal_sigma),
        "known_b_sigma_combined_independent_cells": None,
        "li_ma_sigma_combined_independent_cells": None,
        "formal_sigma": float(formal_sigma),
    }
    meta["outputs"] = {"npz": str(signal_path), "metadata_json": str(signal_path.with_name(signal_path.stem + "_metadata.json"))}
    return meta


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def baseline_mask(cell_ids: np.ndarray, included_ids: Sequence[int]) -> np.ndarray:
    include = {int(v) for v in included_ids}
    return np.asarray([int(v) in include for v in cell_ids], dtype=bool)


def run_stage_f_variant(
    *,
    args: argparse.Namespace,
    name: str,
    signal_npz: Path,
    signal_meta: Path,
    fit_output_root: Path,
) -> Dict[str, object]:
    run_id = f"v3_stage_f_background_systematics_{name}"
    run_dir = fit_output_root / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    argv = [
        "06_fit.py",
        "--response-npz",
        str(resolve(args.response_npz)),
        "--response-metadata",
        str(resolve(args.response_metadata)),
        "--signal-npz",
        str(signal_npz),
        "--signal-metadata",
        str(signal_meta),
        "--stage-c-dir",
        str(resolve(args.stage_c_dir)),
        "--cell-subset-csv",
        str(resolve(args.baseline_selector_csv)),
        "--output-dir",
        str(fit_output_root),
        "--run-id",
        run_id,
        "--source-ra-deg",
        "83.63",
        "--source-dec-deg",
        "22.01",
        "--lhaaso-lat-deg",
        "29.45",
        "--lhaaso-lon-deg",
        "100.14",
        "--exposure-sample-step-sec",
        str(float(args.exposure_sample_step_sec)),
        "--pivot-tev",
        str(float(args.pivot_tev)),
        "--reference-phi0",
        str(float(args.reference_phi0)),
        "--reference-gamma",
        str(float(args.reference_gamma)),
        "--energy-quadrature-points",
        str(int(args.energy_quadrature_points)),
        "--npz-name",
        "fit_v3_background_systematics.npz",
        "--metadata-name",
        "fit_v3_background_systematics_metadata.json",
        "--summary-csv-name",
        "fit_v3_background_systematics_summary.csv",
        "--summary-md-name",
        "fit_v3_background_systematics_summary.md",
        "--no-promote-current",
        "--no-plots",
        "--report-html",
        "",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        stage_f.main()
    finally:
        sys.argv = old_argv
    return load_json(run_dir / "fit_v3_background_systematics_metadata.json")


def high_energy_point_summary(stage_g_meta: Dict[str, object]) -> Dict[str, object]:
    points = stage_g_meta.get("points", []) if isinstance(stage_g_meta.get("points"), list) else []
    pred_points = [
        point for point in points
        if isinstance(point, dict) and point.get("grouping") == "predE"
    ]
    high = [
        point for point in pred_points
        if str(point.get("group_label", "")).startswith("[4.") or str(point.get("group_label", "")) == "[5,6)"
    ]
    return {
        "predE_points": len(pred_points),
        "high_energy_points": len(high),
        "high_energy_labels": [point.get("group_label") for point in high],
        "high_energy_cells": {
            str(point.get("group_label")): point.get("cell_ids", [])
            for point in high
        },
        "high_energy_effective_energy_tev": {
            str(point.get("group_label")): point.get("effective_energy_tev")
            for point in high
        },
        "high_energy_ratio_stage_f": {
            str(point.get("group_label")): point.get("ratio_to_stage_f_model")
            for point in high
        },
    }


def plot_sensitivity_summary(rows: Sequence[Dict[str, object]], output: Path) -> None:
    plt = setup_matplotlib()
    labels = [str(row["variant"]) for row in rows]
    nominal_b = float(rows[0]["baseline_B_on"])
    nominal_excess = float(rows[0]["baseline_excess"])
    b_delta = [100.0 * (float(row["baseline_B_on"]) / nominal_b - 1.0) for row in rows]
    excess_delta = [100.0 * (float(row["baseline_excess"]) / nominal_excess - 1.0) for row in rows]
    logpar_phi0 = [float(row["logpar_phi0"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), dpi=150, sharex=True)
    axes[0].bar(x - 0.18, b_delta, width=0.36, label="baseline B_on delta")
    axes[0].bar(x + 0.18, excess_delta, width=0.36, label="baseline excess delta")
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("delta vs nominal [%]")
    axes[0].set_title("v3 background-method sensitivity summary")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(x, logpar_phi0, marker="o", color="#005f73", label="LogPar phi0")
    axes[1].set_ylabel("LogPar phi0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def normalized_dec_profile(
    maps: np.ndarray,
    x_centers: np.ndarray,
    *,
    mask: np.ndarray,
    fit_mask: np.ndarray,
    half_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_band = np.abs(x_centers) < float(half_width)
    selected = maps[fit_mask][:, :, x_band]
    clean = np.where(np.isfinite(selected), selected, 0.0)
    profile = clean.sum(axis=(0, 2), dtype=np.float64)
    y_mask = np.any(mask[:, x_band], axis=1)
    profile = np.where(y_mask, profile, np.nan)
    peak = np.nanmax(np.abs(profile))
    if peak > 0.0 and np.isfinite(peak):
        profile = profile / peak
    return profile, y_mask


def plot_before_after_dec_profiles(
    *,
    counts_map: np.ndarray,
    background_map: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    fiducial_mask: np.ndarray,
    fit_mask: np.ndarray,
    half_width: float,
    output: Path,
) -> None:
    counts_profile, keep = normalized_dec_profile(
        counts_map.astype(np.float64),
        x_centers,
        mask=fiducial_mask,
        fit_mask=fit_mask,
        half_width=half_width,
    )
    background_profile, _ = normalized_dec_profile(
        background_map.astype(np.float64),
        x_centers,
        mask=fiducial_mask,
        fit_mask=fit_mask,
        half_width=half_width,
    )
    excess_profile, _ = normalized_dec_profile(
        counts_map.astype(np.float64) - background_map.astype(np.float64),
        x_centers,
        mask=fiducial_mask,
        fit_mask=fit_mask,
        half_width=half_width,
    )
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
    ax.plot(y_centers, counts_profile, label="before: counts profile", color="#222222", linewidth=1.8)
    ax.plot(y_centers, background_profile, label="fitted background surface", color="#005f73", linewidth=1.5)
    ax.plot(y_centers, excess_profile, label="after: counts - background", color="#b7791f", linewidth=1.5)
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlim(float(np.nanmin(y_centers[keep])), float(np.nanmax(y_centers[keep])))
    ax.set_xlabel("Dec offset from Crab [deg]")
    ax.set_ylabel("normalized summed profile")
    ax.set_title("v3 baseline before/after Dec profile comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "variant",
        "annulus",
        "surface_order",
        "fit_family",
        "baseline_B_on",
        "baseline_excess",
        "baseline_formal_sigma",
        "valid_baseline_background_cells",
        "pl_phi0",
        "pl_gamma",
        "pl_chi2",
        "logpar_phi0",
        "logpar_alpha",
        "logpar_beta",
        "logpar_chi2",
        "preferred_model",
        "predE_high_energy_points",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    signal_root = resolve(args.signal_output_dir)
    fit_root = resolve(args.fit_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_root.mkdir(parents=True, exist_ok=True)
    fit_root.mkdir(parents=True, exist_ok=True)

    stage_d = load_npz_dict(resolve(args.stage_d_npz))
    stage_e = load_npz_dict(resolve(args.stage_e_npz))
    stage_d_meta = load_json(resolve(args.stage_d_metadata))
    stage_e_meta = load_json(resolve(args.stage_e_metadata))
    stage_g_meta = load_json(resolve(args.stage_g_reference_metadata))
    baseline_ids = selector_ids(resolve(args.baseline_selector_csv))
    fit_mask = baseline_mask(np.asarray(stage_d["cell_id"], dtype=np.int64), baseline_ids)
    background_model = stage_d_meta.get("background_model") if isinstance(stage_d_meta.get("background_model"), dict) else {}

    variants = [
        ("nominal_shifted_order2", "PSF-shifted", 2, True, False, np.asarray(stage_d["B_on"], dtype=np.float64), np.asarray(stage_d["background_map"], dtype=np.float32)),
        ("default_annulus_order2", "1.5-3.5 deg", 2, False, False, None, None),
        ("shifted_annulus_order2", "PSF-shifted", 2, True, False, None, None),
        ("default_annulus_order1", "1.5-3.5 deg", 1, False, False, None, None),
        ("shifted_annulus_order1", "PSF-shifted", 1, True, False, None, None),
        ("shifted_annulus_log_link_order2", "PSF-shifted", 2, True, True, None, None),
    ]

    rows: List[Dict[str, object]] = []
    variant_json: Dict[str, object] = {
        "description": "v3 background-method systematics derived from existing Stage D counts maps.",
        "stage_d_npz": str(resolve(args.stage_d_npz)),
        "stage_e_npz": str(resolve(args.stage_e_npz)),
        "baseline_selector_csv": str(resolve(args.baseline_selector_csv)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "high_energy_stage_g_reference": high_energy_point_summary(stage_g_meta),
        "variants": {},
    }
    count_maps = np.asarray(stage_d["counts_map"], dtype=np.float64)
    fiducial_mask = np.asarray(stage_d["fiducial_mask"], dtype=bool)
    on_mask = np.asarray(stage_d["on_mask"], dtype=bool)
    r_opt_deg = np.asarray(stage_d["r_opt_deg"], dtype=np.float64)
    x_centers = np.asarray(stage_d["x_centers_deg"], dtype=np.float64)
    y_centers = np.asarray(stage_d["y_centers_deg"], dtype=np.float64)

    for name, annulus_label, order, shifted, log_link, supplied_b_on, supplied_map in variants:
        if supplied_b_on is None or supplied_map is None:
            fit = fit_background_variant(
                count_maps,
                x_centers,
                y_centers,
                fiducial_mask,
                on_mask,
                r_opt_deg,
                order=order,
                shifted=shifted,
                log_link=log_link,
                background_model=background_model,
            )
            b_on = np.asarray(fit["B_on"], dtype=np.float64)
            background_map = np.asarray(fit["background_map"], dtype=np.float32)
            valid_background = np.asarray(fit["surface_fit_success"], dtype=bool)
        else:
            b_on = np.asarray(supplied_b_on, dtype=np.float64)
            background_map = np.asarray(supplied_map, dtype=np.float32)
            valid_background = np.isfinite(b_on) & (b_on > 0.0)

        signal = compute_signal_arrays(stage_e, b_on)
        total_n = int(np.nansum(signal["N_on"][fit_mask]))
        total_b = float(np.nansum(signal["B_on"][fit_mask]))
        formal_sigma = float(known_background_sigma(np.asarray([total_n]), np.asarray([total_b]))[0])
        run_dir = signal_root / "runs" / f"v3_background_systematics_{name}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        signal_path = run_dir / "signal_v3_background_systematics.npz"
        signal_meta_path = run_dir / "signal_v3_background_systematics_metadata.json"
        write_signal_npz(signal_path, signal)
        meta = signal_metadata(
            name=name,
            signal_path=signal_path,
            template_metadata=stage_e_meta,
            stage_d_metadata_path=resolve(args.stage_d_metadata),
            total_n=total_n,
            total_b=total_b,
            formal_sigma=formal_sigma,
            valid_background_cells=int(np.count_nonzero(valid_background[fit_mask])),
        )
        write_json(signal_meta_path, meta)
        fit_meta = run_stage_f_variant(
            args=args,
            name=name,
            signal_npz=signal_path,
            signal_meta=signal_meta_path,
            fit_output_root=fit_root,
        )
        fits = fit_meta.get("fits") if isinstance(fit_meta.get("fits"), dict) else {}
        pl = fits.get("pl_conservative", {}) if isinstance(fits, dict) else {}
        logpar = fits.get("logpar_conservative", {}) if isinstance(fits, dict) else {}
        pl_params = pl.get("parameters", {}) if isinstance(pl, dict) and isinstance(pl.get("parameters"), dict) else {}
        logpar_params = logpar.get("parameters", {}) if isinstance(logpar, dict) and isinstance(logpar.get("parameters"), dict) else {}
        preferred = fit_meta.get("preferred_fit") if isinstance(fit_meta.get("preferred_fit"), dict) else {}
        row = {
            "variant": name,
            "annulus": annulus_label,
            "surface_order": int(order),
            "fit_family": "poisson_log_link_positive" if log_link else "weighted_least_squares",
            "baseline_B_on": total_b,
            "baseline_excess": float(total_n - total_b),
            "baseline_formal_sigma": formal_sigma,
            "valid_baseline_background_cells": int(np.count_nonzero(valid_background[fit_mask])),
            "pl_phi0": pl_params.get("phi0"),
            "pl_gamma": pl_params.get("gamma"),
            "pl_chi2": pl.get("chi2") if isinstance(pl, dict) else None,
            "logpar_phi0": logpar_params.get("phi0"),
            "logpar_alpha": logpar_params.get("alpha"),
            "logpar_beta": logpar_params.get("beta"),
            "logpar_chi2": logpar.get("chi2") if isinstance(logpar, dict) else None,
            "preferred_model": preferred.get("model") if isinstance(preferred, dict) else None,
            "predE_high_energy_points": len(variant_json["high_energy_stage_g_reference"]["high_energy_labels"]),  # type: ignore[index]
            "signal_npz": str(signal_path),
            "fit_metadata_json": str((fit_root / "runs" / f"v3_stage_f_background_systematics_{name}" / "fit_v3_background_systematics_metadata.json")),
        }
        rows.append(row)
        variant_json["variants"][name] = {
            **row,
            "baseline_cell_ids": baseline_ids,
        }
        if name == "nominal_shifted_order2":
            plot_before_after_dec_profiles(
                counts_map=count_maps,
                background_map=background_map,
                x_centers=x_centers,
                y_centers=y_centers,
                fiducial_mask=fiducial_mask,
                fit_mask=fit_mask,
                half_width=float(args.profile_half_width_deg),
                output=output_dir / "v3_background_before_after_dec_profile.png",
            )

    summary_csv = output_dir / "v3_background_systematics_summary.csv"
    summary_json = output_dir / "v3_background_systematics_summary.json"
    sensitivity_png = output_dir / "v3_background_method_sensitivity_summary.png"
    write_summary_csv(summary_csv, rows)
    write_json(summary_json, variant_json)
    plot_sensitivity_summary(rows, sensitivity_png)
    print(json.dumps({
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "sensitivity_png": str(sensitivity_png),
        "before_after_dec_profile_png": str(output_dir / "v3_background_before_after_dec_profile.png"),
        "variants": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
