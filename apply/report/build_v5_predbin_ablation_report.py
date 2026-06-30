#!/usr/bin/env python
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_HTML = REPO_ROOT / "apply/report/crab_sed_v5_predbin_ablation_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v5-predbin-ablation"
OFFICIAL_PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
MIGRATION_GROUPS_CSV = REPO_ROOT / "apply/report/assets/v5-migration-binning/v5_migration_binning_groups.csv"


STRATEGIES = {
    "baseline_v4": {
        "label": "baseline_v4",
        "selector": REPO_ROOT / "apply/config/cell_selector_v4_drop4_psfborrow.csv",
        "stage_b_summary": REPO_ROOT / "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate_summary.csv",
        "stage_f_npz": REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz",
        "stage_f_meta": REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_metadata.json",
        "stage_f_summary": REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_summary.csv",
        "stage_g_summary": REPO_ROOT / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4/sed_points_v4_aperture_conditioned_drop4_summary.csv",
    },
    "gap025": {
        "label": "gap025",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_predbin_gap025.csv",
        "stage_b_summary": REPO_ROOT / "apply/output/stage_b_v5_predbin_gap025/current/psf_v5_predbin_gap025_summary.csv",
        "stage_f_npz": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap025_aperture_conditioned/current/fit_v5_predbin_gap025_aperture_conditioned.npz",
        "stage_f_meta": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap025_aperture_conditioned/current/fit_v5_predbin_gap025_aperture_conditioned_metadata.json",
        "stage_f_summary": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap025_aperture_conditioned/current/fit_v5_predbin_gap025_aperture_conditioned_summary.csv",
        "stage_g_summary": REPO_ROOT / "apply/output/stage_g_v5_predbin_gap025_aperture_conditioned/current/sed_points_v5_predbin_gap025_aperture_conditioned_summary.csv",
    },
    "gap1": {
        "label": "gap1",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_predbin_gap1.csv",
        "stage_b_summary": REPO_ROOT / "apply/output/stage_b_v5_predbin_gap1/current/psf_v5_predbin_gap1_summary.csv",
        "stage_f_npz": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap1_aperture_conditioned/current/fit_v5_predbin_gap1_aperture_conditioned.npz",
        "stage_f_meta": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap1_aperture_conditioned/current/fit_v5_predbin_gap1_aperture_conditioned_metadata.json",
        "stage_f_summary": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap1_aperture_conditioned/current/fit_v5_predbin_gap1_aperture_conditioned_summary.csv",
        "stage_g_summary": REPO_ROOT / "apply/output/stage_g_v5_predbin_gap1_aperture_conditioned/current/sed_points_v5_predbin_gap1_aperture_conditioned_summary.csv",
    },
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def html_escape(value: object) -> str:
    return html.escape(str(value))


def relative_path(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start=REPORT_HTML.parent.resolve())
    except OSError:
        return str(path)


def latest_run_artifact(path: Path) -> Path:
    if path.exists():
        return path
    if path.parent.name != "current":
        return path
    runs_dir = path.parent.parent / "runs"
    if not runs_dir.exists():
        return path
    candidates = [candidate for candidate in runs_dir.glob(f"*/{path.name}") if candidate.exists()]
    if not candidates:
        return path
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def resolve_strategy_paths(config: Dict[str, Path]) -> Dict[str, Path]:
    resolved = dict(config)
    for key in ["stage_b_summary", "stage_f_npz", "stage_f_meta", "stage_f_summary", "stage_g_summary"]:
        resolved[key] = latest_run_artifact(config[key])
    return resolved


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_fit_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def stage_b_npz_from_summary(path: Path) -> Path:
    name = path.name
    if name.endswith("_summary.csv"):
        return path.with_name(f"{name[:-len('_summary.csv')]}.npz")
    return path.with_suffix(".npz")


def fit_params_from_meta(meta: Dict[str, object], model: str) -> Dict[str, object]:
    fits = meta.get("fits")
    if not isinstance(fits, dict):
        return {}
    for value in fits.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("model_name") or value.get("model") or "").lower() != model.lower():
            continue
        params = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
        errors = value.get("errors") if isinstance(value.get("errors"), dict) else {}
        return {
            "valid": value.get("valid"),
            "chi2": value.get("chi2"),
            "ndof": value.get("ndof"),
            "p_value": value.get("p_value"),
            "parameters": params,
            "errors": errors,
        }
    direct = fits.get(model) or fits.get(model.lower()) or fits.get(model.upper())
    return direct if isinstance(direct, dict) else {}


def max_abs_pull(fit_npz: Dict[str, np.ndarray], preferred: str = "logpar") -> Optional[float]:
    candidates = [
        f"{preferred}_conservative_pull",
        f"{preferred}_sqrt_n_pull",
        "logpar_conservative_pull",
        "logpar_pull",
    ]
    for key in candidates:
        if key in fit_npz:
            values = np.asarray(fit_npz[key], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                return float(np.nanmax(np.abs(finite)))
    return None


def load_strategy(name: str, config: Dict[str, Path]) -> Dict[str, object]:
    config = resolve_strategy_paths(config)
    selector_rows = read_csv_rows(config["selector"])
    included = [row for row in selector_rows if truthy(row.get("include"))]
    psf_rows = read_csv_rows(config["stage_b_summary"])
    fit_meta = read_json(config["stage_f_meta"])
    fit_npz = load_fit_npz(config["stage_f_npz"])
    stage_g_rows = read_csv_rows(config["stage_g_summary"])
    logpar = fit_params_from_meta(fit_meta, "logpar")
    pl = fit_params_from_meta(fit_meta, "pl")
    risk_rows = []
    for row in psf_rows:
        missing_mass = finite_float(row.get("theta_missing_crab_probability_mass")) or 0.0
        neff = finite_float(row.get("effective_events")) or 0.0
        containment_warning = truthy(row.get("containment_warning"))
        risk = missing_mass > 0.0 or neff < 200.0 or containment_warning
        if risk:
            risk_rows.append(row)
    return {
        "name": name,
        "config": {key: str(value) for key, value in config.items()},
        "selector_rows": selector_rows,
        "included_cells": included,
        "psf_rows": psf_rows,
        "psf_risk_rows": risk_rows,
        "fit_meta": fit_meta,
        "fit_npz": fit_npz,
        "stage_g_rows": stage_g_rows,
        "logpar": logpar,
        "pl": pl,
        "max_pull": max_abs_pull(fit_npz),
        "low_nhit_pass5_ratio": low_nhit_pass5_ratio(stage_g_rows),
        "status": "complete" if config["stage_f_meta"].exists() and config["stage_g_summary"].exists() else "pending",
    }


def official_pass5_points() -> List[Dict[str, float]]:
    rows = read_csv_rows(OFFICIAL_PASS5_CSV)
    out: List[Dict[str, float]] = []
    for row in rows:
        energy = finite_float(row.get("energy_tev") or row.get("E_TeV") or row.get("e_ref_tev"))
        flux = finite_float(row.get("e2_dnde") or row.get("E2_dnde") or row.get("E2dnde"))
        err = finite_float(row.get("e2_dnde_err") or row.get("E2_dnde_err") or row.get("E2dnde_err"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        dnde_err = finite_float(row.get("flux_per_tev_cm2_s_err") or row.get("flux_err_per_tev_cm2_s"))
        if flux is None and energy is not None and dnde is not None:
            flux = energy * energy * dnde
            err = None if dnde_err is None else energy * energy * dnde_err
        if energy is not None and flux is not None:
            out.append({"energy": energy, "flux": flux, "err": err if err is not None else 0.0})
    return out


def fit_logpar_to_e2_points(points: Sequence[Dict[str, float]], pivot_tev: float = 3.0) -> Optional[Dict[str, object]]:
    energy = np.asarray([point["energy"] for point in points], dtype=np.float64)
    e2_flux = np.asarray([point["flux"] for point in points], dtype=np.float64)
    valid = np.isfinite(energy) & np.isfinite(e2_flux) & (energy > 0.0) & (e2_flux > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    x = np.log(energy[valid] / float(pivot_tev))
    y = np.log(e2_flux[valid] / (energy[valid] * energy[valid]))
    c2, c1, c0 = np.polyfit(x, y, 2)
    return {
        "parameters": {
            "phi0": float(np.exp(c0)),
            "alpha": float(-c1),
            "beta": float(-c2),
        },
        "pivot_tev": float(pivot_tev),
        "n_points": int(np.count_nonzero(valid)),
        "fit_note": "unweighted log-space fit to official pass5 SED points",
    }


def low_nhit_pass5_ratio(stage_g_rows: Sequence[Dict[str, str]]) -> Optional[float]:
    nhit_rows = [row for row in stage_g_rows if row.get("grouping") == "nhit"]
    if not nhit_rows:
        return None
    nhit_rows = sorted(
        nhit_rows,
        key=lambda row: interval_key(str(row.get("group_label") or row.get("nhit_bin") or "")),
    )
    low_row = nhit_rows[0]
    energy = finite_float(low_row.get("effective_energy_tev"))
    flux = finite_float(low_row.get("E2_dnde"))
    official = official_pass5_points()
    if energy is None or flux is None or not official:
        return None
    nearest = min(official, key=lambda point: abs(math.log(point["energy"] / energy)))
    if nearest["flux"] <= 0.0:
        return None
    return float(flux / nearest["flux"])


def pass5_point_fit() -> Optional[Dict[str, object]]:
    return fit_logpar_to_e2_points(official_pass5_points())


def logpar_flux(energy_tev: np.ndarray, params: Dict[str, object], pivot_tev: float = 3.0) -> Optional[np.ndarray]:
    values = params.get("parameters") if isinstance(params.get("parameters"), dict) else {}
    phi0 = finite_float(values.get("phi0"))
    alpha = finite_float(values.get("alpha"))
    beta = finite_float(values.get("beta"))
    if phi0 is None or alpha is None or beta is None:
        return None
    x = np.asarray(energy_tev, dtype=np.float64) / float(pivot_tev)
    dnde = phi0 * np.power(x, -(alpha + beta * np.log(x)))
    return energy_tev * energy_tev * dnde


def pass5_fit_flux_at(energy_tev: float, pass5_fit: Optional[Dict[str, object]]) -> Optional[float]:
    if pass5_fit is None:
        return None
    curve = logpar_flux(np.asarray([energy_tev], dtype=np.float64), pass5_fit, pivot_tev=float(pass5_fit.get("pivot_tev", 3.0)))
    if curve is None or not np.isfinite(curve[0]) or curve[0] <= 0.0:
        return None
    return float(curve[0])


def pass5_interp_flux_at(energy_tev: float) -> Optional[float]:
    official = official_pass5_points()
    pairs = sorted(
        (math.log10(point["energy"]), math.log10(point["flux"]))
        for point in official
        if point["energy"] > 0.0 and point["flux"] > 0.0
    )
    if not pairs or energy_tev <= 0.0:
        return None
    lx = math.log10(energy_tev)
    if lx <= pairs[0][0]:
        return 10.0 ** pairs[0][1]
    if lx >= pairs[-1][0]:
        return 10.0 ** pairs[-1][1]
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        if x0 <= lx <= x1:
            frac = (lx - x0) / (x1 - x0)
            return 10.0 ** (y0 + frac * (y1 - y0))
    return 10.0 ** pairs[-1][1]


def plot_sed_overlay(strategies: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=150)
    colors = {"baseline_v4": "#2563eb", "gap025": "#0f766e", "gap1": "#b45309"}
    energy = np.logspace(math.log10(0.2), math.log10(120.0), 240)
    for name, payload in strategies.items():
        curve = logpar_flux(energy, payload.get("logpar", {}))
        if curve is not None:
            ax.plot(energy, curve, color=colors.get(name, "black"), linewidth=1.8, label=f"{name} LogPar")
        rows = [row for row in payload.get("stage_g_rows", []) if row.get("grouping") == "nhit"]
        x: List[float] = []
        y: List[float] = []
        yerr: List[float] = []
        for row in rows:
            ex = finite_float(row.get("effective_energy_tev"))
            ey = finite_float(row.get("E2_dnde"))
            ee = finite_float(row.get("E2_dnde_err"))
            if ex is not None and ey is not None:
                x.append(ex)
                y.append(ey)
                yerr.append(0.0 if ee is None else ee)
        if x:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                markersize=4.2,
                capsize=2,
                color=colors.get(name, "black"),
                alpha=0.9,
                label=f"{name} Nhit points",
            )
    official = official_pass5_points()
    if official:
        ax.errorbar(
            [p["energy"] for p in official],
            [p["flux"] for p in official],
            yerr=[p["err"] for p in official],
            fmt="s",
            markersize=3.2,
            capsize=1.5,
            color="#6b7280",
            alpha=0.65,
            label="official pass5",
        )
        pass5_fit = fit_logpar_to_e2_points(official)
        if pass5_fit:
            pass5_curve = logpar_flux(energy, pass5_fit, pivot_tev=float(pass5_fit.get("pivot_tev", 3.0)))
            if pass5_curve is not None:
                ax.plot(
                    energy,
                    pass5_curve,
                    color="#c026d3",
                    linewidth=2.2,
                    linestyle="--",
                    label="official pass5 point-fit LogPar",
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2 dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Crab SED v5 PredE Binning Ablation")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_psf_heatmap(payload: Dict[str, object], output_path: Path) -> None:
    rows = payload.get("psf_rows", [])
    if not rows:
        return
    plt = setup_matplotlib()
    nhit_bins = sorted({row["nhit_bin"] for row in rows}, key=interval_key)
    pred_bins = sorted({row["predE_bin"] for row in rows}, key=interval_key)
    values = np.full((len(nhit_bins), len(pred_bins)), np.nan, dtype=np.float64)
    for row in rows:
        i = nhit_bins.index(row["nhit_bin"])
        j = pred_bins.index(row["predE_bin"])
        neff = finite_float(row.get("effective_events"))
        missing = finite_float(row.get("theta_missing_crab_probability_mass")) or 0.0
        residual = finite_float(row.get("tail_weight_fraction_above_core_fit")) or 0.0
        score = 0.0
        if neff is None or neff <= 0.0:
            score = 3.0
        else:
            score += max(0.0, math.log10(200.0 / neff))
        score += 5.0 * missing + residual
        values[i, j] = score
    fig, ax = plt.subplots(figsize=(1.25 * len(pred_bins) + 2.4, 0.58 * len(nhit_bins) + 2.0), dpi=150)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_title(f"{payload['name']} PSF risk score")
    ax.set_xlabel("log10(E_pred / GeV) bin")
    ax.set_ylabel("Nhit bin")
    fig.colorbar(im, ax=ax, shrink=0.82, label="risk score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def rayleigh_pdf_deg(r_deg: np.ndarray, sigma_rad: float) -> np.ndarray:
    r_rad = np.radians(r_deg)
    pdf_per_rad = (r_rad / (sigma_rad * sigma_rad)) * np.exp(-0.5 * (r_rad / sigma_rad) ** 2)
    return pdf_per_rad * (math.pi / 180.0)


def plot_psf_profile_grid(payload: Dict[str, object], output_path: Path) -> bool:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        return False
    summary_path = Path(str(config.get("stage_b_summary", "")))
    npz_path = stage_b_npz_from_summary(summary_path)
    if not summary_path.exists() or not npz_path.exists():
        return False

    with np.load(npz_path, allow_pickle=False) as data:
        cell_ids = np.asarray(data["cell_id"], dtype=np.int64)
        nhit_bins = [str(value) for value in data["nhit_bin"]]
        pred_bins = [str(value) for value in data["predE_bin"]]
        profile_edges_deg = np.asarray(data["profile_edges_deg"], dtype=np.float64)
        profile_density = np.asarray(data["profile_density"], dtype=np.float64)

    rows = read_csv_rows(summary_path)
    row_by_cell_id = {int(row["cell_id"]): row for row in rows if row.get("cell_id")}
    fit_cell_ids = {
        int(row["cell_id"])
        for row in payload.get("included_cells", [])
        if isinstance(row, dict) and row.get("cell_id") and truthy(row.get("include"))
    }
    ordered_nhit = sorted(set(nhit_bins), key=interval_key)
    ordered_pred = sorted(set(pred_bins), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    centers = 0.5 * (profile_edges_deg[:-1] + profile_edges_deg[1:])
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
            row = row_by_cell_id.get(cell_id, {})
            is_fit_cell = cell_id in fit_cell_ids
            if is_fit_cell:
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
            density = profile_density[idx]
            ax.step(centers, density, where="mid", color="#1f4e79", linewidth=0.9)
            has_profile = bool(np.isfinite(density).any() and np.nansum(density) > 0.0)
            events = finite_float(row.get("events"))
            sigma_rad = finite_float(row.get("sigma_rad"))
            sigma_deg = finite_float(row.get("sigma_deg"))
            if sigma_rad is None and sigma_deg is not None:
                sigma_rad = math.radians(sigma_deg)
            if has_profile and events is not None and events > 0.0 and sigma_rad is not None and sigma_rad > 0.0:
                ax.plot(centers, rayleigh_pdf_deg(centers, sigma_rad), color="#c9501a", linewidth=0.8, alpha=0.9)
            r_opt = finite_float(row.get("r_opt_deg"))
            if has_profile and events is not None and events > 0.0 and r_opt is not None:
                ax.axvline(r_opt, color="#444444", linewidth=0.7, linestyle="--")
            ax.set_title(f"cell {cell_id}: {pred}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            ax.grid(alpha=0.22, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(nhit, fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("r (deg)", fontsize=6.7)

    handles = [
        Line2D([0], [0], color="#1f4e79", linewidth=0.9, label="MC histogram"),
        Line2D([0], [0], color="#c9501a", linewidth=0.9, label="Rayleigh fit"),
        Line2D([0], [0], color="#444444", linewidth=0.8, linestyle="--", label="r_opt"),
        Patch(facecolor="#ecfdf5", edgecolor="#059669", label="included in fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle(f"{payload['name']} Stage B weighted radial PSF profiles", fontsize=11, y=0.999)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def parse_interval(label: str) -> Tuple[Optional[float], Optional[float]]:
    label = label.strip()
    if label.startswith("[") and label.endswith(")"):
        low, high = label[1:-1].split(",", 1)
        return float(low), float(high)
    if label.startswith("<"):
        return None, float(label[1:])
    if label.startswith(">="):
        return float(label[2:]), None
    return None, None


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def fmt_float(value: object, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "pending"
    return f"{number:.{digits}g}"


def strategy_summary_table(strategies: Dict[str, Dict[str, object]]) -> str:
    rows: List[str] = []
    for name, payload in strategies.items():
        logpar = payload.get("logpar", {})
        params = logpar.get("parameters") if isinstance(logpar, dict) and isinstance(logpar.get("parameters"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{html_escape(name)}</td>"
            f"<td>{html_escape(payload.get('status'))}</td>"
            f"<td class=\"num\">{len(payload.get('included_cells', []))}</td>"
            f"<td class=\"num\">{len(payload.get('psf_risk_rows', []))}</td>"
            f"<td class=\"num\">{fmt_float(params.get('phi0'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(params.get('alpha'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(params.get('beta'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(logpar.get('chi2'), 5)}/{fmt_float(logpar.get('ndof'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(payload.get('max_pull'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(payload.get('low_nhit_pass5_ratio'), 4)}</td>"
            "</tr>"
        )
    return "".join(rows)


def psf_table(payload: Dict[str, object], limit: int = 24) -> str:
    rows: List[str] = []
    for row in list(payload.get("psf_risk_rows", []))[:limit]:
        rows.append(
            "<tr>"
            f"<td>{html_escape(row.get('cell_id'))}</td>"
            f"<td>{html_escape(row.get('nhit_bin'))}</td>"
            f"<td>{html_escape(row.get('predE_bin'))}</td>"
            f"<td class=\"num\">{fmt_float(row.get('effective_events'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(row.get('theta_missing_crab_probability_mass'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(row.get('sigma_deg'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(row.get('r_opt_deg'), 4)}</td>"
            "</tr>"
        )
    return "".join(rows) if rows else "<tr><td colspan=\"7\">No PSF risk rows found or Stage B pending.</td></tr>"


def fmt_percent(value: object, digits: int = 1) -> str:
    number = finite_float(value)
    if number is None:
        return "pending"
    return f"{100.0 * number:.{digits}f}%"


def fmt_sigma(value: object, digits: int = 2) -> str:
    number = finite_float(value)
    if number is None:
        return "pending"
    return f"{number:.{digits}g}σ"


def flux_point_table_from_rows(rows: Sequence[Dict[str, object]], source: str) -> str:
    if not rows:
        return "<p>Stage G Nhit points pending.</p>"
    body: List[str] = []
    for row in rows:
        point_label = str(row.get("point") or row.get("group_label") or row.get("cell_ids") or "").replace(";", "+")
        nhit_label = str(row.get("nhit_bin") or row.get("nhit_span") or row.get("group_label") or "")
        energy = finite_float(row.get("E_med_TeV") or row.get("E_p50_TeV") or row.get("true_energy_p50_tev") or row.get("effective_energy_tev"))
        flux = finite_float(row.get("E2_dnde"))
        flux_err = finite_float(row.get("E2_dnde_err"))
        rel_err = finite_float(row.get("relative_error"))
        if rel_err is None:
            rel_err = None if flux is None or flux <= 0.0 or flux_err is None else flux_err / flux
        significance = finite_float(row.get("significance"))
        if significance is None:
            significance = None if rel_err is None or rel_err <= 0.0 else 1.0 / rel_err
        pass5_ratio = finite_float(row.get("pass5_ratio") or row.get("ratio_to_pass5"))
        if pass5_ratio is None:
            pass5_flux = pass5_interp_flux_at(energy) if energy is not None else None
            pass5_ratio = None if flux is None or pass5_flux is None else flux / pass5_flux
        body.append(
            "<tr>"
            f"<td>{html_escape(point_label)}</td>"
            f"<td>{html_escape(nhit_label)}</td>"
            f"<td class=\"num\">{fmt_float(energy, 3)}</td>"
            f"<td class=\"num\">{fmt_sigma(significance, 3)}</td>"
            f"<td class=\"num\">{fmt_percent(rel_err, 1)}</td>"
            f"<td class=\"num\">{fmt_float(pass5_ratio, 3)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact"><table>'
        "<thead><tr><th>点</th><th>Nhit bin</th><th class=\"num\">E_med [TeV]</th><th class=\"num\">significance</th><th class=\"num\">相对误差</th><th class=\"num\">pass5 ratio</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
        f'<p class="caption-note">source: {html_escape(source)}</p>'
    )


def conservative_7bin_rows() -> List[Dict[str, object]]:
    rows = read_csv_rows(MIGRATION_GROUPS_CSV)
    out: List[Dict[str, object]] = []
    for row in rows:
        if row.get("analysis") != "conservative_7bin":
            continue
        excess = finite_float(row.get("excess"))
        error = finite_float(row.get("error"))
        flux = finite_float(row.get("E2_dnde"))
        flux_err = finite_float(row.get("E2_dnde_err"))
        out.append(
            {
                "point": row.get("group_label"),
                "nhit_span": row.get("nhit_span"),
                "E_p50_TeV": row.get("E_p50_TeV"),
                "significance": (
                    None
                    if excess is None or error in (None, 0.0)
                    else excess / error
                ),
                "relative_error": (
                    None
                    if flux is None or flux_err is None or flux <= 0.0
                    else flux_err / flux
                ),
                "ratio_to_pass5": row.get("ratio_to_pass5"),
                "E2_dnde": row.get("E2_dnde"),
                "E2_dnde_err": row.get("E2_dnde_err"),
            }
        )
    return out


def nhit_flux_point_table(name: str, payload: Dict[str, object]) -> str:
    if name == "baseline_v4" and MIGRATION_GROUPS_CSV.exists():
        return flux_point_table_from_rows(conservative_7bin_rows(), "v5_migration_binning_groups.csv / conservative_7bin")
    rows = [
        row
        for row in payload.get("stage_g_rows", [])
        if isinstance(row, dict) and row.get("grouping") == "nhit"
    ]
    rows = sorted(rows, key=lambda row: interval_key(str(row.get("group_label") or "")))
    normalized = [
        {
            **row,
            "point": str(row.get("cell_ids") or "").replace(";", "+"),
            "nhit_bin": row.get("group_label"),
        }
        for row in rows
    ]
    return flux_point_table_from_rows(normalized, "Stage G summary.csv / grouping=nhit")


def nhit_flux_point_sections(strategies: Dict[str, Dict[str, object]]) -> str:
    sections: List[str] = []
    for name, payload in strategies.items():
        sections.append(
            f"""
<section>
<h3>{html_escape(name)}</h3>
{nhit_flux_point_table(name, payload)}
</section>
"""
        )
    return "".join(sections)


def write_report(strategies: Dict[str, Dict[str, object]], figures: Dict[str, Path]) -> None:
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    generated = time_now_string()
    psf_sections: List[str] = []
    for name, payload in strategies.items():
        fig = figures.get(f"psf_{name}")
        fig_html = f'<figure><img src="{html_escape(relative_path(fig))}" alt="{html_escape(name)} PSF risk"></figure>' if fig and fig.exists() else "<p>PSF heatmap pending.</p>"
        profiles = figures.get(f"psf_profiles_{name}")
        profiles_html = (
            f'<figure><img src="{html_escape(relative_path(profiles))}" alt="{html_escape(name)} fit-cell shaded PSF radial profiles"><figcaption>{html_escape(name)} weighted radial PSF profiles. Green shaded panels are cells included in the final SED fit.</figcaption></figure>'
            if profiles and profiles.exists()
            else ""
        )
        psf_sections.append(
            f"""
<section>
<h3>{html_escape(name)}</h3>
{fig_html}
{profiles_html}
<div class="table-wrap"><table>
<thead><tr><th>cell</th><th>Nhit</th><th>predE</th><th class="num">Neff</th><th class="num">missing theta</th><th class="num">sigma deg</th><th class="num">r_opt deg</th></tr></thead>
<tbody>{psf_table(payload)}</tbody>
</table></div>
</section>
"""
        )
    sed_fig = figures.get("sed_overlay")
    sed_html = f'<figure><img src="{html_escape(relative_path(sed_fig))}" alt="SED overlay"></figure>' if sed_fig and sed_fig.exists() else "<p>SED overlay pending.</p>"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crab SED v5 PredE 分箱消融报告</title>
<style>
body {{ margin:0; background:#f7f8f9; color:#17212b; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.62; }}
main {{ max-width:1240px; margin:0 auto; padding:38px 20px 64px; }}
h1 {{ margin:0 0 10px; font-size:36px; line-height:1.15; }}
h2 {{ margin:38px 0 12px; font-size:25px; border-bottom:1px solid #d7dee3; padding-bottom:8px; }}
h3 {{ margin:26px 0 10px; font-size:19px; }}
.lead {{ color:#53606a; max-width:960px; font-size:16px; }}
.table-wrap {{ overflow-x:auto; border:1px solid #d7dee3; border-radius:8px; background:white; margin:14px 0; }}
table {{ width:100%; border-collapse:collapse; min-width:920px; font-size:14px; }}
.compact table {{ min-width:760px; }}
th,td {{ border-bottom:1px solid #d7dee3; padding:9px 11px; text-align:left; vertical-align:top; }}
th {{ background:#eef2f4; white-space:nowrap; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
figure {{ margin:16px 0; padding:12px; border:1px solid #d7dee3; border-radius:8px; background:white; }}
figure img {{ display:block; width:100%; height:auto; }}
figcaption {{ margin-top:8px; color:#53606a; font-size:13px; }}
.caption-note {{ margin:-6px 0 10px; color:#66727c; font-size:12px; }}
code {{ background:#edf1f3; border-radius:4px; padding:1px 4px; }}
.note {{ border-left:4px solid #0f766e; background:white; border-radius:8px; padding:14px 16px; margin:16px 0; }}
</style>
</head>
<body><main>
<h1>Crab SED v5 PredE 分箱消融报告</h1>
<p class="lead">对比 baseline_v4、gap025 与 gap1 三套 <code>log10(E_pred/GeV)</code> 分箱。gap025/gap1 使用独立 prefit MC selector，主 fit 排除 <code>&lt;2</code> 和 <code>&gt;=6</code> 尾箱；PSF 风险仅标注，不后验删除。</p>
<p class="lead">Generated: {html_escape(generated)}</p>

<section>
<h2>汇总</h2>
<div class="table-wrap"><table>
<thead><tr><th>strategy</th><th>status</th><th class="num">fit cells</th><th class="num">PSF risk cells</th><th class="num">LogPar phi0</th><th class="num">alpha</th><th class="num">beta</th><th class="num">chi2/ndof</th><th class="num">max pull</th><th class="num">low-Nhit pass5 ratio</th></tr></thead>
<tbody>{strategy_summary_table(strategies)}</tbody>
</table></div>
<div class="note">Status 为 pending 时表示对应 Slurm 全量产物还没生成；脚本会在产物出现后自动纳入同一张报告。</div>
</section>

<section>
<h2>SED Overlay</h2>
{sed_html}
<div class="note">The official pass5 curve is an unweighted log-space LogPar fit to the plotted official pass5 SED points.</div>
</section>

<section>
<h2>Final Nhit Flux Point Diagnostics</h2>
<div class="note">baseline_v4 uses the <code>conservative_7bin</code> final flux points from the migration report; gap025 and gap1 use their final Stage G <code>grouping=nhit</code> points. <code>相对误差</code> is <code>E2_dnde_err / E2_dnde</code>; <code>significance</code> is <code>excess/error</code> for baseline_v4 and equivalent flux/error for the Stage G rows. <code>pass5 ratio</code> uses log-log interpolation of the official pass5 SED points at the listed median/effective energy.</div>
{nhit_flux_point_sections(strategies)}
</section>

<section>
<h2>Stage B Rayleigh PSF Diagnostics</h2>
{''.join(psf_sections)}
</section>

<section>
<h2>Artifacts</h2>
<div class="table-wrap"><table>
<thead><tr><th>strategy</th><th>selector</th><th>Stage F</th><th>Stage G</th></tr></thead>
<tbody>
{''.join(
    '<tr>'
    f'<td>{html_escape(name)}</td>'
    f'<td><code>{html_escape(payload["config"]["selector"])}</code></td>'
    f'<td><code>{html_escape(payload["config"]["stage_f_meta"])}</code></td>'
    f'<td><code>{html_escape(payload["config"]["stage_g_summary"])}</code></td>'
    '</tr>'
    for name, payload in strategies.items()
)}
</tbody>
</table></div>
</section>
</main></body></html>
"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")


def time_now_string() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    strategies = {name: load_strategy(name, config) for name, config in STRATEGIES.items()}
    figures: Dict[str, Path] = {}
    sed_overlay = ASSET_DIR / "v5_predbin_ablation_sed_overlay.png"
    plot_sed_overlay(strategies, sed_overlay)
    figures["sed_overlay"] = sed_overlay
    for name, payload in strategies.items():
        out = ASSET_DIR / f"{name}_psf_risk_heatmap.png"
        plot_psf_heatmap(payload, out)
        if out.exists():
            figures[f"psf_{name}"] = out
        profiles = ASSET_DIR / f"{name}_psf_radial_profiles_fit_shaded.png"
        if plot_psf_profile_grid(payload, profiles):
            figures[f"psf_profiles_{name}"] = profiles
    write_report(strategies, figures)
    print(f"Wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
