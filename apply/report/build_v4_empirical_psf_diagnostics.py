#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "apply/report/assets/v4-empirical-psf"
DEFAULT_STAGE_F_META = (
    REPO_ROOT
    / "apply/output/stage_f_v4_drop4_annnorm/runs/v4_stage_f_annnorm_drop_cells_4_17_39_43"
    / "fit_v4_drop4_annnorm_metadata.json"
)
DEFAULT_PSF_NPZ = (
    REPO_ROOT
    / "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate.npz"
)

PROFILE_MAX_DEG = 4.0
PROFILE_BIN_WIDTH_DEG = 0.1
FIDUCIAL_RHO_DEG = 6.0
SINGLE_FIT_MAX_DEG = 3.0
MIN_N_ON = 200.0
MIN_EXCESS = 100.0
MIN_SIGNIFICANCE = 5.0
R68_FACTOR = math.sqrt(-2.0 * math.log(1.0 - 0.68))
PSF_RISK_CELLS = {39, 52, 65}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build observed-data empirical PSF diagnostics for the v4 drop4 fit cells."
    )
    parser.add_argument("--stage-f-metadata", type=str, default=str(DEFAULT_STAGE_F_META))
    parser.add_argument("--psf-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--profile-max-deg", type=float, default=PROFILE_MAX_DEG)
    parser.add_argument("--profile-bin-width-deg", type=float, default=PROFILE_BIN_WIDTH_DEG)
    parser.add_argument("--fiducial-rho-deg", type=float, default=FIDUCIAL_RHO_DEG)
    parser.add_argument("--single-fit-max-deg", type=float, default=SINGLE_FIT_MAX_DEG)
    return parser.parse_args()


def load_pil():
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


def font(size: int, *, bold: bool = False):
    _, _, ImageFont = load_pil()
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(draw: Any, xy: tuple[float, float], text: str, *, fill: str = "#111827", size: int = 11, bold: bool = False) -> None:
    draw.text(xy, text, fill=fill, font=font(size, bold=bold))


def finite_range(arrays: list[np.ndarray], *, include_zero: bool = True, pad_fraction: float = 0.08) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for array in arrays:
        a = np.asarray(array, dtype=np.float64)
        values.append(a[np.isfinite(a)])
    if include_zero:
        values.append(np.asarray([0.0], dtype=np.float64))
    merged = np.concatenate([v for v in values if v.size]) if any(v.size for v in values) else np.asarray([0.0, 1.0])
    lo = float(np.nanmin(merged))
    hi = float(np.nanmax(merged))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    span = hi - lo
    return lo - span * pad_fraction, hi + span * pad_fraction


def polyline_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    rect: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = rect
    points: list[tuple[int, int]] = []
    for xv, yv in zip(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)):
        if not (math.isfinite(float(xv)) and math.isfinite(float(yv))):
            continue
        xp = x0 + (float(xv) - x_min) / max(x_max - x_min, 1.0e-9) * (x1 - x0)
        yp = y1 - (float(yv) - y_min) / max(y_max - y_min, 1.0e-9) * (y1 - y0)
        points.append((int(round(xp)), int(round(yp))))
    return points


def draw_axes(draw: Any, rect: tuple[int, int, int, int], *, x_label: str | None = None, y_label: str | None = None) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="#d1d5db", width=1)
    for frac in [0.25, 0.5, 0.75]:
        x = int(round(x0 + frac * (x1 - x0)))
        y = int(round(y0 + frac * (y1 - y0)))
        draw.line([(x, y0), (x, y1)], fill="#eef2f7", width=1)
        draw.line([(x0, y), (x1, y)], fill="#eef2f7", width=1)
    if x_label:
        draw_text(draw, (x0, y1 + 6), x_label, fill="#4b5563", size=9)
    if y_label:
        draw_text(draw, (x0 - 36, y0 + 4), y_label, fill="#4b5563", size=9)


def draw_line(
    draw: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rect: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    fill: str,
    width: int = 2,
) -> None:
    pts = polyline_points(x, y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, rect=rect)
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width, joint="curve")


def draw_hline(
    draw: Any,
    value: float,
    *,
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    fill: str = "#9ca3af",
    width: int = 1,
) -> None:
    x0, y0, x1, y1 = rect
    if not (y_min <= value <= y_max):
        return
    y = int(round(y1 - (value - y_min) / max(y_max - y_min, 1.0e-9) * (y1 - y0)))
    draw.line([(x0, y), (x1, y)], fill=fill, width=width)


def draw_vline(
    draw: Any,
    value: float,
    *,
    rect: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    fill: str = "#9ca3af",
    width: int = 1,
) -> None:
    x0, y0, x1, y1 = rect
    if not (x_min <= value <= x_max):
        return
    x = int(round(x0 + (value - x_min) / max(x_max - x_min, 1.0e-9) * (x1 - x0)))
    draw.line([(x, y0), (x, y1)], fill=fill, width=width)


def save_canvas(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None or str(path_like).strip() == "":
        return None
    path = Path(path_like)
    if path.exists():
        return path
    if path.is_absolute():
        parts = path.parts
        if "apply" in parts:
            idx = parts.index("apply")
            local = REPO_ROOT.joinpath(*parts[idx:])
            if local.exists():
                return local
            return local
    local = REPO_ROOT / path
    return local


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    out = finite(value)
    if out is None:
        return "n/a"
    return f"{out:.{digits}g}"


def normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    peak = float(np.nanmax(finite_values)) if finite_values.size else float("nan")
    if not math.isfinite(peak) or peak <= 0.0:
        return np.full(values.shape, np.nan, dtype=np.float64)
    return values / peak


def shell_probabilities(edges: np.ndarray, sigma_deg: float) -> np.ndarray:
    sigma = max(float(sigma_deg), 1.0e-6)
    return np.exp(-0.5 * (edges[:-1] / sigma) ** 2) - np.exp(-0.5 * (edges[1:] / sigma) ** 2)


def fit_rayleigh_shells(
    edges: np.ndarray,
    excess_profile: np.ndarray,
    variance_profile: np.ndarray,
    fit_radius_deg: float,
) -> dict[str, float | str]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    use = (
        np.isfinite(excess_profile)
        & np.isfinite(variance_profile)
        & (variance_profile > 0.0)
        & (centers <= float(fit_radius_deg))
    )
    if np.count_nonzero(use) < 4:
        return {"status": "not_fit", "reason": "too_few_fit_bins"}
    y = np.asarray(excess_profile[use], dtype=np.float64)
    var = np.maximum(np.asarray(variance_profile[use], dtype=np.float64), 1.0)
    fit_edges = np.concatenate([edges[:-1][use], [edges[1:][use][-1]]])
    sigma_grid = np.linspace(0.05, 3.0, 900, dtype=np.float64)
    best: tuple[float, float, float] | None = None
    for sigma in sigma_grid:
        p = shell_probabilities(fit_edges, float(sigma))
        denom = float(np.sum((p * p) / var))
        if denom <= 0.0:
            continue
        amp = float(np.sum(y * p / var) / denom)
        amp = max(amp, 0.0)
        chi2 = float(np.sum(((y - amp * p) ** 2) / var))
        if best is None or chi2 < best[0]:
            best = (chi2, amp, float(sigma))
    if best is None:
        return {"status": "not_fit", "reason": "no_valid_sigma_grid"}
    chi2, amp, sigma = best
    ndof = max(int(np.count_nonzero(use)) - 2, 1)
    model = amp * shell_probabilities(edges, sigma)
    residual = np.asarray(excess_profile, dtype=np.float64) - model
    return {
        "status": "fit",
        "reason": "",
        "amplitude": amp,
        "sigma_obs_deg": sigma,
        "r68_obs_deg": R68_FACTOR * sigma,
        "chi2": chi2,
        "ndof": ndof,
        "chi2_over_ndof": chi2 / ndof,
        "model_profile": model,
        "residual_profile": residual,
    }


def reliability(n_on: float, b_on: float, excess: float) -> tuple[float, bool, str]:
    denom = math.sqrt(max(n_on + b_on, 1.0))
    significance = excess / denom
    reasons: list[str] = []
    if n_on < MIN_N_ON:
        reasons.append("N_on_lt_200")
    if excess < MIN_EXCESS:
        reasons.append("excess_lt_100")
    if significance < MIN_SIGNIFICANCE:
        reasons.append("significance_lt_5")
    return significance, not reasons, ";".join(reasons)


def selected_cell_ids(stage_f_meta: dict[str, Any]) -> list[int]:
    validation = stage_f_meta.get("validation", {}) if isinstance(stage_f_meta.get("validation"), dict) else {}
    subset = validation.get("cell_subset", {}) if isinstance(validation.get("cell_subset"), dict) else {}
    ids = subset.get("included_cell_ids")
    if isinstance(ids, list) and ids:
        return [int(v) for v in ids]
    raise ValueError("Stage F metadata does not contain validation.cell_subset.included_cell_ids")


def load_inputs(stage_f_meta_path: Path, psf_npz_override: Path | None) -> dict[str, Any]:
    stage_f_meta = load_json(stage_f_meta_path)
    inputs = stage_f_meta.get("inputs", {}) if isinstance(stage_f_meta.get("inputs"), dict) else {}
    signal_npz = resolve_path(inputs.get("signal_npz"))
    signal_meta_path = resolve_path(inputs.get("signal_metadata_json"))
    selector_csv = resolve_path(inputs.get("cell_subset_csv"))
    if signal_npz is None or not signal_npz.exists():
        raise FileNotFoundError(f"Missing signal NPZ from Stage F metadata: {signal_npz}")
    if signal_meta_path is None or not signal_meta_path.exists():
        raise FileNotFoundError(f"Missing signal metadata from Stage F metadata: {signal_meta_path}")

    signal_meta = load_json(signal_meta_path)
    signal_inputs = signal_meta.get("inputs", {}) if isinstance(signal_meta.get("inputs"), dict) else {}
    background_npz = resolve_path(signal_inputs.get("background_npz"))
    background_meta_path = resolve_path(signal_inputs.get("background_metadata_json"))
    if background_npz is None or not background_npz.exists():
        raise FileNotFoundError(f"Missing background NPZ from signal metadata: {background_npz}")
    if background_meta_path is None or not background_meta_path.exists():
        raise FileNotFoundError(f"Missing background metadata from signal metadata: {background_meta_path}")

    background_meta = load_json(background_meta_path)
    bkg_inputs = background_meta.get("inputs", {}) if isinstance(background_meta.get("inputs"), dict) else {}
    psf_npz = psf_npz_override or resolve_path(bkg_inputs.get("psf_npz")) or DEFAULT_PSF_NPZ
    if psf_npz is None or not psf_npz.exists():
        raise FileNotFoundError(f"Missing Stage B PSF NPZ: {psf_npz}")

    return {
        "stage_f_meta": stage_f_meta,
        "stage_f_meta_path": stage_f_meta_path,
        "signal_npz": signal_npz,
        "signal_meta": signal_meta,
        "signal_meta_path": signal_meta_path,
        "background_npz": background_npz,
        "background_meta": background_meta,
        "background_meta_path": background_meta_path,
        "psf_npz": psf_npz,
        "selector_csv": selector_csv,
        "cell_ids": selected_cell_ids(stage_f_meta),
    }


def radial_profile(values: np.ndarray, rho: np.ndarray, mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(values) & np.isfinite(rho)
    sums, _ = np.histogram(rho[valid], bins=edges, weights=values[valid])
    return sums.astype(np.float64)


def cell_label(nhit: str, prede: str) -> str:
    return f"{nhit}\n{prede}"


def build_cell_profiles(
    *,
    inputs: dict[str, Any],
    profile_max_deg: float,
    profile_bin_width_deg: float,
    fiducial_rho_deg: float,
    single_fit_max_deg: float,
) -> dict[str, Any]:
    edges = np.arange(0.0, float(profile_max_deg) + 0.5 * float(profile_bin_width_deg), float(profile_bin_width_deg))
    centers = 0.5 * (edges[:-1] + edges[1:])
    cell_ids = np.asarray(inputs["cell_ids"], dtype=np.int64)

    with np.load(inputs["background_npz"], allow_pickle=False) as data:
        bkg_cell_id = data["cell_id"].astype(np.int64)
        bkg_id_to_idx = {int(cell_id): idx for idx, cell_id in enumerate(bkg_cell_id)}
        rho = data["rho_grid_deg"].astype(np.float64)
        fiducial_mask = data["fiducial_mask"].astype(bool) & (rho < float(fiducial_rho_deg))
        counts_map = data["counts_map"].astype(np.float64)
        background_map = data["background_map"].astype(np.float64)
        excess_map = data["excess_map"].astype(np.float64)
        annulus_inner = data["annulus_inner_deg"].astype(np.float64)

    with np.load(inputs["signal_npz"], allow_pickle=False) as data:
        signal_cell_id = data["cell_id"].astype(np.int64)
        signal_id_to_idx = {int(cell_id): idx for idx, cell_id in enumerate(signal_cell_id)}
        n_on_arr = data["N_on"].astype(np.float64)
        b_on_arr = data["B_on"].astype(np.float64)
        excess_arr = data["excess"].astype(np.float64)

    with np.load(inputs["psf_npz"], allow_pickle=False) as data:
        psf_cell_id = data["cell_id"].astype(np.int64)
        psf_id_to_idx = {int(cell_id): idx for idx, cell_id in enumerate(psf_cell_id)}
        psf_edges = data["profile_edges_deg"].astype(np.float64)
        psf_centers = 0.5 * (psf_edges[:-1] + psf_edges[1:])
        psf_density = data["profile_density"].astype(np.float64)
        sigma_mc = data["sigma_deg"].astype(np.float64)
        r68_mc = data["r68_deg"].astype(np.float64)
        r90_mc = data["r90_deg"].astype(np.float64)

    n_cells = cell_ids.size
    counts_profiles = np.zeros((n_cells, centers.size), dtype=np.float64)
    background_profiles = np.zeros_like(counts_profiles)
    excess_profiles = np.zeros_like(counts_profiles)
    model_profiles = np.full_like(counts_profiles, np.nan)
    residual_profiles = np.full_like(counts_profiles, np.nan)
    mc_profiles_interp = np.full_like(counts_profiles, np.nan)
    rows: list[dict[str, Any]] = []
    profile_check_errors: list[float] = []

    for out_idx, cell_id_raw in enumerate(cell_ids):
        cell_id = int(cell_id_raw)
        bkg_idx = bkg_id_to_idx[cell_id]
        sig_idx = signal_id_to_idx[cell_id]
        psf_idx = psf_id_to_idx[cell_id]
        profile_mask = fiducial_mask & (rho < float(profile_max_deg))

        counts_profile = radial_profile(counts_map[bkg_idx], rho, profile_mask, edges)
        background_profile = radial_profile(background_map[bkg_idx], rho, profile_mask, edges)
        excess_profile = radial_profile(excess_map[bkg_idx], rho, profile_mask, edges)
        variance_profile = np.maximum(counts_profile + background_profile, 1.0)

        counts_profiles[out_idx] = counts_profile
        background_profiles[out_idx] = background_profile
        excess_profiles[out_idx] = excess_profile

        total_mask = profile_mask & np.isfinite(counts_map[bkg_idx]) & np.isfinite(background_map[bkg_idx])
        profile_check_errors.extend(
            [
                abs(float(np.sum(counts_map[bkg_idx][total_mask])) - float(np.sum(counts_profile))),
                abs(float(np.sum(background_map[bkg_idx][total_mask])) - float(np.sum(background_profile))),
                abs(float(np.sum(excess_map[bkg_idx][total_mask])) - float(np.sum(excess_profile))),
            ]
        )

        mc_norm = normalize_curve(psf_density[psf_idx])
        if np.any(np.isfinite(mc_norm)):
            mc_profiles_interp[out_idx] = np.interp(centers, psf_centers, mc_norm, left=np.nan, right=np.nan)

        n_on = float(n_on_arr[sig_idx])
        b_on = float(b_on_arr[sig_idx])
        excess = float(excess_arr[sig_idx])
        significance, reliable, unreliable_reason = reliability(n_on, b_on, excess)
        fit_radius = min(float(single_fit_max_deg), float(annulus_inner[bkg_idx]))
        fit = {"status": "not_fit", "reason": "low_stat_empirical_psf_unreliable"}
        if n_on >= MIN_N_ON and excess > 0.0:
            fit = fit_rayleigh_shells(edges, excess_profile, variance_profile, fit_radius)
            if fit.get("status") == "fit":
                model_profiles[out_idx] = np.asarray(fit["model_profile"], dtype=np.float64)
                residual_profiles[out_idx] = np.asarray(fit["residual_profile"], dtype=np.float64)

        sigma_obs = finite(fit.get("sigma_obs_deg"))
        r68_obs = finite(fit.get("r68_obs_deg"))
        sigma_ratio = sigma_obs / float(sigma_mc[psf_idx]) if sigma_obs and sigma_mc[psf_idx] > 0.0 else None
        r68_ratio = r68_obs / float(r68_mc[psf_idx]) if r68_obs and r68_mc[psf_idx] > 0.0 else None
        obs_norm = normalize_curve(excess_profile)
        residual_rms = None
        valid_residual = np.isfinite(obs_norm) & np.isfinite(mc_profiles_interp[out_idx]) & (centers <= fit_radius)
        if np.any(valid_residual):
            residual_rms = float(np.sqrt(np.nanmean((obs_norm[valid_residual] - mc_profiles_interp[out_idx][valid_residual]) ** 2)))

        row = {
            "cell_id": cell_id,
            "nhit_bin": str(np.asarray(inputs["background_meta"].get("cells", []))).strip() if False else "",
            "predE_bin": "",
            "N_on": n_on,
            "B_on": b_on,
            "excess": excess,
            "significance": significance,
            "fit_reliable": int(bool(reliable)),
            "unreliable_reason": unreliable_reason,
            "fit_status": fit.get("status", ""),
            "fit_reason": fit.get("reason", ""),
            "fit_radius_deg": fit_radius,
            "sigma_obs_deg": sigma_obs if sigma_obs is not None else "",
            "sigma_mc_deg": float(sigma_mc[psf_idx]),
            "sigma_obs_over_mc": sigma_ratio if sigma_ratio is not None else "",
            "r68_obs_deg": r68_obs if r68_obs is not None else "",
            "r68_mc_deg": float(r68_mc[psf_idx]),
            "r68_obs_over_mc": r68_ratio if r68_ratio is not None else "",
            "r90_mc_deg": float(r90_mc[psf_idx]),
            "profile_residual_rms": residual_rms if residual_rms is not None else "",
            "fit_chi2_over_ndof": fit.get("chi2_over_ndof", ""),
            "psf_risk_cell": int(cell_id in PSF_RISK_CELLS),
        }
        with np.load(inputs["background_npz"], allow_pickle=False) as data:
            row["nhit_bin"] = str(data["nhit_bin"][bkg_idx])
            row["predE_bin"] = str(data["predE_bin"][bkg_idx])
        rows.append(row)

    return {
        "edges": edges,
        "centers": centers,
        "cell_ids": cell_ids,
        "counts_profiles": counts_profiles,
        "background_profiles": background_profiles,
        "excess_profiles": excess_profiles,
        "model_profiles": model_profiles,
        "residual_profiles": residual_profiles,
        "mc_profiles_interp": mc_profiles_interp,
        "rows": rows,
        "profile_check_max_abs_error": max(profile_check_errors) if profile_check_errors else 0.0,
    }


def build_group_profiles(profiles: dict[str, Any]) -> dict[str, Any]:
    rows = profiles["rows"]
    edges = profiles["edges"]
    grouped: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        nhit = str(row["nhit_bin"])
        item = grouped.setdefault(
            nhit,
            {
                "nhit_bin": nhit,
                "cell_ids": [],
                "counts_profile": np.zeros_like(profiles["counts_profiles"][idx]),
                "background_profile": np.zeros_like(profiles["background_profiles"][idx]),
                "excess_profile": np.zeros_like(profiles["excess_profiles"][idx]),
                "mc_profiles": [],
                "mc_weights": [],
                "N_on": 0.0,
                "B_on": 0.0,
                "excess": 0.0,
            },
        )
        item["cell_ids"].append(int(row["cell_id"]))
        item["counts_profile"] += profiles["counts_profiles"][idx]
        item["background_profile"] += profiles["background_profiles"][idx]
        item["excess_profile"] += profiles["excess_profiles"][idx]
        item["mc_profiles"].append(profiles["mc_profiles_interp"][idx])
        item["mc_weights"].append(max(float(row["excess"]), 0.0))
        item["N_on"] += float(row["N_on"])
        item["B_on"] += float(row["B_on"])
        item["excess"] += float(row["excess"])

    group_rows: list[dict[str, Any]] = []
    group_profiles: dict[str, dict[str, np.ndarray]] = {}
    for nhit, item in sorted(grouped.items(), key=lambda kv: interval_key(kv[0])):
        variance = np.maximum(item["counts_profile"] + item["background_profile"], 1.0)
        fit = fit_rayleigh_shells(edges, item["excess_profile"], variance, SINGLE_FIT_MAX_DEG)
        significance, reliable, unreliable_reason = reliability(float(item["N_on"]), float(item["B_on"]), float(item["excess"]))
        mc_stack = np.vstack(item["mc_profiles"])
        weights = np.asarray(item["mc_weights"], dtype=np.float64)
        if not np.any(weights > 0.0):
            weights = np.ones_like(weights)
        group_mc = np.average(mc_stack, axis=0, weights=weights)
        group_mc = normalize_curve(group_mc)
        obs_norm = normalize_curve(item["excess_profile"])
        group_profiles[nhit] = {
            "obs_norm": obs_norm,
            "mc_norm": group_mc,
            "counts_profile": item["counts_profile"],
            "background_profile": item["background_profile"],
            "excess_profile": item["excess_profile"],
            "model_profile": np.asarray(fit.get("model_profile", np.full_like(obs_norm, np.nan)), dtype=np.float64),
        }
        group_rows.append(
            {
                "nhit_bin": nhit,
                "cell_ids": ",".join(str(v) for v in item["cell_ids"]),
                "n_cells": len(item["cell_ids"]),
                "N_on": float(item["N_on"]),
                "B_on": float(item["B_on"]),
                "excess": float(item["excess"]),
                "significance": significance,
                "fit_reliable": int(bool(reliable)),
                "unreliable_reason": unreliable_reason,
                "fit_status": fit.get("status", ""),
                "sigma_obs_deg": fit.get("sigma_obs_deg", ""),
                "r68_obs_deg": fit.get("r68_obs_deg", ""),
                "fit_chi2_over_ndof": fit.get("chi2_over_ndof", ""),
            }
        )
    return {"rows": group_rows, "profiles": group_profiles}


def interval_key(label: str) -> float:
    text = str(label)
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


def plot_profile_components(profiles: dict[str, Any], output_path: Path) -> None:
    Image, ImageDraw, _ = load_pil()
    rows = profiles["rows"]
    centers = profiles["centers"]
    ncols = 5
    nrows = int(math.ceil(len(rows) / ncols))
    panel_w, panel_h = 255, 205
    margin_x, margin_y = 24, 68
    width = margin_x * 2 + ncols * panel_w
    height = margin_y + nrows * panel_h + 36
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (margin_x, 18), "Observed radial profile components for v4 fit cells", size=16, bold=True)
    draw_text(draw, (margin_x, 42), "blue=counts, gray=fitted background, orange=counts-background", size=11, fill="#4b5563")

    for idx, row in enumerate(rows):
        r = idx // ncols
        c = idx % ncols
        px = margin_x + c * panel_w
        py = margin_y + r * panel_h
        rect = (px + 38, py + 39, px + panel_w - 14, py + panel_h - 31)
        y_min, y_max = finite_range(
            [
                profiles["counts_profiles"][idx],
                profiles["background_profiles"][idx],
                profiles["excess_profiles"][idx],
            ]
        )
        draw_axes(draw, rect, x_label="rho" if r == nrows - 1 else None, y_label="counts" if c == 0 else None)
        draw_hline(draw, 0.0, rect=rect, y_min=y_min, y_max=y_max, fill="#9ca3af")
        draw_line(
            draw,
            centers,
            profiles["counts_profiles"][idx],
            rect=rect,
            x_min=0.0,
            x_max=PROFILE_MAX_DEG,
            y_min=y_min,
            y_max=y_max,
            fill="#1f77b4",
            width=2,
        )
        draw_line(
            draw,
            centers,
            profiles["background_profiles"][idx],
            rect=rect,
            x_min=0.0,
            x_max=PROFILE_MAX_DEG,
            y_min=y_min,
            y_max=y_max,
            fill="#6b7280",
            width=2,
        )
        draw_line(
            draw,
            centers,
            profiles["excess_profiles"][idx],
            rect=rect,
            x_min=0.0,
            x_max=PROFILE_MAX_DEG,
            y_min=y_min,
            y_max=y_max,
            fill="#d97706",
            width=2,
        )
        color = "#047857" if row["fit_reliable"] else "#991b1b"
        draw_text(draw, (px + 38, py + 7), f"{row['cell_id']} {row['nhit_bin']}", fill=color, size=10, bold=True)
        draw_text(draw, (px + 38, py + 21), str(row["predE_bin"]), fill=color, size=10)
    save_canvas(output_path, image)


def plot_observed_vs_mc(profiles: dict[str, Any], output_path: Path) -> None:
    Image, ImageDraw, _ = load_pil()
    rows = profiles["rows"]
    centers = profiles["centers"]
    ncols = 5
    nrows = int(math.ceil(len(rows) / ncols))
    panel_w, panel_h = 255, 210
    margin_x, margin_y = 24, 72
    width = margin_x * 2 + ncols * panel_w
    height = margin_y + nrows * panel_h + 36
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (margin_x, 18), "Observed empirical PSF versus MC PSF for v4 fit cells", size=16, bold=True)
    draw_text(draw, (margin_x, 42), "orange=observed excess, blue=MC PSF, black=observed Rayleigh fit; gray line=fit radius", size=11, fill="#4b5563")

    for idx, row in enumerate(rows):
        r = idx // ncols
        c = idx % ncols
        px = margin_x + c * panel_w
        py = margin_y + r * panel_h
        rect = (px + 38, py + 45, px + panel_w - 14, py + panel_h - 31)
        obs = normalize_curve(profiles["excess_profiles"][idx])
        mc = profiles["mc_profiles_interp"][idx]
        model = normalize_curve(profiles["model_profiles"][idx])
        y_min, y_max = -0.25, 1.12
        draw_axes(draw, rect, x_label="rho" if r == nrows - 1 else None, y_label="norm" if c == 0 else None)
        draw_hline(draw, 0.0, rect=rect, y_min=y_min, y_max=y_max, fill="#d1d5db")
        draw_vline(draw, float(row["fit_radius_deg"]), rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, fill="#9ca3af")
        draw_line(draw, centers, obs, rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, y_min=y_min, y_max=y_max, fill="#d97706", width=2)
        draw_line(draw, centers, mc, rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, y_min=y_min, y_max=y_max, fill="#2563eb", width=2)
        if np.any(np.isfinite(model)):
            draw_line(draw, centers, model, rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, y_min=y_min, y_max=y_max, fill="#111827", width=2)
        title_color = "#047857" if row["fit_reliable"] else "#991b1b"
        risk = " risk" if row["psf_risk_cell"] else ""
        draw_text(draw, (px + 38, py + 6), f"{row['cell_id']}{risk} {row['nhit_bin']}", fill=title_color, size=10, bold=True)
        draw_text(
            draw,
            (px + 38, py + 21),
            f"s/MC={fmt(row['sigma_obs_over_mc'], 3)} r68/MC={fmt(row['r68_obs_over_mc'], 3)}",
            fill=title_color,
            size=10,
        )
    save_canvas(output_path, image)


def plot_ratio_grid(rows: list[dict[str, Any]], key: str, output_path: Path, title: str) -> None:
    Image, ImageDraw, _ = load_pil()
    grid = np.full((7, 12), np.nan, dtype=np.float64)
    reliable_grid = np.zeros((7, 12), dtype=bool)
    for row in rows:
        cell_id = int(row["cell_id"])
        idx = cell_id - 1
        r = idx // 12
        c = idx % 12
        value = finite(row.get(key))
        if value is not None:
            grid[r, c] = value
        reliable_grid[r, c] = bool(row.get("fit_reliable"))

    cell_w, cell_h = 82, 62
    left, top = 70, 70
    width = left + 12 * cell_w + 155
    height = top + 7 * cell_h + 92
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (32, 18), title, size=17, bold=True)
    draw_text(draw, (32, 42), "blue < 1, white ~ 1, red > 1; * = low-stat / unreliable empirical fit", size=11, fill="#4b5563")

    def color_for(value: float) -> tuple[int, int, int, int]:
        v = max(0.5, min(1.5, float(value)))
        if v >= 1.0:
            t = (v - 1.0) / 0.5
            start = np.asarray([255, 255, 255])
            end = np.asarray([185, 28, 28])
        else:
            t = (1.0 - v) / 0.5
            start = np.asarray([255, 255, 255])
            end = np.asarray([37, 99, 235])
        rgb = (start * (1.0 - t) + end * t).astype(int)
        return int(rgb[0]), int(rgb[1]), int(rgb[2]), 255

    for r in range(7):
        for c in range(12):
            cell_id = r * 12 + c + 1
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            value = grid[r, c]
            if np.isfinite(value):
                draw.rectangle((x0, y0, x1, y1), fill=color_for(value), outline="white", width=2)
                color = "white" if value < 0.78 or value > 1.25 else "#111827"
                marker = "" if reliable_grid[r, c] else "*"
                draw_text(draw, (x0 + 9, y0 + 13), f"{cell_id}", fill=color, size=10, bold=True)
                draw_text(draw, (x0 + 9, y0 + 31), f"{value:.2f}{marker}", fill=color, size=11)
            else:
                draw.rectangle((x0, y0, x1, y1), fill="#f3f4f6", outline="white", width=2)
                draw_text(draw, (x0 + 9, y0 + 22), str(cell_id), fill="#9ca3af", size=9)

    for c in range(12):
        draw_text(draw, (left + c * cell_w + 28, top - 22), str(c + 1), fill="#4b5563", size=10)
    for r in range(7):
        draw_text(draw, (left - 32, top + r * cell_h + 22), str(r + 1), fill="#4b5563", size=10)
    draw_text(draw, (left + 12 * cell_w + 20, top), "observed / MC", size=11, bold=True)
    for i, value in enumerate(np.linspace(1.5, 0.5, 101)):
        y0 = top + 28 + i * 3
        draw.rectangle((left + 12 * cell_w + 28, y0, left + 12 * cell_w + 56, y0 + 3), fill=color_for(float(value)))
    draw_text(draw, (left + 12 * cell_w + 62, top + 24), "1.5", size=9, fill="#4b5563")
    draw_text(draw, (left + 12 * cell_w + 62, top + 174), "1.0", size=9, fill="#4b5563")
    draw_text(draw, (left + 12 * cell_w + 62, top + 324), "0.5", size=9, fill="#4b5563")
    draw_text(draw, (left + 290, height - 42), "predE cell column", fill="#4b5563", size=11)
    draw_text(draw, (22, top + 190), "Nhit row", fill="#4b5563", size=11)
    save_canvas(output_path, image)


def plot_group_overlays(profiles: dict[str, Any], groups: dict[str, Any], output_path: Path) -> None:
    Image, ImageDraw, _ = load_pil()
    centers = profiles["centers"]
    group_profiles = groups["profiles"]
    n = len(group_profiles)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    panel_w, panel_h = 500, 265
    left, top = 34, 74
    width = left * 2 + ncols * panel_w
    height = top + nrows * panel_h + 42
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (left, 18), "Nhit-group empirical PSF fallback overlays", size=17, bold=True)
    draw_text(draw, (left, 42), "orange=observed summed excess, blue=weighted MC PSF, black=observed Rayleigh fit", size=11, fill="#4b5563")
    rows_by_nhit = {row["nhit_bin"]: row for row in groups["rows"]}
    for idx, (nhit, payload) in enumerate(sorted(group_profiles.items(), key=lambda kv: interval_key(kv[0]))):
        r = idx // ncols
        c = idx % ncols
        px = left + c * panel_w
        py = top + r * panel_h
        rect = (px + 52, py + 54, px + panel_w - 22, py + panel_h - 42)
        row = rows_by_nhit[nhit]
        draw_axes(draw, rect, x_label="rho [deg]" if r == nrows - 1 else None, y_label="norm" if c == 0 else None)
        draw_hline(draw, 0.0, rect=rect, y_min=-0.25, y_max=1.12, fill="#d1d5db")
        draw_line(draw, centers, payload["obs_norm"], rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, y_min=-0.25, y_max=1.12, fill="#d97706", width=3)
        draw_line(draw, centers, payload["mc_norm"], rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, y_min=-0.25, y_max=1.12, fill="#2563eb", width=3)
        model = normalize_curve(payload["model_profile"])
        if np.any(np.isfinite(model)):
            draw_line(draw, centers, model, rect=rect, x_min=0.0, x_max=PROFILE_MAX_DEG, y_min=-0.25, y_max=1.12, fill="#111827", width=2)
        title_color = "#047857" if row["fit_reliable"] else "#991b1b"
        draw_text(
            draw,
            (px + 52, py + 9),
            f"{nhit}: {row['n_cells']} cells, sigma_obs={fmt(row['sigma_obs_deg'], 3)}, sig={fmt(row['significance'], 3)}",
            fill=title_color,
            size=11,
            bold=True,
        )
        draw_text(draw, (px + 52, py + 27), f"cells: {row['cell_ids']}", fill="#4b5563", size=9)
    save_canvas(output_path, image)


def save_npz(path: Path, profiles: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        profile_edges_deg=profiles["edges"].astype(np.float32),
        profile_centers_deg=profiles["centers"].astype(np.float32),
        cell_id=profiles["cell_ids"].astype(np.int32),
        counts_profile=profiles["counts_profiles"].astype(np.float32),
        background_profile=profiles["background_profiles"].astype(np.float32),
        excess_profile=profiles["excess_profiles"].astype(np.float32),
        empirical_model_profile=profiles["model_profiles"].astype(np.float32),
        empirical_residual_profile=profiles["residual_profiles"].astype(np.float32),
        mc_profile_normalized_interp=profiles["mc_profiles_interp"].astype(np.float32),
    )


def build_diagnostics(
    *,
    stage_f_metadata: Path = DEFAULT_STAGE_F_META,
    psf_npz: Path | None = None,
    output_dir: Path = OUT_DIR,
    profile_max_deg: float = PROFILE_MAX_DEG,
    profile_bin_width_deg: float = PROFILE_BIN_WIDTH_DEG,
    fiducial_rho_deg: float = FIDUCIAL_RHO_DEG,
    single_fit_max_deg: float = SINGLE_FIT_MAX_DEG,
) -> dict[str, Path]:
    stage_f_metadata = resolve_path(stage_f_metadata) or stage_f_metadata
    if not stage_f_metadata.exists():
        raise FileNotFoundError(stage_f_metadata)
    resolved_psf = resolve_path(psf_npz) if psf_npz else None
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(stage_f_metadata, resolved_psf)
    profiles = build_cell_profiles(
        inputs=inputs,
        profile_max_deg=profile_max_deg,
        profile_bin_width_deg=profile_bin_width_deg,
        fiducial_rho_deg=fiducial_rho_deg,
        single_fit_max_deg=single_fit_max_deg,
    )
    groups = build_group_profiles(profiles)

    summary_csv = output_dir / "empirical_psf_cell_summary.csv"
    group_csv = output_dir / "empirical_psf_nhit_group_summary.csv"
    profile_npz = output_dir / "empirical_psf_profiles.npz"
    summary_json = output_dir / "empirical_psf_summary.json"
    components_png = output_dir / "observed_radial_profile_components_grid.png"
    overlay_png = output_dir / "observed_vs_mc_radial_profiles_grid.png"
    sigma_grid_png = output_dir / "sigma_obs_over_mc_grid.png"
    r68_grid_png = output_dir / "r68_obs_over_mc_grid.png"
    group_png = output_dir / "nhit_group_empirical_psf_overlays.png"

    cell_fields = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "N_on",
        "B_on",
        "excess",
        "significance",
        "fit_reliable",
        "unreliable_reason",
        "fit_status",
        "fit_reason",
        "fit_radius_deg",
        "sigma_obs_deg",
        "sigma_mc_deg",
        "sigma_obs_over_mc",
        "r68_obs_deg",
        "r68_mc_deg",
        "r68_obs_over_mc",
        "r90_mc_deg",
        "profile_residual_rms",
        "fit_chi2_over_ndof",
        "psf_risk_cell",
    ]
    group_fields = [
        "nhit_bin",
        "cell_ids",
        "n_cells",
        "N_on",
        "B_on",
        "excess",
        "significance",
        "fit_reliable",
        "unreliable_reason",
        "fit_status",
        "sigma_obs_deg",
        "r68_obs_deg",
        "fit_chi2_over_ndof",
    ]
    write_csv(summary_csv, profiles["rows"], cell_fields)
    write_csv(group_csv, groups["rows"], group_fields)
    save_npz(profile_npz, profiles)
    plot_profile_components(profiles, components_png)
    plot_observed_vs_mc(profiles, overlay_png)
    plot_ratio_grid(profiles["rows"], "sigma_obs_over_mc", sigma_grid_png, "Empirical sigma / MC sigma for v4 fit cells")
    plot_ratio_grid(profiles["rows"], "r68_obs_over_mc", r68_grid_png, "Empirical r68 / MC r68 for v4 fit cells")
    plot_group_overlays(profiles, groups, group_png)

    reliable_rows = [row for row in profiles["rows"] if int(row["fit_reliable"]) == 1]
    ratios = [finite(row.get("sigma_obs_over_mc")) for row in reliable_rows]
    ratios = [v for v in ratios if v is not None]
    r68_ratios = [finite(row.get("r68_obs_over_mc")) for row in reliable_rows]
    r68_ratios = [v for v in r68_ratios if v is not None]
    payload = {
        "description": "Observed-data empirical/effective PSF diagnostics for current v4 drop4 fit cells.",
        "inputs": {
            "stage_f_metadata": str(inputs["stage_f_meta_path"]),
            "signal_npz": str(inputs["signal_npz"]),
            "signal_metadata": str(inputs["signal_meta_path"]),
            "background_npz": str(inputs["background_npz"]),
            "background_metadata": str(inputs["background_meta_path"]),
            "psf_npz": str(inputs["psf_npz"]),
            "selector_csv": str(inputs["selector_csv"]),
        },
        "method": {
            "background": "fixed latest annnorm Stage D background_map; no background refit in empirical PSF fit",
            "profile": "radial sums around Crab tangent-plane center (x,y)=(0,0)",
            "radial_range_deg": [0.0, float(profile_max_deg)],
            "radial_bin_width_deg": float(profile_bin_width_deg),
            "fiducial_rho_deg": float(fiducial_rho_deg),
            "single_cell_fit_radius": "min(single_fit_max_deg, annulus_inner_deg)",
            "single_fit_max_deg": float(single_fit_max_deg),
            "model": "2D circular Gaussian/Rayleigh shell probabilities with fitted amplitude and sigma_obs",
            "grouped_fallback": "sum observed profiles by Nhit; MC comparison is positive-excess-weighted average of normalized MC profiles",
        },
        "statistics_gate": {
            "min_N_on": MIN_N_ON,
            "min_excess": MIN_EXCESS,
            "min_significance": MIN_SIGNIFICANCE,
            "unreliable_behavior": "single-cell profiles are plotted; ratios marked with * in grids and excluded from headline medians",
        },
        "summary": {
            "cells": len(profiles["rows"]),
            "reliable_cells": len(reliable_rows),
            "unreliable_cells": len(profiles["rows"]) - len(reliable_rows),
            "median_sigma_obs_over_mc_reliable": float(np.nanmedian(ratios)) if ratios else None,
            "median_r68_obs_over_mc_reliable": float(np.nanmedian(r68_ratios)) if r68_ratios else None,
            "profile_check_max_abs_error": profiles["profile_check_max_abs_error"],
        },
        "outputs": {
            "cell_summary_csv": str(summary_csv),
            "group_summary_csv": str(group_csv),
            "profiles_npz": str(profile_npz),
            "profile_components_png": str(components_png),
            "observed_vs_mc_png": str(overlay_png),
            "sigma_ratio_grid_png": str(sigma_grid_png),
            "r68_ratio_grid_png": str(r68_grid_png),
            "nhit_group_png": str(group_png),
        },
    }
    write_json(summary_json, payload)
    return {
        "summary_csv": summary_csv,
        "group_csv": group_csv,
        "profile_npz": profile_npz,
        "summary_json": summary_json,
        "components_png": components_png,
        "overlay_png": overlay_png,
        "sigma_grid_png": sigma_grid_png,
        "r68_grid_png": r68_grid_png,
        "group_png": group_png,
    }


def main() -> None:
    args = parse_args()
    outputs = build_diagnostics(
        stage_f_metadata=Path(args.stage_f_metadata),
        psf_npz=Path(args.psf_npz) if args.psf_npz else None,
        output_dir=Path(args.output_dir),
        profile_max_deg=float(args.profile_max_deg),
        profile_bin_width_deg=float(args.profile_bin_width_deg),
        fiducial_rho_deg=float(args.fiducial_rho_deg),
        single_fit_max_deg=float(args.single_fit_max_deg),
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
