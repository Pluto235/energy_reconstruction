#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 annulus-normalized background diagnostic assets.")
    parser.add_argument("--nominal-stage-d-npz", type=str, default="apply/output/stage_d_v3_candidate/runs/v3_stage_d_slurm_42024/background_v3_candidate.npz")
    parser.add_argument("--nominal-stage-d-summary", type=str, default="apply/output/stage_d_v3_candidate/runs/v3_stage_d_slurm_42024/background_v3_candidate_summary.csv")
    parser.add_argument("--nominal-stage-e-metadata", type=str, default="apply/output/stage_e_v3_candidate/runs/v3_stage_e_slurm_42024/signal_v3_candidate_metadata.json")
    parser.add_argument("--nominal-stage-f-metadata", type=str, default="apply/output/stage_f_v3_baseline_psfborrow/current/fit_v3_baseline_psfborrow_metadata.json")
    parser.add_argument("--nominal-stage-g-metadata", type=str, default="apply/output/stage_g_v3_baseline_psfborrow/current/sed_points_v3_baseline_psfborrow_metadata.json")
    parser.add_argument("--annnorm-stage-d-npz", type=str, default="apply/output/stage_d_v3_candidate_annnorm/current/background_v3_candidate_annnorm.npz")
    parser.add_argument("--annnorm-stage-d-summary", type=str, default="apply/output/stage_d_v3_candidate_annnorm/current/background_v3_candidate_annnorm_summary.csv")
    parser.add_argument("--annnorm-stage-e-metadata", type=str, default="apply/output/stage_e_v3_candidate_annnorm/current/signal_v3_candidate_annnorm_metadata.json")
    parser.add_argument("--annnorm-stage-f-metadata", type=str, default="apply/output/stage_f_v3_baseline_annnorm/current/fit_v3_baseline_annnorm_metadata.json")
    parser.add_argument("--annnorm-stage-g-metadata", type=str, default="apply/output/stage_g_v3_baseline_annnorm/current/sed_points_v3_baseline_annnorm_metadata.json")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline_psfborrow.csv")
    parser.add_argument("--output-dir", type=str, default="apply/report/assets/v3-annnorm")
    parser.add_argument("--summary-json", type=str, default="v3_annnorm_summary.json")
    parser.add_argument("--summary-csv", type=str, default="v3_annnorm_summary.csv")
    parser.add_argument("--scale-grid-png", type=str, default="v3_annnorm_surface_scale_grid.png")
    parser.add_argument("--dec-profile-png", type=str, default="v3_annnorm_dec_profile_comparison.png")
    parser.add_argument("--sed-overlay-png", type=str, default="v3_annnorm_stage_g_sed_overlay.png")
    return parser.parse_args()


def path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_json(path_: Path) -> Dict[str, object]:
    if not path_.exists():
        return {}
    with path_.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path_: Path) -> List[Dict[str, str]]:
    if not path_.exists():
        return []
    with path_.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: object, digits: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def selector_ids(path_: Path) -> List[int]:
    ids: List[int] = []
    for row in read_csv_rows(path_):
        include = str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}
        if include:
            ids.append(int(row["cell_id"]))
    return ids


def row_map(rows: Sequence[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        try:
            out[int(row["cell_id"])] = row
        except (KeyError, ValueError):
            continue
    return out


def sorted_unique(values: np.ndarray) -> List[str]:
    return sorted({str(v) for v in values.tolist()}, key=interval_key)


def interval_key(label: str) -> float:
    text = str(label).strip().strip('"')
    if text.startswith("[") and "," in text:
        return float(text[1:].split(",", 1)[0])
    if text.startswith(">="):
        return float(text[2:])
    if text.startswith("<"):
        return -1.0e9
    return 1.0e9


def heatmap_from_cells(values: np.ndarray, cell_id: np.ndarray, nhit: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
    y_labels = sorted_unique(nhit)
    x_labels = sorted_unique(pred)
    matrix = np.full((len(y_labels), len(x_labels)), np.nan, dtype=np.float64)
    y_index = {label: idx for idx, label in enumerate(y_labels)}
    x_index = {label: idx for idx, label in enumerate(x_labels)}
    for idx in range(cell_id.size):
        y = y_index[str(nhit[idx])]
        x = x_index[str(pred[idx])]
        matrix[y, x] = values[idx]
    return matrix, y_labels, x_labels


def plot_scale_grid(npz_path: Path, output: Path, fit_ids: Sequence[int]) -> Optional[Path]:
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"annulus_surface_scale", "cell_id", "nhit_bin", "predE_bin"}
        if not required.issubset(set(data.files)):
            return None
        values = np.asarray(data["annulus_surface_scale"], dtype=np.float64)
        cell_id = np.asarray(data["cell_id"], dtype=np.int64)
        nhit = np.asarray(data["nhit_bin"]).astype(str)
        pred = np.asarray(data["predE_bin"]).astype(str)
    matrix, y_labels, x_labels = heatmap_from_cells(values, cell_id, nhit, pred)
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
    finite = matrix[np.isfinite(matrix)]
    span = max(abs(float(np.nanmin(finite)) - 1.0), abs(float(np.nanmax(finite)) - 1.0), 0.02) if finite.size else 0.1
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=1.0 - span, vmax=1.0 + span)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("predE bin")
    ax.set_ylabel("Nhit bin")
    ax.set_title("Annulus-normalized surface scale per candidate cell")
    fit_set = set(int(v) for v in fit_ids)
    for idx, cid in enumerate(cell_id):
        y = y_labels.index(str(nhit[idx]))
        x = x_labels.index(str(pred[idx]))
        text = f"{int(cid)}\n{values[idx]:.3f}" if np.isfinite(values[idx]) else f"{int(cid)}"
        ax.text(x, y, text, ha="center", va="center", fontsize=5.7, color="#111827")
        if int(cid) in fit_set:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="#111827", lw=1.2))
    fig.colorbar(im, ax=ax, label="surface scale")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def dec_profile(npz_path: Path, fit_ids: Sequence[int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"counts_map", "background_map", "cell_id"}
        if not required.issubset(set(data.files)):
            return None
        if "xy_edges_deg" in data.files:
            edges = np.asarray(data["xy_edges_deg"], dtype=np.float64)
        elif "x_edges_deg" in data.files:
            edges = np.asarray(data["x_edges_deg"], dtype=np.float64)
        else:
            return None
        counts = np.asarray(data["counts_map"], dtype=np.float64)
        background = np.asarray(data["background_map"], dtype=np.float64)
        cell_id = np.asarray(data["cell_id"], dtype=np.int64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_mask = np.abs(centers) < 1.0
    selected = np.isin(cell_id, np.asarray(list(fit_ids), dtype=np.int64))
    if not np.any(selected) or not np.any(x_mask):
        return None
    counts_profile = np.nansum(counts[selected][:, :, x_mask], axis=(0, 2))
    background_profile = np.nansum(background[selected][:, :, x_mask], axis=(0, 2))
    return centers, counts_profile, counts_profile - background_profile


def plot_dec_profiles(nominal_npz: Path, annnorm_npz: Path, output: Path, fit_ids: Sequence[int]) -> Optional[Path]:
    nominal = dec_profile(nominal_npz, fit_ids)
    annnorm = dec_profile(annnorm_npz, fit_ids)
    if nominal is None or annnorm is None:
        return None
    y, counts_nom, excess_nom = nominal
    _, counts_new, excess_new = annnorm
    roi_mask = np.abs(y) <= 6.0
    y = y[roi_mask]
    counts_nom = counts_nom[roi_mask]
    excess_nom = excess_nom[roi_mask]
    counts_new = counts_new[roi_mask]
    excess_new = excess_new[roi_mask]
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    ax.plot(y, counts_nom, color="#6b7280", lw=1.0, label="counts, nominal")
    ax.plot(y, counts_new, color="#111827", lw=1.0, ls=":", label="counts, annnorm")
    ax.plot(y, excess_nom, color="#d97706", lw=1.5, label="counts - background, nominal")
    ax.plot(y, excess_new, color="#2563eb", lw=1.5, label="counts - background, annulus-normalized")
    ax.axhline(0.0, color="#111111", lw=0.8, ls="--")
    ax.axvspan(-1.0, 1.0, color="#e5e7eb", alpha=0.45, label="central |Dec offset|<1 deg")
    ax.set_xlim(-6.0, 6.0)
    ax.set_xlabel("Dec offset from Crab (deg)")
    ax.set_ylabel("summed counts in |RA offset|<1 deg")
    ax.set_title("Before/after Dec profile comparison for active v3 fit cells")
    ax.grid(alpha=0.25, lw=0.45)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def stage_g_arrays(meta: Dict[str, object], grouping: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = meta.get("points") if isinstance(meta.get("points"), list) else []
    selected = [p for p in points if isinstance(p, dict) and str(p.get("grouping")) == grouping]
    selected.sort(key=lambda p: finite_float(p.get("effective_energy_tev")) or 0.0)
    x: List[float] = []
    y: List[float] = []
    yerr: List[float] = []
    for point in selected:
        energy = finite_float(point.get("effective_energy_tev"))
        flux = finite_float(point.get("E2_dnde"))
        if energy is None or flux is None or energy <= 0.0 or flux <= 0.0:
            continue
        x.append(energy)
        y.append(flux)
        yerr.append(finite_float(point.get("E2_dnde_err")) or 0.0)
    return np.asarray(x), np.asarray(y), np.asarray(yerr)


def plot_sed_overlay(nominal_meta: Dict[str, object], annnorm_meta: Dict[str, object], output: Path) -> Optional[Path]:
    if not nominal_meta or not annnorm_meta:
        return None
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    plotted = False
    for grouping, marker in [("nhit", "o"), ("predE", "s")]:
        x, y, yerr = stage_g_arrays(nominal_meta, grouping)
        if x.size:
            plotted = True
            ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=4.5, capsize=2, alpha=0.45, label=f"active reference {grouping}")
        x, y, yerr = stage_g_arrays(annnorm_meta, grouping)
        if x.size:
            plotted = True
            ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=4.8, capsize=2, alpha=0.85, label=f"annulus-normalized {grouping}")
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective energy (TeV)")
    ax.set_ylabel("E^2 dN/dE (TeV cm^-2 s^-1)")
    ax.set_title("Stage G SED points: active reference vs annulus-normalized background")
    ax.grid(alpha=0.25, which="both", lw=0.45)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def totals_row(label: str, e_meta: Dict[str, object], f_meta: Dict[str, object], g_meta: Dict[str, object]) -> Dict[str, object]:
    totals = e_meta.get("totals") if isinstance(e_meta.get("totals"), dict) else {}
    preferred = f_meta.get("preferred_fit") if isinstance(f_meta.get("preferred_fit"), dict) else {}
    fits = f_meta.get("fits") if isinstance(f_meta.get("fits"), dict) else {}
    key = f"{str(preferred.get('model', 'pl')).lower()}_{str(preferred.get('error_mode', 'conservative')).lower()}"
    fit = fits.get(key) if isinstance(fits, dict) and isinstance(fits.get(key), dict) else {}
    params = fit.get("parameters") if isinstance(fit.get("parameters"), dict) else {}
    points = g_meta.get("points") if isinstance(g_meta.get("points"), list) else []
    return {
        "variant": label,
        "stage_e_N_on": totals.get("N_on"),
        "stage_e_B_on": totals.get("B_on"),
        "stage_e_excess": totals.get("excess"),
        "stage_e_sigma": totals.get("formal_sigma"),
        "stage_f_model": preferred.get("model"),
        "stage_f_error": preferred.get("error_mode"),
        "stage_f_phi0": params.get("phi0"),
        "stage_f_gamma_or_alpha": params.get("gamma", params.get("alpha")),
        "stage_f_beta": params.get("beta"),
        "stage_f_chi2": fit.get("chi2"),
        "stage_f_ndof": fit.get("ndof"),
        "stage_g_points": len(points),
        "stage_g_run": g_meta.get("run_id"),
    }


def annulus_rows(summary_rows: Sequence[Dict[str, str]], fit_ids: Sequence[int]) -> List[Dict[str, object]]:
    fit_set = set(int(v) for v in fit_ids)
    out: List[Dict[str, object]] = []
    for row in summary_rows:
        try:
            cid = int(row["cell_id"])
        except (KeyError, ValueError):
            continue
        if cid not in fit_set:
            continue
        out.append(
            {
                "cell_id": cid,
                "nhit_bin": row.get("nhit_bin", ""),
                "predE_bin": row.get("predE_bin", ""),
                "B_on": finite_float(row.get("B_on")),
                "annulus_surface_scale": finite_float(row.get("annulus_surface_scale")),
                "annulus_count_residual_raw": finite_float(row.get("annulus_count_residual_raw")),
                "annulus_count_residual_final": finite_float(row.get("annulus_count_residual_final")),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_ids = selector_ids(path(args.baseline_selector_csv))
    nominal_stage_e = load_json(path(args.nominal_stage_e_metadata))
    nominal_stage_f = load_json(path(args.nominal_stage_f_metadata))
    nominal_stage_g = load_json(path(args.nominal_stage_g_metadata))
    ann_stage_e = load_json(path(args.annnorm_stage_e_metadata))
    ann_stage_f = load_json(path(args.annnorm_stage_f_metadata))
    ann_stage_g = load_json(path(args.annnorm_stage_g_metadata))
    scale_png = plot_scale_grid(path(args.annnorm_stage_d_npz), output_dir / args.scale_grid_png, fit_ids)
    dec_png = plot_dec_profiles(path(args.nominal_stage_d_npz), path(args.annnorm_stage_d_npz), output_dir / args.dec_profile_png, fit_ids)
    sed_png = plot_sed_overlay(nominal_stage_g, ann_stage_g, output_dir / args.sed_overlay_png)
    comparison_rows = [
        totals_row("active_reference", nominal_stage_e, nominal_stage_f, nominal_stage_g),
        totals_row("annulus_normalized_surface", ann_stage_e, ann_stage_f, ann_stage_g),
    ]
    ann_rows = annulus_rows(read_csv_rows(path(args.annnorm_stage_d_summary)), fit_ids)
    summary = {
        "description": "v3 annulus-normalized 2D background diagnostic assets",
        "fit_cell_ids": fit_ids,
        "comparison": comparison_rows,
        "annulus_fit_cell_rows": ann_rows,
        "assets": {
            "scale_grid_png": str(scale_png) if scale_png else None,
            "dec_profile_png": str(dec_png) if dec_png else None,
            "sed_overlay_png": str(sed_png) if sed_png else None,
        },
    }
    summary_path = output_dir / args.summary_json
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output_dir / args.summary_csv
    fieldnames = [
        "variant",
        "stage_e_N_on",
        "stage_e_B_on",
        "stage_e_excess",
        "stage_e_sigma",
        "stage_f_model",
        "stage_f_error",
        "stage_f_phi0",
        "stage_f_gamma_or_alpha",
        "stage_f_beta",
        "stage_f_chi2",
        "stage_f_ndof",
        "stage_g_points",
        "stage_g_run",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    for maybe in [scale_png, dec_png, sed_png]:
        if maybe:
            print(f"Wrote {maybe}")


if __name__ == "__main__":
    main()
