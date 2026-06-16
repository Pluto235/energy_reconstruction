#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
M2_TO_CM2 = 1.0e4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3 official-spectrum forward-fold and fake-source residual diagnostics."
    )
    parser.add_argument("--output-dir", type=str, default="apply/report/assets/v3-normalization-diagnostics")
    parser.add_argument("--response-npz", type=str, default="apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz")
    parser.add_argument(
        "--active-stage-f-npz",
        type=str,
        default=(
            "apply/output/stage_f_v3_baseline_psfborrow/runs/"
            "v3_stage_f_psfborrow_slurm_42029/fit_v3_baseline_psfborrow.npz"
        ),
    )
    parser.add_argument(
        "--legacy-stage-f-npz",
        type=str,
        default="apply/output/stage_f_v3_baseline/runs/v3_stage_f_slurm_42024/fit_v3_baseline.npz",
    )
    parser.add_argument(
        "--official-pass5-sed-csv",
        type=str,
        default="apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv",
    )
    parser.add_argument(
        "--official-v099-sed-csv",
        type=str,
        default="apply/report/assets/official-v099/wcda_crab_sed_v099_20250731_20260616_123624.csv",
    )
    parser.add_argument(
        "--offsource-stage-e-npz",
        action="append",
        default=[
            "apply/output/stage_e_v3_offsource_control/runs/v3_stage_e_offsource_ra93p63/signal_v3_offsource.npz",
            "apply/output/stage_e_v3_offsource_control/runs/v3_stage_e_offsource_ra73p63/signal_v3_offsource.npz",
        ],
    )
    parser.add_argument("--quadrature-points", type=int, default=96)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pass5_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    energy = np.asarray([float(row["energy_tev"]) for row in rows], dtype=np.float64)
    dnde = np.asarray([float(row["flux_per_tev_cm2_s"]) for row in rows], dtype=np.float64)
    return sort_spectrum(energy, dnde)


def load_v099_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    energy = np.asarray([float(row["energy_tev"]) for row in rows], dtype=np.float64)
    e2_flux = np.asarray([float(row["e2_flux_scaled_1e14_tev_cm2_s"]) * 1.0e-14 for row in rows], dtype=np.float64)
    dnde = e2_flux / np.square(energy)
    return sort_spectrum(energy, dnde)


def sort_spectrum(energy: np.ndarray, dnde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(energy) & np.isfinite(dnde) & (energy > 0.0) & (dnde > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two valid spectrum points")
    order = np.argsort(energy[valid])
    return energy[valid][order], dnde[valid][order]


def loglog_powerlaw_interp(energy_tev: np.ndarray, knots_energy: np.ndarray, knots_dnde: np.ndarray) -> np.ndarray:
    """Piecewise power-law interpolation, with endpoint slopes used for extrapolation."""
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


def integrate_flux_bins(
    loge_edges: np.ndarray,
    spectrum_energy: np.ndarray,
    spectrum_dnde: np.ndarray,
    quadrature_points: int,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    out = np.zeros(loge_edges.size - 1, dtype=np.float64)
    for idx, (lo, hi) in enumerate(zip(loge_edges[:-1], loge_edges[1:])):
        xs = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        energy_tev = np.power(10.0, xs) / 1000.0
        flux = loglog_powerlaw_interp(energy_tev, spectrum_energy, spectrum_dnde)
        # Stage A logE is log10(E/GeV); E_TeV = 10**logE / 1000.
        integrand = flux * math.log(10.0) * energy_tev
        out[idx] = 0.5 * (hi - lo) * float(np.sum(weights * integrand))
    return out


def model_counts(
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    loge_edges: np.ndarray,
    spectrum_energy: np.ndarray,
    spectrum_dnde: np.ndarray,
    quadrature_points: int,
) -> np.ndarray:
    flux_integral = integrate_flux_bins(loge_edges, spectrum_energy, spectrum_dnde, quadrature_points)
    counts = M2_TO_CM2 * np.einsum("bet,e,t->b", a_eff_m2, flux_integral, theta_exposure_sec)
    return np.asarray(containment, dtype=np.float64) * np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)


def known_background_combined_sigma(n_on: np.ndarray, b_on: np.ndarray) -> float:
    n = np.asarray(n_on, dtype=np.float64)
    b = np.asarray(b_on, dtype=np.float64)
    valid = np.isfinite(n) & np.isfinite(b) & (b > 0.0)
    if not np.any(valid):
        return float("nan")
    term = np.zeros(n.shape, dtype=np.float64)
    positive = valid & (n > 0.0)
    term[positive] = n[positive] * np.log(n[positive] / b[positive]) - (n[positive] - b[positive])
    term[valid & (n <= 0.0)] = b[valid & (n <= 0.0)]
    sigma = math.sqrt(2.0 * max(float(np.sum(term[valid])), 0.0))
    if float(np.sum(n[valid] - b[valid])) < 0.0:
        sigma *= -1.0
    return sigma


def ratio_summary(ratio: np.ndarray) -> Dict[str, object]:
    valid = np.isfinite(ratio) & (ratio > 0.0)
    if not np.any(valid):
        return {
            "valid_ratio_cells": 0,
            "median_observed_over_expected": "",
            "cells_observed_over_expected_gt_1": 0,
            "cells_observed_over_expected_gt_1p5": 0,
        }
    values = ratio[valid]
    return {
        "valid_ratio_cells": int(values.size),
        "median_observed_over_expected": float(np.nanmedian(values)),
        "cells_observed_over_expected_gt_1": int(np.count_nonzero(values > 1.0)),
        "cells_observed_over_expected_gt_1p5": int(np.count_nonzero(values > 1.5)),
    }


def run_forward_fold(
    *,
    run_label: str,
    stage_f_npz: Path,
    response: Dict[str, np.ndarray],
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    quadrature_points: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    with np.load(stage_f_npz, allow_pickle=False) as stage_f:
        cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)
        nhit_bin = np.asarray(stage_f["nhit_bin"]).astype(str)
        predE_bin = np.asarray(stage_f["predE_bin"]).astype(str)
        n_on = np.asarray(stage_f["N_on"], dtype=np.float64)
        b_on = np.asarray(stage_f["B_on"], dtype=np.float64)
        excess = np.asarray(stage_f["excess"], dtype=np.float64)
        err = np.asarray(stage_f["excess_err_conservative"], dtype=np.float64)
        containment = np.asarray(stage_f["containment_r_opt"], dtype=np.float64)
        theta_exposure = np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64)

    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    index = np.asarray([int(np.where(response_ids == cid)[0][0]) for cid in cell_id], dtype=np.int64)
    a_eff = np.asarray(response["a_eff"], dtype=np.float64)[index]
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)

    cell_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    nhit_rows: List[Dict[str, object]] = []
    for spectrum_name, (spectrum_energy, spectrum_dnde) in spectra.items():
        expected = model_counts(
            a_eff_m2=a_eff,
            containment=containment,
            theta_exposure_sec=theta_exposure,
            loge_edges=loge_edges,
            spectrum_energy=spectrum_energy,
            spectrum_dnde=spectrum_dnde,
            quadrature_points=quadrature_points,
        )
        ratio = np.divide(excess, expected, out=np.full_like(excess, np.nan), where=expected > 0.0)
        pull = np.divide(excess - expected, err, out=np.full_like(excess, np.nan), where=err > 0.0)
        for idx, cid in enumerate(cell_id):
            cell_rows.append(
                {
                    "run": run_label,
                    "spectrum": spectrum_name,
                    "cell_id": int(cid),
                    "nhit_bin": nhit_bin[idx],
                    "predE_bin": predE_bin[idx],
                    "N_on": int(n_on[idx]),
                    "B_on": float(b_on[idx]),
                    "excess": float(excess[idx]),
                    "expected_counts": float(expected[idx]),
                    "observed_over_expected": float(ratio[idx]) if np.isfinite(ratio[idx]) else "",
                    "excess_minus_expected": float(excess[idx] - expected[idx]),
                    "pull_conservative": float(pull[idx]) if np.isfinite(pull[idx]) else "",
                    "containment_r_opt": float(containment[idx]),
                }
            )
        total_excess = float(np.nansum(excess))
        total_expected = float(np.nansum(expected))
        summary = {
            "run": run_label,
            "spectrum": spectrum_name,
            "cells": int(cell_id.size),
            "total_excess": total_excess,
            "total_expected_counts": total_expected,
            "total_observed_over_expected": total_excess / total_expected if total_expected > 0.0 else "",
            "source_visible_live_days": float(np.nansum(theta_exposure) / 86400.0),
            **ratio_summary(ratio),
            "spectrum_method": "loglog_piecewise_powerlaw_interpolation_with_endpoint_slope_extrapolation",
            "energy_min_tev": float(np.nanmin(spectrum_energy)),
            "energy_max_tev": float(np.nanmax(spectrum_energy)),
        }
        summary_rows.append(summary)

        for nhit in sorted(set(nhit_bin.tolist())):
            mask = nhit_bin == nhit
            nhit_excess = float(np.nansum(excess[mask]))
            nhit_expected = float(np.nansum(expected[mask]))
            nhit_ratio = np.divide(excess[mask], expected[mask], out=np.full(np.count_nonzero(mask), np.nan), where=expected[mask] > 0.0)
            nhit_rows.append(
                {
                    "run": run_label,
                    "spectrum": spectrum_name,
                    "nhit_bin": nhit,
                    "cells": int(np.count_nonzero(mask)),
                    "total_excess": nhit_excess,
                    "total_expected_counts": nhit_expected,
                    "total_observed_over_expected": nhit_excess / nhit_expected if nhit_expected > 0.0 else "",
                    **ratio_summary(nhit_ratio),
                }
            )
    return cell_rows, summary_rows, nhit_rows


def offsource_rows(
    *,
    label: str,
    stage_e_npz: Path,
    active_ids: np.ndarray,
    legacy_ids: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    with np.load(stage_e_npz, allow_pickle=False) as data:
        ids = np.asarray(data["cell_id"], dtype=np.int64)
        nhit = np.asarray(data["nhit_bin"]).astype(str)
        pred = np.asarray(data["predE_bin"]).astype(str)
        n_on = np.asarray(data["N_on"], dtype=np.float64)
        b_on = np.asarray(data["B_on"], dtype=np.float64)
        excess = np.asarray(data["excess"], dtype=np.float64)
        err = np.asarray(data["excess_err_conservative"], dtype=np.float64)

    cell_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for selector_label, selector_ids in [("active_psfborrow_30cell", active_ids), ("legacy_nominal_stagef31", legacy_ids)]:
        mask = np.isin(ids, selector_ids)
        pull = np.divide(excess[mask], err[mask], out=np.full(np.count_nonzero(mask), np.nan), where=err[mask] > 0.0)
        for local_idx, global_idx in enumerate(np.where(mask)[0]):
            cell_rows.append(
                {
                    "fake_source": label,
                    "selector": selector_label,
                    "cell_id": int(ids[global_idx]),
                    "nhit_bin": nhit[global_idx],
                    "predE_bin": pred[global_idx],
                    "N_on": int(n_on[global_idx]),
                    "B_on": float(b_on[global_idx]),
                    "excess": float(excess[global_idx]),
                    "excess_over_sqrt_N_plus_B": float(pull[local_idx]) if np.isfinite(pull[local_idx]) else "",
                }
            )
        total_n = np.asarray(n_on[mask], dtype=np.float64)
        total_b = np.asarray(b_on[mask], dtype=np.float64)
        total_excess = float(np.nansum(excess[mask]))
        summary_rows.append(
            {
                "fake_source": label,
                "selector": selector_label,
                "cells": int(np.count_nonzero(mask)),
                "N_on": int(np.nansum(total_n)),
                "B_on": float(np.nansum(total_b)),
                "excess": total_excess,
                "combined_known_background_sigma": known_background_combined_sigma(total_n, total_b),
                "positive_excess_cells": int(np.count_nonzero(excess[mask] > 0.0)),
                "negative_excess_cells": int(np.count_nonzero(excess[mask] < 0.0)),
            }
        )
    return cell_rows, summary_rows


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_forward_fold_counts(rows: Sequence[Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    labels = sorted({str(row["run"]) for row in rows})
    spectra = sorted({str(row["spectrum"]) for row in rows})
    fig, axes = plt.subplots(len(labels), len(spectra), figsize=(5.4 * len(spectra), 4.4 * len(labels)), squeeze=False)
    for i, run in enumerate(labels):
        for j, spectrum in enumerate(spectra):
            ax = axes[i][j]
            selected = [row for row in rows if row["run"] == run and row["spectrum"] == spectrum]
            x = np.asarray([float(row["expected_counts"]) for row in selected], dtype=np.float64)
            y = np.asarray([float(row["excess"]) for row in selected], dtype=np.float64)
            c = np.asarray([int(row["cell_id"]) for row in selected], dtype=np.int64)
            good = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
            ax.scatter(x[good], y[good], s=28, c=c[good], cmap="viridis", alpha=0.82)
            lim_hi = max(float(np.nanmax(x[good])) if np.any(good) else 1.0, float(np.nanmax(y[good])) if np.any(good) else 1.0) * 1.15
            lim_lo = max(0.1, min(float(np.nanmin(x[good])) if np.any(good) else 0.1, float(np.nanmin(y[good])) if np.any(good) else 0.1) / 1.5)
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#555555", lw=1.0, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_title(f"{run}\n{spectrum}")
            ax.set_xlabel("official/tutorial expected counts")
            ax.set_ylabel("Stage E excess")
            ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_forward_fold_ratio(rows: Sequence[Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    selected = [row for row in rows if row["run"] == "active_psfborrow_30cell"]
    spectra = sorted({str(row["spectrum"]) for row in selected})
    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    width = 0.36
    for offset, spectrum in zip(np.linspace(-width / 2, width / 2, max(len(spectra), 1)), spectra):
        subset = sorted([row for row in selected if row["spectrum"] == spectrum], key=lambda row: int(row["cell_id"]))
        x = np.arange(len(subset), dtype=np.float64) + offset
        y = [float(row["observed_over_expected"]) for row in subset]
        ax.bar(x, y, width=width, label=spectrum, alpha=0.78)
    cell_labels = [str(row["cell_id"]) for row in sorted([row for row in selected if row["spectrum"] == spectra[0]], key=lambda row: int(row["cell_id"]))] if spectra else []
    ax.axhline(1.0, color="#333333", ls="--", lw=1.0)
    ax.axhline(1.5, color="#b45309", ls=":", lw=1.1)
    ax.set_xticks(np.arange(len(cell_labels)))
    ax.set_xticklabels(cell_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Stage E excess / official expected")
    ax.set_title("Official/tutorial forward-fold normalization ratios, active 30-cell branch")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_offsource_core(rows: Sequence[Dict[str, object]], path: Path) -> None:
    plt = setup_matplotlib()
    selected = [row for row in rows if row["selector"] == "active_psfborrow_30cell"]
    fake_sources = sorted({str(row["fake_source"]) for row in selected})
    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    width = 0.36
    for offset, source in zip(np.linspace(-width / 2, width / 2, max(len(fake_sources), 1)), fake_sources):
        subset = sorted([row for row in selected if row["fake_source"] == source], key=lambda row: int(row["cell_id"]))
        x = np.arange(len(subset), dtype=np.float64) + offset
        y = [float(row["excess"]) for row in subset]
        ax.bar(x, y, width=width, label=source, alpha=0.78)
    cell_labels = [str(row["cell_id"]) for row in sorted([row for row in selected if row["fake_source"] == fake_sources[0]], key=lambda row: int(row["cell_id"]))] if fake_sources else []
    ax.axhline(0.0, color="#333333", lw=1.0)
    ax.set_xticks(np.arange(len(cell_labels)))
    ax.set_xticklabels(cell_labels, rotation=90, fontsize=7)
    ax.set_ylabel("fake-source core excess = N_on - B_on")
    ax.set_title("Off-source core residuals, active 30-cell branch")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(resolve(args.response_npz), allow_pickle=False) as response_npz:
        response = {key: response_npz[key] for key in response_npz.files}

    spectra = {
        "official_pass5": load_pass5_spectrum(resolve(args.official_pass5_sed_csv)),
        "tutorial_v099": load_v099_spectrum(resolve(args.official_v099_sed_csv)),
    }

    forward_cell_rows: List[Dict[str, object]] = []
    forward_summary_rows: List[Dict[str, object]] = []
    forward_nhit_rows: List[Dict[str, object]] = []
    for run_label, stage_f_npz in [
        ("active_psfborrow_30cell", resolve(args.active_stage_f_npz)),
        ("legacy_nominal_stagef31", resolve(args.legacy_stage_f_npz)),
    ]:
        cell_rows, summary_rows, nhit_rows = run_forward_fold(
            run_label=run_label,
            stage_f_npz=stage_f_npz,
            response=response,
            spectra=spectra,
            quadrature_points=int(args.quadrature_points),
        )
        forward_cell_rows.extend(cell_rows)
        forward_summary_rows.extend(summary_rows)
        forward_nhit_rows.extend(nhit_rows)

    with np.load(resolve(args.active_stage_f_npz), allow_pickle=False) as active_stage_f:
        active_ids = np.asarray(active_stage_f["cell_id"], dtype=np.int64)
    with np.load(resolve(args.legacy_stage_f_npz), allow_pickle=False) as legacy_stage_f:
        legacy_ids = np.asarray(legacy_stage_f["cell_id"], dtype=np.int64)

    offsource_cell_rows: List[Dict[str, object]] = []
    offsource_summary_rows: List[Dict[str, object]] = []
    for path in args.offsource_stage_e_npz:
        stage_e_npz = resolve(path)
        label = stage_e_npz.parent.name.replace("v3_stage_e_", "")
        cell_rows, summary_rows = offsource_rows(
            label=label,
            stage_e_npz=stage_e_npz,
            active_ids=active_ids,
            legacy_ids=legacy_ids,
        )
        offsource_cell_rows.extend(cell_rows)
        offsource_summary_rows.extend(summary_rows)

    forward_cell_csv = output_dir / "v3_official_forward_fold_cell_counts.csv"
    forward_summary_csv = output_dir / "v3_official_forward_fold_summary.csv"
    forward_nhit_csv = output_dir / "v3_official_forward_fold_nhit_summary.csv"
    offsource_cell_csv = output_dir / "v3_offsource_core_residual_cells.csv"
    offsource_summary_csv = output_dir / "v3_offsource_core_residual_summary.csv"
    summary_json = output_dir / "v3_normalization_diagnostics_summary.json"
    counts_png = output_dir / "v3_official_forward_fold_counts_vs_excess.png"
    ratio_png = output_dir / "v3_official_forward_fold_ratio_by_cell.png"
    offsource_png = output_dir / "v3_offsource_core_residual_by_cell.png"

    write_csv(
        forward_cell_csv,
        forward_cell_rows,
        [
            "run",
            "spectrum",
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "N_on",
            "B_on",
            "excess",
            "expected_counts",
            "observed_over_expected",
            "excess_minus_expected",
            "pull_conservative",
            "containment_r_opt",
        ],
    )
    write_csv(
        forward_summary_csv,
        forward_summary_rows,
        [
            "run",
            "spectrum",
            "cells",
            "total_excess",
            "total_expected_counts",
            "total_observed_over_expected",
            "source_visible_live_days",
            "valid_ratio_cells",
            "median_observed_over_expected",
            "cells_observed_over_expected_gt_1",
            "cells_observed_over_expected_gt_1p5",
            "spectrum_method",
            "energy_min_tev",
            "energy_max_tev",
        ],
    )
    write_csv(
        forward_nhit_csv,
        forward_nhit_rows,
        [
            "run",
            "spectrum",
            "nhit_bin",
            "cells",
            "total_excess",
            "total_expected_counts",
            "total_observed_over_expected",
            "valid_ratio_cells",
            "median_observed_over_expected",
            "cells_observed_over_expected_gt_1",
            "cells_observed_over_expected_gt_1p5",
        ],
    )
    write_csv(
        offsource_cell_csv,
        offsource_cell_rows,
        [
            "fake_source",
            "selector",
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "N_on",
            "B_on",
            "excess",
            "excess_over_sqrt_N_plus_B",
        ],
    )
    write_csv(
        offsource_summary_csv,
        offsource_summary_rows,
        [
            "fake_source",
            "selector",
            "cells",
            "N_on",
            "B_on",
            "excess",
            "combined_known_background_sigma",
            "positive_excess_cells",
            "negative_excess_cells",
        ],
    )

    plot_forward_fold_counts(forward_cell_rows, counts_png)
    plot_forward_fold_ratio(forward_cell_rows, ratio_png)
    plot_offsource_core(offsource_cell_rows, offsource_png)

    active_forward = [
        row for row in forward_summary_rows if row.get("run") == "active_psfborrow_30cell"
    ]
    active_offsource = [
        row for row in offsource_summary_rows if row.get("selector") == "active_psfborrow_30cell"
    ]
    forward_conclusion = "stage_e_excess_above_official_forward_fold"
    if active_forward and all(float(row["total_observed_over_expected"]) <= 1.0 for row in active_forward):
        forward_conclusion = "stage_e_excess_not_above_official_forward_fold"
    offsource_conclusion = "fake_sources_do_not_show_positive_core_residuals"
    if active_offsource and any(float(row["excess"]) > 0.0 for row in active_offsource):
        offsource_conclusion = "at_least_one_fake_source_has_positive_core_residual"

    write_json(
        summary_json,
        {
            "inputs": {
                "response_npz": str(resolve(args.response_npz)),
                "active_stage_f_npz": str(resolve(args.active_stage_f_npz)),
                "legacy_stage_f_npz": str(resolve(args.legacy_stage_f_npz)),
                "official_pass5_sed_csv": str(resolve(args.official_pass5_sed_csv)),
                "official_v099_sed_csv": str(resolve(args.official_v099_sed_csv)),
                "offsource_stage_e_npz": [str(resolve(path)) for path in args.offsource_stage_e_npz],
            },
            "method": {
                "spectrum_interpolation": "log-log piecewise power-law interpolation; endpoint slopes extrapolate beyond transferred SED points",
                "counts_formula": "containment_r_opt * 1e4 * sum_E,theta(A_eff_m2 * theta_exposure_sec * integral_flux_TeV)",
                "energy_grid": "Stage A log10(E_true/GeV) bin edges",
                "theta_exposure": "Stage F Crab-visible theta exposure from Stage C source_files.csv",
                "quadrature_points": int(args.quadrature_points),
            },
            "outputs": {
                "forward_cell_csv": str(forward_cell_csv),
                "forward_summary_csv": str(forward_summary_csv),
                "forward_nhit_csv": str(forward_nhit_csv),
                "offsource_cell_csv": str(offsource_cell_csv),
                "offsource_summary_csv": str(offsource_summary_csv),
                "counts_vs_excess_png": str(counts_png),
                "ratio_by_cell_png": str(ratio_png),
                "offsource_core_residual_png": str(offsource_png),
            },
            "forward_summary": forward_summary_rows,
            "forward_nhit_summary": forward_nhit_rows,
            "offsource_summary": offsource_summary_rows,
            "conclusions": {
                "official_forward_fold": forward_conclusion,
                "offsource_core_residual": offsource_conclusion,
                "interpretation": (
                    "Active v3 Stage E excess is compared against official/tutorial spectra using the same Stage A response "
                    "and Stage F exposure. Fake-source controls test the sign of core residuals in source-free positions."
                ),
            },
        },
    )
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
