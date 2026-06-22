#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import build_v3_normalization_diagnostics as norm
import importlib.util
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "apply/report/assets/v4-root-cause-diagnostics"
M2_TO_CM2 = 1.0e4

RESPONSE_NPZ = REPO_ROOT / "apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
V099_CSV = REPO_ROOT / "apply/report/assets/official-v099/wcda_crab_sed_v099_20250731_20260616_123624.csv"
PSF_CSV = REPO_ROOT / "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate_summary.csv"

ACTIVE30_F_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v3_baseline_annnorm/runs/v3_stage_f_annnorm_from_psfborrow/fit_v3_baseline_annnorm.npz"
)
DROP4_F_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v4_drop4_annnorm/runs/v4_stage_f_annnorm_drop_cells_4_17_39_43/fit_v4_drop4_annnorm.npz"
)
NHIT_ONLY_F_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v3_nhit_only/runs/v3_stage_f_nhit_only_slurm_42036/fit_v3_nhit_only.npz"
)

ACTIVE30_FOLD_CSV = OUT_DIR / "active30_forward_fold_cell_counts.csv"
DROP4_FOLD_CSV = OUT_DIR / "drop4_forward_fold_cell_counts.csv"
ACTIVE30_NHIT_CSV = OUT_DIR / "active30_forward_fold_nhit_summary.csv"
DROP4_NHIT_CSV = OUT_DIR / "drop4_forward_fold_nhit_summary.csv"
ACTIVE30_SUMMARY_CSV = OUT_DIR / "active30_forward_fold_summary.csv"
DROP4_SUMMARY_CSV = OUT_DIR / "drop4_forward_fold_summary.csv"

OFFSOURCE_NPZS = [
    REPO_ROOT / "apply/output/stage_e_v3_offsource_control/runs/v3_stage_e_offsource_ra93p63/signal_v3_offsource.npz",
    REPO_ROOT / "apply/output/stage_e_v3_offsource_control/runs/v3_stage_e_offsource_ra73p63/signal_v3_offsource.npz",
]


def load_stage06():
    module_path = REPO_ROOT / "apply/stages/06_fit.py"
    spec = importlib.util.spec_from_file_location("stage06_fit_for_v4_diagnostics", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def interval_key(label: str) -> float:
    text = str(label or "")
    if text.startswith("[") and "," in text:
        try:
            return float(text[1:].split(",", 1)[0])
        except ValueError:
            return 1.0e9
    return 1.0e9


def stage_f_specs() -> list[tuple[str, Path]]:
    specs = [("active30", ACTIVE30_F_NPZ), ("drop4", DROP4_F_NPZ)]
    if NHIT_ONLY_F_NPZ.exists():
        specs.append(("nhit_only", NHIT_ONLY_F_NPZ))
    return [(label, path) for label, path in specs if path.exists()]


def load_response() -> dict[str, np.ndarray]:
    with np.load(RESPONSE_NPZ, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def forward_fold_outputs() -> dict[str, dict[str, list[dict[str, Any]]]]:
    response = load_response()
    spectra = {
        "official_pass5": norm.load_pass5_spectrum(PASS5_CSV),
        "tutorial_v099": norm.load_v099_spectrum(V099_CSV),
    }
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, npz_path in stage_f_specs():
        cells, summary, nhit = norm.run_forward_fold(
            run_label=label,
            stage_f_npz=npz_path,
            response=response,
            spectra=spectra,
            quadrature_points=96,
        )
        out[label] = {"cells": cells, "summary": summary, "nhit": nhit}
    return out


def write_forward_fold_tables(outputs: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    cell_fields = [
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
    ]
    summary_fields = [
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
    ]
    nhit_fields = [
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
    ]
    for label, paths in {
        "active30": (ACTIVE30_FOLD_CSV, ACTIVE30_SUMMARY_CSV, ACTIVE30_NHIT_CSV),
        "drop4": (DROP4_FOLD_CSV, DROP4_SUMMARY_CSV, DROP4_NHIT_CSV),
    }.items():
        if label not in outputs:
            continue
        write_csv(paths[0], outputs[label]["cells"], cell_fields)
        write_csv(paths[1], outputs[label]["summary"], summary_fields)
        write_csv(paths[2], outputs[label]["nhit"], nhit_fields)


def required_b_shift_rows(outputs: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_rows: list[dict[str, Any]] = []
    for selector, payload in outputs.items():
        for row in payload["cells"]:
            if row.get("spectrum") != "official_pass5":
                continue
            b_on = finite(row.get("B_on"))
            delta = finite(row.get("excess_minus_expected"))
            cell_rows.append(
                {
                    **row,
                    "selector": selector,
                    "required_delta_b": delta,
                    "required_delta_b_over_b": (delta / b_on if b_on and delta is not None else ""),
                }
            )
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in cell_rows:
        key = (str(row["selector"]), str(row["nhit_bin"]))
        item = by_key.setdefault(
            key,
            {
                "selector": row["selector"],
                "nhit_bin": row["nhit_bin"],
                "cells": 0,
                "total_required_delta_b": 0.0,
                "total_b_on": 0.0,
                "total_excess": 0.0,
                "total_expected_counts": 0.0,
            },
        )
        item["cells"] += 1
        item["total_required_delta_b"] += float(row["required_delta_b"])
        item["total_b_on"] += float(row["B_on"])
        item["total_excess"] += float(row["excess"])
        item["total_expected_counts"] += float(row["expected_counts"])
    summary_rows = []
    for item in by_key.values():
        total_b = float(item["total_b_on"])
        summary_rows.append(
            {
                **item,
                "required_delta_b_over_b": item["total_required_delta_b"] / total_b if total_b > 0 else "",
                "observed_over_expected": item["total_excess"] / item["total_expected_counts"]
                if item["total_expected_counts"] > 0
                else "",
            }
        )
    summary_rows.sort(key=lambda row: (str(row["selector"]), interval_key(str(row["nhit_bin"]))))
    cell_rows.sort(key=lambda row: (str(row["selector"]), -float(row["required_delta_b"])))
    return cell_rows, summary_rows


def offsource_rows(outputs: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selector_ids: dict[str, np.ndarray] = {}
    for label, npz_path in stage_f_specs():
        with np.load(npz_path, allow_pickle=False) as data:
            selector_ids[label] = np.asarray(data["cell_id"], dtype=np.int64)
    cell_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for path in OFFSOURCE_NPZS:
        if not path.exists():
            continue
        label = path.parent.name.replace("v3_stage_e_", "")
        rows, summary = norm.offsource_rows(label=label, stage_e_npz=path, selector_ids_by_label=selector_ids)
        cell_rows.extend(rows)
        summary_rows.extend(summary)
    return cell_rows, summary_rows


def energy_contribution_rows() -> list[dict[str, Any]]:
    response = load_response()
    pass5_energy, pass5_dnde = norm.load_pass5_spectrum(PASS5_CSV)
    flux_integral = norm.integrate_flux_bins(
        np.asarray(response["logE_true_edges"], dtype=np.float64), pass5_energy, pass5_dnde, 96
    )
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    loge_centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    energy_tev = np.power(10.0, loge_centers) / 1000.0
    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for selector, npz_path in stage_f_specs():
        with np.load(npz_path, allow_pickle=False) as stage_f:
            cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)
            nhit = np.asarray(stage_f["nhit_bin"]).astype(str)
            pred = np.asarray(stage_f["predE_bin"]).astype(str)
            containment = np.asarray(stage_f["containment_r_opt"], dtype=np.float64)
            theta_exposure = np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64)
        for idx, cid in enumerate(cell_id):
            ridx = int(np.where(response_ids == cid)[0][0])
            a_eff = np.asarray(response["a_eff"], dtype=np.float64)[ridx]
            by_energy = M2_TO_CM2 * np.einsum("et,e,t->e", a_eff, flux_integral, theta_exposure)
            by_energy = containment[idx] * np.nan_to_num(by_energy, nan=0.0, posinf=0.0, neginf=0.0)
            total = float(np.sum(by_energy))
            if total <= 0:
                continue
            cdf = np.cumsum(by_energy) / total
            rows.append(
                {
                    "selector": selector,
                    "cell_id": int(cid),
                    "nhit_bin": nhit[idx],
                    "predE_bin": pred[idx],
                    "expected_counts": total,
                    "true_e10_tev": float(np.interp(0.10, cdf, energy_tev)),
                    "true_e50_tev": float(np.interp(0.50, cdf, energy_tev)),
                    "true_e90_tev": float(np.interp(0.90, cdf, energy_tev)),
                    "frac_below_pass5_min": float(np.sum(by_energy[energy_tev < float(np.min(pass5_energy))]) / total),
                    "frac_below_1tev": float(np.sum(by_energy[energy_tev < 1.0]) / total),
                    "frac_above_10tev": float(np.sum(by_energy[energy_tev > 10.0]) / total),
                }
            )
    return rows


def low_energy_extrapolation_sensitivity(outputs: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    response = load_response()
    pass5_energy, pass5_dnde = norm.load_pass5_spectrum(PASS5_CSV)
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "pass5_endpoint_extrap": (pass5_energy, pass5_dnde),
        "pass5_flat_below_min": (
            np.concatenate([[0.05], pass5_energy]),
            np.concatenate([[pass5_dnde[0]], pass5_dnde]),
        ),
        "pass5_cut_below_min": (pass5_energy, pass5_dnde),
    }
    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for selector, npz_path in stage_f_specs():
        with np.load(npz_path, allow_pickle=False) as stage_f:
            cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)
            nhit = np.asarray(stage_f["nhit_bin"]).astype(str)
            containment = np.asarray(stage_f["containment_r_opt"], dtype=np.float64)
            theta_exposure = np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64)
            excess = np.asarray(stage_f["excess"], dtype=np.float64)
        for variant, (energy, dnde) in variants.items():
            flux_integral = norm.integrate_flux_bins(loge_edges, energy, dnde, 96)
            if variant == "pass5_cut_below_min":
                centers_tev = np.power(10.0, 0.5 * (loge_edges[:-1] + loge_edges[1:])) / 1000.0
                flux_integral = np.where(centers_tev < float(np.min(pass5_energy)), 0.0, flux_integral)
            expected = []
            for idx, cid in enumerate(cell_id):
                ridx = int(np.where(response_ids == cid)[0][0])
                a_eff = np.asarray(response["a_eff"], dtype=np.float64)[ridx]
                counts = M2_TO_CM2 * float(np.einsum("et,e,t->", a_eff, flux_integral, theta_exposure))
                expected.append(float(containment[idx]) * counts)
            expected_arr = np.asarray(expected, dtype=np.float64)
            for label in sorted(set(nhit), key=interval_key):
                mask = nhit == label
                total_expected = float(np.sum(expected_arr[mask]))
                total_excess = float(np.sum(excess[mask]))
                rows.append(
                    {
                        "selector": selector,
                        "variant": variant,
                        "nhit_bin": label,
                        "cells": int(np.count_nonzero(mask)),
                        "total_excess": total_excess,
                        "total_expected_counts": total_expected,
                        "observed_over_expected": total_excess / total_expected if total_expected > 0 else "",
                    }
                )
    return rows


def response_closure_rows() -> list[dict[str, Any]]:
    stage06 = load_stage06()
    response = load_response()
    pass5_energy, pass5_dnde = norm.load_pass5_spectrum(PASS5_CSV)
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    flux_integral = norm.integrate_flux_bins(loge_edges, pass5_energy, pass5_dnde, 96)
    response_ids = np.asarray(response["cell_id"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for selector, npz_path in stage_f_specs():
        with np.load(npz_path, allow_pickle=False) as stage_f:
            cell_id = np.asarray(stage_f["cell_id"], dtype=np.int64)
            containment = np.asarray(stage_f["containment_r_opt"], dtype=np.float64)
            theta_exposure = np.asarray(stage_f["theta_exposure_sec"], dtype=np.float64)
        index = np.asarray([int(np.where(response_ids == cid)[0][0]) for cid in cell_id], dtype=np.int64)
        a_eff = np.asarray(response["a_eff"], dtype=np.float64)[index]
        expected = M2_TO_CM2 * np.einsum("bet,e,t->b", a_eff, flux_integral, theta_exposure)
        expected = containment * np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
        errors = np.sqrt(np.maximum(expected, 1.0))
        pl = stage06.fit_model(
            model_name="pl",
            error_mode="closure_sqrt_expected",
            observed=expected,
            errors=errors,
            a_eff_m2=a_eff,
            containment=containment,
            theta_exposure_sec=theta_exposure,
            loge_edges=loge_edges,
            pivot_tev=3.0,
            quadrature_points=96,
            start_gamma=2.69,
            start_phi0=2.114e-12,
        )
        logpar = stage06.fit_model(
            model_name="logpar",
            error_mode="closure_sqrt_expected",
            observed=expected,
            errors=errors,
            a_eff_m2=a_eff,
            containment=containment,
            theta_exposure_sec=theta_exposure,
            loge_edges=loge_edges,
            pivot_tev=3.0,
            quadrature_points=96,
            start_gamma=float(pl.parameters.get("gamma", 2.69)),
            start_phi0=float(pl.parameters.get("phi0", 2.114e-12)),
        )
        for fit in [pl, logpar]:
            params = fit.parameters
            rows.append(
                {
                    "selector": selector,
                    "model": fit.model_name,
                    "cells": int(cell_id.size),
                    "chi2": fit.chi2,
                    "ndof": fit.ndof,
                    "chi2_over_ndof": fit.chi2 / fit.ndof if fit.ndof > 0 else "",
                    "phi0": params.get("phi0", ""),
                    "gamma": params.get("gamma", ""),
                    "alpha": params.get("alpha", ""),
                    "beta": params.get("beta", ""),
                    "max_abs_pull": float(np.nanmax(np.abs(fit.pull))),
                    "total_expected_counts": float(np.sum(expected)),
                    "closure_note": "official pass5 folded counts used as pseudo-observed excess",
                }
            )
    return rows


def selector_dependence_rows(outputs: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selector, payload in outputs.items():
        if "summary" not in payload:
            continue
        for row in payload["summary"]:
            if row.get("spectrum") == "official_pass5":
                rows.append({"selector": selector, **row})
    return rows


def plot_required_b_shift(summary_rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    labels = sorted({str(row["nhit_bin"]) for row in summary_rows}, key=interval_key)
    selectors = sorted({str(row["selector"]) for row in summary_rows})
    x = np.arange(len(labels))
    width = 0.8 / max(len(selectors), 1)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
    for i, selector in enumerate(selectors):
        vals = []
        for label in labels:
            row = next((r for r in summary_rows if r["selector"] == selector and r["nhit_bin"] == label), None)
            vals.append(100.0 * float(row["required_delta_b_over_b"]) if row else np.nan)
        ax.bar(x + (i - (len(selectors) - 1) / 2) * width, vals, width=width * 0.9, label=selector)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Required B_on shift to match official (%)")
    ax.set_title("How much background increase would make official pass5 match the excess?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_extrapolation_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    labels = sorted({str(row["nhit_bin"]) for row in rows}, key=interval_key)
    selectors = ["active30", "drop4"]
    variants = ["pass5_endpoint_extrap", "pass5_flat_below_min", "pass5_cut_below_min"]
    fig, axes = plt.subplots(1, len(selectors), figsize=(12.0, 4.4), dpi=160, sharey=True)
    for ax, selector in zip(axes, selectors):
        for variant in variants:
            vals = []
            for label in labels:
                row = next(
                    (
                        r
                        for r in rows
                        if r["selector"] == selector and r["variant"] == variant and r["nhit_bin"] == label
                    ),
                    None,
                )
                vals.append(float(row["observed_over_expected"]) if row and row["observed_over_expected"] != "" else np.nan)
            ax.plot(labels, vals, marker="o", lw=1.5, label=variant)
        ax.axhline(1.0, color="#111827", lw=0.8)
        ax.set_title(selector)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed excess / expected counts")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Sensitivity to official pass5 low-energy extrapolation")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_offsource_summary(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    selectors = sorted({str(row["selector"]) for row in rows})
    fake_sources = sorted({str(row["fake_source"]) for row in rows})
    x = np.arange(len(selectors))
    width = 0.8 / max(len(fake_sources), 1)
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=160)
    for i, source in enumerate(fake_sources):
        vals = []
        for selector in selectors:
            row = next((r for r in rows if r["selector"] == selector and r["fake_source"] == source), None)
            vals.append(float(row["combined_known_background_sigma"]) if row else np.nan)
        ax.bar(x + (i - (len(fake_sources) - 1) / 2) * width, vals, width=width * 0.9, label=source)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(selectors, rotation=25, ha="right")
    ax.set_ylabel("Known-background combined sigma")
    ax.set_title("Off-source pseudo-Crab residual control")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = forward_fold_outputs()
    write_forward_fold_tables(outputs)

    b_cell, b_summary = required_b_shift_rows(outputs)
    write_csv(
        OUT_DIR / "required_background_shift_cells.csv",
        b_cell,
        [
            "selector",
            "run",
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "N_on",
            "B_on",
            "excess",
            "expected_counts",
            "observed_over_expected",
            "required_delta_b",
            "required_delta_b_over_b",
            "pull_conservative",
            "containment_r_opt",
        ],
    )
    write_csv(
        OUT_DIR / "required_background_shift_nhit_summary.csv",
        b_summary,
        [
            "selector",
            "nhit_bin",
            "cells",
            "total_required_delta_b",
            "total_b_on",
            "required_delta_b_over_b",
            "total_excess",
            "total_expected_counts",
            "observed_over_expected",
        ],
    )

    off_cells, off_summary = offsource_rows(outputs)
    write_csv(
        OUT_DIR / "offsource_core_residual_cells.csv",
        off_cells,
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
        OUT_DIR / "offsource_core_residual_summary.csv",
        off_summary,
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

    energy_rows = energy_contribution_rows()
    write_csv(
        OUT_DIR / "official_pass5_true_energy_contribution_by_cell.csv",
        energy_rows,
        [
            "selector",
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "expected_counts",
            "true_e10_tev",
            "true_e50_tev",
            "true_e90_tev",
            "frac_below_pass5_min",
            "frac_below_1tev",
            "frac_above_10tev",
        ],
    )

    extrap_rows = low_energy_extrapolation_sensitivity(outputs)
    write_csv(
        OUT_DIR / "official_pass5_low_energy_extrapolation_sensitivity.csv",
        extrap_rows,
        ["selector", "variant", "nhit_bin", "cells", "total_excess", "total_expected_counts", "observed_over_expected"],
    )

    closure_rows = response_closure_rows()
    write_csv(
        OUT_DIR / "official_pass5_response_closure_fit_summary.csv",
        closure_rows,
        [
            "selector",
            "model",
            "cells",
            "chi2",
            "ndof",
            "chi2_over_ndof",
            "phi0",
            "gamma",
            "alpha",
            "beta",
            "max_abs_pull",
            "total_expected_counts",
            "closure_note",
        ],
    )

    selector_rows = selector_dependence_rows(outputs)
    write_csv(
        OUT_DIR / "selector_dependence_forward_fold_summary.csv",
        selector_rows,
        [
            "selector",
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

    plot_required_b_shift(b_summary, OUT_DIR / "required_background_shift_by_nhit.png")
    plot_extrapolation_sensitivity(extrap_rows, OUT_DIR / "low_energy_extrapolation_sensitivity.png")
    plot_offsource_summary(off_summary, OUT_DIR / "offsource_pseudocrab_residual_sigma.png")

    active30 = next(row for row in selector_rows if row["selector"] == "active30" and row["spectrum"] == "official_pass5")
    drop4 = next(row for row in selector_rows if row["selector"] == "drop4" and row["spectrum"] == "official_pass5")
    conclusions = {
        "required_background_shift": (
            "The lowest Nhit bin can be moved by a few-percent B_on shift, but [200,800) requires much larger "
            "on-core B_on increases, so background alone is not a uniform explanation unless off-source controls show comparable positive residuals."
        ),
        "offsource": (
            "Existing RA-shifted pseudo-Crab controls are strongly negative for active/drop4 selectors, not positive; "
            "this argues against a simple positive excess caused by annnorm background underestimation."
        ),
        "low_energy_extrapolation": (
            "The lowest Nhit expected counts are sensitive to pass5 extrapolation below the first official point; this affects interpretation of [125,200)."
        ),
        "selector_dependence": (
            f"Drop4 does not reduce official underprediction: active30 ratio={float(active30['total_observed_over_expected']):.4g}, "
            f"drop4 ratio={float(drop4['total_observed_over_expected']):.4g}."
        ),
    }
    write_json(
        OUT_DIR / "v4_root_cause_diagnostics_summary.json",
        {
            "outputs": {
                "required_background_shift_cells": str(OUT_DIR / "required_background_shift_cells.csv"),
                "required_background_shift_nhit_summary": str(OUT_DIR / "required_background_shift_nhit_summary.csv"),
                "offsource_core_residual_summary": str(OUT_DIR / "offsource_core_residual_summary.csv"),
                "official_pass5_true_energy_contribution_by_cell": str(
                    OUT_DIR / "official_pass5_true_energy_contribution_by_cell.csv"
                ),
                "official_pass5_low_energy_extrapolation_sensitivity": str(
                    OUT_DIR / "official_pass5_low_energy_extrapolation_sensitivity.csv"
                ),
                "selector_dependence_forward_fold_summary": str(OUT_DIR / "selector_dependence_forward_fold_summary.csv"),
                "official_pass5_response_closure_fit_summary": str(
                    OUT_DIR / "official_pass5_response_closure_fit_summary.csv"
                ),
            },
            "conclusions": conclusions,
        },
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
