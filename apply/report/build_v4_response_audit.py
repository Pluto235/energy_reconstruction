#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import build_v3_latest_bkg_report as v3
import build_v3_normalization_diagnostics as norm


REPO_ROOT = v3.REPO_ROOT
OUT_DIR = REPO_ROOT / "apply/report/assets/v4-response-audit"
M2_TO_CM2 = 1.0e4

RESPONSE_NPZ = REPO_ROOT / "apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz"
RESPONSE_META = REPO_ROOT / "apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json"
SIGNAL_NPZ = (
    REPO_ROOT
    / "apply/output/stage_e_v3_candidate_annnorm/runs/v3_stage_e_annnorm_from_psfborrow/signal_v3_candidate_annnorm.npz"
)
ACTIVE30_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v3_baseline_annnorm/runs/v3_stage_f_annnorm_from_psfborrow/fit_v3_baseline_annnorm.npz"
)
DROP4_NPZ = (
    REPO_ROOT
    / "apply/output/stage_f_v4_drop4_annnorm/runs/v4_stage_f_annnorm_drop_cells_4_17_39_43/fit_v4_drop4_annnorm.npz"
)
CONTAINMENT1_F_META = (
    REPO_ROOT
    / "apply/output/stage_f_v4_containment1_drop4_annnorm/runs/v4_stage_f_annnorm_containment1_drop4/fit_v4_containment1_drop4_annnorm_metadata.json"
)
CONTAINMENT1_G_META = (
    REPO_ROOT
    / "apply/output/stage_g_v4_containment1_drop4_annnorm/runs/v4_stage_g_annnorm_containment1_drop4/sed_points_v4_containment1_drop4_annnorm_metadata.json"
)
DROP4_G_META = (
    REPO_ROOT
    / "apply/output/stage_g_v4_drop4_annnorm/runs/v4_stage_g_annnorm_drop_cells_4_17_39_43/sed_points_v4_drop4_annnorm_metadata.json"
)
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
ACTIVE30_SELECTOR = REPO_ROOT / "apply/config/cell_selector_v3_baseline_psfborrow.csv"
DROP4_SELECTOR = REPO_ROOT / "apply/config/cell_selector_v4_drop4_psfborrow.csv"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selector_mask(cell_id: np.ndarray, path: Path | None) -> np.ndarray:
    if path is None:
        return np.ones(cell_id.shape, dtype=bool)
    include_ids: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}:
                include_ids.add(int(row["cell_id"]))
    return np.asarray([int(cid) in include_ids for cid in cell_id], dtype=bool)


def interval_key(label: str) -> float:
    text = str(label or "")
    if text.startswith("[") and "," in text:
        try:
            return float(text[1:].split(",", 1)[0])
        except ValueError:
            return 1.0e9
    return 1.0e9


def expected_counts(response: dict[str, np.ndarray], signal: dict[str, np.ndarray], theta_exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass5_energy, pass5_dnde = norm.load_pass5_spectrum(PASS5_CSV)
    flux_integral = norm.integrate_flux_bins(
        np.asarray(response["logE_true_edges"], dtype=np.float64),
        pass5_energy,
        pass5_dnde,
        96,
    )
    expected_raw = M2_TO_CM2 * np.einsum(
        "bet,e,t->b",
        np.asarray(response["a_eff"], dtype=np.float64),
        flux_integral,
        np.asarray(theta_exposure, dtype=np.float64),
    )
    containment = np.asarray(signal["containment_r_opt"], dtype=np.float64)
    return expected_raw * containment, expected_raw


def selector_summary_rows(
    signal: dict[str, np.ndarray],
    expected_nominal: np.ndarray,
    expected_containment1: np.ndarray,
) -> list[dict[str, Any]]:
    cell_id = np.asarray(signal["cell_id"], dtype=np.int64)
    nhit = np.asarray(signal["nhit_bin"]).astype(str)
    excess = np.asarray(signal["excess"], dtype=np.float64)
    n_on = np.asarray(signal["N_on"], dtype=np.float64)
    b_on = np.asarray(signal["B_on"], dtype=np.float64)
    selectors = {
        "all84": selector_mask(cell_id, None),
        "positive_excess_all84": excess > 0,
        "active30": selector_mask(cell_id, ACTIVE30_SELECTOR),
        "drop4": selector_mask(cell_id, DROP4_SELECTOR),
    }
    rows: list[dict[str, Any]] = []
    for selector, mask in selectors.items():
        for containment_mode, expected in [
            ("nominal_containment", expected_nominal),
            ("containment_1", expected_containment1),
        ]:
            total_excess = float(np.nansum(excess[mask]))
            total_expected = float(np.nansum(expected[mask]))
            total_b = float(np.nansum(b_on[mask]))
            total_n = float(np.nansum(n_on[mask]))
            rows.append(
                {
                    "selector": selector,
                    "containment_mode": containment_mode,
                    "nhit_bin": "all",
                    "cells": int(np.count_nonzero(mask)),
                    "N_on": total_n,
                    "B_on": total_b,
                    "excess": total_excess,
                    "official_expected_counts": total_expected,
                    "observed_over_expected": total_excess / total_expected if total_expected > 0 else "",
                    "N_on_over_B_on": total_n / total_b if total_b > 0 else "",
                    "effective_containment_factor": (
                        float(np.nansum(expected_nominal[mask]) / np.nansum(expected_containment1[mask]))
                        if np.nansum(expected_containment1[mask]) > 0
                        else ""
                    ),
                }
            )
        for label in sorted(set(nhit), key=interval_key):
            nhit_mask = mask & (nhit == label)
            if not np.any(nhit_mask):
                continue
            for containment_mode, expected in [
                ("nominal_containment", expected_nominal),
                ("containment_1", expected_containment1),
            ]:
                total_excess = float(np.nansum(excess[nhit_mask]))
                total_expected = float(np.nansum(expected[nhit_mask]))
                total_b = float(np.nansum(b_on[nhit_mask]))
                total_n = float(np.nansum(n_on[nhit_mask]))
                rows.append(
                    {
                        "selector": selector,
                        "containment_mode": containment_mode,
                        "nhit_bin": label,
                        "cells": int(np.count_nonzero(nhit_mask)),
                        "N_on": total_n,
                        "B_on": total_b,
                        "excess": total_excess,
                        "official_expected_counts": total_expected,
                        "observed_over_expected": total_excess / total_expected if total_expected > 0 else "",
                        "N_on_over_B_on": total_n / total_b if total_b > 0 else "",
                        "effective_containment_factor": (
                            float(np.nansum(expected_nominal[nhit_mask]) / np.nansum(expected_containment1[nhit_mask]))
                            if np.nansum(expected_containment1[nhit_mask]) > 0
                            else ""
                        ),
                    }
                )
    return rows


def cell_rows(
    signal: dict[str, np.ndarray],
    expected_nominal: np.ndarray,
    expected_containment1: np.ndarray,
) -> list[dict[str, Any]]:
    cell_id = np.asarray(signal["cell_id"], dtype=np.int64)
    active = selector_mask(cell_id, ACTIVE30_SELECTOR)
    drop4 = selector_mask(cell_id, DROP4_SELECTOR)
    rows: list[dict[str, Any]] = []
    for idx, cid in enumerate(cell_id):
        nominal = float(expected_nominal[idx])
        containment1 = float(expected_containment1[idx])
        excess = float(np.asarray(signal["excess"], dtype=np.float64)[idx])
        b_on = float(np.asarray(signal["B_on"], dtype=np.float64)[idx])
        rows.append(
            {
                "cell_id": int(cid),
                "nhit_bin": str(np.asarray(signal["nhit_bin"]).astype(str)[idx]),
                "predE_bin": str(np.asarray(signal["predE_bin"]).astype(str)[idx]),
                "active30": int(bool(active[idx])),
                "drop4": int(bool(drop4[idx])),
                "N_on": float(np.asarray(signal["N_on"], dtype=np.float64)[idx]),
                "B_on": b_on,
                "excess": excess,
                "containment_r_opt": float(np.asarray(signal["containment_r_opt"], dtype=np.float64)[idx]),
                "official_expected_nominal": nominal,
                "official_expected_containment1": containment1,
                "ratio_nominal": excess / nominal if nominal > 0 else "",
                "ratio_containment1": excess / containment1 if containment1 > 0 else "",
                "required_delta_b_over_b_nominal": (excess - nominal) / b_on if b_on > 0 else "",
                "required_delta_b_over_b_containment1": (excess - containment1) / b_on if b_on > 0 else "",
            }
        )
    rows.sort(key=lambda row: (interval_key(str(row["nhit_bin"])), interval_key(str(row["predE_bin"])), int(row["cell_id"])))
    return rows


def stage_f_comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in [
        ("v4_drop4_nominal_containment", REPO_ROOT / "apply/output/stage_f_v4_drop4_annnorm/runs/v4_stage_f_annnorm_drop_cells_4_17_39_43/fit_v4_drop4_annnorm_metadata.json"),
        ("v4_drop4_containment_1", CONTAINMENT1_F_META),
    ]:
        meta = load_json(path)
        if not meta:
            continue
        preferred = meta.get("preferred_fit", {}) if isinstance(meta.get("preferred_fit"), dict) else {}
        key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
        fit = meta.get("fits", {}).get(key, {}) if isinstance(meta.get("fits"), dict) else {}
        params = fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {}
        validation = meta.get("validation", {}) if isinstance(meta.get("validation"), dict) else {}
        subset = validation.get("cell_subset", {}) if isinstance(validation.get("cell_subset"), dict) else {}
        rows.append(
            {
                "run": label,
                "run_id": meta.get("run_id", ""),
                "cells": subset.get("n_included_cells", ""),
                "preferred_model": preferred.get("model", ""),
                "error_mode": preferred.get("error_mode", ""),
                "phi0": params.get("phi0", ""),
                "gamma": params.get("gamma", ""),
                "alpha": params.get("alpha", ""),
                "beta": params.get("beta", ""),
                "chi2": fit.get("chi2", ""),
                "ndof": fit.get("ndof", ""),
                "chi2_over_ndof": fit.get("chi2_over_ndof", ""),
            }
        )
    return rows


def plot_selector_ratios(rows: list[dict[str, Any]], path: Path) -> None:
    plt = setup_matplotlib()
    selectors = ["all84", "active30", "drop4"]
    labels = sorted(
        {
            str(row["nhit_bin"])
            for row in rows
            if str(row.get("nhit_bin")) != "all" and row.get("selector") in selectors
        },
        key=interval_key,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160, sharey=True)
    for ax, mode, title in [
        (axes[0], "nominal_containment", "Current Stage F: response x containment_r_opt"),
        (axes[1], "containment_1", "Ablation: response x 1"),
    ]:
        for selector in selectors:
            vals = []
            for label in labels:
                row = next(
                    (
                        item
                        for item in rows
                        if item["selector"] == selector and item["containment_mode"] == mode and item["nhit_bin"] == label
                    ),
                    None,
                )
                vals.append(float(row["observed_over_expected"]) if row and row["observed_over_expected"] != "" else np.nan)
            ax.plot(labels, vals, marker="o", lw=1.5, label=selector)
        ax.axhline(1.0, color="#111827", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Stage E excess / official pass5 folded counts")
    axes[1].legend(fontsize=8)
    fig.suptitle("Containment ablation changes low-Nhit official forward-fold ratios")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_containment_factors(signal: dict[str, np.ndarray], path: Path) -> None:
    plt = setup_matplotlib()
    cell_id = np.asarray(signal["cell_id"], dtype=np.int64)
    nhit = np.asarray(signal["nhit_bin"]).astype(str)
    cont = np.asarray(signal["containment_r_opt"], dtype=np.float64)
    active = selector_mask(cell_id, ACTIVE30_SELECTOR)
    labels = sorted(set(nhit), key=interval_key)
    values = [cont[nhit == label] for label in labels]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.scatter(
        [labels.index(nhit[idx]) + 1 for idx in np.where(active)[0]],
        cont[active],
        s=20,
        color="#2563eb",
        alpha=0.75,
        label="active30 cells",
        zorder=3,
    )
    ax.axhline(1.0, color="#111827", lw=0.8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("containment_r_opt")
    ax.set_title("Stage B containment factors carried into Stage F/G")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sed_comparison(path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=160)
    e_pass5, y_pass5 = v3.pass5_points()
    if e_pass5:
        ax.plot(e_pass5, y_pass5, "o", ms=5.5, color="#111827", label="Official pass5 WCDA")
    for meta_path, color, marker, label_prefix in [
        (DROP4_G_META, "#2563eb", "o", "v4 drop4 nominal"),
        (CONTAINMENT1_G_META, "#dc2626", "s", "v4 drop4 containment=1"),
    ]:
        meta = load_json(meta_path)
        if not meta:
            continue
        for grouping, alpha in [("nhit", 0.95), ("predE", 0.72)]:
            energy, flux, err = v3.point_arrays(meta, grouping)
            if energy:
                ax.errorbar(
                    energy,
                    flux,
                    yerr=err,
                    fmt=marker if grouping == "nhit" else "D",
                    ms=5.0,
                    lw=1.0,
                    color=color,
                    ecolor=color,
                    alpha=alpha,
                    markerfacecolor="none" if grouping == "predE" else color,
                    capsize=2.3,
                    label=f"{label_prefix} {grouping}",
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("SED containment ablation: nominal v4 versus containment=1")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    response = load_npz(RESPONSE_NPZ)
    signal = load_npz(SIGNAL_NPZ)
    active30 = load_npz(ACTIVE30_NPZ)
    theta_exposure = np.asarray(active30["theta_exposure_sec"], dtype=np.float64)
    expected_nominal, expected_containment1 = expected_counts(response, signal, theta_exposure)

    summary_rows = selector_summary_rows(signal, expected_nominal, expected_containment1)
    write_csv(
        OUT_DIR / "official_pass5_containment_ablation_by_selector_nhit.csv",
        summary_rows,
        [
            "selector",
            "containment_mode",
            "nhit_bin",
            "cells",
            "N_on",
            "B_on",
            "excess",
            "official_expected_counts",
            "observed_over_expected",
            "N_on_over_B_on",
            "effective_containment_factor",
        ],
    )

    per_cell = cell_rows(signal, expected_nominal, expected_containment1)
    write_csv(
        OUT_DIR / "official_pass5_containment_ablation_by_cell.csv",
        per_cell,
        [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "active30",
            "drop4",
            "N_on",
            "B_on",
            "excess",
            "containment_r_opt",
            "official_expected_nominal",
            "official_expected_containment1",
            "ratio_nominal",
            "ratio_containment1",
            "required_delta_b_over_b_nominal",
            "required_delta_b_over_b_containment1",
        ],
    )

    fit_rows = stage_f_comparison_rows()
    write_csv(
        OUT_DIR / "stage_f_nominal_vs_containment1_summary.csv",
        fit_rows,
        [
            "run",
            "run_id",
            "cells",
            "preferred_model",
            "error_mode",
            "phi0",
            "gamma",
            "alpha",
            "beta",
            "chi2",
            "ndof",
            "chi2_over_ndof",
        ],
    )

    plot_selector_ratios(summary_rows, OUT_DIR / "official_pass5_containment_ablation_by_nhit.png")
    plot_containment_factors(signal, OUT_DIR / "containment_r_opt_by_nhit.png")
    plot_sed_comparison(OUT_DIR / "v4_sed_nominal_vs_containment1.png")

    drop4_nominal_all = next(
        row
        for row in summary_rows
        if row["selector"] == "drop4" and row["containment_mode"] == "nominal_containment" and row["nhit_bin"] == "all"
    )
    drop4_cont1_all = next(
        row for row in summary_rows if row["selector"] == "drop4" and row["containment_mode"] == "containment_1" and row["nhit_bin"] == "all"
    )
    response_meta = load_json(RESPONSE_META)
    write_json(
        OUT_DIR / "v4_response_audit_summary.json",
        {
            "summary": {
                "dominant_current_hypothesis": "Stage F/G currently multiply by containment_r_opt even though the response numerator is built from a source-region MC cache; this likely underpredicts expected counts and drives high SED normalization.",
                "drop4_nominal_official_obs_over_expected": drop4_nominal_all["observed_over_expected"],
                "drop4_containment1_official_obs_over_expected": drop4_cont1_all["observed_over_expected"],
                "drop4_nominal_effective_containment_factor": drop4_nominal_all["effective_containment_factor"],
                "stage_a_response_type": response_meta.get("response_type"),
                "stage_a_effective_area_formula": response_meta.get("effective_area_formula"),
                "stage_a_upstream_cache": response_meta.get("binned_root"),
                "stage_a_metadata_mc_dangle_cut": response_meta.get("cuts", {}).get("mc_dangle_cut")
                if isinstance(response_meta.get("cuts"), dict)
                else None,
                "caveat": "The upstream binned cache does not preserve full CLI provenance in its run_summary.json. The next definitive test is to rebuild Stage A from an uncut all-direction numerator cache or explicitly encode aperture conditioning in the response.",
            },
            "outputs": {
                "selector_nhit_csv": str(OUT_DIR / "official_pass5_containment_ablation_by_selector_nhit.csv"),
                "cell_csv": str(OUT_DIR / "official_pass5_containment_ablation_by_cell.csv"),
                "stage_f_summary_csv": str(OUT_DIR / "stage_f_nominal_vs_containment1_summary.csv"),
                "ratio_plot": str(OUT_DIR / "official_pass5_containment_ablation_by_nhit.png"),
                "containment_plot": str(OUT_DIR / "containment_r_opt_by_nhit.png"),
                "sed_plot": str(OUT_DIR / "v4_sed_nominal_vs_containment1.png"),
            },
        },
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
