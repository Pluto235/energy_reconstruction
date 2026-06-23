#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import build_v3_latest_bkg_report as v3
import build_v4_empirical_psf_diagnostics as empirical_psf


REPO_ROOT = v3.REPO_ROOT
REPORT_DIR = v3.REPORT_DIR
REPORT_PATH = REPORT_DIR / "crab_sed_v4_stage_a_to_g_report.html"
V4_ASSET_DIR = REPORT_DIR / "assets" / "v4-annnorm"
V4_FINAL_SED_PNG = V4_ASSET_DIR / "v4_annnorm_final_sed_with_official_and_old_v3.png"
ACTIVE30_FORWARD_DIR = REPORT_DIR / "assets" / "v4-annnorm-normalization-diagnostics"
DROP4_FORWARD_DIR = REPORT_DIR / "assets" / "v4-drop4-normalization-diagnostics"
FORWARD_DIR = DROP4_FORWARD_DIR
DROP4_SELECTOR_CSV = REPO_ROOT / "apply/config/cell_selector_v4_drop4_psfborrow.csv"
PRIMARY_STAGE_A_META = (
    REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned_metadata.json"
)
PRIMARY_STAGE_E_DIR = (
    REPO_ROOT
    / "apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow"
)
PRIMARY_STAGE_E_META = PRIMARY_STAGE_E_DIR / "signal_v4_containment1_annnorm_metadata.json"
DROP4_STAGE_F_DIR = (
    REPO_ROOT
    / "apply/output/stage_f_v4_drop4_annnorm/runs/v4_stage_f_annnorm_drop_cells_4_17_39_43"
)
DROP4_STAGE_G_DIR = (
    REPO_ROOT
    / "apply/output/stage_g_v4_drop4_annnorm/runs/v4_stage_g_annnorm_drop_cells_4_17_39_43"
)
DROP4_STAGE_F_META = DROP4_STAGE_F_DIR / "fit_v4_drop4_annnorm_metadata.json"
DROP4_STAGE_G_META = DROP4_STAGE_G_DIR / "sed_points_v4_drop4_annnorm_metadata.json"
ROOT_CAUSE_DIR = REPORT_DIR / "assets" / "v4-root-cause-diagnostics"
V4_PSF_RADIAL_PROFILE_GRID_PNG = V4_ASSET_DIR / "v4_stage_b_candidate_radial_psf_profiles_fit_highlight.png"
RESPONSE_AUDIT_DIR = REPORT_DIR / "assets" / "v4-response-audit"
V4_RESPONSE_CONTRACT_OVERLAY_PNG = RESPONSE_AUDIT_DIR / "v4_response_contract_stage_g_external_overlay.png"
CONTAINMENT1_STAGE_G_DIR = (
    REPO_ROOT / "apply/output/stage_g_v4_containment1_drop4_annnorm/runs/v4_stage_g_annnorm_containment1_drop4"
)
APERTURE_CONDITIONED_STAGE_G_DIR = (
    REPO_ROOT / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4"
)
APERTURE_CONDITIONED_STAGE_F_DIR = (
    REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4"
)
APERTURE_CONDITIONED_STAGE_F_META = (
    APERTURE_CONDITIONED_STAGE_F_DIR / "fit_v4_aperture_conditioned_drop4_metadata.json"
)
APERTURE_CONDITIONED_STAGE_G_META = (
    APERTURE_CONDITIONED_STAGE_G_DIR / "sed_points_v4_aperture_conditioned_drop4_metadata.json"
)
PRIMARY_STAGE_F_DIR = APERTURE_CONDITIONED_STAGE_F_DIR
PRIMARY_STAGE_G_DIR = APERTURE_CONDITIONED_STAGE_G_DIR
PRIMARY_STAGE_F_META = APERTURE_CONDITIONED_STAGE_F_META
PRIMARY_STAGE_G_META = APERTURE_CONDITIONED_STAGE_G_META
R68_STAGE_B_DIR = REPO_ROOT / "apply/output/stage_b_v4_aperture_variants/runs/v4_r68_from_psfborrow"
R68_STAGE_D_DIR = REPO_ROOT / "apply/output/stage_d_v4_r68_aperture/runs/v4_r68_aperture_drop4_stage_d"
R68_STAGE_E_DIR = REPO_ROOT / "apply/output/stage_e_v4_r68_aperture/runs/v4_r68_aperture_drop4_stage_e"
R68_STAGE_F_DIR = REPO_ROOT / "apply/output/stage_f_v4_r68_aperture/runs/v4_r68_aperture_drop4_stage_f"
R68_STAGE_G_DIR = REPO_ROOT / "apply/output/stage_g_v4_r68_aperture/runs/v4_r68_aperture_drop4_stage_g"
R68_STAGE_B_META = R68_STAGE_B_DIR / "psf_v4_r68_aperture_metadata.json"
R68_STAGE_D_META = R68_STAGE_D_DIR / "background_v4_r68_aperture_metadata.json"
R68_STAGE_E_META = R68_STAGE_E_DIR / "signal_v4_r68_aperture_metadata.json"
R68_STAGE_F_META = R68_STAGE_F_DIR / "fit_v4_r68_aperture_metadata.json"
R68_STAGE_G_META = R68_STAGE_G_DIR / "sed_points_v4_r68_aperture_metadata.json"
R68_SED_COMPARE_PNG = V4_ASSET_DIR / "v4_r68_aperture_sed_comparison.png"
EMPIRICAL_PSF_DIR = REPORT_DIR / "assets" / "v4-empirical-psf"
EMPIRICAL_PSF_SUMMARY_JSON = EMPIRICAL_PSF_DIR / "empirical_psf_summary.json"
EMPIRICAL_PSF_CELL_CSV = EMPIRICAL_PSF_DIR / "empirical_psf_cell_summary.csv"
EMPIRICAL_PSF_GROUP_CSV = EMPIRICAL_PSF_DIR / "empirical_psf_nhit_group_summary.csv"
CELL_ROOT_CAUSE_CROSSMATCH_CSV = ROOT_CAUSE_DIR / "v4_drop4_cell_root_cause_crossmatch.csv"
CELL_ROOT_CAUSE_PNG = ROOT_CAUSE_DIR / "v4_drop4_cell_root_cause_crossmatch.png"


def to_float(value: Any) -> float | None:
    return v3.finite_float(value)


def ratio_cell_rows(rows: list[dict[str, str]], *, max_rows: int = 14) -> list[dict[str, str]]:
    official = [row for row in rows if row.get("spectrum") == "official_pass5"]
    low_nhit = {"[125,200)", "[200,300)", "[300,500)", "[500,800)"}
    selected = [row for row in official if row.get("nhit_bin") in low_nhit]

    def residual(row: dict[str, str]) -> float:
        return (to_float(row.get("excess_minus_expected")) or -1.0e99)

    selected.sort(key=residual, reverse=True)
    return selected[:max_rows]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else sum(finite) / len(finite)


def by_cell(rows: list[dict[str, str]], *, selector: str | None = None) -> dict[int, dict[str, str]]:
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        if selector is not None and row.get("selector") != selector:
            continue
        cell_id = v3.finite_float(row.get("cell_id"))
        if cell_id is None:
            continue
        indexed[int(cell_id)] = row
    return indexed


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_cell_root_cause_crossmatch() -> list[dict[str, Any]]:
    """Merge current drop4 cell diagnostics into one lightweight table."""
    fit_rows = v3.read_csv_rows(PRIMARY_STAGE_F_DIR / "fit_v4_aperture_conditioned_drop4_summary.csv")
    fold_rows = [
        row
        for row in v3.read_csv_rows(RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_cell.csv")
        if truthy(row.get("drop4"))
    ]
    stage_a_rows = v3.read_csv_rows(REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/cell_response_summary.csv")
    psf_rows = v3.read_csv_rows(
        REPO_ROOT
        / "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate_summary.csv"
    )
    empirical_rows = v3.read_csv_rows(EMPIRICAL_PSF_CELL_CSV) if EMPIRICAL_PSF_CELL_CSV.exists() else []
    energy_rows = [
        row
        for row in v3.read_csv_rows(ROOT_CAUSE_DIR / "official_pass5_true_energy_contribution_by_cell.csv")
        if row.get("selector") == "drop4"
    ]
    offsource_rows = [
        row
        for row in v3.read_csv_rows(ROOT_CAUSE_DIR / "offsource_core_residual_cells.csv")
        if row.get("selector") == "drop4"
    ]

    fold_by = by_cell(fold_rows)
    stage_a_by = by_cell(stage_a_rows)
    psf_by = by_cell(psf_rows)
    empirical_by = by_cell(empirical_rows)
    energy_by = by_cell(energy_rows)
    offsource_by: dict[int, dict[str, Any]] = {}
    for row in offsource_rows:
        cell_id = v3.finite_float(row.get("cell_id"))
        if cell_id is None:
            continue
        bucket = offsource_by.setdefault(int(cell_id), {"sigmas": [], "excesses": []})
        sigma = v3.finite_float(row.get("excess_over_sqrt_N_plus_B"))
        excess = v3.finite_float(row.get("excess"))
        if sigma is not None:
            bucket["sigmas"].append(sigma)
        if excess is not None:
            bucket["excesses"].append(excess)

    rows: list[dict[str, Any]] = []
    for fit in fit_rows:
        cell_id_number = v3.finite_float(fit.get("cell_id"))
        if cell_id_number is None:
            continue
        cell_id = int(cell_id_number)
        fold = fold_by.get(cell_id, {})
        stage_a = stage_a_by.get(cell_id, {})
        psf = psf_by.get(cell_id, {})
        empirical = empirical_by.get(cell_id, {})
        energy = energy_by.get(cell_id, {})
        off = offsource_by.get(cell_id, {"sigmas": [], "excesses": []})

        excess = v3.finite_float(fit.get("excess"))
        logpar_model = v3.finite_float(fit.get("logpar_model"))
        err = v3.finite_float(fit.get("error_conservative"))
        official_expected = v3.finite_float(fold.get("official_expected_aperture_response"))
        official_delta = None if excess is None or official_expected is None else excess - official_expected
        official_pull = None if official_delta is None or err is None or err <= 0 else official_delta / err
        stagef_ratio = None if excess is None or logpar_model is None or logpar_model <= 0 else excess / logpar_model
        sigmas = list(off.get("sigmas", []))
        excesses = list(off.get("excesses", []))

        rows.append(
            {
                "cell_id": cell_id,
                "nhit_bin": fit.get("nhit_bin"),
                "predE_bin": fit.get("predE_bin"),
                "N_on": fit.get("N_on"),
                "B_on": fit.get("B_on"),
                "excess": fit.get("excess"),
                "error_conservative": fit.get("error_conservative"),
                "logpar_model": fit.get("logpar_model"),
                "logpar_pull": fit.get("logpar_pull"),
                "stagef_model_ratio": stagef_ratio,
                "official_expected_aperture_response": official_expected,
                "ratio_aperture_response": fold.get("ratio_aperture_response"),
                "fit_minus_official_pass5_aperture_counts": official_delta,
                "fit_vs_official_pass5_aperture_pull": official_pull,
                "required_delta_b_over_b_aperture_response": fold.get(
                    "required_delta_b_over_b_aperture_response"
                ),
                "offsource_mean_sigma": mean(sigmas),
                "offsource_min_sigma": min(sigmas) if sigmas else None,
                "offsource_max_sigma": max(sigmas) if sigmas else None,
                "offsource_mean_excess": mean(excesses),
                "events": stage_a.get("events"),
                "aperture_kept_fraction": stage_a.get("aperture_kept_fraction"),
                "truth_range_events": stage_a.get("truth_range_events"),
                "effective_events": psf.get("effective_events"),
                "core_fit_effective_events": psf.get("core_fit_effective_events"),
                "theta_missing_crab_probability_mass": psf.get("theta_missing_crab_probability_mass"),
                "sigma_deg": psf.get("sigma_deg"),
                "r_opt_deg": psf.get("r_opt_deg"),
                "stageb_containment_r_opt": psf.get("containment_r_opt"),
                "r68_deg": psf.get("r68_deg"),
                "r90_deg": psf.get("r90_deg"),
                "containment_warning": psf.get("containment_warning"),
                "psf_borrowed": psf.get("psf_borrowed"),
                "empirical_significance": empirical.get("significance"),
                "fit_reliable": empirical.get("fit_reliable"),
                "sigma_obs_over_mc": empirical.get("sigma_obs_over_mc"),
                "r68_obs_over_mc": empirical.get("r68_obs_over_mc"),
                "profile_residual_rms": empirical.get("profile_residual_rms"),
                "true_e50_tev": energy.get("true_e50_tev"),
                "frac_below_pass5_min": energy.get("frac_below_pass5_min"),
                "frac_below_1tev": energy.get("frac_below_1tev"),
                "frac_above_10tev": energy.get("frac_above_10tev"),
            }
        )

    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "N_on",
        "B_on",
        "excess",
        "error_conservative",
        "logpar_model",
        "logpar_pull",
        "stagef_model_ratio",
        "official_expected_aperture_response",
        "ratio_aperture_response",
        "fit_minus_official_pass5_aperture_counts",
        "fit_vs_official_pass5_aperture_pull",
        "required_delta_b_over_b_aperture_response",
        "offsource_mean_sigma",
        "offsource_min_sigma",
        "offsource_max_sigma",
        "offsource_mean_excess",
        "events",
        "aperture_kept_fraction",
        "truth_range_events",
        "effective_events",
        "core_fit_effective_events",
        "theta_missing_crab_probability_mass",
        "sigma_deg",
        "r_opt_deg",
        "stageb_containment_r_opt",
        "r68_deg",
        "r90_deg",
        "containment_warning",
        "psf_borrowed",
        "empirical_significance",
        "fit_reliable",
        "sigma_obs_over_mc",
        "r68_obs_over_mc",
        "profile_residual_rms",
        "true_e50_tev",
        "frac_below_pass5_min",
        "frac_below_1tev",
        "frac_above_10tev",
    ]
    write_csv(CELL_ROOT_CAUSE_CROSSMATCH_CSV, rows, fieldnames)
    plot_cell_root_cause_crossmatch(rows)
    return rows


def plot_cell_root_cause_crossmatch(rows: list[dict[str, Any]]) -> Path:
    try:
        plt = v3.setup_matplotlib()
        import numpy as np
    except ModuleNotFoundError:
        if v3.exists(CELL_ROOT_CAUSE_PNG):
            return CELL_ROOT_CAUSE_PNG
        raise

    if not rows:
        return CELL_ROOT_CAUSE_PNG

    ordered = sorted(rows, key=lambda row: int(row["cell_id"]))
    x = np.arange(len(ordered))
    labels = [str(row["cell_id"]) for row in ordered]
    logpar_pull = np.asarray([v3.finite_float(row.get("logpar_pull")) or 0.0 for row in ordered], dtype=float)
    ratio = np.asarray([v3.finite_float(row.get("ratio_aperture_response")) or np.nan for row in ordered], dtype=float)
    db_over_b = np.asarray(
        [100.0 * (v3.finite_float(row.get("required_delta_b_over_b_aperture_response")) or 0.0) for row in ordered],
        dtype=float,
    )
    sigma_obs_mc = np.asarray([v3.finite_float(row.get("sigma_obs_over_mc")) or np.nan for row in ordered], dtype=float)
    off_sigma = np.asarray([v3.finite_float(row.get("offsource_mean_sigma")) or np.nan for row in ordered], dtype=float)

    colors = ["#dc2626" if value > 0 else "#2563eb" for value in logpar_pull]
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 9.2), dpi=160, sharex=True)

    axes[0].bar(x, logpar_pull, color=colors, alpha=0.82)
    axes[0].axhline(0.0, color="#111827", lw=0.8)
    for y in [-3, -2, 2, 3]:
        axes[0].axhline(y, color="#9ca3af", lw=0.7, ls="--", alpha=0.7)
    axes[0].set_ylabel("Stage F pull")
    axes[0].set_title("v4 drop4 cell-level root-cause crossmatch")

    axes[1].plot(x, ratio, "o-", color="#7c3aed", lw=1.3, ms=4.4)
    axes[1].axhline(1.0, color="#111827", lw=0.8)
    axes[1].axhline(1.5, color="#9ca3af", lw=0.7, ls="--", alpha=0.7)
    axes[1].set_ylabel("obs / official")
    axes[1].set_ylim(bottom=min(0.0, float(np.nanmin(ratio)) if np.isfinite(ratio).any() else 0.0))

    axes[2].bar(x, db_over_b, color="#f97316", alpha=0.75)
    axes[2].axhline(0.0, color="#111827", lw=0.8)
    axes[2].set_ylabel("dB/B (%)")

    axes[3].plot(x, sigma_obs_mc, "s", color="#059669", ms=4.2, label="empirical sigma / MC sigma")
    axes[3].plot(x, off_sigma, "x", color="#6b7280", ms=4.2, label="mean off-source sigma")
    axes[3].axhline(1.0, color="#059669", lw=0.8, ls="--", alpha=0.7)
    axes[3].axhline(0.0, color="#6b7280", lw=0.8, ls="--", alpha=0.7)
    axes[3].set_ylabel("PSF / offsrc")
    axes[3].legend(fontsize=7.5, loc="best")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=8)
    axes[-1].set_xlabel("cell id")

    fig.tight_layout()
    CELL_ROOT_CAUSE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CELL_ROOT_CAUSE_PNG)
    plt.close(fig)
    return CELL_ROOT_CAUSE_PNG


def v4_summary_cards(e_meta: dict[str, Any], f_meta: dict[str, Any], g_meta: dict[str, Any], d_meta: dict[str, Any]) -> str:
    totals = e_meta.get("totals", {}) if isinstance(e_meta.get("totals"), dict) else {}
    preferred = f_meta.get("preferred_fit", {}) if isinstance(f_meta.get("preferred_fit"), dict) else {}
    key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
    fit = f_meta.get("fits", {}).get(key, {}) if isinstance(f_meta.get("fits"), dict) else {}
    params = fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {}
    validation = g_meta.get("validation", {}) if isinstance(g_meta.get("validation"), dict) else {}
    cells = validation.get("required_cell_ids") or validation.get("stage_f_subset_included") or []
    return (
        '<div class="cards">'
        f'<div class="card"><div class="k">latest background</div><div class="v">B_on {v3.fmt(totals.get("B_on"), 6)}</div><p>N_on {v3.fmt_int(totals.get("N_on"))}; excess {v3.fmt(totals.get("excess"), 6)}</p></div>'
        f'<div class="card"><div class="k">detection diagnostic</div><div class="v">{v3.fmt(totals.get("known_b_sigma_aggregate"), 4)} sigma</div><p>known-background Poisson aggregate; Li-Ma is not defined for direct expectation bkg.</p></div>'
        f'<div class="card"><div class="k">Stage F preferred fit</div><div class="v">{v3.esc(preferred.get("model"))}</div><p>phi0 {v3.fmt(params.get("phi0"), 5)}, alpha/gamma {v3.fmt(params.get("alpha", params.get("gamma")), 5)}, beta {v3.fmt(params.get("beta"), 5)}; chi2/ndof {v3.fmt(fit.get("chi2"), 4)}/{v3.fmt(fit.get("ndof"), 3)}</p></div>'
        f'<div class="card"><div class="k">fit cells / SED points</div><div class="v">{len(cells)} cells / {len(g_meta.get("points", []))} points</div><p>Stage D active-fit warnings: {len(d_meta.get("quality", {}).get("active_fit_warning_cell_ids", [])) if isinstance(d_meta.get("quality"), dict) else "n/a"}</p></div>'
        "</div>"
    )


def v4_stage_table(
    a_meta: dict[str, Any],
    b_meta: dict[str, Any],
    c_meta: dict[str, Any],
    d_meta: dict[str, Any],
    e_meta: dict[str, Any],
    f_meta: dict[str, Any],
    g_meta: dict[str, Any],
) -> str:
    rows = [
        [
            "A",
            "aperture-conditioned 2D response",
            v3.esc(a_meta.get("response_type")),
            v3.esc(PRIMARY_STAGE_A_META.relative_to(REPO_ROOT)),
        ],
        ["B", "direct own-cell PSF for r_opt", v3.esc(b_meta.get("run_id")), v3.esc(v3.STAGE_B_META.relative_to(REPO_ROOT))],
        ["C", "observation event reduction", v3.esc(c_meta.get("run_id")), v3.esc(v3.STAGE_C_META.relative_to(REPO_ROOT))],
        ["D", "annulus-normalized 2D background", v3.esc(d_meta.get("run_id")), v3.esc(v3.STAGE_D_META.relative_to(REPO_ROOT))],
        [
            "E",
            "on-region excess; containment forced to one for response contract",
            v3.esc(e_meta.get("run_id")),
            v3.esc(PRIMARY_STAGE_E_META.relative_to(REPO_ROOT)),
        ],
        [
            "F",
            "global forward-folding fit",
            v3.esc(f_meta.get("run_id")),
            v3.esc(PRIMARY_STAGE_F_META.relative_to(REPO_ROOT)),
        ],
        [
            "G",
            "diagnostic SED points",
            v3.esc(g_meta.get("run_id")),
            v3.esc(PRIMARY_STAGE_G_META.relative_to(REPO_ROOT)),
        ],
    ]
    return v3.table(["Stage", "current role", "run / type", "metadata"], rows, cls="compact")


def plot_v4_final_sed(current_meta: dict[str, Any], old_meta: dict[str, Any]) -> Path:
    try:
        plt = v3.setup_matplotlib()
    except ModuleNotFoundError:
        if v3.exists(V4_FINAL_SED_PNG):
            return V4_FINAL_SED_PNG
        raise
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)

    e_pass5, y_pass5 = v3.pass5_points()
    if e_pass5:
        ax.plot(e_pass5, y_pass5, "o", ms=5.5, color="#111827", label="Official pass5 WCDA")

    e_v099, y_v099, ylo_v099, yhi_v099 = v3.v099_points()
    if e_v099:
        ax.errorbar(
            e_v099,
            y_v099,
            yerr=[ylo_v099, yhi_v099],
            fmt="s",
            ms=5.0,
            lw=1.0,
            color="#7c2d12",
            ecolor="#7c2d12",
            capsize=2.5,
            label="Tutorial v0.99 WCDA",
        )

    for grouping, marker, color, label in [
        ("nhit", "o", "#2563eb", "v4 aperture-response Nhit points"),
        ("predE", "D", "#059669", "v4 aperture-response predE points"),
    ]:
        energy, flux, err = v3.point_arrays(current_meta, grouping)
        if energy:
            ax.errorbar(
                energy,
                flux,
                yerr=err,
                fmt=marker,
                ms=5.2,
                lw=1.0,
                color=color,
                ecolor=color,
                capsize=2.4,
                label=label,
                zorder=5,
            )

    for grouping, marker, label in [
        ("nhit", "o", "old v3 psfborrow Nhit points"),
        ("predE", "D", "old v3 psfborrow predE points"),
    ]:
        energy, flux, err = v3.point_arrays(old_meta, grouping)
        if energy:
            ax.errorbar(
                energy,
                flux,
                yerr=err,
                fmt=marker,
                ms=4.6,
                lw=0.8,
                color="#9ca3af",
                ecolor="#c4c7cc",
                markerfacecolor="none",
                markeredgewidth=1.0,
                alpha=0.72,
                capsize=2.0,
                label=label,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Crab SED v4: aperture-conditioned response result")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.2, ncol=1)
    fig.tight_layout()
    V4_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(V4_FINAL_SED_PNG)
    plt.close(fig)
    return V4_FINAL_SED_PNG


def e2_curve_for_spectrum(energy_tev: Any, spectrum: dict[str, Any]) -> Any:
    import numpy as np

    energy = np.asarray(energy_tev, dtype=float)
    pivot = float(spectrum.get("pivot_tev") or 3.0)
    phi0 = float(spectrum["phi0"])
    model = str(spectrum.get("model") or "logpar").lower()
    ratio = energy / pivot
    if model == "pl":
        gamma = float(spectrum["gamma"])
        dnde = phi0 * np.power(ratio, -gamma)
    elif model == "logpar":
        alpha = float(spectrum["alpha"])
        beta = float(spectrum["beta"])
        log_ratio = np.log(ratio)
        dnde = phi0 * np.exp((-alpha - beta * log_ratio) * log_ratio)
    else:
        raise ValueError(f"Unsupported spectrum model for plotting: {model}")
    return energy * energy * dnde


def fit_logpar_to_e2_points(label: str, energy: list[float], e2_flux: list[float], *, pivot_tev: float = 3.0) -> dict[str, Any] | None:
    import numpy as np

    e = np.asarray(energy, dtype=float)
    e2 = np.asarray(e2_flux, dtype=float)
    valid = np.isfinite(e) & np.isfinite(e2) & (e > 0.0) & (e2 > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    x = np.log(e[valid] / float(pivot_tev))
    y = np.log(e2[valid] / (e[valid] * e[valid]))
    c2, c1, c0 = np.polyfit(x, y, 2)
    return {
        "label": label,
        "model": "logpar",
        "phi0": float(np.exp(c0)),
        "alpha": float(-c1),
        "beta": float(-c2),
        "pivot_tev": float(pivot_tev),
        "n_points": int(np.count_nonzero(valid)),
        "fit_note": "unweighted log-space fit to plotted SED points",
    }


def v4_frozen_spectrum(g_meta: dict[str, Any], f_meta: dict[str, Any]) -> dict[str, Any]:
    frozen = g_meta.get("frozen_spectrum") if isinstance(g_meta.get("frozen_spectrum"), dict) else {}
    if frozen:
        return dict(frozen)
    preferred = f_meta.get("preferred_fit", {}) if isinstance(f_meta.get("preferred_fit"), dict) else {}
    key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
    fit = f_meta.get("fits", {}).get(key, {}) if isinstance(f_meta.get("fits"), dict) else {}
    params = dict(fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {})
    params["model"] = preferred.get("model")
    params["pivot_tev"] = 3.0
    params["stage_f_chi2"] = fit.get("chi2")
    params["stage_f_ndof"] = fit.get("ndof")
    return params


def spectrum_param_line(label: str, spectrum: dict[str, Any], *, include_chi2: bool = False) -> str:
    model = str(spectrum.get("model") or "logpar").lower()
    pivot = float(spectrum.get("pivot_tev") or 3.0)
    phi0 = float(spectrum["phi0"])
    if model == "pl":
        line = f"{label}: PL phi0={phi0:.3e}, gamma={float(spectrum['gamma']):.3f}, E0={pivot:g} TeV"
    else:
        line = (
            f"{label}: LogPar phi0={phi0:.3e}, alpha={float(spectrum['alpha']):.3f}, "
            f"beta={float(spectrum['beta']):.3f}, E0={pivot:g} TeV"
        )
    if include_chi2 and spectrum.get("stage_f_chi2") is not None and spectrum.get("stage_f_ndof") is not None:
        line += f", chi2/ndof={float(spectrum['stage_f_chi2']):.1f}/{int(spectrum['stage_f_ndof'])}"
    return line


def plot_response_contract_external_overlay(g_meta: dict[str, Any], f_meta: dict[str, Any]) -> Path:
    try:
        plt = v3.setup_matplotlib()
        import numpy as np
    except ModuleNotFoundError:
        if v3.exists(V4_RESPONSE_CONTRACT_OVERLAY_PNG):
            return V4_RESPONSE_CONTRACT_OVERLAY_PNG
        raise

    fig, ax = plt.subplots(figsize=(10.4, 6.6), dpi=170)

    e_pass5, y_pass5 = v3.pass5_points()
    pass5_fit = fit_logpar_to_e2_points("official pass5", e_pass5, y_pass5)
    if e_pass5:
        ax.plot(e_pass5, y_pass5, "o", ms=5.4, color="#111827", label="official pass5 WCDA points", zorder=7)

    e_v099, y_v099, ylo_v099, yhi_v099 = v3.v099_points()
    v099_fit = fit_logpar_to_e2_points("tutorial v0.99", e_v099, y_v099)
    if e_v099:
        ax.errorbar(
            e_v099,
            y_v099,
            yerr=[ylo_v099, yhi_v099],
            fmt="s",
            ms=5.0,
            lw=0.9,
            color="#9a3412",
            ecolor="#9a3412",
            capsize=2.4,
            label="tutorial v0.99 WCDA points",
            zorder=7,
        )

    external_points = (
        g_meta.get("external_crab_sed_references", {}).get("points", [])
        if isinstance(g_meta.get("external_crab_sed_references"), dict)
        else []
    )
    external_styles = {
        "magic_joint_crab": {"fmt": "v", "color": "#7c3aed", "label": "MAGIC"},
        "hess_2024_stereo": {"fmt": "D", "color": "#db2777", "label": "H.E.S.S."},
    }
    for dataset, style in external_styles.items():
        selected = [
            point
            for point in external_points
            if isinstance(point, dict)
            and str(point.get("dataset")) == dataset
            and not bool(point.get("is_upper_limit"))
            and (to_float(point.get("energy_tev")) or 0.0) > 0.0
            and (to_float(point.get("e2_dnde")) or 0.0) > 0.0
        ]
        if not selected:
            continue
        ax.errorbar(
            [float(point["energy_tev"]) for point in selected],
            [float(point["e2_dnde"]) for point in selected],
            yerr=[float(point.get("e2_dnde_err") or 0.0) for point in selected],
            ms=4.0,
            lw=0.65,
            capsize=1.8,
            alpha=0.68,
            zorder=3,
            **style,
        )

    v4_spectrum = v4_frozen_spectrum(g_meta, f_meta)
    curves = [fit for fit in [pass5_fit, v099_fit, v4_spectrum] if fit]
    all_energies: list[float] = []
    for values in [e_pass5, e_v099]:
        all_energies.extend(values)
    for grouping in ["nhit", "predE"]:
        energy, _, _ = v3.point_arrays(g_meta, grouping)
        all_energies.extend(energy)
    if external_points:
        all_energies.extend(
            [
                float(point["energy_tev"])
                for point in external_points
                if isinstance(point, dict)
                and str(point.get("dataset")) in external_styles
                and not bool(point.get("is_upper_limit"))
                and (to_float(point.get("energy_tev")) or 0.0) > 0.0
            ]
        )
    emin = max(0.1, min(all_energies or [0.3]) / 1.35)
    emax = min(250.0, max(all_energies or [120.0]) * 1.35)
    x = np.geomspace(emin, emax, 320)

    if pass5_fit:
        ax.plot(x, e2_curve_for_spectrum(x, pass5_fit), color="#111827", lw=1.7, ls="-", label="official pass5 point-fit LogPar")
    if v099_fit:
        ax.plot(x, e2_curve_for_spectrum(x, v099_fit), color="#9a3412", lw=1.7, ls="--", label="tutorial v0.99 point-fit LogPar")
    if v4_spectrum:
        ax.plot(x, e2_curve_for_spectrum(x, v4_spectrum), color="#2563eb", lw=2.2, label="v4 primary Stage F LogPar")

    for grouping, marker, color, label in [
        ("nhit", "o", "#2563eb", "v4 primary Stage G Nhit points"),
        ("predE", "D", "#059669", "v4 primary Stage G predE points"),
    ]:
        energy, flux, err = v3.point_arrays(g_meta, grouping)
        if not energy:
            continue
        ax.errorbar(
            energy,
            flux,
            yerr=err,
            fmt=marker,
            ms=5.2,
            lw=1.0,
            color=color,
            ecolor=color,
            capsize=2.4,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
            zorder=8,
        )

    note_lines = [
        r"$dN/dE=\phi_0(E/E_0)^{-\alpha-\beta\ln(E/E_0)}$",
        spectrum_param_line("v4 primary", v4_spectrum, include_chi2=True),
    ]
    if pass5_fit:
        note_lines.append(spectrum_param_line("pass5 point-fit", pass5_fit))
    if v099_fit:
        note_lines.append(spectrum_param_line("v0.99 point-fit", v099_fit))
    note_lines.append("pass5/v0.99 curves: unweighted log-space fits to plotted points")
    ax.text(
        0.035,
        0.045,
        "\n".join(note_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d1d5db", "alpha": 0.88},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(emin, emax)
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Native Stage G response-contract overlay")
    ax.grid(True, which="both", alpha=0.23, lw=0.45)
    ax.legend(fontsize=7.1, ncol=2, frameon=True, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    RESPONSE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(V4_RESPONSE_CONTRACT_OVERLAY_PNG)
    plt.close(fig)
    return V4_RESPONSE_CONTRACT_OVERLAY_PNG


def plot_r68_sed_comparison(nominal_meta: dict[str, Any], r68_meta: dict[str, Any]) -> Path | None:
    if not r68_meta:
        return None
    try:
        plt = v3.setup_matplotlib()
    except ModuleNotFoundError:
        return R68_SED_COMPARE_PNG if v3.exists(R68_SED_COMPARE_PNG) else None

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    e_pass5, y_pass5 = v3.pass5_points()
    if e_pass5:
        ax.plot(e_pass5, y_pass5, "o", ms=5.2, color="#111827", label="Official pass5 WCDA")

    for meta, style_prefix, label_prefix in [
        (nominal_meta, "nominal", "nominal r_opt=1.58*sigma"),
        (r68_meta, "r68", "empirical r68 aperture"),
    ]:
        for grouping, marker in [("nhit", "o"), ("predE", "D")]:
            energy, flux, err = v3.point_arrays(meta, grouping)
            if not energy:
                continue
            if style_prefix == "nominal":
                color = "#9ca3af" if grouping == "nhit" else "#6b7280"
                kwargs = {
                    "markerfacecolor": "none",
                    "markeredgewidth": 1.0,
                    "alpha": 0.86,
                    "zorder": 3,
                }
            else:
                color = "#2563eb" if grouping == "nhit" else "#059669"
                kwargs = {"zorder": 5}
            ax.errorbar(
                energy,
                flux,
                yerr=err,
                fmt=marker,
                ms=5.0,
                lw=1.0,
                color=color,
                ecolor=color,
                capsize=2.4,
                label=f"{label_prefix} {grouping}",
                **kwargs,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("V4 aperture control: nominal versus empirical r68")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.0, ncol=1)
    fig.tight_layout()
    V4_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(R68_SED_COMPARE_PNG)
    plt.close(fig)
    return R68_SED_COMPARE_PNG


def point_by_group(meta: dict[str, Any], grouping: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for point in meta.get("points", []) if isinstance(meta.get("points"), list) else []:
        if not isinstance(point, dict):
            continue
        if point.get("grouping") != grouping:
            continue
        label = str(point.get("group_label") or "")
        if label:
            out[label] = point
    return out


def fit_summary_row(label: str, meta: dict[str, Any]) -> list[Any]:
    preferred = meta.get("preferred_fit", {}) if isinstance(meta.get("preferred_fit"), dict) else {}
    key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
    fit = meta.get("fits", {}).get(key, {}) if isinstance(meta.get("fits"), dict) else {}
    params = fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {}
    validation = meta.get("validation", {}) if isinstance(meta.get("validation"), dict) else {}
    subset = validation.get("cell_subset", {}) if isinstance(validation.get("cell_subset"), dict) else {}
    return [
        label,
        v3.esc(subset.get("n_included_cells", validation.get("n_cells", ""))),
        v3.esc(preferred.get("model")),
        v3.fmt(params.get("phi0"), 5),
        v3.fmt(params.get("alpha", params.get("gamma")), 5),
        v3.fmt(params.get("beta"), 5),
        f"{v3.fmt(fit.get('chi2'), 4)} / {v3.fmt(fit.get('ndof'), 3)}",
    ]


def r68_aperture_section(
    *,
    nominal_e_meta: dict[str, Any],
    nominal_f_meta: dict[str, Any],
    nominal_g_meta: dict[str, Any],
    r68_b_meta: dict[str, Any],
    r68_e_meta: dict[str, Any],
    r68_f_meta: dict[str, Any],
    r68_g_meta: dict[str, Any],
) -> str:
    if not (r68_b_meta and r68_e_meta and r68_f_meta and r68_g_meta):
        return (
            "<p>This requested control is configured but the r68 Stage E/F/G outputs are not available yet.</p>"
            '<div class="note">'
            "Expected branch: <code>run_v4_r68_aperture_pipeline.sh</code> builds a Stage B contract with "
            "<code>r_opt_deg := r68_deg</code>, reruns Stage D background integration, rescans Stage E N_on, and then reruns Stage F/G."
            "</div>"
        )

    plot_path = plot_r68_sed_comparison(nominal_g_meta, r68_g_meta)
    aperture_summary = r68_b_meta.get("summary", {}) if isinstance(r68_b_meta.get("summary"), dict) else {}
    nominal_totals = nominal_e_meta.get("totals", {}) if isinstance(nominal_e_meta.get("totals"), dict) else {}
    r68_totals = r68_e_meta.get("totals", {}) if isinstance(r68_e_meta.get("totals"), dict) else {}

    nhit_nom = point_by_group(nominal_g_meta, "nhit")
    nhit_r68 = point_by_group(r68_g_meta, "nhit")
    nhit_rows = []
    for label in sorted(set(nhit_nom) | set(nhit_r68), key=v3.interval_key):
        n = nhit_nom.get(label, {})
        r = nhit_r68.get(label, {})
        n_flux = to_float(n.get("e2_dnde"))
        r_flux = to_float(r.get("e2_dnde"))
        nhit_rows.append(
            [
                v3.esc(label),
                v3.esc(n.get("n_cells", r.get("n_cells", ""))),
                v3.fmt(n.get("effective_energy_tev"), 4),
                v3.fmt(n_flux, 5),
                v3.fmt(r.get("effective_energy_tev"), 4),
                v3.fmt(r_flux, 5),
                v3.fmt((r_flux / n_flux) if n_flux and r_flux else None, 4),
            ]
        )

    fit_rows = [
        fit_summary_row("nominal r_opt=1.58*sigma", nominal_f_meta),
        fit_summary_row("empirical r68 aperture", r68_f_meta),
    ]

    on_rows = [
        [
            "nominal r_opt=1.58*sigma",
            v3.fmt(nominal_totals.get("N_on"), 7),
            v3.fmt(nominal_totals.get("B_on"), 7),
            v3.fmt(nominal_totals.get("excess"), 7),
            v3.fmt(nominal_totals.get("formal_sigma"), 5),
        ],
        [
            "empirical r68 aperture",
            v3.fmt(r68_totals.get("N_on"), 7),
            v3.fmt(r68_totals.get("B_on"), 7),
            v3.fmt(r68_totals.get("excess"), 7),
            v3.fmt(r68_totals.get("formal_sigma"), 5),
        ],
    ]

    first_nom = to_float(nhit_nom.get("[125,200)", {}).get("e2_dnde"))
    first_r68 = to_float(nhit_r68.get("[125,200)", {}).get("e2_dnde"))
    first_ratio = (first_r68 / first_nom) if first_nom and first_r68 else None
    second_nom = to_float(nhit_nom.get("[200,300)", {}).get("e2_dnde"))
    second_r68 = to_float(nhit_r68.get("[200,300)", {}).get("e2_dnde"))
    second_ratio = (second_r68 / second_nom) if second_nom and second_r68 else None

    return (
        "<p>This control replaces the analytic optimum aperture <code>r_opt = 1.58*sigma</code> with the MC empirical "
        "<code>r68</code> radius for every candidate cell. Stage D integrates the fitted 2D background inside the same r68 aperture, "
        "Stage E rescans observation events to recompute N_on, and Stage F/G use <code>containment_r_opt = 0.68</code>.</p>"
        '<div class="note">'
        f"Fit-cell aperture scale changed by median <code>{v3.fmt(aperture_summary.get('median_new_over_original_r_opt'), 4)}x</code> over the full candidate grid. "
        f"For the first two Nhit SED points, r68/nominal flux ratios are <code>{v3.fmt(first_ratio, 4)}</code> and <code>{v3.fmt(second_ratio, 4)}</code>. "
        "If these ratios are well below one, the high low-energy flux was partly driven by the previous aperture/containment convention; "
        "if they stay near one, the problem is not solved by changing aperture alone."
        "</div>"
        + v3.table(["branch", "N_on", "B_on", "excess", "formal sigma"], on_rows, cls="compact")
        + v3.table(["branch", "cells", "model", "phi0", "alpha/gamma", "beta", "chi2/ndof"], fit_rows, cls="compact")
        + "<h3>Nhit SED point movement</h3>"
        + v3.table(
            ["Nhit bin", "cells", "E nominal", "E2dN/dE nominal", "E r68", "E2dN/dE r68", "r68/nominal"],
            nhit_rows,
            cls="compact",
        )
        + '<div class="grid2">'
        + v3.figure(
            plot_path if plot_path is not None else R68_SED_COMPARE_PNG,
            "V4 aperture control SED comparison",
            "Grey open markers are the current nominal v4 drop4 points. Blue/green markers are the empirical r68-aperture rerun. Black points are official pass5.",
        )
        + v3.figure(
            R68_STAGE_G_DIR / "sed_points_stage_f_fullarray_pool1.png",
            "Native Stage G r68-aperture SED points",
            "Stage G rerun after replacing the on-region aperture and response containment by the MC empirical r68 contract.",
        )
        + "</div>"
    )


def selector_rows_from(path: Path, *, included_only: bool = True) -> list[dict[str, str]]:
    rows = v3.read_csv_rows(path)
    if not included_only:
        return rows
    return [row for row in rows if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y"}]


def fit_cell_table_from(rows: list[dict[str, str]]) -> str:
    return v3.fit_cell_table(rows)


def plot_v4_candidate_psf_profiles(fit_rows: list[dict[str, str]]) -> Path | None:
    source_png = v3.STAGE_B_NOMINAL_DIR / "psf_radial_profiles_grid.png"
    if not v3.exists(source_png):
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    fit_ids = {
        int(row["cell_id"])
        for row in fit_rows
        if str(row.get("cell_id", "")).strip().isdigit()
    }
    if not fit_ids:
        return None

    base = Image.open(source_png).convert("RGBA")
    width, height = base.size
    rgb = base.convert("RGB")
    pix = rgb.load()

    def dark_count_x(x: int) -> int:
        total = 0
        for y in range(80, height - 20):
            r, g, b = pix[x, y]
            if r < 80 and g < 80 and b < 80:
                total += 1
        return total

    def dark_count_y(y: int) -> int:
        total = 0
        for x in range(width):
            r, g, b = pix[x, y]
            if r < 80 and g < 80 and b < 80:
                total += 1
        return total

    def contiguous_groups(values: list[int]) -> list[tuple[int, int]]:
        if not values:
            return []
        groups: list[tuple[int, int]] = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
            else:
                groups.append((start, previous))
                start = previous = value
        groups.append((start, previous))
        return groups

    x_groups = contiguous_groups([x for x in range(width) if dark_count_x(x) > 1000])
    y_groups = contiguous_groups([y for y in range(height) if dark_count_y(y) > 2600])
    if len(x_groups) < 24 or len(y_groups) < 14:
        return None

    x_lines = [int(round((lo + hi) / 2.0)) for lo, hi in x_groups[:24]]
    y_lines = [int(round((lo + hi) / 2.0)) for lo, hi in y_groups[:14]]
    x_bounds = [(x_lines[i], x_lines[i + 1]) for i in range(0, 24, 2)]
    y_bounds = [(y_lines[i], y_lines[i + 1]) for i in range(0, 14, 2)]

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (236, 253, 245, 110)
    edge = (0, 158, 115, 235)
    label_fill = (209, 250, 229, 238)
    label_edge = (110, 231, 183, 245)
    label_text = (6, 95, 70, 255)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    ncols = 12
    for cell_id in sorted(fit_ids):
        idx = cell_id - 1
        row = idx // ncols
        col = idx % ncols
        if row >= len(y_bounds) or col >= len(x_bounds):
            continue
        x0, x1 = x_bounds[col]
        y0, y1 = y_bounds[row]
        draw.rectangle((x0, y0, x1, y1), fill=fill)
        for inset in range(3):
            draw.rectangle((x0 - inset, y0 - inset, x1 + inset, y1 + inset), outline=edge)
        label_box = (x0 + 7, y0 + 6, x0 + 39, y0 + 23)
        draw.rounded_rectangle(label_box, radius=4, fill=label_fill, outline=label_edge)
        draw.text((x0 + 14, y0 + 7), "fit", fill=label_text, font=small_font)

    legend_text = f"pale green panels = current v4 fit cells ({len(fit_ids)})"
    legend_box = (70, 43, 590, 72)
    draw.rounded_rectangle(legend_box, radius=7, fill=(255, 255, 255, 230), outline=(209, 213, 219, 255))
    draw.rectangle((88, 52, 112, 64), fill=fill, outline=edge)
    draw.text((122, 48), legend_text, fill=(17, 24, 39, 255), font=font)

    highlighted = Image.alpha_composite(base, overlay)
    V4_PSF_RADIAL_PROFILE_GRID_PNG.parent.mkdir(parents=True, exist_ok=True)
    highlighted.convert("RGB").save(V4_PSF_RADIAL_PROFILE_GRID_PNG)
    return V4_PSF_RADIAL_PROFILE_GRID_PNG


def psf_diagnostics_section(fit_rows: list[dict[str, str]]) -> str:
    highlighted_grid = plot_v4_candidate_psf_profiles(fit_rows) or V4_PSF_RADIAL_PROFILE_GRID_PNG
    fit_ids = ", ".join(str(row.get("cell_id")) for row in fit_rows)
    return (
        "<p>These Stage B diagnostics are MC-side PSF provenance for the current v4 drop4 fit. "
        "The radial profile grid below keeps the nominal full candidate grid, but panels with a pale green background are the cells that actually enter the current v4 Stage F/G fit.</p>"
        '<div class="note">'
        "In each panel, blue is the Crab-theta-reweighted MC radial histogram, orange is the Rayleigh-core model fitted inside the Stage B core range, and the dashed line is r_opt. "
        f"Highlighted v4 fit cells: <code>{v3.esc(fit_ids)}</code>."
        "</div>"
        '<div class="grid2">'
        + v3.figure(
            highlighted_grid,
            "Stage B candidate-grid radial PSF profiles (v4 fit cells highlighted)",
            "Pale green panels are the current v4 drop4 fit cells. Non-highlighted panels remain visible as candidate-grid PSF context.",
        )
        + v3.figure(
            v3.STAGE_B_NOMINAL_DIR / "psf_sigma_deg_grid.png",
            "Stage B PSF sigma grid",
            "Candidate-grid Rayleigh-core PSF width sigma in degrees. Smaller sigma means a narrower reconstructed Crab response for that cell.",
        )
        + v3.figure(
            v3.STAGE_B_NOMINAL_DIR / "psf_r_opt_deg_grid.png",
            "Stage B PSF r_opt grid",
            "Candidate-grid aperture radius used for the on-region integration. In v3/v4 this is tied to the fitted PSF width, approximately r_opt = 1.58 * sigma.",
        )
        + v3.figure(
            v3.STAGE_B_NOMINAL_DIR / "psf_containment_grid.png",
            "Stage B PSF containment at r_opt grid",
            "Fraction of the cell PSF contained inside r_opt. Low containment or warnings indicate broad tails or fragile low-stat PSF behavior.",
        )
        + v3.figure(
            v3.STAGE_B_NOMINAL_DIR / "psf_effective_events_grid.png",
            "Stage B PSF effective-events grid",
            "Effective MC statistics after Crab-declination theta reweighting, Neff = (sum w)^2 / sum(w^2). Low Neff means the PSF is dominated by a small number of weighted MC events.",
        )
        + v3.figure(
            v3.PSFBORROW_ASSET_DIR / "v3_active_fit_cell_theta_profiles.png",
            "Active-cell normalized MC theta profiles",
            "MC theta support after the Stage B cuts; gray is the Crab-visible theta target used for reweighting. This remains useful as PSF-support provenance for the v4 selector.",
        )
        + "</div>"
        "<h3>Current v4 fit-cell PSF table</h3>"
        + v3.active_psf_table(fit_rows)
    )


def fold_summary_by_spectrum(path: Path, spectrum: str = "official_pass5") -> dict[str, str]:
    return next((row for row in v3.read_csv_rows(path) if row.get("spectrum") == spectrum), {})


def fold_nhit_rows(path: Path, spectrum: str = "official_pass5") -> list[dict[str, str]]:
    rows = [row for row in v3.read_csv_rows(path) if row.get("spectrum") == spectrum]
    rows.sort(key=lambda row: v3.interval_key(row.get("nhit_bin")))
    return rows


def active30_vs_drop4_section(active_f_meta: dict[str, Any], drop4_f_meta: dict[str, Any]) -> str:
    active_summary = fold_summary_by_spectrum(ACTIVE30_FORWARD_DIR / "v3_official_forward_fold_summary.csv")
    drop_summary = fold_summary_by_spectrum(DROP4_FORWARD_DIR / "v3_official_forward_fold_summary.csv")
    active_nhit = {row.get("nhit_bin"): row for row in fold_nhit_rows(ACTIVE30_FORWARD_DIR / "v3_official_forward_fold_nhit_summary.csv")}
    drop_nhit = {row.get("nhit_bin"): row for row in fold_nhit_rows(DROP4_FORWARD_DIR / "v3_official_forward_fold_nhit_summary.csv")}

    def fit_row(label: str, meta: dict[str, Any], summary: dict[str, str]) -> list[Any]:
        preferred = meta.get("preferred_fit", {}) if isinstance(meta.get("preferred_fit"), dict) else {}
        key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
        fit = meta.get("fits", {}).get(key, {}) if isinstance(meta.get("fits"), dict) else {}
        params = fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {}
        subset = meta.get("validation", {}).get("cell_subset", {}) if isinstance(meta.get("validation"), dict) else {}
        return [
            v3.esc(label),
            v3.esc(subset.get("n_included_cells")),
            v3.fmt(summary.get("total_observed_over_expected"), 4),
            v3.fmt(params.get("phi0"), 5),
            v3.fmt(params.get("alpha", params.get("gamma")), 5),
            v3.fmt(params.get("beta"), 5),
            f"{v3.fmt(fit.get('chi2'), 4)} / {v3.fmt(fit.get('ndof'), 3)}",
        ]

    nhit_rows = []
    for label in sorted(set(active_nhit) | set(drop_nhit), key=v3.interval_key):
        active = active_nhit.get(label, {})
        drop = drop_nhit.get(label, {})
        nhit_rows.append(
            [
                v3.esc(label),
                v3.esc(active.get("cells")),
                v3.fmt(active.get("total_observed_over_expected"), 4),
                v3.esc(drop.get("cells")),
                v3.fmt(drop.get("total_observed_over_expected"), 4),
                v3.fmt(
                    (to_float(drop.get("total_observed_over_expected")) or float("nan"))
                    - (to_float(active.get("total_observed_over_expected")) or float("nan")),
                    4,
                ),
            ]
        )

    return (
        "<p>This control removes cells <code>4, 17, 39, 43</code> from the original active30 list, then reruns Stage F/G using the same latest annulus-normalized Stage E signal. "
        "The purpose is to test whether these visually/diagnostically suspect cells are driving the low-Nhit excess.</p>"
        '<div class="note">'
        "Result: dropping these four cells does not reduce the official-pass5 underprediction. "
        "The total observed/expected ratio changes from 1.422x to 1.448x, and the two lowest Nhit bins remain at 1.536x and 1.516x. "
        "So this specific four-cell removal is not evidence that those cells are the main source of the low-energy high flux."
        "</div>"
        + v3.table(
            ["selector", "cells", "official obs/exp", "phi0", "alpha/gamma", "beta", "LogPar chi2/ndof"],
            [
                fit_row("active30", active_f_meta, active_summary),
                fit_row("drop4: remove 4,17,39,43", drop4_f_meta, drop_summary),
            ],
            cls="compact",
        )
        + v3.table(
            ["Nhit bin", "active cells", "active obs/exp", "drop4 cells", "drop4 obs/exp", "drop-active"],
            nhit_rows,
            cls="compact",
        )
    )


def forward_fold_section() -> str:
    summary_rows_all = v3.read_csv_rows(RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_selector_nhit.csv")
    cell_rows_all = v3.read_csv_rows(RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_cell.csv")
    primary_mode = "aperture_response_containment_1"
    legacy_mode = "nominal_containment"
    summary_rows = [
        row
        for row in summary_rows_all
        if row.get("selector") == "drop4" and row.get("containment_mode") in {primary_mode, legacy_mode}
    ]
    primary_summary = next((row for row in summary_rows if row.get("containment_mode") == primary_mode and row.get("nhit_bin") == "all"), {})
    legacy_summary = next((row for row in summary_rows if row.get("containment_mode") == legacy_mode and row.get("nhit_bin") == "all"), {})
    primary_nhit = [
        row
        for row in summary_rows
        if row.get("containment_mode") == primary_mode and row.get("nhit_bin") != "all"
    ]
    primary_nhit.sort(key=lambda row: v3.interval_key(row.get("nhit_bin")))

    low_bins = [row for row in primary_nhit if row.get("nhit_bin") in {"[125,200)", "[200,300)", "[300,500)", "[500,800)"}]
    low_ratio_text = ", ".join(
        f"<code>{v3.esc(row.get('nhit_bin'))}</code>: {v3.fmt(row.get('observed_over_expected'), 4)}x"
        for row in low_bins
    )

    total_table = v3.table(
        [
            "contract",
            "cells",
            "excess",
            "official expected",
            "observed/expected",
            "expected / all-dir c=1",
        ],
        [
            [
                "aperture response x 1" if row.get("containment_mode") == primary_mode else "legacy all-dir response x containment",
                v3.esc(row.get("cells")),
                v3.fmt(row.get("excess"), 6),
                v3.fmt(row.get("official_expected_counts"), 6),
                v3.fmt(row.get("observed_over_expected"), 4),
                v3.fmt(row.get("expected_over_all_direction_c1"), 4),
            ]
            for row in [primary_summary, legacy_summary]
            if row
        ],
        cls="compact",
    )

    nhit_table = v3.table(
        ["Nhit bin", "cells", "excess", "official expected", "observed/expected", "N_on/B_on"],
        [
            [
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("cells")),
                v3.fmt(row.get("excess"), 6),
                v3.fmt(row.get("official_expected_counts"), 6),
                v3.fmt(row.get("observed_over_expected"), 4),
                v3.fmt(row.get("N_on_over_B_on"), 4),
            ]
            for row in primary_nhit
        ],
        cls="compact",
    )

    primary_cell_rows = [
        row
        for row in cell_rows_all
        if row.get("drop4") == "1" and row.get("nhit_bin") in {"[125,200)", "[200,300)", "[300,500)", "[500,800)"}
    ]
    primary_cell_rows.sort(
        key=lambda row: v3.finite_float(row.get("ratio_aperture_response")) or -1.0e99,
        reverse=True,
    )

    cell_table = v3.table(
        ["cell", "Nhit", "predE", "excess", "official expected", "obs/exp", "excess - exp", "pull", "containment"],
        [
            (
                lambda expected, excess, err: [
                    v3.esc(row.get("cell_id")),
                    v3.esc(row.get("nhit_bin")),
                    v3.esc(row.get("predE_bin")),
                    v3.fmt(excess, 6),
                    v3.fmt(expected, 6),
                    v3.fmt((excess / expected) if expected and expected > 0 else None, 4),
                    v3.fmt((excess - expected) if expected is not None and excess is not None else None, 5),
                    v3.fmt(((excess - expected) / err) if expected is not None and excess is not None and err and err > 0 else None, 4),
                    "1.0",
                ]
            )(
                v3.finite_float(row.get("official_expected_aperture_response")),
                v3.finite_float(row.get("excess")),
                math.sqrt(
                    max(
                        0.0,
                        (v3.finite_float(row.get("N_on")) or 0.0)
                        + (v3.finite_float(row.get("B_on")) or 0.0),
                    )
                ),
            )
            for row in primary_cell_rows[:14]
        ],
        cls="compact",
    )

    legacy_cell_table = v3.table(
        ["cell", "Nhit", "predE", "legacy obs/exp", "aperture obs/exp", "legacy expected", "aperture expected"],
        [
            [
                v3.esc(row.get("cell_id")),
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("predE_bin")),
                v3.fmt(row.get("ratio_nominal"), 4),
                v3.fmt(row.get("ratio_aperture_response"), 4),
                v3.fmt(row.get("official_expected_nominal"), 6),
                v3.fmt(row.get("official_expected_aperture_response"), 6),
            ]
            for row in primary_cell_rows[:12]
        ],
        cls="compact",
    )

    total_ratio = v3.fmt(primary_summary.get("observed_over_expected"), 4)
    legacy_ratio = v3.fmt(legacy_summary.get("observed_over_expected"), 4)
    total_excess = v3.fmt(primary_summary.get("excess"), 6)
    total_expected = v3.fmt(primary_summary.get("official_expected_counts"), 6)
    return (
        "<p>This primary v4 test folds the official pass5 WCDA spectrum through the aperture-conditioned Stage A response, "
        "the drop4 26-cell list, and theta exposure with downstream <code>containment_r_opt=1</code>.</p>"
        '<div class="note">'
        f"With the new primary contract, official pass5 predicts {total_expected} counts for the drop4 26-cell selector, "
        f"while the latest annnorm excess is {total_excess}; observed/expected is {total_ratio}x. "
        f"The old all-direction x containment contract gave {legacy_ratio}x. Low-Nhit aperture-response ratios are: {low_ratio_text}."
        "</div>"
        "<h3>Total fold summary</h3>"
        + total_table
        + "<h3>Official pass5 fold by Nhit: aperture response x 1</h3>"
        + nhit_table
        + "<h3>Largest low-Nhit cell residuals versus official pass5: aperture response x 1</h3>"
        + cell_table
        + "<h3>Legacy contract comparison for the same cells</h3>"
        + legacy_cell_table
        + '<div class="grid2">'
        + v3.figure(
            RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_nhit.png",
            "Official pass5 forward-fold ratios by response contract",
            "Green markers correspond to the new primary aperture-conditioned response with containment fixed to one.",
        )
        + v3.figure(
            RESPONSE_AUDIT_DIR / "v4_sed_nominal_vs_containment1.png",
            "SED response/aperture contract comparison",
            "Blue is the legacy all-direction response x containment branch; green is the new primary aperture-conditioned response x 1 branch.",
        )
        + "</div>"
    )


def root_cause_diagnostics_section() -> str:
    required_rows = v3.read_csv_rows(ROOT_CAUSE_DIR / "required_background_shift_nhit_summary.csv")
    offsource_rows = v3.read_csv_rows(ROOT_CAUSE_DIR / "offsource_core_residual_summary.csv")
    extrap_rows = v3.read_csv_rows(ROOT_CAUSE_DIR / "official_pass5_low_energy_extrapolation_sensitivity.csv")
    closure_rows = v3.read_csv_rows(ROOT_CAUSE_DIR / "official_pass5_response_closure_fit_summary.csv")
    energy_rows = v3.read_csv_rows(ROOT_CAUSE_DIR / "official_pass5_true_energy_contribution_by_cell.csv")

    required_table_rows = []
    for row in required_rows:
        if row.get("selector") not in {"active30", "drop4"}:
            continue
        required_table_rows.append(
            [
                v3.esc(row.get("selector")),
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("cells")),
                v3.fmt(row.get("observed_over_expected"), 4),
                v3.fmt(100.0 * (v3.finite_float(row.get("required_delta_b_over_b")) or 0.0), 4) + "%",
                v3.fmt(row.get("total_required_delta_b"), 5),
            ]
        )

    offsource_table_rows = [
        [
            v3.esc(row.get("fake_source")),
            v3.esc(row.get("selector")),
            v3.esc(row.get("cells")),
            v3.fmt(row.get("excess"), 5),
            v3.fmt(row.get("combined_known_background_sigma"), 4),
            v3.esc(row.get("positive_excess_cells")),
            v3.esc(row.get("negative_excess_cells")),
        ]
        for row in offsource_rows
        if row.get("selector") in {"active30", "drop4"}
    ]

    closure_table_rows = [
        [
            v3.esc(row.get("selector")),
            v3.esc(row.get("model")),
            v3.esc(row.get("cells")),
            f"{v3.fmt(row.get('chi2'), 4)} / {v3.fmt(row.get('ndof'), 3)}",
            v3.fmt(row.get("phi0"), 5),
            v3.fmt(row.get("gamma") or row.get("alpha"), 5),
            v3.fmt(row.get("beta"), 5),
            v3.fmt(row.get("max_abs_pull"), 4),
        ]
        for row in closure_rows
        if row.get("selector") in {"active30", "drop4"}
    ]

    extrap_keep = {"pass5_endpoint_extrap", "pass5_cut_below_min"}
    extrap_table_rows = [
        [
            v3.esc(row.get("selector")),
            v3.esc(row.get("variant")),
            v3.esc(row.get("nhit_bin")),
            v3.esc(row.get("cells")),
            v3.fmt(row.get("observed_over_expected"), 4),
        ]
        for row in extrap_rows
        if row.get("selector") in {"active30", "drop4"}
        and row.get("variant") in extrap_keep
        and row.get("nhit_bin") in {"[125,200)", "[200,300)", "[300,500)"}
    ]

    energy_table_rows = [
        [
            v3.esc(row.get("selector")),
            v3.esc(row.get("cell_id")),
            v3.esc(row.get("nhit_bin")),
            v3.esc(row.get("predE_bin")),
            v3.fmt(row.get("true_e50_tev"), 4),
            v3.fmt(100.0 * (v3.finite_float(row.get("frac_below_pass5_min")) or 0.0), 4) + "%",
            v3.fmt(100.0 * (v3.finite_float(row.get("frac_below_1tev")) or 0.0), 4) + "%",
        ]
        for row in energy_rows
        if row.get("selector") == "drop4" and row.get("nhit_bin") in {"[125,200)", "[200,300)"}
    ]

    return (
        "<p>These diagnostics turn the current hypotheses into falsifiable checks: how much B_on shift would be needed, whether off-source pseudo-Crab regions show a positive residual, whether official pass5 low-energy extrapolation matters, whether Stage F closes on pseudo-data, and whether selector changes reduce the discrepancy.</p>"
        '<div class="note">'
        "Current evidence: the off-source controls are strongly negative rather than positive, Stage F closes on official-pass5 pseudo-data with LogPar chi2/ndof near 0.1, and the drop4 selector does not reduce the low-Nhit official underprediction. "
        "The most likely remaining issue is therefore response normalization / energy migration / active-cell conditioning, with the caveat that the lowest [125,200) bin is also sensitive to official-pass5 extrapolation below 0.56 TeV."
        "</div>"
        "<h3>Required background shift</h3>"
        + v3.table(
            ["selector", "Nhit", "cells", "obs/official", "required delta B / B_on", "required delta B"],
            required_table_rows,
            cls="compact",
        )
        + '<div class="grid2">'
        + v3.figure(
            ROOT_CAUSE_DIR / "required_background_shift_by_nhit.png",
            "Required B_on shift by Nhit",
            "The B_on increase needed to force official pass5 folded source counts to match the observed excess. A few percent is plausible; tens of percent would require a large background failure.",
        )
        + v3.figure(
            ROOT_CAUSE_DIR / "offsource_pseudocrab_residual_sigma.png",
            "Off-source pseudo-Crab residual sigma",
            "RA-shifted fake-source controls using existing off-source Stage E products. Strong negative residuals argue against a simple positive bkg-underestimate explanation.",
        )
        + "</div>"
        + "<h3>Off-source pseudo-Crab summary</h3>"
        + v3.table(
            ["fake source", "selector", "cells", "excess", "combined sigma", "positive cells", "negative cells"],
            offsource_table_rows,
            cls="compact",
        )
        + "<h3>Low-energy extrapolation sensitivity</h3>"
        + v3.table(
            ["selector", "variant", "Nhit", "cells", "obs/expected"],
            extrap_table_rows,
            cls="compact",
        )
        + '<div class="grid2">'
        + v3.figure(
            ROOT_CAUSE_DIR / "low_energy_extrapolation_sensitivity.png",
            "Official pass5 low-energy extrapolation sensitivity",
            "Endpoint extrapolation, flat-below-min, and cut-below-min variants. The lowest Nhit bins depend strongly on how flux below the first official pass5 point is handled.",
        )
        + v3.figure(
            ROOT_CAUSE_DIR / "required_background_shift_by_nhit.png",
            "Background-shift diagnostic repeated for comparison",
            "Shown again beside the extrapolation plot to compare the background and response interpretations.",
        )
        + "</div>"
        + "<h3>True-energy support of low-Nhit cells</h3>"
        + v3.table(
            ["selector", "cell", "Nhit", "predE", "E50 TeV", "below pass5 min", "below 1 TeV"],
            energy_table_rows,
            cls="compact",
        )
        + "<h3>Official-pass5 response closure</h3>"
        + v3.table(
            ["selector", "model", "cells", "chi2/ndof", "phi0", "gamma/alpha", "beta", "max abs pull"],
            closure_table_rows,
            cls="compact",
        )
    )


def cell_root_cause_crossmatch_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>Cell-level crossmatch was not generated.</p>"

    def f(row: dict[str, Any], key: str) -> float | None:
        return v3.finite_float(row.get(key))

    by_logpar = sorted(rows, key=lambda row: abs(f(row, "logpar_pull") or 0.0), reverse=True)
    by_official = sorted(rows, key=lambda row: f(row, "ratio_aperture_response") or -1.0e99, reverse=True)
    low_mid = [row for row in rows if row.get("nhit_bin") in {"[125,200)", "[200,300)", "[300,500)"}]
    high = [row for row in rows if row.get("nhit_bin") in {"[800,1100)", "[1100,2000)", "[2000,3000)"}]

    low_mid_excess = sum(f(row, "excess") or 0.0 for row in low_mid)
    low_mid_official = sum(f(row, "official_expected_aperture_response") or 0.0 for row in low_mid)
    high_excess = sum(f(row, "excess") or 0.0 for row in high)
    high_official = sum(f(row, "official_expected_aperture_response") or 0.0 for row in high)
    positive_offsrc = [row for row in rows if (f(row, "offsource_mean_sigma") or 0.0) > 0.0]
    reliable_empirical = [row for row in rows if truthy(row.get("fit_reliable"))]
    empirical_narrow = [row for row in reliable_empirical if (f(row, "sigma_obs_over_mc") or 1.0e9) < 0.9]

    top_stagef_table = v3.table(
        [
            "cell",
            "Nhit",
            "predE",
            "excess",
            "LogPar model",
            "pull",
            "obs/official",
            "dB/B",
            "emp sigma/MC",
            "E50 TeV",
            "offsrc sigma",
        ],
        [
            [
                v3.esc(row.get("cell_id")),
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("predE_bin")),
                v3.fmt(row.get("excess"), 5),
                v3.fmt(row.get("logpar_model"), 5),
                v3.fmt(row.get("logpar_pull"), 4),
                v3.fmt(row.get("ratio_aperture_response"), 4),
                v3.fmt(100.0 * (f(row, "required_delta_b_over_b_aperture_response") or 0.0), 4) + "%",
                v3.fmt(row.get("sigma_obs_over_mc"), 4),
                v3.fmt(row.get("true_e50_tev"), 4),
                v3.fmt(row.get("offsource_mean_sigma"), 4),
            ]
            for row in by_logpar[:12]
        ],
        cls="compact",
    )

    top_official_table = v3.table(
        [
            "cell",
            "Nhit",
            "predE",
            "excess",
            "official exp",
            "obs/official",
            "official pull",
            "dB/B",
            "Stage F pull",
            "below 1 TeV",
        ],
        [
            [
                v3.esc(row.get("cell_id")),
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("predE_bin")),
                v3.fmt(row.get("excess"), 5),
                v3.fmt(row.get("official_expected_aperture_response"), 5),
                v3.fmt(row.get("ratio_aperture_response"), 4),
                v3.fmt(row.get("fit_vs_official_pass5_aperture_pull"), 4),
                v3.fmt(100.0 * (f(row, "required_delta_b_over_b_aperture_response") or 0.0), 4) + "%",
                v3.fmt(row.get("logpar_pull"), 4),
                v3.fmt(100.0 * (f(row, "frac_below_1tev") or 0.0), 4) + "%",
            ]
            for row in by_official[:12]
        ],
        cls="compact",
    )

    selector_rows_all = v3.read_csv_rows(RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_selector_nhit.csv")
    selector_rows = [
        row
        for row in selector_rows_all
        if row.get("containment_mode") == "aperture_response_containment_1"
        and row.get("selector") in {"all84", "active30", "drop4"}
        and row.get("nhit_bin") in {"all", "[125,200)", "[200,300)", "[300,500)", "[500,800)"}
    ]
    selector_rows.sort(key=lambda row: (row.get("nhit_bin") != "all", v3.interval_key(row.get("nhit_bin")), row.get("selector")))
    selector_table = v3.table(
        ["selector", "Nhit", "cells", "excess", "official exp", "obs/official"],
        [
            [
                v3.esc(row.get("selector")),
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("cells")),
                v3.fmt(row.get("excess"), 5),
                v3.fmt(row.get("official_expected_counts"), 5),
                v3.fmt(row.get("observed_over_expected"), 4),
            ]
            for row in selector_rows
        ],
        cls="compact",
    )

    low_mid_ratio = low_mid_excess / low_mid_official if low_mid_official > 0 else None
    high_ratio = high_excess / high_official if high_official > 0 else None
    return (
        "<p>This table joins the current Stage F residuals, official-pass5 forward-fold residuals, background-shift requirement, off-source pseudo-Crab residuals, Stage A MC occupancy, Stage B PSF, observed empirical PSF, and true-energy support for each current drop4 fit cell.</p>"
        '<div class="note">'
        f"Current localization: low/mid Nhit drop4 cells [125,500) have obs/official <code>{v3.fmt(low_mid_ratio, 4)}x</code>, while the high-Nhit diagnostic group has <code>{v3.fmt(high_ratio, 4)}x</code>. "
        f"Only <code>{len(positive_offsrc)}/{len(rows)}</code> fit cells have positive mean off-source residual, so the existing off-source controls do not support a uniform positive background under-subtraction. "
        f"Among reliable empirical-PSF cells, <code>{len(empirical_narrow)}/{len(reliable_empirical)}</code> have observed sigma/MC &lt; 0.9, so the observed Crab core is often narrower than the MC PSF rather than broader."
        "</div>"
        '<div class="grid2">'
        + v3.figure(
            CELL_ROOT_CAUSE_PNG,
            "V4 drop4 cell-level root-cause crossmatch",
            "Top panel is Stage F LogPar pull; second is observed/official pass5 under the aperture-conditioned response; third is the B_on shift needed to match official pass5; bottom overlays empirical PSF width ratio and off-source residual sign.",
        )
        + v3.figure(
            RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_nhit.png",
            "Selector and response-contract comparison",
            "Use this beside the cell crossmatch: selector choice changes the low-Nhit ratio, but all84 is still above official in [125,500).",
        )
        + "</div>"
        + "<h3>Selector localization: aperture response x 1</h3>"
        + selector_table
        + "<h3>Largest Stage F residual cells</h3>"
        + top_stagef_table
        + "<h3>Largest official-pass5 aperture-response residual cells</h3>"
        + top_official_table
        + f"<p>Machine-readable joined table: <code>{v3.esc(v3.rel(CELL_ROOT_CAUSE_CROSSMATCH_CSV))}</code>.</p>"
    )


def response_audit_section() -> str:
    summary_rows = v3.read_csv_rows(RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_selector_nhit.csv")
    fit_rows = v3.read_csv_rows(RESPONSE_AUDIT_DIR / "stage_f_nominal_vs_containment1_summary.csv")
    cell_rows = v3.read_csv_rows(RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_cell.csv")
    summary = v3.load_json(RESPONSE_AUDIT_DIR / "v4_response_audit_summary.json")
    summary_payload = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    has_aperture_response = any(row.get("containment_mode") == "aperture_response_containment_1" for row in summary_rows)

    def row_for(selector: str, mode: str, nhit: str) -> dict[str, str]:
        return next(
            (
                row
                for row in summary_rows
                if row.get("selector") == selector and row.get("containment_mode") == mode and row.get("nhit_bin") == nhit
            ),
            {},
        )

    def expected_factor(row: dict[str, str]) -> str:
        return v3.fmt(
            row.get("expected_over_all_direction_c1")
            if row.get("expected_over_all_direction_c1") not in {None, ""}
            else row.get("effective_containment_factor"),
            4,
        )

    overview_rows = []
    for selector in ["all84", "active30", "drop4"]:
        nominal = row_for(selector, "nominal_containment", "all")
        cont1 = row_for(selector, "containment_1", "all")
        aperture = row_for(selector, "aperture_response_containment_1", "all")
        overview_rows.append(
            [
                v3.esc(selector),
                v3.esc(nominal.get("cells")),
                v3.fmt(nominal.get("observed_over_expected"), 4),
                v3.fmt(cont1.get("observed_over_expected"), 4),
                v3.fmt(aperture.get("observed_over_expected"), 4),
                expected_factor(nominal),
                expected_factor(aperture),
                v3.fmt(nominal.get("excess"), 6),
                v3.fmt(nominal.get("official_expected_counts"), 6),
                v3.fmt(cont1.get("official_expected_counts"), 6),
                v3.fmt(aperture.get("official_expected_counts"), 6),
            ]
        )

    low_nhit_rows = []
    for nhit in ["[125,200)", "[200,300)", "[300,500)", "[500,800)"]:
        nominal = row_for("drop4", "nominal_containment", nhit)
        cont1 = row_for("drop4", "containment_1", nhit)
        aperture = row_for("drop4", "aperture_response_containment_1", nhit)
        low_nhit_rows.append(
            [
                v3.esc(nhit),
                v3.esc(nominal.get("cells")),
                v3.fmt(nominal.get("observed_over_expected"), 4),
                v3.fmt(cont1.get("observed_over_expected"), 4),
                v3.fmt(aperture.get("observed_over_expected"), 4),
                expected_factor(nominal),
                expected_factor(aperture),
                v3.fmt(nominal.get("excess"), 5),
                v3.fmt(nominal.get("B_on"), 5),
            ]
        )

    fit_table_rows = [
        [
            v3.esc(row.get("run")),
            v3.esc(row.get("preferred_model")),
            v3.esc(row.get("cells")),
            v3.fmt(row.get("phi0"), 5),
            v3.fmt(row.get("alpha") or row.get("gamma"), 5),
            v3.fmt(row.get("beta"), 5),
            f"{v3.fmt(row.get('chi2'), 4)} / {v3.fmt(row.get('ndof'), 3)}",
        ]
        for row in fit_rows
    ]

    high_residual_cells = [
        row
        for row in cell_rows
        if row.get("drop4") == "1" and row.get("nhit_bin") in {"[125,200)", "[200,300)", "[300,500)", "[500,800)"}
    ]
    high_residual_cells.sort(
        key=lambda row: v3.finite_float(row.get("ratio_nominal")) or -1.0e99,
        reverse=True,
    )
    cell_table_rows = [
        [
            v3.esc(row.get("cell_id")),
            v3.esc(row.get("nhit_bin")),
            v3.esc(row.get("predE_bin")),
            v3.fmt(row.get("containment_r_opt"), 4),
            v3.fmt(row.get("ratio_nominal"), 4),
            v3.fmt(row.get("ratio_containment1"), 4),
            v3.fmt(row.get("ratio_aperture_response"), 4),
            v3.fmt(100.0 * (v3.finite_float(row.get("required_delta_b_over_b_nominal")) or 0.0), 3) + "%",
            v3.fmt(100.0 * (v3.finite_float(row.get("required_delta_b_over_b_containment1")) or 0.0), 3) + "%",
            v3.fmt(100.0 * (v3.finite_float(row.get("required_delta_b_over_b_aperture_response")) or 0.0), 3) + "%",
        ]
        for row in high_residual_cells[:12]
    ]

    interpretation = str(summary_payload.get("double_containment_interpretation") or "pending_aperture_conditioned_stage_a")
    if interpretation == "scalar_containment_suppresses_expectation_beyond_aperture_conditioned_response":
        verdict = (
            "Reading the contract from expected counts, aperture-conditioned Stage A with <code>containment_r_opt=1</code> predicts "
            f"<code>{v3.fmt(summary_payload.get('drop4_aperture_expected_over_nominal'), 4)}x</code> the current expected counts. "
            "So the current scalar-containment branch suppresses the source expectation more than a direct aperture-conditioned response does. "
            "This supports a response/containment contract bias, but it is not a full factor-of-containment double count."
        )
    elif interpretation == "aperture_conditioned_response_matches_containment1_nominal_containment_inconsistent":
        verdict = (
            "Aperture-conditioned Stage A with <code>containment_r_opt=1</code> lands closer to the all-direction response x 1 ablation. "
            "That would mean the scalar containment convention used by the current branch is inconsistent with the response contract."
        )
    elif interpretation == "aperture_conditioned_response_matches_nominal_containment_no_double_containment":
        verdict = (
            "Aperture-conditioned Stage A with <code>containment_r_opt=1</code> lands closer to the current all-direction response x containment branch. "
            "That argues against a simple double-containment error."
        )
    else:
        verdict = (
            "The aperture-conditioned Stage A full response is still pending. Until that branch exists, the containment=1 run is only an ablation, not the final response-contract answer."
        )

    return (
        "<p>This audit isolates response/aperture effects without changing the Stage E background. "
        "It compares three contracts: the current all-direction Stage A response multiplied by <code>containment_r_opt</code>, "
        "the same all-direction response with containment fixed to one, and the definitive aperture-conditioned Stage A response with containment fixed to one.</p>"
        '<div class="note">'
        f"For drop4, official pass5 obs/expected is <code>{v3.fmt(summary_payload.get('drop4_nominal_official_obs_over_expected'), 4)}x</code> in the current branch, "
        f"<code>{v3.fmt(summary_payload.get('drop4_containment1_official_obs_over_expected'), 4)}x</code> with all-direction response x 1, and "
        f"<code>{v3.fmt(summary_payload.get('drop4_aperture_conditioned_official_obs_over_expected'), 4)}x</code> with aperture-conditioned response x 1. "
        f"{verdict}"
        "</div>"
        + v3.table(
            [
                "selector",
                "cells",
                "obs/official current",
                "obs/official all-dir c=1",
                "obs/official aperture c=1",
                "current expected / all-dir",
                "aperture expected / all-dir",
                "excess",
                "expected current",
                "expected all-dir",
                "expected aperture",
            ],
            overview_rows,
            cls="compact",
        )
        + "<h3>Drop4 low-Nhit response-contract comparison</h3>"
        + v3.table(
            [
                "Nhit",
                "cells",
                "obs/official current",
                "obs/official all-dir c=1",
                "obs/official aperture c=1",
                "current expected / all-dir",
                "aperture expected / all-dir",
                "excess",
                "B_on",
            ],
            low_nhit_rows,
            cls="compact",
        )
        + '<div class="grid2">'
        + v3.figure(
            RESPONSE_AUDIT_DIR / "official_pass5_containment_ablation_by_nhit.png",
            "Official pass5 forward-fold ratios by response contract",
            "Current branch: all-direction Stage A response x containment_r_opt. Ablation: all-direction response x 1. Definitive check, when available: aperture-conditioned response x 1.",
        )
        + v3.figure(
            RESPONSE_AUDIT_DIR / "containment_r_opt_by_nhit.png",
            "Containment factors by Nhit",
            "The current Stage B containment factors are typically 0.6-0.7 in low/mid Nhit. Multiplying by these factors suppresses expected counts by the same scale and raises fitted flux.",
        )
        + "</div>"
        + "<h3>Stage F fit impact</h3>"
        + v3.table(
            ["run", "model", "cells", "phi0", "alpha/gamma", "beta", "chi2/ndof"],
            fit_table_rows,
            cls="compact",
        )
        + '<div class="grid2">'
        + v3.figure(
            RESPONSE_AUDIT_DIR / "v4_sed_nominal_vs_containment1.png",
            "SED impact of response/aperture contract",
            "Blue is the current branch, red is all-direction response x 1, and green is aperture-conditioned response x 1 when the full Stage A rebuild is available.",
        )
        + v3.figure(
            V4_RESPONSE_CONTRACT_OVERLAY_PNG,
            "Native Stage G response-contract plot",
            "Overlay version of the native Stage G diagnostic: v4 primary Stage G points and Stage F fit, official pass5 and tutorial v0.99 points with point-fit LogPar curves, plus H.E.S.S. and MAGIC external measurements. The fit parameters are annotated inside the plot.",
        )
        + "</div>"
        + "<h3>Largest drop4 low-Nhit cell changes</h3>"
        + v3.table(
            [
                "cell",
                "Nhit",
                "predE",
                "containment",
                "ratio current",
                "ratio all-dir c=1",
                "ratio aperture c=1",
                "delta B/B current",
                "delta B/B all-dir c=1",
                "delta B/B aperture c=1",
            ],
            cell_table_rows,
            cls="compact",
        )
    )


def empirical_psf_section() -> str:
    summary_meta = v3.load_json(EMPIRICAL_PSF_SUMMARY_JSON) if EMPIRICAL_PSF_SUMMARY_JSON.exists() else {}
    summary = summary_meta.get("summary", {}) if isinstance(summary_meta.get("summary"), dict) else {}
    cell_rows = v3.read_csv_rows(EMPIRICAL_PSF_CELL_CSV) if EMPIRICAL_PSF_CELL_CSV.exists() else []
    group_rows = v3.read_csv_rows(EMPIRICAL_PSF_GROUP_CSV) if EMPIRICAL_PSF_GROUP_CSV.exists() else []

    def reliable_label(row: dict[str, str]) -> str:
        reliable = str(row.get("fit_reliable", "")).strip() == "1"
        if reliable:
            return '<span class="pill">reliable</span>'
        reason = row.get("unreliable_reason") or "unreliable"
        return v3.esc(reason)

    cell_table_rows = [
        [
            v3.esc(row.get("cell_id")),
            v3.esc(row.get("nhit_bin")),
            v3.esc(row.get("predE_bin")),
            v3.fmt(row.get("N_on"), 6),
            v3.fmt(row.get("B_on"), 6),
            v3.fmt(row.get("excess"), 5),
            v3.fmt(row.get("significance"), 4),
            reliable_label(row),
            v3.fmt(row.get("sigma_obs_over_mc"), 4),
            v3.fmt(row.get("r68_obs_over_mc"), 4),
            v3.fmt(row.get("profile_residual_rms"), 4),
        ]
        for row in cell_rows
    ]
    group_table_rows = [
        [
            v3.esc(row.get("nhit_bin")),
            v3.esc(row.get("n_cells")),
            v3.esc(row.get("cell_ids")),
            v3.fmt(row.get("N_on"), 6),
            v3.fmt(row.get("B_on"), 6),
            v3.fmt(row.get("excess"), 5),
            v3.fmt(row.get("significance"), 4),
            reliable_label(row),
            v3.fmt(row.get("sigma_obs_deg"), 4),
            v3.fmt(row.get("r68_obs_deg"), 4),
        ]
        for row in group_rows
    ]
    unreliable = [row for row in cell_rows if str(row.get("fit_reliable", "")).strip() != "1"]
    risk_rows = [row for row in cell_rows if str(row.get("psf_risk_cell", "")).strip() == "1"]
    risk_text = ", ".join(
        f"cell {v3.esc(row.get('cell_id'))}: {reliable_label(row)}"
        for row in risk_rows
    ) or "none in current selector"

    return (
        "<p>This diagnostic fits an empirical/effective PSF directly from observed Crab excess maps. "
        "It keeps the latest annulus-normalized Stage D background fixed, so it is a PSF/containment check rather than a new background fit or a replacement for the MC response.</p>"
        '<div class="note">'
        f"Current v4 drop4 fit cells: <code>{v3.esc(str(summary.get('cells', 'n/a')))}</code>; "
        f"single-cell empirical PSF fits passing the preset statistics gate: <code>{v3.esc(str(summary.get('reliable_cells', 'n/a')))}</code>. "
        f"Reliable-cell median sigma_obs/MC is <code>{v3.fmt(summary.get('median_sigma_obs_over_mc_reliable'), 4)}</code>; "
        f"median r68_obs/MC is <code>{v3.fmt(summary.get('median_r68_obs_over_mc_reliable'), 4)}</code>. "
        f"Profile integration closure max absolute error is <code>{v3.fmt(summary.get('profile_check_max_abs_error'), 4)}</code>. "
        "Interpretation: if low-energy high-excess cells showed a coherent sigma_obs/MC or r68_obs/MC shift, spatial containment would be implicated; if not, continue prioritizing response normalization, energy migration, and cell-selection normalization."
        "</div>"
        '<div class="grid2">'
        + v3.figure(
            EMPIRICAL_PSF_DIR / "observed_vs_mc_radial_profiles_grid.png",
            "Observed empirical PSF versus MC PSF",
            "Orange curves are peak-normalized observed excess radial profiles from counts minus fixed Stage D background. Blue curves are the Stage B MC PSF profiles. Black curves are simple Rayleigh/Gaussian fits to the observed excess core. Red titles mark cells failing the statistics gate.",
        )
        + v3.figure(
            EMPIRICAL_PSF_DIR / "observed_radial_profile_components_grid.png",
            "Observed radial profile components",
            "Raw radial sums for counts, fitted background, and counts-background. This checks whether the empirical PSF fit is being driven by source excess or by residual background shape.",
        )
        + v3.figure(
            EMPIRICAL_PSF_DIR / "sigma_obs_over_mc_grid.png",
            "sigma_obs / sigma_MC grid",
            "Cell-grid ratio of observed empirical Rayleigh width to Stage B MC sigma. Starred cells are plotted but not used for strong single-cell conclusions because they fail the statistics gate.",
        )
        + v3.figure(
            EMPIRICAL_PSF_DIR / "r68_obs_over_mc_grid.png",
            "r68_obs / r68_MC grid",
            "Cell-grid ratio of observed empirical r68 to Stage B MC r68. This is the containment-width comparison most directly related to aperture consistency.",
        )
        + v3.figure(
            EMPIRICAL_PSF_DIR / "nhit_group_empirical_psf_overlays.png",
            "Nhit-group empirical PSF fallback overlays",
            "Fit-cell profiles summed by Nhit bin. This grouped fallback is the safer view when individual high-Nhit cells have too few observed counts for stable PSF fitting.",
        )
        + "</div>"
        "<h3>Single-cell empirical PSF reliability</h3>"
        + v3.table(
            ["cell", "Nhit", "predE", "N_on", "B_on", "excess", "sig", "gate", "sigma/MC", "r68/MC", "profile RMS"],
            cell_table_rows,
            cls="compact",
        )
        + "<h3>Nhit-group fallback fits</h3>"
        + v3.table(
            ["Nhit", "cells", "cell ids", "N_on", "B_on", "excess", "sig", "gate", "sigma_obs", "r68_obs"],
            group_table_rows,
            cls="compact",
        )
        + "<h3>Low-stat and PSF-risk notes</h3>"
        + f"<p>Cells failing the single-cell gate are kept in the plots but should not drive a PSF conclusion by themselves. Historical PSF-risk cells in the current v4 selector: {risk_text}.</p>"
        "<p>Because this is an observed effective PSF, it contains the true Crab spectrum, energy migration, zenith distribution, cell selection, and residual background-model effects. It should not be used to replace MC effective area or energy dispersion directly.</p>"
    )


def css() -> str:
    return """
    :root { color-scheme: light; --ink:#111827; --muted:#4b5563; --line:#d1d5db; --soft:#f3f4f6; --accent:#2563eb; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; color:var(--ink); background:#fff; line-height:1.45; }
    header { padding:28px 36px 22px; border-bottom:1px solid var(--line); background:#f9fafb; }
    main { padding:24px 36px 44px; max-width:1240px; margin:0 auto; }
    h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; }
    h2 { margin:0 0 14px; font-size:21px; border-bottom:1px solid var(--line); padding-bottom:6px; }
    h3 { margin:0 0 8px; font-size:16px; }
    p { margin:8px 0 12px; color:var(--muted); }
    code { background:#eef2ff; border:1px solid #c7d2fe; border-radius:4px; padding:1px 4px; font-size:12px; }
    .section { margin:0 0 30px; }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:14px 0 18px; }
    .card { border:1px solid var(--line); border-radius:8px; padding:13px 14px; background:#fff; min-width:0; }
    .card .k { color:#6b7280; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
    .card .v { font-size:20px; font-weight:650; margin-top:5px; }
    .card p { font-size:12.5px; margin-bottom:0; }
    table { border-collapse:collapse; width:100%; margin:12px 0 16px; font-size:13px; }
    th, td { border:1px solid var(--line); padding:7px 8px; vertical-align:top; }
    th { background:var(--soft); text-align:left; }
    table.compact { font-size:12px; }
    table.compact th, table.compact td { padding:5px 6px; }
    .grid2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; align-items:start; }
    .figure { margin:0 0 16px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }
    .figure img { width:100%; display:block; background:#fff; }
    figcaption { padding:10px 12px 12px; color:var(--muted); font-size:12.5px; border-top:1px solid var(--line); }
    .missing { padding:14px; color:#991b1b; border-color:#fecaca; background:#fff7f7; }
    .pill { display:inline-block; padding:1px 6px; border-radius:999px; color:#1d4ed8; background:#dbeafe; font-size:11px; font-weight:600; }
    .note { padding:11px 13px; border-left:4px solid var(--accent); background:#eff6ff; color:#1e3a8a; margin:12px 0 16px; }
    details { margin:10px 0 16px; }
    summary { cursor:pointer; color:#1d4ed8; font-weight:600; }
    @media (max-width: 900px) { main, header { padding-left:18px; padding-right:18px; } .cards, .grid2 { grid-template-columns:1fr; } }
    """


def build_report() -> None:
    a_meta = v3.load_json(PRIMARY_STAGE_A_META)
    b_meta = v3.load_json(v3.STAGE_B_META)
    c_meta = v3.load_json(v3.STAGE_C_META)
    d_meta = v3.load_json(v3.STAGE_D_META)
    e_meta = v3.load_json(PRIMARY_STAGE_E_META)
    active_f_meta = v3.load_json(v3.STAGE_F_META)
    legacy_f_meta = v3.load_json(DROP4_STAGE_F_META)
    f_meta = v3.load_json(PRIMARY_STAGE_F_META)
    g_meta = v3.load_json(PRIMARY_STAGE_G_META)
    old_g_meta = v3.load_json(v3.OLD_STAGE_G_META)
    r68_b_meta = v3.load_json(R68_STAGE_B_META) if R68_STAGE_B_META.exists() else {}
    r68_e_meta = v3.load_json(R68_STAGE_E_META) if R68_STAGE_E_META.exists() else {}
    r68_f_meta = v3.load_json(R68_STAGE_F_META) if R68_STAGE_F_META.exists() else {}
    r68_g_meta = v3.load_json(R68_STAGE_G_META) if R68_STAGE_G_META.exists() else {}
    fit_rows = selector_rows_from(DROP4_SELECTOR_CSV)

    empirical_psf.build_diagnostics(
        stage_f_metadata=PRIMARY_STAGE_F_META,
        output_dir=EMPIRICAL_PSF_DIR,
    )
    cell_crossmatch_rows = build_cell_root_cause_crossmatch()
    plot_v4_final_sed(g_meta, old_g_meta)
    plot_response_contract_external_overlay(g_meta, f_meta)
    if r68_g_meta:
        plot_r68_sed_comparison(g_meta, r68_g_meta)

    intro = (
        '<div class="note">'
        "This v4 report now uses the aperture-conditioned response as the primary result: Stage A counts only MC events with "
        "<code>mc_dangle <= r_opt</code>, and Stage F/G use a Stage E signal clone with <code>containment_r_opt=1</code>. "
        "The same annulus-normalized Stage D/E excess is used; the changed contract is the response used to convert flux into expected counts."
        "</div>"
        + v4_summary_cards(e_meta, f_meta, g_meta, d_meta)
    )

    cells_body = (
        "<p>The tested cell set starts from the active30 v3_baseline_psfborrow selector but excludes cells <code>4, 17, 39, 43</code>. "
        "Cells 52 and 65 remain included through neighbor PSF borrowing; cell 39 is excluded in this control.</p>"
        f"<p><strong>Included cell ids:</strong> {v3.esc(', '.join(str(row.get('cell_id')) for row in fit_rows))}</p>"
        "<details open><summary>Fit-cell selector table</summary>"
        + fit_cell_table_from(fit_rows)
        + "</details>"
    )

    stage_d_body = (
        v3.current_background_section(d_meta)
        + '<div class="grid2">'
        + v3.figure(v3.STAGE_D_DIR / "roi_counts_grid.png", "Stage D counts map grid", "Observed counts in the 6 deg Crab ROI for the candidate grid.")
        + v3.figure(v3.STAGE_D_DIR / "annulus_training_mask_grid.png", "Annulus training mask grid", "Grey/colored ring pixels show where the 2D surface is trained; the central Crab/source mask is excluded from training.")
        + v3.figure(v3.STAGE_D_DIR / "roi_background_grid.png", "Fitted 2D background surface grid", "The final annulus-normalized 2D background map B_final(x,y) for each cell.")
        + v3.figure(v3.STAGE_D_DIR / "annulus_residual_grid.png", "Annulus residual grid", "Training-ring residuals after the surface fit.")
        + v3.figure(v3.STAGE_D_DIR / "core_background_grid.png", "Core extrapolated background grid", "The fitted background extrapolated into the Crab core/on region used for B_on.")
        + v3.figure(v3.STAGE_D_DIR / "roi_excess_grid.png", "Stage D counts minus fitted 2D background", "Counts map after subtracting the annulus-normalized fitted 2D background.")
        + v3.figure(v3.ASSET_DIR / "v3_annnorm_surface_scale_grid.png", "Annulus surface scale grid", "Per-cell multiplicative scale forcing the annulus-integrated fitted surface to match annulus observed counts.")
        + v3.figure(v3.ASSET_DIR / "v3_annnorm_dec_profile_comparison.png", "Before/after Dec profile comparison", "Unnormalized Dec-offset profiles summed over the active fit cells.")
        + "</div>"
    )

    stage_e_totals = e_meta.get("totals", {}) if isinstance(e_meta.get("totals"), dict) else {}
    stage_e_body = (
        v3.table(
            ["N_on", "B_on", "excess", "sqrt(N_on+B_on)", "formal sigma", "note"],
            [
                [
                    v3.fmt_int(stage_e_totals.get("N_on")),
                    v3.fmt(stage_e_totals.get("B_on"), 6),
                    v3.fmt(stage_e_totals.get("excess"), 6),
                    v3.fmt(math.sqrt(float(stage_e_totals.get("N_on", 0) or 0) + float(stage_e_totals.get("B_on", 0) or 0)), 5),
                    v3.fmt(stage_e_totals.get("formal_sigma"), 5),
                    "direct expectation background; no Li-Ma alpha/N_off",
                ]
            ],
            cls="compact",
        )
        + '<div class="grid2">'
        + v3.figure(v3.STAGE_E_DIR / "on_background_grid.png", "Stage E on/background grid", "Per-cell N_on and B_on entering the latest v3 fit.")
        + v3.figure(v3.STAGE_E_DIR / "excess_grid.png", "Stage E excess grid", "Per-cell N_on - B_on using the latest annulus-normalized background.")
        + v3.figure(v3.STAGE_E_DIR / "known_b_sigma_grid.png", "Stage E known-background sigma grid", "Per-cell known-background Poisson diagnostic significance.")
        + v3.figure(v3.STAGE_E_DIR / "on_over_background_grid.png", "Stage E N_on / B_on grid", "Ratio view to expose local under/over-background behavior.")
        + "</div>"
    )

    stage_f_body = (
        "<p>The preferred spectrum is selected by the Stage F metadata. This is the primary v4 fit using the aperture-conditioned Stage A response and downstream <code>containment_r_opt=1</code>.</p>"
        + v3.stage_f_table(f_meta)
        + '<div class="grid2">'
        + v3.figure(PRIMARY_STAGE_F_DIR / "model_counts_vs_excess.png", "Stage F aperture-response model counts versus excess", "Fit-cell excess compared with the preferred spectral model expectation under the new response contract.")
        + v3.figure(PRIMARY_STAGE_F_DIR / "pull_grid_logpar.png", "Stage F aperture-response LogPar pull grid", "Per-cell residual pull under the current preferred LogPar fit.")
        + v3.figure(PRIMARY_STAGE_F_DIR / "theta_exposure.png", "Stage F aperture-response theta exposure", "Zenith-angle exposure diagnostic used by the forward-folding response.")
        + v3.figure(PRIMARY_STAGE_F_DIR / "pull_grid_pl.png", "Stage F aperture-response PL pull grid", "Power-law residual grid kept as a model-comparison diagnostic.")
        + "</div>"
    )

    stage_g_body = (
        "<p>Stage G fixes the Stage F preferred spectral shape and refits only one normalization per diagnostic energy grouping. "
        "These are the primary v4 SED points from the aperture-conditioned response branch.</p>"
        + v3.stage_g_table(g_meta)
        + '<div class="grid2">'
        + v3.figure(PRIMARY_STAGE_G_DIR / "sed_points_stage_f_fullarray_pool1.png", "Primary v4 aperture-response Stage G SED points", "Native Stage G diagnostic plot from v4_stage_g_aperture_conditioned_drop4.")
        + v3.figure(PRIMARY_STAGE_G_DIR / "sed_points_ratio.png", "Primary v4 aperture-response Stage G ratio plot", "Diagnostic ratios to the frozen Stage F model / reference curves.")
        + v3.figure(PRIMARY_STAGE_G_DIR / "sed_point_cell_counts.png", "Primary v4 aperture-response Stage G cell counts per point", "Which fit cells enter each diagnostic SED point.")
        + v3.figure(V4_FINAL_SED_PNG, "V4 final SED comparison with old v3 reference", "Blue/green markers are the primary aperture-response v4 points. Grey open markers are the previous v3 psfborrow points. Black/brown markers are official pass5/tutorial WCDA references.")
        + "</div>"
    )

    body = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Crab SED v4 aperture-conditioned response report</title>"
        f"<style>{css()}</style></head><body>"
        "<header><h1>Crab SED v4 Aperture-Conditioned Response Report</h1>"
        "<p>Primary result: drop4 cells with aperture-conditioned Stage A response and downstream containment fixed to one.</p></header><main>"
        + v3.section("V4 Result Summary", intro)
        + v3.section("Primary Official Pass5 Forward-Fold Test", forward_fold_section())
        + v3.section("Root-Cause Diagnostics", root_cause_diagnostics_section())
        + v3.section("Cell-Level Localization", cell_root_cause_crossmatch_section(cell_crossmatch_rows))
        + v3.section("Response / Containment Audit", response_audit_section())
        + v3.section("Legacy Cell-Selection Bias Control", active30_vs_drop4_section(active_f_meta, legacy_f_meta))
        + v3.section("Observed PSF Diagnostics", empirical_psf_section())
        + v3.section(
            "R68 Empirical Aperture Control",
            r68_aperture_section(
                nominal_e_meta=e_meta,
                nominal_f_meta=legacy_f_meta,
                nominal_g_meta=v3.load_json(DROP4_STAGE_G_META),
                r68_b_meta=r68_b_meta,
                r68_e_meta=r68_e_meta,
                r68_f_meta=r68_f_meta,
                r68_g_meta=r68_g_meta,
            ),
        )
        + v3.section("Current A-G Inputs", v4_stage_table(a_meta, b_meta, c_meta, d_meta, e_meta, f_meta, g_meta))
        + v3.section("Fit Cell Definition", cells_body)
        + v3.section("Stage B / PSF Diagnostics", psf_diagnostics_section(fit_rows))
        + v3.section("Stage D: Latest 2D Background", stage_d_body)
        + v3.section("Profile Diagnostics: Latest Background", v3.profile_diagnostics_section())
        + v3.section("Stage E: Current Excess", stage_e_body)
        + v3.section("Stage F: Current Forward-Folding Fit", stage_f_body)
        + v3.section("Stage G: Current SED Points", stage_g_body)
        + "</main></body></html>"
    )
    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {V4_FINAL_SED_PNG}")


if __name__ == "__main__":
    build_report()
