#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import build_v3_latest_bkg_report as v3


REPO_ROOT = v3.REPO_ROOT
REPORT_DIR = v3.REPORT_DIR
REPORT_PATH = REPORT_DIR / "crab_sed_v4_stage_a_to_g_report.html"
V4_ASSET_DIR = REPORT_DIR / "assets" / "v4-annnorm"
V4_FINAL_SED_PNG = V4_ASSET_DIR / "v4_annnorm_final_sed_with_official_and_old_v3.png"
ACTIVE30_FORWARD_DIR = REPORT_DIR / "assets" / "v4-annnorm-normalization-diagnostics"
DROP4_FORWARD_DIR = REPORT_DIR / "assets" / "v4-drop4-normalization-diagnostics"
FORWARD_DIR = DROP4_FORWARD_DIR
DROP4_SELECTOR_CSV = REPO_ROOT / "apply/config/cell_selector_v4_drop4_psfborrow.csv"
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


def plot_v4_final_sed(current_meta: dict[str, Any], old_meta: dict[str, Any]) -> Path:
    plt = v3.setup_matplotlib()
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
        ("nhit", "o", "#2563eb", "v4 drop4 annnorm Nhit points"),
        ("predE", "D", "#059669", "v4 drop4 annnorm predE points"),
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
    ax.set_title("Crab SED v4 drop4: annnorm result versus official/tutorial references")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.2, ncol=1)
    fig.tight_layout()
    V4_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(V4_FINAL_SED_PNG)
    plt.close(fig)
    return V4_FINAL_SED_PNG


def selector_rows_from(path: Path, *, included_only: bool = True) -> list[dict[str, str]]:
    rows = v3.read_csv_rows(path)
    if not included_only:
        return rows
    return [row for row in rows if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y"}]


def fit_cell_table_from(rows: list[dict[str, str]]) -> str:
    return v3.fit_cell_table(rows)


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
    summary_rows = v3.read_csv_rows(FORWARD_DIR / "v3_official_forward_fold_summary.csv")
    nhit_rows = v3.read_csv_rows(FORWARD_DIR / "v3_official_forward_fold_nhit_summary.csv")
    cell_rows = v3.read_csv_rows(FORWARD_DIR / "v3_official_forward_fold_cell_counts.csv")

    official_summary = next((row for row in summary_rows if row.get("spectrum") == "official_pass5"), {})
    official_nhit = [row for row in nhit_rows if row.get("spectrum") == "official_pass5"]
    official_nhit.sort(key=lambda row: v3.interval_key(row.get("nhit_bin")))

    low_bins = [row for row in official_nhit if row.get("nhit_bin") in {"[125,200)", "[200,300)", "[300,500)", "[500,800)"}]
    low_ratio_text = ", ".join(
        f"<code>{v3.esc(row.get('nhit_bin'))}</code>: {v3.fmt(row.get('total_observed_over_expected'), 4)}x"
        for row in low_bins
    )

    total_table = v3.table(
        [
            "spectrum",
            "cells",
            "annnorm excess",
            "folded expected",
            "observed/expected",
            "median cell ratio",
            "cells > 1",
            "cells > 1.5",
        ],
        [
            [
                v3.esc(row.get("spectrum")),
                v3.esc(row.get("cells")),
                v3.fmt(row.get("total_excess"), 6),
                v3.fmt(row.get("total_expected_counts"), 6),
                v3.fmt(row.get("total_observed_over_expected"), 4),
                v3.fmt(row.get("median_observed_over_expected"), 4),
                v3.esc(row.get("cells_observed_over_expected_gt_1")),
                v3.esc(row.get("cells_observed_over_expected_gt_1p5")),
            ]
            for row in summary_rows
        ],
        cls="compact",
    )

    nhit_table = v3.table(
        ["Nhit bin", "cells", "annnorm excess", "folded expected", "observed/expected", "median cell ratio", "cells > 1", "cells > 1.5"],
        [
            [
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("cells")),
                v3.fmt(row.get("total_excess"), 6),
                v3.fmt(row.get("total_expected_counts"), 6),
                v3.fmt(row.get("total_observed_over_expected"), 4),
                v3.fmt(row.get("median_observed_over_expected"), 4),
                v3.esc(row.get("cells_observed_over_expected_gt_1")),
                v3.esc(row.get("cells_observed_over_expected_gt_1p5")),
            ]
            for row in official_nhit
        ],
        cls="compact",
    )

    cell_table = v3.table(
        ["cell", "Nhit", "predE", "excess", "official expected", "obs/exp", "excess - exp", "pull", "containment"],
        [
            [
                v3.esc(row.get("cell_id")),
                v3.esc(row.get("nhit_bin")),
                v3.esc(row.get("predE_bin")),
                v3.fmt(row.get("excess"), 6),
                v3.fmt(row.get("expected_counts"), 6),
                v3.fmt(row.get("observed_over_expected"), 4),
                v3.fmt(row.get("excess_minus_expected"), 5),
                v3.fmt(row.get("pull_conservative"), 4),
                v3.fmt(row.get("containment_r_opt"), 4),
            ]
            for row in ratio_cell_rows(cell_rows)
        ],
        cls="compact",
    )

    total_ratio = v3.fmt(official_summary.get("total_observed_over_expected"), 4)
    total_excess = v3.fmt(official_summary.get("total_excess"), 6)
    total_expected = v3.fmt(official_summary.get("total_expected_counts"), 6)
    return (
        "<p>This v4 test folds the official pass5 WCDA spectrum through our Stage A response, the drop4 26-cell list, "
        "cell containment, and theta exposure, then compares the predicted source counts with the latest annulus-normalized Stage E/F excess.</p>"
        '<div class="note">'
        f"Official pass5 predicts {total_expected} counts for the drop4 26-cell selector, while the latest annnorm excess is {total_excess}; "
        f"the total observed/expected ratio is {total_ratio}x. Low-Nhit bins remain high after this forward fold: {low_ratio_text}. "
        "So the low-energy discrepancy is not removed by the latest background map alone; the next suspect should be response normalization, "
        "cell-selection normalization, or energy-migration/containment consistency. The highest-Nhit ratios are less decisive because the absolute excess is small."
        "</div>"
        "<h3>Total fold summary</h3>"
        + total_table
        + "<h3>Official pass5 fold by Nhit</h3>"
        + nhit_table
        + "<h3>Largest low-Nhit cell residuals versus official pass5</h3>"
        + cell_table
        + '<div class="grid2">'
        + v3.figure(
            FORWARD_DIR / "v3_official_forward_fold_counts_vs_excess.png",
            "Drop4 official/tutorial forward-fold counts versus annnorm excess",
            "Each drop4 fit cell compares latest annnorm excess with the expected source counts from official pass5 and tutorial spectra through our response. Points above equality are cells where observed excess exceeds folded official expectation.",
        )
        + v3.figure(
            FORWARD_DIR / "v3_official_forward_fold_ratio_by_cell.png",
            "Observed/expected ratio by drop4 fit cell",
            "Cell-level ratio of latest annnorm excess to folded official/tutorial expected counts. Persistent low-Nhit ratios above 1 point to response/cell normalization rather than only a residual background-map problem.",
        )
        + "</div>"
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
    a_meta = v3.load_json(v3.STAGE_A_META)
    b_meta = v3.load_json(v3.STAGE_B_META)
    c_meta = v3.load_json(v3.STAGE_C_META)
    d_meta = v3.load_json(v3.STAGE_D_META)
    e_meta = v3.load_json(v3.STAGE_E_META)
    active_f_meta = v3.load_json(v3.STAGE_F_META)
    f_meta = v3.load_json(DROP4_STAGE_F_META)
    g_meta = v3.load_json(DROP4_STAGE_G_META)
    old_g_meta = v3.load_json(v3.OLD_STAGE_G_META)
    fit_rows = selector_rows_from(DROP4_SELECTOR_CSV)

    plot_v4_final_sed(g_meta, old_g_meta)

    intro = (
        '<div class="note">'
        "This v4 report does not replace the v3 HTML. It starts from the latest v3 annulus-normalized background result, "
        "then applies a cell-selection-bias control: cells 4, 17, 39, and 43 are removed from the original active30 fit set and Stage F/G are rerun."
        "</div>"
        + v3.summary_cards(e_meta, f_meta, g_meta, d_meta)
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
        "<p>The preferred spectrum is selected by the Stage F metadata. For the latest bkg branch, LogPar remains preferred over PL under conservative sqrt(N_on+B_on) errors.</p>"
        + v3.stage_f_table(f_meta)
        + '<div class="grid2">'
        + v3.figure(DROP4_STAGE_F_DIR / "model_counts_vs_excess.png", "Stage F drop4 model counts versus excess", "Drop4 annnorm fit-cell excess compared with the preferred spectral model expectation.")
        + v3.figure(DROP4_STAGE_F_DIR / "pull_grid_logpar.png", "Stage F drop4 LogPar pull grid", "Per-cell residual pull under the current preferred LogPar fit.")
        + v3.figure(DROP4_STAGE_F_DIR / "theta_exposure.png", "Stage F drop4 theta exposure", "Zenith-angle exposure diagnostic used by the forward-folding response.")
        + v3.figure(DROP4_STAGE_F_DIR / "pull_grid_pl.png", "Stage F drop4 PL pull grid", "Power-law residual grid kept as a model-comparison diagnostic.")
        + "</div>"
    )

    stage_g_body = (
        "<p>Stage G fixes the Stage F preferred spectral shape and refits only one normalization per diagnostic energy grouping. "
        "The final figure overlays official/tutorial WCDA references and the old v3 psfborrow points for visual comparison.</p>"
        + v3.stage_g_table(g_meta)
        + '<div class="grid2">'
        + v3.figure(DROP4_STAGE_G_DIR / "sed_points_stage_f_fullarray_pool1.png", "Drop4 Stage G SED points", "Native Stage G diagnostic plot from v4_stage_g_annnorm_drop_cells_4_17_39_43.")
        + v3.figure(DROP4_STAGE_G_DIR / "sed_points_ratio.png", "Drop4 Stage G ratio plot", "Diagnostic ratios to the frozen Stage F model / reference curves.")
        + v3.figure(DROP4_STAGE_G_DIR / "sed_point_cell_counts.png", "Drop4 Stage G cell counts per point", "Which fit cells enter each diagnostic SED point.")
        + v3.figure(V4_FINAL_SED_PNG, "V4 final SED comparison with old v3 reference", "Blue/green markers are the latest annnorm v4 points. Grey open markers are the previous v3 psfborrow points. Black/brown markers are official pass5/tutorial WCDA references.")
        + "</div>"
    )

    body = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Crab SED v4 drop4 annnorm forward-fold report</title>"
        f"<style>{css()}</style></head><body>"
        "<header><h1>Crab SED v4 Drop4 Annnorm Forward-Fold Report</h1>"
        "<p>Primary question: does removing cells 4, 17, 39, and 43 reduce the low-Nhit excess relative to official pass5 forward-fold expectations?</p></header><main>"
        + v3.section("V4 Result Summary", intro)
        + v3.section("Cell-Selection Bias Control: Active30 Versus Drop4", active30_vs_drop4_section(active_f_meta, f_meta))
        + v3.section("Official Pass5 Forward-Fold Test", forward_fold_section())
        + v3.section("Current A-G Inputs", v3.stage_table(a_meta, b_meta, c_meta, d_meta, e_meta, f_meta, g_meta))
        + v3.section("Fit Cell Definition", cells_body)
        + v3.section("Stage B / PSF Diagnostics", v3.psf_diagnostics_section(fit_rows))
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
