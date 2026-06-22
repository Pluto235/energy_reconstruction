#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "apply" / "report"
REPORT_PATH = REPORT_DIR / "crab_sed_v3_stage_a_to_g_report.html"
ASSET_DIR = REPORT_DIR / "assets" / "v3-annnorm"
PSFBORROW_ASSET_DIR = REPORT_DIR / "assets" / "v3-psfborrow"

STAGE_A_META = REPO_ROOT / "apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json"
STAGE_B_META = REPO_ROOT / "apply/output/stage_b_v3_candidate_directpsf/runs/v3_directpsf_from_psfborrow/psf_v3_candidate_metadata.json"
STAGE_B_NOMINAL_DIR = REPO_ROOT / "apply/output/stage_b_v3_candidate/runs/slurm_42023"
STAGE_B_PSF_DIR = REPO_ROOT / "apply/output/stage_b_v3_candidate_directpsf/runs/v3_directpsf_from_psfborrow"
STAGE_C_META = REPO_ROOT / "apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024/obs_events_metadata.json"
STAGE_D_DIR = REPO_ROOT / "apply/output/stage_d_v3_candidate_annnorm/runs/v3_stage_d_annnorm_from_directpsf"
STAGE_E_DIR = REPO_ROOT / "apply/output/stage_e_v3_candidate_annnorm/runs/v3_stage_e_annnorm_from_directpsf"
STAGE_F_DIR = REPO_ROOT / "apply/output/stage_f_v3_baseline_annnorm/runs/v3_stage_f_annnorm_from_directpsf"
STAGE_G_DIR = REPO_ROOT / "apply/output/stage_g_v3_baseline_annnorm/runs/v3_stage_g_annnorm_from_directpsf"

STAGE_D_META = STAGE_D_DIR / "background_v3_candidate_annnorm_metadata.json"
STAGE_E_META = STAGE_E_DIR / "signal_v3_candidate_annnorm_metadata.json"
STAGE_F_META = STAGE_F_DIR / "fit_v3_baseline_annnorm_metadata.json"
STAGE_G_META = STAGE_G_DIR / "sed_points_v3_baseline_annnorm_metadata.json"
STAGE_B_PSF_SUMMARY = STAGE_B_PSF_DIR / "psf_v3_candidate_summary.csv"
OLD_STAGE_G_META = (
    REPO_ROOT
    / "apply/output/stage_g_v3_baseline_psfborrow/runs/v3_stage_g_psfborrow_slurm_42029/"
    / "sed_points_v3_baseline_psfborrow_metadata.json"
)

SELECTOR_CSV = REPO_ROOT / "apply/config/cell_selector_v3_baseline_directpsf.csv"
PASS5_CSV = REPORT_DIR / "assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
V099_CSV = REPORT_DIR / "assets/official-v099/wcda_crab_sed_v099_20250731_20260616_123624.csv"
SUMMARY_JSON = ASSET_DIR / "v3_annnorm_summary.json"
FINAL_SED_PNG = ASSET_DIR / "v3_annnorm_final_sed_with_official_and_old_v3.png"
PROFILE_DIR = REPORT_DIR / "assets/crab-v3-annnorm-fit-cell-profiles"
PROFILE_PREFIX = "crab_v3_annnorm_fit"
DIRECT_PSF_PROFILE_PNG = PSFBORROW_ASSET_DIR / "v3_active_fit_cell_directpsf_profiles_normalized.png"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def fmt_int(value: Any) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number:,.0f}"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def rel(path: Path) -> str:
    return os.path.relpath(path, REPORT_DIR).replace(os.sep, "/")


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def interval_key(label: Any) -> float:
    text = str(label or "").strip()
    if text.startswith("[") and "," in text:
        try:
            return float(text[1:].split(",", 1)[0])
        except ValueError:
            return 1.0e9
    if text.startswith(">="):
        try:
            return float(text[2:])
        except ValueError:
            return 1.0e9
    if text.startswith(">"):
        try:
            return float(text[1:])
        except ValueError:
            return 1.0e9
    return 1.0e9


def points(meta: dict[str, Any], grouping: str, *, sort_by: str = "energy") -> list[dict[str, Any]]:
    rows = [row for row in meta.get("points", []) if isinstance(row, dict) and row.get("grouping") == grouping]
    if sort_by == "label":
        rows.sort(key=lambda row: interval_key(row.get("group_label")))
    else:
        rows.sort(key=lambda row: finite_float(row.get("effective_energy_tev")) or 0.0)
    return rows


def point_arrays(meta: dict[str, Any], grouping: str) -> tuple[list[float], list[float], list[float]]:
    energy: list[float] = []
    flux: list[float] = []
    err: list[float] = []
    for row in points(meta, grouping):
        e = finite_float(row.get("effective_energy_tev"))
        y = finite_float(row.get("E2_dnde"))
        dy = finite_float(row.get("E2_dnde_err"))
        if e is None or y is None or dy is None or e <= 0 or y <= 0:
            continue
        energy.append(e)
        flux.append(y)
        err.append(dy)
    return energy, flux, err


def pass5_points() -> tuple[list[float], list[float]]:
    energy: list[float] = []
    flux: list[float] = []
    for row in read_csv_rows(PASS5_CSV):
        e = finite_float(row.get("energy_tev"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        if e is None or dnde is None or e <= 0 or dnde <= 0:
            continue
        energy.append(e)
        flux.append(e * e * dnde)
    return energy, flux


def v099_points() -> tuple[list[float], list[float], list[float], list[float]]:
    energy: list[float] = []
    flux: list[float] = []
    err_low: list[float] = []
    err_high: list[float] = []
    for row in read_csv_rows(V099_CSV):
        e = finite_float(row.get("energy_tev"))
        y = finite_float(row.get("e2_flux_scaled_1e14_tev_cm2_s"))
        lo = finite_float(row.get("e2_flux_err_low_scaled_1e14"))
        hi = finite_float(row.get("e2_flux_err_high_scaled_1e14"))
        if e is None or y is None or e <= 0 or y <= 0:
            continue
        energy.append(e)
        flux.append(y * 1.0e-14)
        err_low.append((lo or 0.0) * 1.0e-14)
        err_high.append((hi or 0.0) * 1.0e-14)
    return energy, flux, err_low, err_high


def plot_final_sed(current_meta: dict[str, Any], old_meta: dict[str, Any]) -> Path:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)

    e_pass5, y_pass5 = pass5_points()
    if e_pass5:
        ax.plot(e_pass5, y_pass5, "o", ms=5.5, color="#111827", label="Official pass5 WCDA")

    e_v099, y_v099, ylo_v099, yhi_v099 = v099_points()
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
        ("nhit", "o", "#2563eb", "latest bkg v3 Nhit points"),
        ("predE", "D", "#059669", "latest bkg v3 predE points"),
    ]:
        e, y, dy = point_arrays(current_meta, grouping)
        if e:
            ax.errorbar(
                e,
                y,
                yerr=dy,
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
        e, y, dy = point_arrays(old_meta, grouping)
        if e:
            ax.errorbar(
                e,
                y,
                yerr=dy,
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
    ax.set_title("Crab SED: latest annulus-normalized bkg versus official/tutorial references")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.2, ncol=1)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_SED_PNG)
    plt.close(fig)
    return FINAL_SED_PNG


def tag(text: str, cls: str = "") -> str:
    class_attr = f' class="{esc(cls)}"' if cls else ""
    return f"<span{class_attr}>{esc(text)}</span>"


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]], cls: str = "") -> str:
    class_attr = f' class="{esc(cls)}"' if cls else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def figure(path: Path, title: str, caption: str) -> str:
    if not exists(path):
        return (
            '<div class="figure missing">'
            f"<h3>{esc(title)}</h3><p>Missing asset: <code>{esc(str(path))}</code></p></div>"
        )
    return (
        '<figure class="figure">'
        f'<img src="{esc(rel(path))}" alt="{esc(title)}">'
        f"<figcaption><strong>{esc(title)}</strong><br>{esc(caption)}</figcaption>"
        "</figure>"
    )


def section(title: str, body: str) -> str:
    return f'<section class="section"><h2>{esc(title)}</h2>{body}</section>'


def selector_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(SELECTOR_CSV)
    return [row for row in rows if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y"}]


def fit_cell_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        borrowed = row.get("psf_borrowed_from") or ""
        direct = str(row.get("psf_direct_own_cell", "")).strip().lower() in {"1", "true", "yes", "y"}
        if direct:
            psf_note = "direct own-cell"
        elif borrowed:
            psf_note = f"borrowed from {esc(borrowed)}"
        else:
            psf_note = "native"
        body.append(
            [
                esc(row.get("cell_id")),
                esc(row.get("nhit_bin")),
                esc(row.get("predE_bin")),
                fmt_int(row.get("mc_count")),
                fmt(row.get("ridge_peak_fraction"), 3),
                esc(psf_note),
                esc(row.get("subset_reason")),
            ]
        )
    return table(
        ["cell", "Nhit", "predE", "MC count", "ridge frac.", "PSF", "selection reason"],
        body,
        cls="compact",
    )


def active_psf_table(fit_rows: list[dict[str, str]]) -> str:
    psf_rows = {str(row.get("cell_id")): row for row in read_csv_rows(STAGE_B_PSF_SUMMARY)}
    selector_rows_by_id = {str(row.get("cell_id")): row for row in fit_rows}
    rows = []
    for cell_id in selector_rows_by_id:
        psf = psf_rows.get(cell_id, {})
        selector = selector_rows_by_id[cell_id]
        borrowed = str(psf.get("psf_borrowed", "")).strip().lower() in {"1", "true", "yes", "y"}
        direct = str(psf.get("psf_direct_own_cell", "")).strip().lower() in {"1", "true", "yes", "y"}
        if direct:
            source = "direct own-cell PSF"
            previous = psf.get("previous_psf_source")
            if previous:
                source += f"; previous {previous}"
        elif borrowed:
            source = f"borrowed from {psf.get('borrowed_from') or selector.get('psf_borrowed_from')}"
            method = psf.get("borrow_method") or selector.get("psf_borrow_method")
            weights = psf.get("borrow_weights")
            if method:
                source += f" ({method})"
            if weights:
                source += f"; weights {weights}"
        else:
            source = "direct Stage B PSF"
        rows.append(
            [
                esc(cell_id),
                esc(selector.get("nhit_bin") or psf.get("nhit_bin")),
                esc(selector.get("predE_bin") or psf.get("predE_bin")),
                fmt(psf.get("sigma_deg"), 4),
                fmt(psf.get("r_opt_deg"), 4),
                fmt(psf.get("containment_r_opt"), 4),
                fmt(psf.get("effective_events"), 5),
                fmt(psf.get("theta_missing_crab_probability_mass"), 5),
                esc(source),
                fmt(psf.get("original_theta_missing_crab_probability_mass"), 5),
            ]
        )
    return table(
        [
            "cell",
            "Nhit",
            "predE",
            "sigma deg",
            "r_opt deg",
            "containment",
            "Neff",
            "missing mass",
            "PSF source",
            "orig missing",
        ],
        rows,
        cls="compact",
    )


def plot_active_direct_psf_profiles(fit_rows: list[dict[str, str]]) -> Path | None:
    psf_npz = STAGE_B_PSF_DIR / "psf_v3_candidate.npz"
    if not exists(psf_npz):
        return None
    try:
        import numpy as np
    except Exception:
        return None

    plt = setup_matplotlib()
    with np.load(psf_npz, allow_pickle=False) as data:
        cell_ids = data["cell_id"].astype(int)
        profile_edges = data["profile_edges_deg"].astype(float)
        profile_density = data["profile_density"].astype(float)

    centers = 0.5 * (profile_edges[:-1] + profile_edges[1:])
    id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    psf_rows = {str(row.get("cell_id")): row for row in read_csv_rows(STAGE_B_PSF_SUMMARY)}
    selected_ids = [
        int(row["cell_id"])
        for row in fit_rows
        if str(row.get("cell_id", "")).strip().isdigit() and int(row["cell_id"]) in id_to_index
    ]
    if not selected_ids:
        return None

    ncols = 5
    nrows = int(math.ceil(len(selected_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 2.1 * nrows), dpi=160, sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.set_visible(False)

    curve_color = "#1f77b4"
    title_color = "#111827"
    for ax, cell_id in zip(axes_arr, selected_ids):
        ax.set_visible(True)
        selector_row = next((row for row in fit_rows if str(row.get("cell_id")) == str(cell_id)), {})
        psf_row = psf_rows.get(str(cell_id), {})
        values = profile_density[id_to_index[cell_id]].copy()
        peak = float(np.nanmax(values)) if values.size else 0.0
        if np.isfinite(peak) and peak > 0.0:
            values /= peak

        ax.plot(centers, values, color=curve_color, lw=1.25, label="own cell")
        fit_sigma = finite_float(psf_row.get("sigma_deg"))
        if fit_sigma is not None:
            ax.axvline(fit_sigma, color="#6b7280", lw=1.0, ls="--", alpha=0.9, label="fit PSF sigma")

        direct_own = str(psf_row.get("psf_direct_own_cell", "")).strip().lower() in {"1", "true", "yes", "y"}
        source_note = "direct own-cell" if direct_own else "direct Stage B"
        missing_mass = finite_float(psf_row.get("theta_missing_crab_probability_mass"))
        if direct_own and missing_mass is not None:
            source_note += f"\nmissing theta {missing_mass:.3f}"

        ax.text(
            0.98,
            0.92,
            source_note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.86},
        )
        ax.set_title(
            f"{cell_id} {selector_row.get('nhit_bin', psf_row.get('nhit_bin', ''))}\n"
            f"{selector_row.get('predE_bin', psf_row.get('predE_bin', ''))}",
            fontsize=7.0,
            color=title_color,
        )
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1.08)
        ax.grid(alpha=0.22, lw=0.45)

    visible_axes = [ax for ax in axes_arr if ax.get_visible()]
    if visible_axes:
        handles, labels = visible_axes[0].get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        visible_axes[0].legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=6.4, frameon=True)
    for ax in axes_arr[-ncols:]:
        if ax.get_visible():
            ax.set_xlabel("offset angle [deg]", fontsize=7)
    for row_idx in range(nrows):
        ax = axes_arr[row_idx * ncols]
        if ax.get_visible():
            ax.set_ylabel("own-cell peak-normalized density", fontsize=7)
    fig.suptitle("Active 30-cell own-cell normalized radial profiles (direct PSF)", fontsize=12)
    fig.tight_layout()
    DIRECT_PSF_PROFILE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIRECT_PSF_PROFILE_PNG)
    plt.close(fig)
    return DIRECT_PSF_PROFILE_PNG


def psf_diagnostics_section(fit_rows: list[dict[str, str]]) -> str:
    return (
        "<p>These Stage B diagnostics are restored because the PSF construction is upstream of the latest annulus-normalized Stage D background. "
        "The current mainline uses the direct-own-cell PSF variant for cells 39/52/65, while retaining the latest annnorm background for Stage D/E/F/G. "
        "The neighbor-borrowing result is kept only as an old comparison branch.</p>"
        '<div class="grid2">'
        + figure(
            DIRECT_PSF_PROFILE_PNG,
            "Active 30-cell own-cell normalized radial profiles",
            "Each selected cell's own MC radial distribution is normalized by its peak so the PSF widths can be compared. All panels use the same blue style because cells 39/52/65 now use direct own-cell PSFs in the active branch instead of borrowed PSFs.",
        )
        + figure(
            PSFBORROW_ASSET_DIR / "v3_active_fit_cell_theta_profiles.png",
            "Active 30-cell normalized MC theta profiles",
            "Colored curves are each selected cell's own MC theta support after the Stage B cuts; gray is the Crab-visible theta target used for reweighting. Orange missing-support bins remain the audit trail for 39/52/65.",
        )
        + figure(
            REPO_ROOT
            / "apply/output/stage_b_v3_candidate_direct_ownpsf_focus/diagnostics/direct_owncell_vs_borrowed_psf_cells_39_52_65.png",
            "Direct own-cell versus borrowed PSF for cells 39/52/65",
            "Focused comparison used for the decision to switch the active v3 branch from neighbor borrowing to direct own-cell PSFs.",
        )
        + figure(
            STAGE_B_NOMINAL_DIR / "psf_sigma_deg_grid.png",
            "Stage B PSF sigma grid",
            "Candidate-grid Rayleigh-core PSF width sigma in degrees. Smaller sigma means a narrower reconstructed Crab response for that cell.",
        )
        + figure(
            STAGE_B_NOMINAL_DIR / "psf_r_opt_deg_grid.png",
            "Stage B PSF r_opt grid",
            "Candidate-grid aperture radius used for the on-region integration. In v3 this is tied to the fitted PSF width, approximately r_opt = 1.58 * sigma.",
        )
        + figure(
            STAGE_B_NOMINAL_DIR / "psf_containment_grid.png",
            "Stage B PSF containment at r_opt grid",
            "Fraction of the cell PSF contained inside r_opt. Low containment or warnings indicate broad tails or fragile low-stat PSF behavior.",
        )
        + figure(
            STAGE_B_NOMINAL_DIR / "psf_effective_events_grid.png",
            "Stage B PSF effective-events grid",
            "Effective MC statistics after Crab-declination theta reweighting, Neff = (sum w)^2 / sum(w^2). Low Neff means the PSF is dominated by a small number of weighted MC events.",
        )
        + figure(
            STAGE_B_NOMINAL_DIR / "psf_radial_profiles_grid.png",
            "Stage B candidate-grid radial PSF profiles",
            "Full candidate-grid radial PSF profile diagnostic from nominal Stage B. This is retained as PSF-computation provenance for the active selector.",
        )
        + "</div>"
        "<h3>Active fit-cell PSF table</h3>"
        + active_psf_table(fit_rows)
        + "<h3>PSF theta-support notes</h3>"
        "<p><code>theta_missing_crab_probability_mass</code> is the fraction of the Crab-visible theta exposure for which a cell has no conditional MC support after the cell, true-energy, finite-angle, and positive-weight cuts. "
        "It is therefore a coverage diagnostic for the PSF reweighting, not a statement that the global MC sample is small.</p>"
        "<p>Cells <code>39/52/65</code> are ridge-left physical candidates with visible Crab excess, but their own-cell theta support misses too much Crab theta probability mass. "
        "The active directpsf branch keeps these cells in the 30-cell fit selector and uses their direct own-cell PSFs despite incomplete theta support. "
        "This makes the aperture/background integration self-contained per cell; the previous neighbor-borrowing PSFs are retained only as a systematic comparison.</p>"
    )


def stage_table(
    a_meta: dict[str, Any],
    b_meta: dict[str, Any],
    c_meta: dict[str, Any],
    d_meta: dict[str, Any],
    e_meta: dict[str, Any],
    f_meta: dict[str, Any],
    g_meta: dict[str, Any],
) -> str:
    rows = [
        ["A", "2D response", esc(a_meta.get("response_type", "primary_thrown_response")), esc(STAGE_A_META.relative_to(REPO_ROOT))],
        ["B", "direct own-cell PSF for 39/52/65", esc(b_meta.get("run_id")), esc(STAGE_B_META.relative_to(REPO_ROOT))],
        ["C", "observation event reduction", esc(c_meta.get("run_id")), esc(STAGE_C_META.relative_to(REPO_ROOT))],
        ["D", "annulus-normalized 2D background", esc(d_meta.get("run_id")), esc(STAGE_D_META.relative_to(REPO_ROOT))],
        ["E", "on-region excess from latest bkg", esc(e_meta.get("run_id")), esc(STAGE_E_META.relative_to(REPO_ROOT))],
        ["F", "global forward-folding fit", esc(f_meta.get("run_id")), esc(STAGE_F_META.relative_to(REPO_ROOT))],
        ["G", "diagnostic SED points", esc(g_meta.get("run_id")), esc(STAGE_G_META.relative_to(REPO_ROOT))],
    ]
    return table(["Stage", "current role", "run / type", "metadata"], rows, cls="compact")


def summary_cards(e_meta: dict[str, Any], f_meta: dict[str, Any], g_meta: dict[str, Any], d_meta: dict[str, Any]) -> str:
    totals = e_meta.get("totals", {}) if isinstance(e_meta.get("totals"), dict) else {}
    preferred = f_meta.get("preferred_fit", {}) if isinstance(f_meta.get("preferred_fit"), dict) else {}
    key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
    fit = f_meta.get("fits", {}).get(key, {}) if isinstance(f_meta.get("fits"), dict) else {}
    params = fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {}
    cells = g_meta.get("validation", {}).get("required_cell_ids", []) if isinstance(g_meta.get("validation"), dict) else []
    return (
        '<div class="cards">'
        f'<div class="card"><div class="k">latest background</div><div class="v">B_on {fmt(totals.get("B_on"), 6)}</div><p>N_on {fmt_int(totals.get("N_on"))}; excess {fmt(totals.get("excess"), 6)}</p></div>'
        f'<div class="card"><div class="k">detection diagnostic</div><div class="v">{fmt(totals.get("known_b_sigma_aggregate"), 4)} sigma</div><p>known-background Poisson aggregate; Li-Ma is not defined for direct expectation bkg.</p></div>'
        f'<div class="card"><div class="k">Stage F preferred fit</div><div class="v">{esc(preferred.get("model"))}</div><p>phi0 {fmt(params.get("phi0"), 5)}, alpha/gamma {fmt(params.get("alpha", params.get("gamma")), 5)}, beta {fmt(params.get("beta"), 5)}; chi2/ndof {fmt(fit.get("chi2"), 4)}/{fmt(fit.get("ndof"), 3)}</p></div>'
        f'<div class="card"><div class="k">fit cells / SED points</div><div class="v">{len(cells)} cells / {len(g_meta.get("points", []))} points</div><p>Stage D active-fit warnings: {len(d_meta.get("quality", {}).get("active_fit_warning_cell_ids", [])) if isinstance(d_meta.get("quality"), dict) else "n/a"}</p></div>'
        "</div>"
    )


def current_background_section(d_meta: dict[str, Any]) -> str:
    model = d_meta.get("background_model", {}) if isinstance(d_meta.get("background_model"), dict) else {}
    roi = d_meta.get("roi", {}) if isinstance(d_meta.get("roi"), dict) else {}
    rows = [
        ["ROI", f"rho < {fmt(roi.get('fiducial_radius_deg'), 3)} deg around Crab tangent-plane center"],
        ["training ring", f"inner radius is max({fmt(model.get('annulus_default_inner_deg'), 3)} deg, source mask); width {fmt(model.get('annulus_width_deg'), 3)} deg; capped by {fmt(model.get('annulus_max_inner_deg'), 3)} deg"],
        ["raw surface", "B_raw(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2"],
        ["positivity", "B_pos(x,y) = max(B_raw(x,y), 0)"],
        ["annulus normalization", "scale_b = sum_annulus(counts_b) / sum_annulus(B_pos,b)"],
        ["final bkg", "B_final,b(x,y) = scale_b * B_pos,b(x,y); B_on,b is the integral over the PSF on aperture"],
    ]
    body = (
        "<p>The current report is rebuilt around the annulus-normalized quadratic 2D background. "
        "The quadratic fit keeps the local shape from the training ring, then forces the total fitted background in that ring to match the observed annulus counts for each cell.</p>"
        + table(["item", "definition"], [[esc(a), esc(b)] for a, b in rows], cls="compact")
    )
    return body


def stage_f_table(f_meta: dict[str, Any]) -> str:
    rows = []
    fits = f_meta.get("fits", {}) if isinstance(f_meta.get("fits"), dict) else {}
    preferred = f_meta.get("preferred_fit", {}) if isinstance(f_meta.get("preferred_fit"), dict) else {}
    preferred_key = f"{preferred.get('model')}_{preferred.get('error_mode')}"
    for key in ["pl_conservative", "logpar_conservative", "pl_sqrt_n", "logpar_sqrt_n"]:
        fit = fits.get(key)
        if not isinstance(fit, dict):
            continue
        params = fit.get("parameters", {}) if isinstance(fit.get("parameters"), dict) else {}
        mark = tag("preferred", "pill") if key == preferred_key else ""
        rows.append(
            [
                esc(key),
                esc(fit.get("model_name")),
                fmt(params.get("phi0"), 5),
                fmt(params.get("gamma", params.get("alpha")), 5),
                fmt(params.get("beta"), 5),
                fmt(fit.get("chi2"), 5),
                fmt(fit.get("ndof"), 3),
                fmt(fit.get("chi2_over_ndof"), 4),
                mark,
            ]
        )
    return table(["fit", "model", "phi0", "gamma/alpha", "beta", "chi2", "ndof", "chi2/ndof", ""], rows, cls="compact")


def stage_g_table(g_meta: dict[str, Any]) -> str:
    rows = []
    for row in points(g_meta, "nhit", sort_by="label") + points(g_meta, "predE", sort_by="label"):
        rows.append(
            [
                esc(row.get("grouping")),
                esc(row.get("group_label")),
                esc(",".join(str(v) for v in row.get("cell_ids", []))),
                fmt(row.get("effective_energy_tev"), 4),
                fmt(row.get("E2_dnde"), 5),
                fmt(row.get("E2_dnde_err"), 4),
                fmt(row.get("chi2_over_ndof"), 3),
            ]
        )
    return table(["group", "bin", "cells", "Eeff TeV", "E2 dN/dE", "err", "chi2/ndof"], rows, cls="compact")


def profile_diagnostics_section() -> str:
    counts_ra = PROFILE_DIR / f"{PROFILE_PREFIX}_ra_normalized_counts_profiles.png"
    counts_dec = PROFILE_DIR / f"{PROFILE_PREFIX}_dec_normalized_counts_profiles.png"
    excess_ra = PROFILE_DIR / f"{PROFILE_PREFIX}_ra_normalized_excess_profiles.png"
    excess_dec = PROFILE_DIR / f"{PROFILE_PREFIX}_dec_normalized_excess_profiles.png"
    return (
        "<p>These diagnostics are rebuilt from the latest annulus-normalized Stage D maps, not copied from the old v3 report. "
        "Counts profiles use <code>counts_map</code>; excess profiles use the current <code>excess_map = counts_map - B_final</code>. "
        "Each 1D profile is divided by its own positive peak so that widths and residual asymmetries can be compared across cells.</p>"
        '<div class="grid2">'
        + figure(
            counts_ra,
            "candidate-grid normalized RA-offset counts profiles",
            "For every candidate cell, counts are summed in a |Dec offset|<1 deg band and normalized by that cell's own peak. Green panels are the current 30 fit cells; light gray spans mark that cell's Stage D annulus-training radii projected onto the RA-offset axis.",
        )
        + figure(
            counts_dec,
            "candidate-grid normalized Dec-offset counts profiles",
            "For every candidate cell, counts are summed in a |RA offset cos(dec)|<1 deg band and normalized by that cell's own peak. The same annulus-ring spans are shown in gray.",
        )
        + figure(
            excess_ra,
            "candidate-grid normalized RA-offset excess profiles after latest bkg subtraction",
            "Same RA projection after subtracting the latest annulus-normalized fitted 2D background. This checks whether the Crab-centered excess remains narrow and centered after the new bkg method.",
        )
        + figure(
            excess_dec,
            "candidate-grid normalized Dec-offset excess profiles after latest bkg subtraction",
            "Same Dec projection after subtracting the latest annulus-normalized fitted 2D background. This is the direct Dec-direction residual check for the new bkg method.",
        )
        + "</div>"
    )


def build_report() -> None:
    a_meta = load_json(STAGE_A_META)
    b_meta = load_json(STAGE_B_META)
    c_meta = load_json(STAGE_C_META)
    d_meta = load_json(STAGE_D_META)
    e_meta = load_json(STAGE_E_META)
    f_meta = load_json(STAGE_F_META)
    g_meta = load_json(STAGE_G_META)
    old_g_meta = load_json(OLD_STAGE_G_META)
    summary_meta = load_json(SUMMARY_JSON) if SUMMARY_JSON.exists() else {}
    fit_rows = selector_rows()

    plot_active_direct_psf_profiles(fit_rows)
    plot_final_sed(g_meta, old_g_meta)

    css = """
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

    intro = (
        '<div class="note">'
        "This HTML is the current v3 report rebuilt around the latest annulus-normalized 2D background branch. "
        "Old-background Stage D/E/F/G result sections are removed from the main body, while upstream Stage B PSF diagnostics are retained because they define the active direct-own-cell PSF response. "
        "The previous v3 psfborrow Stage G points are kept only as grey reference markers in the final SED comparison."
        "</div>"
        + summary_cards(e_meta, f_meta, g_meta, d_meta)
    )

    cells_body = (
        "<p>The active baseline uses the same 30-cell v3 selector, evaluated with the annulus-normalized Stage D/E background. "
        "Cells 39, 52, and 65 remain included and now use direct own-cell PSFs instead of neighbor borrowing.</p>"
        f"<p><strong>Included cell ids:</strong> {esc(', '.join(str(row.get('cell_id')) for row in fit_rows))}</p>"
        "<details open><summary>Fit-cell selector table</summary>"
        + fit_cell_table(fit_rows)
        + "</details>"
    )

    stage_d_body = (
        current_background_section(d_meta)
        + '<div class="grid2">'
        + figure(STAGE_D_DIR / "roi_counts_grid.png", "Stage D counts map grid", "Observed counts in the 6 deg Crab ROI for the candidate grid.")
        + figure(STAGE_D_DIR / "annulus_training_mask_grid.png", "Annulus training mask grid", "Grey/colored ring pixels show where the 2D surface is trained; the central Crab/source mask is excluded from training.")
        + figure(STAGE_D_DIR / "roi_background_grid.png", "Fitted 2D background surface grid", "The final annulus-normalized 2D background map B_final(x,y) for each cell, after clipping negative raw-surface pixels and applying the annulus total-count scale.")
        + figure(STAGE_D_DIR / "annulus_residual_grid.png", "Annulus residual grid", "Training-ring residuals after the surface fit; this checks whether the fitted surface tracks the observed annulus counts locally.")
        + figure(STAGE_D_DIR / "core_background_grid.png", "Core extrapolated background grid", "The same fitted background extrapolated into the Crab core/on region used for B_on.")
        + figure(STAGE_D_DIR / "roi_excess_grid.png", "Stage D counts minus fitted 2D background", "Counts map after subtracting the annulus-normalized fitted 2D background; this is the current bkg-subtracted sky map.")
        + figure(ASSET_DIR / "v3_annnorm_surface_scale_grid.png", "Annulus surface scale grid", "Per-cell multiplicative scale applied so the annulus-integrated fitted surface equals annulus observed counts.")
        + figure(ASSET_DIR / "v3_annnorm_dec_profile_comparison.png", "Before/after Dec profile comparison", "Unnormalized Dec-offset profiles summed over the active fit cells; the blue line is the latest annulus-normalized counts-background profile.")
        + "</div>"
    )

    stage_e_totals = e_meta.get("totals", {}) if isinstance(e_meta.get("totals"), dict) else {}
    stage_e_body = (
        table(
            ["N_on", "B_on", "excess", "sqrt(N_on+B_on)", "formal sigma", "note"],
            [
                [
                    fmt_int(stage_e_totals.get("N_on")),
                    fmt(stage_e_totals.get("B_on"), 6),
                    fmt(stage_e_totals.get("excess"), 6),
                    fmt(math.sqrt(float(stage_e_totals.get("N_on", 0) or 0) + float(stage_e_totals.get("B_on", 0) or 0)), 5),
                    fmt(stage_e_totals.get("formal_sigma"), 5),
                    "direct expectation background; no Li-Ma alpha/N_off",
                ]
            ],
            cls="compact",
        )
        + '<div class="grid2">'
        + figure(STAGE_E_DIR / "on_background_grid.png", "Stage E on/background grid", "Per-cell N_on and B_on entering the latest v3 fit.")
        + figure(STAGE_E_DIR / "excess_grid.png", "Stage E excess grid", "Per-cell N_on - B_on using the latest annulus-normalized background.")
        + figure(STAGE_E_DIR / "known_b_sigma_grid.png", "Stage E known-background sigma grid", "Per-cell known-background Poisson diagnostic significance.")
        + figure(STAGE_E_DIR / "on_over_background_grid.png", "Stage E N_on / B_on grid", "Ratio view to expose local under/over-background behavior.")
        + "</div>"
    )

    stage_f_body = (
        "<p>The preferred spectrum is selected by the Stage F metadata. For the latest bkg branch, LogPar remains preferred over PL under conservative sqrt(N_on+B_on) errors.</p>"
        + stage_f_table(f_meta)
        + '<div class="grid2">'
        + figure(STAGE_F_DIR / "model_counts_vs_excess.png", "Stage F model counts versus excess", "Current annnorm fit-cell excess compared with the preferred spectral model expectation.")
        + figure(STAGE_F_DIR / "pull_grid_logpar.png", "Stage F LogPar pull grid", "Per-cell residual pull under the current preferred LogPar fit.")
        + figure(STAGE_F_DIR / "theta_exposure.png", "Stage F theta exposure", "Zenith-angle exposure diagnostic used by the forward-folding response.")
        + figure(STAGE_F_DIR / "pull_grid_pl.png", "Stage F PL pull grid", "Power-law residual grid kept as a model-comparison diagnostic, not the preferred fit.")
        + "</div>"
    )

    stage_g_body = (
        "<p>Stage G fixes the Stage F preferred spectral shape and refits only one normalization per diagnostic energy grouping. "
        "The table below lists the current latest-bkg points; the final figure overlays official/tutorial WCDA references and the old v3 psfborrow points only for visual comparison.</p>"
        + stage_g_table(g_meta)
        + '<div class="grid2">'
        + figure(STAGE_G_DIR / "sed_points_stage_f_fullarray_pool1.png", "Latest-bkg Stage G SED points", "Native Stage G diagnostic plot from v3_stage_g_annnorm_from_directpsf.")
        + figure(STAGE_G_DIR / "sed_points_ratio.png", "Latest-bkg Stage G ratio plot", "Diagnostic ratios to the frozen Stage F model / reference curves.")
        + figure(STAGE_G_DIR / "sed_point_cell_counts.png", "Stage G cell counts per point", "Which fit cells enter each diagnostic SED point.")
        + figure(FINAL_SED_PNG, "Final SED comparison with old v3 reference", "Blue/green markers are the latest-bkg v3 points. Grey open markers are the previous v3 psfborrow points, retained only for comparison. Black/brown markers are official pass5/tutorial WCDA references.")
        + "</div>"
    )

    diagnostic_note = ""
    if isinstance(summary_meta.get("comparison"), list):
        diagnostic_note = (
            "<p>The diagnostic summary asset is retained for provenance at "
            f"<code>{esc(SUMMARY_JSON.relative_to(REPO_ROOT))}</code>, but the HTML main flow uses the annulus-normalized branch as the sole current result.</p>"
        )

    body = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Crab SED v3 latest background report</title>"
        f"<style>{css}</style></head><body>"
        "<header><h1>Crab SED v3 Latest Background Report</h1>"
        "<p>Primary result: annulus-normalized quadratic 2D background, direct own-cell PSF for 39/52/65, Stage A-G diagnostics.</p></header><main>"
        + section("Current Result Summary", intro)
        + section("Current A-G Inputs", stage_table(a_meta, b_meta, c_meta, d_meta, e_meta, f_meta, g_meta) + diagnostic_note)
        + section("Fit Cell Definition", cells_body)
        + section("Stage B / PSF Diagnostics", psf_diagnostics_section(fit_rows))
        + section("Stage D: Latest 2D Background", stage_d_body)
        + section("Profile Diagnostics: Latest Background", profile_diagnostics_section())
        + section("Stage E: Current Excess", stage_e_body)
        + section("Stage F: Current Forward-Folding Fit", stage_f_body)
        + section("Stage G: Current SED Points", stage_g_body)
        + "</main></body></html>"
    )
    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {FINAL_SED_PNG}")


if __name__ == "__main__":
    build_report()
