#!/usr/bin/env python3
from __future__ import annotations

import csv
from html.parser import HTMLParser
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "v6_64748_nhit100_highEplus1_split56"
RUN_ID = os.environ.get("V6_REPORT_RUN_ID", DEFAULT_RUN_ID)
SOURCE_RUN_ID = os.environ.get("V6_REPORT_SOURCE_RUN_ID", RUN_ID)
REPORT_TITLE = os.environ.get("V6_REPORT_TITLE", "Crab SED v6 64748 nhit100 highEplus1 Stage A-G")
REPORT_DIR = REPO_ROOT / "apply" / "report"
if RUN_ID == DEFAULT_RUN_ID:
    REPORT_PATH = REPORT_DIR / "crab_sed_v6_64748_nhit100_highEplus1_stage_a_to_g_report.html"
    ASSET_DIR = REPORT_DIR / "assets" / "v6-64748-nhit100-highEplus1"
else:
    REPORT_PATH = REPORT_DIR / f"crab_sed_{RUN_ID}_stage_a_to_g_report.html"
    ASSET_DIR = REPORT_DIR / "assets" / RUN_ID.replace("_", "-")
VALIDATION_JSON = ASSET_DIR / "report_validation.json"
SOURCE_ASSET_DIR = (
    REPORT_DIR / "assets" / "v6-64748-nhit100-highEplus1"
    if SOURCE_RUN_ID == DEFAULT_RUN_ID
    else REPORT_DIR / "assets" / SOURCE_RUN_ID.replace("_", "-")
)
STAGE_B_THETA_CACHE = SOURCE_ASSET_DIR / f"{SOURCE_RUN_ID}_stage_b_raw_theta_profiles.npz"
STAGE_B_THETA_PROFILE = ASSET_DIR / f"{RUN_ID}_stage_b_raw_theta_profiles.png"
STAGE_B_THETA_PROFILE_PDF = ASSET_DIR / f"{RUN_ID}_stage_b_raw_theta_profiles.pdf"
STAGE_B_FIT_SHADED_PROFILE = ASSET_DIR / f"{RUN_ID}_stage_b_radial_psf_profiles_fit_shaded.png"
STAGE_D_DEC_PROFILE = ASSET_DIR / f"{RUN_ID}_stage_d_dec_profile_before_after.png"
STAGE_D_DEC_PROFILE_PDF = ASSET_DIR / f"{RUN_ID}_stage_d_dec_profile_before_after.pdf"
STAGE_G_EXTERNAL_OVERLAY = ASSET_DIR / f"{RUN_ID}_stage_g_external_overlay.png"
TRUE_ENERGY_CELL_GRID = (
    ASSET_DIR
    / "true-energy-cell-grid"
    / "v6_64748_reselect44_true_energy_cell_grid.png"
)

PASS5_CSV = REPORT_DIR / "assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"
V099_CSV = REPORT_DIR / "assets/official-v099/wcda_crab_sed_v099_20250731_20260616_123624.csv"

LEDGER = REPO_ROOT / f"apply/config/cell_ledger_{SOURCE_RUN_ID}_candidate.csv"
PREFIT_SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{SOURCE_RUN_ID}_prefit.csv"
FIT_SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_fit.csv"
SELECTOR_META = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_fit_metadata.json"
HIGH_E_DECISIONS = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_highEplus1_decisions.csv"

STAGE_A = REPO_ROOT / f"apply/output/stage_a_{SOURCE_RUN_ID}"
STAGE_A_AP = REPO_ROOT / f"apply/output/stage_a_{RUN_ID}_aperture_conditioned"
STAGE_B = REPO_ROOT / f"apply/output/stage_b_{RUN_ID}/runs/{RUN_ID}_stage_b_psf"
STAGE_B_UNFILTERED_DIAGNOSTIC = STAGE_B / f"psf_{RUN_ID}_unfiltered_diagnostic.npz"
STAGE_C = REPO_ROOT / f"apply/output/stage_c_{SOURCE_RUN_ID}/runs/{SOURCE_RUN_ID}_stage_c_halfyear"
STAGE_D = REPO_ROOT / f"apply/output/stage_d_{RUN_ID}_annnorm/runs/{RUN_ID}_stage_d_annnorm"
STAGE_E = REPO_ROOT / f"apply/output/stage_e_{RUN_ID}_containment1_annnorm/runs/{RUN_ID}_stage_e_containment1_annnorm"
STAGE_F = REPO_ROOT / f"apply/output/stage_f_{RUN_ID}/runs/{RUN_ID}_stage_f"
STAGE_G = REPO_ROOT / f"apply/output/stage_g_{RUN_ID}/runs/{RUN_ID}_stage_g"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def rel(path: Path | str) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    return html.escape(os.path.relpath(target, start=REPORT_PATH.parent))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    out = finite_float(value)
    if out is None:
        return "n/a"
    if out == 0:
        return "0"
    if abs(out) >= 1.0e5 or abs(out) < 1.0e-3:
        return f"{out:.{digits}e}"
    return f"{out:.{digits}g}"


def fmt_int(value: Any) -> str:
    out = finite_float(value)
    return "n/a" if out is None else f"{int(round(out)):,}"


def table(headers: list[str], rows: Iterable[Iterable[Any]], classes: str = "") -> str:
    class_attr = f' class="{classes}"' if classes else ""
    parts = [f"<table{class_attr}><thead><tr>"]
    parts.extend(f"<th>{esc(header)}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def figure(path: Path, caption: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected report image: {path}")
    return (
        '<figure class="figure">'
        f'<img src="{rel(path)}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


def archive_report_figures(figures: Iterable[tuple[Path, str]]) -> None:
    """Keep a flat, presentation-ready copy of every report image."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for source, _ in figures:
        target = ASSET_DIR / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def global_cutflow_map(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if row.get("scope") != "global":
            continue
        try:
            out[str(row["step"])] = int(float(row["count"]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def sed_rows_by_group(path: Path, grouping: str) -> list[dict[str, str]]:
    return [row for row in load_csv(path) if row.get("grouping") == grouping]


def fit_metric(meta: dict[str, Any], fit_key: str, metric: str) -> Any:
    return ((meta.get("fits") or {}).get(fit_key) or {}).get(metric)


def parse_interval(label: str) -> tuple[float | None, float | None]:
    label = label.strip()
    if label.lower() in {"all", "*"}:
        return None, None
    if label.startswith("[") and label.endswith(")"):
        low, high = label[1:-1].split(",", 1)
        return float(low), float(high)
    if label.startswith("<"):
        return None, float(label[1:])
    if label.startswith(">="):
        return float(label[2:]), None
    raise ValueError(f"Unsupported interval label: {label}")


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def display_cell_id(cell_id: int, pred_bin: str) -> int | None:
    """Map the internal 13-column ID to the displayed 12-column analysis ID."""
    if pred_bin.strip().startswith(">="):
        return None
    return int(cell_id) - ((int(cell_id) - 1) // 13)


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def rayleigh_pdf_deg(r_deg: Any, sigma_rad: float) -> Any:
    import numpy as np

    r_rad = np.radians(r_deg)
    pdf_per_rad = (r_rad / (sigma_rad * sigma_rad)) * np.exp(-0.5 * (r_rad / sigma_rad) ** 2)
    return pdf_per_rad * (math.pi / 180.0)


def fit_cell_ids_from_selector(rows: list[dict[str, str]]) -> set[int]:
    return {int(float(row["cell_id"])) for row in rows if row.get("cell_id") and truthy(row.get("include"))}


def ensure_stage_b_theta_profile_grid(fit_ids: set[int]) -> None:
    if not STAGE_B_THETA_CACHE.exists():
        raise FileNotFoundError(
            f"Missing complete Stage B theta cache: {STAGE_B_THETA_CACHE}. "
            "Build it on ETO with apply/report/build_v6_stage_b_theta_profile_cache.py."
        )
    source_mtime = max(STAGE_B_THETA_CACHE.stat().st_mtime, FIT_SELECTOR.stat().st_mtime, Path(__file__).stat().st_mtime)
    if (
        STAGE_B_THETA_PROFILE.exists()
        and STAGE_B_THETA_PROFILE_PDF.exists()
        and min(STAGE_B_THETA_PROFILE.stat().st_mtime, STAGE_B_THETA_PROFILE_PDF.stat().st_mtime) >= source_mtime
    ):
        return

    import numpy as np

    with np.load(STAGE_B_THETA_CACHE, allow_pickle=False) as cache:
        cell_ids = np.asarray(cache["cell_id"], dtype=np.int64)
        nhit_bins = np.asarray(cache["nhit_bin"], dtype=str)
        pred_bins = np.asarray(cache["predE_bin"], dtype=str)
        theta_edges = np.asarray(cache["theta_edges_deg"], dtype=np.float64)
        crab_probability = np.asarray(cache["crab_theta_probability"], dtype=np.float64)
        mc_probability = np.asarray(cache["mc_theta_probability"], dtype=np.float64)
        missing_mass = np.asarray(cache["theta_missing_crab_probability_mass"], dtype=np.float64)
        sources = np.asarray(cache["source"], dtype=str)

    if mc_probability.shape != (cell_ids.size, theta_edges.size - 1):
        raise ValueError("Stage B theta cache has incompatible profile dimensions")
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    plt = setup_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(1.78 * len(ordered_pred), 1.55 * len(ordered_nhit)),
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
            shown_cell_id = display_cell_id(cell_id, pred)
            if cell_id in fit_ids:
                ax.set_facecolor("#ecfdf5")
                for spine in ax.spines.values():
                    spine.set_color("#059669")
                    spine.set_linewidth(1.25)
                ax.text(
                    0.03,
                    0.94,
                    "fit",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.8,
                    color="#047857",
                    fontweight="bold",
                )

            probability = mc_probability[idx]
            unsupported = (crab_probability > 0.0) & ~(probability > 0.0)
            if np.any(unsupported):
                ax.fill_between(
                    centers,
                    0.0,
                    crab_probability,
                    where=unsupported,
                    step="mid",
                    color="#E69F00",
                    alpha=0.24,
                    linewidth=0.0,
                )
            ax.step(centers, probability, where="mid", color="#0072B2", linewidth=0.95)
            ax.step(centers, crab_probability, where="mid", color="#D55E00", linewidth=0.8, alpha=0.9)
            cell_line = f"cell {shown_cell_id}\n" if shown_cell_id is not None else ""
            ax.text(
                0.97,
                0.94,
                f"{cell_line}missing={missing_mass[idx]:.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.7,
                color="#0f172a",
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.7},
            )
            if sources[idx] == "no_input_files":
                ax.text(
                    0.5,
                    0.43,
                    "no MC files",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#6b7280",
                )

            ax.set_title(pred, fontsize=6.4)
            ax.set_xlim(float(theta_edges[0]), float(theta_edges[-1]))
            panel_max = float(max(np.nanmax(probability), np.nanmax(crab_probability)))
            ax.set_ylim(0.0, 1.18 * panel_max if panel_max > 0.0 else 1.0)
            ax.tick_params(labelsize=6, length=2, labelleft=(j == 0))
            ax.grid(alpha=0.18, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(nhit, fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("MC true theta (deg)", fontsize=6.7)

    handles = [
        Line2D([0], [0], color="#0072B2", linewidth=1.1, label="raw weighted MC"),
        Line2D([0], [0], color="#D55E00", linewidth=1.0, label="Crab theta target"),
        Patch(facecolor="#E69F00", alpha=0.24, label="missing Crab support"),
        Patch(facecolor="#ecfdf5", edgecolor="#059669", label="included in fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle("Stage B raw normalized MC theta distributions by cell", fontsize=11, y=0.999)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    STAGE_B_THETA_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STAGE_B_THETA_PROFILE, dpi=300)
    fig.savefig(STAGE_B_THETA_PROFILE_PDF)
    plt.close(fig)


def ensure_stage_b_fit_shaded_profile_grid(fit_ids: set[int]) -> None:
    psf_path = STAGE_B / f"psf_{RUN_ID}.npz"
    if not psf_path.exists():
        return
    source_paths = [psf_path, FIT_SELECTOR, Path(__file__)]
    if STAGE_B_UNFILTERED_DIAGNOSTIC.exists():
        source_paths.append(STAGE_B_UNFILTERED_DIAGNOSTIC)
    source_mtime = max(path.stat().st_mtime for path in source_paths)
    if STAGE_B_FIT_SHADED_PROFILE.exists() and STAGE_B_FIT_SHADED_PROFILE.stat().st_mtime >= source_mtime:
        return

    import numpy as np

    with np.load(psf_path, allow_pickle=False) as psf:
        cell_ids = np.asarray(psf["cell_id"], dtype=np.int64)
        nhit_bins = np.asarray(psf["nhit_bin"], dtype=str)
        pred_bins = np.asarray(psf["predE_bin"], dtype=str)
        profile_edges_deg = np.asarray(psf["profile_edges_deg"], dtype=np.float64)
        profile_density = np.asarray(psf["profile_density"], dtype=np.float64)
        sigma_rad = np.asarray(psf["sigma_rad"], dtype=np.float64)
        r_opt_deg = np.asarray(psf["r_opt_deg"], dtype=np.float64)

    diagnostic_by_cell: dict[int, tuple[np.ndarray, float, float, int, str]] = {}
    if STAGE_B_UNFILTERED_DIAGNOSTIC.exists():
        with np.load(STAGE_B_UNFILTERED_DIAGNOSTIC, allow_pickle=False) as diagnostic:
            diagnostic_ids = np.asarray(diagnostic["cell_id"], dtype=np.int64)
            diagnostic_edges = np.asarray(diagnostic["profile_edges_deg"], dtype=np.float64)
            diagnostic_profiles = np.asarray(diagnostic["profile_density"], dtype=np.float64)
            diagnostic_sigma_deg = np.asarray(diagnostic["sigma_deg"], dtype=np.float64)
            diagnostic_r_opt_deg = np.asarray(diagnostic["r_opt_deg"], dtype=np.float64)
            diagnostic_events = np.asarray(diagnostic["events"], dtype=np.int64)
            diagnostic_status = np.asarray(diagnostic["status"], dtype=str)
        if diagnostic_edges.shape != profile_edges_deg.shape or not np.allclose(diagnostic_edges, profile_edges_deg):
            raise ValueError("Formal and unfiltered diagnostic PSF profile edges do not match")
        diagnostic_by_cell = {
            int(cell_id): (
                diagnostic_profiles[idx],
                float(diagnostic_sigma_deg[idx]),
                float(diagnostic_r_opt_deg[idx]),
                int(diagnostic_events[idx]),
                str(diagnostic_status[idx]),
            )
            for idx, cell_id in enumerate(diagnostic_ids)
        }

    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    centers = 0.5 * (profile_edges_deg[:-1] + profile_edges_deg[1:])

    plt = setup_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(1.78 * len(ordered_pred), 1.55 * len(ordered_nhit)),
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
            shown_cell_id = display_cell_id(cell_id, pred)
            if cell_id in fit_ids:
                ax.set_facecolor("#ecfdf5")
                for spine in ax.spines.values():
                    spine.set_color("#059669")
                    spine.set_linewidth(1.25)
                ax.text(
                    0.03,
                    0.94,
                    "fit",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.8,
                    color="#047857",
                    fontweight="bold",
                )
            if shown_cell_id is not None:
                ax.text(
                    0.97,
                    0.94,
                    f"cell {shown_cell_id}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.2,
                    color="#0f172a",
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7},
                )

            density = profile_density[idx]
            formal_profile = np.isfinite(density).any() and np.nansum(density) > 0.0
            if formal_profile:
                ax.step(centers, density, where="mid", color="#1f4e79", linewidth=0.9)
                if idx < sigma_rad.size and np.isfinite(sigma_rad[idx]) and sigma_rad[idx] > 0.0:
                    ax.plot(centers, rayleigh_pdf_deg(centers, float(sigma_rad[idx])), color="#c9501a", linewidth=0.8, alpha=0.9)
                if idx < r_opt_deg.size and np.isfinite(r_opt_deg[idx]):
                    ax.axvline(float(r_opt_deg[idx]), color="#444444", linewidth=0.7, linestyle="--")
            else:
                diagnostic = diagnostic_by_cell.get(cell_id)
                if diagnostic is not None and diagnostic[4] == "ok" and np.nansum(diagnostic[0]) > 0.0:
                    diagnostic_density, diagnostic_sigma, diagnostic_r_opt, diagnostic_count, _ = diagnostic
                    ax.step(centers, diagnostic_density, where="mid", color="#7c3aed", linewidth=0.9)
                    ax.plot(
                        centers,
                        rayleigh_pdf_deg(centers, math.radians(diagnostic_sigma)),
                        color="#c9501a",
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.9,
                    )
                    ax.axvline(diagnostic_r_opt, color="#7c3aed", linewidth=0.7, linestyle=":")
                    ax.text(
                        0.03,
                        0.80,
                        f"diag no E cut\nN={diagnostic_count}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=5.2,
                        color="#6d28d9",
                    )
                else:
                    diagnostic_count = diagnostic[3] if diagnostic is not None else 0
                    label = "no MC events" if diagnostic_count == 0 else "no diagnostic fit"
                    ax.text(
                        0.5,
                        0.45,
                        label,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="#6b7280",
                    )

            ax.set_title(pred, fontsize=6.4)
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
        Line2D([0], [0], color="#7c3aed", linewidth=0.9, label="unfiltered diagnostic only"),
        Patch(facecolor="#ecfdf5", edgecolor="#059669", label="included in fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle(f"Stage B {RUN_ID} radial PSF profiles: fit cells shaded", fontsize=11, y=0.999)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    STAGE_B_FIT_SHADED_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STAGE_B_FIT_SHADED_PROFILE)
    plt.close(fig)


def ensure_stage_d_dec_profile(fit_ids: set[int]) -> None:
    import numpy as np

    stage_d_npz = STAGE_D / f"background_{RUN_ID}_annnorm.npz"
    input_paths = [Path(__file__), stage_d_npz, FIT_SELECTOR]
    existing_inputs = [path for path in input_paths if path.exists()]
    source_mtime = max(path.stat().st_mtime for path in existing_inputs)
    if (
        STAGE_D_DEC_PROFILE.exists()
        and STAGE_D_DEC_PROFILE_PDF.exists()
        and min(STAGE_D_DEC_PROFILE.stat().st_mtime, STAGE_D_DEC_PROFILE_PDF.stat().st_mtime) >= source_mtime
    ):
        return

    with np.load(stage_d_npz, allow_pickle=False) as data:
        required = {"cell_id", "x_centers_deg", "y_centers_deg", "counts_map", "background_map"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Stage D profile input is missing arrays: {sorted(missing)}")
        cell_ids = np.asarray(data["cell_id"], dtype=np.int64)
        x_centers = np.asarray(data["x_centers_deg"], dtype=np.float64)
        y_centers = np.asarray(data["y_centers_deg"], dtype=np.float64)
        counts = np.asarray(data["counts_map"], dtype=np.float64)
        background = np.asarray(data["background_map"], dtype=np.float64)

    selected = np.isin(cell_ids, np.asarray(sorted(fit_ids), dtype=np.int64))
    x_band = np.abs(x_centers) < 1.0
    if np.count_nonzero(selected) != len(fit_ids):
        raise ValueError(f"Stage D profile matched {np.count_nonzero(selected)} of {len(fit_ids)} fit cells")
    counts_profile = np.nansum(counts[selected][:, :, x_band], axis=(0, 2))
    background_profile = np.nansum(background[selected][:, :, x_band], axis=(0, 2))
    excess_profile = counts_profile - background_profile

    roi = np.abs(y_centers) <= 6.0
    core = np.abs(y_centers) < 1.0
    core_counts = float(np.nansum(counts_profile[core]))
    core_background = float(np.nansum(background_profile[core]))
    core_excess = float(np.nansum(excess_profile[core]))
    core_fraction = core_excess / core_counts if core_counts > 0.0 else float("nan")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=170)
    ax.plot(y_centers[roi], counts_profile[roi], color="#111827", lw=1.35, label="Observed counts (before subtraction)")
    ax.plot(y_centers[roi], background_profile[roi], color="#6b7280", lw=1.2, ls=":", label="Annulus-normalized fitted background")
    ax.plot(y_centers[roi], excess_profile[roi], color="#2563eb", lw=2.0, label="Counts - background (after subtraction)")
    ax.axhline(0.0, color="#111827", lw=0.8, ls="--")
    ax.axvspan(-1.0, 1.0, color="#e5e7eb", alpha=0.5, label=r"Central $|\Delta\mathrm{Dec}|<1^\circ$")
    ax.text(
        0.985,
        0.48,
        "Central $|\\Delta\\mathrm{Dec}|<1^\\circ$, $|\\Delta\\mathrm{RA}|<1^\\circ$\n"
        f"observed = {core_counts:,.0f}\n"
        f"fitted background = {core_background:,.0f}\n"
        f"excess = {core_excess:,.0f} ({100.0 * core_fraction:.2f}% of observed)",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )
    ax.set_xlim(-6.0, 6.0)
    ax.set_xlabel("Dec offset from Crab [deg]")
    ax.set_ylabel(r"Summed counts in $|\Delta\mathrm{RA}|<1^\circ$")
    ax.set_title(f"Before/after background subtraction for {len(fit_ids)} active v6 fit cells")
    ax.grid(True, alpha=0.24, lw=0.45)
    ax.legend(fontsize=7.7, ncol=2, loc="upper left", frameon=True)
    fig.tight_layout()
    STAGE_D_DEC_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STAGE_D_DEC_PROFILE)
    fig.savefig(STAGE_D_DEC_PROFILE_PDF)
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return load_csv(path)


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


def fit_logpar_to_e2_points(
    energy: list[float], e2_flux: list[float], *, pivot_tev: float = 3.0
) -> dict[str, float] | None:
    import numpy as np

    e = np.asarray(energy, dtype=np.float64)
    e2 = np.asarray(e2_flux, dtype=np.float64)
    valid = np.isfinite(e) & np.isfinite(e2) & (e > 0.0) & (e2 > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    log_ratio = np.log(e[valid] / float(pivot_tev))
    log_dnde = np.log(e2[valid] / np.square(e[valid]))
    c2, c1, c0 = np.polyfit(log_ratio, log_dnde, 2)
    return {
        "phi0": float(np.exp(c0)),
        "alpha": float(-c1),
        "beta": float(-c2),
        "pivot_tev": float(pivot_tev),
    }


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


def logpar_flux_tev(E_tev: Any, *, phi0: float, alpha: float, beta: float, pivot_tev: float) -> Any:
    import numpy as np

    ratio = np.asarray(E_tev, dtype=np.float64) / float(pivot_tev)
    log_ratio = np.log(ratio)
    return float(phi0) * np.exp((-float(alpha) - float(beta) * log_ratio) * log_ratio)


def pl_flux_tev(E_tev: Any, *, phi0: float, gamma: float, pivot_tev: float) -> Any:
    import numpy as np

    ratio = np.asarray(E_tev, dtype=np.float64) / float(pivot_tev)
    return float(phi0) * np.power(ratio, -float(gamma))


def sed_model_curve(E_tev: Any, model: str, params: dict[str, Any], pivot_tev: float) -> Any:
    if model == "logpar":
        return E_tev * E_tev * logpar_flux_tev(
            E_tev,
            phi0=float(params["phi0"]),
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            pivot_tev=pivot_tev,
        )
    return E_tev * E_tev * pl_flux_tev(
        E_tev,
        phi0=float(params["phi0"]),
        gamma=float(params["gamma"]),
        pivot_tev=pivot_tev,
    )


def sed_uncertainty_band(E_tev: Any, fit: dict[str, Any], pivot_tev: float) -> tuple[Any, Any, Any] | None:
    import numpy as np

    names = [str(name) for name in fit.get("fit_parameter_names") or []]
    covariance = fit.get("covariance")
    params = fit.get("parameters") or {}
    if not names or covariance is None:
        return None
    cov = np.asarray(covariance, dtype=np.float64)
    x = np.asarray(E_tev, dtype=np.float64)
    if cov.shape != (len(names), len(names)):
        return None
    model = str(fit.get("model_name") or "")
    y = sed_model_curve(x, model, params, pivot_tev)
    log_ratio = np.log(x / float(pivot_tev))
    grad = np.zeros((x.size, len(names)), dtype=np.float64)
    for idx, name in enumerate(names):
        if name == "log10_phi0":
            grad[:, idx] = math.log(10.0)
        elif name in {"gamma", "alpha"}:
            grad[:, idx] = -log_ratio
        elif name == "beta":
            grad[:, idx] = -(log_ratio * log_ratio)
    var_log_y = np.einsum("ij,jk,ik->i", grad, cov, grad)
    sigma = np.sqrt(np.clip(var_log_y, 0.0, np.inf))
    return y, y * np.exp(-sigma), y * np.exp(sigma)


def ensure_stage_g_external_overlay(stage_f_meta: dict[str, Any], stage_g_meta: dict[str, Any]) -> None:
    input_paths = [
        Path(__file__),
        STAGE_G / f"sed_points_{RUN_ID}_metadata.json",
        STAGE_G / f"sed_points_{RUN_ID}_summary.csv",
        STAGE_F / f"fit_{RUN_ID}_metadata.json",
        STAGE_G / "external_crab_sed_references.csv",
        STAGE_G / "wcda1_pool1_table1_reference.csv",
        PASS5_CSV,
        V099_CSV,
    ]
    existing_inputs = [path for path in input_paths if path.exists()]
    source_mtime = max(path.stat().st_mtime for path in existing_inputs)
    if STAGE_G_EXTERNAL_OVERLAY.exists() and STAGE_G_EXTERNAL_OVERLAY.stat().st_mtime >= source_mtime:
        return

    import numpy as np

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=170)

    points = [row for row in stage_g_meta.get("points", []) if isinstance(row, dict)]
    energies = [finite_float(row.get("effective_energy_tev")) for row in points]
    energies = [e for e in energies if e is not None and e > 0.0]
    emin = max(0.12, min(energies) / 2.2) if energies else 0.12
    emax = min(320.0, max(energies) * 2.2) if energies else 260.0
    x = np.geomspace(emin, emax, 320)

    frozen = stage_g_meta.get("frozen_spectrum") or {}
    fit_key = str(frozen.get("fit_key") or "logpar_conservative")
    fit = ((stage_f_meta.get("fits") or {}).get(fit_key) or {})
    frozen_params = {str(k): float(v) for k, v in (frozen.get("parameters") or {}).items()}
    if not frozen_params:
        frozen_params = {str(k): float(v) for k, v in (fit.get("parameters") or {}).items()}
    model = str(frozen.get("model") or fit.get("model_name") or "logpar")
    pivot = float(frozen.get("pivot_tev") or stage_f_meta.get("forward_folding", {}).get("pivot_tev") or 3.0)
    y_model = sed_model_curve(x, model, frozen_params, pivot)
    band = sed_uncertainty_band(x, fit, pivot)
    if band is not None:
        _, y_low, y_high = band
        ax.fill_between(x, y_low, y_high, color="#2563eb", alpha=0.18, linewidth=0, label="Stage F LogPar 1-sigma band")
    ax.plot(x, y_model, color="#2563eb", lw=2.1, label="v6 Stage F LogPar")

    ref = stage_g_meta.get("reference_spectrum") or {}
    if ref:
        ax.plot(
            x,
            sed_model_curve(
                x,
                "pl",
                {"phi0": float(ref.get("phi0")), "gamma": float(ref.get("gamma"))},
                float(ref.get("pivot_tev") or pivot),
            ),
            color="#4b5563",
            lw=1.6,
            ls="--",
            label="1LHAASO WCDA full-array PL",
        )

    e_pass5, y_pass5 = pass5_points()
    if e_pass5:
        ax.plot(e_pass5, y_pass5, "o", ms=5.2, color="#111827", label="Official pass5 WCDA", zorder=4)
        pass5_fit = fit_logpar_to_e2_points(e_pass5, y_pass5)
        if pass5_fit is not None:
            ax.plot(
                x,
                sed_model_curve(x, "logpar", pass5_fit, pass5_fit["pivot_tev"]),
                color="#111827",
                lw=1.7,
                ls="-",
                label="Official pass5 point-fit LogPar",
                zorder=3,
            )

    e_v099, y_v099, ylo_v099, yhi_v099 = v099_points()
    if e_v099:
        ax.errorbar(
            e_v099,
            y_v099,
            yerr=[ylo_v099, yhi_v099],
            fmt="s",
            ms=4.9,
            lw=0.9,
            color="#7c2d12",
            ecolor="#7c2d12",
            capsize=2.4,
            label="Official tutorial v0.99 WCDA",
            zorder=4,
        )

    pool1_rows = read_csv_rows(STAGE_G / "wcda1_pool1_table1_reference.csv")
    if pool1_rows:
        ax.errorbar(
            [float(row["emed_tev"]) for row in pool1_rows],
            [float(row["E2_dnde"]) for row in pool1_rows],
            yerr=[float(row["E2_dnde_err"]) for row in pool1_rows],
            fmt="^",
            color="#7f3fbf",
            ecolor="#7f3fbf",
            capsize=2.5,
            ms=4.8,
            lw=0.8,
            label="WCDA-1 Pool-1 Table 1",
            alpha=0.92,
        )

    external_styles = {
        "magic_joint_crab": {"fmt": "v", "color": "#9467bd", "label": "MAGIC"},
        "hess_2024_stereo": {"fmt": "D", "color": "#8c564b", "label": "H.E.S.S."},
        "hawc_2019_nn": {"fmt": "P", "color": "#17becf", "label": "HAWC NN"},
    }
    external_rows = read_csv_rows(STAGE_G / "external_crab_sed_references.csv")
    for dataset, style in external_styles.items():
        selected = [
            row
            for row in external_rows
            if row.get("dataset") == dataset
            and str(row.get("is_upper_limit")).strip().lower() != "true"
            and finite_float(row.get("energy_tev"))
            and finite_float(row.get("e2_dnde"))
        ]
        if not selected:
            continue
        ax.errorbar(
            [float(row["energy_tev"]) for row in selected],
            [float(row["e2_dnde"]) for row in selected],
            yerr=[finite_float(row.get("e2_dnde_err")) or 0.0 for row in selected],
            capsize=1.9,
            ms=3.8,
            lw=0.65,
            alpha=0.58,
            **style,
        )

    point_styles = {
        "nhit": {"fmt": "o", "color": "#2563eb", "label": "v6 Nhit grouped"},
    }
    for grouping, style in point_styles.items():
        selected = [
            row
            for row in points
            if row.get("grouping") == grouping
            and (finite_float(row.get("effective_energy_tev")) or 0.0) > 0.0
            and (finite_float(row.get("E2_dnde")) or 0.0) > 0.0
        ]
        if not selected:
            continue
        ax.errorbar(
            [float(row["effective_energy_tev"]) for row in selected],
            [float(row["E2_dnde"]) for row in selected],
            yerr=[float(row["E2_dnde_err"]) for row in selected],
            capsize=2.8,
            ms=5.1,
            lw=0.9,
            zorder=6,
            **style,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective true energy [TeV]")
    ax.set_ylabel(r"$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    ax.set_title(f"Stage G {RUN_ID} SED overlay with external WCDA references")
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    ax.set_xlim(emin, emax)
    ax.legend(fontsize=7.0, ncol=2, frameon=True)
    fig.tight_layout()
    STAGE_G_EXTERNAL_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STAGE_G_EXTERNAL_OVERLAY)
    plt.close(fig)


def parse_pipeline_jobs() -> list[tuple[str, str]]:
    raw = os.environ.get("PIPELINE_JOB_IDS", "").strip()
    pairs: list[tuple[str, str]] = []
    for item in re.split(r"[;,]", raw):
        item = item.strip()
        if not item or ":" not in item:
            continue
        label, job_id = item.split(":", 1)
        if job_id and job_id != "PENDING":
            pairs.append((label.strip(), job_id.strip()))
    if not pairs and os.environ.get("SLURM_JOB_ID"):
        pairs.append(("current_report_job", str(os.environ["SLURM_JOB_ID"])))
    return pairs


def sacct_rows(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, job_id in pairs:
        record = {"stage": label, "job_id": job_id, "state": "unknown", "elapsed": "unknown", "exit_code": "unknown", "job_name": ""}
        try:
            result = subprocess.run(
                ["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,Elapsed,ExitCode,JobName"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            rows.append(record)
            continue
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 5 or parts[0] != job_id:
                continue
            record.update({"state": parts[1], "elapsed": parts[2], "exit_code": parts[3], "job_name": parts[4]})
            break
        rows.append(record)
    return rows


def collect_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(collect_strings(item, child))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(collect_strings(item, f"{prefix}[{idx}]"))
        return out
    if isinstance(value, str):
        return [(prefix, value)]
    return []


LEGACY_INPUT_RE = re.compile(
    r"(64670|8666|WCDA_simulation_binned_response_v[1-5]|stage_[a-g]_v[1-5]|/v[1-5][_/.-]|_v[1-5]_)",
    re.IGNORECASE,
)


def contamination_audit(metadata_files: list[Path]) -> dict[str, Any]:
    offenders: list[dict[str, str]] = []
    for path in metadata_files:
        payload = load_json(path)
        for key, value in collect_strings(payload):
            if LEGACY_INPUT_RE.search(value):
                offenders.append({"metadata": str(path), "field": key, "value": value})
    return {
        "status": "passed" if not offenders else "failed",
        "metadata_files": [str(path) for path in metadata_files],
        "offenders": offenders,
    }


class ImageRefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        src = attr_map.get("src")
        if src:
            self.images.append(src)


def validate_html_images(path: Path) -> dict[str, Any]:
    parser = ImageRefParser()
    text = path.read_text(encoding="utf-8")
    parser.feed(text)
    missing: list[str] = []
    for src in parser.images:
        if re.match(r"^[a-z]+://", src):
            continue
        image_path = (path.parent / src).resolve()
        if not image_path.exists():
            missing.append(src)
    return {"image_refs": parser.images, "missing_image_refs": missing, "image_ref_count": len(parser.images)}


def status_class(value: str) -> str:
    normalized = value.lower()
    if normalized in {"pass", "passed", "completed", "completed+"}:
        return "pass"
    if normalized in {"warning", "running", "unknown"}:
        return "warn"
    return "fail"


def main() -> None:
    stage_a_meta = load_json(STAGE_A / f"response_2d_{SOURCE_RUN_ID}_metadata.json")
    stage_a_ap_meta = load_json(STAGE_A_AP / f"response_2d_{RUN_ID}_aperture_conditioned_metadata.json")
    stage_b_meta = load_json(STAGE_B / f"psf_{RUN_ID}_metadata.json")
    stage_c_meta = load_json(STAGE_C / "obs_events_metadata.json")
    stage_d_meta = load_json(STAGE_D / f"background_{RUN_ID}_annnorm_metadata.json")
    stage_e_meta = load_json(STAGE_E / f"signal_{RUN_ID}_containment1_annnorm_metadata.json")
    stage_f_meta = load_json(STAGE_F / f"fit_{RUN_ID}_metadata.json")
    stage_g_meta = load_json(STAGE_G / f"sed_points_{RUN_ID}_metadata.json")
    stage_g_summary = load_json(STAGE_G / f"sed_points_{RUN_ID}_summary.json")
    selector_meta = load_json(SELECTOR_META)

    selector_rows = load_csv(FIT_SELECTOR)
    prefit_rows = load_csv(PREFIT_SELECTOR)
    decision_rows = load_csv(HIGH_E_DECISIONS)
    source_rows = load_csv(STAGE_C / "source_files.csv")
    cutflow = global_cutflow_map(load_csv(STAGE_C / "obs_events_cutflow.csv"))
    fit_rows = [row for row in selector_rows if truthy(row.get("include"))]
    tail_rows = [row for row in selector_rows if row.get("predE_bin") == ">=6"]
    tail_included = [row for row in tail_rows if truthy(row.get("include"))]
    high_inc = [row for row in selector_rows if truthy(row.get("highEplus1_included_flag"))]
    high_rej = [
        row
        for row in selector_rows
        if truthy(row.get("highEplus1_rejected_flag")) and not truthy(row.get("include"))
    ]
    original_ridge = [row for row in selector_rows if truthy(row.get("original_ridge_fit_flag"))]
    original_ridge_included = [row for row in original_ridge if truthy(row.get("include"))]
    forced_included = [row for row in selector_rows if row.get("selection_override_flag") == "force_include"]
    forced_excluded = [row for row in selector_rows if row.get("selection_override_flag") == "force_exclude"]
    fit_cell_ids = fit_cell_ids_from_selector(selector_rows)
    ensure_stage_b_theta_profile_grid(fit_cell_ids)
    ensure_stage_b_fit_shaded_profile_grid(fit_cell_ids)
    ensure_stage_d_dec_profile(fit_cell_ids)
    ensure_stage_g_external_overlay(stage_f_meta, stage_g_meta)

    processing = stage_c_meta.get("processing") or {}
    stage_c_files = int(processing.get("processed_file_count") or 0)
    missing_time = int(processing.get("missing_time_file_count") or 0)
    entry_mismatch = int(processing.get("entry_mismatch_file_count") or 0)
    selected_rows = int(processing.get("selected_rows") or 0)
    rough_live_days = float(((stage_c_meta.get("live_time_basis") or {}).get("rough_live_time_days_sum_files")) or 0.0)
    bad_time_rows = [
        row
        for row in source_rows
        if row.get("status") == "missing_time_skipped" or str(row.get("entry_mismatch")).strip().lower() == "true"
    ]

    e_totals = stage_e_meta.get("totals") or {}
    f_pref = stage_f_meta.get("preferred_fit") or {}
    f_quality = stage_f_meta.get("quality") or {}
    v6_logpar = (stage_f_meta.get("fits") or {}).get("logpar_conservative") or {}
    g_quality = stage_g_meta.get("quality") or {}
    g_frozen = stage_g_summary.get("frozen_spectrum") or {}
    g_csv = STAGE_G / f"sed_points_{RUN_ID}_summary.csv"
    g_nhit_rows = sed_rows_by_group(g_csv, "nhit")
    g_pred_rows = sed_rows_by_group(g_csv, "predE")
    job_rows = sacct_rows(parse_pipeline_jobs())

    metadata_files = [
        STAGE_A / f"response_2d_{SOURCE_RUN_ID}_metadata.json",
        STAGE_A_AP / f"response_2d_{RUN_ID}_aperture_conditioned_metadata.json",
        STAGE_B / f"psf_{RUN_ID}_metadata.json",
        STAGE_C / "obs_events_metadata.json",
        STAGE_D / f"background_{RUN_ID}_annnorm_metadata.json",
        STAGE_E / f"signal_{RUN_ID}_containment1_annnorm_metadata.json",
        STAGE_F / f"fit_{RUN_ID}_metadata.json",
        STAGE_G / f"sed_points_{RUN_ID}_metadata.json",
        SELECTOR_META,
    ]
    contamination = contamination_audit(metadata_files)

    validation_rows = [
        ("run id", "pass" if SOURCE_RUN_ID in str(stage_a_meta.get("npz_path")) else "warning", RUN_ID),
        ("Nhit binning", "pass" if selector_rows and selector_rows[0].get("nhit_bin") == "[100,200)" else "fail", selector_rows[0].get("nhit_bin") if selector_rows else "missing"),
        ("tail policy", "pass" if not tail_included else "fail", f"{len(tail_rows)} >=6 tail cells, {len(tail_included)} included"),
        (
            "selector",
            "pass",
            f"{len(fit_rows)} fit cells; {len(forced_included)} forced in, {len(forced_excluded)} forced out",
        ),
        (
            "selector 75/90 swap",
            "pass" if 75 in fit_cell_ids and 90 not in fit_cell_ids and len(fit_cell_ids) == 44 else "fail",
            f"cell 75 included={75 in fit_cell_ids}; cell 90 included={90 in fit_cell_ids}; total={len(fit_cell_ids)}",
        ),
        ("Stage C files", "pass" if stage_c_files > 3000 else "warning", f"{stage_c_files:,} processed, missing time {missing_time}, entry mismatch {entry_mismatch}"),
        ("Stage E signal", "pass" if (stage_e_meta.get("quality_gate") or {}).get("status") == "passed" else "warning", f"formal sigma {fmt(e_totals.get('formal_sigma'), 5)}"),
        ("Stage F fit", "pass" if f_quality.get("fit_status") == "passed" else "warning", f"preferred {f_pref.get('model')}"),
        ("Stage G points", "pass" if g_nhit_rows and g_pred_rows else "warning", f"{len(g_nhit_rows)} Nhit points, {len(g_pred_rows)} predE points"),
        ("metadata pollution", contamination["status"], f"{len(contamination['offenders'])} legacy main-input path/token offenders"),
    ]

    stage_rows = [
        ("Prepare/cache", "/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate"),
        ("Stage A response", f"apply/output/stage_a_{SOURCE_RUN_ID} (reused)"),
        ("Stage B PSF", f"apply/output/stage_b_{RUN_ID}"),
        ("Stage A aperture response", f"apply/output/stage_a_{RUN_ID}_aperture_conditioned"),
        ("Stage C observation", f"apply/output/stage_c_{SOURCE_RUN_ID} (reused)"),
        ("Stage D background", f"apply/output/stage_d_{RUN_ID}_annnorm"),
        ("Stage E signal", f"apply/output/stage_e_{RUN_ID}_containment1_annnorm"),
        ("Stage F fit", f"apply/output/stage_f_{RUN_ID}"),
        ("Stage G SED", f"apply/output/stage_g_{RUN_ID}"),
    ]

    expected_figures = [
        (STAGE_B / "psf_r_opt_deg_grid.png", "Stage B r_opt by cell"),
        (STAGE_B / "psf_effective_events_grid.png", "Stage B effective events by cell"),
        (STAGE_B_THETA_PROFILE, "Stage B raw normalized MC theta distributions; green panels enter the fit"),
        (STAGE_B_FIT_SHADED_PROFILE, "Stage B radial PSF profiles; purple profiles are unfiltered diagnostics only, and green panels enter the final SED fit"),
        (
            TRUE_ENERGY_CELL_GRID,
            "The panels show normalized true-energy distributions for all 91 (Nhit, predE) cells; display IDs 1-84 exclude the unnumbered predE >= 6 tail column. Blue distributions with green borders mark the final 44 cells used by the Stage F/G SED fit, while gray panels are diagnostic-only excluded cells.",
        ),
        (STAGE_D / "roi_excess_grid.png", "Stage D ROI excess map grid"),
        (STAGE_D / "annulus_residual_grid.png", "Stage D annulus residuals"),
        (STAGE_D_DEC_PROFILE, "Stage D aggregate Dec profile before and after annulus-normalized background subtraction for the 44 active fit cells"),
        (STAGE_E / "formal_sigma_grid.png", "Stage E formal sigma grid"),
        (STAGE_E / "on_background_grid.png", "Stage E on/background grid"),
        (STAGE_F / "model_counts_vs_excess.png", "Stage F model counts versus excess"),
        (STAGE_F / "pull_grid_logpar.png", "Stage F LogPar pull grid"),
        (STAGE_G_EXTERNAL_OVERLAY, "Stage G SED overlay with v6 Nhit points, v6 fit band, Pass5 point-fit LogPar, and external references"),
        (STAGE_G / "sed_points_ratio.png", "Stage G SED ratio plot"),
    ]
    archive_report_figures(expected_figures)
    figure_html = "".join(figure(path, caption) for path, caption in expected_figures)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(REPORT_TITLE)}</title>
  <style>
    :root {{
      --ink:#17202a; --muted:#5e6875; --line:#d7dde5; --panel:#f6f8fb;
      --ok:#126b45; --warn:#93620c; --fail:#aa2e25; --accent:#005eb8;
    }}
    body {{ margin:0; color:var(--ink); background:#fff; font-family:Arial,Helvetica,sans-serif; line-height:1.48; }}
    main {{ max-width:1220px; margin:0 auto; padding:32px 24px 58px; }}
    header {{ border-bottom:3px solid var(--ink); padding-bottom:18px; margin-bottom:26px; }}
    h1,h2,h3 {{ margin:0; line-height:1.18; letter-spacing:0; }}
    h1 {{ font-size:32px; }}
    h2 {{ font-size:21px; margin-top:34px; padding-top:14px; border-top:1px solid var(--line); }}
    h3 {{ font-size:16px; margin-top:18px; color:#2f3b48; }}
    p {{ margin:10px 0; }}
    code {{ background:#eef2f6; padding:1px 4px; border-radius:3px; font-size:12px; }}
    table {{ border-collapse:collapse; width:100%; margin:14px 0 22px; font-size:13px; }}
    th,td {{ border:1px solid var(--line); padding:6px 7px; text-align:right; vertical-align:top; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#edf1f6; font-weight:700; }}
    .lede {{ font-size:16px; color:#2f3b48; max-width:960px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:16px 0; }}
    .metric {{ border:1px solid var(--line); border-radius:6px; padding:12px; background:var(--panel); min-height:78px; }}
    .metric .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .metric .value {{ font-size:23px; font-weight:700; margin-top:4px; }}
    .metric .sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
    .status-pass {{ color:var(--ok); font-weight:700; }}
    .status-warn,.status-warning {{ color:var(--warn); font-weight:700; }}
    .status-fail,.status-failed {{ color:var(--fail); font-weight:700; }}
    .okbox {{ border-left:5px solid var(--ok); background:#edf9f1; padding:12px 14px; margin:18px 0; }}
    .callout {{ border-left:5px solid var(--warn); background:#fff8ec; padding:12px 14px; margin:18px 0; }}
    .figgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; margin:14px 0 22px; }}
    .figure {{ margin:0; border:1px solid var(--line); border-radius:6px; padding:10px; background:#fff; }}
    .figure img {{ display:block; width:100%; height:auto; }}
    .figure figcaption {{ margin-top:8px; color:var(--muted); font-size:12px; }}
    .pathlist td {{ text-align:left; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{esc(REPORT_TITLE)}</h1>
    <p class="lede">v6 chain for <code>{RUN_ID}</code>, reusing the <code>{SOURCE_RUN_ID}</code> nominal response and Stage C event reduction. The first Nhit bin is <code>[100,200)</code>; <code>&gt;=6</code> remains a diagnostic tail outside Stage F/G.</p>
  </header>

  <section>
    <h2>Executive Check</h2>
    <div class="grid">
      <div class="metric"><div class="label">candidate cells</div><div class="value">{len(selector_rows)}</div><div class="sub">7 Nhit x 13 predE</div></div>
      <div class="metric"><div class="label">fit cells</div><div class="value">{len(fit_rows)}</div><div class="sub">{len(original_ridge_included)} retained ridge; {len(forced_included)} forced in</div></div>
      <div class="metric"><div class="label">Stage C files</div><div class="value">{stage_c_files:,}</div><div class="sub">missing time {missing_time}; entry mismatch {entry_mismatch}</div></div>
      <div class="metric"><div class="label">selected rows</div><div class="value">{selected_rows:,}</div><div class="sub">rough live {rough_live_days:.3f} d</div></div>
      <div class="metric"><div class="label">Stage E signal</div><div class="value">{fmt(e_totals.get('formal_sigma'), 5)}</div><div class="sub">formal sigma</div></div>
      <div class="metric"><div class="label">Stage F preferred</div><div class="value">{esc(f_pref.get('model')).upper()}</div><div class="sub">chi2/ndof {fmt(v6_logpar.get('chi2'), 4)}/{esc(v6_logpar.get('ndof'))}</div></div>
      <div class="metric"><div class="label">Stage G points</div><div class="value">{len(g_nhit_rows)}/{len(g_pred_rows)}</div><div class="sub">Nhit / predE groupings</div></div>
      <div class="metric"><div class="label">metadata audit</div><div class="value">{esc(contamination['status']).upper()}</div><div class="sub">{len(contamination['offenders'])} offenders</div></div>
    </div>
    {table(["Gate", "Status", "Evidence"], [[esc(name), f'<span class="status-{status_class(status)}">{esc(status)}</span>', esc(evidence)] for name, status, evidence in validation_rows])}
  </section>

  <section>
    <h2>Slurm Jobs</h2>
    <p>All heavy recomputation stages are intended to run through the Slurm dependency chain. This table is populated from <code>PIPELINE_JOB_IDS</code> when available.</p>
    {table(["Stage", "Job", "State", "Elapsed", "Exit", "Name"], [[esc(row["stage"]), esc(row["job_id"]), f'<span class="status-{status_class(row["state"])}">{esc(row["state"])}</span>', esc(row["elapsed"]), esc(row["exit_code"]), esc(row["job_name"])] for row in job_rows]) if job_rows else '<div class="callout">No Slurm job id manifest was provided to the report builder.</div>'}
  </section>

  <section>
    <h2>Inputs And Outputs</h2>
    <p>The main chain uses the 64748 observation eval root and its recovered-time tree. The response, PSF, aperture response, fit, and SED products are all under the new run namespace.</p>
    {table(["Field", "Value"], [
        ["Run id", f"<code>{RUN_ID}</code>"],
        ["Observation root", f"<code>{esc(stage_c_meta.get('obs_root'))}</code>"],
        ["Recovered time root", f"<code>{esc(stage_c_meta.get('time_root'))}</code>"],
        ["MC candidate cache", "<code>/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate</code>"],
        ["Model run dir", f"<code>{esc(stage_a_meta.get('run_dir'))}</code>"],
        ["Fit selector", f"<code>{rel(FIT_SELECTOR)}</code>"],
        ["Stage C input entries", f"{cutflow.get('input_entries', 0):,}"],
        ["After configured-cell selection", f"{cutflow.get('after_cell_selection', 0):,}"],
    ], "pathlist")}
    {table(["Stage", "Primary output"], [[esc(name), f"<code>{esc(path)}</code>"] for name, path in stage_rows], "pathlist")}
  </section>

  <section>
    <h2>Selector Rule</h2>
    <p>{esc(selector_meta.get('rule'))}</p>
    {table(["Nhit", "Status", "Candidate", "MC count", "PSF gate", "Reasons"], [[f"<code>{esc(row.get('nhit_bin'))}</code>", esc(row.get("status")), f"<code>{esc(row.get('candidate_predE_bin'))}</code> cell {esc(row.get('candidate_cell_id'))}", fmt_int(row.get("mc_count")), esc(row.get("psf_quality_flag")), esc(row.get("psf_quality_reasons"))] for row in decision_rows])}
    <h3>Included highEplus1 cells</h3>
    {table(["Cell", "Nhit", "predE", "MC count", "PSF reason"], [[esc(row.get("cell_id")), f"<code>{esc(row.get('nhit_bin'))}</code>", f"<code>{esc(row.get('predE_bin'))}</code>", fmt_int(row.get("mc_count")), esc(row.get("psf_quality_reasons"))] for row in high_inc]) if high_inc else '<div class="callout">No highEplus1 probe passed the Stage B quality gate.</div>'}
    <h3>Rejected highEplus1 probes</h3>
    {table(["Cell", "Nhit", "predE", "MC count", "Reason"], [[esc(row.get("cell_id")), f"<code>{esc(row.get('nhit_bin'))}</code>", f"<code>{esc(row.get('predE_bin'))}</code>", fmt_int(row.get("mc_count")), esc(row.get("exclusion_source"))] for row in high_rej]) if high_rej else '<div class="okbox">No highEplus1 candidate was rejected by the gate.</div>'}
    <h3>Explicit selector overrides</h3>
    {table(["Action", "Cell", "Nhit", "predE", "PSF status"], [["include", esc(row.get("cell_id")), f"<code>{esc(row.get('nhit_bin'))}</code>", f"<code>{esc(row.get('predE_bin'))}</code>", esc(row.get("psf_quality_reasons"))] for row in forced_included] + [["exclude", esc(row.get("cell_id")), f"<code>{esc(row.get('nhit_bin'))}</code>", f"<code>{esc(row.get('predE_bin'))}</code>", esc(row.get("psf_quality_reasons"))] for row in forced_excluded]) if forced_included or forced_excluded else '<div class="callout">No explicit selector overrides.</div>'}
  </section>

  <section>
    <h2>Stage C Time Audit</h2>
    <p>Stage C used <code>{esc(stage_c_meta.get('time_root'))}</code>. Missing-time and entry-mismatch rows are listed below when present.</p>
    {table(["Metric", "Value"], [
        ["Processed files", f"{stage_c_files:,}"],
        ["Missing time files", f"{missing_time:,}"],
        ["Entry mismatch files", f"{entry_mismatch:,}"],
        ["Selected rows", f"{selected_rows:,}"],
        ["Matched MJD min", fmt((stage_c_meta.get("mjd_coverage") or {}).get("matched_mjd_min"), 8)],
        ["Matched MJD max", fmt((stage_c_meta.get("mjd_coverage") or {}).get("matched_mjd_max"), 8)],
    ])}
    {table(["source_file_id", "status", "relative_path", "event_entries", "time_entries", "selected_rows"], [[esc(row.get("source_file_id")), esc(row.get("status")), esc(row.get("relative_path")), esc(row.get("event_entries")), esc(row.get("time_entries")), esc(row.get("selected_rows"))] for row in bad_time_rows[:50]]) if bad_time_rows else '<div class="okbox">No missing recovered-time files or entry mismatches were recorded.</div>'}
  </section>

  <section>
    <h2>Stage A-B-D-E Diagnostics</h2>
    <p>Stage A nominal response is <code>{esc(stage_a_meta.get('response_type'))}</code>; Stage F/G use <code>{esc(stage_a_ap_meta.get('response_type'))}</code>. Stage B wrote {esc(stage_b_meta.get('n_cells'))} formal PSF rows. Figure labels use display IDs 1-84 after excluding the <code>predE &gt;= 6</code> tail column from numbering; stored data and metadata retain the original 91-cell internal IDs. The theta grid shows each cell's raw <code>mc_weight</code>-normalized distribution before Crab reweighting; orange shading marks Crab-positive theta bins without MC support. In the radial grid, purple profiles and dashed Rayleigh curves are diagnostic-only fits made without the formal true-energy cut when the formal profile is empty; they do not replace the Stage B PSF used by Stage A/F. Panels with no raw MC events are labeled explicitly. Pale green panels mark the {len(fit_cell_ids)} cells used by Stage F/G.</p>
    <div class="figgrid">{figure_html}</div>
  </section>

  <section>
    <h2>Stage F Fit</h2>
    <p>Stage F uses the final selector and the aperture-conditioned 64748 response. The preferred model recorded by Stage F is <code>{esc(f_pref.get('model'))}</code>.</p>
    {table(["Fit", "Valid", "chi2/ndof", "p", "phi0", "gamma/alpha", "beta"], [
        ["PL conservative", esc(fit_metric(stage_f_meta, "pl_conservative", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_conservative', 'chi2'), 4)}/{esc(fit_metric(stage_f_meta, 'pl_conservative', 'ndof'))}", fmt(fit_metric(stage_f_meta, "pl_conservative", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_conservative", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_conservative", "gamma"), 4), "n/a"],
        ["LogPar conservative", esc(fit_metric(stage_f_meta, "logpar_conservative", "valid")), f"{fmt(v6_logpar.get('chi2'), 4)}/{esc(v6_logpar.get('ndof'))}", fmt(v6_logpar.get("p_value"), 3), fmt(v6_logpar.get("phi0"), 4), fmt(v6_logpar.get("alpha"), 4), fmt(v6_logpar.get("beta"), 4)],
        ["PL sqrt-N", esc(fit_metric(stage_f_meta, "pl_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_sqrt_n', 'chi2'), 4)}/{esc(fit_metric(stage_f_meta, 'pl_sqrt_n', 'ndof'))}", fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "gamma"), 4), "n/a"],
        ["LogPar sqrt-N", esc(fit_metric(stage_f_meta, "logpar_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'logpar_sqrt_n', 'chi2'), 4)}/{esc(fit_metric(stage_f_meta, 'logpar_sqrt_n', 'ndof'))}", fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "alpha"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "beta"), 4)],
    ])}
  </section>

  <section>
    <h2>Stage G SED</h2>
    <p>Stage G freezes the preferred Stage F spectrum with phi0={fmt(g_frozen.get('phi0'), 4)}, alpha={fmt(g_frozen.get('alpha'), 4)}, beta={fmt(g_frozen.get('beta'), 4)} at {fmt(g_frozen.get('pivot_tev'), 3)} TeV. Quality status: <code>{esc(g_quality.get('status'))}</code>.</p>
    {table(["Nhit group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{esc(row['ndof'])}"] for row in g_nhit_rows])}
    {table(["PredE group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{esc(row['ndof'])}"] for row in g_pred_rows])}
  </section>

  <section>
    <h2>Metadata Audit</h2>
    <p>Main-input metadata files were scanned for legacy cache/model path tokens. Status: <strong>{esc(contamination['status'])}</strong>.</p>
    {table(["Metadata", "Field", "Value"], [[f"<code>{esc(row['metadata'])}</code>", esc(row["field"]), f"<code>{esc(row['value'])}</code>"] for row in contamination["offenders"]], "pathlist") if contamination["offenders"] else '<div class="okbox">No legacy main-input metadata path offenders were found.</div>'}
  </section>
</main>
</body>
</html>
"""

    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    html_validation = validate_html_images(REPORT_PATH)
    validation_payload = {
        "report_path": str(REPORT_PATH),
        "html_image_validation": html_validation,
        "metadata_contamination": contamination,
        "selector_summary": selector_meta,
    }
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    if html_validation["missing_image_refs"]:
        raise SystemExit(f"Report has missing image references: {html_validation['missing_image_refs']}")
    if contamination["offenders"]:
        raise SystemExit("Metadata contamination audit failed; see report and validation JSON")

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {VALIDATION_JSON}")
    print(f"Image refs: {html_validation['image_ref_count']}; missing: {len(html_validation['missing_image_refs'])}")
    print(f"Fit cells: {len(fit_rows)}; highEplus1 included/rejected: {len(high_inc)}/{len(high_rej)}")


if __name__ == "__main__":
    main()
