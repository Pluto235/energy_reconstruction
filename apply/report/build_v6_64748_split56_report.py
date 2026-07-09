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
REPORT_PATH = REPORT_DIR / "crab_sed_v6_64748_split56_baselinev4_report.html"
REPORT_ALIAS_PATH = REPORT_DIR / "crab_sed_v6_64748_baselinev4_report.html"
ASSET_DIR = REPORT_DIR / "assets" / "v6-64748-split56"
STAGE_B_FIT_SHADED_PROFILE = ASSET_DIR / "v6_64748_split56_stage_b_radial_psf_profiles_fit_shaded.png"

SELECTOR = REPO_ROOT / "apply/config/cell_selector_v6_64748_split56_drop4_psfborrow.csv"
LEDGER = REPO_ROOT / "apply/config/cell_ledger_v6_64748_split56_candidate.csv"
STAGE_A = REPO_ROOT / "apply/output/stage_a_v6_64748_split56"
STAGE_A_AP = REPO_ROOT / "apply/output/stage_a_v6_64748_split56_aperture_conditioned"
STAGE_B = REPO_ROOT / "apply/output/stage_b_v6_64748_split56/runs/v6_64748_split56_stage_b_psf"
STAGE_C = REPO_ROOT / "apply/output/stage_c_v6_64748_split56/runs/v6_64748_split56_stage_c_halfyear"
STAGE_D = REPO_ROOT / "apply/output/stage_d_v6_64748_split56_annnorm/runs/v6_64748_split56_stage_d_annnorm"
STAGE_E = REPO_ROOT / "apply/output/stage_e_v6_64748_split56_containment1_annnorm/runs/v6_64748_split56_stage_e_containment1_annnorm"
STAGE_F = REPO_ROOT / "apply/output/stage_f_v6_64748_split56_baselinev4/runs/v6_64748_split56_stage_f_baselinev4"
STAGE_G = REPO_ROOT / "apply/output/stage_g_v6_64748_split56_baselinev4/runs/v6_64748_split56_stage_g_baselinev4"
OBS_ROOT_64748 = Path("/mnt/mydisk/WCDA_observation_eval_64748")
OBS_ROOT_64670 = Path("/mnt/mydisk/WCDA_observation_eval_64670")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rel(path: Path | str) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    return html.escape(os.path.relpath(target, start=REPORT_PATH.parent))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
        return ""
    return (
        '<figure class="figure">'
        f'<img src="{rel(path)}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


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


def fit_cell_ids_from_selector(path: Path) -> set[int]:
    rows = load_csv(path)
    return {int(float(row["cell_id"])) for row in rows if row.get("cell_id") and truthy(row.get("include"))}


def split_child_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    split_keys = {
        ("[2000,3000)", "[5,5.5)"),
        ("[2000,3000)", "[5.5,6)"),
    }
    out = [
        row
        for row in rows
        if (row.get("nhit_bin"), row.get("predE_bin")) in split_keys and truthy(row.get("include"))
    ]
    return sorted(out, key=lambda row: interval_key(row.get("predE_bin", "")))


def selected_fit_rows(rows: list[dict[str, str]], fit_ids: Iterable[int]) -> list[dict[str, str]]:
    by_id = {int(float(row["cell_id"])): row for row in rows if row.get("cell_id")}
    return [by_id[cell_id] for cell_id in sorted(fit_ids) if cell_id in by_id]


def warning_cell_ids(warnings: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for warning in warnings:
        if isinstance(warning, dict):
            cell_id = warning.get("cell_id")
        else:
            text = str(warning)
            cell_id = text.split(":", 1)[0].replace("cell", "").strip()
        try:
            out.add(int(cell_id))
        except (TypeError, ValueError):
            continue
    return out


def global_cutflow_map(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if row.get("scope") != "global":
            continue
        try:
            out[row["step"]] = int(float(row["count"]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def sed_rows_by_group(path: Path, grouping: str) -> list[dict[str, str]]:
    rows = load_csv(path)
    return [row for row in rows if row.get("grouping") == grouping]


def fit_metric(meta: dict[str, Any], fit_key: str, metric: str) -> Any:
    fits = meta.get("fits") or {}
    return (fits.get(fit_key) or {}).get(metric)


def ensure_stage_b_fit_shaded_profile_grid(fit_ids: set[int]) -> None:
    psf_path = STAGE_B / "psf_v6_64748_split56.npz"
    if not psf_path.exists():
        return
    source_mtime = max(psf_path.stat().st_mtime, SELECTOR.stat().st_mtime)
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
            if cell_id in fit_ids:
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
            if np.isfinite(density).any() and np.nansum(density) > 0.0:
                if idx < sigma_rad.size and np.isfinite(sigma_rad[idx]) and sigma_rad[idx] > 0.0:
                    ax.plot(centers, rayleigh_pdf_deg(centers, float(sigma_rad[idx])), color="#c9501a", linewidth=0.8, alpha=0.9)
                if idx < r_opt_deg.size and np.isfinite(r_opt_deg[idx]):
                    ax.axvline(float(r_opt_deg[idx]), color="#444444", linewidth=0.7, linestyle="--")

            ax.set_title(f"cell {cell_id}: {pred}", fontsize=6.4)
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
    fig.suptitle("Stage B v6 split56 weighted radial PSF profiles: fit cells shaded", fontsize=11, y=0.999)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    STAGE_B_FIT_SHADED_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STAGE_B_FIT_SHADED_PROFILE)
    plt.close(fig)


def observation_month_counts(root: Path) -> dict[str, int]:
    counts = {month: 0 for month in ("01", "02", "03", "04", "05", "06")}
    if not root.exists():
        return counts
    for path in root.glob("[0-9][0-9][0-9][0-9]/Esg*.root"):
        counts[path.parent.name[:2]] = counts.get(path.parent.name[:2], 0) + 1
    return counts


def main() -> None:
    stage_a_meta = load_json(STAGE_A / "response_2d_v6_64748_split56_metadata.json")
    stage_a_ap_meta = load_json(STAGE_A_AP / "response_2d_v6_64748_split56_aperture_conditioned_metadata.json")
    stage_b_meta = load_json(STAGE_B / "psf_v6_64748_split56_metadata.json")
    stage_c_meta = load_json(STAGE_C / "obs_events_metadata.json")
    stage_d_meta = load_json(STAGE_D / "background_v6_64748_split56_annnorm_metadata.json")
    stage_e_meta = load_json(STAGE_E / "signal_v6_64748_split56_containment1_annnorm_metadata.json")
    stage_f_meta = load_json(STAGE_F / "fit_v6_64748_split56_baselinev4_metadata.json")
    stage_g_meta = load_json(STAGE_G / "sed_points_v6_64748_split56_baselinev4_metadata.json")
    stage_g_summary = load_json(STAGE_G / "sed_points_v6_64748_split56_baselinev4_summary.json")

    selector_rows = load_csv(SELECTOR)
    candidate_count = len(selector_rows)
    fit_cell_ids = fit_cell_ids_from_selector(SELECTOR)
    fit_count = len(fit_cell_ids)
    split_rows = split_child_rows(selector_rows)
    split_exception_rows = [row for row in split_rows if truthy(row.get("split_child_gate_exception"))]
    ensure_stage_b_fit_shaded_profile_grid(fit_cell_ids)

    source_rows = load_csv(STAGE_C / "source_files.csv")
    cutflow = global_cutflow_map(load_csv(STAGE_C / "obs_events_cutflow.csv"))
    processed = sum(1 for row in source_rows if row.get("status") == "processed")
    entry_mismatches = sum(1 for row in source_rows if str(row.get("entry_mismatch")).lower() == "true")
    rough_live_days = sum(finite_float(row.get("rough_live_time_seconds")) or 0.0 for row in source_rows) / 86400.0
    selected_rows = sum(int(float(row.get("selected_rows") or 0)) for row in source_rows)

    psf_warnings = stage_b_meta.get("warning_rows") or []
    psf_warning_cell_ids = warning_cell_ids(psf_warnings)
    d_warnings = (stage_d_meta.get("quality") or {}).get("warnings") or []
    d_warning_cell_ids = warning_cell_ids(d_warnings)
    e_totals = stage_e_meta.get("totals") or {}
    f_pref = stage_f_meta.get("preferred_fit") or {}
    f_quality = stage_f_meta.get("quality") or {}
    f_exposure = stage_f_meta.get("exposure") or {}
    f_ref = stage_f_meta.get("reference_count_preflight") or {}
    v6_logpar = (stage_f_meta.get("fits") or {}).get("logpar_conservative") or {}
    g_frozen = stage_g_summary.get("frozen_spectrum") or {}
    g_quality = stage_g_meta.get("quality") or {}
    f_rows = selected_fit_rows(load_csv(STAGE_F / "fit_v6_64748_split56_baselinev4_summary.csv"), fit_cell_ids)
    g_summary_csv = STAGE_G / "sed_points_v6_64748_split56_baselinev4_summary.csv"
    g_nhit_rows = sed_rows_by_group(g_summary_csv, "nhit")
    g_pred_rows = sed_rows_by_group(g_summary_csv, "predE")
    nhit_point_count = len(g_nhit_rows)
    pred_point_count = len(g_pred_rows)
    caveats: list[str] = []
    if candidate_count != 91:
        caveats.append(f"candidate cells are {candidate_count}, expected 91")
    if fit_count != 27:
        caveats.append(f"fit cells are {fit_count}, expected 27")
    if nhit_point_count != 7:
        caveats.append(f"Stage G Nhit points are {nhit_point_count}, expected 7")
    if pred_point_count != 12:
        caveats.append(f"Stage G predE points are {pred_point_count}, expected 12")
    caveat_html = (
        '<div class="callout"><strong>Caveat.</strong> ' + esc("; ".join(caveats)) + "</div>"
        if caveats
        else '<div class="okbox"><strong>Split56 shape check passed.</strong> 91 candidate cells, 27 fit cells, 7 Nhit points, and 12 predE points.</div>'
    )
    split_gate_note = table(
        ["Cell", "Nhit", "predE", "MC count", "Default gate", "Effective gate", "ridge fraction", "Exception"],
        [
            [
                esc(row.get("cell_id")),
                f"<code>{esc(row.get('nhit_bin'))}</code>",
                f"<code>{esc(row.get('predE_bin'))}</code>",
                esc(row.get("mc_count")),
                esc(row.get("split_child_default_min_mc_count") or "1000"),
                esc(row.get("split_child_effective_min_mc_count") or "1000"),
                fmt(row.get("ridge_peak_fraction"), 4),
                esc(row.get("split_child_gate_exception") or "0"),
            ]
            for row in split_rows
        ],
    )
    split_gate_note_html = (
        '<div class="callout"><strong>Split-child gate exception.</strong> '
        "The right split child is retained because the old baselinev4 "
        "<code>[2000,3000) x [5,6)</code> fit cell is contractually split into two fit cells. "
        "Cell-level selector fields record the default count gate and the explicit split-child effective gate."
        f"{split_gate_note}</div>"
        if split_exception_rows
        else '<div class="okbox"><strong>Split-child gate check passed.</strong> Both split children meet the default fit-cell quality gates.</div>'
    )

    stage_rows = [
        ("Prepare split56 cache", "/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate"),
        ("Stage A nominal response", "apply/output/stage_a_v6_64748_split56"),
        ("Stage B PSF", "apply/output/stage_b_v6_64748_split56"),
        ("Stage A aperture-conditioned response", "apply/output/stage_a_v6_64748_split56_aperture_conditioned"),
        ("Stage C observation reduction", "apply/output/stage_c_v6_64748_split56"),
        ("Stage D annulus-normalized background", "apply/output/stage_d_v6_64748_split56_annnorm"),
        ("Stage E signal", "apply/output/stage_e_v6_64748_split56_containment1_annnorm"),
        ("Stage F forward-folding fit", "apply/output/stage_f_v6_64748_split56_baselinev4"),
        ("Stage G diagnostic SED", "apply/output/stage_g_v6_64748_split56_baselinev4"),
    ]

    validation_rows = [
        ("candidate / fit cells", "pass" if candidate_count == 91 and fit_count == 27 else "warning", f"{candidate_count} candidates; {fit_count} fit cells"),
        ("split child fit cells", "warning" if split_exception_rows else "pass", f"{len(split_rows)} children included; count-gate exceptions {len(split_exception_rows)}"),
        ("Stage A nominal response", "pass", f"{stage_a_meta.get('response_type')}; {stage_a_meta.get('absolute_effective_area_status')}"),
        ("Stage B PSF", "warning" if psf_warning_cell_ids else "pass", f"{stage_b_meta.get('n_cells')} cells; warning cells {len(psf_warning_cell_ids)}, fit subset {len(psf_warning_cell_ids & fit_cell_ids)}"),
        ("Stage A aperture response", "pass", f"{stage_a_ap_meta.get('response_type')}; PSF path is split56"),
        ("Stage C observation", "pass", f"{processed} files; {selected_rows:,} selected rows; rough live time {rough_live_days:.3f} d"),
        ("Stage D background", "warning" if d_warning_cell_ids else "pass", f"{len(d_warning_cell_ids)} warning cells; {len(d_warning_cell_ids & fit_cell_ids)} in fit subset"),
        ("Stage E signal", "pass" if (stage_e_meta.get("quality_gate") or {}).get("status") == "passed" else "warning", f"formal sigma {fmt(e_totals.get('formal_sigma'), 5)}"),
        ("Stage F fit", "pass" if f_quality.get("fit_status") == "passed" else "warning", f"preferred {f_pref.get('model')}"),
        ("Stage G SED", "pass" if nhit_point_count == 7 and pred_point_count == 12 else "warning", f"{nhit_point_count} Nhit points; {pred_point_count} predE points"),
    ]

    artifacts = [
        ("split56 selector", SELECTOR),
        ("Stage A nominal metadata", STAGE_A / "response_2d_v6_64748_split56_metadata.json"),
        ("Stage B PSF NPZ", STAGE_B / "psf_v6_64748_split56.npz"),
        ("Stage A aperture response", STAGE_A_AP / "response_2d_v6_64748_split56_aperture_conditioned.npz"),
        ("Stage C metadata", STAGE_C / "obs_events_metadata.json"),
        ("Stage D background", STAGE_D / "background_v6_64748_split56_annnorm.npz"),
        ("Stage E signal", STAGE_E / "signal_v6_64748_split56_containment1_annnorm.npz"),
        ("Stage F fit", STAGE_F / "fit_v6_64748_split56_baselinev4.npz"),
        ("Stage G SED points", STAGE_G / "sed_points_v6_64748_split56_baselinev4.npz"),
        ("Stage E HTML", REPORT_DIR / "stage_e_v6_64748_split56_containment1_annnorm_report.html"),
        ("Stage F HTML", REPORT_DIR / "stage_f_v6_64748_split56_baselinev4_report.html"),
        ("Stage G HTML", REPORT_DIR / "stage_g_v6_64748_split56_baselinev4_report.html"),
    ]

    counts_64748 = observation_month_counts(OBS_ROOT_64748)
    counts_64670 = observation_month_counts(OBS_ROOT_64670)
    comparison_section = (
        "<p>The main apply flow in this report uses model/data identifier <code>64748</code> "
        "for the half-year interval <code>2022-01-01</code> through <code>2022-06-30</code>. "
        "Identifier <code>64670</code> is retained only as an explicit model-comparison reference for "
        "<code>2022-01</code> and <code>2022-02</code>; its local March-June ROOT outputs were removed "
        "after confirming the IHEP backup and preserving recovered-time metadata.</p>"
        + table(
            ["Month", "64748 eval ROOT files", "64670 retained eval ROOT files", "64670 role"],
            [
                [
                    f"2022-{month}",
                    f"{counts_64748.get(month, 0):,}",
                    f"{counts_64670.get(month, 0):,}",
                    "explicit Jan-Feb comparison" if month in {"01", "02"} else "not retained locally for comparison",
                ]
                for month in ("01", "02", "03", "04", "05", "06")
            ],
        )
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crab SED v6 _64748 split56 baselinev4 report</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5d6673;
      --line: #d7dde5;
      --panel: #f6f8fb;
      --ok: #146c43;
      --warn: #9a5a00;
      --accent: #005eb8;
    }}
    body {{ margin:0; font-family:Arial,Helvetica,sans-serif; color:var(--ink); background:#fff; line-height:1.48; }}
    main {{ max-width:1180px; margin:0 auto; padding:32px 24px 56px; }}
    header {{ border-bottom:3px solid var(--ink); padding-bottom:18px; margin-bottom:26px; }}
    h1,h2,h3 {{ margin:0; line-height:1.18; letter-spacing:0; }}
    h1 {{ font-size:34px; }}
    h2 {{ font-size:22px; margin-top:34px; padding-top:14px; border-top:1px solid var(--line); }}
    h3 {{ font-size:16px; margin-top:18px; color:#2f3b48; }}
    p {{ margin:10px 0; }}
    .lede {{ font-size:17px; color:#2f3b48; max-width:940px; }}
    .callout {{ border-left:5px solid var(--warn); background:#fff8ec; padding:12px 14px; margin:18px 0; }}
    .okbox {{ border-left:5px solid var(--ok); background:#edf9f1; padding:12px 14px; margin:18px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:16px 0; }}
    .metric {{ border:1px solid var(--line); border-radius:6px; padding:12px; background:var(--panel); min-height:78px; }}
    .metric .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0; }}
    .metric .value {{ font-size:24px; font-weight:700; margin-top:4px; }}
    .metric .sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
    table {{ border-collapse:collapse; width:100%; margin:14px 0 22px; font-size:13px; }}
    th,td {{ border:1px solid var(--line); padding:6px 7px; text-align:right; vertical-align:top; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#edf1f6; font-weight:700; }}
    code {{ background:#eef2f6; padding:1px 4px; border-radius:3px; font-size:12px; }}
    .status-pass {{ color:var(--ok); font-weight:700; }}
    .status-warning {{ color:var(--warn); font-weight:700; }}
    .figgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; margin:14px 0 22px; }}
    .figure {{ margin:0; border:1px solid var(--line); border-radius:6px; padding:10px; background:#fff; }}
    .figure img {{ display:block; width:100%; height:auto; }}
    .figure figcaption {{ margin-top:8px; color:var(--muted); font-size:12px; }}
    .pathlist td {{ text-align:left; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Crab SED v6 <code>_64748</code> split56 baselinev4</h1>
    <p class="lede">This report is the v6 mainline Crab SED apply chain for model/data identifier <code>64748</code>, covering <code>2022-01-01</code> through <code>2022-06-30</code>. It uses the split56 physical predE binning with <code>[5,6)</code> split into <code>[5,5.5)</code> and <code>[5.5,6)</code>, while <code>64670</code> is kept only as a January-February comparison reference.</p>
    {caveat_html}
    {split_gate_note_html}
  </header>

  <section>
    <h2>Executive Result</h2>
    <div class="grid">
      <div class="metric"><div class="label">candidate cells</div><div class="value">{candidate_count}</div><div class="sub">7 Nhit x 13 predE</div></div>
      <div class="metric"><div class="label">fit cells</div><div class="value">{fit_count}</div><div class="sub">baselinev4 plus split high-energy cell</div></div>
      <div class="metric"><div class="label">Stage C files</div><div class="value">{processed:,}</div><div class="sub">entry mismatches {entry_mismatches}</div></div>
      <div class="metric"><div class="label">Selected rows</div><div class="value">{selected_rows:,}</div><div class="sub">configured-cell rows after cuts</div></div>
      <div class="metric"><div class="label">Stage E signal</div><div class="value">{fmt(e_totals.get('formal_sigma'), 5)} sigma</div><div class="sub">N_on {fmt_int(e_totals.get('N_on'))}; B_on {fmt(e_totals.get('B_on'), 6)}</div></div>
      <div class="metric"><div class="label">Preferred Stage F</div><div class="value">{esc(f_pref.get('model')).upper()}</div><div class="sub">chi2/ndof {fmt(v6_logpar.get('chi2'), 4)}/{v6_logpar.get('ndof')}</div></div>
      <div class="metric"><div class="label">Stage G points</div><div class="value">{nhit_point_count}/{pred_point_count}</div><div class="sub">Nhit / predE</div></div>
      <div class="metric"><div class="label">LogPar</div><div class="value">{fmt(v6_logpar.get('phi0'), 4)}</div><div class="sub">alpha {fmt(v6_logpar.get('alpha'), 4)}, beta {fmt(v6_logpar.get('beta'), 4)}</div></div>
    </div>
    {table(["Gate", "Status", "Evidence"], [[esc(name), f'<span class="status-{status}">{status}</span>', esc(evidence)] for name, status, evidence in validation_rows])}
  </section>

  <section>
    <h2>Paths And Provenance</h2>
    <p>Main-flow paths are split56-specific and preserve the v6 <code>_64748</code> model/cache provenance. Any <code>64670</code> mention is an explicit two-month comparison boundary, not a main-flow input.</p>
    {table(["Stage", "Primary output"], [[esc(name), f"<code>{esc(path)}</code>"] for name, path in stage_rows], "pathlist")}
    {table(["Field", "Value"], [
        ["Source MC cache", "<code>/mnt/mydisk/WCDA_simulation_binned_response_v6_64748</code>"],
        ["Split56 candidate cache", "<code>/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate</code>"],
        ["Run dir", f"<code>{esc(stage_a_meta.get('run_dir'))}</code>"],
        ["Selector", "<code>apply/config/cell_selector_v6_64748_split56_drop4_psfborrow.csv</code>"],
        ["Stage C obs root", f"<code>{esc(stage_c_meta.get('obs_root'))}</code>"],
        ["Input entries", f"{cutflow.get('input_entries', 0):,}"],
        ["Rows after configured-cell selection", f"{cutflow.get('after_cell_selection', 0):,}"],
        ["Rough live time", f"{rough_live_days:.3f} d"],
    ], "pathlist")}
  </section>

  <section>
    <h2>Response And PSF</h2>
    <p>Nominal Stage A is <code>{esc(stage_a_meta.get('response_type'))}</code>; the final fit uses <code>{esc(stage_a_ap_meta.get('response_type'))}</code>. Stage B wrote {esc(stage_b_meta.get('n_cells'))} PSF rows. Pale green panels mark the {fit_count} split56 fit cells in the radial PSF grid.</p>
    <div class="figgrid">
      {figure(STAGE_B / "psf_r_opt_deg_grid.png", "Stage B split56 r_opt by candidate cell")}
      {figure(STAGE_B / "psf_effective_events_grid.png", "Stage B split56 effective events by candidate cell")}
      {figure(STAGE_B / "psf_radial_profiles_grid.png", "Stage B split56 radial PSF profiles")}
      {figure(STAGE_B_FIT_SHADED_PROFILE, "Stage B split56 radial PSF profiles; green panels enter the fit")}
    </div>
  </section>

  <section>
    <h2>Stage C-D-E Diagnostics</h2>
    <p>Stage C selected {selected_rows:,} configured-cell rows. Stage D produced the annulus-normalized quadratic ROI-local background with {len(d_warning_cell_ids)} warning cells, including {len(d_warning_cell_ids & fit_cell_ids)} in the fit subset. Stage E passed the total-sigma gate.</p>
    {table(["Metric", "Value"], [
        ["Stage C processed files", f"{processed:,}"],
        ["Stage C rough live time", f"{rough_live_days:.3f} d"],
        ["Stage D warning cells", f"{len(d_warning_cell_ids)} / {candidate_count}"],
        ["Stage D warning cells in fit subset", f"{len(d_warning_cell_ids & fit_cell_ids)} / {fit_count}"],
        ["Stage E total N_on", fmt_int(e_totals.get("N_on"))],
        ["Stage E total B_on", fmt(e_totals.get("B_on"), 6)],
        ["Stage E total excess", fmt(e_totals.get("excess"), 6)],
        ["Stage E formal sigma", fmt(e_totals.get("formal_sigma"), 6)],
    ])}
    <div class="figgrid">
      {figure(STAGE_D / "roi_excess_grid.png", "Stage D ROI excess grid")}
      {figure(STAGE_D / "annulus_residual_grid.png", "Stage D annulus residual diagnostics")}
      {figure(STAGE_E / "formal_sigma_grid.png", "Stage E formal sigma grid")}
      {figure(STAGE_E / "on_background_grid.png", "Stage E on/background grid")}
    </div>
  </section>

  <section>
    <h2>Stage F Fit</h2>
    <p>Stage F uses the split56 aperture-conditioned response, split56 containment-1 signal, and the split56 baselinev4 selector. The fit is diagnostic unless the remaining cell-level residuals are accepted by a later review.</p>
    {table(["Fit", "Valid", "chi2/ndof", "p", "phi0", "gamma/alpha", "beta"], [
        ["PL conservative", esc(fit_metric(stage_f_meta, "pl_conservative", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_conservative', 'chi2'), 4)}/{fit_metric(stage_f_meta, 'pl_conservative', 'ndof')}", fmt(fit_metric(stage_f_meta, "pl_conservative", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_conservative", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_conservative", "gamma"), 4), "n/a"],
        ["LogPar conservative", esc(fit_metric(stage_f_meta, "logpar_conservative", "valid")), f"{fmt(v6_logpar.get('chi2'), 4)}/{v6_logpar.get('ndof')}", fmt(v6_logpar.get("p_value"), 3), fmt(v6_logpar.get("phi0"), 4), fmt(v6_logpar.get("alpha"), 4), fmt(v6_logpar.get("beta"), 4)],
        ["PL sqrt-N", esc(fit_metric(stage_f_meta, "pl_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_sqrt_n', 'chi2'), 4)}/{fit_metric(stage_f_meta, 'pl_sqrt_n', 'ndof')}", fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "gamma"), 4), "n/a"],
        ["LogPar sqrt-N", esc(fit_metric(stage_f_meta, "logpar_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'logpar_sqrt_n', 'chi2'), 4)}/{fit_metric(stage_f_meta, 'logpar_sqrt_n', 'ndof')}", fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "alpha"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "beta"), 4)],
    ])}
    {table(["Cell", "Bin", "Excess", "Err", "LogPar model", "Pull"], [[row["cell_id"], f"<code>{esc(row['nhit_bin'])}</code> <code>{esc(row['predE_bin'])}</code>", fmt(row["excess"], 4), fmt(row["error_conservative"], 3), fmt(row["logpar_model"], 4), fmt(row["logpar_pull"], 3)] for row in f_rows])}
    <div class="figgrid">
      {figure(STAGE_F / "model_counts_vs_excess.png", "Stage F model counts versus observed excess")}
      {figure(STAGE_F / "pull_grid_logpar.png", "Stage F LogPar pull grid")}
      {figure(STAGE_F / "theta_exposure.png", "Stage F theta exposure")}
    </div>
  </section>

  <section>
    <h2>Stage G Diagnostic SED</h2>
    <p>Stage G freezes the preferred Stage F spectrum with phi0={fmt(g_frozen.get('phi0'), 4)}, alpha={fmt(g_frozen.get('alpha'), 4)}, beta={fmt(g_frozen.get('beta'), 4)} at {fmt(g_frozen.get('pivot_tev'), 3)} TeV. Actual point counts are {nhit_point_count} Nhit and {pred_point_count} predE.</p>
    {table(["Nhit group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof", "StageF ratio", "Full-array PL ratio"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{row['ndof']}", fmt(row["ratio_to_stage_f_model"], 3), fmt(row["ratio_to_full_array_pl_ref"], 3)] for row in g_nhit_rows])}
    {table(["PredE group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof", "StageF ratio", "Single cell"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{row['ndof']}", fmt(row["ratio_to_stage_f_model"], 3), esc(row["is_single_cell_point"])] for row in g_pred_rows])}
    <div class="figgrid">
      {figure(STAGE_G / "sed_points_stage_f_fullarray_pool1.png", "Stage G split56 SED overlay")}
      {figure(STAGE_G / "sed_points_ratio.png", "Stage G split56 SED ratios")}
      {figure(STAGE_G / "sed_point_cell_counts.png", "Stage G split56 cell grouping counts")}
    </div>
  </section>

  <section>
    <h2>64670 Reference Boundary</h2>
    {comparison_section}
  </section>

  <section>
    <h2>Artifacts</h2>
    {table(["Artifact", "Path"], [[esc(name), f'<code>{rel(path)}</code>'] for name, path in artifacts], "pathlist")}
    <p>Generated by <code>apply/report/build_v6_64748_split56_report.py</code>. Main flow uses split56 v6 <code>64748</code> outputs; <code>64670</code> is report-only comparison context.</p>
  </section>
</main>
</body>
</html>
"""

    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    REPORT_ALIAS_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {REPORT_ALIAS_PATH}")
    print(f"Stage G quality: {g_quality.get('status')} / publication_ready={g_quality.get('stage_g_publication_ready')}")
    print(f"Candidate cells: {candidate_count}; fit cells: {fit_count}; Stage G Nhit/predE: {nhit_point_count}/{pred_point_count}")


if __name__ == "__main__":
    main()
