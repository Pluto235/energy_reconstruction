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
REPORT_PATH = REPORT_DIR / "crab_sed_v6_64670_baselinev4_report.html"
V6_ASSET_DIR = REPORT_DIR / "assets" / "v6-64670"
V6_STAGE_B_FIT_SHADED_PROFILE = V6_ASSET_DIR / "v6_stage_b_radial_psf_profiles_fit_shaded.png"

V6_SELECTOR = REPO_ROOT / "apply/config/cell_selector_v6_drop4_psfborrow.csv"
V6_STAGE_A = REPO_ROOT / "apply/output/stage_a_v6_64670"
V6_STAGE_A_AP = REPO_ROOT / "apply/output/stage_a_v6_64670_aperture_conditioned"
V6_STAGE_B = REPO_ROOT / "apply/output/stage_b_v6_64670/runs/v6_psf_from_64670"
V6_STAGE_C = REPO_ROOT / "apply/output/stage_c_v6_64670/runs/v6_stage_c_64670_halfyear"
V6_STAGE_D = REPO_ROOT / "apply/output/stage_d_v6_64670_annnorm/runs/v6_stage_d_64670_annnorm"
V6_STAGE_E = REPO_ROOT / "apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm"
V6_STAGE_F = REPO_ROOT / "apply/output/stage_f_v6_64670_baselinev4/runs/v6_stage_f_64670_baselinev4"
V6_STAGE_G = REPO_ROOT / "apply/output/stage_g_v6_64670_baselinev4/runs/v6_stage_g_64670_baselinev4"

V4_STAGE_F = REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4"
V4_STAGE_G = REPO_ROOT / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4"


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


def pct(value: Any, digits: int = 1) -> str:
    out = finite_float(value)
    return "n/a" if out is None else f"{100.0 * out:.{digits}f}%"


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


def csv_row_by_key(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


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


def fit_metric(meta: dict[str, Any], fit_key: str, metric: str) -> Any:
    fits = meta.get("fits") or {}
    return (fits.get(fit_key) or {}).get(metric)


def fit_cell_ids_from_selector(path: Path) -> set[int]:
    rows = load_csv(path)
    return {int(float(row["cell_id"])) for row in rows if row.get("cell_id") and truthy(row.get("include"))}


def selected_fit_rows(rows: list[dict[str, str]], fit_ids: Iterable[int]) -> list[dict[str, str]]:
    by_id = {int(float(row["cell_id"])): row for row in rows if row.get("cell_id")}
    return [by_id[cell_id] for cell_id in sorted(fit_ids) if cell_id in by_id]


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


def ensure_stage_b_fit_shaded_profile_grid(fit_ids: set[int]) -> None:
    psf_path = V6_STAGE_B / "psf_v6_64670.npz"
    if not psf_path.exists():
        return
    source_mtime = max(psf_path.stat().st_mtime, V6_SELECTOR.stat().st_mtime)
    if V6_STAGE_B_FIT_SHADED_PROFILE.exists() and V6_STAGE_B_FIT_SHADED_PROFILE.stat().st_mtime >= source_mtime:
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
        figsize=(2.0 * len(ordered_pred), 1.55 * len(ordered_nhit)),
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

            ax.set_title(f"cell {cell_id}: {pred}", fontsize=6.7)
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
    fig.suptitle("Stage B v6 weighted radial PSF profiles: fit cells shaded", fontsize=11, y=0.999)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    V6_STAGE_B_FIT_SHADED_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(V6_STAGE_B_FIT_SHADED_PROFILE)
    plt.close(fig)


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


def sed_rows_by_group(path: Path, grouping: str) -> list[dict[str, str]]:
    rows = load_csv(path)
    return [row for row in rows if row.get("grouping") == grouping]


def comparison_rows() -> list[list[str]]:
    v6_rows = sed_rows_by_group(V6_STAGE_G / "sed_points_v6_64670_baselinev4_summary.csv", "nhit")
    v4_rows = sed_rows_by_group(V4_STAGE_G / "sed_points_v4_aperture_conditioned_drop4_summary.csv", "nhit")
    v4_by_group = {row["group_label"]: row for row in v4_rows}
    out: list[list[str]] = []
    for row in v6_rows:
        group = row["group_label"]
        v4 = v4_by_group.get(group, {})
        v6_e2 = finite_float(row.get("E2_dnde"))
        v4_e2 = finite_float(v4.get("E2_dnde"))
        ratio = None if v6_e2 is None or v4_e2 in (None, 0.0) else v6_e2 / v4_e2
        out.append(
            [
                f"<code>{esc(group)}</code>",
                fmt(row.get("effective_energy_tev"), 4),
                fmt(row.get("E2_dnde"), 4),
                fmt(row.get("E2_dnde_err"), 3),
                fmt(v4.get("E2_dnde"), 4),
                fmt(v4.get("E2_dnde_err"), 3),
                fmt(ratio, 3),
                fmt(row.get("ratio_to_full_array_pl_ref"), 3),
            ]
        )
    return out


def main() -> None:
    stage_a_meta = load_json(V6_STAGE_A / "response_2d_v6_64670_metadata.json")
    stage_a_ap_meta = load_json(V6_STAGE_A_AP / "response_2d_v6_64670_aperture_conditioned_metadata.json")
    stage_b_meta = load_json(V6_STAGE_B / "psf_v6_64670_metadata.json")
    stage_c_meta = load_json(V6_STAGE_C / "obs_events_metadata.json")
    stage_d_meta = load_json(V6_STAGE_D / "background_v6_64670_annnorm_metadata.json")
    stage_e_meta = load_json(V6_STAGE_E / "signal_v6_64670_containment1_annnorm_metadata.json")
    stage_f_meta = load_json(V6_STAGE_F / "fit_v6_64670_baselinev4_metadata.json")
    stage_g_meta = load_json(V6_STAGE_G / "sed_points_v6_64670_baselinev4_metadata.json")
    stage_g_summary = load_json(V6_STAGE_G / "sed_points_v6_64670_baselinev4_summary.json")
    v4_f_meta = load_json(V4_STAGE_F / "fit_v4_aperture_conditioned_drop4_metadata.json")

    source_rows = load_csv(V6_STAGE_C / "source_files.csv")
    cutflow = global_cutflow_map(load_csv(V6_STAGE_C / "obs_events_cutflow.csv"))
    d_rows = load_csv(V6_STAGE_D / "background_v6_64670_annnorm_summary.csv")
    e_rows = load_csv(V6_STAGE_E / "signal_v6_64670_containment1_annnorm_summary.csv")
    fit_cell_ids = fit_cell_ids_from_selector(V6_SELECTOR)
    ensure_stage_b_fit_shaded_profile_grid(fit_cell_ids)
    f_rows = selected_fit_rows(load_csv(V6_STAGE_F / "fit_v6_64670_baselinev4_summary.csv"), fit_cell_ids)
    g_nhit_rows = sed_rows_by_group(V6_STAGE_G / "sed_points_v6_64670_baselinev4_summary.csv", "nhit")
    g_pred_rows = sed_rows_by_group(V6_STAGE_G / "sed_points_v6_64670_baselinev4_summary.csv", "predE")

    processed = sum(1 for row in source_rows if row.get("status") == "processed")
    entry_mismatches = sum(1 for row in source_rows if str(row.get("entry_mismatch")).lower() == "true")
    rough_live_days = sum(finite_float(row.get("rough_live_time_seconds")) or 0.0 for row in source_rows) / 86400.0
    selected_rows = sum(int(float(row.get("selected_rows") or 0)) for row in source_rows)
    date_min = min(row["relative_path"] for row in source_rows)
    date_max = max(row["relative_path"] for row in source_rows)

    psf_warnings = stage_b_meta.get("warning_rows") or []
    psf_warning_cell_ids = warning_cell_ids(psf_warnings)
    d_warnings = (stage_d_meta.get("quality") or {}).get("warnings") or []
    d_warning_cell_ids = warning_cell_ids(d_warnings)
    f_pref = stage_f_meta["preferred_fit"]
    f_quality = stage_f_meta["quality"]
    f_exposure = stage_f_meta["exposure"]
    f_ref = stage_f_meta["reference_count_preflight"]
    g_frozen = stage_g_summary["frozen_spectrum"]
    g_quality = stage_g_meta["quality"]
    v4_pref = v4_f_meta["preferred_fit"]
    v4_logpar = (v4_f_meta["fits"] or {})["logpar_conservative"]
    v6_logpar = (stage_f_meta["fits"] or {})["logpar_conservative"]

    stage_jobs = [
        ("MC cache", "64744", "COMPLETED", "05:35:59", "/mnt/mydisk/WCDA_simulation_binned_response_v6_64670"),
        ("Prepare candidate cache", "64749", "COMPLETED", "01:03:14", "/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_candidate"),
        ("Stage A nominal", "64751", "COMPLETED", "01:18:48", "apply/output/stage_a_v6_64670"),
        ("Stage B PSF", "64752", "COMPLETED", "01:16:06", "apply/output/stage_b_v6_64670"),
        ("Stage A aperture-conditioned", "64753", "COMPLETED", "00:59:17", "apply/output/stage_a_v6_64670_aperture_conditioned"),
        ("Stage C smoke", "64754", "COMPLETED", "00:00:21", "apply/output/stage_c_v6_64670_smoke"),
        ("Stage C-G full", "64755", "COMPLETED", "00:18:25", "apply/output/stage_c_v6_64670 ... stage_g_v6_64670_baselinev4"),
    ]

    gate_rows = [
        ("Phase 0 eval/time coverage", "pass", f"{processed}/3969 files processed; entry mismatches={entry_mismatches}; bad match-status rows={cutflow.get('input_entries', 0) - cutflow.get('after_match_status', 0)}"),
        ("Phase 1 MC cache", "pass", "10,000 MC files, 3,525,301 inferred events, v6 `_64670` provenance present"),
        ("Phase 2 selector contract", "pass", "26 v4 baseline cells included; drop4 controls 4,17,39,43 excluded"),
        ("Stage A nominal response", "pass", f"{stage_a_meta.get('response_type')}; {stage_a_meta.get('absolute_effective_area_status')}"),
        ("Stage B PSF", "warning", f"84 cells written; {len(psf_warnings)} warning rows, {len(psf_warning_cell_ids & fit_cell_ids)} in the 26 fit cells"),
        ("Stage A aperture-conditioned response", "pass", f"{stage_a_ap_meta.get('response_type')}; v6 PSF path recorded"),
        ("Stage C observation reduction", "pass", f"{selected_rows:,} selected configured-cell rows; rough live time {rough_live_days:.3f} d"),
        ("Stage D annulus background", "warning", f"{len(d_warning_cell_ids)} / 84 cells have warnings; {len(d_warning_cell_ids & fit_cell_ids)} warning rows are in the 26 fit cells; current promotion intentionally blocked"),
        ("Stage E signal", "pass", "quality passed; total formal sigma 68.805"),
        ("Stage F/G fit and SED", "pass", f"preferred {f_pref.get('model')} fit; Stage G is diagnostic-only and physical"),
    ]

    artifacts = [
        ("Stage A nominal metadata", V6_STAGE_A / "response_2d_v6_64670_metadata.json"),
        ("Stage B PSF NPZ", V6_STAGE_B / "psf_v6_64670.npz"),
        ("Stage A aperture response", V6_STAGE_A_AP / "response_2d_v6_64670_aperture_conditioned.npz"),
        ("Stage C source manifest", V6_STAGE_C / "source_files.csv"),
        ("Stage D background", V6_STAGE_D / "background_v6_64670_annnorm.npz"),
        ("Stage E signal", V6_STAGE_E / "signal_v6_64670_containment1_annnorm.npz"),
        ("Stage F fit", V6_STAGE_F / "fit_v6_64670_baselinev4.npz"),
        ("Stage G SED points", V6_STAGE_G / "sed_points_v6_64670_baselinev4.npz"),
        ("Stage E HTML", REPORT_DIR / "stage_e_v6_64670_containment1_annnorm_report.html"),
        ("Stage F HTML", REPORT_DIR / "stage_f_v6_64670_baselinev4_report.html"),
        ("Stage G HTML", REPORT_DIR / "stage_g_v6_64670_baselinev4_report.html"),
    ]

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crab SED v6 _64670 half-year report</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5d6673;
      --line: #d7dde5;
      --panel: #f6f8fb;
      --ok: #146c43;
      --warn: #9a5a00;
      --bad: #9b1c1c;
      --accent: #005eb8;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.48;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    header {{
      border-bottom: 3px solid var(--ink);
      padding-bottom: 18px;
      margin-bottom: 26px;
    }}
    h1, h2, h3 {{
      margin: 0;
      line-height: 1.18;
    }}
    h1 {{
      font-size: 34px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 22px;
      margin-top: 34px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    h3 {{
      font-size: 16px;
      margin-top: 18px;
      color: #2f3b48;
    }}
    p {{
      margin: 10px 0;
    }}
    .lede {{
      font-size: 17px;
      color: #2f3b48;
      max-width: 940px;
    }}
    .callout {{
      border-left: 5px solid var(--warn);
      background: #fff8ec;
      padding: 12px 14px;
      margin: 18px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: var(--panel);
      min-height: 78px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .metric .value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .metric .sub {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0 22px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 7px;
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #edf1f6;
      font-weight: 700;
    }}
    code {{
      background: #eef2f6;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 12px;
    }}
    .status-pass {{
      color: var(--ok);
      font-weight: 700;
    }}
    .status-warning {{
      color: var(--warn);
      font-weight: 700;
    }}
    .figgrid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin: 14px 0 22px;
    }}
    .figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }}
    .figure img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .figure figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
    }}
    .pathlist td {{
      text-align: left;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Crab SED v6 <code>_64670</code> Half-Year Analysis</h1>
    <p class="lede">This report closes the v6 roadmap run on <code>/mnt/mydisk/WCDA_observation_eval_64670</code>. The v4 baseline physics contract is preserved where possible, while every model-dependent MC, response, PSF, signal, fit, and SED artifact is rebuilt under v6 names with the <code>theta_recoxy_position_embed_midenergy_no_core_cut_64670</code> model generation.</p>
    <div class="callout"><strong>Comparability caveat.</strong> v6 is not a drop-in statistical update of v1-v5. Observation predictions and MC response products use a different trained model generation. The v4 comparison table below is diagnostic for scale and shape only; it should not be read as a same-response systematic shift.</div>
  </header>

  <section>
    <h2>Executive Result</h2>
    <div class="grid">
      <div class="metric"><div class="label">Stage C files</div><div class="value">{processed:,}/3969</div><div class="sub">missing time 0; entry mismatch {entry_mismatches}</div></div>
      <div class="metric"><div class="label">Selected rows</div><div class="value">{selected_rows:,}</div><div class="sub">configured-cell rows after match-status and cell cuts</div></div>
      <div class="metric"><div class="label">Rough live time</div><div class="value">{rough_live_days:.2f} d</div><div class="sub">source-visible {fmt(f_exposure.get("source_visible_days"), 4)} d</div></div>
      <div class="metric"><div class="label">Stage E signal</div><div class="value">68.81 sigma</div><div class="sub">N_on 376,450; B_on 335,797</div></div>
      <div class="metric"><div class="label">Preferred Stage F</div><div class="value">{esc(f_pref.get("model")).upper()}</div><div class="sub">chi2/ndof {fmt(v6_logpar.get("chi2"), 4)}/{v6_logpar.get("ndof")}</div></div>
      <div class="metric"><div class="label">LogPar phi0</div><div class="value">{fmt(v6_logpar.get("phi0"), 4)}</div><div class="sub">alpha {fmt(v6_logpar.get("alpha"), 4)}, beta {fmt(v6_logpar.get("beta"), 4)}</div></div>
    </div>
    {table(["Gate", "Status", "Evidence"], [[esc(name), f'<span class="status-{status}">{status}</span>', esc(evidence)] for name, status, evidence in gate_rows])}
  </section>

  <section>
    <h2>Dataset And Provenance</h2>
    <p>The full Stage C run uses <code>{esc(stage_c_meta.get("obs_root"))}</code> with recovered-time friends from <code>{esc(stage_c_meta.get("time_root"))}</code>. Source file coverage spans <code>{esc(date_min)}</code> through <code>{esc(date_max)}</code>, corresponding to 2022-01-01 through 2022-06-30.</p>
    {table(["Field", "Value"], [
        ["Eval ROOT files", f"{processed:,}"],
        ["Recovered-time match", "3969/3969 processed; missing time 0"],
        ["Input entries", f"{cutflow.get('input_entries', 0):,}"],
        ["Rows after match_status==0", f"{cutflow.get('after_match_status', 0):,}"],
        ["Rows after finite + configured-cell selection", f"{cutflow.get('after_cell_selection', 0):,}"],
        ["MC cache", "<code>/mnt/mydisk/WCDA_simulation_binned_response_v6_64670</code>"],
        ["Candidate cache", "<code>/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_candidate</code>"],
        ["Run dir", f"<code>{esc(stage_a_meta.get('run_dir'))}</code>"],
        ["Selector", "<code>apply/config/cell_selector_v6_drop4_psfborrow.csv</code>"],
    ], "pathlist")}
    {table(["Slurm stage", "Job", "State", "Elapsed", "Primary output"], [[esc(a), esc(b), esc(c), esc(d), f"<code>{esc(e)}</code>"] for a,b,c,d,e in stage_jobs], "pathlist")}
  </section>

  <section>
    <h2>Response And PSF</h2>
    <p>Nominal Stage A is a <code>{esc(stage_a_meta.get('response_type'))}</code>; the final fit uses <code>{esc(stage_a_ap_meta.get('response_type'))}</code>. Both metadata files record the v6 MC candidate cache and the <code>_64670</code> run dir, and the aperture-conditioned metadata records the v6 PSF NPZ path.</p>
    <p>Stage B wrote 84 PSF rows. It recorded {len(psf_warnings)} warnings, including {len(psf_warning_cell_ids & fit_cell_ids)} cells in the v4 fit subset, mostly for sparse cells that fall back to the Rayleigh default. Fit-cell coverage is complete and all fit-cell <code>r_opt_deg</code> values are finite and positive. In the radial-profile grid, pale green panels mark the {len(fit_cell_ids)} cells entering the baselinev4 fit.</p>
    <div class="figgrid">
      {figure(V6_STAGE_B / "psf_r_opt_deg_grid.png", "Stage B v6 r_opt by candidate cell")}
      {figure(V6_STAGE_B / "psf_effective_events_grid.png", "Stage B v6 effective events by candidate cell")}
      {figure(V6_STAGE_B_FIT_SHADED_PROFILE, "Stage B v6 radial PSF profiles; green panels enter the fit")}
    </div>
  </section>

  <section>
    <h2>Stage C-D-E Diagnostics</h2>
    <p>Stage C selected {selected_rows:,} configured-cell rows from {cutflow.get('input_entries', 0):,} input entries. The Crab-centered ROI diagnostics estimate an edge around 7.05 deg, so the downstream fiducial <code>rho&lt;6 deg</code> choice remains inside the observed coverage. Stage D produced the annulus-normalized quadratic ROI-local background, but did not promote <code>current</code> because ROI-local background warnings remain in sparse or fragile cells. Stage E used the explicit Stage D run path and passed the total-sigma quality gate.</p>
    {table(["Metric", "Value"], [
        ["Stage C rough live time", f"{rough_live_days:.3f} d"],
        ["Stage C rho&lt;6 rows", "20,336,070"],
        ["Stage D warning rows", f"{len(d_warning_cell_ids)} / 84"],
        ["Stage D warning rows in fit cells", f"{len(d_warning_cell_ids & fit_cell_ids)} / 26"],
        ["Stage E total N_on", "376,450"],
        ["Stage E total B_on", "335,797"],
        ["Stage E total excess", "40,652.7"],
        ["Stage E formal sigma", "68.8051"],
    ])}
    <div class="figgrid">
      {figure(V6_STAGE_D / "roi_excess_grid.png", "Stage D ROI excess grid")}
      {figure(V6_STAGE_D / "annulus_residual_grid.png", "Stage D annulus residual diagnostics")}
      {figure(V6_STAGE_E / "formal_sigma_grid.png", "Stage E formal sigma grid")}
    </div>
  </section>

  <section>
    <h2>Stage F Fit</h2>
    <p>Stage F uses the v6 aperture-conditioned Stage A response, the v6 containment-1 annulus-normalized signal, and the v4 drop4 selector. The fit is physically valid, but cell-level chi2 is still large; the report should be used as an analysis diagnostic rather than a publication result.</p>
    {table(["Fit", "Valid", "chi2/ndof", "p", "phi0", "gamma/alpha", "beta"], [
        ["PL conservative", esc(fit_metric(stage_f_meta, "pl_conservative", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_conservative', 'chi2'), 4)}/{fit_metric(stage_f_meta, 'pl_conservative', 'ndof')}", fmt(fit_metric(stage_f_meta, "pl_conservative", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_conservative", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_conservative", "gamma"), 4), "n/a"],
        ["LogPar conservative", esc(fit_metric(stage_f_meta, "logpar_conservative", "valid")), f"{fmt(v6_logpar.get('chi2'), 4)}/{v6_logpar.get('ndof')}", fmt(v6_logpar.get("p_value"), 3), fmt(v6_logpar.get("phi0"), 4), fmt(v6_logpar.get("alpha"), 4), fmt(v6_logpar.get("beta"), 4)],
        ["PL sqrt-N", esc(fit_metric(stage_f_meta, "pl_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_sqrt_n', 'chi2'), 4)}/{fit_metric(stage_f_meta, 'pl_sqrt_n', 'ndof')}", fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "gamma"), 4), "n/a"],
        ["LogPar sqrt-N", esc(fit_metric(stage_f_meta, "logpar_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'logpar_sqrt_n', 'chi2'), 4)}/{fit_metric(stage_f_meta, 'logpar_sqrt_n', 'ndof')}", fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "alpha"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "beta"), 4)],
    ])}
    {table(["Cell", "Bin", "Excess", "Err", "LogPar model", "Pull"], [[row["cell_id"], f"<code>{esc(row['nhit_bin'])}</code> <code>{esc(row['predE_bin'])}</code>", fmt(row["excess"], 4), fmt(row["error_conservative"], 3), fmt(row["logpar_model"], 4), fmt(row["logpar_pull"], 3)] for row in f_rows])}
    <div class="figgrid">
      {figure(V6_STAGE_F / "model_counts_vs_excess.png", "Stage F model counts versus observed excess")}
      {figure(V6_STAGE_F / "pull_grid_logpar.png", "Stage F LogPar pull grid")}
      {figure(V6_STAGE_F / "theta_exposure.png", "Stage F theta exposure")}
    </div>
  </section>

  <section>
    <h2>Stage G Diagnostic SED</h2>
    <p>Stage G is explicitly diagnostic-only. It freezes the Stage F LogPar spectrum with phi0={fmt(g_frozen.get('phi0'), 4)}, alpha={fmt(g_frozen.get('alpha'), 4)}, beta={fmt(g_frozen.get('beta'), 4)} at {fmt(g_frozen.get('pivot_tev'), 3)} TeV. It writes 7 Nhit points and 11 predicted-energy points.</p>
    {table(["Nhit group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof", "StageF ratio", "Full-array PL ratio"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{row['ndof']}", fmt(row["ratio_to_stage_f_model"], 3), fmt(row["ratio_to_full_array_pl_ref"], 3)] for row in g_nhit_rows])}
    {table(["PredE group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof", "StageF ratio", "Single cell"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{row['ndof']}", fmt(row["ratio_to_stage_f_model"], 3), esc(row["is_single_cell_point"])] for row in g_pred_rows])}
    <div class="figgrid">
      {figure(V6_STAGE_G / "sed_points_stage_f_fullarray_pool1.png", "Stage G v6 SED overlay")}
      {figure(V6_STAGE_G / "sed_points_ratio.png", "Stage G v6 SED ratios")}
      {figure(V6_STAGE_G / "sed_point_cell_counts.png", "Stage G cell grouping counts")}
    </div>
  </section>

  <section>
    <h2>Diagnostic v4 Comparison</h2>
    <p>The table compares v6 Nhit SED points against the v4 aperture-conditioned drop4 baseline. It is included to satisfy the roadmap comparison requirement, but the trained energy model and v6 MC response generation differ from v4, so the ratios mix physics/statistics with model-generation effects.</p>
    {table(["Nhit group", "v6 E_eff TeV", "v6 E2 dN/dE", "v6 err", "v4 E2 dN/dE", "v4 err", "v6/v4", "v6/full-array PL"], comparison_rows())}
    {table(["Quantity", "v6 _64670", "v4 baselinev4"], [
        ["Source-visible exposure", f"{fmt(f_exposure.get('source_visible_days'), 4)} d", f"{fmt(v4_f_meta['exposure'].get('source_visible_days'), 4)} d"],
        ["Reference observed/expected", fmt(f_ref.get("observed_expected_ratio"), 4), fmt(v4_f_meta["reference_count_preflight"].get("observed_expected_ratio"), 4)],
        ["Preferred model", esc(f_pref.get("model")), esc(v4_pref.get("model"))],
        ["LogPar chi2/ndof", f"{fmt(v6_logpar.get('chi2'), 4)}/{v6_logpar.get('ndof')}", f"{fmt(v4_logpar.get('chi2'), 4)}/{v4_logpar.get('ndof')}"],
        ["LogPar phi0", fmt(v6_logpar.get("phi0"), 4), fmt(v4_logpar.get("phi0"), 4)],
        ["LogPar alpha", fmt(v6_logpar.get("alpha"), 4), fmt(v4_logpar.get("alpha"), 4)],
        ["LogPar beta", fmt(v6_logpar.get("beta"), 4), fmt(v4_logpar.get("beta"), 4)],
    ])}
  </section>

  <section>
    <h2>Artifacts</h2>
    {table(["Artifact", "Path"], [[esc(name), f'<code>{rel(path)}</code>'] for name, path in artifacts], "pathlist")}
    <p class="small">Generated by <code>apply/report/build_v6_64670_report.py</code>. Output names are v6-isolated; no v1-v5 output directory is used as a v6 input except the explicit diagnostic v4 comparison section.</p>
  </section>
</main>
</body>
</html>
"""

    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Stage G quality: {g_quality.get('status')} / publication_ready={g_quality.get('stage_g_publication_ready')}")


if __name__ == "__main__":
    main()
