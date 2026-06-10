#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MAPS_NPZ = "apply/plot/crab_cell_skymaps/crab_v1_maps.npz"
DEFAULT_MAPS_METADATA = "apply/plot/crab_cell_skymaps/crab_v1_maps_meta.json"
DEFAULT_STAGE_F_SUMMARY = "apply/output/stage_f/current/fit_v1_summary.csv"
DEFAULT_OUTPUT_DIR = "apply/plot/low_energy_cell_diagnostics"
DEFAULT_REPORT_HTML = "apply/report/low_energy_cell_diagnostics.html"
DEFAULT_REPORT_MD = "apply/report/low_energy_cell_diagnostics.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-energy Crab cell skymap and sideband diagnostics.")
    parser.add_argument("--maps-npz", type=str, default=DEFAULT_MAPS_NPZ)
    parser.add_argument("--maps-metadata", type=str, default=DEFAULT_MAPS_METADATA)
    parser.add_argument("--stage-f-summary", type=str, default=DEFAULT_STAGE_F_SUMMARY)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-html", type=str, default=DEFAULT_REPORT_HTML)
    parser.add_argument("--report-md", type=str, default=DEFAULT_REPORT_MD)
    parser.add_argument("--cell-ids", type=str, default="1-7")
    parser.add_argument("--source-radius-deg", type=float, default=2.0)
    parser.add_argument("--sideband-max-abs-x-deg", type=float, default=5.0)
    parser.add_argument("--profile-half-width-deg", type=float, default=1.0)
    parser.add_argument("--profile-sideband-min-deg", type=float, default=5.0)
    parser.add_argument("--sigma-vmax", type=float, default=12.0)
    parser.add_argument("--summary-csv-name", type=str, default="low_energy_cell_diagnostics.csv")
    return parser.parse_args()


def parse_cell_ids(spec: str) -> List[int]:
    ids: List[int] = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        if "-" in part:
            low_s, high_s = part.split("-", 1)
            low, high = int(low_s), int(high_s)
            if high < low:
                raise ValueError(f"Invalid cell id range: {part}")
            ids.extend(range(low, high + 1))
        else:
            ids.append(int(part))
    if not ids:
        raise ValueError("--cell-ids selected no cells")
    return list(dict.fromkeys(ids))


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_stage_f_rows(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = int(row["cell_id"])
            converted: Dict[str, object] = dict(row)
            for key in [
                "N_on",
                "B_on",
                "excess",
                "error_conservative",
                "pl_model",
                "pl_pull",
                "logpar_model",
                "logpar_pull",
                "preferred_counts",
                "preferred_pull",
            ]:
                if key in row and row[key] != "":
                    converted[key] = float(row[key])
            rows[cell_id] = converted
    return rows


def robust_vmax(values: np.ndarray, percentile: float = 99.5) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(finite, percentile))
    return value if math.isfinite(value) and value > 0.0 else float(finite.max())


def sideband_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    source_radius_deg: float,
    sideband_max_abs_x_deg: float,
) -> np.ndarray:
    rho = np.hypot(xx, yy)
    return (rho >= float(source_radius_deg)) & (np.abs(xx) < float(sideband_max_abs_x_deg))


def compute_metrics(
    *,
    counts: np.ndarray,
    background: np.ndarray,
    approx_sigma: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    selected_ids: Sequence[int],
    meta: Dict[str, object],
    stage_f_rows: Dict[int, Dict[str, object]],
    source_radius_deg: float,
    sideband_max_abs_x_deg: float,
) -> Tuple[List[int], List[Dict[str, object]]]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    rho = np.hypot(xx, yy)
    source_mask = rho < float(source_radius_deg)
    sb_mask = sideband_mask(
        xx,
        yy,
        source_radius_deg=float(source_radius_deg),
        sideband_max_abs_x_deg=float(sideband_max_abs_x_deg),
    )
    left_mask = sb_mask & (xx < -float(source_radius_deg))
    right_mask = sb_mask & (xx > float(source_radius_deg))

    meta_cells = {
        int(cell.get("cell_id")): cell
        for cell in meta.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_id") is not None
    }
    selected_indices: List[int] = []
    rows: List[Dict[str, object]] = []
    id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    for cell_id in selected_ids:
        if cell_id not in id_to_index:
            raise ValueError(f"Cell {cell_id} is not present in {cell_ids.tolist()}")
        idx = id_to_index[cell_id]
        selected_indices.append(idx)
        count_map = counts[idx].astype(np.float64)
        bg_map = background[idx].astype(np.float64)
        sigma_map = approx_sigma[idx].astype(np.float64)

        central_counts = float(np.sum(count_map[source_mask]))
        central_bg = float(np.sum(bg_map[source_mask]))
        central_excess = central_counts - central_bg
        central_sigma = central_excess / math.sqrt(max(central_bg, 1.0))

        side_values = count_map[sb_mask]
        side_mean = float(np.mean(side_values)) if side_values.size else float("nan")
        side_std = float(np.std(side_values)) if side_values.size else float("nan")
        side_cv = side_std / side_mean if side_mean > 0 else float("nan")
        left_sum = float(np.sum(count_map[left_mask]))
        right_sum = float(np.sum(count_map[right_mask]))
        left_right_asym = (left_sum - right_sum) / (left_sum + right_sum) if (left_sum + right_sum) > 0 else float("nan")

        central_sigma_values = sigma_map[source_mask]
        peak_flat = int(np.nanargmax(central_sigma_values)) if central_sigma_values.size else 0
        source_indices = np.argwhere(source_mask)
        peak_y, peak_x = source_indices[peak_flat] if source_indices.size else (0, 0)
        stage_f = stage_f_rows.get(cell_id, {})
        stage_f_pull = float(stage_f.get("pl_pull", float("nan")))
        stage_f_model = float(stage_f.get("pl_model", float("nan")))
        stage_f_excess = float(stage_f.get("excess", float("nan")))
        rows.append(
            {
                "cell_id": int(cell_id),
                "nhit_bin": str(nhit_bins[idx]),
                "predE_bin": str(pred_bins[idx]),
                "roi_events": int(meta_cells.get(cell_id, {}).get("roi_events", int(np.sum(count_map)))),
                "central_counts_r_lt_source_radius": central_counts,
                "central_sideband_background": central_bg,
                "central_excess_like": central_excess,
                "central_approx_sigma": central_sigma,
                "sideband_mean_counts_per_pixel": side_mean,
                "sideband_std_counts_per_pixel": side_std,
                "sideband_cv": side_cv,
                "sideband_left_right_asymmetry": left_right_asym,
                "peak_sigma_in_source_region": float(np.nanmax(central_sigma_values)) if central_sigma_values.size else float("nan"),
                "peak_x_deg": float(x_centers[int(peak_x)]),
                "peak_y_deg": float(y_centers[int(peak_y)]),
                "stage_f_excess": stage_f_excess,
                "stage_f_pl_model": stage_f_model,
                "stage_f_excess_over_model": stage_f_excess / stage_f_model if stage_f_model > 0.0 else float("nan"),
                "stage_f_pl_pull": stage_f_pull,
                "diagnostic_flag": (
                    "exclude_from_stage_g_baseline"
                    if abs(stage_f_pull) >= 5.0 or abs(central_sigma) >= 8.0
                    else "review"
                ),
            }
        )
    return selected_indices, rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_map_grid(
    *,
    counts: np.ndarray,
    approx_sigma: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    selected_indices: Sequence[int],
    source_radius_deg: float,
    sideband_max_abs_x_deg: float,
    sigma_vmax: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    n_cols = len(selected_indices)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.45 * n_cols, 5.25), dpi=170, squeeze=False)
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    counts_vmax = robust_vmax(counts[np.asarray(selected_indices)], 99.3)
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle_x = float(source_radius_deg) * np.cos(theta)
    circle_y = float(source_radius_deg) * np.sin(theta)

    for col, idx in enumerate(selected_indices):
        title = f"cell {int(cell_ids[idx])}\n{nhit_bins[idx]} x {pred_bins[idx]}"
        ax = axes[0, col]
        im0 = ax.imshow(
            counts[idx],
            origin="lower",
            extent=extent,
            cmap="viridis",
            norm=Normalize(vmin=0.0, vmax=counts_vmax),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title, fontsize=7.5)
        ax.plot([0.0], [0.0], marker="+", color="white", markersize=7, markeredgewidth=1.2)
        ax.plot(circle_x, circle_y, color="white", linewidth=0.7, alpha=0.9)
        ax.axvline(-sideband_max_abs_x_deg, color="white", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.axvline(sideband_max_abs_x_deg, color="white", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.tick_params(labelsize=6.2, length=2)
        if col == 0:
            ax.set_ylabel("counts\nDec offset [deg]", fontsize=7.2)

        ax = axes[1, col]
        im1 = ax.imshow(
            approx_sigma[idx],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-float(sigma_vmax), vmax=float(sigma_vmax)),
            interpolation="nearest",
            aspect="equal",
        )
        ax.plot([0.0], [0.0], marker="+", color="black", markersize=7, markeredgewidth=1.2)
        ax.plot(circle_x, circle_y, color="black", linewidth=0.7, alpha=0.9)
        ax.axvline(-sideband_max_abs_x_deg, color="black", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.axvline(sideband_max_abs_x_deg, color="black", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_xlabel("RA offset cos(dec) [deg]", fontsize=7.2)
        ax.tick_params(labelsize=6.2, length=2)
        if col == 0:
            ax.set_ylabel("approx sigma\nDec offset [deg]", fontsize=7.2)

    fig.colorbar(im0, ax=axes[0, :].ravel().tolist(), shrink=0.72, pad=0.008).set_label("counts / pixel", fontsize=7.2)
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.72, pad=0.008).set_label("approx sigma", fontsize=7.2)
    fig.suptitle("Low-energy Crab cell skymap and sideband quicklook", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    fig.savefig(output_path)
    plt.close(fig)


def profile_baseline(profile: np.ndarray, centers: np.ndarray, min_abs_deg: float) -> np.ndarray:
    mask = np.abs(centers) > float(min_abs_deg)
    if not np.any(mask):
        raise ValueError("Profile sideband mask is empty")
    return np.mean(profile[:, mask], axis=1)


def plot_profile_grid(
    *,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    selected_indices: Sequence[int],
    profile_half_width_deg: float,
    profile_sideband_min_deg: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_band = np.abs(y_centers) < float(profile_half_width_deg)
    x_band = np.abs(x_centers) < float(profile_half_width_deg)
    selected = np.asarray(selected_indices, dtype=np.int64)
    ra_profile = counts[selected][:, y_band, :].sum(axis=1).astype(np.float64)
    dec_profile = counts[selected][:, :, x_band].sum(axis=2).astype(np.float64)
    ra_base = profile_baseline(ra_profile, x_centers, profile_sideband_min_deg)
    dec_base = profile_baseline(dec_profile, y_centers, profile_sideband_min_deg)

    n_cols = len(selected_indices)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.45 * n_cols, 5.1), dpi=170, squeeze=False)
    for col, idx in enumerate(selected_indices):
        title = f"cell {int(cell_ids[idx])}\n{nhit_bins[idx]} x {pred_bins[idx]}"
        ax = axes[0, col]
        ax.step(x_centers, ra_profile[col], where="mid", color="#1f4e79", linewidth=0.9)
        ax.axhline(ra_base[col], color="#c9501a", linestyle="--", linewidth=0.8)
        ax.axvline(0.0, color="#333333", linewidth=0.7)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=6.2, length=2)
        if col == 0:
            ax.set_ylabel("RA profile\ncounts / x bin", fontsize=7.2)

        ax = axes[1, col]
        ax.step(y_centers, dec_profile[col], where="mid", color="#1f4e79", linewidth=0.9)
        ax.axhline(dec_base[col], color="#c9501a", linestyle="--", linewidth=0.8)
        ax.axvline(0.0, color="#333333", linewidth=0.7)
        ax.set_xlabel("offset [deg]", fontsize=7.2)
        ax.tick_params(labelsize=6.2, length=2)
        if col == 0:
            ax.set_ylabel("Dec profile\ncounts / y bin", fontsize=7.2)

    fig.suptitle(
        f"Low-energy RA/Dec profiles, center band < {profile_half_width_deg:g} deg, baseline |offset| > {profile_sideband_min_deg:g} deg",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path)
    plt.close(fig)


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 1.0e5 or (abs(number) < 1.0e-3 and number != 0.0):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def write_reports(
    *,
    rows: Sequence[Dict[str, object]],
    output_dir: Path,
    maps_png: Path,
    profiles_png: Path,
    summary_csv: Path,
    report_md: Path,
    report_html: Path,
    source_radius_deg: float,
    sideband_max_abs_x_deg: float,
) -> None:
    md_lines = [
        "# Low-Energy Cell Skymap / Sideband Diagnostics",
        "",
        "This diagnostic focuses on v1 cells 1-7, which dominate the Stage F chi2 after the S0 unit fix.",
        "The sideband quicklook is not a replacement for Stage D physics background.",
        "",
        f"- Source-region quicklook: rho < {source_radius_deg:g} deg",
        f"- Sideband quicklook: same Dec strip, rho >= {source_radius_deg:g} deg, |x| < {sideband_max_abs_x_deg:g} deg",
        f"- Summary CSV: `{summary_csv}`",
        "",
        f"![Low-energy maps]({Path('..') / maps_png.relative_to(REPO_ROOT / 'apply')})",
        "",
        f"![Low-energy profiles]({Path('..') / profiles_png.relative_to(REPO_ROOT / 'apply')})",
        "",
        "| cell | Nhit | predE | central excess-like | central approx sigma | sideband CV | left-right asym | Stage F excess/model | Stage F pull | flag |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
            f"{fmt(row['central_excess_like'], 5)} | {fmt(row['central_approx_sigma'], 4)} | "
            f"{fmt(row['sideband_cv'], 4)} | {fmt(row['sideband_left_right_asymmetry'], 4)} | "
            f"{fmt(row['stage_f_excess_over_model'], 4)} | {fmt(row['stage_f_pl_pull'], 4)} | "
            f"{row['diagnostic_flag']} |"
        )
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['cell_id']))}</td>"
        f"<td>{html.escape(str(row['nhit_bin']))}</td>"
        f"<td>{html.escape(str(row['predE_bin']))}</td>"
        f"<td>{fmt(row['central_excess_like'], 5)}</td>"
        f"<td>{fmt(row['central_approx_sigma'], 4)}</td>"
        f"<td>{fmt(row['sideband_cv'], 4)}</td>"
        f"<td>{fmt(row['sideband_left_right_asymmetry'], 4)}</td>"
        f"<td>{fmt(row['stage_f_excess_over_model'], 4)}</td>"
        f"<td>{fmt(row['stage_f_pl_pull'], 4)}</td>"
        f"<td>{html.escape(str(row['diagnostic_flag']))}</td>"
        "</tr>"
        for row in rows
    )
    maps_rel = Path("..") / maps_png.relative_to(REPO_ROOT / "apply")
    profiles_rel = Path("..") / profiles_png.relative_to(REPO_ROOT / "apply")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Low-Energy Cell Diagnostics</title>
<style>
body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:#1f2933; background:#f7f8fa; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 18px 44px; }}
h1 {{ margin:0 0 8px; font-size:28px; }}
p {{ line-height:1.55; }}
.note {{ padding:12px 14px; border-left:4px solid #c9501a; background:#fff; border:1px solid #d9dee7; }}
img {{ display:block; max-width:100%; height:auto; margin:18px 0; border:1px solid #d9dee7; background:#fff; }}
table {{ width:100%; border-collapse:collapse; margin-top:16px; background:#fff; font-size:13px; }}
th, td {{ border:1px solid #d9dee7; padding:7px 8px; text-align:right; }}
th:nth-child(2), th:nth-child(3), th:nth-child(10), td:nth-child(2), td:nth-child(3), td:nth-child(10) {{ text-align:left; }}
th {{ background:#eef2f6; }}
code {{ background:#eef2f6; padding:2px 4px; border-radius:4px; }}
</style>
</head>
<body>
<main>
<h1>Low-Energy Cell Skymap / Sideband Diagnostics</h1>
<p class="note">This diagnostic focuses on v1 cells 1-7 after the S0 unit fix. The sideband quicklook is diagnostic only and does not replace Stage D.</p>
<p>Source-region quicklook: <code>rho &lt; {source_radius_deg:g} deg</code>. Sideband: same Dec strip, <code>rho &gt;= {source_radius_deg:g} deg</code>, <code>|x| &lt; {sideband_max_abs_x_deg:g} deg</code>.</p>
<img src="{html.escape(str(maps_rel))}" alt="Low-energy skymap grid">
<img src="{html.escape(str(profiles_rel))}" alt="Low-energy sideband profiles">
<table>
<thead><tr><th>cell</th><th>Nhit</th><th>predE</th><th>central excess-like</th><th>central approx sigma</th><th>sideband CV</th><th>left-right asym</th><th>Stage F excess/model</th><th>Stage F pull</th><th>flag</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
<p>Summary CSV: <code>{html.escape(str(summary_csv))}</code></p>
</main>
</body>
</html>
"""
    report_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    maps_npz = Path(args.maps_npz).resolve()
    maps_metadata = Path(args.maps_metadata).resolve()
    stage_f_summary = Path(args.stage_f_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_html = Path(args.report_html).resolve()
    report_md = Path(args.report_md).resolve()
    report_html.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    selected_ids = parse_cell_ids(args.cell_ids)
    meta = load_json(maps_metadata)
    stage_f_rows = load_stage_f_rows(stage_f_summary)
    with np.load(maps_npz) as data:
        required = {"counts", "sideband_background", "approx_sigma", "x_edges", "y_edges", "cell_id", "nhit_bin", "predE_bin"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{maps_npz} is missing required arrays: {sorted(missing)}")
        counts = data["counts"]
        background = data["sideband_background"]
        approx_sigma = data["approx_sigma"]
        x_edges = data["x_edges"]
        y_edges = data["y_edges"]
        cell_ids = data["cell_id"].astype(np.int64)
        nhit_bins = data["nhit_bin"].astype(str)
        pred_bins = data["predE_bin"].astype(str)

    selected_indices, rows = compute_metrics(
        counts=counts,
        background=background,
        approx_sigma=approx_sigma,
        x_edges=x_edges,
        y_edges=y_edges,
        cell_ids=cell_ids,
        nhit_bins=nhit_bins,
        pred_bins=pred_bins,
        selected_ids=selected_ids,
        meta=meta,
        stage_f_rows=stage_f_rows,
        source_radius_deg=float(args.source_radius_deg),
        sideband_max_abs_x_deg=float(args.sideband_max_abs_x_deg),
    )

    summary_csv = output_dir / args.summary_csv_name
    maps_png = output_dir / "low_energy_skymap_sideband_grid.png"
    profiles_png = output_dir / "low_energy_sideband_profiles.png"
    write_csv(summary_csv, rows)
    plot_map_grid(
        counts=counts,
        approx_sigma=approx_sigma,
        x_edges=x_edges,
        y_edges=y_edges,
        cell_ids=cell_ids,
        nhit_bins=nhit_bins,
        pred_bins=pred_bins,
        selected_indices=selected_indices,
        source_radius_deg=float(args.source_radius_deg),
        sideband_max_abs_x_deg=float(args.sideband_max_abs_x_deg),
        sigma_vmax=float(args.sigma_vmax),
        output_path=maps_png,
    )
    plot_profile_grid(
        counts=counts,
        x_edges=x_edges,
        y_edges=y_edges,
        cell_ids=cell_ids,
        nhit_bins=nhit_bins,
        pred_bins=pred_bins,
        selected_indices=selected_indices,
        profile_half_width_deg=float(args.profile_half_width_deg),
        profile_sideband_min_deg=float(args.profile_sideband_min_deg),
        output_path=profiles_png,
    )
    write_reports(
        rows=rows,
        output_dir=output_dir,
        maps_png=maps_png,
        profiles_png=profiles_png,
        summary_csv=summary_csv,
        report_md=report_md,
        report_html=report_html,
        source_radius_deg=float(args.source_radius_deg),
        sideband_max_abs_x_deg=float(args.sideband_max_abs_x_deg),
    )
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote figure: {maps_png}")
    print(f"Wrote figure: {profiles_png}")
    print(f"Wrote report: {report_html}")
    print(f"Wrote report: {report_md}")


if __name__ == "__main__":
    main()
