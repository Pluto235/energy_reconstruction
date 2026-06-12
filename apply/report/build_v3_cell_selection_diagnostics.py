#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 cell-selection diagnostic figures from Stage A response.")
    parser.add_argument("--response-npz", type=str, default="apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz")
    parser.add_argument("--ledger-csv", type=str, default="apply/config/cell_ledger_v3_candidate.csv")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline.csv")
    parser.add_argument("--systematics-selector-csv", type=str, default="apply/config/cell_selector_v3_systematics.csv")
    parser.add_argument("--high-energy-selector-csv", type=str, default="apply/config/cell_selector_v3_high_energy_probes.csv")
    parser.add_argument("--output-dir", type=str, default="apply/report/assets/v3-cell-selection")
    parser.add_argument("--html", type=str, default="apply/report/v3_cell_selection_diagnostics.html")
    return parser.parse_args()


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def rel(path: str | Path, start: Path) -> str:
    p = abs_path(path)
    try:
        return os.path.relpath(p, start=start.resolve())
    except ValueError:
        return str(p)


def h(value: object) -> str:
    return html.escape(str(value))


def parse_interval(label: str) -> Tuple[Optional[float], Optional[float]]:
    label = label.strip()
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
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def selector_ids(rows: Sequence[Dict[str, str]]) -> set[int]:
    ids: set[int] = set()
    for row in rows:
        if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "include"}:
            ids.add(int(row["cell_id"]))
    return ids


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_matrix(rows: Sequence[Dict[str, str]], values: Dict[int, float]) -> Tuple[np.ndarray, List[str], List[str]]:
    nhit_bins = sorted({row["nhit_bin"] for row in rows}, key=interval_key)
    pred_bins = sorted({row["predE_bin"] for row in rows}, key=interval_key)
    matrix = np.full((len(nhit_bins), len(pred_bins)), np.nan, dtype=np.float64)
    by_key = {(row["nhit_bin"], row["predE_bin"]): row for row in rows}
    for i, nhit in enumerate(nhit_bins):
        for j, pred in enumerate(pred_bins):
            row = by_key.get((nhit, pred))
            if row is not None:
                matrix[i, j] = float(values.get(int(row["cell_id"]), np.nan))
    return matrix, nhit_bins, pred_bins


def plot_central_mask(rows: Sequence[Dict[str, str]], baseline_ids: set[int], path: Path) -> None:
    plt = setup_matplotlib()
    values = {}
    labels = {}
    for row in rows:
        cell_id = int(row["cell_id"])
        central = str(row.get("central99_flag", "0")).strip() in {"1", "true", "True"}
        value = 1.0 if central else 0.0
        if cell_id in baseline_ids:
            value = 2.0
        values[cell_id] = value
        labels[cell_id] = int(row.get("mc_count") or 0)
    matrix, nhit_bins, pred_bins = build_matrix(rows, values)
    fig, ax = plt.subplots(figsize=(1.28 * len(pred_bins) + 2.8, 0.62 * len(nhit_bins) + 2.2), dpi=150)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_xlabel("log10(E_pred / GeV) bin")
    ax.set_ylabel("Nhit bin")
    ax.set_title("v3 MC central-99 mask and frozen baseline selector")
    by_id = {int(row["cell_id"]): row for row in rows}
    for i, nhit in enumerate(nhit_bins):
        for j, pred in enumerate(pred_bins):
            match = [row for row in rows if row["nhit_bin"] == nhit and row["predE_bin"] == pred]
            if not match:
                continue
            cell_id = int(match[0]["cell_id"])
            ax.text(j, i, f"{cell_id}\n{labels[cell_id]:.2g}", ha="center", va="center", fontsize=6.2, color="white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["excluded", "central99", "baseline"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_selection_matrix(
    rows: Sequence[Dict[str, str]],
    baseline_ids: set[int],
    systematics_ids: set[int],
    high_energy_ids: set[int],
    path: Path,
) -> None:
    plt = setup_matplotlib()
    values: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for row in rows:
        cell_id = int(row["cell_id"])
        counts[cell_id] = int(row.get("mc_count") or 0)
        value = 0.0
        if cell_id in systematics_ids:
            value = 1.0
        if cell_id in high_energy_ids:
            value = 2.0
        if cell_id in baseline_ids:
            value = 3.0
        values[cell_id] = value
    matrix, nhit_bins, pred_bins = build_matrix(rows, values)
    fig, ax = plt.subplots(figsize=(1.28 * len(pred_bins) + 2.8, 0.62 * len(nhit_bins) + 2.2), dpi=150)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_xlabel("log10(E_pred / GeV) bin")
    ax.set_ylabel("Nhit bin")
    ax.set_title("v3 candidate / baseline / probe / excluded cell matrix")
    for i, nhit in enumerate(nhit_bins):
        for j, pred in enumerate(pred_bins):
            match = [row for row in rows if row["nhit_bin"] == nhit and row["predE_bin"] == pred]
            if not match:
                continue
            cell_id = int(match[0]["cell_id"])
            ax.text(j, i, f"{cell_id}\n{counts[cell_id]:.2g}", ha="center", va="center", fontsize=6.2, color="white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["excluded", "systematics", "high-energy", "baseline"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_true_energy_overlay(response: Dict[str, np.ndarray], rows: Sequence[Dict[str, str]], path: Path) -> None:
    plt = setup_matplotlib()
    cell_ids = np.asarray(response["cell_id"], dtype=np.int32)
    id_to_idx = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
    centers = 0.5 * (loge_edges[:-1] + loge_edges[1:])
    widths = np.diff(loge_edges)
    numerator = np.asarray(response["numerator_sumw"], dtype=np.float64)
    pred_bins = sorted({row["predE_bin"] for row in rows}, key=interval_key)
    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=150)
    cmap = plt.get_cmap("tab20")
    for idx, pred in enumerate(pred_bins):
        selected = [row for row in rows if row["predE_bin"] == pred and int(row["cell_id"]) in id_to_idx]
        if not selected:
            continue
        hist = np.zeros(centers.shape, dtype=np.float64)
        for row in selected:
            cell_idx = id_to_idx[int(row["cell_id"])]
            hist += numerator[cell_idx].sum(axis=1)
        area = float(np.sum(hist * widths))
        if area <= 0:
            continue
        density = hist / area
        ax.plot(centers, density, lw=1.6, color=cmap(idx % cmap.N), label=pred)
        cdf = np.cumsum(hist)
        if cdf[-1] > 0:
            q16, q50, q84 = np.interp([0.16, 0.50, 0.84], cdf / cdf[-1], centers)
            ax.axvline(q50, color=cmap(idx % cmap.N), lw=0.8, alpha=0.35)
            ax.fill_betweenx([0, np.nanmax(density) if density.size else 1], q16, q84, color=cmap(idx % cmap.N), alpha=0.04)
    ax.set_xlabel("log10(E_true / GeV)")
    ax.set_ylabel("normalized MC counts density")
    ax.set_title("v3 predicted-energy bins: normalized true-energy distributions")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7, title="predE bin")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_html_report(path: Path, payload: Dict[str, object]) -> None:
    report_dir = path.parent
    figures = [
        ("v3 cell selection matrix", payload["selection_matrix_png"]),
        ("MC central-99% selection mask", payload["central99_mask_png"]),
        ("MC normalized true-energy distribution overlay", payload["mc_true_energy_overlay_png"]),
    ]
    figure_html = []
    for caption, figure_path in figures:
        p = abs_path(str(figure_path))
        if not p.exists():
            continue
        figure_html.append(
            f'<figure><img src="{h(rel(p, report_dir))}" alt="{h(caption)}"><figcaption>{h(caption)}</figcaption></figure>'
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v3 Cell Selection Diagnostics</title>
<style>
:root {{ --bg:#f7f8f9; --fg:#182027; --muted:#53606a; --panel:#ffffff; --border:#d7dee3; --accent:#005f73; --code:#edf1f3; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans CJK SC","Microsoft YaHei",sans-serif; line-height:1.58; }}
main {{ max-width:1240px; margin:0 auto; padding:38px 20px 66px; }}
header {{ border-bottom:1px solid var(--border); padding-bottom:20px; margin-bottom:26px; }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
h1 {{ margin:8px 0 12px; font-size:clamp(30px,4vw,46px); line-height:1.12; }}
h2 {{ margin:34px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--border); font-size:23px; }}
.lead {{ max-width:940px; color:var(--muted); font-size:17px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.metric {{ min-height:104px; padding:15px; border:1px solid var(--border); background:var(--panel); border-radius:8px; }}
.label {{ color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
.value {{ margin-top:8px; font-size:26px; font-weight:800; }}
code {{ padding:2px 5px; background:var(--code); border-radius:4px; font-size:13px; }}
.figure-grid {{ display:grid; grid-template-columns:1fr; gap:18px; }}
figure {{ margin:0; padding:12px; border:1px solid var(--border); background:var(--panel); border-radius:8px; }}
img {{ display:block; width:100%; height:auto; background:#fff; border-radius:4px; }}
figcaption {{ margin-top:8px; color:var(--muted); font-size:13px; }}
footer {{ margin-top:42px; padding-top:16px; border-top:1px solid var(--border); color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
@media (max-width:800px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">LHAASO-WCDA · Crab SED v3</div>
    <h1>Cell Selection Diagnostics</h1>
    <p class="lead">Candidate grid starts at <code>Nhit [125,200)</code>, uses the v3 mixed predicted-energy bins, and freezes the baseline selector from MC/prefit information rather than Crab on-source excess or fit residuals.</p>
  </header>
  <section class="metrics">
    <div class="metric"><div class="label">candidate</div><div class="value">{h(payload['candidate_cells'])}</div></div>
    <div class="metric"><div class="label">baseline</div><div class="value">{h(payload['baseline_cells'])}</div></div>
    <div class="metric"><div class="label">systematics</div><div class="value">{h(payload['systematics_cells'])}</div></div>
    <div class="metric"><div class="label">high-energy probes</div><div class="value">{h(payload['high_energy_probe_cells'])}</div></div>
  </section>
  <section>
    <h2>Figures</h2>
    <div class="figure-grid">{''.join(figure_html) or '<p>Figures are not available.</p>'}</div>
  </section>
  <footer>Generated from Stage A response and selector CSVs. Metadata JSON: <code>{h(rel(payload['metadata_json'], report_dir))}</code>.</footer>
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in html_text.splitlines()) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = abs_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(abs_path(args.ledger_csv))
    baseline = read_csv(abs_path(args.baseline_selector_csv))
    systematics = read_csv(abs_path(args.systematics_selector_csv))
    high_energy = read_csv(abs_path(args.high_energy_selector_csv))
    baseline_ids = selector_ids(baseline)
    systematics_ids = selector_ids(systematics)
    high_energy_ids = selector_ids(high_energy)
    matrix_png = output_dir / "v3_cell_selection_matrix.png"
    central_png = output_dir / "v3_central99_mask.png"
    overlay_png = output_dir / "v3_mc_true_energy_overlay.png"
    plot_selection_matrix(rows, baseline_ids, systematics_ids, high_energy_ids, matrix_png)
    plot_central_mask(rows, baseline_ids, central_png)
    with np.load(abs_path(args.response_npz), allow_pickle=False) as data:
        response = {key: data[key] for key in data.files}
    plot_true_energy_overlay(response, rows, overlay_png)

    payload = {
        "selection_matrix_png": str(matrix_png),
        "central99_mask_png": str(central_png),
        "mc_true_energy_overlay_png": str(overlay_png),
        "html": str(abs_path(args.html)),
        "metadata_json": str(output_dir / "v3_cell_selection_diagnostics_meta.json"),
        "candidate_cells": len(rows),
        "baseline_cells": len(baseline_ids),
        "systematics_cells": len(systematics_ids),
        "high_energy_probe_cells": len(high_energy_ids),
    }
    (output_dir / "v3_cell_selection_diagnostics_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html_report(abs_path(args.html), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
