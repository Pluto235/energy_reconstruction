#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import uproot

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot normalized true-energy distributions in a single grid: columns=predE bins, rows=nhit bins (a,b,c...)."
    )
    parser.add_argument(
        "--binned-root",
        type=str,
        default="/mnt/mydisk/WCDA_simulation_binned_selectedcuts",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="apply/summary_selectedcuts/bin_counts.csv",
    )
    parser.add_argument(
        "--cell-selection-csv",
        type=str,
        default=None,
        help="Optional CSV with nhit_bin and predE_bin columns. If set, only these cells are plotted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="apply/plot",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="acceptable_true_energy_distribution_grid.png",
    )
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--branch", type=str, default="mc_energy")
    parser.add_argument("--x-min", type=float, default=2.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--num-bins", type=int, default=60)
    parser.add_argument(
        "--target-events-per-bin",
        type=int,
        default=10000,
        help="Max valid events to read per 2D bin for plotting speed.",
    )
    return parser.parse_args()


def pred_bin_key(pred_bin: str) -> float:
    if pred_bin.startswith("<"):
        return -1e9
    if pred_bin.startswith(">="):
        return 1e9
    lo = pred_bin.split(",")[0].strip("[")
    return float(lo)


def nhit_bin_key(nhit_bin: str) -> float:
    if nhit_bin.startswith("<"):
        return -1e9
    if nhit_bin.startswith(">="):
        return 1e9
    lo = nhit_bin.split(",")[0].strip("[")
    return float(lo)


def nhit_dir_name(nhit_bin: str) -> str:
    if nhit_bin.startswith("[") and nhit_bin.endswith(")"):
        lo, hi = nhit_bin[1:-1].split(",")
        return f"nhit_{lo}_{hi}"
    if nhit_bin.startswith("<"):
        return f"nhit_lt_{nhit_bin[1:]}"
    if nhit_bin.startswith(">="):
        return f"nhit_ge_{nhit_bin[2:]}"
    raise ValueError(f"Unsupported nhit_bin label: {nhit_bin}")


def pred_dir_name(pred_bin: str) -> str:
    if pred_bin.startswith("[") and pred_bin.endswith(")"):
        lo, hi = pred_bin[1:-1].split(",")
        return f"predE_{lo.replace('.', 'p')}_{hi.replace('.', 'p')}"
    if pred_bin.startswith("<"):
        return f"predE_lt_{pred_bin[1:].replace('.', 'p')}"
    if pred_bin.startswith(">="):
        return f"predE_ge_{pred_bin[2:].replace('.', 'p')}"
    raise ValueError(f"Unsupported predE_bin label: {pred_bin}")


def load_selected_cells(selection_csv: Path) -> set:
    selected = set()
    with selection_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected.add((row["nhit_bin"], row["predE_bin"]))
    if not selected:
        raise ValueError(f"No cells loaded from selection CSV: {selection_csv}")
    return selected


def load_acceptable_rows(summary_csv: Path, selected_cells: set = None) -> List[Dict[str, object]]:
    rows = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["statistics_level"] != "acceptable":
                continue
            key = (row["nhit_bin"], row["predE_bin"])
            if selected_cells is not None and key not in selected_cells:
                continue
            c = int(row["count"])
            if c <= 0:
                continue
            rows.append(
                {
                    "nhit_bin": row["nhit_bin"],
                    "predE_bin": row["predE_bin"],
                    "count": c,
                }
            )
    return rows


def hist_density_for_bin(
    bin_dir: Path,
    tree_name: str,
    branch: str,
    hist_edges: np.ndarray,
    target_events_per_bin: int,
) -> Tuple[np.ndarray, int]:
    counts = np.zeros(len(hist_edges) - 1, dtype=np.float64)
    total_valid = 0

    for root_file in sorted(bin_dir.glob("*.root")):
        with uproot.open(root_file) as f:
            b = f[tree_name][branch]
            if b.typename == "double":
                dtype = ">f8"
            elif b.typename == "float":
                dtype = ">f4"
            else:
                raise TypeError(f"Unsupported branch typename: {b.typename}")

            parts = []
            for bi in range(b.num_baskets):
                arr = np.frombuffer(b.basket(bi).data, dtype=dtype)
                if arr.size:
                    parts.append(arr)
            if not parts:
                continue
            values = np.concatenate(parts, dtype=np.float64)

        valid = values[np.isfinite(values) & (values > 0)]
        if valid.size == 0:
            continue

        loge = np.log10(valid)
        h, _ = np.histogram(loge, bins=hist_edges)
        counts += h
        total_valid += int(valid.size)
        if target_events_per_bin > 0 and total_valid >= target_events_per_bin:
            break

    if counts.sum() <= 0:
        return counts, total_valid
    widths = np.diff(hist_edges)
    density = counts / (counts.sum() * widths)
    return density, total_valid


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    cell_selection_csv = Path(args.cell_selection_csv).resolve() if args.cell_selection_csv else None
    binned_root = Path(args.binned_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cells = load_selected_cells(cell_selection_csv) if cell_selection_csv else None
    rows = load_acceptable_rows(summary_csv, selected_cells=selected_cells)
    if not rows:
        raise ValueError("No rows selected for plotting.")
    pred_bins = sorted({r["predE_bin"] for r in rows}, key=pred_bin_key)
    nhit_bins = sorted({r["nhit_bin"] for r in rows}, key=nhit_bin_key)
    letter_map = {nh: chr(ord("a") + i) for i, nh in enumerate(nhit_bins)}

    acceptable_set = {(r["nhit_bin"], r["predE_bin"]) for r in rows}
    summary_count = {(r["nhit_bin"], r["predE_bin"]): int(r["count"]) for r in rows}

    hist_edges = np.linspace(args.x_min, args.x_max, args.num_bins + 1)
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    n_rows = len(nhit_bins)
    n_cols = len(pred_bins)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols, 1.8 * n_rows),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    plot_meta = []
    cell_total = len(nhit_bins) * len(pred_bins)
    cell_idx = 0
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            cell_idx += 1
            ax = axes[i, j]
            key = (nhit_bin, pred_bin)
            if key in acceptable_set:
                print(f"[{cell_idx}/{cell_total}] {nhit_bin} x {pred_bin}", flush=True)
                bin_dir = binned_root / nhit_dir_name(nhit_bin) / pred_dir_name(pred_bin)
                density, used_events = hist_density_for_bin(
                    bin_dir=bin_dir,
                    tree_name=args.tree_name,
                    branch=args.branch,
                    hist_edges=hist_edges,
                    target_events_per_bin=args.target_events_per_bin,
                )
                if density.sum() > 0:
                    ax.plot(centers, density, color="#1f4e79", linewidth=1.2)
                    ax.fill_between(centers, density, color="#7aa6d1", alpha=0.25)
                ax.text(
                    0.03,
                    0.95,
                    f"n={summary_count[key]:,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    color="#333333",
                )
                plot_meta.append(
                    {
                        "nhit_letter": letter_map[nhit_bin],
                        "nhit_bin": nhit_bin,
                        "predE_bin": pred_bin,
                        "summary_count": summary_count[key],
                        "events_used_for_plot": used_events,
                    }
                )
            else:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#888888")

            ax.grid(alpha=0.2, linewidth=0.4)
            if i == 0:
                ax.set_title(pred_bin, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{letter_map[nhit_bin]}: {nhit_bin}", fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("log10(True E / GeV)", fontsize=8)

    title_prefix = "Selected v1 Cells" if cell_selection_csv else "Acceptable Bins"
    fig.suptitle(f"{title_prefix}: Normalized True-Energy Distributions per (predE bin, nhit bin)", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = output_dir / args.output_name
    fig.savefig(out_png)
    plt.close(fig)

    out_meta = output_dir / "acceptable_true_energy_distribution_grid_meta.json"
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_csv": str(summary_csv),
                "cell_selection_csv": str(cell_selection_csv) if cell_selection_csv else None,
                "binned_root": str(binned_root),
                "pred_bins": pred_bins,
                "nhit_letter_map": [{"letter": letter_map[n], "nhit_bin": n} for n in nhit_bins],
                "num_cells_with_acceptable_data": len(plot_meta),
                "target_events_per_bin": args.target_events_per_bin,
                "per_cell": plot_meta,
            },
            f,
            indent=2,
        )

    print(f"Wrote figure: {out_png}")
    print(f"Wrote metadata: {out_meta}")


if __name__ == "__main__":
    main()
