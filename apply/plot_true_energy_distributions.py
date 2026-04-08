#!/usr/bin/env python
import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


@dataclass(frozen=True)
class BinSpec:
    nhit_bin: str
    predE_bin: str
    count: int
    formal_nhit_bin: bool
    statistics_level: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot normalized true-energy distributions for acceptable 2D apply bins."
    )
    parser.add_argument(
        "--binned-root",
        type=str,
        default="/mnt/mydisk/WCDA_simulation_binned_selectedcuts",
        help="Root directory containing nhit/predE-binned ROOT outputs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="apply/summary_selectedcuts/bin_counts.csv",
        help="CSV summary used to choose which bins are worth plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="apply/plot/selectedcuts_true_energy_distributions",
        help="Directory where per-bin plots and metadata will be written.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="mc_energy",
        help="True-energy branch to read from each ROOT file.",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default="t_eventout",
        help="ROOT tree name.",
    )
    parser.add_argument(
        "--min-statistics-level",
        type=str,
        default="acceptable",
        choices=["acceptable", "low statistics", "very low statistics"],
        help="Minimum summary label to include.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=0.0,
        help="Lower edge for log10(true energy / GeV) histogram.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=6.0,
        help="Upper edge for log10(true energy / GeV) histogram.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=80,
        help="Number of histogram bins in log10(true energy / GeV).",
    )
    parser.add_argument(
        "--concat-file-chunk",
        type=int,
        default=500,
        help="Unused in basket-decoding mode; kept for backward compatibility.",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        help="Only process the first N selected bins for quick validation.",
    )
    parser.add_argument(
        "--target-events-per-bin",
        type=int,
        default=200000,
        help="Stop reading more files in a bin once this many valid events are accumulated.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20000,
        help="Minimum bin event count from summary CSV to be considered 'enough statistics'.",
    )
    return parser.parse_args()


def stats_rank(label: str) -> int:
    mapping = {
        "very low statistics": 0,
        "low statistics": 1,
        "acceptable": 2,
    }
    return mapping[label]


def slugify_bin_label(label: str) -> str:
    return (
        label.replace("[", "")
        .replace(")", "")
        .replace(">=", "ge_")
        .replace("<", "lt_")
        .replace(",", "_")
        .replace(".", "p")
        .replace(" ", "")
    )


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


def load_bins(summary_csv: Path, min_statistics_level: str, min_count: int) -> List[BinSpec]:
    min_rank = stats_rank(min_statistics_level)
    bins: List[BinSpec] = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["count"]) <= 0:
                continue
            if stats_rank(row["statistics_level"]) < min_rank:
                continue
            if int(row["count"]) < min_count:
                continue
            bins.append(
                BinSpec(
                    nhit_bin=row["nhit_bin"],
                    predE_bin=row["predE_bin"],
                    count=int(row["count"]),
                    formal_nhit_bin=row["formal_nhit_bin"] == "True",
                    statistics_level=row["statistics_level"],
                )
            )
    return bins


def accumulate_histogram(
    bin_dir: Path,
    tree_name: str,
    branch: str,
    hist_edges: np.ndarray,
    target_events_per_bin: Optional[int],
) -> tuple[np.ndarray, int, int]:
    counts = np.zeros(len(hist_edges) - 1, dtype=np.float64)
    files = sorted(bin_dir.glob("*.root"))
    total_events = 0
    files_used = 0

    for file_path in files:
        with uproot.open(file_path) as root_file:
            tree = root_file[tree_name]
            branch_obj = tree[branch]
            if branch_obj.typename == "double":
                dtype = ">f8"
            elif branch_obj.typename == "float":
                dtype = ">f4"
            else:
                raise TypeError(
                    f"Unsupported branch typename {branch_obj.typename!r} for raw basket decode "
                    f"in file {file_path}"
                )

            basket_arrays = []
            for basket_idx in range(branch_obj.num_baskets):
                basket = branch_obj.basket(basket_idx)
                arr = np.frombuffer(basket.data, dtype=dtype)
                if arr.size > 0:
                    basket_arrays.append(arr)
            if not basket_arrays:
                continue
            values = np.concatenate(basket_arrays, dtype=np.float64)

        if values.size == 0:
            continue
        valid = values[np.isfinite(values) & (values > 0)]
        if valid.size == 0:
            continue
        loge = np.log10(valid)
        hist, _ = np.histogram(loge, bins=hist_edges)
        counts += hist
        total_events += int(valid.size)
        files_used += 1

        if target_events_per_bin is not None and target_events_per_bin > 0 and total_events >= target_events_per_bin:
            break

    return counts, total_events, files_used


def make_plot(
    counts: np.ndarray,
    hist_edges: np.ndarray,
    *,
    out_path: Path,
    title: str,
    subtitle: str,
) -> None:
    widths = np.diff(hist_edges)
    total = counts.sum()
    density = counts / (total * widths)
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    ax.step(centers, density, where="mid", linewidth=1.8, color="#1f4e79")
    ax.fill_between(centers, density, step="mid", alpha=0.18, color="#4f81bd")
    ax.set_xlabel(r"True $\log_{10}(E / \mathrm{GeV})$")
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.text(
        0.02,
        0.97,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    binned_root = Path(args.binned_root).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_bins = load_bins(summary_csv, args.min_statistics_level, args.min_count)
    if args.max_bins is not None:
        selected_bins = selected_bins[: args.max_bins]
    hist_edges = np.linspace(args.x_min, args.x_max, args.num_bins + 1)

    manifest = {
        "binned_root": str(binned_root),
        "summary_csv": str(summary_csv),
        "branch": args.branch,
        "tree_name": args.tree_name,
        "min_statistics_level": args.min_statistics_level,
        "min_count": args.min_count,
        "x_range_log10_GeV": [args.x_min, args.x_max],
        "num_bins": args.num_bins,
        "target_events_per_bin": args.target_events_per_bin,
        "generated_plots": [],
        "skipped_bins": [],
    }

    for bin_spec in selected_bins:
        bin_dir = binned_root / nhit_dir_name(bin_spec.nhit_bin) / pred_dir_name(bin_spec.predE_bin)
        print(f"[start] {bin_spec.nhit_bin} x {bin_spec.predE_bin} @ {bin_dir}", flush=True)
        if not bin_dir.exists():
            manifest["skipped_bins"].append(
                {
                    "nhit_bin": bin_spec.nhit_bin,
                    "predE_bin": bin_spec.predE_bin,
                    "reason": f"missing directory: {bin_dir}",
                }
            )
            continue

        counts, total_events, files_used = accumulate_histogram(
            bin_dir=bin_dir,
            tree_name=args.tree_name,
            branch=args.branch,
            hist_edges=hist_edges,
            target_events_per_bin=args.target_events_per_bin,
        )
        if counts.sum() <= 0:
            manifest["skipped_bins"].append(
                {
                    "nhit_bin": bin_spec.nhit_bin,
                    "predE_bin": bin_spec.predE_bin,
                    "reason": "no valid positive true-energy entries found",
                }
            )
            continue

        out_name = (
            f"true_energy_density__{slugify_bin_label(bin_spec.nhit_bin)}__"
            f"{slugify_bin_label(bin_spec.predE_bin)}.png"
        )
        out_path = output_dir / out_name
        make_plot(
            counts=counts,
            hist_edges=hist_edges,
            out_path=out_path,
            title=f"True-energy distribution: nhit {bin_spec.nhit_bin}, predE {bin_spec.predE_bin}",
            subtitle=(
                f"selected events = {total_events:,} | files contributing = {files_used:,} | "
                f"summary label = {bin_spec.statistics_level}"
            ),
        )
        manifest["generated_plots"].append(
            {
                "nhit_bin": bin_spec.nhit_bin,
                "predE_bin": bin_spec.predE_bin,
                "count_from_summary": bin_spec.count,
                "events_read_for_branch": total_events,
                "files_used": files_used,
                "formal_nhit_bin": bin_spec.formal_nhit_bin,
                "statistics_level": bin_spec.statistics_level,
                "plot_path": str(out_path),
            }
        )
        print(
            f"[done] {bin_spec.nhit_bin} x {bin_spec.predE_bin} -> {out_path.name} ({total_events} events)",
            flush=True,
        )

    manifest_path = output_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
