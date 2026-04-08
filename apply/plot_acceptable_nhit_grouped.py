#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one combined normalized figure: predE bins on x-axis, nhit bins as lettered sub-bars."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="apply/summary_selectedcuts/bin_counts.csv",
        help="Path to bin_counts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="apply/plot",
        help="Directory for output figure and manifest.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="acceptable_nhit_grouped_by_predE_normalized.png",
        help="Output PNG name.",
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


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["statistics_level"] != "acceptable":
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

    pred_bins = sorted({r["predE_bin"] for r in rows}, key=pred_bin_key)
    nhit_bins = sorted({r["nhit_bin"] for r in rows}, key=nhit_bin_key)

    letter_map = {nhit: chr(ord("a") + i) for i, nhit in enumerate(nhit_bins)}
    matrix = np.zeros((len(nhit_bins), len(pred_bins)), dtype=np.float64)

    pred_idx = {p: i for i, p in enumerate(pred_bins)}
    nhit_idx = {n: i for i, n in enumerate(nhit_bins)}
    for r in rows:
        matrix[nhit_idx[r["nhit_bin"]], pred_idx[r["predE_bin"]]] = float(r["count"])

    col_sums = matrix.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(matrix, col_sums, where=col_sums > 0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.arange(len(pred_bins), dtype=np.float64)
    n = len(nhit_bins)
    group_width = 0.86
    bar_w = group_width / max(n, 1)
    offsets = (np.arange(n) - (n - 1) / 2.0) * bar_w
    cmap = plt.get_cmap("tab10", n)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    for i, nhit in enumerate(nhit_bins):
        ax.bar(
            x + offsets[i],
            normalized[i],
            width=bar_w * 0.96,
            color=cmap(i),
            alpha=0.92,
            label=f"{letter_map[nhit]}: {nhit}",
            edgecolor="white",
            linewidth=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pred_bins, rotation=0)
    ax.set_xlabel("Predicted Energy Bin")
    ax.set_ylabel("Normalized Fraction in Each Energy Bin")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Acceptable Bins Only: Normalized Distribution by Energy Bin with nhit Sub-bins (a,b,c...)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()

    out_png = output_dir / args.output_name
    fig.savefig(out_png)
    plt.close(fig)

    letter_table_path = output_dir / "acceptable_nhit_letter_map.json"
    with letter_table_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_csv": str(summary_csv),
                "pred_bins": pred_bins,
                "nhit_bins_in_letter_order": [{"letter": letter_map[n], "nhit_bin": n} for n in nhit_bins],
                "num_acceptable_rows": len(rows),
                "total_acceptable_events": int(matrix.sum()),
                "note": "Heights are normalized within each predE_bin column.",
            },
            f,
            indent=2,
        )

    print(f"Wrote figure: {out_png}")
    print(f"Wrote letter map: {letter_table_path}")


if __name__ == "__main__":
    main()
