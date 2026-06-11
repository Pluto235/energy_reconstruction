#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Cell:
    index: int
    cell_id: int
    nhit_bin: str
    predE_bin: str
    source_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage D maps for cells included in a fit selector.")
    parser.add_argument("--stage-d-npz", type=str, default="apply/output/stage_d_v2_raw65/runs/v2_stage_d_slurm_42014/background_v2_raw65.npz")
    parser.add_argument("--stage-d-metadata", type=str, default="apply/output/stage_d_v2_raw65/runs/v2_stage_d_slurm_42014/background_v2_raw65_metadata.json")
    parser.add_argument("--selector-csv", type=str, default="apply/config/cell_selector_v2_baseline24.csv")
    parser.add_argument("--map-key", choices=["counts_map", "excess_map"], default="counts_map")
    parser.add_argument("--output-png", type=str, default="apply/report/assets/crab-v2-baseline24-fit-cell-skymaps/crab_v2_baseline24_fit_counts_grid.png")
    parser.add_argument("--output-metadata", type=str, default="apply/report/assets/crab-v2-baseline24-fit-cell-skymaps/crab_v2_baseline24_fit_counts_grid_meta.json")
    parser.add_argument("--title", type=str, default="v2_baseline24 fit-cell Stage D counts maps")
    parser.add_argument("--vmax-percentile", type=float, default=99.3)
    parser.add_argument("--roi-circle-deg", type=float, default=0.0, help="Overlay a circle with this radius in degrees. Disabled when <= 0.")
    parser.add_argument("--crop-radius-deg", type=float, default=0.0, help="Crop the map to +/- this offset in both axes. Disabled when <= 0.")
    parser.add_argument("--circular-roi-mask", action="store_true", help="Mask pixels outside crop-radius-deg as NaN after cropping.")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


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


def selector_included_ids(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids: List[int] = []
    for row in rows:
        include = str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}
        if include:
            ids.append(int(row["cell_id"]))
    if not ids:
        raise ValueError(f"No included cells found in selector: {path}")
    return ids


def load_cells(data: np.lib.npyio.NpzFile, included_ids: Sequence[int]) -> List[Cell]:
    source_by_id = {int(cell_id): idx for idx, cell_id in enumerate(data["cell_id"])}
    cells: List[Cell] = []
    for out_index, cell_id in enumerate(included_ids):
        if int(cell_id) not in source_by_id:
            raise ValueError(f"Cell {cell_id} is included by selector but missing from Stage D NPZ")
        source_index = source_by_id[int(cell_id)]
        cells.append(
            Cell(
                index=out_index,
                cell_id=int(cell_id),
                nhit_bin=str(data["nhit_bin"][source_index]),
                predE_bin=str(data["predE_bin"][source_index]),
                source_index=source_index,
            )
        )
    return sorted(cells, key=lambda c: (interval_key(c.nhit_bin), interval_key(c.predE_bin), c.cell_id))


def robust_vmax(values: np.ndarray, percentile: float) -> float:
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        return 1.0
    vmax = float(np.percentile(finite_positive, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(finite_positive.max())
    return max(vmax, 1.0e-6)


def prepare_grid(cells: Sequence[Cell]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Cell]]:
    nhit_bins = sorted({cell.nhit_bin for cell in cells}, key=interval_key)
    pred_bins = sorted({cell.predE_bin for cell in cells}, key=interval_key)
    by_key = {(cell.nhit_bin, cell.predE_bin): cell for cell in cells}
    return nhit_bins, pred_bins, by_key


def robust_abs_vmax(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(finite)))
    return max(vmax, 1.0e-6)


def plot_grid(
    *,
    maps: np.ndarray,
    cells: Sequence[Cell],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    output_png: Path,
    title: str,
    vmax_percentile: float,
    per_cell_roi_events: Dict[int, int],
    map_key: str,
    roi_circle_deg: float,
) -> None:
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    first_visible_col_by_row = {
        i: min(j for j, pred_bin in enumerate(pred_bins) if (nhit_bin, pred_bin) in by_key)
        for i, nhit_bin in enumerate(nhit_bins)
    }
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.05 * len(pred_bins), 1.95 * len(nhit_bins)),
        dpi=170,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    if map_key == "excess_map":
        cmap = plt.get_cmap("RdBu_r").copy()
    else:
        cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")

    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue

            image = maps[cell.source_index].astype(np.float64)
            if map_key == "excess_map":
                vmax = robust_abs_vmax(image, vmax_percentile)
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
            else:
                norm = Normalize(vmin=0.0, vmax=robust_vmax(image, vmax_percentile))
            ax.imshow(
                image,
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            if roi_circle_deg > 0.0:
                ax.add_patch(plt.Circle((0.0, 0.0), roi_circle_deg, fill=False, color="black", linewidth=1.05, alpha=0.9))
                ax.add_patch(plt.Circle((0.0, 0.0), roi_circle_deg, fill=False, color="white", linewidth=0.52, alpha=0.95))
            ax.plot([0.0], [0.0], marker="+", markersize=7, markeredgewidth=1.2, color="black")
            roi_events = per_cell_roi_events.get(cell.cell_id)
            if roi_events is None:
                roi_events = int(np.nansum(maps[cell.source_index]))
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}\nN={roi_events:,}", fontsize=6.9)
            ax.tick_params(labelsize=6.2, length=2)
            ax.grid(color="white", alpha=0.22, linewidth=0.32)
            if j == first_visible_col_by_row[i]:
                ax.set_ylabel(f"{nhit_bin}\nDec offset (deg)", fontsize=7.0)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("RA offset cos(dec) (deg)", fontsize=7.0)

    fig.suptitle(f"{title} (per-panel color scale)", fontsize=12, y=0.996)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.982])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def crop_maps(
    *,
    maps: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    crop_radius_deg: float,
    circular_roi_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if crop_radius_deg <= 0.0:
        return (
            maps,
            x_edges,
            y_edges,
            {
                "enabled": False,
                "crop_radius_deg": None,
                "circular_roi_mask": False,
            },
        )

    x_idx = np.flatnonzero(np.abs(x_centers) <= crop_radius_deg)
    y_idx = np.flatnonzero(np.abs(y_centers) <= crop_radius_deg)
    if x_idx.size == 0 or y_idx.size == 0:
        raise ValueError(f"No bins selected by --crop-radius-deg {crop_radius_deg}")

    cropped = maps[:, y_idx.min() : y_idx.max() + 1, x_idx.min() : x_idx.max() + 1].astype(np.float64, copy=True)
    cropped_x_edges = x_edges[x_idx.min() : x_idx.max() + 2]
    cropped_y_edges = y_edges[y_idx.min() : y_idx.max() + 2]
    cropped_x_centers = x_centers[x_idx.min() : x_idx.max() + 1]
    cropped_y_centers = y_centers[y_idx.min() : y_idx.max() + 1]

    masked_pixel_count = 0
    if circular_roi_mask:
        xx, yy = np.meshgrid(cropped_x_centers, cropped_y_centers)
        outside_roi = (xx * xx + yy * yy) > crop_radius_deg * crop_radius_deg
        masked_pixel_count = int(np.count_nonzero(outside_roi))
        cropped[:, outside_roi] = np.nan

    return (
        cropped,
        cropped_x_edges,
        cropped_y_edges,
        {
            "enabled": True,
            "crop_radius_deg": float(crop_radius_deg),
            "circular_roi_mask": bool(circular_roi_mask),
            "x_range_deg": [float(cropped_x_edges[0]), float(cropped_x_edges[-1])],
            "y_range_deg": [float(cropped_y_edges[0]), float(cropped_y_edges[-1])],
            "masked_pixels_per_cell": masked_pixel_count,
        },
    )


def metadata_cell_events(path: Path) -> Dict[int, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    events: Dict[int, int] = {}
    for cell in meta.get("cells", []):
        if not isinstance(cell, dict):
            continue
        events[int(cell["cell_id"])] = int(cell.get("grid_events", cell.get("selected_events", 0)))
    return events


def main() -> None:
    args = parse_args()
    stage_d_npz = resolve(args.stage_d_npz)
    stage_d_metadata = resolve(args.stage_d_metadata)
    selector_csv = resolve(args.selector_csv)
    output_png = resolve(args.output_png)
    output_metadata = resolve(args.output_metadata)

    data = np.load(stage_d_npz)
    required = {"cell_id", "nhit_bin", "predE_bin", "x_edges_deg", "y_edges_deg", "x_centers_deg", "y_centers_deg", args.map_key}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{stage_d_npz} is missing arrays: {sorted(missing)}")

    included_ids = selector_included_ids(selector_csv)
    cells = load_cells(data, included_ids)
    per_cell_roi_events = metadata_cell_events(stage_d_metadata)
    maps, x_edges, y_edges, crop_metadata = crop_maps(
        maps=data[args.map_key],
        x_edges=data["x_edges_deg"].astype(np.float64),
        y_edges=data["y_edges_deg"].astype(np.float64),
        x_centers=data["x_centers_deg"].astype(np.float64),
        y_centers=data["y_centers_deg"].astype(np.float64),
        crop_radius_deg=float(args.crop_radius_deg),
        circular_roi_mask=bool(args.circular_roi_mask),
    )
    plot_grid(
        maps=maps,
        cells=cells,
        x_edges=x_edges,
        y_edges=y_edges,
        output_png=output_png,
        title=str(args.title),
        vmax_percentile=float(args.vmax_percentile),
        per_cell_roi_events=per_cell_roi_events,
        map_key=str(args.map_key),
        roi_circle_deg=float(args.roi_circle_deg),
    )

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "description": f"Stage D {args.map_key} maps for cells included in the fit selector.",
        "map_key": str(args.map_key),
        "selector_csv": str(selector_csv),
        "stage_d_npz": str(stage_d_npz),
        "stage_d_metadata": str(stage_d_metadata),
        "output_png": str(output_png),
        "included_cell_ids": [cell.cell_id for cell in cells],
        "included_cell_count": len(cells),
        "crop": crop_metadata,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "stage_d_source_index": cell.source_index,
                "grid_events": per_cell_roi_events.get(cell.cell_id),
            }
            for cell in cells
        ],
        "plotting": {
            "counts_scale": "per-panel",
            "vmax_percentile": float(args.vmax_percentile),
            "normalization": "TwoSlopeNorm centered at 0" if args.map_key == "excess_map" else "Normalize from 0",
            "marker": "Crab nominal center at (0,0)",
            "roi_circle_deg": float(args.roi_circle_deg) if args.roi_circle_deg > 0.0 else None,
        },
    }
    with output_metadata.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_metadata}")


if __name__ == "__main__":
    main()
