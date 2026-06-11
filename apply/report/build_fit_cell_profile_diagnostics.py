#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
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


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Cell:
    index: int
    cell_id: int
    nhit_bin: str
    predE_bin: str
    source_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized fit-cell Stage D excess profile diagnostics.")
    parser.add_argument("--stage-d-npz", type=str, default="apply/output/stage_d_v2_raw65/runs/v2_stage_d_slurm_42014/background_v2_raw65.npz")
    parser.add_argument("--stage-d-metadata", type=str, default="apply/output/stage_d_v2_raw65/runs/v2_stage_d_slurm_42014/background_v2_raw65_metadata.json")
    parser.add_argument("--selector-csv", type=str, default="apply/config/cell_selector_v2_baseline24.csv")
    parser.add_argument("--output-dir", type=str, default="apply/report/assets/crab-v2-baseline24-fit-cell-profiles")
    parser.add_argument("--output-prefix", type=str, default="crab_v2_baseline24_fit")
    parser.add_argument("--title-prefix", type=str, default="v2_baseline24 fit-cell")
    parser.add_argument("--profile-half-width-deg", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=-0.35)
    parser.add_argument("--y-max", type=float, default=1.15)
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
    included: List[int] = []
    for row in rows:
        include = str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}
        if include:
            included.append(int(row["cell_id"]))
    if not included:
        raise ValueError(f"No included cells found in selector: {path}")
    return included


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


def prepare_grid(cells: Sequence[Cell]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Cell]]:
    nhit_bins = sorted({cell.nhit_bin for cell in cells}, key=interval_key)
    pred_bins = sorted({cell.predE_bin for cell in cells}, key=interval_key)
    by_key = {(cell.nhit_bin, cell.predE_bin): cell for cell in cells}
    return nhit_bins, pred_bins, by_key


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


def normalize_profiles(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normalized = np.full(profile.shape, np.nan, dtype=np.float64)
    peaks = np.full(profile.shape[0], np.nan, dtype=np.float64)
    for idx, values in enumerate(profile):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        peak = float(np.nanmax(values))
        peaks[idx] = peak
        if peak > 0.0 and np.isfinite(peak):
            normalized[idx] = values / peak
    return normalized, peaks


def compute_profiles(
    excess_map: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    profile_half_width_deg: float,
) -> Dict[str, np.ndarray]:
    y_band = np.abs(y_centers) < float(profile_half_width_deg)
    x_band = np.abs(x_centers) < float(profile_half_width_deg)
    if not np.any(y_band):
        raise ValueError("No y bins selected for RA-offset profile. Increase --profile-half-width-deg.")
    if not np.any(x_band):
        raise ValueError("No x bins selected for Dec-offset profile. Increase --profile-half-width-deg.")

    clean_excess = np.where(np.isfinite(excess_map), excess_map, 0.0)
    ra_profile = np.sum(clean_excess[:, y_band, :], axis=1, dtype=np.float64)
    dec_profile = np.sum(clean_excess[:, :, x_band], axis=2, dtype=np.float64)
    ra_normalized, ra_peaks = normalize_profiles(ra_profile)
    dec_normalized, dec_peaks = normalize_profiles(dec_profile)
    return {
        "ra_profile": ra_profile,
        "dec_profile": dec_profile,
        "ra_normalized": ra_normalized,
        "dec_normalized": dec_normalized,
        "ra_peak": ra_peaks,
        "dec_peak": dec_peaks,
        "ra_band_bin_count": np.array([int(np.sum(y_band))], dtype=np.int32),
        "dec_band_bin_count": np.array([int(np.sum(x_band))], dtype=np.int32),
    }


def plot_profile_grid(
    *,
    normalized_profile: np.ndarray,
    centers: np.ndarray,
    cells: Sequence[Cell],
    per_cell_roi_events: Dict[int, int],
    output_path: Path,
    title: str,
    x_label: str,
    y_min: float,
    y_max: float,
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

    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue

            values = normalized_profile[cell.source_index]
            ax.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.85)
            ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.65, alpha=0.75)
            ax.axvline(0.0, color="#222222", linewidth=0.75, alpha=0.7)
            ax.step(centers, values, where="mid", color="#1f4e79", linewidth=1.05)
            ax.grid(alpha=0.22, linewidth=0.35)
            ax.set_ylim(float(y_min), float(y_max))
            roi_events = per_cell_roi_events.get(cell.cell_id, 0)
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}\nN={roi_events:,}", fontsize=6.9)
            ax.tick_params(labelsize=6.2, length=2)
            if j == first_visible_col_by_row[i]:
                ax.set_ylabel(f"{nhit_bin}\nexcess / peak", fontsize=7.0)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel(x_label, fontsize=7.0)

    fig.suptitle(title, fontsize=12, y=0.996)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.982])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stage_d_npz = resolve(args.stage_d_npz)
    stage_d_metadata = resolve(args.stage_d_metadata)
    selector_csv = resolve(args.selector_csv)
    output_dir = resolve(args.output_dir)
    output_prefix = str(args.output_prefix).strip()
    if not output_prefix:
        raise ValueError("--output-prefix cannot be empty")

    data = np.load(stage_d_npz)
    required = {"cell_id", "nhit_bin", "predE_bin", "x_centers_deg", "y_centers_deg", "excess_map"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{stage_d_npz} is missing arrays: {sorted(missing)}")

    included_ids = selector_included_ids(selector_csv)
    cells = load_cells(data, included_ids)
    per_cell_roi_events = metadata_cell_events(stage_d_metadata)
    profiles = compute_profiles(
        data["excess_map"],
        data["x_centers_deg"].astype(np.float64),
        data["y_centers_deg"].astype(np.float64),
        profile_half_width_deg=float(args.profile_half_width_deg),
    )

    ra_png = output_dir / f"{output_prefix}_ra_normalized_excess_profiles.png"
    dec_png = output_dir / f"{output_prefix}_dec_normalized_excess_profiles.png"
    meta_path = output_dir / f"{output_prefix}_normalized_excess_profiles_meta.json"
    plot_profile_grid(
        normalized_profile=profiles["ra_normalized"],
        centers=data["x_centers_deg"].astype(np.float64),
        cells=cells,
        per_cell_roi_events=per_cell_roi_events,
        output_path=ra_png,
        title=f"{args.title_prefix} normalized RA-offset excess profiles",
        x_label="RA offset cos(dec) (deg)",
        y_min=float(args.y_min),
        y_max=float(args.y_max),
    )
    plot_profile_grid(
        normalized_profile=profiles["dec_normalized"],
        centers=data["y_centers_deg"].astype(np.float64),
        cells=cells,
        per_cell_roi_events=per_cell_roi_events,
        output_path=dec_png,
        title=f"{args.title_prefix} normalized Dec-offset excess profiles",
        x_label="Dec offset (deg)",
        y_min=float(args.y_min),
        y_max=float(args.y_max),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "description": "Normalized Stage D excess profiles for cells included in the fit selector.",
        "selector_csv": str(selector_csv),
        "stage_d_npz": str(stage_d_npz),
        "stage_d_metadata": str(stage_d_metadata),
        "included_cell_ids": [cell.cell_id for cell in cells],
        "included_cell_count": len(cells),
        "profile_half_width_deg": float(args.profile_half_width_deg),
        "normalization": "Each 1D excess profile is divided by its own positive peak.",
        "outputs": {
            "ra_normalized_excess_profiles_png": str(ra_png),
            "dec_normalized_excess_profiles_png": str(dec_png),
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "stage_d_source_index": cell.source_index,
                "grid_events": per_cell_roi_events.get(cell.cell_id),
                "ra_profile_peak": float(profiles["ra_peak"][cell.source_index]),
                "dec_profile_peak": float(profiles["dec_peak"][cell.source_index]),
            }
            for cell in cells
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {ra_png}")
    print(f"Wrote {dec_png}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
