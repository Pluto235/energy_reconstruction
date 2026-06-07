#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uproot
from uproot.source.file import MemmapSource

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELL_SELECTION = REPO_ROOT / "apply" / "config" / "cell_selection_v1.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apply" / "plot" / "crab_cell_skymaps"


@dataclass(frozen=True)
class Cell:
    index: int
    cell_id: int
    nhit_bin: str
    predE_bin: str
    nhit_low: Optional[float]
    nhit_high: Optional[float]
    pred_low: Optional[float]
    pred_high: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Crab-centered sky maps for the selected v1 (Nhit, predicted logE) cells "
            "from observation eval ROOT files and recovered-time friend trees."
        )
    )
    parser.add_argument("--obs-root", type=str, default="/mnt/mydisk/WCDA_observation_eval")
    parser.add_argument("--time-root", type=str, default="/mnt/mydisk/WCDA_observation_eval/recovered_time")
    parser.add_argument("--cell-selection-csv", type=str, default=str(DEFAULT_CELL_SELECTION))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--time-tree-name", type=str, default="t_recovered_time")
    parser.add_argument("--file-glob", type=str, default="Esg*.root")
    parser.add_argument("--day-prefix", type=str, default=None, help="Only process MMDD directories starting with this prefix.")
    parser.add_argument("--max-files", type=int, default=None, help="Process only the first N sorted observation ROOT files.")
    parser.add_argument("--entries-per-chunk", type=int, default=200000)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--allow-missing-time", action="store_true", help="Skip files without matching .time.root instead of failing.")
    parser.add_argument("--allow-entry-mismatch", action="store_true", help="Use the common entry prefix instead of failing.")

    parser.add_argument("--source-name", type=str, default="Crab")
    parser.add_argument("--source-ra-deg", type=float, default=83.63)
    parser.add_argument("--source-dec-deg", type=float, default=22.01)
    parser.add_argument("--half-width-deg", type=float, default=10.0)
    parser.add_argument("--pixel-size-deg", type=float, default=0.1)
    parser.add_argument("--smooth-sigma-deg", type=float, default=0.3)
    parser.add_argument("--source-exclusion-radius-deg", type=float, default=2.0)
    parser.add_argument("--sideband-stat", choices=["mean", "median"], default="mean")

    parser.add_argument("--cut-pinc-max", type=float, default=1.1)
    parser.add_argument("--cut-fitstat-equals", type=int, default=0)
    parser.add_argument("--cut-theta-max-deg", type=float, default=50.0)
    parser.add_argument("--match-status-equals", type=int, default=0)

    parser.add_argument("--counts-scale", choices=["per-panel", "global"], default="per-panel")
    parser.add_argument("--counts-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--significance-max", type=float, default=5.0)
    parser.add_argument("--quicklook-only", action="store_true", help="Only rebuild the approximate significance quicklook from an existing maps NPZ.")
    parser.add_argument(
        "--quicklook-input-npz",
        type=str,
        default=None,
        help="Input maps NPZ for --quicklook-only. Defaults to <output-dir>/crab_v1_maps.npz.",
    )
    parser.add_argument(
        "--quicklook-bg-max-abs-x-deg",
        type=float,
        default=5.0,
        help="Use only sideband bins with |RA offset * cos(dec)| below this value for the quicklook background.",
    )
    parser.add_argument("--profiles-only", action="store_true", help="Only build 1D profile diagnostics from an existing maps NPZ.")
    parser.add_argument(
        "--profile-input-npz",
        type=str,
        default=None,
        help="Input maps NPZ for --profiles-only. Defaults to <output-dir>/crab_v1_maps.npz.",
    )
    parser.add_argument("--profile-half-width-deg", type=float, default=1.0)
    parser.add_argument("--profile-sideband-min-deg", type=float, default=5.0)
    parser.add_argument("--profile-baseline-stat", choices=["median", "mean"], default="median")
    return parser.parse_args()


def open_root(path: Path):
    return uproot.open(path, handler=MemmapSource)


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


def load_cells(selection_csv: Path) -> List[Cell]:
    cells: List[Cell] = []
    with selection_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"cell_id", "nhit_bin", "predE_bin"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{selection_csv} is missing required columns: {sorted(missing)}")

        for row in reader:
            nhit_low, nhit_high = parse_interval(row["nhit_bin"])
            pred_low, pred_high = parse_interval(row["predE_bin"])
            cells.append(
                Cell(
                    index=len(cells),
                    cell_id=int(row["cell_id"]),
                    nhit_bin=row["nhit_bin"],
                    predE_bin=row["predE_bin"],
                    nhit_low=nhit_low,
                    nhit_high=nhit_high,
                    pred_low=pred_low,
                    pred_high=pred_high,
                )
            )

    if not cells:
        raise ValueError(f"No cells loaded from {selection_csv}")

    return sorted(cells, key=lambda c: (interval_key(c.nhit_bin), interval_key(c.predE_bin), c.cell_id))


def discover_observation_files(obs_root: Path, file_glob: str, day_prefix: Optional[str], max_files: Optional[int]) -> List[Path]:
    if obs_root.is_file():
        files = [obs_root]
    elif obs_root.is_dir():
        files = []
        for path in sorted(obs_root.glob(f"[0-9][0-9][0-9][0-9]/{file_glob}")):
            if path.name.endswith(".time.root"):
                continue
            if day_prefix is not None and not path.parent.name.startswith(day_prefix):
                continue
            files.append(path)
    else:
        raise FileNotFoundError(f"Observation root does not exist: {obs_root}")

    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No observation ROOT files found under {obs_root}")
    return files


def observation_relative_path(obs_file: Path, obs_root: Path) -> Path:
    if obs_root.is_file():
        return Path(obs_file.parent.name) / obs_file.name
    try:
        return obs_file.relative_to(obs_root)
    except ValueError as exc:
        raise ValueError(f"{obs_file} is not under {obs_root}") from exc


def time_path_for(obs_file: Path, obs_root: Path, time_root: Path) -> Path:
    rel = observation_relative_path(obs_file, obs_root)
    return time_root / rel.parent / f"{obs_file.stem}.time.root"


def build_edges(half_width_deg: float, pixel_size_deg: float) -> np.ndarray:
    if half_width_deg <= 0:
        raise ValueError("--half-width-deg must be positive")
    if pixel_size_deg <= 0:
        raise ValueError("--pixel-size-deg must be positive")
    n_bins = int(round((2.0 * half_width_deg) / pixel_size_deg))
    if n_bins < 1:
        raise ValueError("Sky map would have fewer than one bin")
    return np.linspace(-half_width_deg, half_width_deg, n_bins + 1, dtype=np.float64)


def build_radec_edges(source_ra_deg: float, source_dec_deg: float, half_width_deg: float, pixel_size_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    cos_dec = math.cos(math.radians(source_dec_deg))
    if cos_dec <= 0:
        raise ValueError("RA/Dec plotting is not supported at the celestial poles.")
    n_bins = int(round((2.0 * half_width_deg) / pixel_size_deg))
    ra_half_width = half_width_deg / cos_dec
    ra_edges = np.linspace(source_ra_deg - ra_half_width, source_ra_deg + ra_half_width, n_bins + 1, dtype=np.float64)
    dec_edges = np.linspace(source_dec_deg - half_width_deg, source_dec_deg + half_width_deg, n_bins + 1, dtype=np.float64)
    return ra_edges, dec_edges


def in_interval(values: np.ndarray, low: Optional[float], high: Optional[float]) -> np.ndarray:
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values >= low
    if high is not None:
        mask &= values < high
    return mask


def wrapped_delta_ra_deg(ra_deg: np.ndarray, source_ra_deg: float) -> np.ndarray:
    return ((ra_deg - source_ra_deg + 180.0) % 360.0) - 180.0


def gaussian_kernel_1d(sigma_px: float) -> np.ndarray:
    radius = max(1, int(math.ceil(4.0 * sigma_px)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_px) ** 2)
    kernel /= kernel.sum()
    return kernel


def convolve_axis_edge(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def smooth_maps(maps: np.ndarray, sigma_px: float) -> np.ndarray:
    maps = maps.astype(np.float64, copy=False)
    if sigma_px <= 0:
        return maps.astype(np.float32, copy=True)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(maps, sigma=(0, sigma_px, sigma_px), mode="nearest").astype(np.float32)
    except ImportError:
        kernel = gaussian_kernel_1d(sigma_px)
        smoothed = convolve_axis_edge(maps, kernel, axis=2)
        smoothed = convolve_axis_edge(smoothed, kernel, axis=1)
        return smoothed.astype(np.float32)


def sideband_background(
    maps: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    exclusion_radius_deg: float,
    stat: str,
    max_abs_x_deg: float,
) -> np.ndarray:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    sideband_mask = ((xx * xx + yy * yy) >= exclusion_radius_deg * exclusion_radius_deg) & (
        np.abs(xx) < float(max_abs_x_deg)
    )
    background = np.zeros_like(maps, dtype=np.float32)

    for i in range(maps.shape[0]):
        for y_idx in range(maps.shape[1]):
            values = maps[i, y_idx, sideband_mask[y_idx]]
            if values.size == 0:
                level = 0.0
            elif stat == "median":
                level = float(np.median(values))
            else:
                level = float(np.mean(values))
            background[i, y_idx, :] = level
    return background


def robust_vmax(values: np.ndarray, percentile: float) -> float:
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        return 1.0
    vmax = float(np.percentile(finite_positive, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(finite_positive.max())
    return max(vmax, 1.0e-6)


def prepare_grid(cells: Sequence[Cell]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Cell]]:
    nhit_bins = sorted({c.nhit_bin for c in cells}, key=interval_key)
    pred_bins = sorted({c.predE_bin for c in cells}, key=interval_key)
    by_key = {(c.nhit_bin, c.predE_bin): c for c in cells}
    return nhit_bins, pred_bins, by_key


def plot_grid(
    maps: np.ndarray,
    cells: Sequence[Cell],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    per_cell_roi_events: np.ndarray,
    output_path: Path,
    *,
    title: str,
    cmap_name: str,
    scale: str,
    vmax_percentile: float,
    significance_max: float,
    colorbar_label: str,
    marker_x: float = 0.0,
    marker_y: float = 0.0,
    x_label: str = "RA offset cos(dec) (deg)",
    y_label: str = "Dec offset (deg)",
) -> None:
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    n_rows = len(nhit_bins)
    n_cols = len(pred_bins)
    first_visible_col_by_row = {
        i: min(j for j, pred_bin in enumerate(pred_bins) if (nhit_bin, pred_bin) in by_key)
        for i, nhit_bin in enumerate(nhit_bins)
    }
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.25 * n_cols, 2.05 * n_rows),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")

    global_norm = None
    if scale == "global":
        if "significance" in output_path.name:
            global_norm = TwoSlopeNorm(vcenter=0.0, vmin=-significance_max, vmax=significance_max)
        else:
            global_norm = Normalize(vmin=0.0, vmax=robust_vmax(maps, vmax_percentile))

    first_im = None
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue

            data = maps[cell.index]
            if "significance" in output_path.name:
                norm = global_norm or TwoSlopeNorm(vcenter=0.0, vmin=-significance_max, vmax=significance_max)
            elif scale == "global":
                norm = global_norm
            else:
                norm = Normalize(vmin=0.0, vmax=robust_vmax(data, vmax_percentile))

            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            if first_im is None:
                first_im = im

            ax.plot([marker_x], [marker_y], marker="+", markersize=7, markeredgewidth=1.2, color="black")
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}\nN={int(per_cell_roi_events[cell.index]):,}", fontsize=7.2)
            ax.tick_params(labelsize=6.5, length=2)
            ax.grid(color="white", alpha=0.25, linewidth=0.35)
            if j == first_visible_col_by_row[i]:
                ax.set_ylabel(f"{nhit_bin}\n{y_label}", fontsize=7.5)
            if i == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=7.5)

    subtitle = "per-panel color scale" if scale == "per-panel" and "significance" not in output_path.name else "shared color scale"
    fig.suptitle(f"{title} ({subtitle})", fontsize=12, y=0.996)
    fig.tight_layout(rect=[0.0, 0.0, 0.94 if scale == "global" or "significance" in output_path.name else 1.0, 0.982])

    if first_im is not None and (scale == "global" or "significance" in output_path.name):
        cbar = fig.colorbar(first_im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        cbar.set_label(colorbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.savefig(output_path)
    plt.close(fig)


def cells_from_npz(data: np.lib.npyio.NpzFile) -> List[Cell]:
    cells: List[Cell] = []
    for idx, (cell_id, nhit_bin, predE_bin) in enumerate(zip(data["cell_id"], data["nhit_bin"], data["predE_bin"])):
        nhit_bin_str = str(nhit_bin)
        predE_bin_str = str(predE_bin)
        nhit_low, nhit_high = parse_interval(nhit_bin_str)
        pred_low, pred_high = parse_interval(predE_bin_str)
        cells.append(
            Cell(
                index=idx,
                cell_id=int(cell_id),
                nhit_bin=nhit_bin_str,
                predE_bin=predE_bin_str,
                nhit_low=nhit_low,
                nhit_high=nhit_high,
                pred_low=pred_low,
                pred_high=pred_high,
            )
        )
    return cells


def profile_baseline(profile: np.ndarray, centers: np.ndarray, sideband_min_deg: float, stat: str) -> np.ndarray:
    sideband = np.abs(centers) > float(sideband_min_deg)
    if not np.any(sideband):
        raise ValueError("Profile sideband is empty. Reduce --profile-sideband-min-deg.")
    sideband_values = profile[:, sideband]
    if stat == "mean":
        return np.mean(sideband_values, axis=1)
    return np.median(sideband_values, axis=1)


def normalized_excess_profile(profile: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    excess = profile - baseline[:, None]
    peak = np.max(excess, axis=1)
    normalized = np.zeros_like(excess, dtype=np.float64)
    valid = peak > 0
    normalized[valid] = excess[valid] / peak[valid, None]
    return normalized


def compute_profiles(
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    profile_half_width_deg: float,
    sideband_min_deg: float,
    baseline_stat: str,
) -> Dict[str, np.ndarray]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_band = np.abs(y_centers) < float(profile_half_width_deg)
    x_band = np.abs(x_centers) < float(profile_half_width_deg)
    if not np.any(y_band):
        raise ValueError("No y bins selected for RA-offset profile. Increase --profile-half-width-deg.")
    if not np.any(x_band):
        raise ValueError("No x bins selected for Dec-offset profile. Increase --profile-half-width-deg.")

    ra_profile = counts[:, y_band, :].sum(axis=1).astype(np.float64)
    dec_profile = counts[:, :, x_band].sum(axis=2).astype(np.float64)
    ra_baseline = profile_baseline(ra_profile, x_centers, sideband_min_deg, baseline_stat)
    dec_baseline = profile_baseline(dec_profile, y_centers, sideband_min_deg, baseline_stat)
    return {
        "x_centers": x_centers,
        "y_centers": y_centers,
        "ra_profile": ra_profile,
        "dec_profile": dec_profile,
        "ra_baseline": ra_baseline,
        "dec_baseline": dec_baseline,
        "ra_normalized": normalized_excess_profile(ra_profile, ra_baseline),
        "dec_normalized": normalized_excess_profile(dec_profile, dec_baseline),
        "ra_band_bin_count": np.array([int(np.sum(y_band))], dtype=np.int32),
        "dec_band_bin_count": np.array([int(np.sum(x_band))], dtype=np.int32),
    }


def load_per_cell_roi_events(meta_path: Path, counts: np.ndarray) -> np.ndarray:
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        cells_meta = meta.get("cells", [])
        if cells_meta:
            roi_events = np.zeros(len(cells_meta), dtype=np.int64)
            for item in cells_meta:
                roi_events[int(item["index"])] = int(item["roi_events"])
            return roi_events
    return counts.sum(axis=(1, 2)).astype(np.int64)


def plot_profile_grid(
    profile: np.ndarray,
    centers: np.ndarray,
    baseline: np.ndarray,
    cells: Sequence[Cell],
    per_cell_roi_events: np.ndarray,
    output_path: Path,
    *,
    title: str,
    x_label: str,
    normalized: bool,
) -> None:
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    first_visible_col_by_row = {
        i: min(j for j, pred_bin in enumerate(pred_bins) if (nhit_bin, pred_bin) in by_key)
        for i, nhit_bin in enumerate(nhit_bins)
    }
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.25 * len(pred_bins), 2.0 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        squeeze=False,
    )

    legend_drawn = False
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue

            values = profile[cell.index]
            ax.axvline(0.0, color="#222222", linewidth=0.7, alpha=0.65)
            ax.grid(alpha=0.22, linewidth=0.35)
            if normalized:
                ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.7, alpha=0.8, label="baseline")
                ax.step(centers, values, where="mid", color="#1f4e79", linewidth=1.0, label="excess / peak")
                ymin = min(-0.25, float(np.nanmin(values)) * 1.05)
                ax.set_ylim(bottom=ymin, top=1.1)
            else:
                excess = values - baseline[cell.index]
                ax.step(centers, values, where="mid", color="#1f4e79", linewidth=0.9, label="raw")
                ax.axhline(baseline[cell.index], color="#777777", linestyle="--", linewidth=0.8, label="sideband median")
                ax.step(centers, excess, where="mid", color="#c9501a", linewidth=0.9, alpha=0.9, label="raw - baseline")

            ax.set_title(f"cell {cell.cell_id}: {pred_bin}\nN={int(per_cell_roi_events[cell.index]):,}", fontsize=7.1)
            ax.tick_params(labelsize=6.3, length=2)
            if j == first_visible_col_by_row[i]:
                ylabel = "excess / peak" if normalized else "counts / bin"
                ax.set_ylabel(f"{nhit_bin}\n{ylabel}", fontsize=7.3)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel(x_label, fontsize=7.3)
            if not legend_drawn:
                ax.legend(fontsize=5.8, frameon=False, loc="best")
                legend_drawn = True

    fig.suptitle(title, fontsize=12, y=0.996)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.982])
    fig.savefig(output_path)
    plt.close(fig)


def update_metadata_profiles(
    meta_path: Path,
    *,
    output_paths: Dict[str, Path],
    profile_half_width_deg: float,
    sideband_min_deg: float,
    baseline_stat: str,
    ra_band_bin_count: int,
    dec_band_bin_count: int,
) -> None:
    meta: Dict[str, object] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    meta["profiles"] = {
        "profile_half_width_deg": float(profile_half_width_deg),
        "sideband_min_abs_offset_deg": float(sideband_min_deg),
        "baseline_stat": baseline_stat,
        "ra_offset_profile": "counts[:, abs(y_center)<profile_half_width, :].sum(axis=1)",
        "dec_offset_profile": "counts[:, :, abs(x_center)<profile_half_width].sum(axis=2)",
        "ra_profile_center_band_y_bins": int(ra_band_bin_count),
        "dec_profile_center_band_x_bins": int(dec_band_bin_count),
        "normalization": "(profile - baseline) / max(profile - baseline) per cell",
        "not_for_physics": True,
    }
    outputs = meta.setdefault("outputs", {})
    for key, path in output_paths.items():
        outputs[key] = str(path)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def generate_profile_outputs(args: argparse.Namespace, *, input_npz: Path, output_dir: Path, meta_path: Path) -> Dict[str, Path]:
    if not input_npz.exists():
        raise FileNotFoundError(f"Profile input NPZ does not exist: {input_npz}")

    data = np.load(input_npz)
    required = {"counts", "x_edges", "y_edges", "cell_id", "nhit_bin", "predE_bin"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{input_npz} is missing required arrays: {sorted(missing)}")

    counts = data["counts"]
    x_edges = data["x_edges"]
    y_edges = data["y_edges"]
    cells = cells_from_npz(data)
    per_cell_roi_events = load_per_cell_roi_events(meta_path, counts)
    profiles = compute_profiles(
        counts,
        x_edges,
        y_edges,
        profile_half_width_deg=float(args.profile_half_width_deg),
        sideband_min_deg=float(args.profile_sideband_min_deg),
        baseline_stat=args.profile_baseline_stat,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ra_raw_png = output_dir / "crab_v1_ra_offset_profiles_grid.png"
    dec_raw_png = output_dir / "crab_v1_dec_offset_profiles_grid.png"
    ra_norm_png = output_dir / "crab_v1_ra_offset_profiles_normalized_grid.png"
    dec_norm_png = output_dir / "crab_v1_dec_offset_profiles_normalized_grid.png"

    half_width = float(args.profile_half_width_deg)
    sideband = float(args.profile_sideband_min_deg)
    baseline_label = args.profile_baseline_stat
    plot_profile_grid(
        profiles["ra_profile"],
        profiles["x_centers"],
        profiles["ra_baseline"],
        cells,
        per_cell_roi_events,
        ra_raw_png,
        title=f"{args.source_name} RA-offset raw profiles, |Dec offset| < {half_width:g} deg, baseline={baseline_label}(|x|>{sideband:g} deg)",
        x_label="RA offset * cos(Crab Dec) [deg]",
        normalized=False,
    )
    plot_profile_grid(
        profiles["dec_profile"],
        profiles["y_centers"],
        profiles["dec_baseline"],
        cells,
        per_cell_roi_events,
        dec_raw_png,
        title=f"{args.source_name} Dec-offset raw profiles, |RA offset| < {half_width:g} deg, baseline={baseline_label}(|y|>{sideband:g} deg)",
        x_label="Dec offset [deg]",
        normalized=False,
    )
    plot_profile_grid(
        profiles["ra_normalized"],
        profiles["x_centers"],
        profiles["ra_baseline"],
        cells,
        per_cell_roi_events,
        ra_norm_png,
        title=f"{args.source_name} RA-offset normalized excess-like profiles",
        x_label="RA offset * cos(Crab Dec) [deg]",
        normalized=True,
    )
    plot_profile_grid(
        profiles["dec_normalized"],
        profiles["y_centers"],
        profiles["dec_baseline"],
        cells,
        per_cell_roi_events,
        dec_norm_png,
        title=f"{args.source_name} Dec-offset normalized excess-like profiles",
        x_label="Dec offset [deg]",
        normalized=True,
    )

    output_paths = {
        "ra_offset_profiles_png": ra_raw_png,
        "dec_offset_profiles_png": dec_raw_png,
        "ra_offset_profiles_normalized_png": ra_norm_png,
        "dec_offset_profiles_normalized_png": dec_norm_png,
    }
    update_metadata_profiles(
        meta_path,
        output_paths=output_paths,
        profile_half_width_deg=float(args.profile_half_width_deg),
        sideband_min_deg=float(args.profile_sideband_min_deg),
        baseline_stat=args.profile_baseline_stat,
        ra_band_bin_count=int(profiles["ra_band_bin_count"][0]),
        dec_band_bin_count=int(profiles["dec_band_bin_count"][0]),
    )

    for path in output_paths.values():
        print(f"Wrote figure: {path}")
    print(f"Updated metadata: {meta_path}")
    return output_paths


def update_metadata_quicklook(
    meta_path: Path,
    *,
    significance_png: Path,
    smooth_sigma_deg: float,
    source_exclusion_radius_deg: float,
    sideband_stat: str,
    bg_max_abs_x_deg: float,
) -> None:
    meta: Dict[str, object] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    meta["quicklook"] = {
        "smooth_sigma_deg": float(smooth_sigma_deg),
        "source_exclusion_radius_deg": float(source_exclusion_radius_deg),
        "sideband_stat": sideband_stat,
        "bg_max_abs_x_deg": float(bg_max_abs_x_deg),
        "sideband_region": (
            "same Dec strip, r >= source_exclusion_radius_deg, "
            "abs(RA offset * cos(source Dec)) < bg_max_abs_x_deg"
        ),
        "excluded_region_note": "The quicklook background does not use ROI edge bins with abs(x) >= bg_max_abs_x_deg.",
        "approx_sigma_formula": "(smoothed_counts - smoothed_sideband_background) / sqrt(smoothed_sideband_background)",
        "not_for_physics": True,
    }
    outputs = meta.setdefault("outputs", {})
    outputs["approx_significance_png"] = str(significance_png)
    outputs["metadata_json"] = str(meta_path)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def generate_quicklook_outputs(args: argparse.Namespace, *, input_npz: Path, output_dir: Path, meta_path: Path) -> Dict[str, Path]:
    if not input_npz.exists():
        raise FileNotFoundError(f"Quicklook input NPZ does not exist: {input_npz}")

    with np.load(input_npz) as data:
        required = {"counts", "x_edges", "y_edges", "cell_id", "nhit_bin", "predE_bin"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{input_npz} is missing required arrays: {sorted(missing)}")

        arrays = {key: data[key] for key in data.files}
        counts = arrays["counts"]
        x_edges = arrays["x_edges"]
        y_edges = arrays["y_edges"]
        cells = cells_from_npz(data)

    pixel_size_deg = float(np.median(np.diff(x_edges)))
    if pixel_size_deg <= 0:
        raise ValueError("Invalid x_edges in quicklook input NPZ")
    sigma_px = float(args.smooth_sigma_deg) / pixel_size_deg
    smoothed_counts = smooth_maps(counts, sigma_px)
    background = sideband_background(
        counts,
        x_edges,
        y_edges,
        float(args.source_exclusion_radius_deg),
        args.sideband_stat,
        float(args.quicklook_bg_max_abs_x_deg),
    )
    smoothed_background = smooth_maps(background, sigma_px)
    excess_like = smoothed_counts - smoothed_background
    approx_sigma = excess_like / np.sqrt(np.maximum(smoothed_background, 1.0e-6))

    arrays["smoothed_counts"] = smoothed_counts.astype(np.float32)
    arrays["sideband_background"] = background.astype(np.float32)
    arrays["smoothed_sideband_background"] = smoothed_background.astype(np.float32)
    arrays["excess_like"] = excess_like.astype(np.float32)
    arrays["approx_sigma"] = approx_sigma.astype(np.float32)
    np.savez_compressed(input_npz, **arrays)

    output_dir.mkdir(parents=True, exist_ok=True)
    significance_png = output_dir / "crab_v1_approx_significance_grid.png"
    per_cell_roi_events = load_per_cell_roi_events(meta_path, counts)
    plot_grid(
        approx_sigma.astype(np.float32),
        cells,
        x_edges,
        y_edges,
        per_cell_roi_events,
        significance_png,
        title=f"{args.source_name} v1 cell approx significance quicklook",
        cmap_name="RdBu_r",
        scale="global",
        vmax_percentile=float(args.counts_vmax_percentile),
        significance_max=float(args.significance_max),
        colorbar_label="approx sigma",
    )
    update_metadata_quicklook(
        meta_path,
        significance_png=significance_png,
        smooth_sigma_deg=float(args.smooth_sigma_deg),
        source_exclusion_radius_deg=float(args.source_exclusion_radius_deg),
        sideband_stat=args.sideband_stat,
        bg_max_abs_x_deg=float(args.quicklook_bg_max_abs_x_deg),
    )

    print(f"Wrote figure: {significance_png}")
    print(f"Updated maps: {input_npz}")
    print(f"Updated metadata: {meta_path}")
    return {"approx_significance_png": significance_png}


def process_files(
    files: Sequence[Path],
    obs_root: Path,
    time_root: Path,
    cells: Sequence[Cell],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    required_event_branches = ["nv", "ml_logE_pred", "pincness", "fitstat", "theta"]
    required_time_branches = ["ra_mean_deg", "dec_mean_deg", "match_status"]

    counts = np.zeros((len(cells), len(y_edges) - 1, len(x_edges) - 1), dtype=np.int64)
    radec_counts = np.zeros((len(cells), len(dec_edges) - 1, len(ra_edges) - 1), dtype=np.int64)
    per_cell_cut_events = np.zeros(len(cells), dtype=np.int64)
    per_cell_roi_events = np.zeros(len(cells), dtype=np.int64)

    total_entries_seen = 0
    cut_match_events = 0
    roi_events = 0
    processed_files = 0
    missing_time_files: List[str] = []
    entry_mismatch_files: List[Dict[str, object]] = []

    theta_cut_rad = math.radians(float(args.cut_theta_max_deg))
    cos_dec = math.cos(math.radians(float(args.source_dec_deg)))
    half_width = float(args.half_width_deg)

    for file_idx, obs_file in enumerate(files, start=1):
        time_file = time_path_for(obs_file, obs_root, time_root)
        if not time_file.exists():
            missing_time_files.append(str(time_file))
            if args.allow_missing_time:
                continue
            raise FileNotFoundError(f"Missing recovered-time friend tree for {obs_file}: {time_file}")

        with open_root(obs_file) as obs_f, open_root(time_file) as time_f:
            if args.tree_name not in obs_f:
                raise KeyError(f"{obs_file} does not contain tree {args.tree_name!r}")
            if args.time_tree_name not in time_f:
                raise KeyError(f"{time_file} does not contain tree {args.time_tree_name!r}")
            tree = obs_f[args.tree_name]
            time_tree = time_f[args.time_tree_name]

            n_event = int(tree.num_entries)
            n_time = int(time_tree.num_entries)
            if n_event != n_time:
                mismatch = {"obs_file": str(obs_file), "time_file": str(time_file), "event_entries": n_event, "time_entries": n_time}
                entry_mismatch_files.append(mismatch)
                if not args.allow_entry_mismatch:
                    raise ValueError(f"Entry mismatch: {mismatch}")
            n_entries = min(n_event, n_time)
            total_entries_seen += n_entries

            for start in range(0, n_entries, int(args.entries_per_chunk)):
                stop = min(start + int(args.entries_per_chunk), n_entries)
                event_arrays = tree.arrays(required_event_branches, entry_start=start, entry_stop=stop, library="np")
                time_arrays = time_tree.arrays(required_time_branches, entry_start=start, entry_stop=stop, library="np")

                nv = np.asarray(event_arrays["nv"])
                loge_pred = np.asarray(event_arrays["ml_logE_pred"], dtype=np.float64)
                ra = np.asarray(time_arrays["ra_mean_deg"], dtype=np.float64)
                dec = np.asarray(time_arrays["dec_mean_deg"], dtype=np.float64)

                mask = np.asarray(time_arrays["match_status"] == int(args.match_status_equals))
                mask &= np.asarray(event_arrays["pincness"] < float(args.cut_pinc_max))
                mask &= np.asarray(event_arrays["fitstat"] == int(args.cut_fitstat_equals))
                mask &= np.asarray(event_arrays["theta"] < theta_cut_rad)
                mask &= np.isfinite(loge_pred) & np.isfinite(ra) & np.isfinite(dec)

                cut_match_events += int(mask.sum())

                dra = wrapped_delta_ra_deg(ra, float(args.source_ra_deg))
                x = dra * cos_dec
                y = dec - float(args.source_dec_deg)
                in_roi = mask & (x >= -half_width) & (x <= half_width) & (y >= -half_width) & (y <= half_width)
                roi_events += int(in_roi.sum())

                for cell in cells:
                    cell_mask = mask & in_interval(nv, cell.nhit_low, cell.nhit_high) & in_interval(
                        loge_pred, cell.pred_low, cell.pred_high
                    )
                    per_cell_cut_events[cell.index] += int(cell_mask.sum())
                    cell_roi_mask = cell_mask & in_roi
                    per_cell_roi_events[cell.index] += int(cell_roi_mask.sum())
                    if not np.any(cell_roi_mask):
                        continue

                    hist, _, _ = np.histogram2d(x[cell_roi_mask], y[cell_roi_mask], bins=(x_edges, y_edges))
                    counts[cell.index] += hist.T.astype(np.int64, copy=False)
                    radec_hist, _, _ = np.histogram2d(
                        ra[cell_roi_mask],
                        dec[cell_roi_mask],
                        bins=(ra_edges, dec_edges),
                    )
                    radec_counts[cell.index] += radec_hist.T.astype(np.int64, copy=False)

        processed_files += 1
        if args.print_every > 0 and (file_idx % args.print_every == 0 or file_idx == len(files)):
            print(
                f"[{file_idx}/{len(files)}] processed={processed_files}, "
                f"entries={total_entries_seen:,}, cut/match={cut_match_events:,}, roi={roi_events:,}",
                flush=True,
            )

    meta = {
        "input_file_count": len(files),
        "processed_file_count": processed_files,
        "total_entries_seen": int(total_entries_seen),
        "cut_match_events": int(cut_match_events),
        "roi_events": int(roi_events),
        "missing_time_file_count": len(missing_time_files),
        "missing_time_files": missing_time_files,
        "entry_mismatch_file_count": len(entry_mismatch_files),
        "entry_mismatch_files": entry_mismatch_files,
        "processed_files_sample": [str(p) for p in files[:5]],
    }
    return counts, radec_counts, {
        **meta,
        "per_cell_cut_events": per_cell_cut_events.tolist(),
        "per_cell_roi_events": per_cell_roi_events.tolist(),
    }


def write_outputs(
    counts: np.ndarray,
    radec_counts: np.ndarray,
    smoothed_counts: np.ndarray,
    background: np.ndarray,
    smoothed_background: np.ndarray,
    approx_sigma: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    cells: Sequence[Cell],
    output_dir: Path,
    metadata: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_cell_roi_events = np.asarray(metadata["per_cell_roi_events"], dtype=np.int64)

    counts_png = output_dir / "crab_v1_counts_grid.png"
    radec_counts_png = output_dir / "crab_v1_counts_radec_grid.png"
    smoothed_png = output_dir / "crab_v1_smoothed_counts_grid.png"
    significance_png = output_dir / "crab_v1_approx_significance_grid.png"

    plot_grid(
        counts.astype(np.float32),
        cells,
        x_edges,
        y_edges,
        per_cell_roi_events,
        counts_png,
        title=f"{args.source_name} v1 cell counts map",
        cmap_name="viridis",
        scale=args.counts_scale,
        vmax_percentile=float(args.counts_vmax_percentile),
        significance_max=float(args.significance_max),
        colorbar_label="counts / pixel",
    )
    plot_grid(
        radec_counts.astype(np.float32),
        cells,
        ra_edges,
        dec_edges,
        per_cell_roi_events,
        radec_counts_png,
        title=f"{args.source_name} v1 cell RA/Dec counts map",
        cmap_name="viridis",
        scale=args.counts_scale,
        vmax_percentile=float(args.counts_vmax_percentile),
        significance_max=float(args.significance_max),
        colorbar_label="counts / pixel",
        marker_x=float(args.source_ra_deg),
        marker_y=float(args.source_dec_deg),
        x_label="RA (deg; increasing right)",
        y_label="Dec (deg)",
    )
    plot_grid(
        smoothed_counts,
        cells,
        x_edges,
        y_edges,
        per_cell_roi_events,
        smoothed_png,
        title=f"{args.source_name} v1 cell smoothed counts map, sigma={args.smooth_sigma_deg:g} deg",
        cmap_name="viridis",
        scale=args.counts_scale,
        vmax_percentile=float(args.counts_vmax_percentile),
        significance_max=float(args.significance_max),
        colorbar_label="smoothed counts / pixel",
    )
    plot_grid(
        approx_sigma,
        cells,
        x_edges,
        y_edges,
        per_cell_roi_events,
        significance_png,
        title=f"{args.source_name} v1 cell approx significance quicklook",
        cmap_name="RdBu_r",
        scale="global",
        vmax_percentile=float(args.counts_vmax_percentile),
        significance_max=float(args.significance_max),
        colorbar_label="approx sigma",
    )

    npz_path = output_dir / "crab_v1_maps.npz"
    np.savez_compressed(
        npz_path,
        counts=counts,
        radec_counts=radec_counts,
        smoothed_counts=smoothed_counts.astype(np.float32),
        sideband_background=background.astype(np.float32),
        smoothed_sideband_background=smoothed_background.astype(np.float32),
        excess_like=(smoothed_counts - smoothed_background).astype(np.float32),
        approx_sigma=approx_sigma.astype(np.float32),
        x_edges=x_edges,
        y_edges=y_edges,
        ra_edges=ra_edges,
        dec_edges=dec_edges,
        cell_id=np.asarray([c.cell_id for c in cells], dtype=np.int32),
        nhit_bin=np.asarray([c.nhit_bin for c in cells]),
        predE_bin=np.asarray([c.predE_bin for c in cells]),
    )

    meta_path = output_dir / "crab_v1_maps_meta.json"
    meta_payload = {
        "description": "Crab-centered v1-cell sky maps. Approx significance is a quicklook only, not the Stage-D background result.",
        "obs_root": str(Path(args.obs_root).resolve()),
        "time_root": str(Path(args.time_root).resolve()),
        "cell_selection_csv": str(Path(args.cell_selection_csv).resolve()),
        "output_dir": str(output_dir.resolve()),
        "source": {"name": args.source_name, "ra_deg": args.source_ra_deg, "dec_deg": args.source_dec_deg},
        "projection": "x=(ra-source_ra wrapped)*cos(source_dec), y=dec-source_dec",
        "radec_projection": {
            "ra_axis": "increasing_right",
            "ra_range_deg": [float(ra_edges[0]), float(ra_edges[-1])],
            "dec_range_deg": [float(dec_edges[0]), float(dec_edges[-1])],
            "source_marker": [float(args.source_ra_deg), float(args.source_dec_deg)],
            "range_definition": "RA: source_ra +/- half_width_deg / cos(source_dec); Dec: source_dec +/- half_width_deg",
        },
        "roi": {"half_width_deg": args.half_width_deg, "pixel_size_deg": args.pixel_size_deg},
        "cuts": {
            "match_status_equals": args.match_status_equals,
            "pincness_lt": args.cut_pinc_max,
            "fitstat_equals": args.cut_fitstat_equals,
            "theta_deg_lt": args.cut_theta_max_deg,
        },
        "quicklook": {
            "smooth_sigma_deg": args.smooth_sigma_deg,
            "source_exclusion_radius_deg": args.source_exclusion_radius_deg,
            "sideband_stat": args.sideband_stat,
            "bg_max_abs_x_deg": args.quicklook_bg_max_abs_x_deg,
            "sideband_region": (
                "same Dec strip, r >= source_exclusion_radius_deg, "
                "abs(RA offset * cos(source Dec)) < bg_max_abs_x_deg"
            ),
            "excluded_region_note": "The quicklook background does not use ROI edge bins with abs(x) >= bg_max_abs_x_deg.",
            "approx_sigma_formula": "(smoothed_counts - smoothed_sideband_background) / sqrt(smoothed_sideband_background)",
            "not_for_physics": True,
        },
        "plotting": {
            "counts_scale": args.counts_scale,
            "counts_vmax_percentile": args.counts_vmax_percentile,
            "significance_max": args.significance_max,
        },
        "cells": [
            {
                "index": c.index,
                "cell_id": c.cell_id,
                "nhit_bin": c.nhit_bin,
                "predE_bin": c.predE_bin,
                "cut_events": int(metadata["per_cell_cut_events"][c.index]),
                "roi_events": int(metadata["per_cell_roi_events"][c.index]),
            }
            for c in cells
        ],
        "processing": {k: v for k, v in metadata.items() if k not in {"per_cell_cut_events", "per_cell_roi_events"}},
        "outputs": {
            "counts_png": str(counts_png),
            "counts_radec_png": str(radec_counts_png),
            "smoothed_counts_png": str(smoothed_png),
            "approx_significance_png": str(significance_png),
            "maps_npz": str(npz_path),
            "metadata_json": str(meta_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"Wrote figure: {counts_png}")
    print(f"Wrote figure: {radec_counts_png}")
    print(f"Wrote figure: {smoothed_png}")
    print(f"Wrote figure: {significance_png}")
    print(f"Wrote maps: {npz_path}")
    print(f"Wrote metadata: {meta_path}")


def main() -> None:
    args = parse_args()
    if args.entries_per_chunk <= 0:
        raise ValueError("--entries-per-chunk must be positive")
    if args.quicklook_bg_max_abs_x_deg <= 0:
        raise ValueError("--quicklook-bg-max-abs-x-deg must be positive")
    if args.profile_half_width_deg <= 0:
        raise ValueError("--profile-half-width-deg must be positive")
    if args.profile_sideband_min_deg <= 0:
        raise ValueError("--profile-sideband-min-deg must be positive")

    obs_root = Path(args.obs_root).resolve()
    time_root = Path(args.time_root).resolve()
    cell_selection_csv = Path(args.cell_selection_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile_input_npz = Path(args.profile_input_npz).resolve() if args.profile_input_npz else output_dir / "crab_v1_maps.npz"
    quicklook_input_npz = Path(args.quicklook_input_npz).resolve() if args.quicklook_input_npz else output_dir / "crab_v1_maps.npz"
    meta_path = output_dir / "crab_v1_maps_meta.json"

    if args.quicklook_only:
        generate_quicklook_outputs(args, input_npz=quicklook_input_npz, output_dir=output_dir, meta_path=meta_path)
        return

    if args.profiles_only:
        generate_profile_outputs(args, input_npz=profile_input_npz, output_dir=output_dir, meta_path=meta_path)
        return

    cells = load_cells(cell_selection_csv)
    cells = [
        Cell(index=i, cell_id=c.cell_id, nhit_bin=c.nhit_bin, predE_bin=c.predE_bin, nhit_low=c.nhit_low, nhit_high=c.nhit_high, pred_low=c.pred_low, pred_high=c.pred_high)
        for i, c in enumerate(cells)
    ]
    files = discover_observation_files(obs_root, args.file_glob, args.day_prefix, args.max_files)
    x_edges = build_edges(float(args.half_width_deg), float(args.pixel_size_deg))
    y_edges = build_edges(float(args.half_width_deg), float(args.pixel_size_deg))
    ra_edges, dec_edges = build_radec_edges(
        float(args.source_ra_deg),
        float(args.source_dec_deg),
        float(args.half_width_deg),
        float(args.pixel_size_deg),
    )

    print(f"Loaded {len(cells)} selected cells from {cell_selection_csv}")
    print(f"Discovered {len(files)} observation files under {obs_root}")
    print(
        f"Sky map: half_width={args.half_width_deg:g} deg, pixel={args.pixel_size_deg:g} deg, "
        f"shape=({len(y_edges) - 1}, {len(x_edges) - 1})",
        flush=True,
    )
    print(
        f"RA/Dec map: ra=[{ra_edges[0]:.4f}, {ra_edges[-1]:.4f}] deg, "
        f"dec=[{dec_edges[0]:.4f}, {dec_edges[-1]:.4f}] deg, RA increasing right",
        flush=True,
    )

    counts, radec_counts, metadata = process_files(files, obs_root, time_root, cells, x_edges, y_edges, ra_edges, dec_edges, args)
    sigma_px = float(args.smooth_sigma_deg) / float(args.pixel_size_deg)
    smoothed_counts = smooth_maps(counts, sigma_px)
    background = sideband_background(
        counts,
        x_edges,
        y_edges,
        float(args.source_exclusion_radius_deg),
        args.sideband_stat,
        float(args.quicklook_bg_max_abs_x_deg),
    )
    smoothed_background = smooth_maps(background, sigma_px)
    excess_like = smoothed_counts - smoothed_background
    approx_sigma = excess_like / np.sqrt(np.maximum(smoothed_background, 1.0e-6))

    write_outputs(
        counts=counts,
        radec_counts=radec_counts,
        smoothed_counts=smoothed_counts,
        background=background,
        smoothed_background=smoothed_background,
        approx_sigma=approx_sigma.astype(np.float32),
        x_edges=x_edges,
        y_edges=y_edges,
        ra_edges=ra_edges,
        dec_edges=dec_edges,
        cells=cells,
        output_dir=output_dir,
        metadata=metadata,
        args=args,
    )
    generate_profile_outputs(args, input_npz=output_dir / "crab_v1_maps.npz", output_dir=output_dir, meta_path=meta_path)


if __name__ == "__main__":
    main()
