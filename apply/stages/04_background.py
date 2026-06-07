#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_STAGE_C_DIR = "apply/output/stage_c/current"
DEFAULT_PSF_NPZ = "apply/output/stage_b/current/psf_v1.npz"
DEFAULT_CELL_SELECTION = "apply/config/cell_selection_v1.csv"
DEFAULT_OUTPUT_DIR = "apply/output/stage_d"
DEFAULT_LHAASO_LAT_DEG = 29.45
DEFAULT_LHAASO_LON_DEG = 100.14
DEFAULT_THETA_MAX_DEG = 50.0
DEFAULT_SOURCE_RA_DEG = 83.63
DEFAULT_SOURCE_DEC_DEG = 22.01


@dataclass(frozen=True)
class CellSpec:
    index: int
    cell_id: int
    nhit_bin: str
    predE_bin: str
    mc_count: int
    selection_version: str
    selection_reason: str


@dataclass(frozen=True)
class SourceMask:
    name: str
    ra_deg: float
    dec_deg: float
    radius_deg: float
    enabled: bool = True


@dataclass
class ScanResult:
    counts_unmasked_flat: np.ndarray
    counts_masked_flat: np.ndarray
    cell_time_counts: np.ndarray
    cell_total_events: np.ndarray
    cell_grid_events: np.ndarray
    cell_source_masked_events: np.ndarray
    cell_out_of_grid_events: np.ndarray
    input_rows: int
    processed_batches: int
    theta_check: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage D direct-integration background for Crab SED v1 cells.")
    parser.add_argument("--stage-c-dir", type=str, default=DEFAULT_STAGE_C_DIR)
    parser.add_argument("--psf-npz", type=str, default=DEFAULT_PSF_NPZ)
    parser.add_argument("--cell-selection-csv", type=str, default=DEFAULT_CELL_SELECTION)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run directory name under <output-dir>/runs. Defaults to Slurm job id or a timestamp.",
    )
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)

    parser.add_argument("--time-bin-minutes", type=float, default=20.0)
    parser.add_argument("--ha-min-deg", type=float, default=-65.0)
    parser.add_argument("--ha-max-deg", type=float, default=65.0)
    parser.add_argument("--dec-min-deg", type=float, default=-25.0)
    parser.add_argument("--dec-max-deg", type=float, default=85.0)
    parser.add_argument("--grid-step-deg", type=float, default=0.25)
    parser.add_argument("--theta-max-deg", type=float, default=DEFAULT_THETA_MAX_DEG)
    parser.add_argument("--lhaaso-lat-deg", type=float, default=DEFAULT_LHAASO_LAT_DEG)
    parser.add_argument("--lhaaso-lon-deg", type=float, default=DEFAULT_LHAASO_LON_DEG)
    parser.add_argument("--source-ra-deg", type=float, default=DEFAULT_SOURCE_RA_DEG)
    parser.add_argument("--source-dec-deg", type=float, default=DEFAULT_SOURCE_DEC_DEG)

    parser.add_argument("--batch-size", type=int, default=500000)
    parser.add_argument("--workers", type=int, default=1, help="Reserved for Slurm resource accounting; scanning is vectorized.")
    parser.add_argument("--max-batches", type=int, default=None, help="Read only the first N parquet batches for smoke tests.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true", default=False)

    parser.add_argument("--theta-check-max-events", type=int, default=20000)
    parser.add_argument("--theta-check-warn-p95-deg", type=float, default=0.5)
    parser.add_argument("--theta-check-fail-p95-deg", type=float, default=1.0)
    parser.add_argument("--npz-name", type=str, default="background_v1.npz")
    parser.add_argument("--metadata-name", type=str, default="background_v1_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="background_v1_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="background_v1_summary.md")
    return parser.parse_args()


def default_source_masks() -> List[SourceMask]:
    return [
        SourceMask("Crab", 83.63, 22.01, 2.0),
        SourceMask("Mrk421", 166.1138, 38.2088, 3.0),
        SourceMask("Mrk501", 253.4676, 39.7602, 3.0),
        SourceMask("Geminga", 98.4756, 17.7703, 3.0),
        SourceMask("Cygnus_Cocoon", 307.7, 41.0, 8.0),
    ]


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


def load_cells(selection_csv: Path) -> List[CellSpec]:
    cells: List[CellSpec] = []
    with selection_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"cell_id", "nhit_bin", "predE_bin"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{selection_csv} is missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            cells.append(
                CellSpec(
                    index=idx,
                    cell_id=int(row.get("cell_id") or (idx + 1)),
                    nhit_bin=row["nhit_bin"],
                    predE_bin=row["predE_bin"],
                    mc_count=int(row.get("mc_count") or 0),
                    selection_version=row.get("selection_version", ""),
                    selection_reason=row.get("selection_reason", ""),
                )
            )
    if not cells:
        raise ValueError(f"No cells loaded from {selection_csv}")
    sorted_cells = sorted(cells, key=lambda c: (interval_key(c.nhit_bin), interval_key(c.predE_bin), c.cell_id))
    return [
        CellSpec(
            index=idx,
            cell_id=cell.cell_id,
            nhit_bin=cell.nhit_bin,
            predE_bin=cell.predE_bin,
            mc_count=cell.mc_count,
            selection_version=cell.selection_version,
            selection_reason=cell.selection_reason,
        )
        for idx, cell in enumerate(sorted_cells)
    ]


def make_default_run_id() -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"slurm_{slurm_job_id}"
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not value:
        raise ValueError("--run-id cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("--run-id may only contain letters, digits, dots, underscores, and hyphens")
    if value in {".", ".."}:
        raise ValueError(f"Invalid --run-id: {value!r}")
    return value


def prepare_run_output_dir(output_root: Path, run_id: str, *, overwrite_run_dir: bool) -> Path:
    run_dir = output_root / "runs" / run_id
    if run_dir.exists():
        if overwrite_run_dir:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"Stage D run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def replace_path_atomic(target: Path, replacement: Path) -> None:
    backup = target.with_name(f".{target.name}.old")
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    if target.exists() or target.is_symlink():
        target.replace(backup)
    replacement.replace(target)
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def copytree_atomic(source: Path, target: Path) -> None:
    tmp = target.with_name(f".{target.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        if tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()
    shutil.copytree(source, tmp)
    replace_path_atomic(target, tmp)


def symlink_atomic(link_path: Path, target: Path) -> None:
    tmp = link_path.with_name(f".{link_path.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    replace_path_atomic(link_path, tmp)


def promote_successful_run(output_root: Path, run_dir: Path) -> None:
    current = output_root / "current"
    latest = output_root / "latest"
    try:
        current_tmp = output_root / ".current.tmp"
        if current_tmp.exists() or current_tmp.is_symlink():
            if current_tmp.is_dir() and not current_tmp.is_symlink():
                shutil.rmtree(current_tmp)
            else:
                current_tmp.unlink()
        current_tmp.symlink_to(run_dir)
        replace_path_atomic(current, current_tmp)
    except OSError:
        copytree_atomic(run_dir, current)
    try:
        symlink_atomic(latest, run_dir)
    except OSError:
        latest.write_text(str(run_dir) + "\n", encoding="utf-8")


def make_edges(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive")
    n_steps = int(round((stop - start) / step))
    if n_steps <= 0:
        raise ValueError(f"Invalid edge range: start={start}, stop={stop}, step={step}")
    edges = start + step * np.arange(n_steps + 1, dtype=np.float64)
    if not np.isclose(edges[-1], stop):
        raise ValueError(f"Step does not land on stop: start={start}, stop={stop}, step={step}")
    return edges


def wrapped_delta_deg(values: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(values) + 180.0) % 360.0) - 180.0


def gmst_deg_from_mjd(mjd: np.ndarray | float) -> np.ndarray:
    jd = np.asarray(mjd, dtype=np.float64) + 2400000.5
    t = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    )
    return np.mod(gmst, 360.0)


def local_sidereal_deg(mjd: np.ndarray | float, longitude_east_deg: float) -> np.ndarray:
    return np.mod(gmst_deg_from_mjd(mjd) + float(longitude_east_deg), 360.0)


def zenith_deg_from_ha_dec(ha_deg: np.ndarray, dec_deg: np.ndarray, lat_deg: float) -> np.ndarray:
    ha = np.radians(ha_deg)
    dec = np.radians(dec_deg)
    lat = math.radians(float(lat_deg))
    cosz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cosz, -1.0, 1.0)))


def event_source_mask(ra_deg: np.ndarray, dec_deg: np.ndarray, sources: Sequence[SourceMask]) -> np.ndarray:
    mask = np.zeros(ra_deg.shape, dtype=bool)
    if not sources:
        return mask
    dec_rad = np.radians(dec_deg)
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    for source in sources:
        if not source.enabled:
            continue
        source_dec = math.radians(source.dec_deg)
        dra = np.radians(((ra_deg - source.ra_deg + 180.0) % 360.0) - 180.0)
        cos_sep = sin_dec * math.sin(source_dec) + cos_dec * math.cos(source_dec) * np.cos(dra)
        mask |= cos_sep >= math.cos(math.radians(source.radius_deg))
    return mask


def disk_indices_for_center(
    center_ha_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    ha_centers: np.ndarray,
    dec_centers: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    n_ha = ha_centers.size
    if radius_deg <= 0:
        return np.empty(0, dtype=np.int64)

    step_ha = float(np.median(np.diff(ha_centers))) if ha_centers.size > 1 else radius_deg
    step_dec = float(np.median(np.diff(dec_centers))) if dec_centers.size > 1 else radius_deg
    dec_min = center_dec_deg - radius_deg - step_dec
    dec_max = center_dec_deg + radius_deg + step_dec
    d0 = max(0, int(np.searchsorted(dec_centers, dec_min, side="left")))
    d1 = min(dec_centers.size, int(np.searchsorted(dec_centers, dec_max, side="right")))
    if d0 >= d1:
        return np.empty(0, dtype=np.int64)

    cos_dec = max(abs(math.cos(math.radians(center_dec_deg))), 0.10)
    ha_half_width = min(180.0, radius_deg / cos_dec + step_ha)
    ha_min = center_ha_deg - ha_half_width
    ha_max = center_ha_deg + ha_half_width
    h0 = max(0, int(np.searchsorted(ha_centers, ha_min, side="left")))
    h1 = min(ha_centers.size, int(np.searchsorted(ha_centers, ha_max, side="right")))
    if h0 >= h1:
        return np.empty(0, dtype=np.int64)

    ha_slice = ha_centers[h0:h1]
    dec_slice = dec_centers[d0:d1]
    dec_grid = np.radians(dec_slice[:, None])
    dha = np.radians(ha_slice[None, :] - center_ha_deg)
    center_dec = math.radians(center_dec_deg)
    cos_sep = np.sin(dec_grid) * math.sin(center_dec) + np.cos(dec_grid) * math.cos(center_dec) * np.cos(dha)
    disk = cos_sep >= math.cos(math.radians(radius_deg))
    disk &= visible_mask[d0:d1, h0:h1]
    if not np.any(disk):
        return np.empty(0, dtype=np.int64)

    yy, xx = np.nonzero(disk)
    return ((yy + d0) * n_ha + (xx + h0)).astype(np.int64, copy=False)


def source_mask_indices_for_lst(
    lst_deg: float,
    sources: Sequence[SourceMask],
    ha_centers: np.ndarray,
    dec_centers: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for source in sources:
        if not source.enabled:
            continue
        source_ha = float(((lst_deg - source.ra_deg + 180.0) % 360.0) - 180.0)
        idx = disk_indices_for_center(source_ha, source.dec_deg, source.radius_deg, ha_centers, dec_centers, visible_mask)
        if idx.size:
            pieces.append(idx)
    if not pieces:
        return np.empty(0, dtype=np.int64)
    if len(pieces) == 1:
        return pieces[0]
    return np.unique(np.concatenate(pieces))


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not np.isfinite(number):
        return None
    return number


def build_time_edges(stage_c_dir: Path, metadata: Dict[str, object], time_bin_minutes: float) -> np.ndarray:
    if time_bin_minutes <= 0:
        raise ValueError("--time-bin-minutes must be positive")
    step_days = float(time_bin_minutes) / (24.0 * 60.0)
    mjd_coverage = metadata.get("mjd_coverage", {}) if metadata else {}
    mjd_min = finite_float(mjd_coverage.get("selected_mjd_min") if isinstance(mjd_coverage, dict) else None)
    mjd_max = finite_float(mjd_coverage.get("selected_mjd_max") if isinstance(mjd_coverage, dict) else None)

    if mjd_min is None or mjd_max is None:
        source_files = stage_c_dir / "source_files.csv"
        mins: List[float] = []
        maxes: List[float] = []
        with source_files.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row_min = finite_float(row.get("selected_mjd_min")) or finite_float(row.get("matched_mjd_min"))
                row_max = finite_float(row.get("selected_mjd_max")) or finite_float(row.get("matched_mjd_max"))
                if row_min is not None:
                    mins.append(row_min)
                if row_max is not None:
                    maxes.append(row_max)
        if not mins or not maxes:
            raise ValueError("Could not determine Stage C MJD coverage.")
        mjd_min = min(mins)
        mjd_max = max(maxes)

    start = math.floor(float(mjd_min) / step_days) * step_days
    stop = math.ceil(float(mjd_max) / step_days) * step_days
    edges = np.arange(start, stop + 0.5 * step_days, step_days, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("Time binning produced fewer than one bin.")
    if edges[-1] <= mjd_max:
        edges = np.append(edges, edges[-1] + step_days)
    return edges


def live_time_by_bin(source_files_csv: Path, time_edges_mjd: np.ndarray) -> np.ndarray:
    live_time_sec = np.zeros(time_edges_mjd.size - 1, dtype=np.float64)
    if not source_files_csv.exists():
        return live_time_sec
    with source_files_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            start = finite_float(row.get("matched_mjd_min"))
            stop = finite_float(row.get("matched_mjd_max"))
            rough_live = finite_float(row.get("rough_live_time_seconds"))
            if start is None or stop is None or rough_live is None or stop <= start or rough_live <= 0:
                continue
            span_sec = (stop - start) * 86400.0
            if span_sec <= 0:
                continue
            first = max(0, int(np.searchsorted(time_edges_mjd, start, side="right") - 1))
            last = min(live_time_sec.size - 1, int(np.searchsorted(time_edges_mjd, stop, side="left")))
            for idx in range(first, last + 1):
                overlap = max(0.0, min(stop, time_edges_mjd[idx + 1]) - max(start, time_edges_mjd[idx]))
                if overlap > 0:
                    live_time_sec[idx] += rough_live * (overlap * 86400.0 / span_sec)
    return live_time_sec


def load_psf_by_cell(psf_npz: Path, cells: Sequence[CellSpec]) -> Dict[int, Dict[str, float]]:
    if not psf_npz.exists():
        raise FileNotFoundError(f"Stage B PSF NPZ does not exist: {psf_npz}")
    data = np.load(psf_npz)
    required = {"cell_id", "r_opt_deg", "sigma_deg", "containment_r_opt"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{psf_npz} is missing required arrays: {sorted(missing)}")
    by_id: Dict[int, Dict[str, float]] = {}
    for idx, cell_id in enumerate(data["cell_id"]):
        by_id[int(cell_id)] = {
            "r_opt_deg": float(data["r_opt_deg"][idx]),
            "sigma_deg": float(data["sigma_deg"][idx]),
            "containment_r_opt": float(data["containment_r_opt"][idx]),
        }
    missing_cells = [cell.cell_id for cell in cells if cell.cell_id not in by_id]
    if missing_cells:
        raise ValueError(f"PSF table is missing selected cells: {missing_cells}")
    return by_id


def column_to_numpy(batch, name: str) -> np.ndarray:
    return batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)


def update_bincount(target: np.ndarray, linear_index: np.ndarray, minlength: int) -> None:
    if linear_index.size == 0:
        return
    target += np.bincount(linear_index, minlength=minlength)


def scan_stage_c_events(
    obs_events_dir: Path,
    cells: Sequence[CellSpec],
    time_edges_mjd: np.ndarray,
    ha_edges_deg: np.ndarray,
    dec_edges_deg: np.ndarray,
    sources: Sequence[SourceMask],
    *,
    latitude_deg: float,
    longitude_east_deg: float,
    batch_size: int,
    max_batches: Optional[int],
    print_every: int,
    theta_check_max_events: int,
) -> ScanResult:
    dataset = ds.dataset(obs_events_dir, format="parquet", partitioning="hive")
    columns = ["mjd", "ra_mean_deg", "dec_mean_deg", "theta", "cell_id"]
    scanner = dataset.scanner(columns=columns, batch_size=int(batch_size), use_threads=True)
    n_cells = len(cells)
    n_time = time_edges_mjd.size - 1
    n_ha = ha_edges_deg.size - 1
    n_dec = dec_edges_deg.size - 1
    n_grid = n_ha * n_dec
    flat_size = n_cells * n_grid

    max_cell_id = max(cell.cell_id for cell in cells)
    cell_index_by_id = np.full(max_cell_id + 1, -1, dtype=np.int16)
    for cell in cells:
        cell_index_by_id[cell.cell_id] = np.int16(cell.index)

    counts_unmasked_flat = np.zeros(flat_size, dtype=np.int64)
    counts_masked_flat = np.zeros(flat_size, dtype=np.int64)
    cell_time_counts = np.zeros((n_cells, n_time), dtype=np.int64)
    cell_total_events = np.zeros(n_cells, dtype=np.int64)
    cell_grid_events = np.zeros(n_cells, dtype=np.int64)
    cell_source_masked_events = np.zeros(n_cells, dtype=np.int64)
    cell_out_of_grid_events = np.zeros(n_cells, dtype=np.int64)

    theta_sample_mjd: List[np.ndarray] = []
    theta_sample_ra: List[np.ndarray] = []
    theta_sample_dec: List[np.ndarray] = []
    theta_sample_theta: List[np.ndarray] = []
    theta_sample_count = 0

    input_rows = 0
    processed_batches = 0
    time_start = float(time_edges_mjd[0])
    time_step = float(time_edges_mjd[1] - time_edges_mjd[0])
    ha_min = float(ha_edges_deg[0])
    dec_min = float(dec_edges_deg[0])
    ha_step = float(ha_edges_deg[1] - ha_edges_deg[0])
    dec_step = float(dec_edges_deg[1] - dec_edges_deg[0])

    for batch_idx, batch in enumerate(scanner.to_batches(), start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        processed_batches += 1
        n_rows = int(batch.num_rows)
        input_rows += n_rows

        mjd = np.asarray(column_to_numpy(batch, "mjd"), dtype=np.float64)
        ra = np.asarray(column_to_numpy(batch, "ra_mean_deg"), dtype=np.float64)
        dec = np.asarray(column_to_numpy(batch, "dec_mean_deg"), dtype=np.float64)
        theta = np.asarray(column_to_numpy(batch, "theta"), dtype=np.float64)
        cell_id = np.asarray(column_to_numpy(batch, "cell_id"), dtype=np.int32)

        valid_id = (cell_id >= 0) & (cell_id < cell_index_by_id.size)
        cell_idx = np.full(cell_id.shape, -1, dtype=np.int16)
        cell_idx[valid_id] = cell_index_by_id[cell_id[valid_id]]
        valid_cell = cell_idx >= 0

        finite = valid_cell & np.isfinite(mjd) & np.isfinite(ra) & np.isfinite(dec)
        if not np.any(finite):
            continue

        valid_indices = np.nonzero(finite)[0]
        mjd_v = mjd[valid_indices]
        ra_v = ra[valid_indices]
        dec_v = dec[valid_indices]
        cell_v = np.asarray(cell_idx[valid_indices], dtype=np.int64)

        update_bincount(cell_total_events, cell_v, n_cells)

        time_idx = np.floor((mjd_v - time_start) / time_step).astype(np.int64)
        valid_time = (time_idx >= 0) & (time_idx < n_time)
        if np.any(valid_time):
            linear_time = cell_v[valid_time] * n_time + time_idx[valid_time]
            cell_time_counts += np.bincount(linear_time, minlength=n_cells * n_time).reshape(n_cells, n_time)

        lst = local_sidereal_deg(mjd_v, longitude_east_deg)
        ha = ((lst - ra_v + 180.0) % 360.0) - 180.0
        ha_idx = np.floor((ha - ha_min) / ha_step).astype(np.int64)
        dec_idx = np.floor((dec_v - dec_min) / dec_step).astype(np.int64)
        valid_grid = (ha_idx >= 0) & (ha_idx < n_ha) & (dec_idx >= 0) & (dec_idx < n_dec)

        if np.any(valid_grid):
            grid_idx = dec_idx[valid_grid] * n_ha + ha_idx[valid_grid]
            linear_grid = cell_v[valid_grid] * n_grid + grid_idx
            update_bincount(counts_unmasked_flat, linear_grid, flat_size)
            update_bincount(cell_grid_events, cell_v[valid_grid], n_cells)

        out_of_grid = ~valid_grid
        if np.any(out_of_grid):
            update_bincount(cell_out_of_grid_events, cell_v[out_of_grid], n_cells)

        source_masked = event_source_mask(ra_v, dec_v, sources)
        if np.any(source_masked):
            update_bincount(cell_source_masked_events, cell_v[source_masked], n_cells)
        masked_training = valid_grid & (~source_masked)
        if np.any(masked_training):
            grid_idx = dec_idx[masked_training] * n_ha + ha_idx[masked_training]
            linear_grid = cell_v[masked_training] * n_grid + grid_idx
            update_bincount(counts_masked_flat, linear_grid, flat_size)

        if theta_sample_count < theta_check_max_events:
            theta_v = theta[valid_indices]
            sanity = np.isfinite(theta_v)
            if np.any(sanity):
                remaining = int(theta_check_max_events) - theta_sample_count
                take_idx = np.nonzero(sanity)[0][:remaining]
                if take_idx.size:
                    theta_sample_mjd.append(mjd_v[take_idx].copy())
                    theta_sample_ra.append(ra_v[take_idx].copy())
                    theta_sample_dec.append(dec_v[take_idx].copy())
                    theta_sample_theta.append(theta_v[take_idx].copy())
                    theta_sample_count += int(take_idx.size)

        if print_every > 0 and (batch_idx % print_every == 0):
            print(
                f"[batch {batch_idx}] rows={input_rows:,} selected={int(cell_total_events.sum()):,}",
                flush=True,
            )

    theta_check: Dict[str, object] = {"sample_events": int(theta_sample_count), "status": "not_run"}
    if theta_sample_count > 0:
        sample_mjd = np.concatenate(theta_sample_mjd)
        sample_ra = np.concatenate(theta_sample_ra)
        sample_dec = np.concatenate(theta_sample_dec)
        sample_theta = np.concatenate(theta_sample_theta)
        sample_lst = local_sidereal_deg(sample_mjd, longitude_east_deg)
        sample_ha = ((sample_lst - sample_ra + 180.0) % 360.0) - 180.0
        theta_pred_deg = zenith_deg_from_ha_dec(sample_ha, sample_dec, latitude_deg)
        theta_observed_deg = np.degrees(sample_theta)
        absdiff = np.abs(theta_pred_deg - theta_observed_deg)
        theta_check = {
            "sample_events": int(theta_sample_count),
            "status": "ok",
            "median_absdiff_deg": float(np.median(absdiff)),
            "p95_absdiff_deg": float(np.percentile(absdiff, 95.0)),
            "max_absdiff_deg": float(np.max(absdiff)),
        }

    return ScanResult(
        counts_unmasked_flat=counts_unmasked_flat,
        counts_masked_flat=counts_masked_flat,
        cell_time_counts=cell_time_counts,
        cell_total_events=cell_total_events,
        cell_grid_events=cell_grid_events,
        cell_source_masked_events=cell_source_masked_events,
        cell_out_of_grid_events=cell_out_of_grid_events,
        input_rows=input_rows,
        processed_batches=processed_batches,
        theta_check=theta_check,
    )


def build_weighted_mask_exposure(
    cell_time_counts: np.ndarray,
    time_centers_mjd: np.ndarray,
    ha_centers: np.ndarray,
    dec_centers: np.ndarray,
    visible_mask: np.ndarray,
    sources: Sequence[SourceMask],
    longitude_east_deg: float,
    print_every: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cells, n_time = cell_time_counts.shape
    n_grid = visible_mask.size
    visible_flat = visible_mask.ravel()
    total_counts = cell_time_counts.sum(axis=1).astype(np.float64)
    weighted_exposure = total_counts[:, None] * visible_flat.astype(np.float64)[None, :]

    active_time = np.nonzero(cell_time_counts.sum(axis=0) > 0)[0]
    available_time_bins = np.zeros(n_grid, dtype=np.int32)
    available_time_bins[visible_flat] = int(active_time.size)
    lst_centers = local_sidereal_deg(time_centers_mjd, longitude_east_deg)

    for done, time_idx in enumerate(active_time, start=1):
        masked_idx = source_mask_indices_for_lst(
            float(lst_centers[time_idx]),
            sources,
            ha_centers,
            dec_centers,
            visible_mask,
        )
        if masked_idx.size:
            counts = cell_time_counts[:, time_idx].astype(np.float64)
            if np.any(counts > 0):
                weighted_exposure[:, masked_idx] -= counts[:, None]
            available_time_bins[masked_idx] -= 1
        if print_every > 0 and (done % print_every == 0 or done == active_time.size):
            print(f"[mask exposure {done}/{active_time.size}] active time bins processed", flush=True)

    weighted_exposure[:, ~visible_flat] = 0.0
    np.maximum(weighted_exposure, 0.0, out=weighted_exposure)
    np.maximum(available_time_bins, 0, out=available_time_bins)
    return weighted_exposure, available_time_bins


def normalize_acceptance(
    counts_unmasked: np.ndarray,
    counts_masked: np.ndarray,
    weighted_exposure: np.ndarray,
    visible_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells, n_dec, n_ha = counts_unmasked.shape
    visible = visible_mask[None, :, :]
    acceptance_unmasked = np.zeros((n_cells, n_dec, n_ha), dtype=np.float32)
    acceptance_masked = np.zeros((n_cells, n_dec, n_ha), dtype=np.float32)
    raw_masked = np.zeros((n_cells, n_dec, n_ha), dtype=np.float32)
    exposure_3d = weighted_exposure.reshape(n_cells, n_dec, n_ha)

    for idx in range(n_cells):
        unmasked = counts_unmasked[idx].astype(np.float64)
        unmasked[~visible_mask] = 0.0
        unmasked_sum = float(unmasked.sum())
        if unmasked_sum > 0:
            acceptance_unmasked[idx] = (unmasked / unmasked_sum).astype(np.float32)

        raw = np.zeros((n_dec, n_ha), dtype=np.float64)
        valid = (exposure_3d[idx] > 0) & visible_mask
        raw[valid] = counts_masked[idx][valid].astype(np.float64) / exposure_3d[idx][valid]
        raw[~np.isfinite(raw)] = 0.0
        raw[raw < 0.0] = 0.0
        raw_sum = float(raw.sum())
        raw_masked[idx] = raw.astype(np.float32)
        if raw_sum > 0:
            acceptance_masked[idx] = (raw / raw_sum).astype(np.float32)

    return acceptance_unmasked, acceptance_masked, raw_masked


def integrate_crab_background(
    acceptance: np.ndarray,
    cell_time_counts: np.ndarray,
    time_centers_mjd: np.ndarray,
    ha_centers: np.ndarray,
    dec_centers: np.ndarray,
    visible_mask: np.ndarray,
    r_opt_deg: np.ndarray,
    source_ra_deg: float,
    source_dec_deg: float,
    longitude_east_deg: float,
    latitude_deg: float,
    print_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cells, n_time = cell_time_counts.shape
    p_on_time = np.zeros((n_cells, n_time), dtype=np.float32)
    b_on_time = np.zeros((n_cells, n_time), dtype=np.float32)
    b_on = np.zeros(n_cells, dtype=np.float64)
    lst_centers = local_sidereal_deg(time_centers_mjd, longitude_east_deg)
    crab_ha_deg = ((lst_centers - float(source_ra_deg) + 180.0) % 360.0) - 180.0
    crab_theta_deg = zenith_deg_from_ha_dec(crab_ha_deg, np.full_like(crab_ha_deg, source_dec_deg), latitude_deg)
    acceptance_flat = acceptance.reshape(n_cells, -1).astype(np.float64)

    for cell_idx in range(n_cells):
        nonzero_time = np.nonzero(cell_time_counts[cell_idx] > 0)[0]
        for done, time_idx in enumerate(nonzero_time, start=1):
            disk_idx = disk_indices_for_center(
                float(crab_ha_deg[time_idx]),
                float(source_dec_deg),
                float(r_opt_deg[cell_idx]),
                ha_centers,
                dec_centers,
                visible_mask,
            )
            if disk_idx.size:
                p_on = float(acceptance_flat[cell_idx, disk_idx].sum())
            else:
                p_on = 0.0
            contribution = float(cell_time_counts[cell_idx, time_idx]) * p_on
            p_on_time[cell_idx, time_idx] = np.float32(p_on)
            b_on_time[cell_idx, time_idx] = np.float32(contribution)
            b_on[cell_idx] += contribution
        if print_every > 0 and ((cell_idx + 1) % print_every == 0 or cell_idx + 1 == n_cells):
            print(f"[background {cell_idx + 1}/{n_cells}] cells integrated", flush=True)
    return b_on, p_on_time, b_on_time, crab_ha_deg.astype(np.float32), crab_theta_deg.astype(np.float32)


def finite_rate(counts: np.ndarray, live_time_sec: np.ndarray) -> np.ndarray:
    rate = np.zeros(counts.shape, dtype=np.float32)
    valid = live_time_sec > 0
    if np.any(valid):
        rate[:, valid] = (counts[:, valid] / live_time_sec[valid][None, :]).astype(np.float32)
    return rate


def prepare_grid(cells: Sequence[CellSpec]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], CellSpec]]:
    nhit_bins = sorted({cell.nhit_bin for cell in cells}, key=interval_key)
    pred_bins = sorted({cell.predE_bin for cell in cells}, key=interval_key)
    return nhit_bins, pred_bins, {(cell.nhit_bin, cell.predE_bin): cell for cell in cells}


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def log_map(data: np.ndarray) -> np.ndarray:
    out = np.full(data.shape, np.nan, dtype=np.float32)
    positive = data > 0
    out[positive] = np.log10(data[positive])
    return out


def plot_acceptance_grid(
    acceptance: np.ndarray,
    cells: Sequence[CellSpec],
    ha_edges: np.ndarray,
    dec_edges: np.ndarray,
    output_path: Path,
    *,
    title: str,
    crab_ha_deg: Optional[np.ndarray] = None,
    crab_theta_deg: Optional[np.ndarray] = None,
    source_dec_deg: float = DEFAULT_SOURCE_DEC_DEG,
    theta_max_deg: float = DEFAULT_THETA_MAX_DEG,
) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.15 * len(pred_bins), 1.75 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(ha_edges[0]), float(ha_edges[-1]), float(dec_edges[0]), float(dec_edges[-1])]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    logged = log_map(acceptance)
    finite = logged[np.isfinite(logged)]
    vmin = float(np.percentile(finite, 2.0)) if finite.size else -8.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else -2.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = -8.0, -2.0
    first_im = None
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue
            im = ax.imshow(
                logged[cell.index],
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if first_im is None:
                first_im = im
            if crab_ha_deg is not None and crab_theta_deg is not None:
                visible = np.asarray(crab_theta_deg) < float(theta_max_deg)
                ax.scatter(crab_ha_deg[visible], np.full(np.count_nonzero(visible), source_dec_deg), s=0.4, c="white", alpha=0.35)
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit_bin}\nDec (deg)", fontsize=6.7)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("HA (deg)", fontsize=6.7)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_im is not None:
        cbar = fig.colorbar(first_im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        cbar.set_label("log10 relative acceptance", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rate_grid(
    rate_hz: np.ndarray,
    live_time_sec: np.ndarray,
    time_centers_mjd: np.ndarray,
    cells: Sequence[CellSpec],
    output_path: Path,
) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.15 * len(pred_bins), 1.5 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        squeeze=False,
    )
    valid_time = live_time_sec > 0
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue
            ax.plot(time_centers_mjd[valid_time], rate_hz[cell.index, valid_time], color="#1f4e79", linewidth=0.45)
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            ax.grid(alpha=0.22, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(f"{nhit_bin}\nHz", fontsize=6.7)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("MJD", fontsize=6.7)
    fig.suptitle("Stage D per-cell rate vs time", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.982])
    fig.savefig(output_path)
    plt.close(fig)


def plot_background_grid(rows: Sequence[Dict[str, object]], cells: Sequence[CellSpec], output_path: Path) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    values_by_cell = {int(row["cell_id"]): float(row["B_on"]) for row in rows}
    matrix = np.full((len(nhit_bins), len(pred_bins)), np.nan, dtype=np.float64)
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is not None:
                matrix[i, j] = values_by_cell[cell.cell_id]
    logged = np.full_like(matrix, np.nan)
    positive = matrix > 0
    logged[positive] = np.log10(matrix[positive])
    fig, ax = plt.subplots(figsize=(1.25 * len(pred_bins) + 2.6, 0.58 * len(nhit_bins) + 2.0), dpi=150)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(logged, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_xlabel("log10(E_pred / GeV) bin", fontsize=8)
    ax.set_ylabel("Nhit bin", fontsize=8)
    ax.set_title("Stage D Crab on-region background prediction", fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3g}", ha="center", va="center", color="white", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("log10 B_on", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_mask_exposure(
    available_time_bins: np.ndarray,
    active_time_bins: int,
    ha_edges: np.ndarray,
    dec_edges: np.ndarray,
    output_path: Path,
) -> None:
    plt = setup_matplotlib()
    n_dec = dec_edges.size - 1
    n_ha = ha_edges.size - 1
    if active_time_bins > 0:
        fraction = available_time_bins.reshape(n_dec, n_ha).astype(np.float64) / float(active_time_bins)
    else:
        fraction = np.zeros((n_dec, n_ha), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=150)
    im = ax.imshow(
        fraction,
        origin="lower",
        extent=[float(ha_edges[0]), float(ha_edges[-1]), float(dec_edges[0]), float(dec_edges[-1])],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("HA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Stage D source-mask exposure availability")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("available active time-bin fraction")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "selected_events",
        "grid_events",
        "out_of_grid_events",
        "source_masked_events",
        "source_masked_fraction",
        "live_time_days",
        "median_rate_hz",
        "mean_rate_hz",
        "r_opt_deg",
        "sigma_deg",
        "containment_r_opt",
        "B_on",
        "max_p_on",
        "r_opt_large_warning",
        "r_opt_extreme_warning",
        "background_form",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_source_masks_csv(path: Path, sources: Sequence[SourceMask]) -> None:
    fieldnames = ["name", "ra_deg", "dec_deg", "radius_deg", "enabled"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    "name": source.name,
                    "ra_deg": f"{source.ra_deg:.8g}",
                    "dec_deg": f"{source.dec_deg:.8g}",
                    "radius_deg": f"{source.radius_deg:.8g}",
                    "enabled": bool(source.enabled),
                }
            )


def write_summary_md(path: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage D Direct-Integration Background Summary\n\n")
        f.write(f"- Run id: `{metadata['run_id']}`\n")
        f.write(f"- Stage C input: `{metadata['inputs']['stage_c_dir']}`\n")
        f.write(f"- PSF input: `{metadata['inputs']['psf_npz']}`\n")
        f.write(f"- Background form: `{metadata['background_model']['background_form']}`\n")
        f.write(f"- Time bin: {metadata['time_binning']['time_bin_minutes']} min\n")
        f.write(f"- Active time bins: {metadata['time_binning']['active_time_bins']}\n")
        f.write(f"- Live time: {metadata['time_binning']['total_live_time_days']:.6g} days\n")
        f.write(f"- Theta sanity p95 absdiff: {metadata['theta_coordinate_check'].get('p95_absdiff_deg', 'n/a')} deg\n\n")
        f.write("| cell | Nhit bin | predE bin | events | masked frac | r_opt deg | B_on | median Hz | warnings |\n")
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            warnings = []
            if row["r_opt_large_warning"]:
                warnings.append("large_r_opt")
            if row["r_opt_extreme_warning"]:
                warnings.append("extreme_r_opt")
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{row['selected_events']} | {row['source_masked_fraction']:.4g} | "
                f"{row['r_opt_deg']:.5g} | {row['B_on']:.6g} | {row['median_rate_hz']:.5g} | "
                f"{', '.join(warnings) if warnings else '-'} |\n"
            )


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    stage_c_dir = Path(args.stage_c_dir).resolve()
    obs_events_dir = stage_c_dir / "obs_events"
    if not obs_events_dir.exists():
        raise FileNotFoundError(f"Stage C obs_events directory does not exist: {obs_events_dir}")
    stage_c_metadata_path = stage_c_dir / "obs_events_metadata.json"
    stage_c_metadata = load_json(stage_c_metadata_path) if stage_c_metadata_path.exists() else {}
    source_files_csv = stage_c_dir / "source_files.csv"

    psf_npz = Path(args.psf_npz).resolve()
    selection_csv = Path(args.cell_selection_csv).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(args.run_id or make_default_run_id())
    run_dir = prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))

    cells = load_cells(selection_csv)
    sources = default_source_masks()
    psf_by_cell = load_psf_by_cell(psf_npz, cells)
    r_opt_deg = np.asarray([psf_by_cell[cell.cell_id]["r_opt_deg"] for cell in cells], dtype=np.float32)
    sigma_deg = np.asarray([psf_by_cell[cell.cell_id]["sigma_deg"] for cell in cells], dtype=np.float32)
    containment_r_opt = np.asarray([psf_by_cell[cell.cell_id]["containment_r_opt"] for cell in cells], dtype=np.float32)

    ha_edges = make_edges(float(args.ha_min_deg), float(args.ha_max_deg), float(args.grid_step_deg))
    dec_edges = make_edges(float(args.dec_min_deg), float(args.dec_max_deg), float(args.grid_step_deg))
    ha_centers = 0.5 * (ha_edges[:-1] + ha_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    ha_grid, dec_grid = np.meshgrid(ha_centers, dec_centers)
    theta_grid_deg = zenith_deg_from_ha_dec(ha_grid, dec_grid, float(args.lhaaso_lat_deg))
    visible_mask = theta_grid_deg < float(args.theta_max_deg)

    time_edges_mjd = build_time_edges(stage_c_dir, stage_c_metadata, float(args.time_bin_minutes))
    time_centers_mjd = 0.5 * (time_edges_mjd[:-1] + time_edges_mjd[1:])
    live_time_sec = live_time_by_bin(source_files_csv, time_edges_mjd)

    print(f"Loaded {len(cells)} cells from {selection_csv}", flush=True)
    print(f"Stage C dataset: {obs_events_dir}", flush=True)
    print(f"Output run dir: {run_dir}", flush=True)
    print(
        f"Grid: HA bins={ha_edges.size - 1}, Dec bins={dec_edges.size - 1}, time bins={time_edges_mjd.size - 1}",
        flush=True,
    )

    scan = scan_stage_c_events(
        obs_events_dir,
        cells,
        time_edges_mjd,
        ha_edges,
        dec_edges,
        sources,
        longitude_east_deg=float(args.lhaaso_lon_deg),
        latitude_deg=float(args.lhaaso_lat_deg),
        batch_size=int(args.batch_size),
        max_batches=args.max_batches,
        print_every=int(args.print_every),
        theta_check_max_events=int(args.theta_check_max_events),
    )
    theta_p95 = finite_float(scan.theta_check.get("p95_absdiff_deg"))
    if theta_p95 is not None:
        if theta_p95 > float(args.theta_check_fail_p95_deg):
            raise RuntimeError(
                f"Theta coordinate sanity check failed: p95 absdiff={theta_p95:.4g} deg "
                f"> {args.theta_check_fail_p95_deg:.4g} deg"
            )
        if theta_p95 > float(args.theta_check_warn_p95_deg):
            scan.theta_check["status"] = "warning"
    print(f"Scanned rows: {scan.input_rows:,}", flush=True)

    n_cells = len(cells)
    n_dec = dec_edges.size - 1
    n_ha = ha_edges.size - 1
    counts_unmasked = scan.counts_unmasked_flat.reshape(n_cells, n_dec, n_ha)
    counts_masked = scan.counts_masked_flat.reshape(n_cells, n_dec, n_ha)

    weighted_exposure, available_time_bins = build_weighted_mask_exposure(
        scan.cell_time_counts,
        time_centers_mjd,
        ha_centers,
        dec_centers,
        visible_mask,
        sources,
        longitude_east_deg=float(args.lhaaso_lon_deg),
        print_every=max(1, int(args.print_every) * 20),
    )
    acceptance_unmasked, acceptance_masked, raw_masked = normalize_acceptance(
        counts_unmasked,
        counts_masked,
        weighted_exposure,
        visible_mask,
    )
    b_on, p_on_time, b_on_time, crab_ha_deg, crab_theta_deg = integrate_crab_background(
        acceptance_masked,
        scan.cell_time_counts,
        time_centers_mjd,
        ha_centers,
        dec_centers,
        visible_mask,
        r_opt_deg,
        float(args.source_ra_deg),
        float(args.source_dec_deg),
        float(args.lhaaso_lon_deg),
        float(args.lhaaso_lat_deg),
        print_every=3,
    )
    rate_hz = finite_rate(scan.cell_time_counts, live_time_sec)
    active_time_bins = int(np.count_nonzero(scan.cell_time_counts.sum(axis=0) > 0))

    rows: List[Dict[str, object]] = []
    total_live_time_days = float(live_time_sec.sum() / 86400.0)
    for cell in cells:
        idx = cell.index
        valid_rate = rate_hz[idx][live_time_sec > 0]
        selected_events = int(scan.cell_total_events[idx])
        source_masked_fraction = (
            float(scan.cell_source_masked_events[idx]) / float(selected_events) if selected_events > 0 else 0.0
        )
        rows.append(
            {
                "cell_index": int(idx),
                "cell_id": int(cell.cell_id),
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "selected_events": selected_events,
                "grid_events": int(scan.cell_grid_events[idx]),
                "out_of_grid_events": int(scan.cell_out_of_grid_events[idx]),
                "source_masked_events": int(scan.cell_source_masked_events[idx]),
                "source_masked_fraction": source_masked_fraction,
                "live_time_days": total_live_time_days,
                "median_rate_hz": float(np.median(valid_rate)) if valid_rate.size else 0.0,
                "mean_rate_hz": float(np.mean(valid_rate)) if valid_rate.size else 0.0,
                "r_opt_deg": float(r_opt_deg[idx]),
                "sigma_deg": float(sigma_deg[idx]),
                "containment_r_opt": float(containment_r_opt[idx]),
                "B_on": float(b_on[idx]),
                "max_p_on": float(np.max(p_on_time[idx])) if p_on_time.shape[1] else 0.0,
                "r_opt_large_warning": bool(r_opt_deg[idx] > 10.0),
                "r_opt_extreme_warning": bool(r_opt_deg[idx] > 20.0),
                "background_form": "direct_expectation",
            }
        )

    npz_path = run_dir / args.npz_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    metadata_path = run_dir / args.metadata_name
    source_masks_csv_path = run_dir / "source_masks_v1.csv"

    np.savez_compressed(
        npz_path,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        ha_edges_deg=ha_edges.astype(np.float32),
        dec_edges_deg=dec_edges.astype(np.float32),
        ha_centers_deg=ha_centers.astype(np.float32),
        dec_centers_deg=dec_centers.astype(np.float32),
        theta_grid_deg=theta_grid_deg.astype(np.float32),
        visible_mask=visible_mask.astype(bool),
        counts_unmasked=counts_unmasked.astype(np.int64),
        counts_masked=counts_masked.astype(np.int64),
        weighted_mask_exposure=weighted_exposure.reshape(n_cells, n_dec, n_ha).astype(np.float32),
        available_time_bins=available_time_bins.reshape(n_dec, n_ha).astype(np.int32),
        acceptance_unmasked=acceptance_unmasked,
        acceptance_masked=acceptance_masked,
        acceptance_masked_raw=raw_masked,
        time_edges_mjd=time_edges_mjd.astype(np.float64),
        time_centers_mjd=time_centers_mjd.astype(np.float64),
        live_time_sec=live_time_sec.astype(np.float32),
        cell_time_counts=scan.cell_time_counts.astype(np.int64),
        rate_hz=rate_hz.astype(np.float32),
        source_ra_deg=np.asarray([source.ra_deg for source in sources], dtype=np.float32),
        source_dec_deg=np.asarray([source.dec_deg for source in sources], dtype=np.float32),
        source_radius_deg=np.asarray([source.radius_deg for source in sources], dtype=np.float32),
        source_name=np.asarray([source.name for source in sources], dtype="U32"),
        crab_ha_deg=crab_ha_deg.astype(np.float32),
        crab_theta_deg=crab_theta_deg.astype(np.float32),
        r_opt_deg=r_opt_deg.astype(np.float32),
        sigma_deg=sigma_deg.astype(np.float32),
        containment_r_opt=containment_r_opt.astype(np.float32),
        p_on_time=p_on_time.astype(np.float32),
        b_on_time=b_on_time.astype(np.float32),
        B_on=b_on.astype(np.float64),
    )

    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = {
            "acceptance_unmasked_png": str(run_dir / "acceptance_unmasked_grid.png"),
            "acceptance_masked_png": str(run_dir / "acceptance_masked_grid.png"),
            "crab_track_acceptance_png": str(run_dir / "crab_track_acceptance_grid.png"),
            "rate_vs_time_png": str(run_dir / "rate_vs_time_grid.png"),
            "background_prediction_png": str(run_dir / "background_prediction_grid.png"),
            "mask_exposure_png": str(run_dir / "mask_exposure_coverage.png"),
        }
        plot_acceptance_grid(acceptance_unmasked, cells, ha_edges, dec_edges, Path(plot_outputs["acceptance_unmasked_png"]), title="Stage D unmasked acceptance")
        plot_acceptance_grid(acceptance_masked, cells, ha_edges, dec_edges, Path(plot_outputs["acceptance_masked_png"]), title="Stage D source-masked acceptance")
        plot_acceptance_grid(
            acceptance_masked,
            cells,
            ha_edges,
            dec_edges,
            Path(plot_outputs["crab_track_acceptance_png"]),
            title="Stage D source-masked acceptance with Crab track",
            crab_ha_deg=crab_ha_deg,
            crab_theta_deg=crab_theta_deg,
            source_dec_deg=float(args.source_dec_deg),
            theta_max_deg=float(args.theta_max_deg),
        )
        plot_rate_grid(rate_hz, live_time_sec, time_centers_mjd, cells, Path(plot_outputs["rate_vs_time_png"]))
        plot_background_grid(rows, cells, Path(plot_outputs["background_prediction_png"]))
        plot_mask_exposure(available_time_bins, active_time_bins, ha_edges, dec_edges, Path(plot_outputs["mask_exposure_png"]))

    metadata: Dict[str, object] = {
        "description": "Stage D direct-integration background for v1 (Nhit, predicted logE) Crab SED cells.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {
            "stage_c_dir": str(stage_c_dir),
            "obs_events_dir": str(obs_events_dir),
            "stage_c_metadata_json": str(stage_c_metadata_path),
            "source_files_csv": str(source_files_csv),
            "psf_npz": str(psf_npz),
            "cell_selection_csv": str(selection_csv),
        },
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "site": {
            "latitude_deg": float(args.lhaaso_lat_deg),
            "longitude_east_deg": float(args.lhaaso_lon_deg),
            "sidereal_time_formula": "IAU-style GMST polynomial from MJD plus east longitude; no astropy/IERS download.",
        },
        "grid": {
            "coordinate_system": "hour_angle_declination",
            "ha_edges_deg": [float(ha_edges[0]), float(ha_edges[-1])],
            "dec_edges_deg": [float(dec_edges[0]), float(dec_edges[-1])],
            "grid_step_deg": float(args.grid_step_deg),
            "shape": [int(n_dec), int(n_ha)],
            "theta_visible_max_deg": float(args.theta_max_deg),
            "visible_grid_pixels": int(np.count_nonzero(visible_mask)),
        },
        "time_binning": {
            "time_bin_minutes": float(args.time_bin_minutes),
            "time_bins": int(time_edges_mjd.size - 1),
            "active_time_bins": active_time_bins,
            "mjd_min": float(time_edges_mjd[0]),
            "mjd_max": float(time_edges_mjd[-1]),
            "live_time_estimator": "Stage C source_files matched MJD span with rough live time fractionally assigned to overlapping windows.",
            "total_live_time_seconds": float(live_time_sec.sum()),
            "total_live_time_days": total_live_time_days,
        },
        "source_masks": [
            {
                "name": source.name,
                "ra_deg": float(source.ra_deg),
                "dec_deg": float(source.dec_deg),
                "radius_deg": float(source.radius_deg),
                "enabled": bool(source.enabled),
            }
            for source in sources
        ],
        "background_model": {
            "method": "direct_integration_acceptance_times_rate",
            "background_form": "direct_expectation",
            "B_on_formula": "sum_time N_cell,time * sum_grid acceptance_cell,grid * I(grid within Crab aperture at time)",
            "alpha_b": None,
            "N_off_b": None,
            "alpha_N_off_note": "Stage D v1 outputs direct background expectation B_on,b; traditional alpha/N_off is not defined.",
            "on_region_radius_source": "Stage B psf_v1.npz r_opt_deg",
            "mask_exposure_correction": "counts_masked / sum_time(N_cell,time * source_mask_available_time,grid), normalized over visible grid",
        },
        "theta_coordinate_check": scan.theta_check,
        "processing": {
            "input_rows_scanned": int(scan.input_rows),
            "processed_batches": int(scan.processed_batches),
            "batch_size": int(args.batch_size),
            "max_batches": args.max_batches,
            "workers_requested": int(args.workers),
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "cells": rows,
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": {
            "npz": str(npz_path),
            "metadata_json": str(metadata_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "source_masks_csv": str(source_masks_csv_path),
            **plot_outputs,
        },
    }

    write_summary_csv(summary_csv_path, rows)
    write_source_masks_csv(source_masks_csv_path, sources)
    write_summary_md(summary_md_path, metadata, rows)
    write_json(metadata_path, metadata)

    if not args.no_promote_current:
        promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        write_json(metadata_path, metadata)
    else:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]
        write_json(metadata_path, metadata)

    print(f"Wrote {npz_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)
    print(f"Wrote {summary_md_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    if not args.no_promote_current:
        print(f"Promoted current Stage D output to {output_root / 'current'}", flush=True)
    if theta_p95 is not None and theta_p95 > float(args.theta_check_warn_p95_deg):
        print(f"Theta coordinate warning: p95 absdiff={theta_p95:.4g} deg", flush=True)


if __name__ == "__main__":
    main()
