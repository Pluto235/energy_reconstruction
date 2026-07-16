#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apply.stages.poisson_roi_background import (
    PoissonSurfaceFitError,
    fit_profiled_poisson_surface,
    quadratic_rectangle_basis_integrals,
)


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


@dataclass
class RoiScanResult:
    counts_flat: np.ndarray
    cell_total_events: np.ndarray
    cell_map_events: np.ndarray
    cell_out_of_map_events: np.ndarray
    cell_fiducial_events: np.ndarray
    cell_edge_diagnostic_events: np.ndarray
    rho_hist_total: np.ndarray
    rho_hist_by_cell: np.ndarray
    continuous_annulus_counts: np.ndarray
    input_rows: int
    processed_batches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage D background for Crab SED analysis cells.")
    parser.add_argument("--stage-c-dir", type=str, default=DEFAULT_STAGE_C_DIR)
    parser.add_argument("--psf-npz", type=str, default=DEFAULT_PSF_NPZ)
    parser.add_argument("--cell-selection-csv", type=str, default=DEFAULT_CELL_SELECTION)
    parser.add_argument(
        "--included-only",
        action="store_true",
        default=False,
        help="Load only rows whose selector include column is 1.",
    )
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
    parser.add_argument("--mjd-min", type=float, default=None, help="Optional inclusive MJD lower bound for validation splits.")
    parser.add_argument("--mjd-max", type=float, default=None, help="Optional exclusive MJD upper bound for validation splits.")

    parser.add_argument(
        "--background-mode",
        choices=["auto", "full_field_direct_integration", "crab_roi_local"],
        default="auto",
        help="Select the Stage D background model. 'auto' uses Stage C ROI coverage metadata.",
    )
    parser.add_argument(
        "--roi-background-method",
        choices=["dec-sideband", "annulus-quadratic"],
        default="dec-sideband",
        help="Background estimator for --background-mode crab_roi_local.",
    )
    parser.add_argument("--roi-fiducial-deg", type=float, default=6.0)
    parser.add_argument("--roi-edge-diagnostic-deg", type=float, default=8.0)
    parser.add_argument("--roi-grid-step-deg", type=float, default=0.1)
    parser.add_argument(
        "--roi-fit-statistic",
        choices=["weighted-ls", "poisson"],
        default="weighted-ls",
        help="Surface-fit statistic for annulus-quadratic ROI backgrounds.",
    )
    parser.add_argument(
        "--pooling-manifest",
        type=str,
        default=None,
        help="Frozen pooling manifest required by --roi-fit-statistic poisson.",
    )
    parser.add_argument(
        "--roi-grid-offset-x-fraction",
        type=float,
        choices=[0.0, 0.5],
        default=0.0,
    )
    parser.add_argument(
        "--roi-grid-offset-y-fraction",
        type=float,
        choices=[0.0, 0.5],
        default=0.0,
    )
    parser.add_argument(
        "--on-aperture-integration",
        choices=["pixel-center", "analytic-quadratic"],
        default="pixel-center",
        help=(
            "How to integrate the fitted ROI background over the circular on aperture. "
            "The analytic mode is available only for annulus-quadratic backgrounds."
        ),
    )
    parser.add_argument(
        "--analytic-required-cell-ids",
        type=str,
        default="",
        help=(
            "Comma-separated cells that must pass analytic surface positivity. "
            "When omitted, analytic mode requires every loaded cell."
        ),
    )
    parser.add_argument("--roi-edge-margin-deg", type=float, default=0.25)
    parser.add_argument("--roi-source-mask-deg", type=float, default=2.0)
    parser.add_argument("--roi-source-mask-r-opt-factor", type=float, default=2.0)
    parser.add_argument("--roi-coverage-max-deg", type=float, default=12.0)
    parser.add_argument("--roi-coverage-bin-deg", type=float, default=0.1)
    parser.add_argument("--annulus-default-inner-deg", type=float, default=1.5)
    parser.add_argument("--annulus-width-deg", type=float, default=2.0)
    parser.add_argument("--annulus-source-mask-min-deg", type=float, default=1.5)
    parser.add_argument("--annulus-source-mask-r-opt-factor", type=float, default=2.0)
    parser.add_argument("--annulus-source-mask-margin-deg", type=float, default=0.2)
    parser.add_argument("--annulus-max-inner-deg", type=float, default=3.0)
    parser.add_argument("--roi-surface-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--surface-condition-max", type=float, default=1.0e8)
    parser.add_argument("--surface-min-training-pixels", type=int, default=80)
    parser.add_argument(
        "--annulus-normalize-surface",
        action="store_true",
        default=False,
        help=(
            "For annulus-quadratic ROI backgrounds, keep the weighted least-squares surface shape "
            "but rescale each cell so the model integral over the training annulus equals the "
            "observed annulus counts."
        ),
    )

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


def load_cells(selection_csv: Path, *, included_only: bool = False) -> List[CellSpec]:
    cells: List[CellSpec] = []
    with selection_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"cell_id", "nhit_bin", "predE_bin"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{selection_csv} is missing required columns: {sorted(missing)}")
        if included_only and "include" not in set(reader.fieldnames or []):
            raise ValueError(f"{selection_csv} is missing required include column for --included-only")
        for idx, row in enumerate(reader):
            if included_only and str(row.get("include", "")).strip() != "1":
                continue
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


def make_offset_roi_edges(radius_deg: float, step_deg: float, offset_fraction: float) -> np.ndarray:
    """Build a phased grid with exterior padding that fully covers the physical disk."""
    radius = float(radius_deg)
    step = float(step_deg)
    fraction = float(offset_fraction)
    if radius <= 0.0 or step <= 0.0:
        raise ValueError("ROI radius and grid step must be positive")
    if fraction not in {0.0, 0.5}:
        raise ValueError("ROI grid offset fraction must be 0 or 0.5")
    start = -radius + fraction * step
    if start > -radius:
        start -= step
    n_steps = int(math.ceil((radius - start) / step))
    edges = start + step * np.arange(n_steps + 1, dtype=np.float64)
    if edges[-1] < radius - 1.0e-12:
        edges = np.append(edges, edges[-1] + step)
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


def resolve_background_mode(args: argparse.Namespace, stage_c_metadata: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    requested = str(args.background_mode)
    roi = stage_c_metadata.get("roi_coverage") if isinstance(stage_c_metadata, dict) else None
    roi_meta = roi if isinstance(roi, dict) else {}
    if requested != "auto":
        return requested, {
            "requested": requested,
            "resolved": requested,
            "reason": "explicit --background-mode",
            "stage_c_roi_coverage": roi_meta,
        }

    if not roi_meta:
        raise ValueError(
            "--background-mode auto requires Stage C roi_coverage metadata. "
            "Use --background-mode full_field_direct_integration only if the Stage C input is confirmed full-field."
        )

    edge = finite_float(roi_meta.get("edge_radius_estimate_deg"))
    fiducial = finite_float(roi_meta.get("fiducial_radius_recommendation_deg"))
    fractions = roi_meta.get("counts_within_radius_fraction_of_total")
    rho10_fraction = None
    if isinstance(fractions, dict):
        rho10_fraction = finite_float(fractions.get("rho_lt_10_deg"))

    local_reasons: List[str] = []
    if edge is not None and edge <= max(float(args.roi_edge_diagnostic_deg) + 1.0, float(args.roi_fiducial_deg) + 1.0):
        local_reasons.append(f"Stage C edge_radius_estimate_deg={edge:.4g}")
    if rho10_fraction is not None and rho10_fraction >= 0.10:
        local_reasons.append(f"rho_lt_10 fraction is high ({rho10_fraction:.4g})")
    if fiducial is not None and fiducial <= float(args.roi_edge_diagnostic_deg):
        local_reasons.append(f"Stage C fiducial recommendation={fiducial:.4g} deg")

    if local_reasons:
        return "crab_roi_local", {
            "requested": "auto",
            "resolved": "crab_roi_local",
            "reason": "; ".join(local_reasons),
            "stage_c_roi_coverage": roi_meta,
        }

    raise ValueError(
        "Stage C roi_coverage metadata is present but does not clearly indicate a Crab-local ROI. "
        "Choose --background-mode crab_roi_local or --background-mode full_field_direct_integration explicitly."
    )


def crab_tangent_xy(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    source_ra_deg: float,
    source_dec_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dra = ((np.asarray(ra_deg, dtype=np.float64) - float(source_ra_deg) + 180.0) % 360.0) - 180.0
    x = dra * math.cos(math.radians(float(source_dec_deg)))
    y = np.asarray(dec_deg, dtype=np.float64) - float(source_dec_deg)
    rho = np.hypot(x, y)
    return x, y, rho


def make_cell_index_by_id(cells: Sequence[CellSpec]) -> np.ndarray:
    max_cell_id = max(cell.cell_id for cell in cells)
    cell_index_by_id = np.full(max_cell_id + 1, -1, dtype=np.int16)
    for cell in cells:
        cell_index_by_id[cell.cell_id] = np.int16(cell.index)
    return cell_index_by_id


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


def apply_mjd_window(mjd: np.ndarray, *, mjd_min: Optional[float], mjd_max: Optional[float]) -> np.ndarray:
    mask = np.isfinite(mjd)
    if mjd_min is not None:
        mask &= mjd >= float(mjd_min)
    if mjd_max is not None:
        mask &= mjd < float(mjd_max)
    return mask


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
    mjd_min: Optional[float] = None,
    mjd_max: Optional[float] = None,
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

        finite = valid_cell & apply_mjd_window(mjd, mjd_min=mjd_min, mjd_max=mjd_max) & np.isfinite(ra) & np.isfinite(dec)
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


def scan_stage_c_roi_events(
    obs_events_dir: Path,
    cells: Sequence[CellSpec],
    *,
    source_ra_deg: float,
    source_dec_deg: float,
    x_edges_deg: np.ndarray,
    y_edges_deg: np.ndarray,
    rho_hist_edges_deg: np.ndarray,
    roi_fiducial_deg: float,
    roi_edge_diagnostic_deg: float,
    annulus_inner_deg: Optional[np.ndarray] = None,
    annulus_outer_deg: Optional[np.ndarray] = None,
    batch_size: int,
    max_batches: Optional[int],
    print_every: int,
    mjd_min: Optional[float] = None,
    mjd_max: Optional[float] = None,
) -> RoiScanResult:
    dataset = ds.dataset(obs_events_dir, format="parquet", partitioning="hive")
    columns = ["mjd", "ra_mean_deg", "dec_mean_deg", "cell_id"]
    scanner = dataset.scanner(columns=columns, batch_size=int(batch_size), use_threads=True)
    n_cells = len(cells)
    n_x = x_edges_deg.size - 1
    n_y = y_edges_deg.size - 1
    n_map = n_x * n_y
    flat_size = n_cells * n_map
    cell_index_by_id = make_cell_index_by_id(cells)

    counts_flat = np.zeros(flat_size, dtype=np.int64)
    cell_total_events = np.zeros(n_cells, dtype=np.int64)
    cell_map_events = np.zeros(n_cells, dtype=np.int64)
    cell_out_of_map_events = np.zeros(n_cells, dtype=np.int64)
    cell_fiducial_events = np.zeros(n_cells, dtype=np.int64)
    cell_edge_diagnostic_events = np.zeros(n_cells, dtype=np.int64)
    rho_hist_total = np.zeros(rho_hist_edges_deg.size - 1, dtype=np.int64)
    rho_hist_by_cell = np.zeros((n_cells, rho_hist_edges_deg.size - 1), dtype=np.int64)
    continuous_annulus_counts = np.zeros(n_cells, dtype=np.int64)

    if (annulus_inner_deg is None) != (annulus_outer_deg is None):
        raise ValueError("Both annulus_inner_deg and annulus_outer_deg are required together")
    if annulus_inner_deg is not None:
        annulus_inner_deg = np.asarray(annulus_inner_deg, dtype=np.float64)
        annulus_outer_deg = np.asarray(annulus_outer_deg, dtype=np.float64)
        if annulus_inner_deg.shape != (n_cells,) or annulus_outer_deg.shape != (n_cells,):
            raise ValueError("Continuous annulus boundaries must match the loaded cells")

    x_min = float(x_edges_deg[0])
    y_min = float(y_edges_deg[0])
    x_step = float(x_edges_deg[1] - x_edges_deg[0])
    y_step = float(y_edges_deg[1] - y_edges_deg[0])
    input_rows = 0
    processed_batches = 0

    for batch_idx, batch in enumerate(scanner.to_batches(), start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        processed_batches += 1
        input_rows += int(batch.num_rows)

        mjd = np.asarray(column_to_numpy(batch, "mjd"), dtype=np.float64)
        ra = np.asarray(column_to_numpy(batch, "ra_mean_deg"), dtype=np.float64)
        dec = np.asarray(column_to_numpy(batch, "dec_mean_deg"), dtype=np.float64)
        cell_id = np.asarray(column_to_numpy(batch, "cell_id"), dtype=np.int32)

        valid_id = (cell_id >= 0) & (cell_id < cell_index_by_id.size)
        cell_idx = np.full(cell_id.shape, -1, dtype=np.int16)
        cell_idx[valid_id] = cell_index_by_id[cell_id[valid_id]]
        finite = (cell_idx >= 0) & apply_mjd_window(mjd, mjd_min=mjd_min, mjd_max=mjd_max) & np.isfinite(ra) & np.isfinite(dec)
        if not np.any(finite):
            continue

        ra_v = ra[finite]
        dec_v = dec[finite]
        cell_v = np.asarray(cell_idx[finite], dtype=np.int64)
        x, y, rho = crab_tangent_xy(ra_v, dec_v, source_ra_deg, source_dec_deg)

        update_bincount(cell_total_events, cell_v, n_cells)
        rho_hist_total += np.histogram(rho[np.isfinite(rho)], bins=rho_hist_edges_deg)[0]
        rho_bin = np.searchsorted(rho_hist_edges_deg, rho, side="right") - 1
        valid_rho_bin = np.isfinite(rho) & (rho_bin >= 0) & (rho_bin < rho_hist_edges_deg.size - 1)
        if np.any(valid_rho_bin):
            linear_rho = cell_v[valid_rho_bin] * (rho_hist_edges_deg.size - 1) + rho_bin[valid_rho_bin]
            rho_hist_by_cell += np.bincount(
                linear_rho,
                minlength=n_cells * (rho_hist_edges_deg.size - 1),
            ).reshape(n_cells, rho_hist_edges_deg.size - 1)

        fiducial = rho < float(roi_fiducial_deg)
        if np.any(fiducial):
            update_bincount(cell_fiducial_events, cell_v[fiducial], n_cells)
        edge_diag = (rho >= float(roi_fiducial_deg)) & (rho < float(roi_edge_diagnostic_deg))
        if np.any(edge_diag):
            update_bincount(cell_edge_diagnostic_events, cell_v[edge_diag], n_cells)

        if annulus_inner_deg is not None and annulus_outer_deg is not None:
            continuous_annulus = (
                (rho >= annulus_inner_deg[cell_v])
                & (rho < annulus_outer_deg[cell_v])
                & (rho < float(roi_fiducial_deg))
            )
            if np.any(continuous_annulus):
                update_bincount(continuous_annulus_counts, cell_v[continuous_annulus], n_cells)

        x_idx = np.floor((x - x_min) / x_step).astype(np.int64)
        y_idx = np.floor((y - y_min) / y_step).astype(np.int64)
        valid_map = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y)
        if np.any(valid_map):
            map_idx = y_idx[valid_map] * n_x + x_idx[valid_map]
            linear_map = cell_v[valid_map] * n_map + map_idx
            update_bincount(counts_flat, linear_map, flat_size)
            update_bincount(cell_map_events, cell_v[valid_map], n_cells)
        out_of_map = ~valid_map
        if np.any(out_of_map):
            update_bincount(cell_out_of_map_events, cell_v[out_of_map], n_cells)

        if print_every > 0 and (batch_idx % print_every == 0):
            print(
                f"[roi batch {batch_idx}] rows={input_rows:,} selected={int(cell_total_events.sum()):,}",
                flush=True,
            )

    return RoiScanResult(
        counts_flat=counts_flat,
        cell_total_events=cell_total_events,
        cell_map_events=cell_map_events,
        cell_out_of_map_events=cell_out_of_map_events,
        cell_fiducial_events=cell_fiducial_events,
        cell_edge_diagnostic_events=cell_edge_diagnostic_events,
        rho_hist_total=rho_hist_total,
        rho_hist_by_cell=rho_hist_by_cell,
        continuous_annulus_counts=continuous_annulus_counts,
        input_rows=input_rows,
        processed_batches=processed_batches,
    )


def build_roi_masks(
    x_centers: np.ndarray,
    r_opt_deg: np.ndarray,
    *,
    y_centers: Optional[np.ndarray] = None,
    roi_fiducial_deg: float,
    roi_edge_margin_deg: float,
    roi_source_mask_deg: float,
    source_mask_r_opt_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if y_centers is None:
        y_centers = x_centers
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    rho_grid = np.hypot(x_grid, y_grid)
    fiducial_mask = rho_grid < float(roi_fiducial_deg)
    training_edge_limit = max(0.0, float(roi_fiducial_deg) - float(roi_edge_margin_deg))
    edge_safe_mask = rho_grid < training_edge_limit
    on_masks = np.zeros((r_opt_deg.size, y_centers.size, x_centers.size), dtype=bool)
    source_masks = np.zeros_like(on_masks)
    for idx, r_opt in enumerate(r_opt_deg):
        on_masks[idx] = rho_grid <= float(r_opt)
        source_radius = max(float(roi_source_mask_deg), float(source_mask_r_opt_factor) * float(r_opt))
        source_masks[idx] = rho_grid <= source_radius
    return rho_grid.astype(np.float32), fiducial_mask, edge_safe_mask, on_masks, source_masks


def estimate_roi_dec_sideband_background(
    counts_map: np.ndarray,
    fiducial_mask: np.ndarray,
    edge_safe_mask: np.ndarray,
    on_masks: np.ndarray,
    source_masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cells, n_y, n_x = counts_map.shape
    background_map = np.zeros((n_cells, n_y, n_x), dtype=np.float32)
    training_mask = np.zeros((n_cells, n_y, n_x), dtype=bool)
    off_counts = np.zeros(n_cells, dtype=np.float64)
    off_pixels = np.zeros(n_cells, dtype=np.int64)
    on_pixels = np.zeros(n_cells, dtype=np.int64)
    b_on = np.zeros(n_cells, dtype=np.float64)

    base_training = fiducial_mask & edge_safe_mask
    for cell_idx in range(n_cells):
        on_mask = on_masks[cell_idx] & fiducial_mask
        train = base_training & (~source_masks[cell_idx])
        training_mask[cell_idx] = train
        on_pixels[cell_idx] = int(np.count_nonzero(on_mask))
        off_pixels[cell_idx] = int(np.count_nonzero(train))
        off_counts[cell_idx] = float(counts_map[cell_idx][train].sum()) if off_pixels[cell_idx] else 0.0

        for y_idx in range(n_y):
            row_train = train[y_idx]
            row_on = on_mask[y_idx]
            if not np.any(row_on):
                continue
            row_train_pixels = int(np.count_nonzero(row_train))
            if row_train_pixels <= 0:
                continue
            row_density = float(counts_map[cell_idx, y_idx, row_train].sum()) / float(row_train_pixels)
            background_map[cell_idx, y_idx, fiducial_mask[y_idx]] = np.float32(row_density)
            b_on[cell_idx] += row_density * float(np.count_nonzero(row_on))

    return b_on, background_map, training_mask, off_counts, off_pixels, on_pixels


def surface_design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    if int(order) == 1:
        return np.column_stack([np.ones_like(x), x, y]).astype(np.float64)
    if int(order) == 2:
        return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y]).astype(np.float64)
    raise ValueError(f"Unsupported surface order: {order}")


def polynomial_coefficients6(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if values.size < 1 or values.size > 6 or not np.all(np.isfinite(values)):
        raise ValueError("Background polynomial must contain 1 to 6 finite coefficients")
    out = np.zeros(6, dtype=np.float64)
    out[: values.size] = values
    return out


def integrate_centered_disk_polynomial(
    coefficients: np.ndarray,
    radius_deg: float,
    grid_step_deg: float,
    scale: float = 1.0,
) -> float:
    """Integrate a per-grid-cell polynomial density over a centered disk."""
    radius = float(radius_deg)
    step = float(grid_step_deg)
    normalization = float(scale)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Disk radius must be positive and finite")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Grid step must be positive and finite")
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Surface scale must be positive and finite")
    c = polynomial_coefficients6(coefficients)
    disk_integral = math.pi * radius * radius * c[0]
    disk_integral += 0.25 * math.pi * radius**4 * (c[3] + c[5])
    return float(normalization * disk_integral / (step * step))


def evaluate_quadratic(coefficients: np.ndarray, x: float, y: float) -> float:
    c = polynomial_coefficients6(coefficients)
    return float(c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y)


def minimum_quadratic_on_centered_disk(coefficients: np.ndarray, radius_deg: float) -> float:
    """Return the deterministic global minimum of a quadratic on a centered disk."""
    c = polynomial_coefficients6(coefficients)
    radius = float(radius_deg)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Disk radius must be positive and finite")

    candidates = [evaluate_quadratic(c, 0.0, 0.0)]
    hessian = np.asarray([[2.0 * c[3], c[4]], [c[4], 2.0 * c[5]]], dtype=np.float64)
    gradient = np.asarray([c[1], c[2]], dtype=np.float64)
    stationary, _, _, _ = np.linalg.lstsq(hessian, -gradient, rcond=None)
    if np.linalg.norm(hessian @ stationary + gradient) <= 1.0e-10 * max(1.0, np.linalg.norm(gradient)):
        if float(np.hypot(stationary[0], stationary[1])) <= radius * (1.0 + 1.0e-12):
            candidates.append(evaluate_quadratic(c, float(stationary[0]), float(stationary[1])))

    r = radius
    derivative = np.asarray(
        [
            -r * c[2] + r * r * c[4],
            -2.0 * r * c[1] - 4.0 * r * r * (c[5] - c[3]),
            -6.0 * r * r * c[4],
            -2.0 * r * c[1] + 4.0 * r * r * (c[5] - c[3]),
            r * c[2] + r * r * c[4],
        ],
        dtype=np.float64,
    )
    nonzero = np.flatnonzero(np.abs(derivative) > 1.0e-14)
    if nonzero.size:
        roots = np.roots(derivative[nonzero[0] :])
        for root in roots:
            if abs(float(root.imag)) <= 1.0e-9 * max(1.0, abs(float(root.real))):
                theta = 2.0 * math.atan(float(root.real))
                candidates.append(evaluate_quadratic(c, r * math.cos(theta), r * math.sin(theta)))
    candidates.extend([evaluate_quadratic(c, r, 0.0), evaluate_quadratic(c, -r, 0.0)])
    return float(min(candidates))


def select_on_aperture_background(
    *,
    pixel_center_b_on: float,
    analytic_b_on: float,
    analytic_surface_min: float,
    method: str,
    cell_id: int,
    analytic_required: bool = True,
) -> float:
    if method == "pixel-center":
        return float(pixel_center_b_on)
    if method != "analytic-quadratic":
        raise ValueError(f"Unsupported on-aperture integration method: {method}")
    analytic_valid = (
        math.isfinite(analytic_surface_min)
        and analytic_surface_min > 0.0
        and math.isfinite(analytic_b_on)
        and analytic_b_on > 0.0
    )
    if not analytic_valid and not analytic_required:
        return float(pixel_center_b_on)
    if not math.isfinite(analytic_surface_min) or analytic_surface_min <= 0.0:
        raise ValueError(
            f"Cell {int(cell_id)} has a non-positive fitted background inside the analytic on aperture: "
            f"minimum={analytic_surface_min:.12g}"
        )
    if not math.isfinite(analytic_b_on) or analytic_b_on <= 0.0:
        raise ValueError(f"Cell {int(cell_id)} has invalid analytic B_on={analytic_b_on}")
    return float(analytic_b_on)


def padded_coefficients(coeff: np.ndarray, order: int) -> np.ndarray:
    out = np.full(6, np.nan, dtype=np.float64)
    coeff = np.asarray(coeff, dtype=np.float64)
    out[: coeff.size] = coeff
    return out


def padded_covariance(cov: np.ndarray, order: int) -> np.ndarray:
    out = np.full((6, 6), np.nan, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    out[: cov.shape[0], : cov.shape[1]] = cov
    return out


def annulus_placement(
    r_opt_deg: np.ndarray,
    *,
    default_inner_deg: float,
    width_deg: float,
    source_mask_min_deg: float,
    source_mask_r_opt_factor: float,
    source_mask_margin_deg: float,
    max_inner_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_opt = np.asarray(r_opt_deg, dtype=np.float64)
    source_mask = np.maximum(float(source_mask_min_deg), float(source_mask_r_opt_factor) * r_opt)
    shifted_inner = source_mask + float(source_mask_margin_deg)
    inner = np.full(r_opt.shape, float(default_inner_deg), dtype=np.float64)
    needs_shift = shifted_inner > float(default_inner_deg)
    if np.any(needs_shift):
        inner[needs_shift] = np.minimum(np.ceil(shifted_inner[needs_shift] * 2.0) / 2.0, float(max_inner_deg))
    outer = inner + float(width_deg)
    return source_mask, inner, outer, needs_shift


def estimate_roi_annulus_surface_background(
    counts_map: np.ndarray,
    xy_centers: np.ndarray,
    r_opt_deg: np.ndarray,
    *,
    cell_ids: Optional[np.ndarray] = None,
    analytic_required_mask: Optional[np.ndarray] = None,
    roi_fiducial_deg: float,
    annulus_default_inner_deg: float,
    annulus_width_deg: float,
    annulus_source_mask_min_deg: float,
    annulus_source_mask_r_opt_factor: float,
    annulus_source_mask_margin_deg: float,
    annulus_max_inner_deg: float,
    surface_order: int,
    condition_max: float,
    min_training_pixels: int,
    annulus_normalize_surface: bool = False,
    on_aperture_integration: str = "pixel-center",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
]:
    n_cells, n_y, n_x = counts_map.shape
    x_grid, y_grid = np.meshgrid(xy_centers.astype(np.float64), xy_centers.astype(np.float64))
    rho_grid = np.hypot(x_grid, y_grid)
    fiducial_mask = rho_grid < float(roi_fiducial_deg)
    on_masks = np.zeros((n_cells, n_y, n_x), dtype=bool)
    source_masks = np.zeros_like(on_masks)
    training_mask = np.zeros_like(on_masks)
    background_map = np.full((n_cells, n_y, n_x), np.nan, dtype=np.float32)
    residual_map = np.full((n_cells, n_y, n_x), np.nan, dtype=np.float32)
    core_background_map = np.full((n_cells, n_y, n_x), np.nan, dtype=np.float32)
    b_on = np.zeros(n_cells, dtype=np.float64)
    b_on_pixel_center = np.full(n_cells, np.nan, dtype=np.float64)
    b_on_analytic = np.full(n_cells, np.nan, dtype=np.float64)
    on_area_deg2 = math.pi * np.asarray(r_opt_deg, dtype=np.float64) ** 2
    on_effective_pixels = np.full(n_cells, np.nan, dtype=np.float64)
    on_aperture_surface_min = np.full(n_cells, np.nan, dtype=np.float64)
    on_aperture_integration_by_cell = np.full(n_cells, "pixel-center", dtype="U32")
    on_pixels = np.zeros(n_cells, dtype=np.int64)
    if cell_ids is None:
        cell_ids = np.arange(1, n_cells + 1, dtype=np.int64)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    if cell_ids.shape != (n_cells,):
        raise ValueError(f"cell_ids shape {cell_ids.shape} does not match {n_cells} background cells")
    if analytic_required_mask is None:
        analytic_required_mask = np.ones(n_cells, dtype=bool)
    analytic_required_mask = np.asarray(analytic_required_mask, dtype=bool)
    if analytic_required_mask.shape != (n_cells,):
        raise ValueError(
            f"analytic_required_mask shape {analytic_required_mask.shape} does not match {n_cells} background cells"
        )
    center_steps = np.diff(np.asarray(xy_centers, dtype=np.float64))
    if center_steps.size == 0 or not np.allclose(center_steps, center_steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Analytic on-aperture integration requires a uniform ROI grid")
    grid_step_deg = float(center_steps[0])
    if grid_step_deg <= 0.0:
        raise ValueError("ROI grid step must be positive")
    on_effective_pixels[:] = on_area_deg2 / (grid_step_deg * grid_step_deg)
    source_mask_radius, annulus_inner, annulus_outer, shifted = annulus_placement(
        r_opt_deg,
        default_inner_deg=float(annulus_default_inner_deg),
        width_deg=float(annulus_width_deg),
        source_mask_min_deg=float(annulus_source_mask_min_deg),
        source_mask_r_opt_factor=float(annulus_source_mask_r_opt_factor),
        source_mask_margin_deg=float(annulus_source_mask_margin_deg),
        max_inner_deg=float(annulus_max_inner_deg),
    )
    coeffs = np.full((n_cells, 6), np.nan, dtype=np.float64)
    covariances = np.full((n_cells, 6, 6), np.nan, dtype=np.float64)
    chi2 = np.full(n_cells, np.nan, dtype=np.float64)
    ndof = np.zeros(n_cells, dtype=np.int64)
    condition = np.full(n_cells, np.inf, dtype=np.float64)
    annulus_counts = np.zeros(n_cells, dtype=np.float64)
    annulus_pixels = np.zeros(n_cells, dtype=np.int64)
    residual_mean = np.full(n_cells, np.nan, dtype=np.float64)
    residual_rms = np.full(n_cells, np.nan, dtype=np.float64)
    surface_scale = np.ones(n_cells, dtype=np.float64)
    annulus_model_counts_raw = np.full(n_cells, np.nan, dtype=np.float64)
    annulus_model_counts_final = np.full(n_cells, np.nan, dtype=np.float64)
    annulus_count_residual_raw = np.full(n_cells, np.nan, dtype=np.float64)
    annulus_count_residual_final = np.full(n_cells, np.nan, dtype=np.float64)
    rank = np.zeros(n_cells, dtype=np.int64)
    negative_pixels = np.zeros(n_cells, dtype=np.int64)
    core_warning = np.zeros(n_cells, dtype=bool)
    fit_success = np.zeros(n_cells, dtype=bool)
    off_counts = np.zeros(n_cells, dtype=np.float64)
    off_pixels = np.zeros(n_cells, dtype=np.int64)

    full_design = surface_design_matrix(x_grid.ravel(), y_grid.ravel(), int(surface_order))
    for cell_idx in range(n_cells):
        on_mask = (rho_grid <= float(r_opt_deg[cell_idx])) & fiducial_mask
        source_mask = (rho_grid <= float(source_mask_radius[cell_idx])) & fiducial_mask
        annulus_mask = (
            (rho_grid >= float(annulus_inner[cell_idx]))
            & (rho_grid < float(annulus_outer[cell_idx]))
            & fiducial_mask
        )
        on_masks[cell_idx] = on_mask
        source_masks[cell_idx] = source_mask
        training_mask[cell_idx] = annulus_mask
        on_pixels[cell_idx] = int(np.count_nonzero(on_mask))
        annulus_pixels[cell_idx] = int(np.count_nonzero(annulus_mask))
        off_pixels[cell_idx] = int(annulus_pixels[cell_idx])
        annulus_counts[cell_idx] = float(counts_map[cell_idx][annulus_mask].sum())
        off_counts[cell_idx] = float(annulus_counts[cell_idx])

        warnings = []
        if annulus_pixels[cell_idx] < int(min_training_pixels):
            core_warning[cell_idx] = True
            continue

        x_train = x_grid[annulus_mask].ravel()
        y_train = y_grid[annulus_mask].ravel()
        z = counts_map[cell_idx][annulus_mask].astype(np.float64).ravel()
        design = surface_design_matrix(x_train, y_train, int(surface_order))
        weights = 1.0 / np.sqrt(np.maximum(z, 1.0))
        weighted_design = design * weights[:, None]
        weighted_z = z * weights
        rank[cell_idx] = int(np.linalg.matrix_rank(weighted_design))
        if rank[cell_idx] < design.shape[1]:
            core_warning[cell_idx] = True
            continue
        condition[cell_idx] = float(np.linalg.cond(weighted_design))
        if not np.isfinite(condition[cell_idx]) or condition[cell_idx] > float(condition_max):
            core_warning[cell_idx] = True
            continue

        coeff, _, _, _ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)
        pred_train = design @ coeff
        variance = np.maximum(z, 1.0)
        resid = z - pred_train
        chi2[cell_idx] = float(np.sum((resid * resid) / variance))
        ndof[cell_idx] = max(0, int(z.size - coeff.size))
        xtwx = weighted_design.T @ weighted_design
        try:
            cov = np.linalg.inv(xtwx) * (chi2[cell_idx] / max(ndof[cell_idx], 1))
        except np.linalg.LinAlgError:
            cov = np.full((coeff.size, coeff.size), np.nan, dtype=np.float64)
            core_warning[cell_idx] = True

        pred_full = (full_design @ coeff).reshape(n_y, n_x)
        neg = fiducial_mask & (pred_full < 0.0)
        negative_pixels[cell_idx] = int(np.count_nonzero(neg))
        if negative_pixels[cell_idx] > 0:
            core_warning[cell_idx] = True
        pred_clipped = np.maximum(pred_full, 0.0)
        pred_clipped[~fiducial_mask] = np.nan
        valid_annulus_model = annulus_mask & np.isfinite(pred_clipped)
        raw_annulus_sum = float(np.nansum(pred_clipped[valid_annulus_model]))
        annulus_model_counts_raw[cell_idx] = raw_annulus_sum
        annulus_count_residual_raw[cell_idx] = float(annulus_counts[cell_idx] - raw_annulus_sum)
        scale = 1.0
        if annulus_normalize_surface:
            if raw_annulus_sum > 0.0 and np.isfinite(raw_annulus_sum):
                scale = float(annulus_counts[cell_idx] / raw_annulus_sum)
            else:
                core_warning[cell_idx] = True
        surface_scale[cell_idx] = scale
        pred_final = pred_clipped * scale
        pred_final[~fiducial_mask] = np.nan
        final_annulus_sum = float(np.nansum(pred_final[valid_annulus_model]))
        annulus_model_counts_final[cell_idx] = final_annulus_sum
        annulus_count_residual_final[cell_idx] = float(annulus_counts[cell_idx] - final_annulus_sum)
        background_map[cell_idx] = pred_final.astype(np.float32)
        core_background_map[cell_idx][on_mask] = pred_final[on_mask].astype(np.float32)
        residual = np.full((n_y, n_x), np.nan, dtype=np.float64)
        valid_bg = annulus_mask & np.isfinite(pred_final)
        residual[valid_bg] = (counts_map[cell_idx][valid_bg] - pred_final[valid_bg]) / np.sqrt(
            np.maximum(pred_final[valid_bg], 1.0)
        )
        residual_map[cell_idx] = residual.astype(np.float32)
        ann_resid = residual[valid_bg]
        residual_mean[cell_idx] = float(np.nanmean(ann_resid)) if ann_resid.size else np.nan
        residual_rms[cell_idx] = float(np.sqrt(np.nanmean(ann_resid * ann_resid))) if ann_resid.size else np.nan
        coeffs[cell_idx] = padded_coefficients(coeff, int(surface_order))
        covariances[cell_idx] = padded_covariance(cov, int(surface_order))
        pixel_center_value = float(np.nansum(pred_final[on_mask]))
        analytic_value = integrate_centered_disk_polynomial(
            coeff,
            float(r_opt_deg[cell_idx]),
            grid_step_deg,
            scale,
        )
        analytic_minimum = scale * minimum_quadratic_on_centered_disk(
            coeff,
            float(r_opt_deg[cell_idx]),
        )
        b_on_pixel_center[cell_idx] = pixel_center_value
        b_on_analytic[cell_idx] = analytic_value
        on_aperture_surface_min[cell_idx] = analytic_minimum
        b_on[cell_idx] = select_on_aperture_background(
            pixel_center_b_on=pixel_center_value,
            analytic_b_on=analytic_value,
            analytic_surface_min=analytic_minimum,
            method=str(on_aperture_integration),
            cell_id=int(cell_ids[cell_idx]),
            analytic_required=bool(analytic_required_mask[cell_idx]),
        )
        if str(on_aperture_integration) == "analytic-quadratic":
            if analytic_minimum > 0.0 and analytic_value > 0.0:
                on_aperture_integration_by_cell[cell_idx] = "analytic-quadratic"
            else:
                on_aperture_integration_by_cell[cell_idx] = "pixel-center-excluded-fallback"
        if b_on[cell_idx] <= 0.0 or on_pixels[cell_idx] <= 0:
            core_warning[cell_idx] = True
        fit_success[cell_idx] = not bool(core_warning[cell_idx])

    diagnostics = {
        "annulus_inner_deg": annulus_inner.astype(np.float32),
        "annulus_outer_deg": annulus_outer.astype(np.float32),
        "source_mask_radius_deg": source_mask_radius.astype(np.float32),
        "annulus_shifted_flag": shifted.astype(bool),
        "surface_coefficients": coeffs.astype(np.float64),
        "surface_covariance": covariances.astype(np.float64),
        "fit_chi2": chi2.astype(np.float64),
        "fit_ndof": ndof.astype(np.int64),
        "fit_rank": rank.astype(np.int64),
        "fit_condition_number": condition.astype(np.float64),
        "annulus_counts": annulus_counts.astype(np.float64),
        "annulus_pixels": annulus_pixels.astype(np.int64),
        "annulus_residual_mean": residual_mean.astype(np.float64),
        "annulus_residual_rms": residual_rms.astype(np.float64),
        "annulus_surface_scale": surface_scale.astype(np.float64),
        "annulus_model_counts_raw": annulus_model_counts_raw.astype(np.float64),
        "annulus_model_counts_final": annulus_model_counts_final.astype(np.float64),
        "annulus_count_residual_raw": annulus_count_residual_raw.astype(np.float64),
        "annulus_count_residual_final": annulus_count_residual_final.astype(np.float64),
        "core_extrapolation_warning": core_warning.astype(bool),
        "negative_background_pixels": negative_pixels.astype(np.int64),
        "surface_fit_success": fit_success.astype(bool),
        "annulus_residual_map": residual_map.astype(np.float32),
        "core_background_map": core_background_map.astype(np.float32),
        "annulus_off_counts": off_counts.astype(np.float64),
        "annulus_off_pixels": off_pixels.astype(np.int64),
        "B_on_pixel_center": b_on_pixel_center.astype(np.float64),
        "B_on_analytic": b_on_analytic.astype(np.float64),
        "on_area_deg2": on_area_deg2.astype(np.float64),
        "on_effective_pixels": on_effective_pixels.astype(np.float64),
        "on_aperture_surface_min": on_aperture_surface_min.astype(np.float64),
        "on_aperture_grid_step_deg": np.asarray([grid_step_deg], dtype=np.float64),
        "analytic_required_mask": analytic_required_mask.astype(bool),
        "on_aperture_integration_by_cell": on_aperture_integration_by_cell.astype("U32"),
    }
    return b_on, background_map, training_mask, on_masks, source_masks, diagnostics


def rectangle_basis_grid(x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    x0, y0 = np.meshgrid(x_edges[:-1], y_edges[:-1])
    x1, y1 = np.meshgrid(x_edges[1:], y_edges[1:])
    dx = x1 - x0
    dy = y1 - y0
    return np.stack(
        [
            dx * dy,
            0.5 * (x1 * x1 - x0 * x0) * dy,
            dx * 0.5 * (y1 * y1 - y0 * y0),
            (x1**3 - x0**3) * dy / 3.0,
            0.25 * (x1 * x1 - x0 * x0) * (y1 * y1 - y0 * y0),
            dx * (y1**3 - y0**3) / 3.0,
        ],
        axis=-1,
    ).astype(np.float64)


def integrate_centered_annulus_density(coefficients: np.ndarray, inner_deg: float, outer_deg: float) -> float:
    coeff = polynomial_coefficients6(coefficients)
    inner = float(inner_deg)
    outer = float(outer_deg)
    if inner < 0.0 or outer <= inner:
        raise ValueError("Annulus radii must satisfy 0 <= inner < outer")
    return float(
        math.pi * (outer * outer - inner * inner) * coeff[0]
        + math.pi * (outer**4 - inner**4) * (coeff[3] + coeff[5]) / 4.0
    )


def integrate_density_disk_via_analytic_baseline(
    coefficients: np.ndarray, radius_deg: float, grid_step_deg: float
) -> float:
    """Adapt per-deg2 density coefficients to the prerequisite per-pixel analytic API."""
    step = float(grid_step_deg)
    return integrate_centered_disk_polynomial(
        np.asarray(coefficients, dtype=np.float64) * step * step,
        float(radius_deg),
        step,
    )


def estimate_roi_poisson_pooled_background(
    counts_map: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cells: Sequence[CellSpec],
    r_opt_deg: np.ndarray,
    continuous_annulus_counts: np.ndarray,
    manifest: Dict[str, object],
    *,
    roi_fiducial_deg: float,
    annulus_default_inner_deg: float,
    annulus_width_deg: float,
    annulus_source_mask_min_deg: float,
    annulus_source_mask_r_opt_factor: float,
    annulus_source_mask_margin_deg: float,
    annulus_max_inner_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    n_cells, n_y, n_x = counts_map.shape
    if len(cells) != n_cells:
        raise ValueError("Poisson background cell axis does not match loaded cells")
    cell_ids = np.asarray([cell.cell_id for cell in cells], dtype=np.int64)
    index_by_id = {int(cell_id): index for index, cell_id in enumerate(cell_ids)}
    basis_grid = rectangle_basis_grid(np.asarray(x_edges), np.asarray(y_edges))
    basis_flat = basis_grid.reshape(-1, 6)
    x_centers = 0.5 * (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:]))
    y_centers = 0.5 * (np.asarray(y_edges[:-1]) + np.asarray(y_edges[1:]))
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    rho_grid = np.hypot(x_grid, y_grid)
    fiducial_mask = rho_grid < float(roi_fiducial_deg)
    source_radius, annulus_inner, annulus_outer, shifted = annulus_placement(
        np.asarray(r_opt_deg, dtype=np.float64),
        default_inner_deg=annulus_default_inner_deg,
        width_deg=annulus_width_deg,
        source_mask_min_deg=annulus_source_mask_min_deg,
        source_mask_r_opt_factor=annulus_source_mask_r_opt_factor,
        source_mask_margin_deg=annulus_source_mask_margin_deg,
        max_inner_deg=annulus_max_inner_deg,
    )
    training_mask = np.zeros((n_cells, n_y, n_x), dtype=bool)
    on_masks = np.zeros_like(training_mask)
    source_masks = np.zeros_like(training_mask)
    for index in range(n_cells):
        training_mask[index] = (
            (rho_grid >= annulus_inner[index])
            & (rho_grid < annulus_outer[index])
            & fiducial_mask
        )
        on_masks[index] = (rho_grid <= float(r_opt_deg[index])) & fiducial_mask
        source_masks[index] = (rho_grid <= source_radius[index]) & fiducial_mask

    expected_counts_raw = manifest.get("continuous_annulus_counts")
    if not isinstance(expected_counts_raw, dict):
        raise ValueError("Pooling manifest is missing continuous_annulus_counts for all donor cells")
    expected_counts = np.asarray([int(expected_counts_raw[str(cell_id)]) for cell_id in cell_ids], dtype=np.int64)
    actual_counts = np.asarray(continuous_annulus_counts, dtype=np.int64)
    if not np.array_equal(actual_counts, expected_counts):
        mismatches = [
            {"cell_id": int(cell_ids[index]), "manifest": int(expected_counts[index]), "stage_d": int(actual_counts[index])}
            for index in np.flatnonzero(actual_counts != expected_counts)[:10]
        ]
        raise ValueError(f"Continuous annulus counts do not match the frozen manifest: {mismatches}")

    assignments = manifest["execution_cell_assignments"]
    if not isinstance(assignments, dict):
        raise ValueError("Pooling manifest execution assignments are invalid")
    fit_by_pool: Dict[Tuple[Tuple[int, ...], int], object] = {}
    pool_key_by_cell: Dict[int, Tuple[Tuple[int, ...], int]] = {}
    for cell_id in cell_ids:
        assignment = assignments[str(int(cell_id))]
        donors = tuple(int(value) for value in assignment["donor_cell_ids"])
        order = int(assignment["surface_order"])
        key = (donors, order)
        pool_key_by_cell[int(cell_id)] = key
        if key in fit_by_pool:
            continue
        missing = sorted(set(donors) - set(index_by_id))
        if missing:
            raise ValueError(f"Frozen pool {donors} references unloaded cells: {missing}")
        donor_counts = {cell: counts_map[index_by_id[cell]].reshape(-1) for cell in donors}
        donor_basis = {cell: basis_flat for cell in donors}
        donor_masks = {cell: training_mask[index_by_id[cell]].reshape(-1) for cell in donors}
        shape_contributors = {
            cell: int(expected_counts_raw[str(cell)]) >= 100 for cell in donors
        }
        fit_by_pool[key] = fit_profiled_poisson_surface(
            donor_counts,
            donor_basis,
            donor_masks,
            donors,
            order,
            float(roi_fiducial_deg),
            shape_contributors,
        )

    background_map = np.full((n_cells, n_y, n_x), np.nan, dtype=np.float64)
    residual_map = np.full_like(background_map, np.nan)
    core_background_map = np.full_like(background_map, np.nan)
    b_on = np.full(n_cells, np.nan, dtype=np.float64)
    b_on_pixel = np.full(n_cells, np.nan, dtype=np.float64)
    density_coefficients = np.full((n_cells, 6), np.nan, dtype=np.float64)
    shape_coefficients = np.full((n_cells, 6), np.nan, dtype=np.float64)
    surface_order = np.full(n_cells, -1, dtype=np.int16)
    poisson_deviance = np.full(n_cells, np.nan, dtype=np.float64)
    fit_ndof = np.zeros(n_cells, dtype=np.int64)
    positive_minimum = np.full(n_cells, np.nan, dtype=np.float64)
    positive_minimum_xy = np.full((n_cells, 2), np.nan, dtype=np.float64)
    pool_target_cell_id = np.zeros(n_cells, dtype=np.int64)
    donor_ids_json = np.empty(n_cells, dtype="U256")
    optimizer_status_json = np.empty(n_cells, dtype="U2048")
    cv_scores_json = np.empty(n_cells, dtype="U16384")
    target_cells = manifest.get("cells", {})
    target_cell_ids = {int(value) for value in manifest.get("target_cell_ids", [])}
    grid_step = float(np.asarray(x_edges)[1] - np.asarray(x_edges)[0])
    for index, cell_id_raw in enumerate(cell_ids):
        cell_id = int(cell_id_raw)
        assignment = assignments[str(cell_id)]
        fit = fit_by_pool[pool_key_by_cell[cell_id]]
        shape = np.asarray(fit.shape_coefficients, dtype=np.float64)
        annulus_integral = integrate_centered_annulus_density(
            shape, float(annulus_inner[index]), min(float(annulus_outer[index]), float(roi_fiducial_deg))
        )
        if not np.isfinite(annulus_integral) or annulus_integral <= 0.0:
            raise PoissonSurfaceFitError(f"Cell {cell_id} has non-positive analytic annulus integral")
        normalization = float(actual_counts[index]) / annulus_integral
        density = normalization * shape
        prediction = (basis_flat @ density).reshape(n_y, n_x)
        prediction[~fiducial_mask] = np.nan
        if np.any(prediction[fiducial_mask] < 0.0):
            valid_values = prediction[fiducial_mask]
            worst_flat = int(np.nanargmin(valid_values))
            y_index, x_index = np.argwhere(fiducial_mask)[worst_flat]
            raise PoissonSurfaceFitError(
                f"Cell {cell_id} has non-positive fitted pixel expectation: "
                f"value={prediction[y_index, x_index]:.17g}, "
                f"center=({x_centers[x_index]:.17g},{y_centers[y_index]:.17g}), "
                f"rho={rho_grid[y_index, x_index]:.17g}, "
                f"shape_integral={float(basis_grid[y_index, x_index] @ shape):.17g}, "
                f"shape_coefficients={shape.tolist()}, optimizer_status={fit.optimizer_status}"
            )
        background_map[index] = prediction
        valid = training_mask[index]
        residual_map[index, valid] = (
            counts_map[index, valid] - prediction[valid]
        ) / np.sqrt(np.maximum(prediction[valid], 1.0))
        core_background_map[index, on_masks[index]] = prediction[on_masks[index]]
        analytic_b_on = integrate_density_disk_via_analytic_baseline(
            density, float(r_opt_deg[index]), grid_step
        )
        if not np.isfinite(analytic_b_on) or analytic_b_on < 0.0:
            raise PoissonSurfaceFitError(f"Cell {cell_id} has non-positive analytic B_on")
        if cell_id in target_cell_ids and analytic_b_on <= 0.0:
            raise PoissonSurfaceFitError(f"Target cell {cell_id} has non-positive analytic B_on")
        b_on[index] = analytic_b_on
        b_on_pixel[index] = float(np.nansum(prediction[on_masks[index]]))
        density_coefficients[index] = density
        shape_coefficients[index] = shape
        surface_order[index] = int(fit.order)
        poisson_deviance[index] = float(fit.poisson_deviance)
        fit_ndof[index] = int(fit.ndof)
        positive_minimum[index] = float(fit.positive_minimum)
        positive_minimum_xy[index] = np.asarray(fit.positive_minimum_xy)
        pool_target_cell_id[index] = int(assignment["pool_target_cell_id"])
        donor_ids_json[index] = json.dumps(list(fit.donor_cell_ids), separators=(",", ":"))
        optimizer_status_json[index] = json.dumps(fit.optimizer_status, sort_keys=True, separators=(",", ":"))
        pool_target = str(int(assignment["pool_target_cell_id"]))
        cv = target_cells.get(pool_target, {}).get("cross_validation", {}) if isinstance(target_cells, dict) else {}
        cv_scores_json[index] = json.dumps(cv, sort_keys=True, separators=(",", ":"))

    annulus_pixels = np.count_nonzero(training_mask, axis=(1, 2)).astype(np.int64)
    annulus_model = np.asarray(
        [float(np.nansum(background_map[index][training_mask[index]])) for index in range(n_cells)]
    )
    diagnostics = {
        "annulus_inner_deg": annulus_inner.astype(np.float64),
        "annulus_outer_deg": annulus_outer.astype(np.float64),
        "source_mask_radius_deg": source_radius.astype(np.float64),
        "annulus_shifted_flag": shifted.astype(bool),
        "surface_coefficients": density_coefficients,
        "surface_shape_coefficients": shape_coefficients,
        "surface_density_coefficients": density_coefficients,
        "surface_covariance": np.full((n_cells, 6, 6), np.nan),
        "surface_order": surface_order,
        "fit_chi2": poisson_deviance.copy(),
        "poisson_deviance": poisson_deviance,
        "fit_ndof": fit_ndof,
        "fit_rank": np.asarray([1 if value == 0 else 3 if value == 1 else 6 for value in surface_order]),
        "fit_condition_number": np.full(n_cells, np.nan),
        "annulus_counts": actual_counts.astype(np.float64),
        "continuous_annulus_counts": actual_counts,
        "annulus_pixels": annulus_pixels,
        "annulus_residual_mean": np.nanmean(residual_map, axis=(1, 2)),
        "annulus_residual_rms": np.sqrt(np.nanmean(residual_map * residual_map, axis=(1, 2))),
        "annulus_surface_scale": np.ones(n_cells, dtype=np.float64),
        "annulus_model_counts_raw": annulus_model,
        "annulus_model_counts_final": annulus_model,
        "annulus_count_residual_raw": actual_counts - annulus_model,
        "annulus_count_residual_final": actual_counts - annulus_model,
        "core_extrapolation_warning": np.zeros(n_cells, dtype=bool),
        "negative_background_pixels": np.zeros(n_cells, dtype=np.int64),
        "surface_fit_success": np.ones(n_cells, dtype=bool),
        "annulus_residual_map": residual_map.astype(np.float32),
        "core_background_map": core_background_map.astype(np.float32),
        "annulus_off_counts": actual_counts.astype(np.float64),
        "annulus_off_pixels": annulus_pixels,
        "B_on_pixel_center": b_on_pixel,
        "B_on_analytic": b_on.copy(),
        "on_area_deg2": math.pi * np.asarray(r_opt_deg, dtype=np.float64) ** 2,
        "on_effective_pixels": math.pi * np.asarray(r_opt_deg, dtype=np.float64) ** 2 / (grid_step * grid_step),
        "on_aperture_surface_min": positive_minimum,
        "positive_minimum": positive_minimum,
        "density_positive_minimum": np.asarray(
            [
                (float(actual_counts[index]) / integrate_centered_annulus_density(
                    shape_coefficients[index],
                    float(annulus_inner[index]),
                    min(float(annulus_outer[index]), float(roi_fiducial_deg)),
                )) * positive_minimum[index]
                for index in range(n_cells)
            ],
            dtype=np.float64,
        ),
        "positive_minimum_xy": positive_minimum_xy,
        "on_aperture_grid_step_deg": np.asarray([grid_step]),
        "analytic_required_mask": np.ones(n_cells, dtype=bool),
        "on_aperture_integration_by_cell": np.full(n_cells, "analytic-quadratic", dtype="U32"),
        "pool_target_cell_id": pool_target_cell_id,
        "pool_donor_cell_ids_json": donor_ids_json,
        "optimizer_status_json": optimizer_status_json,
        "cv_scores_json": cv_scores_json,
        "roi_fit_statistic": np.full(n_cells, "poisson", dtype="U16"),
    }
    return b_on, background_map.astype(np.float32), training_mask, on_masks, source_masks, diagnostics


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


def known_background_sigma_map(
    counts_map: np.ndarray,
    background_map: np.ndarray,
    fiducial_mask: np.ndarray,
) -> np.ndarray:
    counts = counts_map.astype(np.float64)
    background = background_map.astype(np.float64)
    term = np.zeros(counts.shape, dtype=np.float64)
    valid = np.isfinite(counts) & np.isfinite(background) & (background > 0.0)
    positive_counts = valid & (counts > 0.0)
    term[positive_counts] = (
        counts[positive_counts] * np.log(counts[positive_counts] / background[positive_counts])
        - (counts[positive_counts] - background[positive_counts])
    )
    zero_counts = valid & (counts <= 0.0)
    term[zero_counts] = background[zero_counts]
    term = np.maximum(term, 0.0)
    signed = np.sign(counts - background) * np.sqrt(2.0 * term)

    sigma = np.full(counts.shape, np.nan, dtype=np.float32)
    fiducial = fiducial_mask.astype(bool)[None, :, :]
    keep = valid & fiducial
    sigma[keep] = signed[keep].astype(np.float32)
    return sigma


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
    validate_roi_poisson_args(args)
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


def plot_roi_counts_grid(
    counts_map: np.ndarray,
    cells: Sequence[CellSpec],
    xy_edges: np.ndarray,
    output_path: Path,
    *,
    title: str,
    roi_fiducial_deg: float,
) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.05 * len(pred_bins), 1.75 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    logged = log_map(counts_map.astype(np.float32))
    finite = logged[np.isfinite(logged)]
    vmin = float(np.percentile(finite, 5.0)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.0, 1.0
    first_im = None
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle_x = float(roi_fiducial_deg) * np.cos(theta)
    circle_y = float(roi_fiducial_deg) * np.sin(theta)
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
                aspect="equal",
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if first_im is None:
                first_im = im
            ax.plot(circle_x, circle_y, color="white", linewidth=0.35, alpha=0.85)
            ax.scatter([0.0], [0.0], marker="+", s=18, c="white", linewidths=0.6)
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit_bin}\ny (deg)", fontsize=6.7)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_im is not None:
        cbar = fig.colorbar(first_im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        cbar.set_label("log10 counts", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_roi_signed_grid(
    values: np.ndarray,
    cells: Sequence[CellSpec],
    xy_edges: np.ndarray,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    roi_fiducial_deg: float,
    r_opt_deg: np.ndarray,
    limit_percentile: float = 99.0,
) -> None:
    plt = setup_matplotlib()
    from matplotlib.colors import TwoSlopeNorm

    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.05 * len(pred_bins), 1.75 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(np.abs(finite), float(limit_percentile))) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    first_im = None
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    fiducial_x = float(roi_fiducial_deg) * np.cos(theta)
    fiducial_y = float(roi_fiducial_deg) * np.sin(theta)
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue
            im = ax.imshow(
                values[cell.index],
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            if first_im is None:
                first_im = im
            ax.plot(fiducial_x, fiducial_y, color="#222222", linewidth=0.35, alpha=0.8)
            on_radius = float(r_opt_deg[cell.index])
            ax.plot(on_radius * np.cos(theta), on_radius * np.sin(theta), color="#111111", linewidth=0.45, alpha=0.9)
            ax.scatter([0.0], [0.0], marker="+", s=18, c="#111111", linewidths=0.6)
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit_bin}\ny (deg)", fontsize=6.7)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_im is not None:
        cbar = fig.colorbar(first_im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        cbar.set_label(colorbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_roi_mask_summary(
    fiducial_mask: np.ndarray,
    training_mask: np.ndarray,
    on_masks: np.ndarray,
    xy_edges: np.ndarray,
    output_path: Path,
    *,
    cell_index: int = 0,
) -> None:
    plt = setup_matplotlib()
    view = np.zeros(fiducial_mask.shape, dtype=np.float32)
    view[fiducial_mask] = 1.0
    view[training_mask[cell_index]] = 2.0
    view[on_masks[cell_index]] = 3.0
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=150)
    im = ax.imshow(
        view,
        origin="lower",
        extent=[float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])],
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=3,
    )
    ax.set_xlabel("x = ΔRA cos(Dec_Crab) (deg)")
    ax.set_ylabel("y = ΔDec (deg)")
    ax.set_title(f"Stage D ROI masks, cell index {cell_index}")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["outside", "fiducial", "training", "on"])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roi_annulus_mask_grid(
    training_mask: np.ndarray,
    source_masks: np.ndarray,
    on_masks: np.ndarray,
    xy_edges: np.ndarray,
    cells: Sequence[CellSpec],
    output_path: Path,
) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.05 * len(pred_bins), 1.75 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue
            view = np.zeros(training_mask.shape[1:], dtype=np.float32)
            view[source_masks[cell.index]] = 1.0
            view[training_mask[cell.index]] = 2.0
            view[on_masks[cell.index]] = 3.0
            ax.imshow(view, origin="lower", extent=extent, aspect="equal", interpolation="nearest", cmap="viridis", vmin=0, vmax=3)
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit_bin}\ny (deg)", fontsize=6.7)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)
    fig.suptitle("Stage D annulus training masks", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.982])
    fig.savefig(output_path)
    plt.close(fig)


def plot_roi_coverage_profile(
    rho_hist_total: np.ndarray,
    rho_edges: np.ndarray,
    output_path: Path,
    *,
    roi_fiducial_deg: float,
    roi_edge_diagnostic_deg: float,
) -> None:
    plt = setup_matplotlib()
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    widths = np.diff(rho_edges)
    annulus_area = np.pi * (rho_edges[1:] ** 2 - rho_edges[:-1] ** 2)
    density = np.divide(rho_hist_total, annulus_area, out=np.zeros_like(centers), where=annulus_area > 0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    ax.step(centers, density, where="mid", color="#1f4e79", linewidth=1.0)
    ax.axvline(float(roi_fiducial_deg), color="#d62728", linestyle="--", linewidth=0.9, label=f"fiducial {roi_fiducial_deg:g} deg")
    ax.axvline(float(roi_edge_diagnostic_deg), color="#7f7f7f", linestyle=":", linewidth=0.9, label=f"edge diag {roi_edge_diagnostic_deg:g} deg")
    ax.set_xlabel("rho from Crab (deg)")
    ax.set_ylabel("counts per deg^2")
    ax.set_title("Stage D ROI coverage profile")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(fontsize=8)
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
        "B_on_pixel_center",
        "B_on_analytic",
        "on_area_deg2",
        "on_effective_pixels",
        "on_aperture_surface_min",
        "on_aperture_integration",
        "analytic_required",
        "N_off",
        "alpha",
        "max_p_on",
        "on_pixels",
        "off_pixels",
        "off_counts",
        "fiducial_events",
        "edge_diagnostic_events",
        "background_mode",
        "background_method",
        "r_opt_large_warning",
        "r_opt_extreme_warning",
        "background_form",
        "annulus_inner_deg",
        "annulus_outer_deg",
        "surface_fit_chi2",
        "surface_fit_ndof",
        "roi_fit_statistic",
        "surface_order",
        "continuous_annulus_count",
        "pool_target_cell_id",
        "pool_donor_cell_ids",
        "poisson_deviance",
        "positive_minimum",
        "surface_condition_number",
        "annulus_residual_rms",
        "annulus_surface_scale",
        "annulus_count_residual_raw",
        "annulus_count_residual_final",
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
        f.write("# Stage D Background Summary\n\n")
        f.write(f"- Run id: `{metadata['run_id']}`\n")
        f.write(f"- Stage C input: `{metadata['inputs']['stage_c_dir']}`\n")
        f.write(f"- PSF input: `{metadata['inputs']['psf_npz']}`\n")
        f.write(f"- Background mode: `{metadata['background_model'].get('background_mode', 'n/a')}`\n")
        f.write(f"- Background method: `{metadata['background_model'].get('method', 'n/a')}`\n")
        f.write(f"- Background form: `{metadata['background_model']['background_form']}`\n")
        if "roi" in metadata:
            f.write(f"- ROI fiducial radius: {metadata['roi'].get('fiducial_radius_deg', 'n/a')} deg\n")
            f.write(f"- ROI edge diagnostic radius: {metadata['roi'].get('edge_diagnostic_radius_deg', 'n/a')} deg\n")
        if "time_binning" in metadata:
            f.write(f"- Time bin: {metadata['time_binning'].get('time_bin_minutes', 'n/a')} min\n")
            f.write(f"- Active time bins: {metadata['time_binning'].get('active_time_bins', 'n/a')}\n")
            live_days = finite_float(metadata["time_binning"].get("total_live_time_days"))
            if live_days is not None:
                f.write(f"- Live time: {live_days:.6g} days\n")
        if "theta_coordinate_check" in metadata:
            f.write(f"- Theta sanity p95 absdiff: {metadata['theta_coordinate_check'].get('p95_absdiff_deg', 'n/a')} deg\n")
        f.write("\n| cell | Nhit bin | predE bin | events | masked frac | r_opt deg | B_on | off pixels | warnings |\n")
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            warnings = []
            if row["r_opt_large_warning"]:
                warnings.append("large_r_opt")
            if row["r_opt_extreme_warning"]:
                warnings.append("extreme_r_opt")
            extra_warnings = row.get("warnings")
            if isinstance(extra_warnings, list):
                warnings.extend(str(item) for item in extra_warnings)
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{row['selected_events']} | {row['source_masked_fraction']:.4g} | "
                f"{row['r_opt_deg']:.5g} | {row['B_on']:.6g} | {row.get('off_pixels', '')} | "
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


def canonical_manifest_sha256(payload: Dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_poisson_manifest(path: Path) -> Dict[str, object]:
    payload = load_json(path)
    expected_sha = str(payload.get("manifest_sha256", ""))
    actual_sha = canonical_manifest_sha256(payload)
    if not expected_sha or expected_sha != actual_sha:
        raise ValueError(f"Pooling manifest self-hash mismatch: expected={expected_sha!r}, actual={actual_sha}")
    baseline_sha = str(payload.get("analytic_bon_baseline_sha", "")).strip()
    required_sha = os.environ.get("ANALYTIC_BON_BASELINE_SHA", "").strip()
    if not required_sha:
        raise ValueError("ANALYTIC_BON_BASELINE_SHA is required in Poisson mode")
    if baseline_sha != required_sha:
        raise ValueError(
            f"Pooling manifest analytic baseline {baseline_sha!r} does not match required {required_sha!r}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline_sha, "HEAD"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"Analytic B_on baseline {baseline_sha} is not an ancestor of HEAD")
    donor_ids = [int(value) for value in payload.get("donor_universe_cell_ids", [])]
    target_ids = [int(value) for value in payload.get("target_cell_ids", [])]
    if len(donor_ids) != 84 or len(set(donor_ids)) != 84:
        raise ValueError("Pooling manifest must contain exactly 84 unique donor-universe cells")
    if len(target_ids) != 44 or len(set(target_ids)) != 44:
        raise ValueError("Pooling manifest must contain exactly 44 unique target cells")
    assignments = payload.get("execution_cell_assignments")
    if not isinstance(assignments, dict) or {int(value) for value in assignments} != set(donor_ids):
        raise ValueError("Pooling manifest must freeze one execution assignment for every donor cell")
    return payload


def validate_roi_poisson_args(args: argparse.Namespace) -> None:
    if str(args.roi_fit_statistic) != "poisson":
        return
    if not args.pooling_manifest:
        raise ValueError("--roi-fit-statistic poisson requires --pooling-manifest")
    if str(args.roi_background_method) != "annulus-quadratic":
        raise ValueError("Poisson fitting requires --roi-background-method annulus-quadratic")
    if not bool(args.annulus_normalize_surface):
        raise ValueError("Poisson fitting requires --annulus-normalize-surface")
    if str(args.on_aperture_integration) != "analytic-quadratic":
        raise ValueError("Poisson fitting requires --on-aperture-integration analytic-quadratic")


def filter_cells_to_manifest(cells: Sequence[CellSpec], manifest: Dict[str, object]) -> List[CellSpec]:
    donor_ids = [int(value) for value in manifest["donor_universe_cell_ids"]]  # type: ignore[index]
    by_id = {int(cell.cell_id): cell for cell in cells}
    missing = sorted(set(donor_ids) - set(by_id))
    if missing:
        raise ValueError(f"Cell selection is missing pooling-manifest donor cells: {missing}")
    filtered: List[CellSpec] = []
    for index, cell_id in enumerate(donor_ids):
        cell = by_id[cell_id]
        filtered.append(
            CellSpec(
                index=index,
                cell_id=cell.cell_id,
                nhit_bin=cell.nhit_bin,
                predE_bin=cell.predE_bin,
                mc_count=cell.mc_count,
                selection_version=cell.selection_version,
                selection_reason=cell.selection_reason,
            )
        )
    return filtered


def total_live_time_seconds_from_stage_c(metadata: Dict[str, object], source_files_csv: Path) -> float:
    live = metadata.get("live_time_basis") if isinstance(metadata, dict) else None
    if isinstance(live, dict):
        value = finite_float(live.get("rough_live_time_seconds_sum_files"))
        if value is not None:
            return value
    total = 0.0
    if source_files_csv.exists():
        with source_files_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                value = finite_float(row.get("rough_live_time_seconds"))
                if value is not None:
                    total += value
    return total


def run_crab_roi_local_background(
    *,
    args: argparse.Namespace,
    start_time: float,
    stage_c_dir: Path,
    obs_events_dir: Path,
    stage_c_metadata_path: Path,
    stage_c_metadata: Dict[str, object],
    source_files_csv: Path,
    psf_npz: Path,
    selection_csv: Path,
    output_root: Path,
    run_dir: Path,
    run_id: str,
    cells: Sequence[CellSpec],
    r_opt_deg: np.ndarray,
    sigma_deg: np.ndarray,
    containment_r_opt: np.ndarray,
    mode_resolution: Dict[str, object],
) -> None:
    if float(args.roi_fiducial_deg) <= 0:
        raise ValueError("--roi-fiducial-deg must be positive")
    if float(args.roi_edge_diagnostic_deg) <= float(args.roi_fiducial_deg):
        raise ValueError("--roi-edge-diagnostic-deg must be greater than --roi-fiducial-deg")
    if float(args.roi_grid_step_deg) <= 0:
        raise ValueError("--roi-grid-step-deg must be positive")
    if float(args.roi_coverage_bin_deg) <= 0:
        raise ValueError("--roi-coverage-bin-deg must be positive")
    if float(args.roi_coverage_max_deg) <= float(args.roi_edge_diagnostic_deg):
        raise ValueError("--roi-coverage-max-deg must be greater than --roi-edge-diagnostic-deg")
    if str(args.on_aperture_integration) == "analytic-quadratic":
        if str(args.roi_background_method) != "annulus-quadratic":
            raise ValueError("analytic-quadratic on-aperture integration requires --roi-background-method annulus-quadratic")
        if not bool(args.annulus_normalize_surface):
            raise ValueError("analytic-quadratic on-aperture integration requires --annulus-normalize-surface")

    x_edges = make_offset_roi_edges(
        float(args.roi_edge_diagnostic_deg),
        float(args.roi_grid_step_deg),
        float(args.roi_grid_offset_x_fraction),
    )
    y_edges = make_offset_roi_edges(
        float(args.roi_edge_diagnostic_deg),
        float(args.roi_grid_step_deg),
        float(args.roi_grid_offset_y_fraction),
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    rho_hist_edges = make_edges(0.0, float(args.roi_coverage_max_deg), float(args.roi_coverage_bin_deg))

    continuous_inner = None
    continuous_outer = None
    poisson_manifest: Optional[Dict[str, object]] = None
    if str(args.roi_fit_statistic) == "poisson":
        poisson_manifest = load_and_validate_poisson_manifest(Path(args.pooling_manifest).resolve())
        _, continuous_inner, continuous_outer, _ = annulus_placement(
            np.asarray(r_opt_deg, dtype=np.float64),
            default_inner_deg=float(args.annulus_default_inner_deg),
            width_deg=float(args.annulus_width_deg),
            source_mask_min_deg=float(args.annulus_source_mask_min_deg),
            source_mask_r_opt_factor=float(args.annulus_source_mask_r_opt_factor),
            source_mask_margin_deg=float(args.annulus_source_mask_margin_deg),
            max_inner_deg=float(args.annulus_max_inner_deg),
        )

    print("Stage D background mode: crab_roi_local", flush=True)
    print(
        f"ROI grid: {x_edges.size - 1} x {y_edges.size - 1}, "
        f"fiducial rho<{float(args.roi_fiducial_deg):g} deg",
        flush=True,
    )

    scan = scan_stage_c_roi_events(
        obs_events_dir,
        cells,
        source_ra_deg=float(args.source_ra_deg),
        source_dec_deg=float(args.source_dec_deg),
        x_edges_deg=x_edges,
        y_edges_deg=y_edges,
        rho_hist_edges_deg=rho_hist_edges,
        roi_fiducial_deg=float(args.roi_fiducial_deg),
        roi_edge_diagnostic_deg=float(args.roi_edge_diagnostic_deg),
        annulus_inner_deg=continuous_inner,
        annulus_outer_deg=continuous_outer,
        batch_size=int(args.batch_size),
        max_batches=args.max_batches,
        print_every=int(args.print_every),
        mjd_min=args.mjd_min,
        mjd_max=args.mjd_max,
    )
    print(f"Scanned rows: {scan.input_rows:,}", flush=True)

    n_cells = len(cells)
    n_x = x_edges.size - 1
    n_y = y_edges.size - 1
    counts_map = scan.counts_flat.reshape(n_cells, n_y, n_x)
    rho_grid, fiducial_mask, edge_safe_mask, on_masks, source_masks = build_roi_masks(
        x_centers,
        r_opt_deg.astype(np.float64),
        y_centers=y_centers,
        roi_fiducial_deg=float(args.roi_fiducial_deg),
        roi_edge_margin_deg=float(args.roi_edge_margin_deg),
        roi_source_mask_deg=float(args.roi_source_mask_deg),
        source_mask_r_opt_factor=float(args.roi_source_mask_r_opt_factor),
    )
    annulus_diagnostics: Dict[str, np.ndarray] = {}
    background_method = str(args.roi_background_method).replace("-", "_")
    loaded_cell_ids = np.asarray([cell.cell_id for cell in cells], dtype=np.int64)
    analytic_required_mask = np.ones(n_cells, dtype=bool)
    if str(args.on_aperture_integration) == "analytic-quadratic" and str(args.analytic_required_cell_ids).strip():
        required_ids = {
            int(value.strip())
            for value in str(args.analytic_required_cell_ids).split(",")
            if value.strip()
        }
        unknown_ids = sorted(required_ids - set(int(value) for value in loaded_cell_ids))
        if unknown_ids:
            raise ValueError(f"--analytic-required-cell-ids contains cells not loaded by Stage D: {unknown_ids}")
        analytic_required_mask = np.isin(loaded_cell_ids, np.asarray(sorted(required_ids), dtype=np.int64))
        if not np.any(analytic_required_mask):
            raise ValueError("--analytic-required-cell-ids selected no Stage D cells")
    if str(args.roi_background_method) == "annulus-quadratic":
        if str(args.roi_fit_statistic) == "poisson":
            if poisson_manifest is None:
                raise RuntimeError("Validated pooling manifest was not loaded")
            b_on, background_map, training_mask, on_masks, source_masks, annulus_diagnostics = estimate_roi_poisson_pooled_background(
                counts_map,
                x_edges,
                y_edges,
                cells,
                r_opt_deg.astype(np.float64),
                scan.continuous_annulus_counts,
                poisson_manifest,
                roi_fiducial_deg=float(args.roi_fiducial_deg),
                annulus_default_inner_deg=float(args.annulus_default_inner_deg),
                annulus_width_deg=float(args.annulus_width_deg),
                annulus_source_mask_min_deg=float(args.annulus_source_mask_min_deg),
                annulus_source_mask_r_opt_factor=float(args.annulus_source_mask_r_opt_factor),
                annulus_source_mask_margin_deg=float(args.annulus_source_mask_margin_deg),
                annulus_max_inner_deg=float(args.annulus_max_inner_deg),
            )
        else:
            if not np.array_equal(x_edges, y_edges):
                raise ValueError("Independent x/y grid phases are available only in Poisson mode")
            b_on, background_map, training_mask, on_masks, source_masks, annulus_diagnostics = estimate_roi_annulus_surface_background(
                counts_map,
                x_centers,
                r_opt_deg.astype(np.float64),
                cell_ids=loaded_cell_ids,
                analytic_required_mask=analytic_required_mask,
                roi_fiducial_deg=float(args.roi_fiducial_deg),
                annulus_default_inner_deg=float(args.annulus_default_inner_deg),
                annulus_width_deg=float(args.annulus_width_deg),
                annulus_source_mask_min_deg=float(args.annulus_source_mask_min_deg),
                annulus_source_mask_r_opt_factor=float(args.annulus_source_mask_r_opt_factor),
                annulus_source_mask_margin_deg=float(args.annulus_source_mask_margin_deg),
                annulus_max_inner_deg=float(args.annulus_max_inner_deg),
                surface_order=int(args.roi_surface_order),
                condition_max=float(args.surface_condition_max),
                min_training_pixels=int(args.surface_min_training_pixels),
                annulus_normalize_surface=bool(args.annulus_normalize_surface),
                on_aperture_integration=str(args.on_aperture_integration),
            )
        off_counts = np.asarray(annulus_diagnostics["annulus_off_counts"], dtype=np.float64)
        off_pixels = np.asarray(annulus_diagnostics["annulus_off_pixels"], dtype=np.int64)
        on_pixels = np.asarray([int(np.count_nonzero(mask)) for mask in on_masks], dtype=np.int64)
    else:
        b_on, background_map, training_mask, off_counts, off_pixels, on_pixels = estimate_roi_dec_sideband_background(
            counts_map,
            fiducial_mask,
            edge_safe_mask,
            on_masks,
            source_masks,
        )
    if not np.all(np.isfinite(b_on)) or np.any(b_on < 0.0):
        raise RuntimeError("ROI-local background produced non-finite or negative B_on values")
    excess_map = counts_map.astype(np.float32) - background_map.astype(np.float32)
    excess_map[:, ~fiducial_mask] = np.nan
    known_b_sigma_grid = known_background_sigma_map(counts_map, background_map, fiducial_mask)

    total_live_time_sec = total_live_time_seconds_from_stage_c(stage_c_metadata, source_files_csv)
    total_live_time_days = total_live_time_sec / 86400.0 if total_live_time_sec > 0 else 0.0

    rows: List[Dict[str, object]] = []
    poisson_target_ids = (
        {int(value) for value in poisson_manifest.get("target_cell_ids", [])}
        if poisson_manifest is not None
        else set()
    )
    for cell in cells:
        idx = cell.index
        selected_events = int(scan.cell_total_events[idx])
        fiducial_events = int(scan.cell_fiducial_events[idx])
        source_masked_events = int(counts_map[idx][source_masks[idx] & fiducial_mask].sum())
        source_masked_fraction = float(source_masked_events) / float(fiducial_events) if fiducial_events > 0 else 0.0
        warnings: List[str] = []
        if off_pixels[idx] <= 0:
            warnings.append("no_training_pixels")
        if on_pixels[idx] <= 0:
            warnings.append("no_on_pixels")
        if b_on[idx] <= 0.0 and (
            str(args.roi_fit_statistic) != "poisson" or int(cell.cell_id) in poisson_target_ids
        ):
            warnings.append("non_positive_B_on")
        if scan.cell_fiducial_events[idx] <= 0:
            warnings.append("no_fiducial_events")
        if annulus_diagnostics:
            if bool(annulus_diagnostics["core_extrapolation_warning"][idx]):
                warnings.append("core_extrapolation_warning")
            if not bool(annulus_diagnostics["surface_fit_success"][idx]):
                warnings.append("surface_fit_not_successful")
        rows.append(
            {
                "cell_index": int(idx),
                "cell_id": int(cell.cell_id),
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "selected_events": selected_events,
                "grid_events": int(scan.cell_map_events[idx]),
                "out_of_grid_events": int(scan.cell_out_of_map_events[idx]),
                "source_masked_events": source_masked_events,
                "source_masked_fraction": source_masked_fraction,
                "fiducial_events": fiducial_events,
                "edge_diagnostic_events": int(scan.cell_edge_diagnostic_events[idx]),
                "live_time_days": total_live_time_days,
                "median_rate_hz": 0.0,
                "mean_rate_hz": float(selected_events / total_live_time_sec) if total_live_time_sec > 0 else 0.0,
                "r_opt_deg": float(r_opt_deg[idx]),
                "sigma_deg": float(sigma_deg[idx]),
                "containment_r_opt": float(containment_r_opt[idx]),
                "B_on": float(b_on[idx]),
                "B_on_pixel_center": (
                    float(annulus_diagnostics["B_on_pixel_center"][idx]) if annulus_diagnostics else ""
                ),
                "B_on_analytic": (
                    float(annulus_diagnostics["B_on_analytic"][idx]) if annulus_diagnostics else ""
                ),
                "on_area_deg2": (
                    float(annulus_diagnostics["on_area_deg2"][idx]) if annulus_diagnostics else ""
                ),
                "on_effective_pixels": (
                    float(annulus_diagnostics["on_effective_pixels"][idx]) if annulus_diagnostics else ""
                ),
                "on_aperture_surface_min": (
                    float(annulus_diagnostics["on_aperture_surface_min"][idx]) if annulus_diagnostics else ""
                ),
                "on_aperture_integration": (
                    str(annulus_diagnostics["on_aperture_integration_by_cell"][idx])
                    if annulus_diagnostics
                    else str(args.on_aperture_integration)
                ),
                "analytic_required": (
                    bool(annulus_diagnostics["analytic_required_mask"][idx]) if annulus_diagnostics else False
                ),
                "N_off": "",
                "alpha": "",
                "max_p_on": 0.0,
                "on_pixels": int(on_pixels[idx]),
                "off_pixels": int(off_pixels[idx]),
                "off_counts": float(off_counts[idx]),
                "r_opt_large_warning": bool(r_opt_deg[idx] > 10.0),
                "r_opt_extreme_warning": bool(r_opt_deg[idx] > 20.0),
                "background_form": "direct_expectation",
                "background_mode": "crab_roi_local",
                "background_method": background_method,
                "annulus_inner_deg": (
                    float(annulus_diagnostics["annulus_inner_deg"][idx]) if annulus_diagnostics else ""
                ),
                "annulus_outer_deg": (
                    float(annulus_diagnostics["annulus_outer_deg"][idx]) if annulus_diagnostics else ""
                ),
                "surface_fit_chi2": float(annulus_diagnostics["fit_chi2"][idx]) if annulus_diagnostics else "",
                "surface_fit_ndof": int(annulus_diagnostics["fit_ndof"][idx]) if annulus_diagnostics else "",
                "roi_fit_statistic": str(args.roi_fit_statistic),
                "surface_order": (
                    int(annulus_diagnostics["surface_order"][idx])
                    if annulus_diagnostics and "surface_order" in annulus_diagnostics
                    else int(args.roi_surface_order)
                ),
                "continuous_annulus_count": (
                    int(annulus_diagnostics["continuous_annulus_counts"][idx])
                    if annulus_diagnostics and "continuous_annulus_counts" in annulus_diagnostics
                    else ""
                ),
                "pool_target_cell_id": (
                    int(annulus_diagnostics["pool_target_cell_id"][idx])
                    if annulus_diagnostics and "pool_target_cell_id" in annulus_diagnostics
                    else ""
                ),
                "pool_donor_cell_ids": (
                    str(annulus_diagnostics["pool_donor_cell_ids_json"][idx])
                    if annulus_diagnostics and "pool_donor_cell_ids_json" in annulus_diagnostics
                    else ""
                ),
                "poisson_deviance": (
                    float(annulus_diagnostics["poisson_deviance"][idx])
                    if annulus_diagnostics and "poisson_deviance" in annulus_diagnostics
                    else ""
                ),
                "positive_minimum": (
                    float(annulus_diagnostics["positive_minimum"][idx])
                    if annulus_diagnostics and "positive_minimum" in annulus_diagnostics
                    else ""
                ),
                "surface_condition_number": (
                    float(annulus_diagnostics["fit_condition_number"][idx]) if annulus_diagnostics else ""
                ),
                "annulus_residual_rms": (
                    float(annulus_diagnostics["annulus_residual_rms"][idx]) if annulus_diagnostics else ""
                ),
                "annulus_surface_scale": (
                    float(annulus_diagnostics["annulus_surface_scale"][idx]) if annulus_diagnostics else ""
                ),
                "annulus_count_residual_raw": (
                    float(annulus_diagnostics["annulus_count_residual_raw"][idx]) if annulus_diagnostics else ""
                ),
                "annulus_count_residual_final": (
                    float(annulus_diagnostics["annulus_count_residual_final"][idx]) if annulus_diagnostics else ""
                ),
                "warnings": warnings,
            }
        )

    npz_path = run_dir / args.npz_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    metadata_path = run_dir / args.metadata_name
    source_masks_csv_path = run_dir / "source_masks.csv"

    np.savez_compressed(
        npz_path,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        x_edges_deg=x_edges.astype(np.float32),
        y_edges_deg=y_edges.astype(np.float32),
        x_centers_deg=x_centers.astype(np.float32),
        y_centers_deg=y_centers.astype(np.float32),
        rho_grid_deg=rho_grid.astype(np.float32),
        fiducial_mask=fiducial_mask.astype(bool),
        edge_safe_mask=edge_safe_mask.astype(bool),
        on_mask=on_masks.astype(bool),
        source_mask=source_masks.astype(bool),
        training_mask=training_mask.astype(bool),
        counts_map=counts_map.astype(np.int64),
        background_map=background_map.astype(np.float32),
        excess_map=excess_map.astype(np.float32),
        known_b_sigma_map=known_b_sigma_grid.astype(np.float32),
        rho_hist_edges_deg=rho_hist_edges.astype(np.float32),
        rho_hist_total=scan.rho_hist_total.astype(np.int64),
        rho_hist_by_cell=scan.rho_hist_by_cell.astype(np.int64),
        cell_total_events=scan.cell_total_events.astype(np.int64),
        cell_map_events=scan.cell_map_events.astype(np.int64),
        cell_fiducial_events=scan.cell_fiducial_events.astype(np.int64),
        cell_edge_diagnostic_events=scan.cell_edge_diagnostic_events.astype(np.int64),
        off_counts=off_counts.astype(np.float64),
        off_pixels=off_pixels.astype(np.int64),
        on_pixels=on_pixels.astype(np.int64),
        alpha=np.full(n_cells, np.nan, dtype=np.float32),
        N_off=np.full(n_cells, np.nan, dtype=np.float64),
        source_ra_deg=np.asarray([float(args.source_ra_deg)], dtype=np.float32),
        source_dec_deg=np.asarray([float(args.source_dec_deg)], dtype=np.float32),
        source_name=np.asarray(["Crab"], dtype="U32"),
        r_opt_deg=r_opt_deg.astype(np.float32),
        sigma_deg=sigma_deg.astype(np.float32),
        containment_r_opt=containment_r_opt.astype(np.float32),
        B_on=b_on.astype(np.float64),
        on_aperture_integration=np.asarray([str(args.on_aperture_integration)], dtype="U32"),
        **annulus_diagnostics,
    )

    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        background_title = (
            "Stage D ROI-local annulus quadratic surface background"
            if annulus_diagnostics
            else "Stage D ROI-local Dec-sideband background"
        )
        excess_title = (
            "Stage D ROI-local counts minus annulus quadratic surface background"
            if annulus_diagnostics
            else "Stage D ROI-local counts minus Dec-sideband background"
        )
        plot_outputs = {
            "roi_coverage_profile_png": str(run_dir / "roi_coverage_profile.png"),
            "roi_counts_grid_png": str(run_dir / "roi_counts_grid.png"),
            "roi_background_grid_png": str(run_dir / "roi_background_grid.png"),
            "roi_excess_grid_png": str(run_dir / "roi_excess_grid.png"),
            "roi_known_b_sigma_grid_png": str(run_dir / "roi_known_b_sigma_grid.png"),
            "roi_mask_summary_png": str(run_dir / "roi_mask_summary.png"),
            "annulus_training_mask_grid_png": str(run_dir / "annulus_training_mask_grid.png"),
            "annulus_residual_grid_png": str(run_dir / "annulus_residual_grid.png"),
            "core_background_grid_png": str(run_dir / "core_background_grid.png"),
            "background_prediction_png": str(run_dir / "background_prediction_grid.png"),
        }
        plot_roi_coverage_profile(
            scan.rho_hist_total,
            rho_hist_edges,
            Path(plot_outputs["roi_coverage_profile_png"]),
            roi_fiducial_deg=float(args.roi_fiducial_deg),
            roi_edge_diagnostic_deg=float(args.roi_edge_diagnostic_deg),
        )
        plot_roi_counts_grid(
            counts_map,
            cells,
            x_edges,
            Path(plot_outputs["roi_counts_grid_png"]),
            title="Stage D ROI-local counts",
            roi_fiducial_deg=float(args.roi_fiducial_deg),
        )
        plot_roi_counts_grid(
            background_map,
            cells,
            x_edges,
            Path(plot_outputs["roi_background_grid_png"]),
            title=background_title,
            roi_fiducial_deg=float(args.roi_fiducial_deg),
        )
        plot_roi_signed_grid(
            excess_map,
            cells,
            x_edges,
            Path(plot_outputs["roi_excess_grid_png"]),
            title=excess_title,
            colorbar_label="counts - background",
            roi_fiducial_deg=float(args.roi_fiducial_deg),
            r_opt_deg=r_opt_deg,
        )
        plot_roi_signed_grid(
            known_b_sigma_grid,
            cells,
            x_edges,
            Path(plot_outputs["roi_known_b_sigma_grid_png"]),
            title="Stage D ROI-local known-background residual",
            colorbar_label="known-B sigma",
            roi_fiducial_deg=float(args.roi_fiducial_deg),
            r_opt_deg=r_opt_deg,
        )
        plot_roi_mask_summary(
            fiducial_mask,
            training_mask,
            on_masks,
            x_edges,
            Path(plot_outputs["roi_mask_summary_png"]),
            cell_index=0,
        )
        plot_roi_annulus_mask_grid(
            training_mask,
            source_masks,
            on_masks,
            x_edges,
            cells,
            Path(plot_outputs["annulus_training_mask_grid_png"]),
        )
        if annulus_diagnostics:
            plot_roi_signed_grid(
                annulus_diagnostics["annulus_residual_map"],
                cells,
                x_edges,
                Path(plot_outputs["annulus_residual_grid_png"]),
                title="Stage D annulus fit residuals",
                colorbar_label="annulus residual sigma",
                roi_fiducial_deg=float(args.roi_fiducial_deg),
                r_opt_deg=r_opt_deg,
            )
            plot_roi_counts_grid(
                annulus_diagnostics["core_background_map"],
                cells,
                x_edges,
                Path(plot_outputs["core_background_grid_png"]),
                title="Stage D core extrapolated background",
                roi_fiducial_deg=float(args.roi_fiducial_deg),
            )
        plot_background_grid(rows, cells, Path(plot_outputs["background_prediction_png"]))

    severe_warnings = [
        f"cell {row['cell_id']}: {','.join(row['warnings'])}"
        for row in rows
        if row.get("warnings")
    ]
    if args.max_batches is not None:
        quality_status = "smoke_warning" if severe_warnings else "smoke"
        promotable = False
        quality_reason = "max_batches set; partial scans are smoke tests and are not promoted"
    elif severe_warnings:
        quality_status = "failed"
        promotable = False
        quality_reason = "ROI-local background has cells with invalid or fragile background estimates"
    else:
        quality_status = "ok"
        promotable = True
        quality_reason = "ROI-local background passed basic positivity and training-pixel checks"
    if annulus_diagnostics:
        if str(args.roi_fit_statistic) == "poisson":
            b_on_formula = (
                "continuous_N_annulus * analytic_disk_integral(q,r_opt) / "
                "analytic_annulus_integral(q,r_inner,r_outer_clipped_to_fiducial)"
            )
        elif str(args.on_aperture_integration) == "analytic-quadratic":
            b_on_formula = (
                "annulus_scale/grid_step_deg^2 * "
                "(pi*r_opt_deg^2*c0 + pi*r_opt_deg^4*(c_xx+c_yy)/4)"
            )
        elif bool(args.annulus_normalize_surface):
            b_on_formula = (
                "integral of annulus-count-normalized weighted least-squares annulus "
                "quadratic surface over pixel-center on-aperture mask"
            )
        else:
            b_on_formula = "integral of weighted least-squares annulus quadratic surface over on aperture"
    else:
        b_on_formula = "sum_y mean(counts in same y strip training pixels) * on_pixels_y"
    metadata: Dict[str, object] = {
        "description": "Stage D ROI-local background for configured (Nhit, predicted logE) Crab SED cells.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {
            "stage_c_dir": str(stage_c_dir),
            "obs_events_dir": str(obs_events_dir),
            "stage_c_metadata_json": str(stage_c_metadata_path),
            "source_files_csv": str(source_files_csv),
            "psf_npz": str(psf_npz),
            "cell_selection_csv": str(selection_csv),
            "pooling_manifest": str(Path(args.pooling_manifest).resolve()) if args.pooling_manifest else None,
        },
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "background_mode_resolution": mode_resolution,
        "roi": {
            "source": "Crab",
            "source_ra_deg": float(args.source_ra_deg),
            "source_dec_deg": float(args.source_dec_deg),
            "coordinate_system": "crab_tangent_plane_small_angle",
            "x_formula": "wrap(ra_mean_deg - source_ra_deg) * cos(source_dec_deg)",
            "y_formula": "dec_mean_deg - source_dec_deg",
            "rho_formula": "sqrt(x^2 + y^2)",
            "fiducial_radius_deg": float(args.roi_fiducial_deg),
            "edge_diagnostic_radius_deg": float(args.roi_edge_diagnostic_deg),
            "edge_margin_deg": float(args.roi_edge_margin_deg),
            "source_mask_min_radius_deg": float(args.roi_source_mask_deg),
            "source_mask_r_opt_factor": float(args.roi_source_mask_r_opt_factor),
            "stage_c_roi_coverage": mode_resolution.get("stage_c_roi_coverage", {}),
        },
        "source_masks": [
            {
                "name": source.name,
                "ra_deg": float(source.ra_deg),
                "dec_deg": float(source.dec_deg),
                "radius_deg": float(source.radius_deg),
                "enabled_for_full_field_direct_integration": bool(source.enabled),
                "enabled_for_crab_roi_local": bool(source.name == "Crab"),
                "roi_local_note": (
                    "used as the central training exclusion"
                    if source.name == "Crab"
                    else "not applied in the Crab-local tangent-plane baseline"
                ),
            }
            for source in default_source_masks()
        ],
        "grid": {
            "coordinate_system": "crab_tangent_plane",
            "x_edges_deg": [float(x_edges[0]), float(x_edges[-1])],
            "y_edges_deg": [float(y_edges[0]), float(y_edges[-1])],
            "grid_step_deg": float(args.roi_grid_step_deg),
            "offset_x_fraction": float(args.roi_grid_offset_x_fraction),
            "offset_y_fraction": float(args.roi_grid_offset_y_fraction),
            "offset_x_deg": float(args.roi_grid_offset_x_fraction) * float(args.roi_grid_step_deg),
            "offset_y_deg": float(args.roi_grid_offset_y_fraction) * float(args.roi_grid_step_deg),
            "physical_coverage_radius_deg": float(args.roi_edge_diagnostic_deg),
            "shape": [int(n_y), int(n_x)],
        },
        "background_model": {
            "background_mode": "crab_roi_local",
            "method": background_method,
            "fit_statistic": str(args.roi_fit_statistic),
            "pooling_manifest_sha256": (
                str(poisson_manifest.get("manifest_sha256")) if poisson_manifest is not None else None
            ),
            "analytic_bon_baseline_sha": (
                str(poisson_manifest.get("analytic_bon_baseline_sha")) if poisson_manifest is not None else None
            ),
            "background_form": "direct_expectation",
            "B_on_formula": b_on_formula,
            "on_aperture_integration": str(args.on_aperture_integration),
            "on_aperture_integration_note": (
                "B_on is the closed-form integral of the fitted polynomial over the true centered disk; "
                "background_map and on_mask remain pixelized morphology diagnostics only."
                if str(args.on_aperture_integration) == "analytic-quadratic"
                else "B_on is summed over pixels whose centers lie inside r_opt."
            ),
            "analytic_on_aperture_formula": (
                "scale/grid_step_deg^2 * (pi*r^2*c0 + pi*r^4*(c_xx+c_yy)/4)"
                if str(args.on_aperture_integration) == "analytic-quadratic"
                else None
            ),
            "analytic_positive_surface_required": (
                True if str(args.on_aperture_integration) == "analytic-quadratic" else None
            ),
            "analytic_required_cell_ids": (
                [int(cell.cell_id) for cell, required in zip(cells, analytic_required_mask) if bool(required)]
                if str(args.on_aperture_integration) == "analytic-quadratic"
                else []
            ),
            "excluded_fallback_cell_ids": (
                [
                    int(cell.cell_id)
                    for cell, method in zip(cells, annulus_diagnostics.get("on_aperture_integration_by_cell", []))
                    if str(method) == "pixel-center-excluded-fallback"
                ]
                if annulus_diagnostics
                else []
            ),
            "alpha_b": None,
            "N_off_b": None,
            "alpha_N_off_note": "ROI-local direct expectation outputs B_on,b; traditional alpha/N_off is not defined.",
            "on_region_radius_source": f"Stage B PSF NPZ r_opt_deg: {Path(args.psf_npz).resolve()}",
            "surface_order": int(args.roi_surface_order) if annulus_diagnostics else None,
            "surface_basis": ["1", "x", "y", "x^2", "x*y", "y^2"] if int(args.roi_surface_order) == 2 else ["1", "x", "y"],
            "annulus_normalize_surface": bool(args.annulus_normalize_surface) if annulus_diagnostics else None,
            "surface_normalization_formula": (
                "N_annulus_b / analytic_annulus_integral(q_b); no clipping and no second scale"
                if annulus_diagnostics and str(args.roi_fit_statistic) == "poisson"
                else "scale_b=sum_annulus(counts_b)/sum_annulus(max(B_raw_b,0)); B_final_b=scale_b*max(B_raw_b,0)"
                if annulus_diagnostics and bool(args.annulus_normalize_surface)
                else None
            ),
            "annulus_default_inner_deg": float(args.annulus_default_inner_deg) if annulus_diagnostics else None,
            "annulus_width_deg": float(args.annulus_width_deg) if annulus_diagnostics else None,
            "annulus_max_inner_deg": float(args.annulus_max_inner_deg) if annulus_diagnostics else None,
            "annulus_source_mask_min_deg": float(args.annulus_source_mask_min_deg) if annulus_diagnostics else None,
            "annulus_source_mask_r_opt_factor": (
                float(args.annulus_source_mask_r_opt_factor) if annulus_diagnostics else None
            ),
            "annulus_source_mask_margin_deg": (
                float(args.annulus_source_mask_margin_deg) if annulus_diagnostics else None
            ),
            "surface_condition_max": float(args.surface_condition_max) if annulus_diagnostics else None,
            "surface_min_training_pixels": int(args.surface_min_training_pixels) if annulus_diagnostics else None,
            "li_ma_applicable": False,
        },
        "processing": {
            "input_rows_scanned": int(scan.input_rows),
            "processed_batches": int(scan.processed_batches),
            "batch_size": int(args.batch_size),
            "max_batches": args.max_batches,
            "workers_requested": int(args.workers),
            "mjd_min": args.mjd_min,
            "mjd_max": args.mjd_max,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
        "quality": {
            "status": quality_status,
            "promotable": promotable,
            "reason": quality_reason,
            "warnings": severe_warnings,
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
    write_source_masks_csv(source_masks_csv_path, [SourceMask("Crab", float(args.source_ra_deg), float(args.source_dec_deg), float(args.roi_source_mask_deg))])
    write_summary_md(summary_md_path, metadata, rows)
    write_json(metadata_path, metadata)

    if not args.no_promote_current and promotable:
        promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        write_json(metadata_path, metadata)
    elif not args.no_promote_current:
        metadata["promotion"]["status"] = "blocked_quality_gate"  # type: ignore[index]
        metadata["promotion"]["reason"] = quality_reason  # type: ignore[index]
        write_json(metadata_path, metadata)
    else:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]
        write_json(metadata_path, metadata)

    print(f"Wrote {npz_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)
    print(f"Wrote {summary_md_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    if not args.no_promote_current and promotable:
        print(f"Promoted current Stage D output to {output_root / 'current'}", flush=True)
    elif not args.no_promote_current:
        print(f"Stage D promotion blocked: {quality_reason}", flush=True)


def main() -> None:
    args = parse_args()
    validate_roi_poisson_args(args)
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

    cells = load_cells(selection_csv, included_only=bool(args.included_only))
    if str(args.roi_fit_statistic) == "poisson":
        cells = filter_cells_to_manifest(
            cells,
            load_and_validate_poisson_manifest(Path(args.pooling_manifest).resolve()),
        )
    sources = default_source_masks()
    psf_by_cell = load_psf_by_cell(psf_npz, cells)
    r_opt_deg = np.asarray([psf_by_cell[cell.cell_id]["r_opt_deg"] for cell in cells], dtype=np.float32)
    sigma_deg = np.asarray([psf_by_cell[cell.cell_id]["sigma_deg"] for cell in cells], dtype=np.float32)
    containment_r_opt = np.asarray([psf_by_cell[cell.cell_id]["containment_r_opt"] for cell in cells], dtype=np.float32)

    background_mode, mode_resolution = resolve_background_mode(args, stage_c_metadata)
    if background_mode == "crab_roi_local":
        run_crab_roi_local_background(
            args=args,
            start_time=start,
            stage_c_dir=stage_c_dir,
            obs_events_dir=obs_events_dir,
            stage_c_metadata_path=stage_c_metadata_path,
            stage_c_metadata=stage_c_metadata,
            source_files_csv=source_files_csv,
            psf_npz=psf_npz,
            selection_csv=selection_csv,
            output_root=output_root,
            run_dir=run_dir,
            run_id=run_id,
            cells=cells,
            r_opt_deg=r_opt_deg,
            sigma_deg=sigma_deg,
            containment_r_opt=containment_r_opt,
            mode_resolution=mode_resolution,
        )
        return
    if background_mode != "full_field_direct_integration":
        raise ValueError(f"Unsupported resolved background mode: {background_mode}")

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
        mjd_min=args.mjd_min,
        mjd_max=args.mjd_max,
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
    source_masks_csv_path = run_dir / "source_masks.csv"

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
        "description": "Stage D full-field direct-integration background for configured (Nhit, predicted logE) Crab SED cells.",
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
        "background_mode_resolution": mode_resolution,
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
            "background_mode": "full_field_direct_integration",
            "method": "direct_integration_acceptance_times_rate",
            "background_form": "direct_expectation",
            "B_on_formula": "sum_time N_cell,time * sum_grid acceptance_cell,grid * I(grid within Crab aperture at time)",
            "alpha_b": None,
            "N_off_b": None,
            "alpha_N_off_note": "Stage D outputs direct background expectation B_on,b; traditional alpha/N_off is not defined.",
            "on_region_radius_source": f"Stage B PSF NPZ r_opt_deg: {psf_npz}",
            "mask_exposure_correction": "counts_masked / sum_time(N_cell,time * source_mask_available_time,grid), normalized over visible grid",
        },
        "theta_coordinate_check": scan.theta_check,
        "processing": {
            "input_rows_scanned": int(scan.input_rows),
            "processed_batches": int(scan.processed_batches),
            "batch_size": int(args.batch_size),
            "max_batches": args.max_batches,
            "workers_requested": int(args.workers),
            "mjd_min": args.mjd_min,
            "mjd_max": args.mjd_max,
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
