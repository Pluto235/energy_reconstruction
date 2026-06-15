#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OBS_ROOT = "/mnt/mydisk/WCDA_observation_eval"
DEFAULT_TIME_ROOT = "/mnt/mydisk/WCDA_observation_eval/recovered_time"
DEFAULT_CELL_SELECTION = "apply/config/cell_selection_v1.csv"
DEFAULT_OUTPUT_DIR = "apply/output/stage_c"
DEFAULT_TREE_NAME = "t_eventout"
DEFAULT_TIME_TREE_NAME = "t_recovered_time"
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

CUTFLOW_KEYS = [
    "input_entries",
    "after_match_status",
    "after_pincness",
    "after_fitstat",
    "after_theta",
    "after_dcedge",
    "after_finite",
    "after_cell_selection",
]

ROI_SOURCE_NAME = "Crab"
ROI_RA_DEG = 83.63
ROI_DEC_DEG = 22.01
ROI_COORDINATE = "tangent_plane_small_angle"
ROI_HIST_MIN_DEG = 0.0
ROI_HIST_MAX_DEG = 12.0
ROI_HIST_BIN_WIDTH_DEG = 0.1
ROI_COUNT_RADII_DEG = [2.0, 4.0, 5.5, 6.0, 6.5, 8.0, 10.0]
ROI_FIDUCIAL_RADIUS_RECOMMENDATION_DEG = 6.0
XY_QUANTILE_BIN_WIDTH_DEG = 0.02
XY_QUANTILES = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]


@dataclass(frozen=True)
class CellSpec:
    index: int
    cell_id: int
    nhit_bin: str
    predE_bin: str
    nhit_low: Optional[float]
    nhit_high: Optional[float]
    pred_low: Optional[float]
    pred_high: Optional[float]
    mc_count: int
    selection_version: str
    selection_reason: str


@dataclass(frozen=True)
class InputFileSpec:
    source_file_id: int
    obs_path: str
    time_path: str
    relative_path: str
    yyyymm: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage C observation reduction: eval ROOT + recovered time friend tree to configured-cell parquet."
    )
    parser.add_argument("--obs-root", type=str, default=DEFAULT_OBS_ROOT)
    parser.add_argument("--time-root", type=str, default=DEFAULT_TIME_ROOT)
    parser.add_argument("--cell-selection-csv", type=str, default=DEFAULT_CELL_SELECTION)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run directory name under <output-dir>/runs. Defaults to the Slurm job id or a timestamp.",
    )
    parser.add_argument(
        "--no-promote-current",
        action="store_true",
        default=False,
        help="Do not update <output-dir>/current and <output-dir>/latest after a successful run.",
    )
    parser.add_argument(
        "--overwrite-run-dir",
        action="store_true",
        default=False,
        help="Delete and recreate an existing run directory. Intended for smoke tests only.",
    )
    parser.add_argument("--tree-name", type=str, default=DEFAULT_TREE_NAME)
    parser.add_argument("--time-tree-name", type=str, default=DEFAULT_TIME_TREE_NAME)
    parser.add_argument("--file-glob", type=str, default="Esg*.root")
    parser.add_argument("--day-prefix", type=str, default=None, help="Only process MMDD directories starting with this prefix.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--entries-per-chunk", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--allow-missing-time", action="store_true", default=False)
    parser.add_argument(
        "--allow-entry-mismatch",
        action="store_true",
        default=False,
        help="Use the common entry prefix instead of failing when eval and time trees differ.",
    )

    parser.add_argument("--match-status-equals", type=int, default=0)
    parser.add_argument("--cut-pinc-max", type=float, default=1.1)
    parser.add_argument("--cut-fitstat-equals", type=int, default=0)
    parser.add_argument("--cut-theta-max-deg", type=float, default=50.0)
    parser.add_argument("--cut-dcedge-min", type=float, default=20.0)
    parser.add_argument("--gap-threshold-sec", type=float, default=60.0)
    parser.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    parser.add_argument("--roi-batch-size", type=int, default=1000000)
    parser.add_argument(
        "--skip-roi-coverage",
        action="store_true",
        default=False,
        help="Skip Crab-centered ROI coverage diagnostics. Intended only for debugging.",
    )
    parser.add_argument(
        "--update-existing-run",
        type=str,
        default=None,
        help="Only add/update metadata and summary diagnostics for an existing Stage C run directory.",
    )
    return parser.parse_args()


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
            nhit_low, nhit_high = parse_interval(row["nhit_bin"])
            pred_low, pred_high = parse_interval(row["predE_bin"])
            cells.append(
                CellSpec(
                    index=idx,
                    cell_id=int(row.get("cell_id") or (idx + 1)),
                    nhit_bin=row["nhit_bin"],
                    predE_bin=row["predE_bin"],
                    nhit_low=nhit_low,
                    nhit_high=nhit_high,
                    pred_low=pred_low,
                    pred_high=pred_high,
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
            nhit_low=cell.nhit_low,
            nhit_high=cell.nhit_high,
            pred_low=cell.pred_low,
            pred_high=cell.pred_high,
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
            raise FileExistsError(f"Stage C run directory already exists: {run_dir}")
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


def month_from_filename(path: Path) -> str:
    match = re.search(r"Esg(\d{8})(?:_|\.root)", path.name)
    if not match:
        raise ValueError(f"Cannot derive YYYYMM from observation file name: {path}")
    return match.group(1)[:6]


def discover_observation_files(
    obs_root: Path,
    time_root: Path,
    file_glob: str,
    day_prefix: Optional[str],
    max_files: Optional[int],
) -> List[InputFileSpec]:
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

    specs: List[InputFileSpec] = []
    for source_file_id, obs_file in enumerate(files):
        rel = observation_relative_path(obs_file, obs_root)
        specs.append(
            InputFileSpec(
                source_file_id=source_file_id,
                obs_path=str(obs_file),
                time_path=str(time_path_for(obs_file, obs_root, time_root)),
                relative_path=str(rel),
                yyyymm=month_from_filename(obs_file),
            )
        )
    return specs


def make_parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("ra_mean_deg", pa.float32()),
            pa.field("dec_mean_deg", pa.float32()),
            pa.field("mjd", pa.float64()),
            pa.field("theta", pa.float32()),
            pa.field("phi", pa.float32()),
            pa.field("cell_id", pa.int16()),
            pa.field("nv", pa.int32()),
            pa.field("ml_logE_pred", pa.float32()),
            pa.field("dcedge", pa.float32()),
            pa.field("source_file_id", pa.int32()),
            pa.field("entry", pa.int64()),
        ]
    )


def open_tree(root_file, path: Path, tree_name: str):
    if tree_name in root_file:
        return root_file[tree_name]
    versioned = f"{tree_name};1"
    if versioned in root_file:
        return root_file[versioned]
    raise KeyError(f"{path} does not contain tree {tree_name!r}")


def require_branches(tree, path: Path, branches: Sequence[str]) -> None:
    available = set(tree.keys())
    missing = [name for name in branches if name not in available]
    if missing:
        raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")


def in_interval(values: np.ndarray, low: Optional[float], high: Optional[float]) -> np.ndarray:
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values >= float(low)
    if high is not None:
        mask &= values < float(high)
    return mask


def assign_cell_ids(nv: np.ndarray, loge_pred: np.ndarray, cells: Sequence[CellSpec]) -> np.ndarray:
    cell_ids = np.zeros(nv.shape, dtype=np.int16)
    for cell in cells:
        mask = in_interval(nv, cell.nhit_low, cell.nhit_high) & in_interval(loge_pred, cell.pred_low, cell.pred_high)
        cell_ids[mask] = np.int16(cell.cell_id)
    return cell_ids


def update_mjd_stats(stats: Dict[str, object], mjd_values: np.ndarray, gap_threshold_days: float) -> None:
    finite_mjd = np.asarray(mjd_values[np.isfinite(mjd_values)], dtype=np.float64)
    if finite_mjd.size == 0:
        return

    current_min = float(finite_mjd.min())
    current_max = float(finite_mjd.max())
    stats["mjd_min"] = current_min if stats["mjd_min"] is None else min(float(stats["mjd_min"]), current_min)
    stats["mjd_max"] = current_max if stats["mjd_max"] is None else max(float(stats["mjd_max"]), current_max)

    previous = stats["last_mjd"]
    diffs = np.diff(finite_mjd)
    if previous is not None:
        first_gap = finite_mjd[0] - float(previous)
        diffs = np.concatenate([np.asarray([first_gap], dtype=np.float64), diffs])
    positive_gaps = diffs[diffs > gap_threshold_days]
    stats["gap_count"] = int(stats["gap_count"]) + int(positive_gaps.size)
    stats["gap_days"] = float(stats["gap_days"]) + float(positive_gaps.sum())
    stats["last_mjd"] = float(finite_mjd[-1])


def mjd_to_yyyymm(mjd: Optional[float]) -> Optional[str]:
    if mjd is None or not np.isfinite(mjd):
        return None
    dt = MJD_EPOCH + timedelta(days=float(mjd))
    return f"{dt.year:04d}{dt.month:02d}"


def table_from_selected(
    event_arrays: Dict[str, np.ndarray],
    time_arrays: Dict[str, np.ndarray],
    selected_mask: np.ndarray,
    selected_cell_ids: np.ndarray,
    source_file_id: int,
    start_entry: int,
) -> pa.Table:
    selected_entries = np.arange(start_entry, start_entry + selected_mask.size, dtype=np.int64)[selected_mask]
    count = int(selected_mask.sum())
    schema = make_parquet_schema()
    arrays = [
        pa.array(np.asarray(time_arrays["ra_mean_deg"][selected_mask], dtype=np.float32), type=pa.float32()),
        pa.array(np.asarray(time_arrays["dec_mean_deg"][selected_mask], dtype=np.float32), type=pa.float32()),
        pa.array(np.asarray(time_arrays["mjd"][selected_mask], dtype=np.float64), type=pa.float64()),
        pa.array(np.asarray(event_arrays["theta"][selected_mask], dtype=np.float32), type=pa.float32()),
        pa.array(np.asarray(event_arrays["phi"][selected_mask], dtype=np.float32), type=pa.float32()),
        pa.array(np.asarray(selected_cell_ids[selected_mask], dtype=np.int16), type=pa.int16()),
        pa.array(np.asarray(event_arrays["nv"][selected_mask], dtype=np.int32), type=pa.int32()),
        pa.array(np.asarray(event_arrays["ml_logE_pred"][selected_mask], dtype=np.float32), type=pa.float32()),
        pa.array(np.asarray(event_arrays["dcedge"][selected_mask], dtype=np.float32), type=pa.float32()),
        pa.array(np.full(count, int(source_file_id), dtype=np.int32), type=pa.int32()),
        pa.array(selected_entries, type=pa.int64()),
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def init_cutflow() -> Dict[str, int]:
    return {key: 0 for key in CUTFLOW_KEYS}


def finite_cell_lower_bounds(cells: Sequence[CellSpec]) -> Dict[str, Optional[float]]:
    nhit_lows = [cell.nhit_low for cell in cells if cell.nhit_low is not None]
    pred_lows = [cell.pred_low for cell in cells if cell.pred_low is not None]
    return {
        "nhit_min_inclusive": min(nhit_lows) if nhit_lows else None,
        "predE_min_inclusive": min(pred_lows) if pred_lows else None,
    }


def process_input_file(task: Tuple[InputFileSpec, Dict[str, object]]) -> Dict[str, object]:
    spec, config = task
    obs_path = Path(spec.obs_path)
    time_path = Path(spec.time_path)
    run_dir = Path(str(config["run_dir"]))
    cells: Sequence[CellSpec] = config["cells"]  # type: ignore[assignment]
    compression = None if str(config["compression"]) == "none" else str(config["compression"])
    lower_bounds = finite_cell_lower_bounds(cells)
    nhit_min = lower_bounds["nhit_min_inclusive"]
    pred_min = lower_bounds["predE_min_inclusive"]

    result: Dict[str, object] = {
        "source_file_id": int(spec.source_file_id),
        "status": "processed",
        "obs_path": str(obs_path),
        "time_path": str(time_path),
        "relative_path": spec.relative_path,
        "yyyymm": spec.yyyymm,
        "event_entries": None,
        "time_entries": None,
        "common_entries": 0,
        "entry_mismatch": False,
        "selected_rows": 0,
        "parquet_path": None,
        "parquet_relative_path": None,
        "cutflow": init_cutflow(),
        "cell_counts": [0 for _ in cells],
        "nhit_below_candidate_min": 0,
        "predE_below_candidate_min": 0,
        "out_of_ledger_after_finite": 0,
        "matched_mjd_min": None,
        "matched_mjd_max": None,
        "matched_mjd_start_utc_yyyymm": None,
        "matched_mjd_end_utc_yyyymm": None,
        "matched_span_seconds": 0.0,
        "matched_gap_count": 0,
        "matched_gap_seconds": 0.0,
        "rough_live_time_seconds": 0.0,
        "selected_mjd_min": None,
        "selected_mjd_max": None,
    }

    if not time_path.exists():
        if bool(config["allow_missing_time"]):
            result["status"] = "missing_time_skipped"
            return result
        raise FileNotFoundError(f"Missing recovered-time friend tree for {obs_path}: {time_path}")

    event_branches = ["nv", "ml_logE_pred", "theta", "phi", "dcedge", "pincness", "fitstat"]
    time_branches = ["mjd", "ra_mean_deg", "dec_mean_deg", "match_status"]
    theta_cut_rad = math.radians(float(config["cut_theta_max_deg"]))
    gap_threshold_days = float(config["gap_threshold_sec"]) / 86400.0
    entries_per_chunk = int(config["entries_per_chunk"])
    partition_dir = run_dir / "obs_events" / f"yyyymm={spec.yyyymm}"
    final_path = partition_dir / f"part-{spec.source_file_id:06d}.parquet"
    tmp_path = partition_dir / f".part-{spec.source_file_id:06d}.{os.getpid()}.tmp.parquet"

    writer: Optional[pq.ParquetWriter] = None
    matched_mjd_stats: Dict[str, object] = {"mjd_min": None, "mjd_max": None, "last_mjd": None, "gap_count": 0, "gap_days": 0.0}
    selected_mjd_stats: Dict[str, object] = {"mjd_min": None, "mjd_max": None, "last_mjd": None, "gap_count": 0, "gap_days": 0.0}

    try:
        with uproot.open(obs_path) as obs_file, uproot.open(time_path) as recovered_time_file:
            event_tree = open_tree(obs_file, obs_path, str(config["tree_name"]))
            time_tree = open_tree(recovered_time_file, time_path, str(config["time_tree_name"]))
            require_branches(event_tree, obs_path, event_branches)
            require_branches(time_tree, time_path, time_branches)

            n_event = int(event_tree.num_entries)
            n_time = int(time_tree.num_entries)
            result["event_entries"] = n_event
            result["time_entries"] = n_time
            if n_event != n_time:
                result["entry_mismatch"] = True
                if not bool(config["allow_entry_mismatch"]):
                    raise ValueError(
                        "Entry mismatch: "
                        f"obs_file={obs_path}, time_file={time_path}, event_entries={n_event}, time_entries={n_time}"
                    )

            common_entries = min(n_event, n_time)
            result["common_entries"] = common_entries
            for start in range(0, common_entries, entries_per_chunk):
                stop = min(start + entries_per_chunk, common_entries)
                event_arrays = event_tree.arrays(event_branches, entry_start=start, entry_stop=stop, library="np")
                time_arrays = time_tree.arrays(time_branches, entry_start=start, entry_stop=stop, library="np")
                n_chunk = stop - start

                cutflow = result["cutflow"]
                cutflow["input_entries"] += int(n_chunk)  # type: ignore[index]

                match_mask = np.asarray(time_arrays["match_status"] == int(config["match_status_equals"]))
                update_mjd_stats(matched_mjd_stats, np.asarray(time_arrays["mjd"])[match_mask], gap_threshold_days)
                cutflow["after_match_status"] += int(match_mask.sum())  # type: ignore[index]

                pinc_mask = match_mask & np.asarray(event_arrays["pincness"] < float(config["cut_pinc_max"]))
                cutflow["after_pincness"] += int(pinc_mask.sum())  # type: ignore[index]

                fit_mask = pinc_mask & np.asarray(event_arrays["fitstat"] == int(config["cut_fitstat_equals"]))
                cutflow["after_fitstat"] += int(fit_mask.sum())  # type: ignore[index]

                theta_mask = fit_mask & np.asarray(event_arrays["theta"] < theta_cut_rad)
                cutflow["after_theta"] += int(theta_mask.sum())  # type: ignore[index]

                dcedge_mask = theta_mask & np.asarray(event_arrays["dcedge"] > float(config["cut_dcedge_min"]))
                cutflow["after_dcedge"] += int(dcedge_mask.sum())  # type: ignore[index]

                nv = np.asarray(event_arrays["nv"], dtype=np.float64)
                loge_pred = np.asarray(event_arrays["ml_logE_pred"], dtype=np.float64)
                finite_mask = (
                    dcedge_mask
                    & np.isfinite(nv)
                    & np.isfinite(loge_pred)
                    & np.isfinite(np.asarray(event_arrays["theta"], dtype=np.float64))
                    & np.isfinite(np.asarray(event_arrays["phi"], dtype=np.float64))
                    & np.isfinite(np.asarray(event_arrays["dcedge"], dtype=np.float64))
                    & np.isfinite(np.asarray(time_arrays["mjd"], dtype=np.float64))
                    & np.isfinite(np.asarray(time_arrays["ra_mean_deg"], dtype=np.float64))
                    & np.isfinite(np.asarray(time_arrays["dec_mean_deg"], dtype=np.float64))
                )
                cutflow["after_finite"] += int(finite_mask.sum())  # type: ignore[index]

                cell_ids = assign_cell_ids(nv, loge_pred, cells)
                selected_mask = finite_mask & (cell_ids > 0)
                selected_count = int(selected_mask.sum())
                cutflow["after_cell_selection"] += selected_count  # type: ignore[index]
                result["out_of_ledger_after_finite"] = int(result["out_of_ledger_after_finite"]) + int(
                    np.count_nonzero(finite_mask & (cell_ids <= 0))
                )
                if nhit_min is not None:
                    result["nhit_below_candidate_min"] = int(result["nhit_below_candidate_min"]) + int(
                        np.count_nonzero(finite_mask & (nv < float(nhit_min)))
                    )
                if pred_min is not None:
                    result["predE_below_candidate_min"] = int(result["predE_below_candidate_min"]) + int(
                        np.count_nonzero(finite_mask & (loge_pred < float(pred_min)))
                    )
                if selected_count == 0:
                    continue

                result["selected_rows"] = int(result["selected_rows"]) + selected_count
                update_mjd_stats(selected_mjd_stats, np.asarray(time_arrays["mjd"])[selected_mask], gap_threshold_days)

                for cell in cells:
                    result["cell_counts"][cell.index] += int(np.count_nonzero(selected_mask & (cell_ids == cell.cell_id)))  # type: ignore[index]

                if writer is None:
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(tmp_path, make_parquet_schema(), compression=compression)
                writer.write_table(table_from_selected(event_arrays, time_arrays, selected_mask, cell_ids, spec.source_file_id, start))
    finally:
        if writer is not None:
            writer.close()

    if int(result["selected_rows"]) > 0:
        tmp_path.replace(final_path)
        result["parquet_path"] = str(final_path)
        result["parquet_relative_path"] = str(final_path.relative_to(run_dir))
    elif tmp_path.exists():
        tmp_path.unlink()

    matched_min = matched_mjd_stats["mjd_min"]
    matched_max = matched_mjd_stats["mjd_max"]
    selected_min = selected_mjd_stats["mjd_min"]
    selected_max = selected_mjd_stats["mjd_max"]

    matched_span_days = 0.0 if matched_min is None or matched_max is None else max(0.0, float(matched_max) - float(matched_min))
    matched_gap_days = float(matched_mjd_stats["gap_days"])
    result["matched_mjd_min"] = matched_min
    result["matched_mjd_max"] = matched_max
    result["matched_mjd_start_utc_yyyymm"] = mjd_to_yyyymm(matched_min)  # type: ignore[arg-type]
    result["matched_mjd_end_utc_yyyymm"] = mjd_to_yyyymm(matched_max)  # type: ignore[arg-type]
    result["matched_span_seconds"] = matched_span_days * 86400.0
    result["matched_gap_count"] = int(matched_mjd_stats["gap_count"])
    result["matched_gap_seconds"] = matched_gap_days * 86400.0
    result["rough_live_time_seconds"] = max(0.0, (matched_span_days - matched_gap_days) * 86400.0)
    result["selected_mjd_min"] = selected_min
    result["selected_mjd_max"] = selected_max
    return result


def merge_cutflow(results: Iterable[Dict[str, object]]) -> Dict[str, int]:
    merged = init_cutflow()
    for result in results:
        cutflow = result["cutflow"]
        for key in CUTFLOW_KEYS:
            merged[key] += int(cutflow[key])  # type: ignore[index]
    return merged


def merge_cell_counts(results: Iterable[Dict[str, object]], n_cells: int) -> List[int]:
    counts = np.zeros(n_cells, dtype=np.int64)
    for result in results:
        counts += np.asarray(result["cell_counts"], dtype=np.int64)
    return [int(value) for value in counts]


def group_by_month(results: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["yyyymm"]), []).append(result)
    return grouped


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.12g}"
    return value


def write_source_files_csv(path: Path, results: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "source_file_id",
        "status",
        "yyyymm",
        "relative_path",
        "obs_path",
        "time_path",
        "event_entries",
        "time_entries",
        "common_entries",
        "entry_mismatch",
        "selected_rows",
        "parquet_relative_path",
        "matched_mjd_min",
        "matched_mjd_max",
        "matched_mjd_start_utc_yyyymm",
        "matched_mjd_end_utc_yyyymm",
        "matched_span_seconds",
        "matched_gap_count",
        "matched_gap_seconds",
        "rough_live_time_seconds",
        "selected_mjd_min",
        "selected_mjd_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: csv_value(result.get(key)) for key in fieldnames})


def write_cutflow_csv(path: Path, results: Sequence[Dict[str, object]]) -> None:
    grouped = group_by_month(results)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "yyyymm", "source_file_id", "step", "count"])
        writer.writeheader()
        global_cutflow = merge_cutflow(results)
        for key in CUTFLOW_KEYS:
            writer.writerow({"scope": "global", "yyyymm": "", "source_file_id": "", "step": key, "count": global_cutflow[key]})
        for yyyymm, month_results in sorted(grouped.items()):
            month_cutflow = merge_cutflow(month_results)
            for key in CUTFLOW_KEYS:
                writer.writerow({"scope": "month", "yyyymm": yyyymm, "source_file_id": "", "step": key, "count": month_cutflow[key]})
        for result in results:
            cutflow = result["cutflow"]
            for key in CUTFLOW_KEYS:
                writer.writerow(
                    {
                        "scope": "file",
                        "yyyymm": result["yyyymm"],
                        "source_file_id": result["source_file_id"],
                        "step": key,
                        "count": cutflow[key],  # type: ignore[index]
                    }
                )


def write_cell_counts_csv(path: Path, results: Sequence[Dict[str, object]], cells: Sequence[CellSpec]) -> None:
    grouped = group_by_month(results)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scope", "yyyymm", "source_file_id", "cell_id", "nhit_bin", "predE_bin", "count"],
        )
        writer.writeheader()

        global_counts = merge_cell_counts(results, len(cells))
        for cell, count in zip(cells, global_counts):
            writer.writerow(
                {
                    "scope": "global",
                    "yyyymm": "",
                    "source_file_id": "",
                    "cell_id": cell.cell_id,
                    "nhit_bin": cell.nhit_bin,
                    "predE_bin": cell.predE_bin,
                    "count": count,
                }
            )
        for yyyymm, month_results in sorted(grouped.items()):
            month_counts = merge_cell_counts(month_results, len(cells))
            for cell, count in zip(cells, month_counts):
                writer.writerow(
                    {
                        "scope": "month",
                        "yyyymm": yyyymm,
                        "source_file_id": "",
                        "cell_id": cell.cell_id,
                        "nhit_bin": cell.nhit_bin,
                        "predE_bin": cell.predE_bin,
                        "count": count,
                    }
                )
        for result in results:
            counts = result["cell_counts"]
            for cell in cells:
                writer.writerow(
                    {
                        "scope": "file",
                        "yyyymm": result["yyyymm"],
                        "source_file_id": result["source_file_id"],
                        "cell_id": cell.cell_id,
                        "nhit_bin": cell.nhit_bin,
                        "predE_bin": cell.predE_bin,
                        "count": counts[cell.index],  # type: ignore[index]
                    }
                )


def finite_min(values: Sequence[object]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return min(finite_values) if finite_values else None


def finite_max(values: Sequence[object]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return max(finite_values) if finite_values else None


def build_manifest(run_dir: Path, results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    parquet_results = [result for result in results if result.get("parquet_relative_path")]
    partitions: Dict[str, Dict[str, object]] = {}
    for result in parquet_results:
        yyyymm = str(result["yyyymm"])
        item = partitions.setdefault(yyyymm, {"yyyymm": yyyymm, "files": 0, "rows": 0})
        item["files"] = int(item["files"]) + 1
        item["rows"] = int(item["rows"]) + int(result["selected_rows"])

    return {
        "dataset_path": str(run_dir / "obs_events"),
        "dataset_relative_path": "obs_events",
        "partitioning": "hive",
        "partition_columns": ["yyyymm"],
        "partition_column_types": {"yyyymm": "filename-derived YYYYMM label"},
        "schema": [{"name": field.name, "type": str(field.type)} for field in make_parquet_schema()],
        "partitions": [partitions[key] for key in sorted(partitions)],
        "files": [
            {
                "source_file_id": int(result["source_file_id"]),
                "yyyymm": result["yyyymm"],
                "rows": int(result["selected_rows"]),
                "path": str(result["parquet_path"]),
                "relative_path": str(result["parquet_relative_path"]),
            }
            for result in parquet_results
        ],
    }


def radius_key(radius_deg: float) -> str:
    label = f"{radius_deg:g}".replace(".", "p")
    return f"rho_lt_{label}_deg"


def wrap_ra_delta_deg(ra_deg: np.ndarray, center_ra_deg: float) -> np.ndarray:
    return ((ra_deg - float(center_ra_deg) + 180.0) % 360.0) - 180.0


def roi_xy_rho(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_deg = wrap_ra_delta_deg(ra_deg, ROI_RA_DEG) * math.cos(math.radians(ROI_DEC_DEG))
    y_deg = dec_deg - ROI_DEC_DEG
    rho_deg = np.sqrt((x_deg * x_deg) + (y_deg * y_deg))
    return x_deg, y_deg, rho_deg


def make_roi_hist_edges() -> np.ndarray:
    bins = int(round((ROI_HIST_MAX_DEG - ROI_HIST_MIN_DEG) / ROI_HIST_BIN_WIDTH_DEG))
    return np.linspace(ROI_HIST_MIN_DEG, ROI_HIST_MAX_DEG, bins + 1, dtype=np.float64)


def iter_roi_record_batches(run_dir: Path, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dataset_path = run_dir / "obs_events"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Stage C parquet dataset does not exist: {dataset_path}")

    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    scanner = dataset.scanner(columns=["ra_mean_deg", "dec_mean_deg", "cell_id"], batch_size=batch_size, use_threads=True)
    for batch in scanner.to_batches():
        names = batch.schema.names
        ra = np.asarray(batch.column(names.index("ra_mean_deg")).to_numpy(zero_copy_only=False), dtype=np.float64)
        dec = np.asarray(batch.column(names.index("dec_mean_deg")).to_numpy(zero_copy_only=False), dtype=np.float64)
        cell_id = np.asarray(batch.column(names.index("cell_id")).to_numpy(zero_copy_only=False), dtype=np.int16)
        yield ra, dec, cell_id


def make_xy_quantile_edges(min_value: float, max_value: float) -> np.ndarray:
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        raise ValueError("Cannot build ROI x/y quantile bins from non-finite bounds")
    if max_value <= min_value:
        min_value -= 0.5
        max_value += 0.5
    bins = max(1, int(math.ceil((max_value - min_value) / XY_QUANTILE_BIN_WIDTH_DEG)))
    return np.linspace(min_value, max_value, bins + 1, dtype=np.float64)


def quantiles_from_histogram(counts: np.ndarray, edges: np.ndarray, quantiles: Sequence[float]) -> Dict[str, Optional[float]]:
    total = int(counts.sum())
    if total <= 0:
        return {f"q{quantile:g}": None for quantile in quantiles}

    cumulative = np.cumsum(counts, dtype=np.int64)
    values: Dict[str, Optional[float]] = {}
    for quantile in quantiles:
        target = max(1, int(math.ceil(float(quantile) * total)))
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(idx, len(counts) - 1)
        previous = int(cumulative[idx - 1]) if idx > 0 else 0
        bin_count = int(counts[idx])
        if bin_count <= 0:
            values[f"q{quantile:g}"] = float(edges[idx])
            continue
        fraction = (target - previous) / float(bin_count)
        values[f"q{quantile:g}"] = float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))
    return values


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    window = max(1, int(window))
    if window == 1:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64)
    numerator = np.convolve(values.astype(np.float64), kernel, mode="same")
    denominator = np.convolve(np.ones(values.size, dtype=np.float64), kernel, mode="same")
    return numerator / np.maximum(denominator, 1.0)


def percentile_radius_from_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    percentile: float,
) -> Optional[float]:
    total = int(counts.sum())
    if total <= 0:
        return None
    cumulative = np.cumsum(counts, dtype=np.int64)
    target = max(1, int(math.ceil(float(percentile) * total)))
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(idx, len(counts) - 1)
    previous = int(cumulative[idx - 1]) if idx > 0 else 0
    bin_count = int(counts[idx])
    if bin_count <= 0:
        return float(edges[idx])
    fraction = (target - previous) / float(bin_count)
    return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def estimate_roi_edge(
    counts: np.ndarray,
    edges: np.ndarray,
    density_per_sqdeg: np.ndarray,
) -> Dict[str, object]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth_density = moving_average(density_per_sqdeg, window=7)
    cdf_99 = percentile_radius_from_histogram(counts, edges, 0.99)
    cdf_995 = percentile_radius_from_histogram(counts, edges, 0.995)

    plateau_mask = (centers >= 2.0) & (centers <= 5.5) & np.isfinite(smooth_density) & (smooth_density > 0)
    plateau = float(np.median(smooth_density[plateau_mask])) if np.any(plateau_mask) else None

    edge_radius: Optional[float] = None
    method = "no_clear_density_edge; cdf99_and_cdf99p5_reported"
    if plateau is not None and plateau > 0.0:
        search_mask = (centers >= 5.5) & (centers <= ROI_HIST_MAX_DEG) & np.isfinite(smooth_density)
        search_indices = np.flatnonzero(search_mask)
        below = smooth_density < (0.5 * plateau)
        for idx in search_indices:
            stop = min(idx + 3, below.size)
            if stop - idx >= 3 and bool(np.all(below[idx:stop])):
                edge_radius = float(centers[idx])
                method = "smoothed_annular_density_below_half_plateau_for_3_bins"
                break

    if edge_radius is None and cdf_995 is not None:
        edge_radius = float(cdf_995)
        method = "rho_cumulative_99p5_within_12deg_proxy_no_clear_density_edge"

    return {
        "edge_radius_estimate_deg": edge_radius,
        "edge_radius_method": method,
        "plateau_density_per_sqdeg": plateau,
        "rho_percentile_99_deg_within_12deg": cdf_99,
        "rho_percentile_99p5_deg_within_12deg": cdf_995,
        "smoothed_density_per_sqdeg": [float(value) for value in smooth_density],
    }


def build_roi_warnings(
    total_rows: int,
    counts_within: Dict[str, int],
    centers: np.ndarray,
    smooth_density: np.ndarray,
    plateau_density: Optional[float],
    by_cell_within: np.ndarray,
) -> List[str]:
    warnings: List[str] = []
    rho6 = int(counts_within[radius_key(6.0)])
    rho8 = int(counts_within[radius_key(8.0)])
    rho10 = int(counts_within[radius_key(10.0)])

    if total_rows > 0 and (rho6 / float(total_rows)) < 0.001:
        warnings.append("rho<6 deg events are below 0.1% of all selected Stage C events; Crab is a small local ROI in this all-sky dataset.")

    if plateau_density is not None and plateau_density > 0.0:
        pre6_mask = (centers >= 5.5) & (centers < 6.0) & np.isfinite(smooth_density)
        if np.any(pre6_mask):
            pre6_density = float(np.median(smooth_density[pre6_mask]))
            if pre6_density < 0.5 * float(plateau_density):
                warnings.append("Crab-centered rho profile is already below 50% of its 2-5.5 deg plateau before rho=6 deg.")

    if rho10 > 0 and ((rho10 - rho8) / float(rho10)) < 0.05:
        warnings.append("rho<8 and rho<10 counts differ by less than 5%; available Crab-centered coverage may be below 8 deg.")

    idx6 = ROI_COUNT_RADII_DEG.index(6.0)
    idx10 = ROI_COUNT_RADII_DEG.index(10.0)
    denominators = by_cell_within[:, idx10]
    usable = denominators >= 1000
    if np.count_nonzero(usable) >= 2:
        ratios = by_cell_within[usable, idx6].astype(np.float64) / denominators[usable].astype(np.float64)
        min_ratio = float(np.min(ratios))
        max_ratio = float(np.max(ratios))
        if min_ratio > 0.0 and (max_ratio / min_ratio) > 2.0:
            warnings.append("Per-cell rho<6/rho<10 ratios differ by more than a factor of 2 across cells with at least 1000 rho<10 events.")
        elif (max_ratio - min_ratio) > 0.25:
            warnings.append("Per-cell rho<6/rho<10 ratios span more than 0.25 across cells with at least 1000 rho<10 events.")

    return warnings


def write_roi_coverage_csv(
    path: Path,
    edges: np.ndarray,
    counts: np.ndarray,
    density_per_sqdeg: np.ndarray,
) -> None:
    cumulative = np.cumsum(counts, dtype=np.int64)
    total_in_hist = int(cumulative[-1]) if cumulative.size else 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rho_bin_low_deg",
                "rho_bin_high_deg",
                "rho_bin_center_deg",
                "count",
                "annulus_area_sqdeg",
                "density_per_sqdeg",
                "cumulative_count_within_12deg",
                "cumulative_fraction_within_12deg",
            ],
        )
        writer.writeheader()
        for idx, count in enumerate(counts):
            low = float(edges[idx])
            high = float(edges[idx + 1])
            area = math.pi * ((high * high) - (low * low))
            fraction = (int(cumulative[idx]) / float(total_in_hist)) if total_in_hist > 0 else 0.0
            writer.writerow(
                {
                    "rho_bin_low_deg": f"{low:.6g}",
                    "rho_bin_high_deg": f"{high:.6g}",
                    "rho_bin_center_deg": f"{0.5 * (low + high):.6g}",
                    "count": int(count),
                    "annulus_area_sqdeg": f"{area:.12g}",
                    "density_per_sqdeg": f"{float(density_per_sqdeg[idx]):.12g}",
                    "cumulative_count_within_12deg": int(cumulative[idx]),
                    "cumulative_fraction_within_12deg": f"{fraction:.12g}",
                }
            )


def write_roi_coverage_by_cell_csv(
    path: Path,
    cells: Sequence[CellSpec],
    edges: np.ndarray,
    by_cell_counts: np.ndarray,
) -> None:
    cumulative = np.cumsum(by_cell_counts, axis=1, dtype=np.int64)
    totals = cumulative[:, -1] if cumulative.shape[1] > 0 else np.zeros(len(cells), dtype=np.int64)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id",
                "nhit_bin",
                "predE_bin",
                "rho_bin_low_deg",
                "rho_bin_high_deg",
                "rho_bin_center_deg",
                "count",
                "annulus_area_sqdeg",
                "density_per_sqdeg",
                "cumulative_count_within_12deg",
                "cumulative_fraction_within_12deg",
            ],
        )
        writer.writeheader()
        for cell_idx, cell in enumerate(cells):
            total = int(totals[cell_idx])
            for bin_idx, count in enumerate(by_cell_counts[cell_idx]):
                low = float(edges[bin_idx])
                high = float(edges[bin_idx + 1])
                area = math.pi * ((high * high) - (low * low))
                fraction = (int(cumulative[cell_idx, bin_idx]) / float(total)) if total > 0 else 0.0
                writer.writerow(
                    {
                        "cell_id": cell.cell_id,
                        "nhit_bin": cell.nhit_bin,
                        "predE_bin": cell.predE_bin,
                        "rho_bin_low_deg": f"{low:.6g}",
                        "rho_bin_high_deg": f"{high:.6g}",
                        "rho_bin_center_deg": f"{0.5 * (low + high):.6g}",
                        "count": int(count),
                        "annulus_area_sqdeg": f"{area:.12g}",
                        "density_per_sqdeg": f"{int(count) / area if area > 0 else 0.0:.12g}",
                        "cumulative_count_within_12deg": int(cumulative[cell_idx, bin_idx]),
                        "cumulative_fraction_within_12deg": f"{fraction:.12g}",
                    }
                )


def write_roi_coverage_plot(
    path: Path,
    cells: Sequence[CellSpec],
    edges: np.ndarray,
    counts: np.ndarray,
    density_per_sqdeg: np.ndarray,
    smoothed_density_per_sqdeg: Sequence[float],
    by_cell_counts: np.ndarray,
    edge_radius_estimate_deg: Optional[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[:-1] + edges[1:])
    cumulative = np.cumsum(counts, dtype=np.int64)
    cumulative_fraction = cumulative / float(cumulative[-1]) if cumulative.size and cumulative[-1] > 0 else np.zeros_like(cumulative, dtype=np.float64)
    annulus_area = math.pi * ((edges[1:] * edges[1:]) - (edges[:-1] * edges[:-1]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(centers, density_per_sqdeg, color="#3a6ea5", lw=1.0, label="annular density")
    axes[0].plot(centers, np.asarray(smoothed_density_per_sqdeg, dtype=np.float64), color="#d95f02", lw=1.5, label="smoothed")
    axes[0].axvline(ROI_FIDUCIAL_RADIUS_RECOMMENDATION_DEG, color="#2ca25f", ls="--", lw=1.2, label="fiducial 6 deg")
    if edge_radius_estimate_deg is not None:
        axes[0].axvline(float(edge_radius_estimate_deg), color="#b2182b", ls=":", lw=1.4, label="edge estimate")
    axes[0].set_xlim(ROI_HIST_MIN_DEG, ROI_HIST_MAX_DEG)
    axes[0].set_ylabel("events / deg^2")
    axes[0].set_title("Crab-centered Stage C ROI coverage diagnostic")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.2)

    for cell_idx, cell in enumerate(cells):
        density = by_cell_counts[cell_idx].astype(np.float64) / np.maximum(annulus_area, 1.0e-12)
        peak = float(np.nanmax(density)) if density.size else 0.0
        if peak <= 0.0:
            continue
        axes[1].plot(centers, density / peak, lw=0.8, alpha=0.65, label=str(cell.cell_id))
    axes[1].plot(centers, cumulative_fraction, color="black", lw=1.5, label="global cumulative fraction")
    axes[1].axvline(ROI_FIDUCIAL_RADIUS_RECOMMENDATION_DEG, color="#2ca25f", ls="--", lw=1.2)
    if edge_radius_estimate_deg is not None:
        axes[1].axvline(float(edge_radius_estimate_deg), color="#b2182b", ls=":", lw=1.4)
    axes[1].set_xlim(ROI_HIST_MIN_DEG, ROI_HIST_MAX_DEG)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("rho from Crab (deg)")
    axes[1].set_ylabel("normalized per-cell density / cumulative")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left", ncol=6, fontsize=6, title="cell")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_roi_coverage_artifacts(
    run_dir: Path,
    cells: Sequence[CellSpec],
    batch_size: int,
) -> Dict[str, object]:
    hist_edges = make_roi_hist_edges()
    n_bins = hist_edges.size - 1
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    annulus_area = math.pi * ((hist_edges[1:] * hist_edges[1:]) - (hist_edges[:-1] * hist_edges[:-1]))
    hist_counts = np.zeros(n_bins, dtype=np.int64)
    by_cell_counts = np.zeros((len(cells), n_bins), dtype=np.int64)
    within_counts = {radius_key(radius): 0 for radius in ROI_COUNT_RADII_DEG}
    by_cell_within = np.zeros((len(cells), len(ROI_COUNT_RADII_DEG)), dtype=np.int64)
    cell_totals = np.zeros(len(cells), dtype=np.int64)

    cell_ids = [int(cell.cell_id) for cell in cells]
    max_cell_id = max(cell_ids) if cell_ids else 0
    cell_lookup = np.full(max_cell_id + 1, -1, dtype=np.int16)
    for idx, cell_id in enumerate(cell_ids):
        cell_lookup[cell_id] = np.int16(idx)

    total_rows = 0
    rows_in_rho_histogram = 0
    invalid_cell_id_count = 0
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for ra, dec, parquet_cell_ids in iter_roi_record_batches(run_dir, batch_size):
        if ra.size == 0:
            continue
        finite = np.isfinite(ra) & np.isfinite(dec)
        if not bool(np.all(finite)):
            bad = int(ra.size - np.count_nonzero(finite))
            raise ValueError(f"Cannot compute Crab ROI coverage: found {bad} rows with non-finite RA/Dec")

        x_deg, y_deg, rho_deg = roi_xy_rho(ra, dec)
        total_rows += int(rho_deg.size)
        x_min = min(x_min, float(np.min(x_deg)))
        x_max = max(x_max, float(np.max(x_deg)))
        y_min = min(y_min, float(np.min(y_deg)))
        y_max = max(y_max, float(np.max(y_deg)))

        batch_counts, _ = np.histogram(rho_deg, bins=hist_edges)
        hist_counts += batch_counts.astype(np.int64)
        rows_in_rho_histogram += int(batch_counts.sum())

        for idx, radius in enumerate(ROI_COUNT_RADII_DEG):
            mask = rho_deg < float(radius)
            within_counts[radius_key(radius)] += int(np.count_nonzero(mask))

        valid_cell_ids = (parquet_cell_ids >= 0) & (parquet_cell_ids < cell_lookup.size)
        mapped_cell = np.full(parquet_cell_ids.shape, -1, dtype=np.int16)
        mapped_cell[valid_cell_ids] = cell_lookup[parquet_cell_ids[valid_cell_ids]]
        valid_cell = mapped_cell >= 0
        invalid_cell_id_count += int(parquet_cell_ids.size - np.count_nonzero(valid_cell))
        if np.any(valid_cell):
            cell_totals += np.bincount(mapped_cell[valid_cell].astype(np.int64), minlength=len(cells))

        bin_indices = np.searchsorted(hist_edges, rho_deg, side="right") - 1
        valid_bin = (bin_indices >= 0) & (bin_indices < n_bins) & valid_cell
        if np.any(valid_bin):
            flat = (mapped_cell[valid_bin].astype(np.int64) * n_bins) + bin_indices[valid_bin].astype(np.int64)
            by_cell_counts += np.bincount(flat, minlength=len(cells) * n_bins).reshape(len(cells), n_bins)

        for idx, radius in enumerate(ROI_COUNT_RADII_DEG):
            mask = (rho_deg < float(radius)) & valid_cell
            if np.any(mask):
                by_cell_within[:, idx] += np.bincount(mapped_cell[mask].astype(np.int64), minlength=len(cells))

    if total_rows <= 0:
        raise ValueError(f"Cannot compute Crab ROI coverage: no selected rows in {run_dir / 'obs_events'}")
    if invalid_cell_id_count:
        raise ValueError(f"Cannot compute Crab ROI coverage: found {invalid_cell_id_count} rows with cell_id outside selection table")

    x_edges = make_xy_quantile_edges(x_min, x_max)
    y_edges = make_xy_quantile_edges(y_min, y_max)
    x_hist = np.zeros(x_edges.size - 1, dtype=np.int64)
    y_hist = np.zeros(y_edges.size - 1, dtype=np.int64)
    for ra, dec, _ in iter_roi_record_batches(run_dir, batch_size):
        x_deg, y_deg, _ = roi_xy_rho(ra, dec)
        x_hist += np.histogram(x_deg, bins=x_edges)[0].astype(np.int64)
        y_hist += np.histogram(y_deg, bins=y_edges)[0].astype(np.int64)

    density_per_sqdeg = hist_counts.astype(np.float64) / np.maximum(annulus_area, 1.0e-12)
    edge = estimate_roi_edge(hist_counts, hist_edges, density_per_sqdeg)
    smooth_density = np.asarray(edge["smoothed_density_per_sqdeg"], dtype=np.float64)
    warnings = build_roi_warnings(
        total_rows,
        within_counts,
        centers,
        smooth_density,
        edge["plateau_density_per_sqdeg"],  # type: ignore[arg-type]
        by_cell_within,
    )

    coverage_csv = run_dir / "obs_events_roi_coverage.csv"
    coverage_by_cell_csv = run_dir / "obs_events_roi_coverage_by_cell.csv"
    coverage_json = run_dir / "obs_events_roi_coverage.json"
    coverage_png = run_dir / "obs_events_roi_coverage.png"

    write_roi_coverage_csv(coverage_csv, hist_edges, hist_counts, density_per_sqdeg)
    write_roi_coverage_by_cell_csv(coverage_by_cell_csv, cells, hist_edges, by_cell_counts)
    write_roi_coverage_plot(
        coverage_png,
        cells,
        hist_edges,
        hist_counts,
        density_per_sqdeg,
        smooth_density,
        by_cell_counts,
        edge["edge_radius_estimate_deg"],  # type: ignore[arg-type]
    )

    by_cell_counts_within: List[Dict[str, object]] = []
    for cell_idx, cell in enumerate(cells):
        row: Dict[str, object] = {
            "cell_id": int(cell.cell_id),
            "nhit_bin": cell.nhit_bin,
            "predE_bin": cell.predE_bin,
            "total_selected_events": int(cell_totals[cell_idx]),
            "rho_histogram_events_0_to_12_deg": int(by_cell_counts[cell_idx].sum()),
        }
        for radius_idx, radius in enumerate(ROI_COUNT_RADII_DEG):
            row[radius_key(radius)] = int(by_cell_within[cell_idx, radius_idx])
        by_cell_counts_within.append(row)

    cumulative = np.cumsum(hist_counts, dtype=np.int64)
    total_in_hist = int(cumulative[-1]) if cumulative.size else 0
    histogram_rows = []
    for idx, count in enumerate(hist_counts):
        histogram_rows.append(
            {
                "rho_bin_low_deg": float(hist_edges[idx]),
                "rho_bin_high_deg": float(hist_edges[idx + 1]),
                "rho_bin_center_deg": float(centers[idx]),
                "count": int(count),
                "annulus_area_sqdeg": float(annulus_area[idx]),
                "density_per_sqdeg": float(density_per_sqdeg[idx]),
                "smoothed_density_per_sqdeg": float(smooth_density[idx]),
                "cumulative_count_within_12deg": int(cumulative[idx]),
                "cumulative_fraction_within_12deg": (int(cumulative[idx]) / float(total_in_hist)) if total_in_hist > 0 else 0.0,
            }
        )

    payload: Dict[str, object] = {
        "source": ROI_SOURCE_NAME,
        "ra_deg": ROI_RA_DEG,
        "dec_deg": ROI_DEC_DEG,
        "coordinate": ROI_COORDINATE,
        "status": "diagnostic_only_no_cut_applied",
        "run_dir": str(run_dir),
        "dataset_path": str(run_dir / "obs_events"),
        "total_selected_events": int(total_rows),
        "rho_histogram_range_deg": [ROI_HIST_MIN_DEG, ROI_HIST_MAX_DEG],
        "rho_histogram_bin_width_deg": ROI_HIST_BIN_WIDTH_DEG,
        "rho_histogram_events_0_to_12_deg": int(rows_in_rho_histogram),
        "rho_histogram_fraction_of_total": float(rows_in_rho_histogram / float(total_rows)),
        "counts_within_radius": {key: int(value) for key, value in within_counts.items()},
        "counts_within_radius_fraction_of_total": {
            key: float(value / float(total_rows)) for key, value in within_counts.items()
        },
        "x_deg": {
            "min": float(x_min),
            "max": float(x_max),
            "quantiles": quantiles_from_histogram(x_hist, x_edges, XY_QUANTILES),
            "quantile_bin_width_deg": XY_QUANTILE_BIN_WIDTH_DEG,
        },
        "y_deg": {
            "min": float(y_min),
            "max": float(y_max),
            "quantiles": quantiles_from_histogram(y_hist, y_edges, XY_QUANTILES),
            "quantile_bin_width_deg": XY_QUANTILE_BIN_WIDTH_DEG,
        },
        "edge_radius_estimate_deg": edge["edge_radius_estimate_deg"],
        "edge_radius_method": edge["edge_radius_method"],
        "edge_diagnostics": {
            "plateau_density_per_sqdeg": edge["plateau_density_per_sqdeg"],
            "rho_percentile_99_deg_within_12deg": edge["rho_percentile_99_deg_within_12deg"],
            "rho_percentile_99p5_deg_within_12deg": edge["rho_percentile_99p5_deg_within_12deg"],
            "density_profile": "annulus-area-normalized counts, smoothed with a 7-bin moving average",
        },
        "fiducial_radius_recommendation_deg": ROI_FIDUCIAL_RADIUS_RECOMMENDATION_DEG,
        "warnings": warnings,
        "histogram": histogram_rows,
        "counts_within_radius_by_cell": by_cell_counts_within,
        "artifacts": {
            "rho_histogram_json": str(coverage_json),
            "coverage_csv": str(coverage_csv),
            "coverage_by_cell_csv": str(coverage_by_cell_csv),
            "coverage_plot_png": str(coverage_png),
        },
    }
    write_json(coverage_json, payload)
    return payload


def roi_metadata_block(run_dir: Path, coverage: Dict[str, object]) -> Dict[str, object]:
    artifacts = coverage["artifacts"]  # type: ignore[index]
    return {
        "source": ROI_SOURCE_NAME,
        "ra_deg": ROI_RA_DEG,
        "dec_deg": ROI_DEC_DEG,
        "coordinate": ROI_COORDINATE,
        "rho_histogram_json": str(artifacts["rho_histogram_json"]),  # type: ignore[index]
        "coverage_csv": str(artifacts["coverage_csv"]),  # type: ignore[index]
        "coverage_by_cell_csv": str(artifacts["coverage_by_cell_csv"]),  # type: ignore[index]
        "coverage_plot_png": str(artifacts["coverage_plot_png"]),  # type: ignore[index]
        "fiducial_radius_recommendation_deg": ROI_FIDUCIAL_RADIUS_RECOMMENDATION_DEG,
        "edge_radius_estimate_deg": coverage.get("edge_radius_estimate_deg"),
        "edge_radius_method": coverage.get("edge_radius_method"),
        "counts_within_radius": coverage.get("counts_within_radius"),
        "counts_within_radius_fraction_of_total": coverage.get("counts_within_radius_fraction_of_total"),
        "warnings": coverage.get("warnings", []),
        "status": "diagnostic_only_no_cut_applied",
        "note": "Stage C does not apply a Crab ROI cut; these diagnostics are for Stage D fiducial-ROI decisions.",
        "run_dir": str(run_dir),
    }


def attach_roi_coverage_to_metadata(run_dir: Path, metadata: Dict[str, object], coverage: Dict[str, object]) -> None:
    metadata["roi_coverage"] = roi_metadata_block(run_dir, coverage)
    outputs = metadata.setdefault("outputs", {})
    if isinstance(outputs, dict):
        artifacts = coverage["artifacts"]  # type: ignore[index]
        outputs["roi_coverage_json"] = str(artifacts["rho_histogram_json"])  # type: ignore[index]
        outputs["roi_coverage_csv"] = str(artifacts["coverage_csv"])  # type: ignore[index]
        outputs["roi_coverage_by_cell_csv"] = str(artifacts["coverage_by_cell_csv"])  # type: ignore[index]
        outputs["roi_coverage_png"] = str(artifacts["coverage_plot_png"])  # type: ignore[index]


def build_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    output_root: Path,
    selection_csv: Path,
    cells: Sequence[CellSpec],
    results: Sequence[Dict[str, object]],
    elapsed_seconds: float,
) -> Dict[str, object]:
    grouped = group_by_month(results)
    global_cutflow = merge_cutflow(results)
    global_cell_counts = merge_cell_counts(results, len(cells))
    lower_bounds = finite_cell_lower_bounds(cells)
    out_of_ledger_after_finite = sum(int(result.get("out_of_ledger_after_finite") or 0) for result in results)
    nhit_below_candidate_min = sum(int(result.get("nhit_below_candidate_min") or 0) for result in results)
    pred_below_candidate_min = sum(int(result.get("predE_below_candidate_min") or 0) for result in results)
    matched_mjd_min = finite_min([result.get("matched_mjd_min") for result in results])
    matched_mjd_max = finite_max([result.get("matched_mjd_max") for result in results])
    selected_mjd_min = finite_min([result.get("selected_mjd_min") for result in results])
    selected_mjd_max = finite_max([result.get("selected_mjd_max") for result in results])
    rough_live_time_seconds = sum(float(result.get("rough_live_time_seconds") or 0.0) for result in results)

    return {
        "description": "Stage C observation reduction for configured (Nhit, predicted logE) cells.",
        "run_id": run_dir.name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "obs_root": str(Path(args.obs_root).resolve()),
        "time_root": str(Path(args.time_root).resolve()),
        "cell_selection_csv": str(selection_csv),
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "tree_name": args.tree_name,
        "time_tree_name": args.time_tree_name,
        "n_cells": len(cells),
        "cells": [
            {
                "cell_index": cell.index,
                "cell_id": cell.cell_id,
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "mc_count_reference": cell.mc_count,
                "selection_version": cell.selection_version,
                "selection_reason": cell.selection_reason,
                "selected_events": int(global_cell_counts[cell.index]),
            }
            for cell in cells
        ],
        "cuts": {
            "match_status_equals": int(args.match_status_equals),
            "pincness_lt": float(args.cut_pinc_max),
            "fitstat_equals": int(args.cut_fitstat_equals),
            "theta_rad_lt": math.radians(float(args.cut_theta_max_deg)),
            "theta_deg_lt": float(args.cut_theta_max_deg),
            "dcedge_gt_m": float(args.cut_dcedge_min),
            "candidate_nhit_min_inclusive": lower_bounds["nhit_min_inclusive"],
            "candidate_predE_min_inclusive": lower_bounds["predE_min_inclusive"],
        },
        "columns": {
            "theta": "radians",
            "phi": "radians",
            "mjd": "modified_julian_date",
            "source_file_id": "0-based index in source_files.csv",
            "entry": "0-based entry index in the source eval ROOT tree",
        },
        "dataset": build_manifest(run_dir, results),
        "processing": {
            "input_file_count": len(results),
            "processed_file_count": sum(1 for result in results if result["status"] == "processed"),
            "missing_time_file_count": sum(1 for result in results if result["status"] == "missing_time_skipped"),
            "entry_mismatch_file_count": sum(1 for result in results if result.get("entry_mismatch")),
            "selected_rows": int(global_cutflow["after_cell_selection"]),
            "workers": int(args.workers),
            "entries_per_chunk": int(args.entries_per_chunk),
            "compression": args.compression,
            "elapsed_seconds": float(elapsed_seconds),
        },
        "cutflow": global_cutflow,
        "assignment_audit": {
            "candidate_nhit_min_inclusive": lower_bounds["nhit_min_inclusive"],
            "candidate_predE_min_inclusive": lower_bounds["predE_min_inclusive"],
            "nhit_below_candidate_min_after_quality_cuts": int(nhit_below_candidate_min),
            "predE_below_candidate_min_after_quality_cuts": int(pred_below_candidate_min),
            "out_of_ledger_after_finite": int(out_of_ledger_after_finite),
            "selected_rows": int(global_cutflow["after_cell_selection"]),
            "note": "Counts are after match/status/quality/finite cuts and before final cell selection unless noted.",
        },
        "months": {
            yyyymm: {
                "input_files": len(month_results),
                "selected_rows": int(merge_cutflow(month_results)["after_cell_selection"]),
                "rough_live_time_seconds": float(sum(float(result.get("rough_live_time_seconds") or 0.0) for result in month_results)),
            }
            for yyyymm, month_results in sorted(grouped.items())
        },
        "mjd_coverage": {
            "matched_mjd_min": matched_mjd_min,
            "matched_mjd_max": matched_mjd_max,
            "selected_mjd_min": selected_mjd_min,
            "selected_mjd_max": selected_mjd_max,
        },
        "live_time_basis": {
            "scope": "rough Stage C basis only; Stage D owns final rate/live-time weighting",
            "gap_threshold_sec": float(args.gap_threshold_sec),
            "rough_live_time_seconds_sum_files": float(rough_live_time_seconds),
            "rough_live_time_days_sum_files": float(rough_live_time_seconds / 86400.0),
            "formula": "sum over files of max(0, matched_mjd_span_seconds - matched_mjd_gaps_above_threshold)",
            "mjd_sample": "match_status-passing entries before event-quality cuts",
        },
        "quality_flags": {
            "allow_missing_time": bool(args.allow_missing_time),
            "allow_entry_mismatch": bool(args.allow_entry_mismatch),
            "month_source": "EsgYYYYMMDD_HH.root filename",
            "quicklook_reference_total_entries": 127692389,
        },
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": {
            "manifest_json": str(run_dir / "obs_events_manifest.json"),
            "metadata_json": str(run_dir / "obs_events_metadata.json"),
            "source_files_csv": str(run_dir / "source_files.csv"),
            "cutflow_csv": str(run_dir / "obs_events_cutflow.csv"),
            "cell_counts_csv": str(run_dir / "obs_events_cell_counts.csv"),
            "summary_md": str(run_dir / "obs_events_summary.md"),
            "roi_coverage_json": str(run_dir / "obs_events_roi_coverage.json"),
            "roi_coverage_csv": str(run_dir / "obs_events_roi_coverage.csv"),
            "roi_coverage_by_cell_csv": str(run_dir / "obs_events_roi_coverage_by_cell.csv"),
            "roi_coverage_png": str(run_dir / "obs_events_roi_coverage.png"),
        },
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_optional_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"


def write_summary(path: Path, metadata: Dict[str, object]) -> None:
    processing = metadata["processing"]
    cutflow = metadata["cutflow"]
    live_time = metadata["live_time_basis"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage C Observation Reduction Summary\n\n")
        f.write(f"- Input files: {processing['input_file_count']}\n")
        f.write(f"- Processed files: {processing['processed_file_count']}\n")
        f.write(f"- Missing time files: {processing['missing_time_file_count']}\n")
        f.write(f"- Entry mismatch files: {processing['entry_mismatch_file_count']}\n")
        f.write(f"- Input entries: {cutflow['input_entries']:,}\n")
        f.write(f"- Selected configured-cell rows: {cutflow['after_cell_selection']:,}\n")
        f.write(f"- Rough live-time basis: {live_time['rough_live_time_days_sum_files']:.6g} days\n")
        f.write("- Live-time note: Stage D owns final rate/live-time weighting.\n\n")
        f.write("| yyyymm | files | selected rows | rough live-time days |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for yyyymm, row in metadata["months"].items():  # type: ignore[union-attr]
            days = float(row["rough_live_time_seconds"]) / 86400.0
            f.write(f"| {yyyymm} | {row['input_files']} | {row['selected_rows']:,} | {days:.6g} |\n")

        roi = metadata.get("roi_coverage")
        if isinstance(roi, dict):
            f.write("\n## Crab ROI coverage diagnostics\n\n")
            f.write("Stage C does not apply a Crab ROI cut.\n")
            f.write("These diagnostics only characterize available sky coverage for Stage D.\n")
            f.write(
                "Current downstream baseline expects Stage D to choose a fiducial ROI, "
                "likely rho<6 deg if the coverage edge is around 8 deg.\n\n"
            )
            f.write(f"- Coordinate: {roi.get('coordinate', ROI_COORDINATE)} around Crab ({ROI_RA_DEG}, {ROI_DEC_DEG}) deg\n")
            f.write(f"- Diagnostic status: {roi.get('status', 'diagnostic_only_no_cut_applied')}\n")
            f.write(f"- Fiducial radius recommendation: {format_optional_float(roi.get('fiducial_radius_recommendation_deg'))} deg\n")
            f.write(f"- Edge radius estimate: {format_optional_float(roi.get('edge_radius_estimate_deg'))} deg\n")
            f.write(f"- Edge estimate method: {roi.get('edge_radius_method', 'n/a')}\n")
            counts = roi.get("counts_within_radius")
            fractions = roi.get("counts_within_radius_fraction_of_total")
            if isinstance(counts, dict) and isinstance(fractions, dict):
                f.write("\n| radius | count | fraction of selected rows |\n")
                f.write("| ---: | ---: | ---: |\n")
                for radius in ROI_COUNT_RADII_DEG:
                    key = radius_key(radius)
                    count = int(counts.get(key, 0))
                    fraction = float(fractions.get(key, 0.0))
                    f.write(f"| rho<{radius:g} deg | {count:,} | {fraction:.6g} |\n")
            warnings = roi.get("warnings")
            if isinstance(warnings, list) and warnings:
                f.write("\nWarnings:\n")
                for warning in warnings:
                    f.write(f"- {warning}\n")


def run_processing(
    specs: Sequence[InputFileSpec],
    config: Dict[str, object],
    workers: int,
    print_every: int,
) -> List[Dict[str, object]]:
    tasks = [(spec, config) for spec in specs]
    results_by_id: Dict[int, Dict[str, object]] = {}
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            result = process_input_file(task)
            results_by_id[int(result["source_file_id"])] = result
            if print_every > 0 and (idx % print_every == 0 or idx == len(tasks)):
                print(
                    f"[{idx}/{len(tasks)}] selected={sum(int(r['selected_rows']) for r in results_by_id.values()):,}",
                    flush=True,
                )
    else:
        print(f"Processing {len(tasks)} observation files with {workers} workers.", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_by_id = {executor.submit(process_input_file, task): task[0].source_file_id for task in tasks}
            for done_count, future in enumerate(as_completed(future_by_id), start=1):
                source_file_id = future_by_id[future]
                result = future.result()
                results_by_id[int(source_file_id)] = result
                if print_every > 0 and (done_count % print_every == 0 or done_count == len(tasks)):
                    print(
                        f"[{done_count}/{len(tasks)}] selected={sum(int(r['selected_rows']) for r in results_by_id.values()):,}",
                        flush=True,
                    )
    return [results_by_id[idx] for idx in sorted(results_by_id)]


def load_metadata(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata JSON is not an object: {path}")
    return payload


def update_existing_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.update_existing_run).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Stage C run directory does not exist: {run_dir}")
    selection_csv = Path(args.cell_selection_csv).resolve()
    cells = load_cells(selection_csv)
    metadata_path = run_dir / "obs_events_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage C metadata does not exist: {metadata_path}")
    metadata = load_metadata(metadata_path)

    if args.roi_batch_size <= 0:
        raise ValueError("--roi-batch-size must be positive")
    coverage = build_roi_coverage_artifacts(run_dir, cells, batch_size=int(args.roi_batch_size))
    attach_roi_coverage_to_metadata(run_dir, metadata, coverage)
    write_json(metadata_path, metadata)
    write_summary(run_dir / "obs_events_summary.md", metadata)

    print(f"Updated existing Stage C run: {run_dir}", flush=True)
    print(f"Wrote ROI coverage: {run_dir / 'obs_events_roi_coverage.json'}", flush=True)
    print(f"Edge radius estimate: {coverage.get('edge_radius_estimate_deg')}", flush=True)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.update_existing_run:
        update_existing_run(args)
        return
    if args.entries_per_chunk <= 0:
        raise ValueError("--entries-per-chunk must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.gap_threshold_sec <= 0:
        raise ValueError("--gap-threshold-sec must be positive")
    if args.roi_batch_size <= 0:
        raise ValueError("--roi-batch-size must be positive")

    obs_root = Path(args.obs_root).resolve()
    time_root = Path(args.time_root).resolve()
    output_root = Path(args.output_dir).resolve()
    selection_csv = Path(args.cell_selection_csv).resolve()
    cells = load_cells(selection_csv)
    specs = discover_observation_files(obs_root, time_root, args.file_glob, args.day_prefix, args.max_files)
    run_id = sanitize_run_id(args.run_id or make_default_run_id())
    run_dir = prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))

    print(f"Loaded {len(cells)} selected cells from {selection_csv}", flush=True)
    print(f"Discovered {len(specs)} observation files under {obs_root}", flush=True)
    print(f"Output run dir: {run_dir}", flush=True)

    config: Dict[str, object] = {
        "run_dir": str(run_dir),
        "cells": cells,
        "tree_name": args.tree_name,
        "time_tree_name": args.time_tree_name,
        "entries_per_chunk": int(args.entries_per_chunk),
        "allow_missing_time": bool(args.allow_missing_time),
        "allow_entry_mismatch": bool(args.allow_entry_mismatch),
        "match_status_equals": int(args.match_status_equals),
        "cut_pinc_max": float(args.cut_pinc_max),
        "cut_fitstat_equals": int(args.cut_fitstat_equals),
        "cut_theta_max_deg": float(args.cut_theta_max_deg),
        "cut_dcedge_min": float(args.cut_dcedge_min),
        "gap_threshold_sec": float(args.gap_threshold_sec),
        "compression": args.compression,
    }
    results = run_processing(specs, config, workers=int(args.workers), print_every=int(args.print_every))
    elapsed = time.perf_counter() - start

    manifest = build_manifest(run_dir, results)
    metadata = build_metadata(args, run_dir, output_root, selection_csv, cells, results, elapsed)
    metadata["dataset"] = manifest

    write_source_files_csv(run_dir / "source_files.csv", results)
    write_cutflow_csv(run_dir / "obs_events_cutflow.csv", results)
    write_cell_counts_csv(run_dir / "obs_events_cell_counts.csv", results, cells)
    if not args.skip_roi_coverage:
        coverage = build_roi_coverage_artifacts(run_dir, cells, batch_size=int(args.roi_batch_size))
        attach_roi_coverage_to_metadata(run_dir, metadata, coverage)
    write_json(run_dir / "obs_events_manifest.json", manifest)
    write_json(run_dir / "obs_events_metadata.json", metadata)
    write_summary(run_dir / "obs_events_summary.md", metadata)

    if not args.no_promote_current:
        promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        write_json(run_dir / "obs_events_metadata.json", metadata)
    else:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]
        write_json(run_dir / "obs_events_metadata.json", metadata)

    print(f"Wrote dataset: {run_dir / 'obs_events'}", flush=True)
    print(f"Wrote manifest: {run_dir / 'obs_events_manifest.json'}", flush=True)
    print(f"Wrote metadata: {run_dir / 'obs_events_metadata.json'}", flush=True)
    print(f"Selected rows: {metadata['processing']['selected_rows']:,}", flush=True)  # type: ignore[index]
    if not args.no_promote_current:
        print(f"Promoted current Stage C output to {output_root / 'current'}", flush=True)


if __name__ == "__main__":
    main()
