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
        description="Stage C observation reduction: eval ROOT + recovered time friend tree to v1-cell parquet."
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


def process_input_file(task: Tuple[InputFileSpec, Dict[str, object]]) -> Dict[str, object]:
    spec, config = task
    obs_path = Path(spec.obs_path)
    time_path = Path(spec.time_path)
    run_dir = Path(str(config["run_dir"]))
    cells: Sequence[CellSpec] = config["cells"]  # type: ignore[assignment]
    compression = None if str(config["compression"]) == "none" else str(config["compression"])

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
    matched_mjd_min = finite_min([result.get("matched_mjd_min") for result in results])
    matched_mjd_max = finite_max([result.get("matched_mjd_max") for result in results])
    selected_mjd_min = finite_min([result.get("selected_mjd_min") for result in results])
    selected_mjd_max = finite_max([result.get("selected_mjd_max") for result in results])
    rough_live_time_seconds = sum(float(result.get("rough_live_time_seconds") or 0.0) for result in results)

    return {
        "description": "Stage C observation reduction for v1 (Nhit, predicted logE) cells.",
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
        },
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        f.write(f"- Selected v1-cell rows: {cutflow['after_cell_selection']:,}\n")
        f.write(f"- Rough live-time basis: {live_time['rough_live_time_days_sum_files']:.6g} days\n")
        f.write("- Live-time note: Stage D owns final rate/live-time weighting.\n\n")
        f.write("| yyyymm | files | selected rows | rough live-time days |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for yyyymm, row in metadata["months"].items():  # type: ignore[union-attr]
            days = float(row["rough_live_time_seconds"]) / 86400.0
            f.write(f"| {yyyymm} | {row['input_files']} | {row['selected_rows']:,} | {days:.6g} |\n")


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


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.entries_per_chunk <= 0:
        raise ValueError("--entries-per-chunk must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.gap_threshold_sec <= 0:
        raise ValueError("--gap-threshold-sec must be positive")

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
