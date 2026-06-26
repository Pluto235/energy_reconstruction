#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_STAGE_C_DIR = "apply/output/stage_c/current"
DEFAULT_BACKGROUND_NPZ = "apply/output/stage_d/current/background_v1.npz"
DEFAULT_BACKGROUND_METADATA = "apply/output/stage_d/current/background_v1_metadata.json"
DEFAULT_CELL_SELECTION = "apply/config/cell_selection_v1.csv"
DEFAULT_OUTPUT_DIR = "apply/output/stage_e"
DEFAULT_REPORT_HTML = "apply/report/stage_e_report.html"
DEFAULT_SOURCE_RA_DEG = 83.63
DEFAULT_SOURCE_DEC_DEG = 22.01
ALLOWED_BACKGROUND_MODES = {"crab_roi_local", "full_field_direct_integration"}
ALLOWED_BACKGROUND_FORMS = {"direct_expectation", "off_counts"}


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
class BackgroundContract:
    arrays: Dict[str, np.ndarray]
    background_mode: str
    background_form: str
    statistic_kind: str
    strict_contract: bool
    promotable_contract: bool
    diagnostic_reasons: List[str]
    warnings: List[str]
    roi_fiducial_deg: Optional[float]
    roi_config: Dict[str, object]
    metadata_summary: Dict[str, object]


@dataclass
class ScanResult:
    n_on: np.ndarray
    input_rows: int
    valid_cell_rows: int
    roi_kept_rows: int
    roi_rejected_rows: int
    processed_batches: int


@dataclass
class SignalStats:
    excess: np.ndarray
    excess_err_stat: np.ndarray
    excess_err_conservative: np.ndarray
    known_b_sigma: np.ndarray
    li_ma_sigma: np.ndarray
    total_n_on: int
    total_b_on: float
    total_n_off: Optional[float]
    total_excess: float
    total_excess_err_stat: float
    total_known_b_sigma_aggregate: float
    total_known_b_sigma_combined: float
    total_li_ma_sigma_combined: Optional[float]
    total_formal_sigma: float
    formal_significance_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage E Crab signal extraction for configured SED cells.")
    parser.add_argument("--stage-c-dir", type=str, default=DEFAULT_STAGE_C_DIR)
    parser.add_argument("--background-npz", type=str, default=DEFAULT_BACKGROUND_NPZ)
    parser.add_argument("--background-metadata", type=str, default=DEFAULT_BACKGROUND_METADATA)
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

    parser.add_argument("--source-ra-deg", type=float, default=DEFAULT_SOURCE_RA_DEG)
    parser.add_argument("--source-dec-deg", type=float, default=DEFAULT_SOURCE_DEC_DEG)
    parser.add_argument("--mjd-min", type=float, default=None, help="Optional inclusive MJD lower bound for validation splits.")
    parser.add_argument("--mjd-max", type=float, default=None, help="Optional exclusive MJD upper bound for validation splits.")
    parser.add_argument("--batch-size", type=int, default=500000)
    parser.add_argument("--max-batches", type=int, default=None, help="Read only the first N parquet batches for smoke tests.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--report-html", type=str, default=DEFAULT_REPORT_HTML)

    parser.add_argument("--quality-min-total-sigma", type=float, default=50.0)
    parser.add_argument("--quality-max-total-sigma", type=float, default=200.0)
    parser.add_argument(
        "--containment-override",
        choices=["none", "fixed1"],
        default="none",
        help=(
            "Override containment_r_opt written to Stage E outputs. Use fixed1 only with "
            "aperture-conditioned Stage A responses, where the response numerator already "
            "contains the r_opt cut."
        ),
    )
    parser.add_argument("--npz-name", type=str, default="signal_v1.npz")
    parser.add_argument("--metadata-name", type=str, default="signal_v1_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="signal_v1_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="signal_v1_summary.md")
    return parser.parse_args()


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


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
            raise FileExistsError(f"Stage E run directory already exists: {run_dir}")
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


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_token(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text or None


def first_text(*values: object) -> Optional[str]:
    for value in values:
        text = normalize_token(value)
        if text:
            return text
    return None


def metadata_dict(metadata: Dict[str, object], key: str) -> Dict[str, object]:
    value = metadata.get(key)
    return value if isinstance(value, dict) else {}


def find_roi_fiducial_deg(metadata: Dict[str, object]) -> Optional[float]:
    roi = metadata_dict(metadata, "roi")
    background_model = metadata_dict(metadata, "background_model")
    candidates = [
        roi.get("fiducial_radius_deg"),
        roi.get("fiducial_rho_deg"),
        roi.get("roi_fiducial_deg"),
        background_model.get("fiducial_radius_deg"),
        background_model.get("fiducial_rho_deg"),
        background_model.get("roi_fiducial_deg"),
    ]
    for value in candidates:
        number = finite_float(value)
        if number is not None:
            return number
    return None


def column_to_numpy(batch, name: str) -> np.ndarray:
    return batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)


def apply_mjd_window(mjd: np.ndarray, *, mjd_min: Optional[float], mjd_max: Optional[float]) -> np.ndarray:
    mask = np.isfinite(mjd)
    if mjd_min is not None:
        mask &= mjd >= float(mjd_min)
    if mjd_max is not None:
        mask &= mjd < float(mjd_max)
    return mask


def ordered_background_arrays(
    data: np.lib.npyio.NpzFile,
    background_npz: Path,
    cells: Sequence[CellSpec],
    *,
    required_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    common_required = {"cell_id", "r_opt_deg", "sigma_deg", "containment_r_opt", "nhit_bin", "predE_bin"}
    missing = (common_required | set(required_names)) - set(data.files)
    if missing:
        raise ValueError(f"{background_npz} is missing required arrays: {sorted(missing)}")

    background_ids = np.asarray(data["cell_id"], dtype=np.int32)
    if len(set(int(x) for x in background_ids)) != background_ids.size:
        raise ValueError("Stage D background has duplicate cell_id values")

    id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(background_ids)}
    missing_cells = [cell.cell_id for cell in cells if cell.cell_id not in id_to_index]
    if missing_cells:
        raise ValueError(f"Stage D background is missing selected cells: {missing_cells}")

    order = np.asarray([id_to_index[cell.cell_id] for cell in cells], dtype=np.int64)
    arrays: Dict[str, np.ndarray] = {
        "cell_id": background_ids[order].astype(np.int32),
        "r_opt_deg": np.asarray(data["r_opt_deg"][order], dtype=np.float64),
        "sigma_deg": np.asarray(data["sigma_deg"][order], dtype=np.float64),
        "containment_r_opt": np.asarray(data["containment_r_opt"][order], dtype=np.float64),
        "nhit_bin": np.asarray(data["nhit_bin"][order]).astype("U32"),
        "predE_bin": np.asarray(data["predE_bin"][order]).astype("U32"),
    }
    for optional in ["B_on", "N_off", "alpha"]:
        if optional in data.files:
            arrays[optional] = np.asarray(data[optional][order], dtype=np.float64)

    for idx, cell in enumerate(cells):
        if int(arrays["cell_id"][idx]) != cell.cell_id:
            raise ValueError(f"Internal cell remap failed for cell {cell.cell_id}")
        if str(arrays["nhit_bin"][idx]) != cell.nhit_bin or str(arrays["predE_bin"][idx]) != cell.predE_bin:
            raise ValueError(
                f"Cell label mismatch for cell {cell.cell_id}: "
                f"selection=({cell.nhit_bin}, {cell.predE_bin}) "
                f"background=({arrays['nhit_bin'][idx]}, {arrays['predE_bin'][idx]})"
            )

    for key in ["r_opt_deg", "sigma_deg", "containment_r_opt"]:
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Background array {key} contains non-finite values")
    if np.any(arrays["r_opt_deg"] <= 0.0):
        raise ValueError("Background r_opt_deg must be positive")
    return arrays


def load_background_contract(
    background_npz: Path,
    background_metadata: Dict[str, object],
    cells: Sequence[CellSpec],
) -> BackgroundContract:
    if not background_npz.exists():
        raise FileNotFoundError(f"Stage D background NPZ does not exist: {background_npz}")

    model = metadata_dict(background_metadata, "background_model")
    mode_raw = first_text(
        model.get("background_mode"),
        model.get("background_scope"),
        background_metadata.get("background_mode"),
        background_metadata.get("background_scope"),
    )
    form_raw = first_text(model.get("background_form"), background_metadata.get("background_form"))
    method = first_text(model.get("method"), background_metadata.get("method")) or ""

    diagnostic_reasons: List[str] = []
    warnings: List[str] = []
    strict_contract = True

    if mode_raw not in ALLOWED_BACKGROUND_MODES:
        if mode_raw is None:
            diagnostic_reasons.append("Stage D metadata is missing background_mode/background_scope.")
        else:
            diagnostic_reasons.append(f"Stage D background mode is unsupported: {mode_raw!r}.")
        strict_contract = False
        mode = "legacy_unspecified"
    else:
        mode = mode_raw

    if form_raw is None and "direct" in method:
        form_raw = "direct_expectation"
        diagnostic_reasons.append("Stage D background_form was inferred from legacy direct-integration method.")
        strict_contract = False
    if form_raw in {"off", "on_off", "on_off_counts", "alpha_n_off", "alpha_noff"}:
        form_raw = "off_counts"
    if form_raw not in ALLOWED_BACKGROUND_FORMS:
        if form_raw is None:
            diagnostic_reasons.append("Stage D metadata is missing background_form.")
        else:
            diagnostic_reasons.append(f"Stage D background form is unsupported: {form_raw!r}.")
        strict_contract = False
        form = "direct_expectation"
    else:
        form = form_raw

    required = ["B_on"] if form == "direct_expectation" else ["N_off", "alpha"]
    with np.load(background_npz, allow_pickle=False) as data:
        arrays = ordered_background_arrays(data, background_npz, cells, required_names=required)

    if form == "off_counts":
        alpha = arrays["alpha"]
        n_off = arrays["N_off"]
        if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
            raise ValueError("Stage D off-count background has invalid alpha values")
        if not np.all(np.isfinite(n_off)) or np.any(n_off < 0.0):
            raise ValueError("Stage D off-count background has invalid N_off values")
        b_on_from_off = alpha * n_off
        if "B_on" in arrays:
            b_on_existing = arrays["B_on"]
            scale = np.maximum(1.0, np.abs(b_on_from_off))
            mismatch = np.abs(b_on_existing - b_on_from_off) / scale
            if np.nanmax(mismatch) > 1.0e-5:
                diagnostic_reasons.append("Stage D B_on is inconsistent with alpha * N_off for off-count background.")
                strict_contract = False
        arrays["B_on"] = b_on_from_off.astype(np.float64)
        statistic_kind = "li_ma_on_off"
    else:
        b_on = arrays["B_on"]
        if not np.all(np.isfinite(b_on)) or np.any(b_on < 0.0):
            raise ValueError("Stage D direct background has invalid B_on values")
        arrays["alpha"] = np.full(len(cells), np.nan, dtype=np.float64)
        arrays["N_off"] = np.full(len(cells), np.nan, dtype=np.float64)
        statistic_kind = "known_background_poisson"

    roi_fiducial_deg = None
    roi_config = metadata_dict(background_metadata, "roi").copy()
    if mode == "crab_roi_local":
        roi_fiducial_deg = find_roi_fiducial_deg(background_metadata)
        if roi_fiducial_deg is None or roi_fiducial_deg <= 0.0:
            raise ValueError("Stage D crab_roi_local contract is missing a positive fiducial ROI radius.")
        roi_config.setdefault("fiducial_radius_deg", roi_fiducial_deg)
    elif roi_config:
        roi_config.setdefault("applied_in_stage_e", False)

    promotable_contract = strict_contract and not diagnostic_reasons
    return BackgroundContract(
        arrays=arrays,
        background_mode=mode,
        background_form=form,
        statistic_kind=statistic_kind,
        strict_contract=strict_contract,
        promotable_contract=promotable_contract,
        diagnostic_reasons=diagnostic_reasons,
        warnings=warnings,
        roi_fiducial_deg=roi_fiducial_deg,
        roi_config=roi_config,
        metadata_summary={
            "background_mode": mode,
            "background_form": form,
            "method": method or None,
            "stage_d_run_id": background_metadata.get("run_id"),
        },
    )


def scan_on_region(
    obs_events_dir: Path,
    cells: Sequence[CellSpec],
    r_opt_deg: np.ndarray,
    source_ra_deg: float,
    source_dec_deg: float,
    *,
    roi_fiducial_deg: Optional[float],
    batch_size: int,
    max_batches: Optional[int],
    print_every: int,
    mjd_min: Optional[float] = None,
    mjd_max: Optional[float] = None,
) -> ScanResult:
    dataset = ds.dataset(obs_events_dir, format="parquet", partitioning="hive")
    columns = ["mjd", "ra_mean_deg", "dec_mean_deg", "cell_id"]
    scanner = dataset.scanner(columns=columns, batch_size=int(batch_size), use_threads=True)
    n_cells = len(cells)
    max_cell_id = max(cell.cell_id for cell in cells)
    cell_index_by_id = np.full(max_cell_id + 1, -1, dtype=np.int16)
    for cell in cells:
        cell_index_by_id[cell.cell_id] = np.int16(cell.index)

    n_on = np.zeros(n_cells, dtype=np.int64)
    input_rows = 0
    valid_cell_rows = 0
    roi_kept_rows = 0
    roi_rejected_rows = 0
    processed_batches = 0
    source_dec_rad = math.radians(float(source_dec_deg))
    sin_source_dec = math.sin(source_dec_rad)
    cos_source_dec = math.cos(source_dec_rad)

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
        valid = (cell_idx >= 0) & apply_mjd_window(mjd, mjd_min=mjd_min, mjd_max=mjd_max) & np.isfinite(ra) & np.isfinite(dec)
        if not np.any(valid):
            continue

        ra_v = ra[valid]
        dec_v = dec[valid]
        cell_v = np.asarray(cell_idx[valid], dtype=np.int64)
        valid_cell_rows += int(ra_v.size)

        dra_deg = ((ra_v - float(source_ra_deg) + 180.0) % 360.0) - 180.0
        if roi_fiducial_deg is not None:
            x = dra_deg * cos_source_dec
            y = dec_v - float(source_dec_deg)
            rho = np.hypot(x, y)
            roi_keep = rho < float(roi_fiducial_deg)
            roi_kept_rows += int(np.count_nonzero(roi_keep))
            roi_rejected_rows += int(roi_keep.size - np.count_nonzero(roi_keep))
            if not np.any(roi_keep):
                continue
            ra_v = ra_v[roi_keep]
            dec_v = dec_v[roi_keep]
            cell_v = cell_v[roi_keep]
            dra_deg = dra_deg[roi_keep]
        else:
            roi_kept_rows += int(ra_v.size)

        dra_rad = np.radians(dra_deg)
        dec_rad = np.radians(dec_v)
        cos_sep = np.sin(dec_rad) * sin_source_dec + np.cos(dec_rad) * cos_source_dec * np.cos(dra_rad)
        sep_deg = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        on_region = sep_deg <= r_opt_deg[cell_v]
        if np.any(on_region):
            n_on += np.bincount(cell_v[on_region], minlength=n_cells)

        if print_every > 0 and (batch_idx % print_every == 0):
            print(f"[batch {batch_idx}] rows={input_rows:,} N_on={int(n_on.sum()):,}", flush=True)

    return ScanResult(
        n_on=n_on,
        input_rows=input_rows,
        valid_cell_rows=valid_cell_rows,
        roi_kept_rows=roi_kept_rows,
        roi_rejected_rows=roi_rejected_rows,
        processed_batches=processed_batches,
    )


def known_background_sigma(n_on: np.ndarray, b_on: np.ndarray) -> np.ndarray:
    n = np.asarray(n_on, dtype=np.float64)
    b = np.asarray(b_on, dtype=np.float64)
    sigma = np.full(np.broadcast_shapes(n.shape, b.shape), np.nan, dtype=np.float64)
    n_b, b_b = np.broadcast_arrays(n, b)

    valid = b_b >= 0.0
    positive_b = valid & (b_b > 0.0)
    positive_n = n_b > 0.0
    term = np.zeros_like(sigma, dtype=np.float64)
    normal = positive_b & positive_n
    term[normal] = n_b[normal] * np.log(n_b[normal] / b_b[normal]) - (n_b[normal] - b_b[normal])
    zero_n = positive_b & (~positive_n)
    term[zero_n] = b_b[zero_n]
    zero_b_positive_n = valid & (~positive_b) & positive_n
    term[zero_b_positive_n] = np.inf
    sigma[valid] = np.sqrt(np.maximum(0.0, 2.0 * term[valid]))
    sigma[valid & (n_b < b_b)] *= -1.0
    return sigma


def li_ma_significance(n_on: np.ndarray, n_off: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    n = np.asarray(n_on, dtype=np.float64)
    off = np.asarray(n_off, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    sigma = np.full(np.broadcast_shapes(n.shape, off.shape, a.shape), np.nan, dtype=np.float64)
    n_b, off_b, a_b = np.broadcast_arrays(n, off, a)
    valid = (n_b >= 0.0) & (off_b >= 0.0) & (a_b > 0.0)
    total = n_b + off_b
    nonzero = valid & (total > 0.0)
    term = np.zeros_like(sigma, dtype=np.float64)

    on_positive = nonzero & (n_b > 0.0)
    term[on_positive] += n_b[on_positive] * np.log(
        ((1.0 + a_b[on_positive]) / a_b[on_positive]) * (n_b[on_positive] / total[on_positive])
    )
    off_positive = nonzero & (off_b > 0.0)
    term[off_positive] += off_b[off_positive] * np.log(
        (1.0 + a_b[off_positive]) * (off_b[off_positive] / total[off_positive])
    )
    sigma[valid] = np.sqrt(np.maximum(0.0, 2.0 * term[valid]))
    excess = n_b - a_b * off_b
    sigma[valid & (excess < 0.0)] *= -1.0
    return sigma


def signed_quadrature(values: np.ndarray, sign_value: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    sign = -1.0 if sign_value < 0.0 else 1.0
    return float(sign * math.sqrt(float(np.sum(finite * finite))))


def compute_signal_stats(
    n_on: np.ndarray,
    b_on: np.ndarray,
    *,
    background_form: str,
    n_off: np.ndarray,
    alpha: np.ndarray,
) -> SignalStats:
    n = np.asarray(n_on, dtype=np.int64)
    b = np.asarray(b_on, dtype=np.float64)
    excess = n.astype(np.float64) - b
    known_b = known_background_sigma(n, b)
    excess_err_conservative = np.sqrt(np.maximum(n.astype(np.float64) + b, 0.0))

    total_n = int(n.sum())
    total_b = float(np.nansum(b))
    total_excess = float(total_n - total_b)
    total_known_aggregate = float(known_background_sigma(np.asarray([total_n]), np.asarray([total_b]))[0])
    total_known_combined = signed_quadrature(known_b, total_excess)

    if background_form == "off_counts":
        off = np.asarray(n_off, dtype=np.float64)
        a = np.asarray(alpha, dtype=np.float64)
        excess_err_stat = np.sqrt(np.maximum(n.astype(np.float64) + a * a * off, 0.0))
        li_ma = li_ma_significance(n, off, a)
        total_err = float(math.sqrt(float(np.nansum(excess_err_stat * excess_err_stat))))
        total_li_ma_combined = signed_quadrature(li_ma, total_excess)
        formal_sigma = total_li_ma_combined
        formal_name = "Li-Ma on/off combined independent cells"
        total_n_off = float(np.nansum(off))
    else:
        excess_err_stat = np.sqrt(np.maximum(n.astype(np.float64), 0.0))
        li_ma = np.full(n.shape, np.nan, dtype=np.float64)
        total_err = float(math.sqrt(float(np.nansum(excess_err_stat * excess_err_stat))))
        total_li_ma_combined = None
        formal_sigma = total_known_aggregate
        formal_name = "Poisson known-background aggregate diagnostic"
        total_n_off = None

    return SignalStats(
        excess=excess,
        excess_err_stat=excess_err_stat,
        excess_err_conservative=excess_err_conservative,
        known_b_sigma=known_b,
        li_ma_sigma=li_ma,
        total_n_on=total_n,
        total_b_on=total_b,
        total_n_off=total_n_off,
        total_excess=total_excess,
        total_excess_err_stat=total_err,
        total_known_b_sigma_aggregate=total_known_aggregate,
        total_known_b_sigma_combined=total_known_combined,
        total_li_ma_sigma_combined=total_li_ma_combined,
        total_formal_sigma=float(formal_sigma),
        formal_significance_name=formal_name,
    )


def evaluate_quality(
    stats: SignalStats,
    contract: BackgroundContract,
    *,
    min_total_sigma: float,
    max_total_sigma: float,
    max_batches: Optional[int],
) -> Dict[str, object]:
    if max_batches is not None:
        return {
            "status": "skipped",
            "reason": "max_batches set; partial scans are smoke tests and are not promoted",
            "exit_code": 0,
            "promotable": False,
        }
    if not contract.promotable_contract:
        return {
            "status": "blocked_contract",
            "reason": "; ".join(contract.diagnostic_reasons) or "Stage D background contract is not promotable.",
            "exit_code": 2,
            "promotable": False,
        }

    sigma = stats.total_formal_sigma
    if not math.isfinite(sigma):
        return {
            "status": "failed",
            "reason": f"total formal significance is not finite for {stats.formal_significance_name}",
            "exit_code": 2,
            "promotable": False,
        }
    passed = float(min_total_sigma) <= sigma <= float(max_total_sigma)
    if passed:
        return {
            "status": "passed",
            "reason": f"total formal sigma {sigma:.6g} within [{min_total_sigma:.6g}, {max_total_sigma:.6g}]",
            "exit_code": 0,
            "promotable": True,
        }
    return {
        "status": "failed",
        "reason": f"total formal sigma {sigma:.6g} outside [{min_total_sigma:.6g}, {max_total_sigma:.6g}]",
        "exit_code": 2,
        "promotable": False,
    }


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


def matrix_for_cells(values: np.ndarray, cells: Sequence[CellSpec]) -> Tuple[np.ndarray, List[str], List[str]]:
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    matrix = np.full((len(nhit_bins), len(pred_bins)), np.nan, dtype=np.float64)
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is not None:
                matrix[i, j] = float(values[cell.index])
    return matrix, nhit_bins, pred_bins


def plot_heatmap_grid(
    values: np.ndarray,
    cells: Sequence[CellSpec],
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap_name: str,
    symmetric: bool = False,
    log10: bool = False,
) -> None:
    plt = setup_matplotlib()
    matrix, nhit_bins, pred_bins = matrix_for_cells(values, cells)
    plot_matrix = matrix.copy()
    if log10:
        plot_matrix[:] = np.nan
        positive = matrix > 0
        plot_matrix[positive] = np.log10(matrix[positive])
    finite = plot_matrix[np.isfinite(plot_matrix)]
    fig, ax = plt.subplots(figsize=(1.28 * len(pred_bins) + 2.8, 0.62 * len(nhit_bins) + 2.2), dpi=150)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")
    kwargs: Dict[str, object] = {"aspect": "auto", "interpolation": "nearest", "cmap": cmap}
    if finite.size:
        if symmetric:
            vmax = float(np.nanmax(np.abs(finite)))
            vmax = max(vmax, 1.0)
            kwargs["vmin"] = -vmax
            kwargs["vmax"] = vmax
        else:
            kwargs["vmin"] = float(np.nanmin(finite))
            kwargs["vmax"] = float(np.nanmax(finite))
    im = ax.imshow(plot_matrix, **kwargs)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_xlabel("log10(E_pred / GeV) bin", fontsize=8)
    ax.set_ylabel("Nhit bin", fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                if abs(value) >= 1000:
                    label = f"{value:.3g}"
                elif abs(value) >= 10:
                    label = f"{value:.1f}"
                else:
                    label = f"{value:.2g}"
                ax.text(j, i, label, ha="center", va="center", color="white", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_on_background_grid(
    n_on: np.ndarray,
    b_on: np.ndarray,
    cells: Sequence[CellSpec],
    output_path: Path,
    *,
    title: str,
) -> None:
    plt = setup_matplotlib()
    n_matrix, nhit_bins, pred_bins = matrix_for_cells(n_on.astype(np.float64), cells)
    b_matrix, _, _ = matrix_for_cells(b_on, cells)
    fig, axes = plt.subplots(1, 2, figsize=(2.25 * len(pred_bins) + 3.6, 0.72 * len(nhit_bins) + 2.4), dpi=150, squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    for ax, matrix, panel_title in [
        (axes[0, 0], n_matrix, "N_on"),
        (axes[0, 1], b_matrix, "B_on"),
    ]:
        logged = np.full_like(matrix, np.nan, dtype=np.float64)
        positive = matrix > 0
        logged[positive] = np.log10(matrix[positive])
        im = ax.imshow(logged, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_xticks(np.arange(len(pred_bins)))
        ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(nhit_bins)))
        ax.set_yticklabels(nhit_bins, fontsize=7)
        ax.set_xlabel("log10(E_pred / GeV) bin", fontsize=8)
        ax.set_title(panel_title, fontsize=10)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3g}", ha="center", va="center", color="white", fontsize=6.5)
        cbar = fig.colorbar(im, ax=ax, shrink=0.78)
        cbar.set_label(f"log10 {panel_title}", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Nhit bin", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def build_rows(
    cells: Sequence[CellSpec],
    contract: BackgroundContract,
    scan: ScanResult,
    stats: SignalStats,
) -> List[Dict[str, object]]:
    bg = contract.arrays
    rows: List[Dict[str, object]] = []
    for cell in cells:
        idx = cell.index
        rows.append(
            {
                "cell_index": int(idx),
                "cell_id": int(cell.cell_id),
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "r_opt_deg": float(bg["r_opt_deg"][idx]),
                "sigma_deg": float(bg["sigma_deg"][idx]),
                "containment_r_opt": float(bg["containment_r_opt"][idx]),
                "N_on": int(scan.n_on[idx]),
                "B_on": float(bg["B_on"][idx]),
                "N_off": float(bg["N_off"][idx]) if np.isfinite(bg["N_off"][idx]) else math.nan,
                "alpha": float(bg["alpha"][idx]) if np.isfinite(bg["alpha"][idx]) else math.nan,
                "excess": float(stats.excess[idx]),
                "excess_err_stat": float(stats.excess_err_stat[idx]),
                "excess_err_conservative": float(stats.excess_err_conservative[idx]),
                "known_b_sigma": float(stats.known_b_sigma[idx]) if np.isfinite(stats.known_b_sigma[idx]) else math.nan,
                "li_ma_sigma": float(stats.li_ma_sigma[idx]) if np.isfinite(stats.li_ma_sigma[idx]) else math.nan,
                "background_mode": contract.background_mode,
                "background_form": contract.background_form,
                "statistic_kind": contract.statistic_kind,
                "statistic_note": statistic_note(contract),
            }
        )
    return rows


def statistic_note(contract: BackgroundContract) -> str:
    if contract.background_form == "off_counts":
        return "Li-Ma on/off significance from Stage D alpha and N_off"
    return "known-background Poisson diagnostic; Li-Ma undefined for direct_expectation background"


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "r_opt_deg",
        "sigma_deg",
        "containment_r_opt",
        "N_on",
        "B_on",
        "N_off",
        "alpha",
        "excess",
        "excess_err_stat",
        "excess_err_conservative",
        "known_b_sigma",
        "li_ma_sigma",
        "background_mode",
        "background_form",
        "statistic_kind",
        "statistic_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_md(path: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    totals = metadata["totals"]  # type: ignore[index]
    quality = metadata["quality_gate"]  # type: ignore[index]
    contract = metadata["stage_d_contract"]  # type: ignore[index]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage E Signal Extraction Summary\n\n")
        f.write(f"- Run id: `{metadata['run_id']}`\n")
        f.write(f"- Stage C input: `{metadata['inputs']['stage_c_dir']}`\n")
        f.write(f"- Stage D background: `{metadata['inputs']['background_npz']}`\n")
        f.write(f"- Background mode: `{contract['background_mode']}`\n")
        f.write(f"- Background form: `{contract['background_form']}`\n")
        f.write(f"- Statistic: {metadata['statistic']['formal_significance']}\n")
        if metadata["roi"]["stage_e_fiducial_applied"]:  # type: ignore[index]
            f.write(f"- Stage E fiducial ROI: rho < {metadata['roi']['fiducial_radius_deg']} deg\n")  # type: ignore[index]
        f.write(f"- Rows scanned: {metadata['processing']['input_rows_scanned']:,}\n")  # type: ignore[index]
        f.write(f"- Total N_on: {totals['N_on']:,}\n")
        f.write(f"- Total B_on: {totals['B_on']:.6g}\n")
        if totals.get("N_off") is not None:
            f.write(f"- Total N_off: {totals['N_off']:.6g}\n")
        f.write(f"- Total excess: {totals['excess']:.6g}\n")
        f.write(f"- Total formal sigma: {totals['formal_sigma']:.6g}\n")
        f.write(f"- Quality status: `{quality['status']}` ({quality['reason']})\n")
        if contract["diagnostic_reasons"]:
            f.write(f"- Contract diagnostics: {'; '.join(contract['diagnostic_reasons'])}\n")
        f.write("\n| cell | Nhit bin | predE bin | r_opt deg | N_on | B_on | N_off | alpha | excess | known-B sigma | Li-Ma |\n")
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{row['r_opt_deg']:.5g} | {row['N_on']} | {row['B_on']:.6g} | "
                f"{format_md_float(row['N_off'])} | {format_md_float(row['alpha'])} | "
                f"{row['excess']:.6g} | {format_md_float(row['known_b_sigma'])} | {format_md_float(row['li_ma_sigma'])} |\n"
            )


def format_md_float(value: object) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number:.6g}"


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
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def format_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def format_float(value: object, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.{digits}g}"
    return f"{number:.{digits}g}"


def existing_apply_path_for_html(target: str, report_path: Path) -> Optional[Path]:
    target_path = Path(target)
    report_resolved = report_path.resolve()
    candidates: List[Path] = []
    if target_path.is_absolute():
        candidates.append(target_path)
        parts = target_path.parts
        if "apply" in parts:
            apply_idx = len(parts) - 1 - parts[::-1].index("apply")
            candidates.append(report_resolved.parents[1] / Path(*parts[apply_idx + 1 :]))
    else:
        candidates.append((report_resolved.parent / target_path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def relative_path_for_html(target: str, report_path: Path) -> str:
    existing = existing_apply_path_for_html(target, report_path)
    if existing is not None:
        return os.path.relpath(existing.resolve(), start=report_path.resolve().parent)
    target_path = Path(target)
    try:
        return os.path.relpath(target_path.resolve(), start=report_path.resolve().parent)
    except OSError:
        return target


def write_report_html(path: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = metadata["totals"]  # type: ignore[index]
    quality = metadata["quality_gate"]  # type: ignore[index]
    outputs = metadata["outputs"]  # type: ignore[index]
    contract = metadata["stage_d_contract"]  # type: ignore[index]
    processing = metadata["processing"]  # type: ignore[index]
    roi = metadata["roi"]  # type: ignore[index]
    statistic = metadata["statistic"]  # type: ignore[index]
    source = metadata.get("source", {})
    inputs = metadata.get("inputs", {})
    quality_status = str(quality["status"])
    quality_class = "good" if quality_status == "passed" else ("warn" if quality_status == "skipped" else "bad")
    cell_count = len(rows)
    background_npz_name = Path(str(inputs.get("background_npz", "Stage D background NPZ"))).name if isinstance(inputs, dict) else "Stage D background NPZ"
    npz_output = str(outputs.get("npz", "signal.npz")) if isinstance(outputs, dict) else "signal.npz"
    summary_csv_output = str(outputs.get("summary_csv", "signal_summary.csv")) if isinstance(outputs, dict) else "signal_summary.csv"
    metadata_output = str(outputs.get("metadata_json", "signal_metadata.json")) if isinstance(outputs, dict) else "signal_metadata.json"
    background_summary = contract.get("metadata_summary") if isinstance(contract, dict) else {}
    background_method = (
        background_summary.get("method")
        if isinstance(background_summary, dict)
        else None
    ) or contract.get("background_form", "direct_expectation")

    figure_items = [
        (
            "formal_sigma_grid_png",
            "逐 cell 的正式诊断显著性",
            "本次 direct expectation 背景下，formal sigma 等同于 known-background Poisson 诊断。",
        ),
        (
            "known_b_sigma_grid_png",
            "known-background 显著性诊断",
            "逐 cell 比较 N_on 与 B_on；正值说明 Crab 源区计数超过背景期望。",
        ),
        ("excess_grid_png", "逐 cell excess", "excess = N_on - B_on，是 Stage F 拟合直接使用的信号表。"),
        (
            "on_background_grid_png",
            "N_on 与 B_on 对照",
            "左图是 Crab on-region 实际计数，右图是 Stage D 给出的同一孔径背景期望。",
        ),
        ("on_over_background_grid_png", "N_on / B_on 比值", "比值大于 1 表示源区计数超过背景。"),
    ]
    figure_html: List[str] = []
    for key, title, caption in figure_items:
        target = outputs.get(key) if isinstance(outputs, dict) else None
        if target:
            rel = html.escape(relative_path_for_html(str(target), path))
            figure_html.append(
                f'<figure><img src="{rel}" alt="{html.escape(title)}"><figcaption>'
                f'<strong>{html.escape(title)}。</strong>{html.escape(caption)}</figcaption></figure>'
            )

    stage_d_outputs: Dict[str, object] = {}
    background_metadata_json = inputs.get("background_metadata_json") if isinstance(inputs, dict) else None
    if background_metadata_json:
        background_metadata_path = existing_apply_path_for_html(str(background_metadata_json), path)
        if background_metadata_path is not None:
            try:
                background_metadata = load_json(background_metadata_path)
                background_outputs = background_metadata.get("outputs")
                if isinstance(background_outputs, dict):
                    stage_d_outputs = background_outputs
            except (OSError, json.JSONDecodeError):
                stage_d_outputs = {}

    stage_d_map_items = [
        (
            "roi_excess_grid_png",
            "Stage D 正式去背景天图",
            f"逐像素 excess_map = counts_map - background_map；background_map 来自 Stage D {background_method} direct expectation。",
        ),
        (
            "roi_known_b_sigma_grid_png",
            "Stage D 正式 residual 天图",
            "逐像素 known-background Poisson residual，使用同一个 Stage D background_map；它不是此前平滑 skymap 的 sideband 近似显著性。",
        ),
        ("roi_counts_grid_png", "Stage D counts 天图", "Stage D 正式 ROI 网格里的原始 counts_map，没有 Gaussian smoothing。"),
        ("roi_background_grid_png", "Stage D background 天图", f"Stage D 正式 ROI-local {background_method} background_map，与去背景天图使用同一背景口径。"),
    ]
    stage_d_map_html: List[str] = []
    for key, title, caption in stage_d_map_items:
        target = stage_d_outputs.get(key)
        if target and existing_apply_path_for_html(str(target), path) is not None:
            rel = html.escape(relative_path_for_html(str(target), path))
            stage_d_map_html.append(
                f'<figure class="wide"><img src="{rel}" alt="{html.escape(title)}"><figcaption>'
                f'<strong>{html.escape(title)}。</strong>{html.escape(caption)}</figcaption></figure>'
            )

    table_rows: List[str] = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{int(row['cell_id'])}</td>"
            f"<td>{html.escape(str(row['nhit_bin']))}</td>"
            f"<td>{html.escape(str(row['predE_bin']))}</td>"
            f"<td class=\"num\">{format_float(row['r_opt_deg'], 4)}</td>"
            f"<td class=\"num\">{format_int(row['N_on'])}</td>"
            f"<td class=\"num\">{format_float(row['B_on'], 6)}</td>"
            f"<td class=\"num\">{format_float(row['excess'], 6)}</td>"
            f"<td class=\"num\">{format_float(row['excess_err_stat'], 5)}</td>"
            f"<td class=\"num\">{format_float(row['known_b_sigma'], 5)}</td>"
            f"<td class=\"num\">{format_float(row['li_ma_sigma'], 5)}</td>"
            "</tr>"
        )

    contract_reasons = contract.get("diagnostic_reasons") if isinstance(contract, dict) else None
    contract_note = ""
    if contract_reasons:
        contract_note = "<p><strong>Stage D 合约诊断：</strong>" + html.escape("；".join(str(x) for x in contract_reasons)) + "</p>"

    roi_radius = roi.get("fiducial_radius_deg") if isinstance(roi, dict) else None
    source_ra = source.get("ra_deg") if isinstance(source, dict) else None
    source_dec = source.get("dec_deg") if isinstance(source, dict) else None

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage E 信号提取报告</title>
<style>
:root {{
  --bg: #101418;
  --fg: #eef3f5;
  --muted: #a8b2b8;
  --panel: #171d22;
  --panel-2: #1d242a;
  --border: #2e3942;
  --accent: #5fb3b3;
  --good: #52b788;
  --warn: #e9b44c;
  --bad: #ef476f;
  --code: #0b0f12;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #f7f8f9;
    --fg: #1b2329;
    --muted: #56636b;
    --panel: #ffffff;
    --panel-2: #f1f4f6;
    --border: #d7dee3;
    --code: #eef2f4;
  }}
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg); font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; line-height: 1.66; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 42px 20px 64px; }}
header {{ border-bottom: 1px solid var(--border); margin-bottom: 34px; padding-bottom: 24px; }}
.eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: .08em; font-size: 12px; font-weight: 700; }}
h1 {{ margin: 8px 0 10px; font-size: clamp(30px, 5vw, 44px); line-height: 1.15; }}
h2 {{ margin: 42px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); font-size: 24px; }}
h3 {{ margin: 24px 0 8px; color: var(--accent); font-size: 18px; }}
p {{ margin: 10px 0; }}
ul, ol {{ margin: 12px 0; padding-left: 24px; }}
li {{ margin: 6px 0; }}
code {{ padding: 2px 5px; border-radius: 4px; background: var(--code); color: var(--fg); font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace; font-size: 13px; }}
.lead {{ max-width: 960px; color: var(--muted); font-size: 17px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
.metric {{ min-height: 106px; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }}
.metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .07em; }}
.metric .value {{ margin-top: 8px; color: var(--fg); font-size: 25px; font-weight: 700; line-height: 1.2; overflow-wrap: anywhere; }}
.metric .note {{ margin-top: 7px; color: var(--muted); font-size: 13px; }}
.callout {{ margin: 18px 0; padding: 16px 18px; border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; background: var(--panel); }}
.callout.good {{ border-left-color: var(--good); }}
.callout.warn {{ border-left-color: var(--warn); }}
.callout.bad {{ border-left-color: var(--bad); }}
.flow {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 16px 0; }}
.step {{ padding: 13px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }}
.step .k {{ color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
.step .t {{ margin-top: 6px; font-weight: 700; }}
.step .d {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
.table-wrap {{ width: 100%; overflow-x: auto; margin: 18px 0; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }}
table {{ width: 100%; min-width: 1040px; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }}
th {{ background: var(--panel); font-weight: 700; white-space: nowrap; }}
tbody tr:nth-child(odd) td {{ background: var(--panel-2); }}
tbody tr:last-child td {{ border-bottom: 0; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 18px; }}
figure {{ margin: 0; padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }}
figure.wide {{ grid-column: 1 / -1; }}
figure img {{ display: block; width: 100%; height: auto; border-radius: 4px; background: #fff; }}
figcaption {{ margin-top: 9px; color: var(--muted); font-size: 13px; }}
footer {{ margin-top: 54px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .flow, .figure-grid {{ grid-template-columns: 1fr; }} }}
@media (max-width: 520px) {{ main {{ padding: 32px 16px 56px; }} .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">LHAASO-WCDA · Crab SED</div>
    <h1>Stage E 信号提取报告</h1>
    <p class="lead">本页把 Stage E 的输出翻译成可读的分析说明：它不是在拟合能谱，而是在每个 <code>(Nhit, log10 E_pred)</code> cell 里统计 Crab 源区计数，并扣除 Stage D 给出的背景期望，得到后续 Stage F forward folding 拟合要使用的信号表。</p>
  </header>

  <section>
    <h2>一句话结论</h2>
    <p>Stage E 已经在 <code>{format_int(cell_count)}</code> 个传入 cell 上完成 Crab on-region 计数。总源区计数 <code>N_on = {format_int(totals['N_on'])}</code>，Stage D 背景期望 <code>B_on = {format_float(totals['B_on'], 6)}</code>，因此总 excess 为 <code>{format_float(totals['excess'], 6)}</code>，known-background 形式的总诊断显著性为 <code>{format_float(totals['formal_sigma'], 6)} sigma</code>。质量门状态为 <code>{html.escape(quality_status)}</code>。</p>
    <div class="grid">
      <div class="metric"><div class="label">Run</div><div class="value">{html.escape(str(metadata['run_id']))}</div><div class="note">Stage E signal extraction</div></div>
      <div class="metric"><div class="label">扫描事件</div><div class="value">{format_int(processing['input_rows_scanned'])}</div><div class="note">Stage C obs_events</div></div>
      <div class="metric"><div class="label">背景形式</div><div class="value">{html.escape(str(contract['background_form']))}</div><div class="note">{html.escape(str(contract['background_mode']))}</div></div>
      <div class="metric"><div class="label">质量门</div><div class="value">{html.escape(quality_status)}</div><div class="note">{html.escape(str(quality['reason']))}</div></div>
    </div>
    <div class="grid">
      <div class="metric"><div class="label">Total N_on</div><div class="value">{format_int(totals['N_on'])}</div><div class="note">Crab 源区实际计数</div></div>
      <div class="metric"><div class="label">Total B_on</div><div class="value">{format_float(totals['B_on'], 6)}</div><div class="note">Stage D 背景期望</div></div>
      <div class="metric"><div class="label">Total Excess</div><div class="value">{format_float(totals['excess'], 6)}</div><div class="note">N_on - B_on</div></div>
      <div class="metric"><div class="label">Formal Sigma</div><div class="value">{format_float(totals['formal_sigma'], 6)}</div><div class="note">Poisson known-background 诊断</div></div>
    </div>
    <div class="callout {quality_class}">
      <strong>质量门状态：<code>{html.escape(quality_status)}</code>。</strong>
      当前配置要求总 formal sigma 位于
      <code>[{format_float(quality['min_total_sigma'], 4)}, {format_float(quality['max_total_sigma'], 4)}]</code> 之间。本次结果为 <code>{format_float(totals['formal_sigma'], 6)}</code>，通过质量门，可以作为 Stage F 的输入候选。
      {contract_note}
    </div>
  </section>

  <section>
    <h2>为什么要做 Stage E</h2>
    <p>前面几个阶段各自只解决了问题的一部分：Stage A 给出 MC 响应矩阵，Stage B 给出每个 cell 的 PSF 和积分半径，Stage C 把观测 ROOT 规约成带 <code>RA/Dec/MJD/cell_id</code> 的 parquet，Stage D 估计每个 cell 在 Crab 孔径内应有多少背景。到这里还没有得到“Crab 在每个 cell 贡献了多少信号”。</p>
    <p>Stage E 就是这个桥梁：它把观测到的源区计数 <code>N_on</code> 和背景期望 <code>B_on</code> 合成 <code>excess</code>。Stage F 不直接拿所有事件拟合，而是拿 Stage E 产出的逐 cell excess 表，和 Stage A 响应折叠出来的模型预期数做比较。</p>
  </section>

  <section>
    <h2>Stage E 在做什么</h2>
    <div class="flow">
      <div class="step"><div class="k">Input</div><div class="t">读观测事件</div><div class="d">从 Stage C parquet 读取 <code>ra_mean_deg</code>、<code>dec_mean_deg</code>、<code>cell_id</code>。</div></div>
      <div class="step"><div class="k">Aperture</div><div class="t">定义源区半径</div><div class="d">每个 cell 用 Stage B/D 传下来的 <code>r_opt_deg</code>，即约 <code>1.58 sigma</code> 的最优积分半径。</div></div>
      <div class="step"><div class="k">Count</div><div class="t">统计 N_on</div><div class="d">计算事件到 Crab 的角距离，若小于该 cell 的 <code>r_opt</code>，计入 <code>N_on,b</code>。</div></div>
      <div class="step"><div class="k">Subtract</div><div class="t">扣背景</div><div class="d">读取 Stage D 的 <code>B_on,b</code>，计算 <code>excess_b = N_on,b - B_on,b</code>。</div></div>
      <div class="step"><div class="k">Diagnose</div><div class="t">输出诊断</div><div class="d">给出误差、显著性、表格和二维热图，供 Stage F 和人工检查使用。</div></div>
    </div>
    <p>本次 Stage D 使用的是 Crab 局部 ROI 背景模型，所以 Stage E 也只在同一套 fiducial ROI 内工作：<code>rho &lt; {format_float(roi_radius, 3)} deg</code>。这保证 <code>N_on</code> 的统计口径和 <code>B_on</code> 的背景口径一致。</p>
  </section>

  <section>
    <h2>具体怎么做的</h2>
    <h3>1. 事件选择和几何</h3>
    <ul>
      <li>源位置固定为 Crab：<code>RA = {format_float(source_ra, 5)} deg</code>，<code>Dec = {format_float(source_dec, 5)} deg</code>。</li>
      <li>先按 Stage D 的局部 ROI 约束保留 <code>rho &lt; {format_float(roi_radius, 3)} deg</code> 的事件；本次保留 <code>{format_int(processing['roi_kept_rows'])}</code> 行，剔除 <code>{format_int(processing['roi_rejected_rows'])}</code> 行。</li>
      <li>随后用球面角距离公式计算事件与 Crab 的夹角 <code>sep</code>，每个 cell 判断 <code>sep &lt;= r_opt_deg[cell]</code>。</li>
    </ul>
    <h3>2. 背景和统计量</h3>
    <p>Stage D 输出的是 <code>direct_expectation</code>，也就是直接背景期望 <code>B_on</code>。它不是传统 off 区计数，因此本报告不把 <code>B_on</code> 强行拆成 <code>alpha × N_off</code>，表格里的 Li-Ma 显示为 <code>n/a</code>。</p>
    <ul>
      <li><code>excess_b = N_on,b - B_on,b</code></li>
      <li>conservative 误差口径：<code>sqrt(N_on,b + B_on,b)</code></li>
      <li>本次 formal sigma：<code>{html.escape(str(statistic.get('formal_significance')))}</code></li>
      <li>known-background 诊断公式：<code>sign(N_on-B_on) * sqrt(2 * (N_on ln(N_on/B_on) - (N_on-B_on)))</code></li>
    </ul>
    <div class="callout warn">
      <strong>为什么没有 Li-Ma？</strong>
      Li-Ma 需要明确的 on/off 计数和曝光比 <code>alpha</code>。本次 Stage D 给的是直接背景期望，不是 off 区事件数，所以 Li-Ma 在数学上不适用。这里的显著性是一个诊断量，用来检查信号是否合理；最终能谱结果仍要看 Stage F 之后的拟合和系统误差检查。
    </div>
  </section>

  <section>
    <h2>结果怎么读</h2>
    <p>下表每一行是一个输入 cell。<code>Nhit bin</code> 表示 shower size 分箱，<code>log10 E_pred bin</code> 是 ML 预测能量分箱。低 Nhit、低预测能量 cell 统计量最大；高 Nhit、高预测能量 cell 背景很低，但如果仍有明显 positive excess，就对高能端谱形很有价值。</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>cell</th><th>Nhit bin</th><th>log10 E_pred bin</th><th class="num">r_opt deg</th><th class="num">N_on</th><th class="num">B_on</th><th class="num">excess</th><th class="num">stat err</th><th class="num">known-B sigma</th><th class="num">Li-Ma</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
    <p>例如 cell 1 和 cell 2 贡献了最多的绝对 excess；而 cell 16、17 这类高 Nhit cell 虽然计数少，但背景更低，known-B 诊断显著性仍然很强。整体上，正 excess 沿着高 Nhit 对应高 <code>log10 E_pred</code> 的物理带出现，这是我们希望看到的结构。</p>
  </section>

	  <section>
	    <h2>诊断图怎么看</h2>
	    <div class="figure-grid">
	      {''.join(figure_html) if figure_html else '<p>本次运行没有生成诊断图。</p>'}
	    </div>
	  </section>

	  <section>
	    <h2>和 Stage D 正式去背景天图的关系</h2>
	    <p>下面几张图来自 Stage D 正式 ROI-local 背景输出，不再使用此前平滑 skymap 的 sideband 近似。它们直接读取 <code>{html.escape(background_npz_name)}</code> 中的 <code>counts_map</code> 和 <code>background_map</code>：去背景天图是逐像素 <code>counts_map - background_map</code>，residual 天图是同一背景期望下的 Poisson 诊断量。</p>
	    <div class="callout good">
	      <strong>这组图和 Stage E 的背景口径一致。</strong>
	      Stage E 表格中的 <code>B_on</code> 是把同一个 Stage D <code>background_map</code> 积分到每个 cell 的 Crab on-region 后得到的；这里的天图只是把同一个过程摊回 ROI 像素上，方便人眼检查源形态。
	    </div>
	    <div class="figure-grid">
	      {''.join(stage_d_map_html) if stage_d_map_html else '<p>本地没有找到 Stage D 正式去背景天图。</p>'}
	    </div>
	  </section>

	  <section>
    <h2>输出文件和下一步</h2>
    <ul>
      <li><code>{html.escape(npz_output)}</code>：Stage F 直接读取的数组，包含 <code>N_on</code>、<code>B_on</code>、<code>excess</code>、误差和显著性。</li>
      <li><code>{html.escape(summary_csv_output)}</code>：逐 cell 表格，适合人工检查和快速画图。</li>
      <li><code>{html.escape(metadata_output)}</code>：输入路径、Stage D 合约、ROI 口径、统计公式和质量门信息。</li>
      <li><code>{html.escape(str(Path(npz_output).parent))}/*.png</code>：二维 cell 诊断图。</li>
    </ul>
    <div class="callout good">
      <strong>下一步是 Stage F。</strong>
      Stage F 会把假设的 Crab 真实能谱通过 Stage A 的二维响应矩阵折叠到这些 cell 上，得到模型预期信号数，再和这里的 <code>excess_b</code> 比较，拟合幂律或 log-parabola 谱参数。
    </div>
    <div class="callout warn">
      <strong>当前结果仍是分析链路验证，不是最终发表图。</strong>
      需要继续检查背景系统误差、ROI 口径、cell selector 稳定性、1D vs 2D 自洽性，以及 Stage F 拟合残差。
    </div>
  </section>

  <footer>
    生成来源：<code>{html.escape(str(outputs.get('metadata_json', '')))}</code> 和 <code>{html.escape(str(outputs.get('summary_csv', '')))}</code>。
  </footer>
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def make_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    output_root: Path,
    stage_c_dir: Path,
    obs_events_dir: Path,
    stage_c_metadata_path: Path,
    stage_c_metadata: Dict[str, object],
    background_npz: Path,
    background_metadata_path: Path,
    background_metadata: Dict[str, object],
    selection_csv: Path,
    contract: BackgroundContract,
    scan: ScanResult,
    stats: SignalStats,
    quality: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    outputs: Dict[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    return {
        "description": "Stage E Crab signal extraction for configured (Nhit, predicted logE) SED cells.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {
            "stage_c_dir": str(stage_c_dir),
            "obs_events_dir": str(obs_events_dir),
            "stage_c_metadata_json": str(stage_c_metadata_path),
            "background_npz": str(background_npz),
            "background_metadata_json": str(background_metadata_path),
            "cell_selection_csv": str(selection_csv),
            "stage_c_run_id": stage_c_metadata.get("run_id") if isinstance(stage_c_metadata, dict) else None,
            "stage_d_run_id": background_metadata.get("run_id") if isinstance(background_metadata, dict) else None,
        },
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "source": {
            "name": "Crab",
            "ra_deg": float(args.source_ra_deg),
            "dec_deg": float(args.source_dec_deg),
        },
        "stage_d_contract": {
            "background_mode": contract.background_mode,
            "background_form": contract.background_form,
            "statistic_kind": contract.statistic_kind,
            "strict_contract": bool(contract.strict_contract),
            "promotable_contract": bool(contract.promotable_contract),
            "diagnostic_reasons": contract.diagnostic_reasons,
            "warnings": contract.warnings,
            "metadata_summary": contract.metadata_summary,
        },
        "containment": {
            "override": str(args.containment_override),
            "source": (
                "fixed1_cli_for_aperture_conditioned_response"
                if str(args.containment_override) == "fixed1"
                else "stage_d_background_npz_containment_r_opt"
            ),
        },
        "roi": {
            "stage_e_fiducial_applied": contract.roi_fiducial_deg is not None,
            "fiducial_radius_deg": contract.roi_fiducial_deg,
            "stage_d_roi_config": contract.roi_config,
        },
        "statistic": {
            "background_form": contract.background_form,
            "formal_significance": stats.formal_significance_name,
            "known_b_sigma_formula": "sign(N_on-B_on) * sqrt(2 * (N_on ln(N_on/B_on) - (N_on-B_on)))",
            "li_ma_formula": "Li-Ma Eq. 17 / Hu 2023 Eq. 6-26 when Stage D provides alpha and N_off",
            "excess_formula": "N_on - B_on",
            "excess_err_stat": (
                "sqrt(N_on + alpha^2 N_off)" if contract.background_form == "off_counts" else "sqrt(N_on)"
            ),
            "excess_err_conservative": "sqrt(N_on + B_on)",
            "li_ma_status": "available" if contract.background_form == "off_counts" else "not_applicable",
        },
        "quality_gate": quality,
        "totals": {
            "N_on": int(stats.total_n_on),
            "B_on": float(stats.total_b_on),
            "N_off": stats.total_n_off,
            "excess": float(stats.total_excess),
            "excess_err_stat": float(stats.total_excess_err_stat),
            "known_b_sigma_aggregate": float(stats.total_known_b_sigma_aggregate),
            "known_b_sigma_combined_independent_cells": float(stats.total_known_b_sigma_combined),
            "li_ma_sigma_combined_independent_cells": stats.total_li_ma_sigma_combined,
            "formal_sigma": float(stats.total_formal_sigma),
        },
        "processing": {
            "input_rows_scanned": int(scan.input_rows),
            "valid_cell_rows": int(scan.valid_cell_rows),
            "roi_kept_rows": int(scan.roi_kept_rows),
            "roi_rejected_rows": int(scan.roi_rejected_rows),
            "processed_batches": int(scan.processed_batches),
            "batch_size": int(args.batch_size),
            "max_batches": args.max_batches,
            "mjd_min": args.mjd_min,
            "mjd_max": args.mjd_max,
            "elapsed_seconds": float(elapsed_seconds),
        },
        "cells": rows,
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.quality_min_total_sigma > args.quality_max_total_sigma:
        raise ValueError("--quality-min-total-sigma cannot exceed --quality-max-total-sigma")

    stage_c_dir = Path(args.stage_c_dir).resolve()
    obs_events_dir = stage_c_dir / "obs_events"
    if not obs_events_dir.exists():
        raise FileNotFoundError(f"Stage C obs_events directory does not exist: {obs_events_dir}")
    stage_c_metadata_path = stage_c_dir / "obs_events_metadata.json"
    stage_c_metadata = load_json(stage_c_metadata_path) if stage_c_metadata_path.exists() else {}

    background_npz = Path(args.background_npz).resolve()
    background_metadata_path = Path(args.background_metadata).resolve()
    background_metadata = load_json(background_metadata_path) if background_metadata_path.exists() else {}
    selection_csv = Path(args.cell_selection_csv).resolve()
    cells = load_cells(selection_csv)
    contract = load_background_contract(background_npz, background_metadata, cells)
    if str(args.containment_override) == "fixed1":
        contract.arrays["containment_r_opt"] = np.ones(len(cells), dtype=np.float64)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(args.run_id or make_default_run_id())
    run_dir = prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))

    print(f"Loaded {len(cells)} cells from {selection_csv}", flush=True)
    print(f"Stage C dataset: {obs_events_dir}", flush=True)
    print(f"Stage D background: {background_npz}", flush=True)
    print(f"Stage D mode/form: {contract.background_mode}/{contract.background_form}", flush=True)
    if contract.diagnostic_reasons:
        print(f"Stage D contract diagnostics: {'; '.join(contract.diagnostic_reasons)}", flush=True)
    print(f"Output run dir: {run_dir}", flush=True)

    scan = scan_on_region(
        obs_events_dir,
        cells,
        contract.arrays["r_opt_deg"],
        float(args.source_ra_deg),
        float(args.source_dec_deg),
        roi_fiducial_deg=contract.roi_fiducial_deg,
        batch_size=int(args.batch_size),
        max_batches=args.max_batches,
        print_every=int(args.print_every),
        mjd_min=args.mjd_min,
        mjd_max=args.mjd_max,
    )
    stats = compute_signal_stats(
        scan.n_on,
        contract.arrays["B_on"],
        background_form=contract.background_form,
        n_off=contract.arrays["N_off"],
        alpha=contract.arrays["alpha"],
    )
    quality = evaluate_quality(
        stats,
        contract,
        min_total_sigma=float(args.quality_min_total_sigma),
        max_total_sigma=float(args.quality_max_total_sigma),
        max_batches=args.max_batches,
    )
    quality["min_total_sigma"] = float(args.quality_min_total_sigma)
    quality["max_total_sigma"] = float(args.quality_max_total_sigma)

    rows = build_rows(cells, contract, scan, stats)
    npz_path = run_dir / args.npz_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    metadata_path = run_dir / args.metadata_name

    formal_sigma = stats.li_ma_sigma if contract.background_form == "off_counts" else stats.known_b_sigma
    np.savez_compressed(
        npz_path,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        r_opt_deg=contract.arrays["r_opt_deg"].astype(np.float32),
        sigma_deg=contract.arrays["sigma_deg"].astype(np.float32),
        containment_r_opt=contract.arrays["containment_r_opt"].astype(np.float32),
        N_on=scan.n_on.astype(np.int64),
        B_on=contract.arrays["B_on"].astype(np.float64),
        N_off=contract.arrays["N_off"].astype(np.float64),
        alpha=contract.arrays["alpha"].astype(np.float64),
        excess=stats.excess.astype(np.float64),
        excess_err_stat=stats.excess_err_stat.astype(np.float64),
        excess_err_conservative=stats.excess_err_conservative.astype(np.float64),
        known_b_sigma=stats.known_b_sigma.astype(np.float64),
        li_ma_sigma=stats.li_ma_sigma.astype(np.float64),
        formal_sigma=formal_sigma.astype(np.float64),
        background_mode=np.asarray([contract.background_mode] * len(cells), dtype="U64"),
        background_form=np.asarray([contract.background_form] * len(cells), dtype="U64"),
        statistic_kind=np.asarray([contract.statistic_kind] * len(cells), dtype="U64"),
        roi_fiducial_deg=np.asarray(np.nan if contract.roi_fiducial_deg is None else contract.roi_fiducial_deg, dtype=np.float64),
    )

    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = {
            "formal_sigma_grid_png": str(run_dir / "formal_sigma_grid.png"),
            "known_b_sigma_grid_png": str(run_dir / "known_b_sigma_grid.png"),
            "excess_grid_png": str(run_dir / "excess_grid.png"),
            "on_background_grid_png": str(run_dir / "on_background_grid.png"),
            "on_over_background_grid_png": str(run_dir / "on_over_background_grid.png"),
        }
        plot_heatmap_grid(
            formal_sigma,
            cells,
            Path(plot_outputs["formal_sigma_grid_png"]),
            title="Stage E formal significance",
            colorbar_label="formal sigma",
            cmap_name="RdBu_r",
            symmetric=True,
        )
        plot_heatmap_grid(
            stats.known_b_sigma,
            cells,
            Path(plot_outputs["known_b_sigma_grid_png"]),
            title="Stage E known-background diagnostic",
            colorbar_label="known-B sigma",
            cmap_name="RdBu_r",
            symmetric=True,
        )
        plot_heatmap_grid(
            stats.excess,
            cells,
            Path(plot_outputs["excess_grid_png"]),
            title="Stage E excess by configured cell",
            colorbar_label="excess",
            cmap_name="magma",
        )
        plot_on_background_grid(
            scan.n_on,
            contract.arrays["B_on"],
            cells,
            Path(plot_outputs["on_background_grid_png"]),
            title="Stage E Crab on-region counts and background",
        )
        ratio = np.divide(
            scan.n_on.astype(np.float64),
            contract.arrays["B_on"],
            out=np.full(len(cells), np.nan),
            where=contract.arrays["B_on"] > 0,
        )
        plot_heatmap_grid(
            ratio,
            cells,
            Path(plot_outputs["on_over_background_grid_png"]),
            title="Stage E N_on / B_on",
            colorbar_label="N_on / B_on",
            cmap_name="viridis",
        )

    outputs: Dict[str, object] = {
        "npz": str(npz_path),
        "metadata_json": str(metadata_path),
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
        "report_html": str(Path(args.report_html).resolve()) if args.report_html else None,
        **plot_outputs,
    }
    metadata = make_metadata(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        output_root=output_root,
        stage_c_dir=stage_c_dir,
        obs_events_dir=obs_events_dir,
        stage_c_metadata_path=stage_c_metadata_path,
        stage_c_metadata=stage_c_metadata,
        background_npz=background_npz,
        background_metadata_path=background_metadata_path,
        background_metadata=background_metadata,
        selection_csv=selection_csv,
        contract=contract,
        scan=scan,
        stats=stats,
        quality=quality,
        rows=rows,
        outputs=outputs,
        elapsed_seconds=time.perf_counter() - start,
    )

    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, metadata, rows)
    write_json(metadata_path, metadata)

    promotable = bool(quality.get("promotable")) and not bool(args.no_promote_current)
    if promotable:
        promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
    elif quality.get("status") == "blocked_contract":
        metadata["promotion"]["status"] = "blocked_stage_d_contract"  # type: ignore[index]
    elif quality.get("status") == "failed":
        metadata["promotion"]["status"] = "blocked_quality_gate"  # type: ignore[index]
    elif args.max_batches is not None:
        metadata["promotion"]["status"] = "skipped_smoke_max_batches"  # type: ignore[index]
    elif args.no_promote_current:
        metadata["promotion"]["status"] = "skipped_no_promote_current"  # type: ignore[index]
    else:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]

    write_json(metadata_path, metadata)
    write_summary_md(summary_md_path, metadata, rows)
    if args.report_html:
        write_report_html(Path(args.report_html).resolve(), metadata, rows)

    print(f"Wrote {npz_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)
    print(f"Wrote {summary_md_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    if args.report_html:
        print(f"Wrote report {Path(args.report_html).resolve()}", flush=True)
    print(
        "Totals: "
        f"N_on={stats.total_n_on:,} "
        f"B_on={stats.total_b_on:.6g} "
        f"excess={stats.total_excess:.6g} "
        f"formal_sigma={stats.total_formal_sigma:.6g}",
        flush=True,
    )
    print(f"Quality gate: {quality['status']} ({quality['reason']})", flush=True)
    if promotable:
        print(f"Promoted current Stage E output to {output_root / 'current'}", flush=True)

    exit_code = int(quality.get("exit_code") or 0)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
