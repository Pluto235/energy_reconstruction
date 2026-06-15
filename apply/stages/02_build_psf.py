#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v1"
DEFAULT_CELL_SELECTION = "apply/config/cell_selection_v1.csv"
DEFAULT_STAGE_A_METADATA = "apply/output/stage_a/response_2d_metadata.json"
DEFAULT_OUTPUT_DIR = "apply/output/stage_b"
DEFAULT_TREE_NAME = "t_eventout"
DEFAULT_LHAASO_LAT_DEG = 29.45
DEFAULT_SOURCE_DEC_DEG = 22.01
DEFAULT_THETA_MAX_DEG = 50.0
RAYLEIGH_OPT_RADIUS_FACTOR = 1.58
RAYLEIGH_OPT_CONTAINMENT = 1.0 - math.exp(-0.5 * RAYLEIGH_OPT_RADIUS_FACTOR**2)


@dataclass(frozen=True)
class CellSpec:
    index: int
    cell_id: int
    nhit_bin: str
    predE_bin: str
    mc_count: int
    selection_version: str
    selection_reason: str


@dataclass
class CellEvents:
    dangle_rad: np.ndarray
    mc_theta_deg: np.ndarray
    mc_weight: np.ndarray
    loge_true: np.ndarray
    input_files: int
    angle_check_absdiff_rad: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage B Crab-declination PSF table for configured (Nhit, predicted-energy) cells."
    )
    parser.add_argument("--binned-root", type=str, default=DEFAULT_BINNED_ROOT)
    parser.add_argument("--cell-selection-csv", type=str, default=DEFAULT_CELL_SELECTION)
    parser.add_argument("--stage-a-metadata", type=str, default=DEFAULT_STAGE_A_METADATA)
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
        help="Allow deleting and recreating an existing run directory. Intended for smoke tests only.",
    )
    parser.add_argument("--tree-name", type=str, default=DEFAULT_TREE_NAME)
    parser.add_argument("--max-files-per-cell", type=int, default=None)
    parser.add_argument("--allow-missing-stage-a-metadata", action="store_true", default=False)
    parser.add_argument("--allow-missing-cell-dirs", action="store_true", default=False)
    parser.add_argument("--weight-branch", type=str, default="mc_weight")
    parser.add_argument("--allow-missing-weight", action="store_true", default=False)
    parser.add_argument(
        "--logE-min",
        type=float,
        default=None,
        help="Minimum log10(E_true/GeV) included in the PSF. Defaults to Stage A response edges when available.",
    )
    parser.add_argument(
        "--logE-max",
        type=float,
        default=None,
        help="Maximum log10(E_true/GeV) included in the PSF. Defaults to Stage A response edges when available.",
    )

    parser.add_argument("--lhaaso-lat-deg", type=float, default=DEFAULT_LHAASO_LAT_DEG)
    parser.add_argument("--source-dec-deg", type=float, default=DEFAULT_SOURCE_DEC_DEG)
    parser.add_argument("--theta-min-deg", type=float, default=0.0)
    parser.add_argument("--theta-max-deg", type=float, default=DEFAULT_THETA_MAX_DEG)
    parser.add_argument("--theta-step-deg", type=float, default=1.0)
    parser.add_argument("--hour-angle-samples", type=int, default=200000)
    parser.add_argument(
        "--allow-incomplete-theta-support",
        action="store_true",
        default=False,
        help="Permit Crab theta bins with no MC support. Intended for small smoke tests only.",
    )

    parser.add_argument("--min-events-per-cell", type=int, default=1000)
    parser.add_argument("--min-effective-events", type=float, default=200.0)
    parser.add_argument(
        "--allow-low-stat-psf-fallback",
        action="store_true",
        default=False,
        help=(
            "Write finite fallback PSF rows for cells that cannot support a PSF fit. "
            "Intended for full candidate ledgers where low-stat cells are excluded by a downstream selector."
        ),
    )
    parser.add_argument(
        "--core-fit-max-deg",
        type=float,
        default=3.0,
        help="Maximum radial residual included in the baseline Rayleigh core fit.",
    )
    parser.add_argument(
        "--theta-missing-mass-fail-threshold",
        type=float,
        default=0.10,
        help="Fail if missing Crab-positive theta support exceeds this probability mass in any cell.",
    )
    parser.add_argument("--containment-warning-tolerance", type=float, default=0.12)
    parser.add_argument("--angle-check-max-events", type=int, default=20000)
    parser.add_argument("--angle-check-warn-rad", type=float, default=1.0e-4)
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel cell workers. Each worker processes one selected cell at a time.",
    )

    parser.add_argument("--profile-max-deg", type=float, default=5.0)
    parser.add_argument("--profile-bin-width-deg", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--npz-name", type=str, default="psf_v1.npz")
    parser.add_argument("--metadata-name", type=str, default="psf_v1_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="psf_v1_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="psf_v1_summary.md")
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    return (
        label.replace(">=", "ge_")
        .replace("<", "lt_")
        .replace("[", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


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


def make_edges(start: float, stop: float, step: float) -> np.ndarray:
    n_steps = int(round((stop - start) / step))
    if n_steps <= 0:
        raise ValueError(f"Invalid edges: start={start}, stop={stop}, step={step}")
    edges = start + step * np.arange(n_steps + 1, dtype=np.float64)
    if not np.isclose(edges[-1], stop):
        raise ValueError(f"Step does not land on stop: start={start}, stop={stop}, step={step}")
    return edges


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


def binned_cell_dir(binned_root: Path, cell: CellSpec) -> Path:
    return binned_root / f"nhit_{sanitize_label(cell.nhit_bin)}" / f"predE_{sanitize_label(cell.predE_bin)}"


def open_tree(path: Path, tree_name: str):
    root_file = uproot.open(path)
    try:
        if tree_name in root_file:
            return root_file, root_file[tree_name]
        versioned = f"{tree_name};1"
        if versioned in root_file:
            return root_file, root_file[versioned]
        raise KeyError(f"{path} does not contain tree {tree_name!r}")
    except Exception:
        root_file.close()
        raise


def arrays_for_tree(
    path: Path,
    tree_name: str,
    branches: Sequence[str],
    optional_branches: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    root_file, tree = open_tree(path, tree_name)
    try:
        available = set(tree.keys())
        missing = [name for name in branches if name not in available]
        if missing:
            raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
        selected = list(dict.fromkeys(list(branches) + [name for name in optional_branches if name in available]))
        return tree.arrays(selected, library="np")
    finally:
        root_file.close()


def load_stage_a_metadata(path: Path, allow_missing: bool) -> Dict[str, object]:
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Stage A metadata does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_stage_a_metadata_for_production(stage_a_metadata: Dict[str, object]) -> None:
    if not stage_a_metadata:
        return
    expected = {
        "response_type": "primary_thrown_response",
        "absolute_effective_area_status": "available",
        "weighting": "mc_weight_baseline",
    }
    mismatches = [
        f"{key}={stage_a_metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if stage_a_metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError("Stage A metadata is not the current production response: " + "; ".join(mismatches))


def validate_stage_a_cells(cells: Sequence[CellSpec], stage_a_metadata: Dict[str, object]) -> None:
    if not stage_a_metadata:
        return
    stage_cells = stage_a_metadata.get("cells")
    if not stage_cells:
        raise ValueError("Stage A metadata does not contain a `cells` list.")
    if len(stage_cells) != len(cells):
        raise ValueError(f"Stage A cell count {len(stage_cells)} does not match Stage B cell count {len(cells)}")

    for idx, (cell, stage_cell) in enumerate(zip(cells, stage_cells)):
        expected = (cell.cell_id, cell.nhit_bin, cell.predE_bin)
        observed = (int(stage_cell["cell_id"]), str(stage_cell["nhit_bin"]), str(stage_cell["predE_bin"]))
        if observed != expected:
            raise ValueError(f"Stage A cell mismatch at index {idx}: expected {expected}, got {observed}")


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
            raise FileExistsError(f"Stage B run directory already exists: {run_dir}")
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


def discover_cell_files(cell_dir: Path, max_files: Optional[int], *, allow_missing_cell_dirs: bool) -> List[Path]:
    if not cell_dir.exists():
        if allow_missing_cell_dirs:
            return []
        raise FileNotFoundError(f"Missing binned ROOT directory: {cell_dir}")
    files = sorted(cell_dir.glob("*.root"))
    if max_files is not None:
        files = files[:max_files]
    if not files and not allow_missing_cell_dirs:
        raise FileNotFoundError(f"No ROOT files found in selected cell directory: {cell_dir}")
    return files


def load_weight(
    arrays: Dict[str, np.ndarray],
    *,
    weight_branch: str,
    allow_missing_weight: bool,
) -> np.ndarray:
    if weight_branch in arrays:
        return np.asarray(arrays[weight_branch], dtype=np.float64)
    if allow_missing_weight:
        ref = next(iter(arrays.values()))
        return np.ones(len(ref), dtype=np.float64)
    raise KeyError(f"Missing weight branch {weight_branch!r}")


def spherical_separation_rad(theta: np.ndarray, phi: np.ndarray, mc_theta: np.ndarray, mc_phi: np.ndarray) -> np.ndarray:
    cos_sep = (
        np.sin(theta) * np.sin(mc_theta) * np.cos(phi - mc_phi)
        + np.cos(theta) * np.cos(mc_theta)
    )
    return np.arccos(np.clip(cos_sep, -1.0, 1.0))


def append_concat(chunks: Sequence[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not chunks:
        return np.asarray([], dtype=dtype)
    return np.concatenate([np.asarray(chunk, dtype=dtype) for chunk in chunks])


def read_cell_events(
    files: Sequence[Path],
    *,
    tree_name: str,
    weight_branch: str,
    allow_missing_weight: bool,
    angle_check_max_events: int,
    file_progress_every: int,
    progress_label: str,
) -> CellEvents:
    dangle_chunks: List[np.ndarray] = []
    theta_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []
    loge_chunks: List[np.ndarray] = []
    angle_check_chunks: List[np.ndarray] = []
    angle_check_remaining = max(0, int(angle_check_max_events))

    base_required_branches = ["mc_dangle", "mc_theta", "mc_energy"]
    optional_branches: List[str] = []
    if allow_missing_weight:
        optional_branches.append(weight_branch)
    else:
        base_required_branches.append(weight_branch)

    file_progress_every = max(0, int(file_progress_every))
    for file_idx, path in enumerate(files, start=1):
        required_branches = list(base_required_branches)
        if angle_check_remaining > 0:
            required_branches.extend(["theta", "phi", "mc_phi"])
        arrays = arrays_for_tree(path, tree_name, required_branches, optional_branches=optional_branches)
        dangle = np.asarray(arrays["mc_dangle"], dtype=np.float64)
        mc_theta = np.asarray(arrays["mc_theta"], dtype=np.float64)
        weight = load_weight(arrays, weight_branch=weight_branch, allow_missing_weight=allow_missing_weight)
        mc_energy = np.asarray(arrays["mc_energy"], dtype=np.float64)

        dangle_chunks.append(dangle)
        theta_chunks.append(np.degrees(mc_theta))
        weight_chunks.append(weight)
        loge_chunks.append(np.log10(mc_energy, where=mc_energy > 0, out=np.full_like(mc_energy, np.nan, dtype=np.float64)))

        if angle_check_remaining > 0 and dangle.size > 0:
            take = min(angle_check_remaining, dangle.size)
            sep = spherical_separation_rad(
                np.asarray(arrays["theta"][:take], dtype=np.float64),
                np.asarray(arrays["phi"][:take], dtype=np.float64),
                np.asarray(arrays["mc_theta"][:take], dtype=np.float64),
                np.asarray(arrays["mc_phi"][:take], dtype=np.float64),
            )
            angle_check_chunks.append(np.abs(sep - dangle[:take]))
            angle_check_remaining -= take
        if file_progress_every > 0 and (file_idx % file_progress_every == 0 or file_idx == len(files)):
            print(
                f"[{progress_label}] read {file_idx}/{len(files)} files | events={sum(len(chunk) for chunk in dangle_chunks)}",
                flush=True,
            )

    return CellEvents(
        dangle_rad=append_concat(dangle_chunks, np.float64),
        mc_theta_deg=append_concat(theta_chunks, np.float64),
        mc_weight=append_concat(weight_chunks, np.float64),
        loge_true=append_concat(loge_chunks, np.float64),
        input_files=len(files),
        angle_check_absdiff_rad=append_concat(angle_check_chunks, np.float64),
    )


def crab_theta_probability(
    theta_edges_deg: np.ndarray,
    *,
    latitude_deg: float,
    declination_deg: float,
    theta_max_deg: float,
    hour_angle_samples: int,
) -> np.ndarray:
    if hour_angle_samples <= 0:
        raise ValueError("--hour-angle-samples must be positive")
    lat = math.radians(float(latitude_deg))
    dec = math.radians(float(declination_deg))
    hour_angle = np.linspace(-math.pi, math.pi, int(hour_angle_samples), endpoint=False, dtype=np.float64)
    cos_theta = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(hour_angle)
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    mask = theta_deg < float(theta_max_deg)
    hist, _ = np.histogram(theta_deg[mask], bins=theta_edges_deg)
    total = float(np.sum(hist))
    if total <= 0:
        raise ValueError("Crab theta track has no samples inside the requested theta range.")
    return hist.astype(np.float64) / total


def theta_bin_indices(theta_deg: np.ndarray, theta_edges_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(theta_edges_deg, theta_deg, side="right") - 1
    valid = np.isfinite(theta_deg) & (idx >= 0) & (idx < len(theta_edges_deg) - 1)
    return idx, valid


def theta_reweight_ratio(
    theta_deg: np.ndarray,
    mc_weight: np.ndarray,
    theta_edges_deg: np.ndarray,
    crab_prob: np.ndarray,
    *,
    support_mask: Optional[np.ndarray] = None,
    allow_incomplete_theta_support: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    idx, valid = theta_bin_indices(theta_deg, theta_edges_deg)
    finite_weight = np.isfinite(mc_weight) & (mc_weight > 0)
    hist_mask = valid & finite_weight
    if support_mask is not None:
        hist_mask &= np.asarray(support_mask, dtype=bool)
    mc_sumw, _ = np.histogram(theta_deg[hist_mask], bins=theta_edges_deg, weights=mc_weight[hist_mask])
    mc_total = float(np.sum(mc_sumw))
    if mc_total <= 0:
        raise ValueError("Cell has no positive mc_weight inside theta bin range.")
    mc_prob = mc_sumw / mc_total

    crab_positive = crab_prob > 0
    mc_positive = mc_prob > 0
    missing_support = crab_positive & ~mc_positive
    if np.any(missing_support) and not allow_incomplete_theta_support:
        missing_bins = [
            [float(theta_edges_deg[i]), float(theta_edges_deg[i + 1])]
            for i in np.nonzero(missing_support)[0].tolist()
        ]
        raise ValueError(f"MC theta support is missing Crab-positive bins: {missing_bins[:8]}")
    missing_crab_probability_mass = float(np.sum(crab_prob[missing_support]))

    ratio = np.zeros_like(mc_prob, dtype=np.float64)
    supported = crab_positive & mc_positive
    np.divide(crab_prob, mc_prob, out=ratio, where=supported)

    meta = {
        "mc_theta_sumw": mc_sumw.tolist(),
        "mc_theta_probability": mc_prob.tolist(),
        "crab_theta_probability": crab_prob.tolist(),
        "missing_crab_support_bin_count": int(np.count_nonzero(missing_support)),
        "missing_crab_support_bins": [
            [float(theta_edges_deg[i]), float(theta_edges_deg[i + 1])]
            for i in np.nonzero(missing_support)[0].tolist()
        ],
        "missing_crab_probability_mass": missing_crab_probability_mass,
        "zero_crab_positive_mc_bin_count": int(np.count_nonzero((~crab_positive) & mc_positive)),
    }
    return ratio, meta


def effective_event_count(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return 0.0
    sumw = float(np.sum(weight[valid]))
    sumw2 = float(np.sum(weight[valid] ** 2))
    if sumw2 <= 0:
        return 0.0
    return (sumw * sumw) / sumw2


def rayleigh_sigma_mle(r_rad: np.ndarray, weight: np.ndarray) -> float:
    r_rad = np.asarray(r_rad, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(r_rad) & (r_rad >= 0.0) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return float("nan")
    sumw = float(np.sum(weight[valid]))
    if sumw <= 0:
        return float("nan")
    second_moment = float(np.sum(weight[valid] * r_rad[valid] ** 2)) / sumw
    return math.sqrt(max(second_moment / 2.0, 0.0))


def weighted_quantile(values: np.ndarray, quantiles: Sequence[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles_arr = np.asarray(quantiles, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(quantiles_arr.shape, np.nan, dtype=np.float64)
    x = values[valid]
    w = weights[valid]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 0:
        return np.full(quantiles_arr.shape, np.nan, dtype=np.float64)
    return np.interp(quantiles_arr * total, cumulative, x)


def profile_histogram(r_deg: np.ndarray, weight: np.ndarray, profile_edges_deg: np.ndarray) -> np.ndarray:
    valid = np.isfinite(r_deg) & np.isfinite(weight) & (weight > 0)
    hist, _ = np.histogram(r_deg[valid], bins=profile_edges_deg, weights=weight[valid])
    total = float(np.sum(hist))
    widths = np.diff(profile_edges_deg)
    if total <= 0:
        return np.zeros(len(profile_edges_deg) - 1, dtype=np.float64)
    return hist.astype(np.float64) / (total * widths)


def finite_percentiles(values: np.ndarray, percentiles: Sequence[float]) -> List[Optional[float]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [None for _ in percentiles]
    return [float(v) for v in np.percentile(finite, percentiles)]


def fallback_psf_row(
    cell: CellSpec,
    *,
    cell_dir: Path,
    input_files: int,
    events: int,
    reason: str,
    profile_edges_deg: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray]:
    sigma_deg = 1.0
    r_opt_deg = RAYLEIGH_OPT_RADIUS_FACTOR * sigma_deg
    row: Dict[str, object] = {
        "cell_index": int(cell.index),
        "cell_id": int(cell.cell_id),
        "nhit_bin": cell.nhit_bin,
        "predE_bin": cell.predE_bin,
        "input_dir": str(cell_dir),
        "input_files": int(input_files),
        "events": int(events),
        "logE_range_events": 0,
        "valid_events": 0,
        "positive_baseline_weight_events": 0,
        "sumw_baseline": 0.0,
        "sumw_mc_weight": 0.0,
        "effective_events": 0.0,
        "core_fit_max_deg": None,
        "core_fit_events": 0,
        "core_fit_sumw": 0.0,
        "core_fit_effective_events": 0.0,
        "core_fit_weight_fraction": 0.0,
        "tail_weight_fraction_above_core_fit": 1.0,
        "sigma_rad": math.radians(sigma_deg),
        "sigma_deg": sigma_deg,
        "sigma_mc_weight_deg": sigma_deg,
        "sigma_unweighted_deg": sigma_deg,
        "sigma_full_rayleigh_rad": math.radians(sigma_deg),
        "sigma_full_rayleigh_deg": sigma_deg,
        "sigma_full_mc_weight_deg": sigma_deg,
        "sigma_full_unweighted_deg": sigma_deg,
        "r_opt_rad": math.radians(r_opt_deg),
        "r_opt_deg": r_opt_deg,
        "r_opt_factor": float(RAYLEIGH_OPT_RADIUS_FACTOR),
        "containment_r_opt": float(RAYLEIGH_OPT_CONTAINMENT),
        "containment_r_opt_core_fit_full_distribution": float(RAYLEIGH_OPT_CONTAINMENT),
        "rayleigh_expected_containment_r_opt": float(RAYLEIGH_OPT_CONTAINMENT),
        "containment_minus_expected": 0.0,
        "containment_warning": True,
        "r68_deg": sigma_deg * math.sqrt(-2.0 * math.log(1.0 - 0.68)),
        "r90_deg": sigma_deg * math.sqrt(-2.0 * math.log(1.0 - 0.90)),
        "r95_deg": sigma_deg * math.sqrt(-2.0 * math.log(1.0 - 0.95)),
        "core_r68_deg": sigma_deg * math.sqrt(-2.0 * math.log(1.0 - 0.68)),
        "core_r90_deg": sigma_deg * math.sqrt(-2.0 * math.log(1.0 - 0.90)),
        "core_r95_deg": sigma_deg * math.sqrt(-2.0 * math.log(1.0 - 0.95)),
        "mc_logE_true_p05": None,
        "mc_logE_true_p50": None,
        "mc_logE_true_p95": None,
        "theta_missing_crab_probability_mass": 1.0,
        "theta_reweight": {
            "status": "fallback",
            "reason": reason,
            "missing_crab_probability_mass": 1.0,
        },
        "angle_check_absdiff_rad_p50": None,
        "angle_check_absdiff_rad_p90": None,
        "angle_check_absdiff_rad_p99": None,
        "angle_check_absdiff_rad_max": None,
        "angle_check_warning": True,
        "psf_quality_flag": "fallback_low_stat",
        "warnings": [reason],
    }
    return row, np.zeros(profile_edges_deg.size - 1, dtype=np.float32)


def process_cell(
    cell: CellSpec,
    *,
    binned_root: Path,
    tree_name: str,
    weight_branch: str,
    allow_missing_weight: bool,
    max_files_per_cell: Optional[int],
    allow_missing_cell_dirs: bool,
    theta_edges_deg: np.ndarray,
    crab_prob: np.ndarray,
    loge_min: float,
    loge_max: float,
    allow_incomplete_theta_support: bool,
    min_events_per_cell: int,
    min_effective_events: float,
    allow_low_stat_psf_fallback: bool,
    core_fit_max_deg: float,
    theta_missing_mass_fail_threshold: float,
    containment_warning_tolerance: float,
    angle_check_max_events: int,
    angle_check_warn_rad: float,
    file_progress_every: int,
    profile_edges_deg: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray]:
    cell_dir = binned_cell_dir(binned_root, cell)
    files = discover_cell_files(cell_dir, max_files_per_cell, allow_missing_cell_dirs=allow_missing_cell_dirs)
    events = read_cell_events(
        files,
        tree_name=tree_name,
        weight_branch=weight_branch,
        allow_missing_weight=allow_missing_weight,
        angle_check_max_events=angle_check_max_events,
        file_progress_every=file_progress_every,
        progress_label=f"cell {cell.cell_id}",
    )

    n_events = int(events.dangle_rad.size)
    if n_events < int(min_events_per_cell):
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"events_below_min_events_per_cell:{n_events}<{min_events_per_cell}",
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(
            f"Cell {cell.cell_id} has {n_events} events, below --min-events-per-cell={min_events_per_cell}"
        )

    loge_valid = (
        np.isfinite(events.loge_true)
        & (events.loge_true >= float(loge_min))
        & (events.loge_true < float(loge_max))
    )
    ratio_support = (
        loge_valid
        & np.isfinite(events.dangle_rad)
        & (events.dangle_rad >= 0.0)
        & np.isfinite(events.mc_weight)
        & (events.mc_weight > 0.0)
    )
    theta_ratio, theta_meta = theta_reweight_ratio(
        events.mc_theta_deg,
        events.mc_weight,
        theta_edges_deg,
        crab_prob,
        support_mask=ratio_support,
        allow_incomplete_theta_support=allow_incomplete_theta_support,
    )
    missing_crab_mass = float(theta_meta.get("missing_crab_probability_mass") or 0.0)
    if missing_crab_mass > float(theta_missing_mass_fail_threshold):
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=(
                    f"theta_missing_crab_probability_mass:{missing_crab_mass:.6g}>"
                    f"{theta_missing_mass_fail_threshold}"
                ),
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(
            f"Cell {cell.cell_id} is missing Crab theta support mass {missing_crab_mass:.4g}, "
            f"above --theta-missing-mass-fail-threshold={theta_missing_mass_fail_threshold}"
        )
    theta_idx, theta_valid = theta_bin_indices(events.mc_theta_deg, theta_edges_deg)
    base_valid = (
        theta_valid
        & ratio_support
    )
    full_weight = np.zeros(n_events, dtype=np.float64)
    full_weight[base_valid] = events.mc_weight[base_valid] * theta_ratio[theta_idx[base_valid]]
    mc_weight_only = np.where(base_valid, events.mc_weight, 0.0)
    unweighted = np.where(base_valid, 1.0, 0.0)

    positive_full = full_weight > 0
    sumw_full = float(np.sum(full_weight[positive_full]))
    if sumw_full <= 0:
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason="no_positive_baseline_weight_after_theta_reweighting",
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(f"Cell {cell.cell_id} has no positive baseline weight after Crab theta reweighting.")

    neff = effective_event_count(full_weight)
    if neff < float(min_effective_events):
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"effective_events_below_min:{neff:.6g}<{min_effective_events}",
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(
            f"Cell {cell.cell_id} has effective events {neff:.3g}, below --min-effective-events={min_effective_events}"
        )

    core_fit_max_rad = math.radians(float(core_fit_max_deg))
    core_mask = positive_full & np.isfinite(events.dangle_rad) & (events.dangle_rad <= core_fit_max_rad)
    core_weight = np.where(core_mask, full_weight, 0.0)
    mc_weight_core = np.where(core_mask, events.mc_weight, 0.0)
    unweighted_core = np.where(core_mask, 1.0, 0.0)
    core_sumw = float(np.sum(core_weight[core_weight > 0]))
    core_neff = effective_event_count(core_weight)
    if core_sumw <= 0:
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason="no_positive_baseline_weight_inside_core_fit_range",
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(f"Cell {cell.cell_id} has no positive baseline weight inside core fit range.")
    if core_neff < float(min_effective_events):
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"core_effective_events_below_min:{core_neff:.6g}<{min_effective_events}",
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(
            f"Cell {cell.cell_id} has core effective events {core_neff:.3g}, "
            f"below --min-effective-events={min_effective_events}"
        )

    sigma_rad = rayleigh_sigma_mle(events.dangle_rad, core_weight)
    sigma_mc_weight_rad = rayleigh_sigma_mle(events.dangle_rad, mc_weight_core)
    sigma_unweighted_rad = rayleigh_sigma_mle(events.dangle_rad, unweighted_core)
    sigma_full_rad = rayleigh_sigma_mle(events.dangle_rad, full_weight)
    sigma_full_mc_weight_rad = rayleigh_sigma_mle(events.dangle_rad, mc_weight_only)
    sigma_full_unweighted_rad = rayleigh_sigma_mle(events.dangle_rad, unweighted)
    if not np.isfinite(sigma_rad) or sigma_rad <= 0:
        if allow_low_stat_psf_fallback:
            return fallback_psf_row(
                cell,
                cell_dir=cell_dir,
                input_files=events.input_files,
                events=n_events,
                reason=f"invalid_baseline_sigma:{sigma_rad}",
                profile_edges_deg=profile_edges_deg,
            )
        raise ValueError(f"Cell {cell.cell_id} has invalid baseline sigma: {sigma_rad}")

    r_opt_rad = RAYLEIGH_OPT_RADIUS_FACTOR * sigma_rad
    containment = float(np.sum(full_weight[positive_full & (events.dangle_rad <= r_opt_rad)]) / sumw_full)
    if not np.isfinite(containment) or containment <= 0.0 or containment > 1.0:
        raise ValueError(f"Cell {cell.cell_id} has invalid r_opt containment: {containment}")

    containment_warning = abs(containment - RAYLEIGH_OPT_CONTAINMENT) > float(containment_warning_tolerance)
    quantiles_rad = weighted_quantile(events.dangle_rad, [0.68, 0.90, 0.95], full_weight)
    core_quantiles_rad = weighted_quantile(events.dangle_rad, [0.68, 0.90, 0.95], core_weight)
    r_deg = np.degrees(events.dangle_rad)
    profile_density = profile_histogram(r_deg, full_weight, profile_edges_deg)

    angle_check_percentiles = finite_percentiles(events.angle_check_absdiff_rad, [50.0, 90.0, 99.0, 100.0])
    angle_check_warning = (
        angle_check_percentiles[0] is not None
        and float(angle_check_percentiles[0]) > float(angle_check_warn_rad)
    )

    loge_percentiles = finite_percentiles(events.loge_true[loge_valid], [5.0, 50.0, 95.0])
    row: Dict[str, object] = {
        "cell_index": int(cell.index),
        "cell_id": int(cell.cell_id),
        "nhit_bin": cell.nhit_bin,
        "predE_bin": cell.predE_bin,
        "input_dir": str(cell_dir),
        "input_files": int(events.input_files),
        "events": n_events,
        "logE_range_events": int(np.count_nonzero(loge_valid)),
        "valid_events": int(np.count_nonzero(base_valid)),
        "positive_baseline_weight_events": int(np.count_nonzero(positive_full)),
        "sumw_baseline": sumw_full,
        "sumw_mc_weight": float(np.sum(mc_weight_only[mc_weight_only > 0])),
        "effective_events": float(neff),
        "core_fit_max_deg": float(core_fit_max_deg),
        "core_fit_events": int(np.count_nonzero(core_mask)),
        "core_fit_sumw": core_sumw,
        "core_fit_effective_events": float(core_neff),
        "core_fit_weight_fraction": float(core_sumw / sumw_full),
        "tail_weight_fraction_above_core_fit": float(max(0.0, 1.0 - core_sumw / sumw_full)),
        "sigma_rad": float(sigma_rad),
        "sigma_deg": float(math.degrees(sigma_rad)),
        "sigma_mc_weight_deg": float(math.degrees(sigma_mc_weight_rad)) if np.isfinite(sigma_mc_weight_rad) else None,
        "sigma_unweighted_deg": float(math.degrees(sigma_unweighted_rad)) if np.isfinite(sigma_unweighted_rad) else None,
        "sigma_full_rayleigh_rad": float(sigma_full_rad) if np.isfinite(sigma_full_rad) else None,
        "sigma_full_rayleigh_deg": float(math.degrees(sigma_full_rad)) if np.isfinite(sigma_full_rad) else None,
        "sigma_full_mc_weight_deg": (
            float(math.degrees(sigma_full_mc_weight_rad)) if np.isfinite(sigma_full_mc_weight_rad) else None
        ),
        "sigma_full_unweighted_deg": (
            float(math.degrees(sigma_full_unweighted_rad)) if np.isfinite(sigma_full_unweighted_rad) else None
        ),
        "r_opt_rad": float(r_opt_rad),
        "r_opt_deg": float(math.degrees(r_opt_rad)),
        "r_opt_factor": float(RAYLEIGH_OPT_RADIUS_FACTOR),
        "containment_r_opt": containment,
        "containment_r_opt_core_fit_full_distribution": containment,
        "rayleigh_expected_containment_r_opt": float(RAYLEIGH_OPT_CONTAINMENT),
        "containment_minus_expected": float(containment - RAYLEIGH_OPT_CONTAINMENT),
        "containment_warning": bool(containment_warning),
        "r68_deg": float(math.degrees(quantiles_rad[0])) if np.isfinite(quantiles_rad[0]) else None,
        "r90_deg": float(math.degrees(quantiles_rad[1])) if np.isfinite(quantiles_rad[1]) else None,
        "r95_deg": float(math.degrees(quantiles_rad[2])) if np.isfinite(quantiles_rad[2]) else None,
        "core_r68_deg": float(math.degrees(core_quantiles_rad[0])) if np.isfinite(core_quantiles_rad[0]) else None,
        "core_r90_deg": float(math.degrees(core_quantiles_rad[1])) if np.isfinite(core_quantiles_rad[1]) else None,
        "core_r95_deg": float(math.degrees(core_quantiles_rad[2])) if np.isfinite(core_quantiles_rad[2]) else None,
        "mc_logE_true_p05": loge_percentiles[0],
        "mc_logE_true_p50": loge_percentiles[1],
        "mc_logE_true_p95": loge_percentiles[2],
        "theta_missing_crab_probability_mass": missing_crab_mass,
        "theta_reweight": theta_meta,
        "angle_check_absdiff_rad_p50": angle_check_percentiles[0],
        "angle_check_absdiff_rad_p90": angle_check_percentiles[1],
        "angle_check_absdiff_rad_p99": angle_check_percentiles[2],
        "angle_check_absdiff_rad_max": angle_check_percentiles[3],
        "angle_check_warning": bool(angle_check_warning),
        "psf_quality_flag": "warning" if (containment_warning or angle_check_warning) else "ok",
        "warnings": [],
    }
    return row, profile_density


def process_cell_task(task: Tuple[CellSpec, Dict[str, object]]) -> Tuple[int, Dict[str, object], np.ndarray]:
    cell, kwargs = task
    row, profile_density = process_cell(cell, **kwargs)
    return int(cell.index), row, profile_density


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "input_files",
        "events",
        "logE_range_events",
        "valid_events",
        "positive_baseline_weight_events",
        "sumw_baseline",
        "effective_events",
        "core_fit_events",
        "core_fit_effective_events",
        "core_fit_weight_fraction",
        "tail_weight_fraction_above_core_fit",
        "theta_missing_crab_probability_mass",
        "sigma_deg",
        "sigma_mc_weight_deg",
        "sigma_unweighted_deg",
        "sigma_full_rayleigh_deg",
        "r_opt_deg",
        "containment_r_opt",
        "r68_deg",
        "r90_deg",
        "r95_deg",
        "containment_warning",
        "angle_check_warning",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown_summary(
    path: Path,
    *,
    metadata: Dict[str, object],
    rows: Sequence[Dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage B PSF Summary\n\n")
        f.write(f"- Cells: {metadata['n_cells']}\n")
        f.write(f"- Weighting: `{metadata['weighting']['baseline']}`\n")
        f.write(f"- Theta bins: {metadata['theta_edges_deg'][0]} to {metadata['theta_edges_deg'][-1]} deg\n")
        f.write(f"- PSF baseline fit: core Rayleigh MLE inside {metadata['psf_model']['core_fit_max_deg']} deg\n")
        f.write(f"- Rayleigh r_opt factor: {RAYLEIGH_OPT_RADIUS_FACTOR:.3f}\n")
        f.write(f"- Output NPZ: `{metadata['outputs']['npz']}`\n\n")
        f.write(
            "| cell | Nhit bin | predE bin | events | Neff | core Neff | "
            "sigma deg | full sigma deg | r_opt deg | containment | tail weight |\n"
        )
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{row['events']} | {row['effective_events']:.1f} | "
                f"{row['core_fit_effective_events']:.1f} | "
                f"{row['sigma_deg']:.5g} | {row['sigma_full_rayleigh_deg']:.5g} | "
                f"{row['r_opt_deg']:.5g} | {row['containment_r_opt']:.4f} | "
                f"{row['tail_weight_fraction_above_core_fit']:.4f} |\n"
            )


def metric_array(rows: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def prepare_grid(cells: Sequence[CellSpec]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], CellSpec]]:
    nhit_bins = sorted({cell.nhit_bin for cell in cells}, key=interval_key)
    pred_bins = sorted({cell.predE_bin for cell in cells}, key=interval_key)
    return nhit_bins, pred_bins, {(cell.nhit_bin, cell.predE_bin): cell for cell in cells}


def plot_metric_grid(
    rows: Sequence[Dict[str, object]],
    cells: Sequence[CellSpec],
    output_path: Path,
    *,
    key: str,
    title: str,
    colorbar_label: str,
    fmt: str,
) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    values_by_cell = {int(row["cell_id"]): float(row[key]) for row in rows}
    matrix = np.full((len(nhit_bins), len(pred_bins)), np.nan, dtype=np.float64)
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is not None:
                matrix[i, j] = values_by_cell[cell.cell_id]

    fig, ax = plt.subplots(figsize=(1.25 * len(pred_bins) + 2.6, 0.58 * len(nhit_bins) + 2.0), dpi=150)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_xlabel("log10(E_pred / GeV) bin", fontsize=8)
    ax.set_ylabel("Nhit bin", fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def rayleigh_pdf_deg(r_deg: np.ndarray, sigma_rad: float) -> np.ndarray:
    r_rad = np.radians(r_deg)
    pdf_per_rad = (r_rad / (sigma_rad * sigma_rad)) * np.exp(-0.5 * (r_rad / sigma_rad) ** 2)
    return pdf_per_rad * (math.pi / 180.0)


def plot_profile_grid(
    profile_density: np.ndarray,
    profile_edges_deg: np.ndarray,
    rows: Sequence[Dict[str, object]],
    cells: Sequence[CellSpec],
    output_path: Path,
) -> None:
    plt = setup_matplotlib()
    nhit_bins, pred_bins, by_key = prepare_grid(cells)
    row_by_cell_id = {int(row["cell_id"]): row for row in rows}
    centers = 0.5 * (profile_edges_deg[:-1] + profile_edges_deg[1:])
    fig, axes = plt.subplots(
        len(nhit_bins),
        len(pred_bins),
        figsize=(2.0 * len(pred_bins), 1.55 * len(nhit_bins)),
        dpi=150,
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for i, nhit_bin in enumerate(nhit_bins):
        for j, pred_bin in enumerate(pred_bins):
            ax = axes[i, j]
            cell = by_key.get((nhit_bin, pred_bin))
            if cell is None:
                ax.set_axis_off()
                continue
            row = row_by_cell_id[cell.cell_id]
            density = profile_density[cell.index]
            ax.step(centers, density, where="mid", color="#1f4e79", linewidth=0.9)
            model = rayleigh_pdf_deg(centers, float(row["sigma_rad"]))
            ax.plot(centers, model, color="#c9501a", linewidth=0.8, alpha=0.9)
            ax.axvline(float(row["r_opt_deg"]), color="#444444", linewidth=0.7, linestyle="--")
            ax.set_title(f"cell {cell.cell_id}: {pred_bin}", fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            ax.grid(alpha=0.22, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(nhit_bin, fontsize=6.7)
            if i == len(nhit_bins) - 1:
                ax.set_xlabel("r (deg)", fontsize=6.7)
    fig.suptitle("Stage B weighted radial PSF profiles: MC histogram, Rayleigh fit, r_opt", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(output_path)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    *,
    rows: Sequence[Dict[str, object]],
    cells: Sequence[CellSpec],
    profile_density: np.ndarray,
    profile_edges_deg: np.ndarray,
) -> Dict[str, str]:
    paths = {
        "sigma_grid_png": output_dir / "psf_sigma_deg_grid.png",
        "r_opt_grid_png": output_dir / "psf_r_opt_deg_grid.png",
        "containment_grid_png": output_dir / "psf_containment_grid.png",
        "effective_events_grid_png": output_dir / "psf_effective_events_grid.png",
        "profiles_grid_png": output_dir / "psf_radial_profiles_grid.png",
    }
    plot_metric_grid(
        rows,
        cells,
        paths["sigma_grid_png"],
        key="sigma_deg",
        title="Stage B PSF sigma by configured cell",
        colorbar_label="sigma (deg)",
        fmt=".3g",
    )
    plot_metric_grid(
        rows,
        cells,
        paths["r_opt_grid_png"],
        key="r_opt_deg",
        title="Stage B optimal integration radius by configured cell",
        colorbar_label="1.58 sigma (deg)",
        fmt=".3g",
    )
    plot_metric_grid(
        rows,
        cells,
        paths["containment_grid_png"],
        key="containment_r_opt",
        title="Stage B containment inside 1.58 sigma",
        colorbar_label="containment",
        fmt=".3f",
    )
    plot_metric_grid(
        rows,
        cells,
        paths["effective_events_grid_png"],
        key="effective_events",
        title="Stage B effective event count by configured cell",
        colorbar_label="Neff",
        fmt=".2g",
    )
    plot_profile_grid(profile_density, profile_edges_deg, rows, cells, paths["profiles_grid_png"])
    return {key: str(path) for key, path in paths.items()}


def compact_stage_a_snapshot(stage_a_metadata: Dict[str, object]) -> Dict[str, object]:
    if not stage_a_metadata:
        return {}
    return {
        "response_type": stage_a_metadata.get("response_type"),
        "weighting": stage_a_metadata.get("weighting"),
        "absolute_effective_area_status": stage_a_metadata.get("absolute_effective_area_status"),
        "npz_path": stage_a_metadata.get("npz_path"),
        "binned_root": stage_a_metadata.get("binned_root"),
        "cell_selection_csv": stage_a_metadata.get("cell_selection_csv"),
        "n_cells": stage_a_metadata.get("n_cells"),
        "truth_shape": stage_a_metadata.get("truth_shape"),
        "logE_true_edges": stage_a_metadata.get("logE_true_edges"),
        "theta_true_edges_deg": stage_a_metadata.get("theta_true_edges_deg"),
        "s0_m2": stage_a_metadata.get("s0_m2"),
        "effective_area_formula": stage_a_metadata.get("effective_area_formula"),
        "cuts": stage_a_metadata.get("cuts"),
    }


def resolve_loge_range(args: argparse.Namespace, stage_a_metadata: Dict[str, object]) -> Tuple[float, float, str]:
    if args.logE_min is not None and args.logE_max is not None:
        source = "cli"
        loge_min = float(args.logE_min)
        loge_max = float(args.logE_max)
    elif args.logE_min is None and args.logE_max is None and stage_a_metadata.get("logE_true_edges"):
        edges = [float(value) for value in stage_a_metadata["logE_true_edges"]]
        loge_min = float(edges[0])
        loge_max = float(edges[-1])
        source = "stage_a_metadata.logE_true_edges"
    elif args.logE_min is None and args.logE_max is None:
        loge_min = 2.0
        loge_max = 6.0
        source = "default_2_6"
    else:
        raise ValueError("--logE-min and --logE-max must be provided together.")

    if not np.isfinite(loge_min) or not np.isfinite(loge_max) or loge_min >= loge_max:
        raise ValueError(f"Invalid logE range: [{loge_min}, {loge_max})")
    return loge_min, loge_max, source


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    binned_root = Path(args.binned_root).resolve()
    selection_csv = Path(args.cell_selection_csv).resolve()
    stage_a_metadata_path = Path(args.stage_a_metadata).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(args.run_id or make_default_run_id())
    output_dir = prepare_run_output_dir(
        output_root,
        run_id,
        overwrite_run_dir=bool(args.overwrite_run_dir),
    )

    cells = load_cells(selection_csv)
    stage_a_metadata = load_stage_a_metadata(stage_a_metadata_path, bool(args.allow_missing_stage_a_metadata))
    validate_stage_a_metadata_for_production(stage_a_metadata)
    validate_stage_a_cells(cells, stage_a_metadata)
    loge_min, loge_max, loge_range_source = resolve_loge_range(args, stage_a_metadata)

    theta_edges_deg = make_edges(float(args.theta_min_deg), float(args.theta_max_deg), float(args.theta_step_deg))
    profile_edges_deg = make_edges(0.0, float(args.profile_max_deg), float(args.profile_bin_width_deg))
    crab_prob = crab_theta_probability(
        theta_edges_deg,
        latitude_deg=float(args.lhaaso_lat_deg),
        declination_deg=float(args.source_dec_deg),
        theta_max_deg=float(args.theta_max_deg),
        hour_angle_samples=int(args.hour_angle_samples),
    )

    cell_kwargs: Dict[str, object] = {
        "binned_root": binned_root,
        "tree_name": args.tree_name,
        "weight_branch": args.weight_branch,
        "allow_missing_weight": bool(args.allow_missing_weight),
        "max_files_per_cell": args.max_files_per_cell,
        "allow_missing_cell_dirs": bool(args.allow_missing_cell_dirs),
        "theta_edges_deg": theta_edges_deg,
        "crab_prob": crab_prob,
        "loge_min": loge_min,
        "loge_max": loge_max,
        "allow_incomplete_theta_support": bool(args.allow_incomplete_theta_support),
        "min_events_per_cell": int(args.min_events_per_cell),
        "min_effective_events": float(args.min_effective_events),
        "allow_low_stat_psf_fallback": bool(args.allow_low_stat_psf_fallback),
        "core_fit_max_deg": float(args.core_fit_max_deg),
        "theta_missing_mass_fail_threshold": float(args.theta_missing_mass_fail_threshold),
        "containment_warning_tolerance": float(args.containment_warning_tolerance),
        "angle_check_max_events": int(args.angle_check_max_events),
        "angle_check_warn_rad": float(args.angle_check_warn_rad),
        "file_progress_every": int(args.file_progress_every),
        "profile_edges_deg": profile_edges_deg,
    }
    workers = max(1, int(args.workers))
    rows_by_index: Dict[int, Dict[str, object]] = {}
    profiles_by_index: Dict[int, np.ndarray] = {}

    if workers == 1:
        for done_count, cell in enumerate(cells, start=1):
            row, profile_density = process_cell(cell, **cell_kwargs)
            rows_by_index[int(cell.index)] = row
            profiles_by_index[int(cell.index)] = profile_density
            print(
                f"[{done_count}/{len(cells)}] cell={cell.cell_id} {cell.nhit_bin} {cell.predE_bin} "
                f"events={row['events']} Neff={row['effective_events']:.1f} "
                f"sigma={row['sigma_deg']:.4g} deg r_opt={row['r_opt_deg']:.4g} deg",
                flush=True,
            )
    else:
        print(f"Processing {len(cells)} cells with {workers} workers.", flush=True)
        tasks = [(cell, cell_kwargs) for cell in cells]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_by_cell = {executor.submit(process_cell_task, task): task[0] for task in tasks}
            for done_count, future in enumerate(as_completed(future_by_cell), start=1):
                cell = future_by_cell[future]
                cell_index, row, profile_density = future.result()
                rows_by_index[cell_index] = row
                profiles_by_index[cell_index] = profile_density
                print(
                    f"[{done_count}/{len(cells)}] cell={cell.cell_id} {cell.nhit_bin} {cell.predE_bin} "
                    f"events={row['events']} Neff={row['effective_events']:.1f} "
                    f"sigma={row['sigma_deg']:.4g} deg r_opt={row['r_opt_deg']:.4g} deg",
                    flush=True,
                )

    rows = [rows_by_index[int(cell.index)] for cell in cells]
    profile_rows = [profiles_by_index[int(cell.index)] for cell in cells]

    profile_density = np.vstack(profile_rows).astype(np.float32)
    npz_path = output_dir / args.npz_name
    np.savez_compressed(
        npz_path,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        sigma_rad=metric_array(rows, "sigma_rad").astype(np.float32),
        sigma_deg=metric_array(rows, "sigma_deg").astype(np.float32),
        sigma_mc_weight_deg=metric_array(rows, "sigma_mc_weight_deg").astype(np.float32),
        sigma_unweighted_deg=metric_array(rows, "sigma_unweighted_deg").astype(np.float32),
        sigma_full_rayleigh_rad=metric_array(rows, "sigma_full_rayleigh_rad").astype(np.float32),
        sigma_full_rayleigh_deg=metric_array(rows, "sigma_full_rayleigh_deg").astype(np.float32),
        r_opt_rad=metric_array(rows, "r_opt_rad").astype(np.float32),
        r_opt_deg=metric_array(rows, "r_opt_deg").astype(np.float32),
        containment_r_opt=metric_array(rows, "containment_r_opt").astype(np.float32),
        r68_deg=metric_array(rows, "r68_deg").astype(np.float32),
        r90_deg=metric_array(rows, "r90_deg").astype(np.float32),
        r95_deg=metric_array(rows, "r95_deg").astype(np.float32),
        core_r68_deg=metric_array(rows, "core_r68_deg").astype(np.float32),
        core_r90_deg=metric_array(rows, "core_r90_deg").astype(np.float32),
        core_r95_deg=metric_array(rows, "core_r95_deg").astype(np.float32),
        effective_events=metric_array(rows, "effective_events").astype(np.float32),
        core_fit_effective_events=metric_array(rows, "core_fit_effective_events").astype(np.float32),
        core_fit_weight_fraction=metric_array(rows, "core_fit_weight_fraction").astype(np.float32),
        tail_weight_fraction_above_core_fit=metric_array(rows, "tail_weight_fraction_above_core_fit").astype(np.float32),
        theta_missing_crab_probability_mass=metric_array(rows, "theta_missing_crab_probability_mass").astype(np.float32),
        events=metric_array(rows, "events").astype(np.int64),
        sumw_baseline=metric_array(rows, "sumw_baseline").astype(np.float64),
        theta_edges_deg=theta_edges_deg.astype(np.float32),
        crab_theta_probability=crab_prob.astype(np.float32),
        profile_edges_deg=profile_edges_deg.astype(np.float32),
        profile_density=profile_density,
    )

    summary_csv_path = output_dir / args.summary_csv_name
    summary_md_path = output_dir / args.summary_md_name
    metadata_path = output_dir / args.metadata_name
    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = write_plots(
            output_dir,
            rows=rows,
            cells=cells,
            profile_density=profile_density,
            profile_edges_deg=profile_edges_deg,
        )

    warning_rows = [
        {
            "cell_id": row["cell_id"],
            "containment_warning": row["containment_warning"],
            "angle_check_warning": row["angle_check_warning"],
            "missing_crab_probability_mass": row["theta_reweight"].get("missing_crab_probability_mass"),
            "theta_missing_support_warning": (
                row["theta_reweight"].get("missing_crab_probability_mass", 0.0) > 0.0
            ),
        }
        for row in rows
        if (
            row["containment_warning"]
            or row["angle_check_warning"]
            or row["theta_reweight"].get("missing_crab_probability_mass", 0.0) > 0.0
        )
    ]
    metadata: Dict[str, object] = {
        "description": "Stage B Crab-declination PSF table for configured (Nhit, predicted logE) cells.",
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "binned_root": str(binned_root),
        "cell_selection_csv": str(selection_csv),
        "stage_a_metadata": str(stage_a_metadata_path),
        "stage_a_snapshot": compact_stage_a_snapshot(stage_a_metadata),
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "tree_name": args.tree_name,
        "n_cells": len(cells),
        "cells": rows,
        "weighting": {
            "baseline": f"{args.weight_branch} * crab_declination_theta_reweight",
            "diagnostics": ["unweighted", args.weight_branch],
            "allow_missing_weight": bool(args.allow_missing_weight),
        },
        "logE_true_filter": {
            "min_inclusive": float(loge_min),
            "max_exclusive": float(loge_max),
            "source": loge_range_source,
        },
        "crab_track": {
            "lhaaso_lat_deg": float(args.lhaaso_lat_deg),
            "source_dec_deg": float(args.source_dec_deg),
            "theta_max_deg": float(args.theta_max_deg),
            "hour_angle_samples": int(args.hour_angle_samples),
            "interpretation": "uniform hour-angle samples conditioned on theta < theta_max_deg",
        },
        "theta_edges_deg": theta_edges_deg.tolist(),
        "crab_theta_probability": crab_prob.tolist(),
        "psf_model": {
            "radial_residual_branch": "mc_dangle",
            "model": "2d_circular_gaussian_radial_rayleigh_core_mle",
            "core_fit_max_deg": float(args.core_fit_max_deg),
            "sigma_formula": "sqrt(sum(w * r^2) / (2 * sum(w)))",
            "baseline_sigma": "Rayleigh MLE using events with mc_dangle <= core_fit_max_deg",
            "tail_diagnostics": ["full-distribution Rayleigh MLE", "r68", "r90", "r95", "tail weight fraction"],
            "r_opt_factor": float(RAYLEIGH_OPT_RADIUS_FACTOR),
            "r_opt_expected_containment": float(RAYLEIGH_OPT_CONTAINMENT),
        },
        "quality_thresholds": {
            "min_events_per_cell": int(args.min_events_per_cell),
            "min_effective_events": float(args.min_effective_events),
            "core_fit_max_deg": float(args.core_fit_max_deg),
            "containment_warning_tolerance": float(args.containment_warning_tolerance),
            "allow_incomplete_theta_support": bool(args.allow_incomplete_theta_support),
            "theta_missing_mass_fail_threshold": float(args.theta_missing_mass_fail_threshold),
            "angle_check_max_events": int(args.angle_check_max_events),
            "angle_check_warn_rad": float(args.angle_check_warn_rad),
            "file_progress_every": int(args.file_progress_every),
        },
        "warning_rows": warning_rows,
        "elapsed_seconds": float(time.perf_counter() - start),
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": {
            "npz": str(npz_path),
            "metadata_json": str(metadata_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            **plot_outputs,
        },
    }

    write_summary_csv(summary_csv_path, rows)
    write_markdown_summary(summary_md_path, metadata=metadata, rows=rows)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.no_promote_current:
        promote_successful_run(output_root, output_dir)
        metadata["promotion"]["status"] = "promoted"
        metadata["promotion"]["current_dir"] = str(output_root / "current")
        metadata["promotion"]["latest"] = str(output_root / "latest")
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    else:
        metadata["promotion"]["status"] = "skipped"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {metadata_path}")
    if not args.no_promote_current:
        print(f"Promoted current Stage B output to {output_root / 'current'}")
    if warning_rows:
        print(f"Warnings recorded for {len(warning_rows)} cells; inspect metadata warning_rows.")


if __name__ == "__main__":
    main()
