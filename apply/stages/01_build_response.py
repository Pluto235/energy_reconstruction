#!/usr/bin/env python
import argparse
import concurrent.futures
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import uproot

from apply.simulation_all_bin import sanitize_label


DEFAULT_RUN_DIR = "/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666"
DEFAULT_INPUT_ROOT = "/mnt/mydisk/WCDA_simulation"
DEFAULT_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v1"
DEFAULT_CELL_SELECTION = "apply/config/cell_selection_v1.csv"
DEFAULT_OUTPUT_DIR = "apply/output/stage_a"
DEFAULT_TREE_NAME = "t_eventout"
DEFAULT_PRIMARY_DENOMINATOR_NPZ = "/mnt/mydisk/WCDA_simulation_primary_hist/primary_denominator_stage_a.npz"
DEFAULT_INJECTION_AREA_CM2 = 4.0e10
CM2_PER_M2 = 1.0e4
DEFAULT_S0_M2 = DEFAULT_INJECTION_AREA_CM2 / CM2_PER_M2
DEFAULT_THROWN_GEOMETRY = "square_core_box"
DEFAULT_THROWN_CORE_RANGE_M = [-1000.0, 1000.0]


@dataclass(frozen=True)
class CellSpec:
    cell_id: int
    nhit_bin: str
    predE_bin: str
    mc_count: int
    selection_version: str
    selection_reason: str
    raw_ledger_version: str
    cell_role: str
    role_reason: str
    source_pool: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage A 2D primary-thrown response on configured (Nhit, predicted-energy) cells."
    )
    parser.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--binned-root", type=str, default=DEFAULT_BINNED_ROOT)
    parser.add_argument("--cell-selection-csv", type=str, default=DEFAULT_CELL_SELECTION)
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tree-name", type=str, default=DEFAULT_TREE_NAME)
    parser.add_argument("--primary-denominator-npz", type=str, default=DEFAULT_PRIMARY_DENOMINATOR_NPZ)
    parser.add_argument(
        "--use-eventout-denominator",
        action="store_true",
        default=False,
        help="Fallback to the legacy denominator built from cut-passing eventout trees.",
    )
    parser.add_argument("--max-raw-files", type=int, default=None)
    parser.add_argument("--max-binned-files-per-cell", type=int, default=None)
    parser.add_argument(
        "--denominator-workers",
        type=int,
        default=1,
        help="Parallel workers for scanning raw MC denominator files. 1 keeps sequential execution.",
    )
    parser.add_argument("--denominator-progress-every", type=int, default=100)
    parser.add_argument(
        "--numerator-workers",
        type=int,
        default=8,
        help="Parallel workers for scanning binned MC numerator files. 1 keeps sequential execution.",
    )
    parser.add_argument(
        "--numerator-files-per-task",
        type=int,
        default=250,
        help="Number of binned ROOT files processed by each numerator worker task.",
    )
    parser.add_argument("--numerator-progress-every", type=int, default=20)
    parser.add_argument(
        "--allow-missing-cell-dirs",
        action="store_true",
        default=False,
        help="Allow selected cells with no binned ROOT directory. Intended for small smoke tests only.",
    )
    parser.add_argument("--logE-min", type=float, default=2.0)
    parser.add_argument("--logE-max", type=float, default=6.0)
    parser.add_argument("--logE-step", type=float, default=0.1)
    parser.add_argument("--theta-min-deg", type=float, default=0.0)
    parser.add_argument("--theta-max-deg", type=float, default=50.0)
    parser.add_argument("--theta-step-deg", type=float, default=1.0)
    parser.add_argument("--cut-pinc-max", type=float, default=1.1)
    parser.add_argument("--cut-theta-max-deg", type=float, default=50.0)
    parser.add_argument("--cut-fitstat-equals", type=int, default=0)
    parser.add_argument("--cut-dcedge-min", type=float, default=20.0)
    parser.add_argument(
        "--core-box",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=None,
        help="Optional denominator core-box cut on xc/yc.",
    )
    parser.add_argument(
        "--weight-branch",
        type=str,
        default="mc_weight",
        help="Weight branch for the baseline response. Missing branch falls back to ones only when --allow-missing-weight is set.",
    )
    parser.add_argument("--allow-missing-weight", action="store_true", default=False)
    parser.add_argument("--s0-m2", type=float, default=DEFAULT_S0_M2)
    parser.add_argument(
        "--aperture-psf-npz",
        type=str,
        default="",
        help=(
            "Optional Stage B PSF NPZ used to build an aperture-conditioned response. "
            "When set, numerator MC events are additionally cut with mc_dangle <= r_opt_deg(cell). "
            "Leave empty for the default all-direction cell response."
        ),
    )
    parser.add_argument(
        "--aperture-angle-branch",
        type=str,
        default="mc_dangle",
        help="ROOT branch in radians used for the optional aperture-conditioned numerator cut.",
    )
    parser.add_argument("--npz-name", type=str, default="response_2d.npz")
    parser.add_argument("--metadata-name", type=str, default="response_2d_metadata.json")
    return parser.parse_args()


def make_edges(start: float, stop: float, step: float) -> np.ndarray:
    n_steps = int(round((stop - start) / step))
    if n_steps <= 0:
        raise ValueError(f"Invalid edges: start={start}, stop={stop}, step={step}")
    edges = start + step * np.arange(n_steps + 1, dtype=np.float64)
    if not np.isclose(edges[-1], stop):
        raise ValueError(f"Step does not land on stop: start={start}, stop={stop}, step={step}")
    return edges


def discover_root_files(root: Path, max_files: Optional[int]) -> List[Path]:
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".root")
    if max_files is not None:
        files = files[:max_files]
    return files


def load_cells(selection_csv: Path) -> List[CellSpec]:
    cells: List[CellSpec] = []
    with selection_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cells.append(
                CellSpec(
                    cell_id=int(row.get("cell_id") or idx),
                    nhit_bin=row["nhit_bin"],
                    predE_bin=row["predE_bin"],
                    mc_count=int(row.get("mc_count") or 0),
                    selection_version=row.get("selection_version", ""),
                    selection_reason=row.get("selection_reason", ""),
                    raw_ledger_version=row.get("raw_ledger_version", ""),
                    cell_role=row.get("cell_role", ""),
                    role_reason=row.get("role_reason", ""),
                    source_pool=row.get("source_pool", ""),
                )
            )
    if not cells:
        raise ValueError(f"No cells loaded from {selection_csv}")
    return cells


def load_aperture_radii(cells: Sequence[CellSpec], psf_npz: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, object]]]:
    if psf_npz is None:
        return None, None
    if not psf_npz.exists():
        raise FileNotFoundError(f"Aperture PSF NPZ does not exist: {psf_npz}")
    with np.load(psf_npz, allow_pickle=False) as data:
        required = {"cell_id", "r_opt_deg"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{psf_npz} is missing required arrays for aperture response: {sorted(missing)}")
        source_cell_id = np.asarray(data["cell_id"], dtype=np.int64)
        source_r_opt = np.asarray(data["r_opt_deg"], dtype=np.float64)
    by_cell = {int(cid): float(radius) for cid, radius in zip(source_cell_id, source_r_opt)}
    radii = np.asarray([by_cell.get(int(cell.cell_id), np.nan) for cell in cells], dtype=np.float64)
    invalid = [
        int(cell.cell_id)
        for cell, radius in zip(cells, radii)
        if not math.isfinite(float(radius)) or float(radius) <= 0.0
    ]
    if invalid:
        raise ValueError(f"Aperture PSF has invalid/missing r_opt_deg for cells: {invalid}")
    meta = {
        "mode": "mc_dangle_le_r_opt",
        "psf_npz": str(psf_npz),
        "angle_branch": "mc_dangle",
        "radius_source": "Stage B PSF NPZ r_opt_deg",
        "min_r_opt_deg": float(np.nanmin(radii)),
        "median_r_opt_deg": float(np.nanmedian(radii)),
        "max_r_opt_deg": float(np.nanmax(radii)),
    }
    return radii, meta


def binned_cell_dir(binned_root: Path, cell: CellSpec) -> Path:
    return binned_root / f"nhit_{sanitize_label(cell.nhit_bin)}" / f"predE_{sanitize_label(cell.predE_bin)}"


def open_tree(path: Path, tree_name: str):
    f = uproot.open(path)
    try:
        if f"{tree_name};1" in f:
            return f, f[f"{tree_name};1"]
        return f, f[tree_name]
    except Exception:
        f.close()
        raise


def branch_list(tree) -> List[str]:
    return list(tree.keys())


def arrays_for_tree(
    path: Path,
    tree_name: str,
    branches: Sequence[str],
    optional_branches: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    f, tree = open_tree(path, tree_name)
    try:
        available = set(branch_list(tree))
        missing = [name for name in branches if name not in available]
        if missing:
            raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
        selected = list(dict.fromkeys(list(branches) + [name for name in optional_branches if name in available]))
        return tree.arrays(selected, library="np")
    finally:
        f.close()


def read_run_config(run_dir: Path) -> Dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cut_mask(
    arrays: Dict[str, np.ndarray],
    *,
    cut_pinc_max: float,
    cut_theta_max_deg: float,
    cut_fitstat_equals: int,
    cut_dcedge_min: Optional[float],
    core_box: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    n_events = len(next(iter(arrays.values())) if arrays else [])
    mask = np.ones(n_events, dtype=bool)
    mask &= np.asarray(arrays["pincness"] < float(cut_pinc_max))
    mask &= np.asarray(arrays["fitstat"] == int(cut_fitstat_equals))
    mask &= np.asarray(arrays["theta"] < math.radians(float(cut_theta_max_deg)))
    if cut_dcedge_min is not None:
        mask &= np.asarray(arrays["dcedge"] > float(cut_dcedge_min))
    if core_box is not None:
        x_min, x_max, y_min, y_max = core_box
        mask &= np.asarray(arrays["xc"] >= float(x_min))
        mask &= np.asarray(arrays["xc"] <= float(x_max))
        mask &= np.asarray(arrays["yc"] >= float(y_min))
        mask &= np.asarray(arrays["yc"] <= float(y_max))
    return mask


def valid_truth_mask(loge: np.ndarray, theta_deg: np.ndarray, loge_edges: np.ndarray, theta_edges_deg: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(loge)
        & np.isfinite(theta_deg)
        & (loge >= loge_edges[0])
        & (loge < loge_edges[-1])
        & (theta_deg >= theta_edges_deg[0])
        & (theta_deg < theta_edges_deg[-1])
    )


def fill_histograms(
    loge: np.ndarray,
    theta_deg: np.ndarray,
    weight: np.ndarray,
    loge_edges: np.ndarray,
    theta_edges_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = valid_truth_mask(loge, theta_deg, loge_edges, theta_edges_deg)
    shape = (len(loge_edges) - 1, len(theta_edges_deg) - 1)
    if not np.any(mask):
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)

    hist_sumw, _, _ = np.histogram2d(
        loge[mask],
        theta_deg[mask],
        bins=[loge_edges, theta_edges_deg],
        weights=weight[mask],
    )
    hist_count, _, _ = np.histogram2d(
        loge[mask],
        theta_deg[mask],
        bins=[loge_edges, theta_edges_deg],
    )
    return hist_sumw.astype(np.float64, copy=False), hist_count.astype(np.float64, copy=False)


def load_weight(
    arrays: Dict[str, np.ndarray],
    *,
    weight_branch: str,
    allow_missing_weight: bool,
) -> np.ndarray:
    if weight_branch in arrays:
        weight = np.asarray(arrays[weight_branch], dtype=np.float64)
    elif allow_missing_weight:
        ref = next(iter(arrays.values()))
        weight = np.ones(len(ref), dtype=np.float64)
    else:
        raise KeyError(f"Missing weight branch {weight_branch!r}")
    return weight


def denominator_file_hist_worker(kwargs: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(kwargs["path"]))
    tree_name = str(kwargs["tree_name"])
    branches = list(kwargs["branches"])
    optional_branches = list(kwargs["optional_branches"])
    loge_edges = np.asarray(kwargs["loge_edges"], dtype=np.float64)
    theta_edges_deg = np.asarray(kwargs["theta_edges_deg"], dtype=np.float64)
    weight_branch = str(kwargs["weight_branch"])
    allow_missing_weight = bool(kwargs["allow_missing_weight"])

    arrays = arrays_for_tree(path, tree_name, branches, optional_branches=optional_branches)
    n_events = len(arrays["mc_energy"])
    cut_mask = build_cut_mask(
        arrays,
        cut_pinc_max=float(kwargs["cut_pinc_max"]),
        cut_theta_max_deg=float(kwargs["cut_theta_max_deg"]),
        cut_fitstat_equals=int(kwargs["cut_fitstat_equals"]),
        cut_dcedge_min=kwargs["cut_dcedge_min"],
        core_box=kwargs["core_box"],
    )
    weight = load_weight(arrays, weight_branch=weight_branch, allow_missing_weight=allow_missing_weight)
    loge = np.log10(np.asarray(arrays["mc_energy"], dtype=np.float64))
    theta_deg = np.degrees(np.asarray(arrays["mc_theta"], dtype=np.float64))
    truth_mask = cut_mask & valid_truth_mask(loge, theta_deg, loge_edges, theta_edges_deg)
    sumw, count = fill_histograms(
        loge[cut_mask],
        theta_deg[cut_mask],
        weight[cut_mask],
        loge_edges,
        theta_edges_deg,
    )
    return {
        "path": str(path),
        "sumw": sumw,
        "count": count,
        "total_events": int(n_events),
        "cut_pass_events": int(np.count_nonzero(cut_mask)),
        "truth_range_events": int(np.count_nonzero(truth_mask)),
    }


def accumulate_denominator(
    raw_files: Sequence[Path],
    *,
    tree_name: str,
    loge_edges: np.ndarray,
    theta_edges_deg: np.ndarray,
    weight_branch: str,
    allow_missing_weight: bool,
    cut_pinc_max: float,
    cut_theta_max_deg: float,
    cut_fitstat_equals: int,
    cut_dcedge_min: Optional[float],
    core_box: Optional[Tuple[float, float, float, float]],
    denominator_workers: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    shape = (len(loge_edges) - 1, len(theta_edges_deg) - 1)
    denominator_sumw = np.zeros(shape, dtype=np.float64)
    denominator_count = np.zeros(shape, dtype=np.float64)
    total_events = 0
    cut_pass_events = 0
    truth_range_events = 0

    branches = ["mc_energy", "mc_theta", "pincness", "fitstat", "theta"]
    optional_branches: List[str] = []
    if allow_missing_weight:
        optional_branches.append(weight_branch)
    else:
        branches.append(weight_branch)
    if cut_dcedge_min is not None:
        branches.append("dcedge")
    if core_box is not None:
        branches.extend(["xc", "yc"])

    worker_kwargs = [
        {
            "path": str(path),
            "tree_name": tree_name,
            "branches": branches,
            "optional_branches": optional_branches,
            "loge_edges": loge_edges,
            "theta_edges_deg": theta_edges_deg,
            "weight_branch": weight_branch,
            "allow_missing_weight": allow_missing_weight,
            "cut_pinc_max": cut_pinc_max,
            "cut_theta_max_deg": cut_theta_max_deg,
            "cut_fitstat_equals": cut_fitstat_equals,
            "cut_dcedge_min": cut_dcedge_min,
            "core_box": core_box,
        }
        for path in raw_files
    ]

    progress_every = max(1, int(progress_every))
    denominator_workers = max(1, int(denominator_workers))
    if denominator_workers == 1:
        result_iter = map(denominator_file_hist_worker, worker_kwargs)
        for idx, result in enumerate(result_iter, start=1):
            denominator_sumw += result["sumw"]
            denominator_count += result["count"]
            total_events += int(result["total_events"])
            cut_pass_events += int(result["cut_pass_events"])
            truth_range_events += int(result["truth_range_events"])
            if idx % progress_every == 0 or idx == len(raw_files):
                print(
                    f"[denominator] {idx}/{len(raw_files)} files | "
                    f"total={total_events} cut_pass={cut_pass_events}",
                    flush=True,
                )
    else:
        print(f"[denominator] scanning {len(raw_files)} files with {denominator_workers} workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=denominator_workers) as executor:
            futures = [executor.submit(denominator_file_hist_worker, item) for item in worker_kwargs]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                denominator_sumw += result["sumw"]
                denominator_count += result["count"]
                total_events += int(result["total_events"])
                cut_pass_events += int(result["cut_pass_events"])
                truth_range_events += int(result["truth_range_events"])
                if idx % progress_every == 0 or idx == len(raw_files):
                    print(
                        f"[denominator] {idx}/{len(raw_files)} files | "
                        f"total={total_events} cut_pass={cut_pass_events}",
                        flush=True,
                    )

    meta = {
        "raw_files": len(raw_files),
        "total_events": int(total_events),
        "cut_pass_events": int(cut_pass_events),
        "truth_range_events": int(truth_range_events),
        "denominator_sumw_total": float(np.sum(denominator_sumw)),
        "denominator_count_total": int(np.sum(denominator_count)),
    }
    return denominator_sumw, denominator_count, meta


def _read_scalar_npz_value(data: np.lib.npyio.NpzFile, name: str) -> Optional[object]:
    if name not in data.files:
        return None
    value = data[name]
    if getattr(value, "shape", ()) == ():
        return value.item()
    return value.tolist()


def load_primary_denominator(
    path: Path,
    *,
    loge_edges: np.ndarray,
    theta_edges_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Primary denominator NPZ does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = ["denominator_sumw", "denominator_count", "logE_true_edges", "theta_true_edges_deg"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {', '.join(missing)}")

        denominator_sumw = np.asarray(data["denominator_sumw"], dtype=np.float64)
        denominator_count = np.asarray(data["denominator_count"], dtype=np.float64)
        source_root = _read_scalar_npz_value(data, "source_root")
        source_hist_count = _read_scalar_npz_value(data, "source_hist_count")
        source_hist_sumw = _read_scalar_npz_value(data, "source_hist_sumw")
        denominator_loge_edges = np.asarray(data["logE_true_edges"], dtype=np.float64)
        denominator_theta_edges = np.asarray(data["theta_true_edges_deg"], dtype=np.float64)

    expected_shape = (len(loge_edges) - 1, len(theta_edges_deg) - 1)
    if denominator_sumw.shape != expected_shape:
        raise ValueError(f"denominator_sumw shape {denominator_sumw.shape} does not match {expected_shape}")
    if denominator_count.shape != expected_shape:
        raise ValueError(f"denominator_count shape {denominator_count.shape} does not match {expected_shape}")
    if not np.allclose(denominator_loge_edges, loge_edges, rtol=0.0, atol=1e-9):
        raise ValueError("Primary denominator logE edges do not match requested Stage A edges.")
    if not np.allclose(denominator_theta_edges, theta_edges_deg, rtol=0.0, atol=1e-9):
        raise ValueError("Primary denominator theta edges do not match requested Stage A edges.")

    finite_sumw = np.isfinite(denominator_sumw)
    finite_count = np.isfinite(denominator_count)
    if not np.all(finite_sumw):
        raise ValueError("Primary denominator_sumw contains non-finite values.")
    if not np.all(finite_count):
        raise ValueError("Primary denominator_count contains non-finite values.")

    meta = {
        "type": "primary_thrown_histogram",
        "npz_path": str(path),
        "weighted_sumw_source": "hntotmc",
        "unweighted_count_source": "hntotmc0",
        "source_root": None if source_root is None else str(source_root),
        "source_hist_sumw": None if source_hist_sumw is None else str(source_hist_sumw),
        "source_hist_count": None if source_hist_count is None else str(source_hist_count),
        "denominator_sumw_total": float(np.sum(denominator_sumw)),
        "denominator_count_total": int(round(float(np.sum(denominator_count)))),
        "nonzero_sumw_bins": int(np.count_nonzero(denominator_sumw > 0)),
        "nonzero_count_bins": int(np.count_nonzero(denominator_count > 0)),
        "empty_sumw_bins": int(np.count_nonzero(denominator_sumw <= 0)),
        "empty_count_bins": int(np.count_nonzero(denominator_count <= 0)),
    }
    return denominator_sumw, denominator_count, meta


def accumulate_numerators(
    cells: Sequence[CellSpec],
    *,
    binned_root: Path,
    tree_name: str,
    loge_edges: np.ndarray,
    theta_edges_deg: np.ndarray,
    weight_branch: str,
    allow_missing_weight: bool,
    aperture_r_opt_deg: Optional[np.ndarray],
    aperture_angle_branch: str,
    max_files_per_cell: Optional[int],
    allow_missing_cell_dirs: bool,
    numerator_workers: int,
    files_per_task: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    n_cells = len(cells)
    shape = (n_cells, len(loge_edges) - 1, len(theta_edges_deg) - 1)
    numerator_sumw = np.zeros(shape, dtype=np.float64)
    numerator_count = np.zeros(shape, dtype=np.float64)
    summaries: List[Dict[str, object]] = [
        {
            "cell_id": int(cell.cell_id),
            "cell_index": int(cell_idx),
            "nhit_bin": cell.nhit_bin,
            "predE_bin": cell.predE_bin,
            "input_files": 0,
            "input_dir": str(binned_cell_dir(binned_root, cell)),
            "events": 0,
            "aperture_kept_events": 0 if aperture_r_opt_deg is not None else "",
            "aperture_kept_fraction": "" if aperture_r_opt_deg is None else 0.0,
            "truth_range_events": 0,
            "sumw_total": 0.0,
        }
        for cell_idx, cell in enumerate(cells)
    ]
    worker_kwargs: List[Dict[str, object]] = []

    for cell_idx, cell in enumerate(cells):
        cell_dir = binned_cell_dir(binned_root, cell)
        if not cell_dir.exists() and not allow_missing_cell_dirs:
            raise FileNotFoundError(
                f"Missing binned ROOT directory for selected cell {cell.cell_id} "
                f"({cell.nhit_bin}, {cell.predE_bin}): {cell_dir}"
            )
        files = sorted(cell_dir.glob("*.root")) if cell_dir.exists() else []
        if max_files_per_cell is not None:
            files = files[:max_files_per_cell]
        summaries[cell_idx]["input_files"] = len(files)
        for chunk in chunk_sequence(files, files_per_task):
            worker_kwargs.append(
                {
                    "cell_index": cell_idx,
                    "paths": [str(path) for path in chunk],
                    "tree_name": tree_name,
                    "loge_edges": loge_edges,
                    "theta_edges_deg": theta_edges_deg,
                    "weight_branch": weight_branch,
                    "allow_missing_weight": allow_missing_weight,
                    "aperture_r_opt_deg": (
                        None
                        if aperture_r_opt_deg is None
                        else float(aperture_r_opt_deg[cell_idx])
                    ),
                    "aperture_angle_branch": aperture_angle_branch,
                }
            )

    progress_every = max(1, int(progress_every))
    numerator_workers = max(1, int(numerator_workers))
    if not worker_kwargs:
        return numerator_sumw, numerator_count, summaries

    def consume_result(result: Dict[str, object]) -> None:
        cell_idx = int(result["cell_index"])
        numerator_sumw[cell_idx] += result["sumw"]
        numerator_count[cell_idx] += result["count"]
        summaries[cell_idx]["events"] += int(result["events"])
        if result.get("aperture_kept_events") is not None:
            summaries[cell_idx]["aperture_kept_events"] += int(result["aperture_kept_events"])
        summaries[cell_idx]["truth_range_events"] += int(result["truth_range_events"])
        summaries[cell_idx]["sumw_total"] += float(result["sumw_total"])

    if numerator_workers == 1:
        for task_idx, kwargs in enumerate(worker_kwargs, start=1):
            consume_result(numerator_chunk_hist_worker(kwargs))
            if task_idx % progress_every == 0 or task_idx == len(worker_kwargs):
                print(f"[numerator] {task_idx}/{len(worker_kwargs)} tasks complete", flush=True)
    else:
        print(
            f"[numerator] scanning {len(worker_kwargs)} tasks with {numerator_workers} workers",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=numerator_workers) as executor:
            futures = [executor.submit(numerator_chunk_hist_worker, item) for item in worker_kwargs]
            for task_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                consume_result(future.result())
                if task_idx % progress_every == 0 or task_idx == len(worker_kwargs):
                    print(f"[numerator] {task_idx}/{len(worker_kwargs)} tasks complete", flush=True)

    for cell_idx, cell in enumerate(cells):
        if summaries[cell_idx].get("aperture_kept_events") != "":
            events = int(summaries[cell_idx]["events"])
            kept = int(summaries[cell_idx]["aperture_kept_events"])
            summaries[cell_idx]["aperture_kept_fraction"] = float(kept / events) if events > 0 else 0.0
        print(
            f"[numerator] cell {cell_idx + 1}/{len(cells)} "
            f"{cell.nhit_bin} {cell.predE_bin} | "
            f"files={summaries[cell_idx]['input_files']} events={summaries[cell_idx]['events']}",
            flush=True,
        )

    return numerator_sumw, numerator_count, summaries


def chunk_sequence(items: Sequence[Path], chunk_size: int) -> List[List[Path]]:
    chunk_size = max(1, int(chunk_size))
    return [list(items[start : start + chunk_size]) for start in range(0, len(items), chunk_size)]


def numerator_chunk_hist_worker(kwargs: Dict[str, object]) -> Dict[str, object]:
    paths = [Path(str(path)) for path in kwargs["paths"]]
    tree_name = str(kwargs["tree_name"])
    loge_edges = np.asarray(kwargs["loge_edges"], dtype=np.float64)
    theta_edges_deg = np.asarray(kwargs["theta_edges_deg"], dtype=np.float64)
    weight_branch = str(kwargs["weight_branch"])
    allow_missing_weight = bool(kwargs["allow_missing_weight"])
    aperture_r_opt_deg = kwargs.get("aperture_r_opt_deg")
    aperture_angle_branch = str(kwargs.get("aperture_angle_branch") or "mc_dangle")
    use_aperture = aperture_r_opt_deg is not None
    aperture_r_opt_rad = math.radians(float(aperture_r_opt_deg)) if use_aperture else None
    shape = (len(loge_edges) - 1, len(theta_edges_deg) - 1)
    sumw = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    events = 0
    aperture_kept_events = 0
    truth_range_events = 0
    cell_sumw_total = 0.0

    for path in paths:
        branches = ["mc_energy", "mc_theta"]
        optional_branches: List[str] = []
        if allow_missing_weight:
            optional_branches.append(weight_branch)
        else:
            branches.append(weight_branch)
        if use_aperture:
            branches.append(aperture_angle_branch)
        arrays = arrays_for_tree(path, tree_name, branches, optional_branches=optional_branches)
        n_events = len(arrays["mc_energy"])
        events += int(n_events)
        weight = load_weight(arrays, weight_branch=weight_branch, allow_missing_weight=allow_missing_weight)
        loge = np.log10(np.asarray(arrays["mc_energy"], dtype=np.float64))
        theta_deg = np.degrees(np.asarray(arrays["mc_theta"], dtype=np.float64))
        if use_aperture:
            dangle = np.asarray(arrays[aperture_angle_branch], dtype=np.float64)
            aperture_mask = np.isfinite(dangle) & (dangle <= float(aperture_r_opt_rad))
            aperture_kept_events += int(np.count_nonzero(aperture_mask))
            loge = loge[aperture_mask]
            theta_deg = theta_deg[aperture_mask]
            weight = weight[aperture_mask]
        truth_range_events += int(np.count_nonzero(valid_truth_mask(loge, theta_deg, loge_edges, theta_edges_deg)))
        hist_sumw, hist_count = fill_histograms(loge, theta_deg, weight, loge_edges, theta_edges_deg)
        sumw += hist_sumw
        count += hist_count
        cell_sumw_total += float(np.sum(weight[np.isfinite(weight)]))

    return {
        "cell_index": int(kwargs["cell_index"]),
        "sumw": sumw,
        "count": count,
        "events": int(events),
        "aperture_kept_events": int(aperture_kept_events) if use_aperture else None,
        "truth_range_events": int(truth_range_events),
        "sumw_total": float(cell_sumw_total),
        "files": int(len(paths)),
    }


def divide_response(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    response = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator[None, :, :], out=response, where=denominator[None, :, :] > 0)
    return response


def effective_area(eta: np.ndarray, theta_edges_deg: np.ndarray, s0_m2: float) -> Tuple[np.ndarray, np.ndarray]:
    theta_centers_deg = 0.5 * (theta_edges_deg[:-1] + theta_edges_deg[1:])
    cos_theta = np.cos(np.radians(theta_centers_deg)).astype(np.float64)
    return float(s0_m2) * eta * cos_theta[None, None, :], theta_centers_deg


def thrown_area_metadata(s0_m2: float) -> Dict[str, object]:
    return {
        "s0_m2": float(s0_m2),
        "s0_cm2": float(s0_m2) * CM2_PER_M2,
        "injection_area_value": float(DEFAULT_INJECTION_AREA_CM2),
        "injection_area_unit": "cm^2",
        "geometry": DEFAULT_THROWN_GEOMETRY,
        "core_x_range_m": list(DEFAULT_THROWN_CORE_RANGE_M),
        "core_y_range_m": list(DEFAULT_THROWN_CORE_RANGE_M),
        "source_evidence": (
            "IHEP upstream t_evth.v[48] injectionArea=4.0e10; "
            "coreX_WL/coreY_WL are in cm and map to eventout mc_xc/mc_yc after division by 100."
        ),
        "unit_note": "4.0e10 is cm^2 = 4.0e6 m^2, not mm^2 = 4.0e4 m^2.",
    }


def finite_summary(name: str, values: np.ndarray) -> Dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"name": name, "finite_count": 0, "min": None, "max": None, "sum": None}
    return {
        "name": name,
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "sum": float(np.sum(finite)),
    }


def write_cell_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_denominator_occupancy_csv(
    path: Path,
    *,
    denominator_sumw: np.ndarray,
    denominator_count: np.ndarray,
    loge_edges: np.ndarray,
    theta_edges_deg: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "logE_true_lo",
                "logE_true_hi",
                "theta_true_deg_lo",
                "theta_true_deg_hi",
                "denominator_count",
                "denominator_sumw",
            ],
        )
        writer.writeheader()
        for i in range(denominator_count.shape[0]):
            for j in range(denominator_count.shape[1]):
                writer.writerow(
                    {
                        "logE_true_lo": float(loge_edges[i]),
                        "logE_true_hi": float(loge_edges[i + 1]),
                        "theta_true_deg_lo": float(theta_edges_deg[j]),
                        "theta_true_deg_hi": float(theta_edges_deg[j + 1]),
                        "denominator_count": int(denominator_count[i, j]),
                        "denominator_sumw": float(denominator_sumw[i, j]),
                    }
                )


def write_markdown_summary(
    path: Path,
    *,
    metadata: Dict[str, object],
    cell_summaries: Sequence[Dict[str, object]],
    empty_denominator_bins: int,
    low_denominator_bins: int,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage A Response Summary\n\n")
        f.write(f"- Response type: `{metadata['response_type']}`\n")
        f.write(f"- Weighting: `{metadata['weighting']}`\n")
        f.write(f"- Cells: {metadata['n_cells']}\n")
        f.write(f"- Truth shape: {metadata['truth_shape']}\n")
        f.write(f"- Absolute effective area status: `{metadata['absolute_effective_area_status']}`\n")
        if metadata.get("s0_m2") is not None:
            f.write(f"- S0: {metadata['s0_m2']} m^2\n")
        thrown_area = metadata.get("thrown_area")
        if isinstance(thrown_area, dict):
            f.write(
                "- Injection area: "
                f"{thrown_area.get('injection_area_value')} {thrown_area.get('injection_area_unit')} "
                f"= {thrown_area.get('s0_m2')} m^2\n"
            )
            f.write(f"- Thrown geometry: `{thrown_area.get('geometry')}`\n")
        f.write(f"- Empty denominator bins: {empty_denominator_bins}\n")
        f.write(f"- Denominator bins with count < 10: {low_denominator_bins}\n")
        f.write(f"- NPZ: `{metadata['npz_path']}`\n\n")
        f.write("| cell_id | nhit_bin | predE_bin | files | events | aperture kept | aperture fraction | sumw_total |\n")
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in cell_summaries:
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{row['input_files']} | {row['events']} | {row.get('aperture_kept_events', '')} | "
                f"{row.get('aperture_kept_fraction', '')} | {row['sumw_total']:.6g} |\n"
            )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    input_root = Path(args.input_root).resolve()
    binned_root = Path(args.binned_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_csv = Path(args.cell_selection_csv).resolve()
    run_dir = Path(args.run_dir).resolve()
    core_box = tuple(args.core_box) if args.core_box is not None else None

    cells = load_cells(selection_csv)
    aperture_psf_npz = Path(args.aperture_psf_npz).resolve() if str(args.aperture_psf_npz).strip() else None
    aperture_r_opt_deg, aperture_meta = load_aperture_radii(cells, aperture_psf_npz)
    if aperture_meta is not None:
        aperture_meta["angle_branch"] = str(args.aperture_angle_branch)
    run_config = read_run_config(run_dir)
    loge_edges = make_edges(args.logE_min, args.logE_max, args.logE_step)
    theta_edges_deg = make_edges(args.theta_min_deg, args.theta_max_deg, args.theta_step_deg)

    if args.use_eventout_denominator:
        raw_files = discover_root_files(input_root, args.max_raw_files)
        if not raw_files:
            raise FileNotFoundError(f"No ROOT files found in {input_root}")

        denominator_sumw, denominator_count, denominator_meta = accumulate_denominator(
            raw_files,
            tree_name=args.tree_name,
            loge_edges=loge_edges,
            theta_edges_deg=theta_edges_deg,
            weight_branch=args.weight_branch,
            allow_missing_weight=args.allow_missing_weight,
            cut_pinc_max=args.cut_pinc_max,
            cut_theta_max_deg=args.cut_theta_max_deg,
            cut_fitstat_equals=args.cut_fitstat_equals,
            cut_dcedge_min=args.cut_dcedge_min,
            core_box=core_box,
            denominator_workers=args.denominator_workers,
            progress_every=args.denominator_progress_every,
        )
        response_type = "conditional_eventout_response"
        absolute_effective_area_status = "pending_thrown_normalization"
    else:
        denominator_sumw, denominator_count, denominator_meta = load_primary_denominator(
            Path(args.primary_denominator_npz).resolve(),
            loge_edges=loge_edges,
            theta_edges_deg=theta_edges_deg,
        )
        response_type = "primary_thrown_aperture_conditioned_response" if aperture_r_opt_deg is not None else "primary_thrown_response"
        absolute_effective_area_status = "available"

    numerator_sumw, numerator_count, cell_summaries = accumulate_numerators(
        cells,
        binned_root=binned_root,
        tree_name=args.tree_name,
        loge_edges=loge_edges,
        theta_edges_deg=theta_edges_deg,
        weight_branch=args.weight_branch,
        allow_missing_weight=args.allow_missing_weight,
        aperture_r_opt_deg=aperture_r_opt_deg,
        aperture_angle_branch=str(args.aperture_angle_branch),
        max_files_per_cell=args.max_binned_files_per_cell,
        allow_missing_cell_dirs=args.allow_missing_cell_dirs,
        numerator_workers=args.numerator_workers,
        files_per_task=args.numerator_files_per_task,
        progress_every=args.numerator_progress_every,
    )

    eta = divide_response(numerator_sumw, denominator_sumw)
    eta_count = divide_response(numerator_count, denominator_count)
    theta_centers_deg = 0.5 * (theta_edges_deg[:-1] + theta_edges_deg[1:])
    if absolute_effective_area_status == "available":
        a_eff, theta_centers_deg = effective_area(eta, theta_edges_deg, args.s0_m2)
    else:
        a_eff = np.full_like(eta, np.nan, dtype=np.float64)

    npz_path = output_dir / args.npz_name
    np.savez_compressed(
        npz_path,
        eta=eta.astype(np.float32),
        eta_count=eta_count.astype(np.float32),
        eta_conditional=eta.astype(np.float32),
        eta_conditional_count=eta_count.astype(np.float32),
        a_eff=a_eff.astype(np.float32),
        numerator_sumw=numerator_sumw,
        numerator_count=numerator_count,
        denominator_sumw=denominator_sumw,
        denominator_count=denominator_count,
        logE_true_edges=loge_edges,
        theta_true_edges_deg=theta_edges_deg,
        theta_true_centers_deg=theta_centers_deg,
        cos_theta_center=np.cos(np.radians(theta_centers_deg)).astype(np.float32),
        s0_m2=np.asarray(args.s0_m2, dtype=np.float32),
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        aperture_r_opt_deg=np.asarray(
            np.full(len(cells), np.nan, dtype=np.float32)
            if aperture_r_opt_deg is None
            else aperture_r_opt_deg.astype(np.float32)
        ),
    )

    empty_denominator_bins = int(np.count_nonzero(denominator_count <= 0))
    low_denominator_bins = int(np.count_nonzero((denominator_count > 0) & (denominator_count < 10)))
    max_cell_eta_sum = np.sum(eta, axis=0)
    metadata: Dict[str, object] = {
        "response_type": response_type,
        "weighting": f"{args.weight_branch}_baseline",
        "absolute_effective_area_status": absolute_effective_area_status,
        "npz_path": str(npz_path),
        "input_root": str(input_root),
        "binned_root": str(binned_root),
        "cell_selection_csv": str(selection_csv),
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "checkpoints" / "best_model.pt"),
        "run_config_snapshot": run_config,
        "tree_name": args.tree_name,
        "n_cells": len(cells),
        "truth_shape": [int(len(loge_edges) - 1), int(len(theta_edges_deg) - 1)],
        "cells": [
            {
                "cell_id": int(cell.cell_id),
                "cell_index": int(idx),
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "mc_count_reference": int(cell.mc_count),
                "selection_version": cell.selection_version,
                "selection_reason": cell.selection_reason,
                "raw_ledger_version": cell.raw_ledger_version,
                "cell_role": cell.cell_role,
                "role_reason": cell.role_reason,
                "source_pool": cell.source_pool,
            }
            for idx, cell in enumerate(cells)
        ],
        "logE_true_edges": loge_edges.tolist(),
        "theta_true_edges_deg": theta_edges_deg.tolist(),
        "theta_true_centers_deg": theta_centers_deg.tolist(),
        "s0_m2": float(args.s0_m2),
        "thrown_area": thrown_area_metadata(float(args.s0_m2)),
        "effective_area_formula": "a_eff_m2 = s0_m2 * cos(theta_true_bin_center) * eta",
        "cos_theta_source": "theta_true_bin_center",
        "response_aperture_conditioning": (
            {
                **aperture_meta,
                "note": (
                    "Numerator is restricted by the configured aperture. "
                    "Use with Stage F/G containment_r_opt=1 to avoid applying the same aperture twice."
                ),
            }
            if aperture_meta is not None
            else {"mode": "none", "note": "Default all-direction cell response; Stage F/G should apply PSF containment for on-aperture excess."}
        ),
        "cuts": {
            "pincness_lt": float(args.cut_pinc_max),
            "fitstat_equals": int(args.cut_fitstat_equals),
            "theta_reco_deg_lt": float(args.cut_theta_max_deg),
            "dcedge_gt": None if args.cut_dcedge_min is None else float(args.cut_dcedge_min),
            "core_box": None if core_box is None else [float(v) for v in core_box],
            "mc_dangle_cut": None if aperture_r_opt_deg is None else "cell-dependent r_opt_deg from aperture_psf_npz",
        },
        "allow_missing_cell_dirs": bool(args.allow_missing_cell_dirs),
        "denominator": denominator_meta,
        "primary_denominator_npz": None if args.use_eventout_denominator else str(Path(args.primary_denominator_npz).resolve()),
        "use_eventout_denominator": bool(args.use_eventout_denominator),
        "denominator_workers": int(args.denominator_workers),
        "numerator_workers": int(args.numerator_workers),
        "numerator_files_per_task": int(args.numerator_files_per_task),
        "array_summaries": [
            finite_summary("eta", eta),
            finite_summary("eta_count", eta_count),
            finite_summary("numerator_sumw", numerator_sumw),
            finite_summary("denominator_sumw", denominator_sumw),
            finite_summary("a_eff", a_eff),
        ],
        "empty_denominator_bins": empty_denominator_bins,
        "low_denominator_count_bins_lt10": low_denominator_bins,
        "max_sum_eta_over_configured_cells": float(np.nanmax(max_cell_eta_sum)) if max_cell_eta_sum.size else None,
        "max_sum_eta_over_v1_cells": float(np.nanmax(max_cell_eta_sum)) if max_cell_eta_sum.size else None,
        "a_eff_max_m2": float(np.nanmax(a_eff)) if a_eff.size else None,
        "elapsed_seconds": float(time.perf_counter() - start),
        "obsolete_cache_warning": "/mnt/mydisk/WCDA_simulation_binned_selectedcuts used an older checkpoint/cut set and is not an official Stage A input.",
    }

    metadata_path = output_dir / args.metadata_name
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_cell_summary_csv(output_dir / "cell_response_summary.csv", cell_summaries)
    write_denominator_occupancy_csv(
        output_dir / "denominator_occupancy.csv",
        denominator_sumw=denominator_sumw,
        denominator_count=denominator_count,
        loge_edges=loge_edges,
        theta_edges_deg=theta_edges_deg,
    )
    write_markdown_summary(
        output_dir / "response_2d_summary.md",
        metadata=metadata,
        cell_summaries=cell_summaries,
        empty_denominator_bins=empty_denominator_bins,
        low_denominator_bins=low_denominator_bins,
    )

    print(f"Wrote {npz_path}")
    print(f"Wrote {metadata_path}")
    print(f"Shape: {eta.shape}")


if __name__ == "__main__":
    main()
