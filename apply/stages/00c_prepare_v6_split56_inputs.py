#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Starting in version 5.7.0, Uproot will default to writing RNTuples instead of TTrees.*",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE_ROOT = Path("/mnt/mydisk/WCDA_simulation_binned_response_v6_64670")
TARGET_ROOT = Path("/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_split56_candidate")
RUN_DIR = Path("/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670")

NHIT_BINS = ["[125,200)", "[200,300)", "[300,500)", "[500,800)", "[800,1100)", "[1100,2000)", "[2000,3000)"]
PRED_BINS = [
    "[2,2.5)",
    "[2.5,3)",
    "[3,3.25)",
    "[3.25,3.5)",
    "[3.5,3.75)",
    "[3.75,4.0)",
    "[4.0,4.25)",
    "[4.25,4.5)",
    "[4.5,4.75)",
    "[4.75,5.0)",
    "[5,5.5)",
    "[5.5,6)",
    ">=6",
]
SOURCE_LOW_NHIT = "[100,200)"
TARGET_LOW_NHIT = "[125,200)"
SOURCE_HIGH_NHIT = ">=2000"
TARGET_HIGH_NHIT = "[2000,3000)"
SOURCE_GE5_PRED = ">=5"
VERSION = "v6_64670_split56_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v6 _64670 split56 91-cell candidate cache and prefit selector."
    )
    parser.add_argument("--source-binned-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--target-binned-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--source-bin-counts-csv", type=Path, default=SOURCE_ROOT / "summary" / "bin_counts.csv")
    parser.add_argument("--candidate-ledger-csv", type=Path, default=Path("apply/config/cell_ledger_v6_split56_candidate.csv"))
    parser.add_argument("--prefit-selector-csv", type=Path, default=Path("apply/config/cell_selector_v6_split56_drop4_psfborrow.prefit.csv"))
    parser.add_argument("--systematics-selector-csv", type=Path, default=Path("apply/config/cell_selector_v6_split56_systematics.csv"))
    parser.add_argument("--high-energy-selector-csv", type=Path, default=Path("apply/config/cell_selector_v6_split56_high_energy_probes.csv"))
    parser.add_argument("--manifest-json", type=Path, default=Path("apply/config/v6_64670_split56_strategy_manifest.json"))
    parser.add_argument("--diagnostics-html", type=Path, default=Path("apply/report/v6_64670_split56_cell_selection_diagnostics.html"))
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--write-configs", action="store_true", default=False)
    parser.add_argument("--prepare-cache", action="store_true", default=False)
    parser.add_argument("--overwrite-filtered-cache", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-files-per-source-bin", type=int, default=None)
    parser.add_argument("--min-baseline-mc-count", type=int, default=1000)
    parser.add_argument("--ridge-min-peak-fraction", type=float, default=0.10)
    parser.add_argument("--write-diagnostics", action="store_true", default=False)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def interval_mask(values: np.ndarray, label: str) -> np.ndarray:
    low, high = parse_interval(label)
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values >= float(low)
    if high is not None:
        mask &= values < float(high)
    return mask


def intervals_overlap(left: str, right: str) -> bool:
    a_low, a_high = parse_interval(left)
    b_low, b_high = parse_interval(right)
    lo = max(-math.inf if a_low is None else a_low, -math.inf if b_low is None else b_low)
    hi = min(math.inf if a_high is None else a_high, math.inf if b_high is None else b_high)
    return lo < hi


def interval_contains(container: str, inner: str) -> bool:
    c_low, c_high = parse_interval(container)
    i_low, i_high = parse_interval(inner)
    low_ok = c_low is None or (i_low is not None and i_low >= c_low)
    high_ok = c_high is None or (i_high is not None and i_high <= c_high)
    return low_ok and high_ok


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


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def source_nhit_bin(target_nhit: str) -> str:
    if target_nhit == TARGET_LOW_NHIT:
        return SOURCE_LOW_NHIT
    if target_nhit == TARGET_HIGH_NHIT:
        return SOURCE_HIGH_NHIT
    return target_nhit


def source_pred_bins(target_pred: str) -> List[str]:
    if target_pred == ">=6":
        return [SOURCE_GE5_PRED]
    low, high = parse_interval(target_pred)
    if low is None or high is None:
        return [target_pred]
    source_bins = [
        "[2,3)",
        "[3,3.25)",
        "[3.25,3.5)",
        "[3.5,3.75)",
        "[3.75,4.0)",
        "[4.0,4.25)",
        "[4.25,4.5)",
        "[4.5,4.75)",
        "[4.75,5.0)",
        SOURCE_GE5_PRED,
    ]
    return [label for label in source_bins if intervals_overlap(target_pred, label)]


def target_cache_needs_filter(target_nhit: str, target_pred: str) -> bool:
    if target_nhit in {TARGET_LOW_NHIT, TARGET_HIGH_NHIT}:
        return True
    return any(not interval_contains(target_pred, source_pred) for source_pred in source_pred_bins(target_pred))


def target_cache_can_symlink(target_nhit: str, target_pred: str) -> bool:
    source_preds = source_pred_bins(target_pred)
    return (
        not target_cache_needs_filter(target_nhit, target_pred)
        and len(source_preds) == 1
        and interval_contains(source_preds[0], target_pred)
    )


def source_dir(source_root: Path, target_nhit: str, source_pred: str) -> Path:
    return source_root / f"nhit_{sanitize_label(source_nhit_bin(target_nhit))}" / f"predE_{sanitize_label(source_pred)}"


def target_dir(target_root: Path, target_nhit: str, target_pred: str) -> Path:
    return target_root / f"nhit_{sanitize_label(target_nhit)}" / f"predE_{sanitize_label(target_pred)}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def source_count_lookup(source_counts_csv: Path) -> Dict[Tuple[str, str], int]:
    lookup: Dict[Tuple[str, str], int] = {}
    for row in read_csv(source_counts_csv):
        lookup[(row["nhit_bin"], row["predE_bin"])] = int(row.get("count") or 0)
    return lookup


def estimated_count(target_nhit: str, target_pred: str, source_counts: Dict[Tuple[str, str], int]) -> int:
    if target_cache_needs_filter(target_nhit, target_pred):
        return 0
    return int(sum(source_counts.get((source_nhit_bin(target_nhit), pred), 0) for pred in source_pred_bins(target_pred)))


def build_candidate_rows(
    *,
    source_counts_csv: Path,
    exact_counts: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[Dict[str, object]]:
    source_counts = source_count_lookup(source_counts_csv)
    rows: List[Dict[str, object]] = []
    for cell_id, (nhit_bin, pred_bin) in enumerate(
        [(nhit, pred) for nhit in NHIT_BINS for pred in PRED_BINS],
        start=1,
    ):
        key = (nhit_bin, pred_bin)
        count = int(exact_counts.get(key, 0)) if exact_counts is not None else estimated_count(nhit_bin, pred_bin, source_counts)
        if target_cache_can_symlink(nhit_bin, pred_bin):
            source_pool = "existing_cache_symlink"
        else:
            source_pool = "filtered_existing_cache"
        rows.append(
            {
                "cell_id": cell_id,
                "nhit_bin": nhit_bin,
                "predE_bin": pred_bin,
                "mc_count": count,
                "candidate_version": VERSION,
                "strategy": "split56",
                "central99_flag": 0,
                "selection_version": VERSION,
                "selection_reason": "pending selector evaluation",
                "raw_ledger_version": VERSION,
                "cell_role": "candidate" if pred_bin != ">=6" else "diagnostic_tail",
                "role_reason": "v6 _64670 split56 candidate grid" if pred_bin != ">=6" else "diagnostic tail bin outside [2,6)",
                "source_pool": source_pool,
                "source_nhit_bin": source_nhit_bin(nhit_bin),
                "source_predE_bins": ";".join(source_pred_bins(pred_bin)),
                "fit_predE_range_flag": int(pred_bin != ">=6"),
                "tail_bin_flag": int(pred_bin == ">=6"),
            }
        )
    return rows


def compute_central_flags(rows: Sequence[Dict[str, object]], central_fraction: float = 0.99) -> Dict[int, bool]:
    flags: Dict[int, bool] = {}
    tail = 0.5 * (1.0 - float(central_fraction))
    by_nhit: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)
    for items in by_nhit.values():
        sorted_items = sorted(items, key=lambda row: interval_key(str(row["predE_bin"])))
        counts = np.asarray([int(row.get("mc_count") or 0) for row in sorted_items], dtype=np.float64)
        total = float(np.sum(counts))
        if total <= 0:
            for row in sorted_items:
                flags[int(row["cell_id"])] = False
            continue
        cumulative_lo = np.concatenate([[0.0], np.cumsum(counts[:-1])]) / total
        cumulative_hi = np.cumsum(counts) / total
        for row, lo, hi in zip(sorted_items, cumulative_lo, cumulative_hi):
            flags[int(row["cell_id"])] = bool((hi > tail) and (lo < 1.0 - tail) and int(row.get("mc_count") or 0) > 0)
    return flags


def compute_ridge_flags(
    rows: Sequence[Dict[str, object]],
    *,
    central_flags: Dict[int, bool],
    min_mc_count: int,
    min_peak_fraction: float,
) -> Dict[int, Tuple[bool, float]]:
    flags: Dict[int, Tuple[bool, float]] = {}
    by_nhit: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if str(row["predE_bin"]) != ">=6":
            by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)
    for items in by_nhit.values():
        counts = np.asarray([int(row.get("mc_count") or 0) for row in items], dtype=np.float64)
        peak = float(np.max(counts)) if counts.size else 0.0
        for row, count_value in zip(items, counts):
            cell_id = int(row["cell_id"])
            count = int(count_value)
            peak_fraction = float(count_value / peak) if peak > 0.0 else 0.0
            on_ridge = (
                bool(central_flags.get(cell_id, False))
                and count >= int(min_mc_count)
                and peak_fraction >= float(min_peak_fraction)
            )
            flags[cell_id] = (bool(on_ridge), peak_fraction)
    for row in rows:
        flags.setdefault(int(row["cell_id"]), (False, 0.0))
    return flags


def selector_rows(
    rows: Sequence[Dict[str, object]],
    *,
    central_flags: Dict[int, bool],
    ridge_flags: Dict[int, Tuple[bool, float]],
    min_mc_count: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        cell_id = int(row["cell_id"])
        pred = str(row["predE_bin"])
        count = int(row.get("mc_count") or 0)
        central = bool(central_flags.get(cell_id, False))
        on_ridge, peak_fraction = ridge_flags.get(cell_id, (False, 0.0))
        include = central and on_ridge and count >= int(min_mc_count) and pred != ">=6"
        if include:
            reason = "central99 MC-occupancy-ridge prefit cell inside [2,6)"
            role = "baseline_fit"
            exclusion = ""
        elif pred == ">=6":
            reason = "diagnostic >=6 tail bin excluded from main fit"
            role = "diagnostic_tail"
            exclusion = "fit_predE_range"
        elif pred in {"[5,5.5)", "[5.5,6)"} and str(row["nhit_bin"]) != TARGET_HIGH_NHIT:
            reason = "split56 high-energy probe excluded except preserved highest-Nhit baseline split"
            role = "high_energy_probe"
            exclusion = "high_energy_probe"
        else:
            reason = "excluded by central99/MC-ridge/statistics prefit selector"
            role = "excluded"
            exclusion = "MC_prefit_selector"
        out.append(
            {
                "cell_id": cell_id,
                "include": int(include),
                "subset_version": "v6_64670_split56_prefit",
                "subset_reason": reason,
                "nhit_bin": row["nhit_bin"],
                "predE_bin": pred,
                "mc_count": count,
                "central99_flag": int(central),
                "physical_ridge_flag": int(on_ridge),
                "ridge_peak_fraction": peak_fraction,
                "fit_predE_range_flag": int(pred != ">=6"),
                "tail_bin_flag": int(pred == ">=6"),
                "psf_quality_flag": 1,
                "cell_role": role,
                "exclusion_source": exclusion,
            }
        )
    return out


def apply_selector_roles(
    rows: Sequence[Dict[str, object]],
    selector: Sequence[Dict[str, object]],
    central_flags: Dict[int, bool],
    *,
    min_mc_count: int,
) -> None:
    by_id = {int(row["cell_id"]): row for row in selector}
    for row in rows:
        cell_id = int(row["cell_id"])
        pred = str(row["predE_bin"])
        selector_row = by_id[cell_id]
        row["central99_flag"] = int(central_flags.get(cell_id, False))
        if truthy(selector_row.get("include")):
            row["cell_role"] = "baseline_fit"
            row["role_reason"] = selector_row["subset_reason"]
        elif pred == ">=6":
            row["cell_role"] = "diagnostic_tail"
            row["role_reason"] = "diagnostic >=6 tail bin outside [2,6); not used in main fit"
        elif pred in {"[5,5.5)", "[5.5,6)"}:
            row["cell_role"] = "high_energy_probe"
            row["role_reason"] = "split56 high-energy diagnostic/probe outside preserved highest-Nhit baseline split"
        elif not central_flags.get(cell_id, False):
            row["cell_role"] = "diagnostic_response_tail"
            row["role_reason"] = "outside MC central-99 reconstructed-energy population"
        elif int(row.get("mc_count") or 0) < int(min_mc_count):
            row["cell_role"] = "diagnostic_low_stat"
            row["role_reason"] = "below prefit MC-count threshold"
        else:
            row["cell_role"] = "systematics_probe"
            row["role_reason"] = "central-99 count-qualified probe outside MC occupancy ridge"
        row["selection_reason"] = row["role_reason"]


def open_tree(path: Path, tree_name: str):
    import uproot

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


def split_file_batch(task: Dict[str, object]) -> List[Dict[str, object]]:
    source = Path(str(task["source"]))
    tree_name = str(task["tree_name"])
    targets = list(task["targets"])
    overwrite = bool(task["overwrite"])
    pending: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    for target_spec in targets:
        target = Path(str(target_spec["target"]))
        target_nhit = str(target_spec["target_nhit"])
        target_pred = str(target_spec["target_pred"])
        if target.exists() and not overwrite:
            root_file, tree = open_tree(target, tree_name)
            try:
                kept = int(tree.num_entries)
            finally:
                root_file.close()
            results.append(
                {
                    "status": "exists",
                    "source": str(source),
                    "target": str(target),
                    "target_nhit": target_nhit,
                    "target_pred": target_pred,
                    "input": kept,
                    "kept": kept,
                }
            )
        else:
            pending.append(target_spec)
    if not pending:
        return results

    root_file, tree = open_tree(source, tree_name)
    try:
        arrays = tree.arrays(list(tree.keys()), library="np")
    finally:
        root_file.close()
    if "nv" not in arrays:
        raise KeyError(f"{source} is missing required branch nv")
    if "ml_logE_pred" not in arrays:
        raise KeyError(f"{source} is missing required branch ml_logE_pred")

    nv = np.asarray(arrays["nv"], dtype=np.float64)
    loge_pred = np.asarray(arrays["ml_logE_pred"], dtype=np.float64)
    finite = np.isfinite(nv) & np.isfinite(loge_pred)
    import uproot

    for target_spec in pending:
        target = Path(str(target_spec["target"]))
        target_nhit = str(target_spec["target_nhit"])
        target_pred = str(target_spec["target_pred"])
        mask = finite.copy()
        mask &= interval_mask(nv, target_nhit)
        mask &= interval_mask(loge_pred, target_pred)
        kept = int(np.count_nonzero(mask))
        if kept <= 0:
            if target.exists() and overwrite:
                target.unlink()
            results.append(
                {
                    "status": "empty",
                    "source": str(source),
                    "target": str(target),
                    "target_nhit": target_nhit,
                    "target_pred": target_pred,
                    "input": int(nv.size),
                    "kept": 0,
                }
            )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: np.asarray(value)[mask] for name, value in arrays.items()}
        payload["nhit_bin"] = np.asarray([target_nhit] * kept, dtype=object)
        payload["predE_bin"] = np.asarray([target_pred] * kept, dtype=object)
        with uproot.recreate(target) as output:
            output[tree_name] = payload
        results.append(
            {
                "status": "written",
                "source": str(source),
                "target": str(target),
                "target_nhit": target_nhit,
                "target_pred": target_pred,
                "input": int(nv.size),
                "kept": kept,
            }
        )
    return results


def symlink_dir(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return "existing_symlink"
        if target.is_dir() and not target.is_symlink():
            return "existing_directory"
        raise FileExistsError(f"Refusing to replace existing target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    return "created_symlink"


def prepare_cache(
    *,
    source_root: Path,
    target_root: Path,
    rows: Sequence[Dict[str, object]],
    tree_name: str,
    workers: int,
    progress_every: int,
    overwrite_filtered: bool,
    max_files_per_source_bin: Optional[int],
) -> Dict[str, object]:
    target_root.mkdir(parents=True, exist_ok=True)
    task_targets_by_source: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    exact_counts: Dict[Tuple[str, str], int] = {(str(row["nhit_bin"]), str(row["predE_bin"])): 0 for row in rows}
    status_counts: Dict[str, int] = {}
    direct_links: List[Dict[str, object]] = []

    for row in rows:
        nhit = str(row["nhit_bin"])
        pred = str(row["predE_bin"])
        dst_dir = target_dir(target_root, nhit, pred)
        source_preds = source_pred_bins(pred)
        if target_cache_can_symlink(nhit, pred):
            src_dir = source_dir(source_root, nhit, source_preds[0])
            if not src_dir.exists():
                status_counts["missing_source_dir"] = status_counts.get("missing_source_dir", 0) + 1
                continue
            status = symlink_dir(src_dir, dst_dir)
            status_counts[status] = status_counts.get(status, 0) + 1
            direct_links.append({"target_nhit": nhit, "target_pred": pred, "source": str(src_dir), "target": str(dst_dir), "status": status})
            exact_counts[(nhit, pred)] = int(row.get("mc_count") or 0)
            continue

        for source_pred in source_preds:
            src_dir = source_dir(source_root, nhit, source_pred)
            if not src_dir.exists():
                status_counts["missing_source_dir"] = status_counts.get("missing_source_dir", 0) + 1
                continue
            files = sorted(src_dir.glob("*.root"))
            if max_files_per_source_bin is not None:
                files = files[: int(max_files_per_source_bin)]
            for source_file in files:
                target_name = source_file.name if len(source_preds) == 1 else f"{sanitize_label(source_pred)}__{source_file.name}"
                key = (str(source_file), tree_name)
                task_targets_by_source.setdefault(key, []).append(
                    {
                        "target": str(dst_dir / target_name),
                        "target_nhit": nhit,
                        "target_pred": pred,
                    }
                )

    tasks: List[Dict[str, object]] = [
        {
            "source": source,
            "tree_name": task_tree_name,
            "targets": targets,
            "overwrite": overwrite_filtered,
        }
        for (source, task_tree_name), targets in sorted(task_targets_by_source.items())
    ]
    filtered_target_count = sum(len(task["targets"]) for task in tasks)
    start = time.perf_counter()
    input_events = 0
    kept_events = 0
    progress_every = max(1, int(progress_every))
    workers = max(1, int(workers))

    def consume(result: Dict[str, object]) -> None:
        nonlocal input_events, kept_events
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        input_events += int(result["input"])
        kept_events += int(result["kept"])
        key = (str(result["target_nhit"]), str(result["target_pred"]))
        exact_counts[key] = int(exact_counts.get(key, 0)) + int(result["kept"])

    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            for result in split_file_batch(task):
                consume(result)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"[v6-split56-cache] {idx}/{len(tasks)} source files | targets={filtered_target_count:,} kept={kept_events:,}", flush=True)
    else:
        print(
            f"[v6-split56-cache] filtering {filtered_target_count} targets from {len(tasks)} source files with {workers} workers",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(split_file_batch, task) for task in tasks]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                for result in future.result():
                    consume(result)
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"[v6-split56-cache] {idx}/{len(tasks)} source files | targets={filtered_target_count:,} kept={kept_events:,}", flush=True)

    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "filtered_tasks": int(filtered_target_count),
        "filtered_source_files": int(len(tasks)),
        "direct_links": direct_links,
        "status_counts": status_counts,
        "filtered_input_events": int(input_events),
        "filtered_kept_events": int(kept_events),
        "exact_counts": {f"{k[0]}__{k[1]}": int(v) for k, v in exact_counts.items()},
        "elapsed_seconds": float(time.perf_counter() - start),
        "_exact_counts": exact_counts,
    }


def validate_rows(rows: Sequence[Dict[str, object]], selector: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if len(rows) != 91:
        raise ValueError(f"split56 ledger expected 91 candidate cells, got {len(rows)}")
    by_nhit: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)
    for nhit, items in by_nhit.items():
        labels = [str(row["predE_bin"]) for row in sorted(items, key=lambda r: interval_key(str(r["predE_bin"])))]
        if labels != PRED_BINS:
            raise ValueError(f"split56 ledger predE labels for {nhit} do not match expected bins")
    tail_included = [row for row in selector if truthy(row.get("include")) and str(row.get("predE_bin")) == ">=6"]
    if tail_included:
        raise ValueError(f"selector includes >=6 tail bins: {tail_included}")
    return {
        "candidate_cells": len(rows),
        "prefit_included_cells": sum(1 for row in selector if truthy(row.get("include"))),
        "predE_bins": PRED_BINS,
        "nhit_bins": NHIT_BINS,
        "tail_selector_included": len(tail_included),
    }


def write_diagnostics_html(path: Path, rows: Sequence[Dict[str, object]], selector: Sequence[Dict[str, object]]) -> None:
    import html

    path.parent.mkdir(parents=True, exist_ok=True)
    included = {int(row["cell_id"]) for row in selector if truthy(row.get("include"))}
    by_key = {(str(row["nhit_bin"]), str(row["predE_bin"])): row for row in rows}
    matrix_rows: List[str] = []
    for nhit in NHIT_BINS:
        cells = [f"<th>{html.escape(nhit)}</th>"]
        for pred in PRED_BINS:
            row = by_key[(nhit, pred)]
            cell_id = int(row["cell_id"])
            classes = ["cell"]
            if cell_id in included:
                classes.append("fit")
            elif pred == ">=6":
                classes.append("tail")
            elif str(row.get("cell_role")) == "high_energy_probe":
                classes.append("probe")
            count = int(row.get("mc_count") or 0)
            cells.append(
                f'<td class="{" ".join(classes)}"><strong>{cell_id}</strong>'
                f'<br><span>{html.escape(pred)}</span><br><code>{count:,}</code></td>'
            )
        matrix_rows.append("<tr>" + "".join(cells) + "</tr>")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v6 _64670 split56 cell selection diagnostics</title>
<style>
body {{ margin:0; background:#f7f8f9; color:#17212b; font-family:Arial,Helvetica,sans-serif; line-height:1.55; }}
main {{ max-width:1480px; margin:0 auto; padding:36px 20px 60px; }}
h1 {{ margin:0 0 12px; font-size:32px; letter-spacing:0; }}
.matrix-wrap {{ overflow-x:auto; border:1px solid #d7dee3; border-radius:8px; background:white; }}
table {{ border-collapse:collapse; min-width:1260px; width:100%; font-size:11px; }}
th,td {{ border:1px solid #d7dee3; padding:7px; vertical-align:top; text-align:center; }}
th {{ background:#eef2f4; position:sticky; left:0; z-index:1; }}
td.cell {{ background:#f4f5f6; }}
td.fit {{ background:#d7f0df; }}
td.probe {{ background:#fff2c6; }}
td.tail {{ background:#e2e8f0; color:#4b5563; }}
code {{ background:#edf1f3; border-radius:4px; padding:1px 4px; }}
</style>
</head>
<body><main>
<h1>Crab SED v6 <code>_64670</code> split56 candidate diagnostics</h1>
<p>Candidate grid has <strong>{len(rows)}</strong> cells: Nhit bins unchanged, old predE <code>[5,6)</code> split into <code>[5,5.5)</code> and <code>[5.5,6)</code>, and <code>&gt;=6</code> retained as a diagnostic tail outside the fit.</p>
<p>Green cells are prefit MC-ridge cells before the drop4-baseline contract is enforced.</p>
<div class="matrix-wrap"><table>
<thead><tr><th>Nhit / predE</th>{''.join(f'<th>{html.escape(pred)}</th>' for pred in PRED_BINS)}</tr></thead>
<tbody>{''.join(matrix_rows)}</tbody>
</table></div>
</main></body></html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_configs(
    *,
    ledger_csv: Path,
    prefit_selector_csv: Path,
    systematics_selector_csv: Path,
    high_energy_selector_csv: Path,
    rows: Sequence[Dict[str, object]],
    selector: Sequence[Dict[str, object]],
) -> None:
    ledger_fields = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "mc_count",
        "candidate_version",
        "strategy",
        "central99_flag",
        "fit_predE_range_flag",
        "tail_bin_flag",
        "selection_version",
        "selection_reason",
        "raw_ledger_version",
        "cell_role",
        "role_reason",
        "source_pool",
        "source_nhit_bin",
        "source_predE_bins",
    ]
    selector_fields = [
        "cell_id",
        "include",
        "subset_version",
        "subset_reason",
        "nhit_bin",
        "predE_bin",
        "mc_count",
        "central99_flag",
        "physical_ridge_flag",
        "ridge_peak_fraction",
        "fit_predE_range_flag",
        "tail_bin_flag",
        "psf_quality_flag",
        "cell_role",
        "exclusion_source",
    ]
    write_csv(ledger_csv, rows, ledger_fields)
    write_csv(prefit_selector_csv, selector, selector_fields)

    systematics = []
    high_energy = []
    for row in selector:
        sys_row = dict(row)
        sys_include = bool(row.get("predE_bin") != ">=6" and truthy(row.get("central99_flag")) and int(row.get("mc_count") or 0) >= 1000)
        sys_row["include"] = int(sys_include)
        sys_row["subset_version"] = "v6_64670_split56_systematics"
        sys_row["cell_role"] = "systematics_probe" if sys_include and not truthy(row.get("include")) else row.get("cell_role", "")
        sys_row["exclusion_source"] = "" if sys_include else row.get("exclusion_source", "")
        systematics.append(sys_row)

        he_row = dict(row)
        he_include = str(row.get("predE_bin")) in {"[5,5.5)", "[5.5,6)", ">=6"} and int(row.get("mc_count") or 0) > 0
        he_row["include"] = int(he_include)
        he_row["subset_version"] = "v6_64670_split56_high_energy_probes"
        he_row["cell_role"] = "high_energy_probe" if he_include and not truthy(row.get("include")) else row.get("cell_role", "")
        he_row["exclusion_source"] = "" if he_include else row.get("exclusion_source", "")
        high_energy.append(he_row)
    write_csv(systematics_selector_csv, systematics, selector_fields)
    write_csv(high_energy_selector_csv, high_energy, selector_fields)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    source_root = args.source_binned_root.resolve()
    target_root = args.target_binned_root.resolve()
    source_counts_csv = args.source_bin_counts_csv.resolve()
    ledger_csv = resolve(args.candidate_ledger_csv)
    prefit_selector_csv = resolve(args.prefit_selector_csv)
    systematics_selector_csv = resolve(args.systematics_selector_csv)
    high_energy_selector_csv = resolve(args.high_energy_selector_csv)
    manifest_json = resolve(args.manifest_json)
    diagnostics_html = resolve(args.diagnostics_html)

    rows = build_candidate_rows(source_counts_csv=source_counts_csv)
    cache_manifest: Optional[Dict[str, object]] = None
    if args.prepare_cache:
        cache_manifest = prepare_cache(
            source_root=source_root,
            target_root=target_root,
            rows=rows,
            tree_name=str(args.tree_name),
            workers=int(args.workers),
            progress_every=int(args.progress_every),
            overwrite_filtered=bool(args.overwrite_filtered_cache),
            max_files_per_source_bin=args.max_files_per_source_bin,
        )
        exact_counts = dict(cache_manifest["_exact_counts"])  # type: ignore[index]
        rows = build_candidate_rows(source_counts_csv=source_counts_csv, exact_counts=exact_counts)
        write_json(target_root / "summary" / "v6_64670_split56_prepare_manifest.json", cache_manifest)
        write_csv(
            target_root / "summary" / "bin_counts.csv",
            [{"nhit_bin": row["nhit_bin"], "predE_bin": row["predE_bin"], "count": int(row["mc_count"])} for row in rows],
            ["nhit_bin", "predE_bin", "count"],
        )

    central_flags = compute_central_flags(rows)
    ridge_flags = compute_ridge_flags(
        rows,
        central_flags=central_flags,
        min_mc_count=int(args.min_baseline_mc_count),
        min_peak_fraction=float(args.ridge_min_peak_fraction),
    )
    selector = selector_rows(
        rows,
        central_flags=central_flags,
        ridge_flags=ridge_flags,
        min_mc_count=int(args.min_baseline_mc_count),
    )
    apply_selector_roles(rows, selector, central_flags, min_mc_count=int(args.min_baseline_mc_count))
    validation = validate_rows(rows, selector)

    if args.write_configs:
        write_configs(
            ledger_csv=ledger_csv,
            prefit_selector_csv=prefit_selector_csv,
            systematics_selector_csv=systematics_selector_csv,
            high_energy_selector_csv=high_energy_selector_csv,
            rows=rows,
            selector=selector,
        )
    if args.write_diagnostics:
        write_diagnostics_html(diagnostics_html, rows, selector)

    manifest = {
        "strategy": "split56",
        "version": VERSION,
        "source_binned_root": str(source_root),
        "target_binned_root": str(target_root),
        "run_dir": str(args.run_dir.resolve()),
        "ledger_csv": str(ledger_csv),
        "prefit_selector_csv": str(prefit_selector_csv),
        "systematics_selector_csv": str(systematics_selector_csv),
        "high_energy_selector_csv": str(high_energy_selector_csv),
        "nhit_bins": NHIT_BINS,
        "predE_bins": PRED_BINS,
        "candidate_cells": len(rows),
        "fit_predE_range": "[2,6)",
        "tail_bins": [">=6"],
        "selector_rules": {
            "central99_flag": 1,
            "min_mc_count": int(args.min_baseline_mc_count),
            "ridge_peak_fraction_min": float(args.ridge_min_peak_fraction),
            "final_drop4_contract": "enforced by 00d_enforce_v6_split56_drop4_selector.py using (nhit_bin,predE_bin), not old cell_id",
        },
        "validation": validation,
        "cache_manifest": cache_manifest,
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    if args.write_configs:
        write_json(manifest_json, manifest)
    print(json.dumps(json_ready(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
