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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SOURCE_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v1"
DEFAULT_TARGET_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate"
DEFAULT_SOURCE_BIN_COUNTS = "/mnt/mydisk/WCDA_simulation_binned_response_v1/summary/bin_counts.csv"
DEFAULT_CANDIDATE_LEDGER = "apply/config/cell_ledger_v3_candidate.csv"
DEFAULT_BASELINE_SELECTOR = "apply/config/cell_selector_v3_baseline.csv"
DEFAULT_SYSTEMATICS_SELECTOR = "apply/config/cell_selector_v3_systematics.csv"
DEFAULT_HIGH_ENERGY_SELECTOR = "apply/config/cell_selector_v3_high_energy_probes.csv"
DEFAULT_DIAGNOSTICS_HTML = "apply/report/v3_cell_selection_diagnostics.html"
DEFAULT_PSF_SUMMARY_CSV = ""
DEFAULT_BASELINE_PSF_FOLLOWUP_CELL_IDS = "39,52,65"
PSF_QUALITY_MODES = ("off", "annotate", "strict")

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
    "[5,6)",
    ">=6",
]
SOURCE_HIGH_NHIT = ">=2000"
TARGET_HIGH_NHIT = "[2000,3000)"
SOURCE_LOW_NHIT = "[100,200)"
TARGET_LOW_NHIT = "[125,200)"
SOURCE_GE5_PRED = ">=5"
V3_CANDIDATE_VERSION = "v3_candidate"
V3_BASELINE_VERSION = "v3_baseline"
V3_SYSTEMATICS_VERSION = "v3_systematics"
V3_HIGH_ENERGY_VERSION = "v3_high_energy_probes"
DEFAULT_BASELINE_RIDGE_MIN_PEAK_FRACTION = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v3 Crab SED candidate ledger/selectors and optional v3 binned MC cache."
    )
    parser.add_argument("--source-binned-root", type=str, default=DEFAULT_SOURCE_BINNED_ROOT)
    parser.add_argument("--target-binned-root", type=str, default=DEFAULT_TARGET_BINNED_ROOT)
    parser.add_argument("--source-bin-counts-csv", type=str, default=DEFAULT_SOURCE_BIN_COUNTS)
    parser.add_argument("--candidate-ledger-csv", type=str, default=DEFAULT_CANDIDATE_LEDGER)
    parser.add_argument("--baseline-selector-csv", type=str, default=DEFAULT_BASELINE_SELECTOR)
    parser.add_argument("--systematics-selector-csv", type=str, default=DEFAULT_SYSTEMATICS_SELECTOR)
    parser.add_argument("--high-energy-selector-csv", type=str, default=DEFAULT_HIGH_ENERGY_SELECTOR)
    parser.add_argument("--diagnostics-html", type=str, default=DEFAULT_DIAGNOSTICS_HTML)
    parser.add_argument("--psf-summary-csv", type=str, default=DEFAULT_PSF_SUMMARY_CSV)
    parser.add_argument("--require-psf-quality", action="store_true", default=False)
    parser.add_argument(
        "--psf-quality-mode",
        choices=PSF_QUALITY_MODES,
        default="off",
        help=(
            "How baseline selection uses PSF summary quality. 'strict' excludes bad-PSF ridge cells; "
            "'annotate' writes psf_quality_flag but keeps ridge-core cells; 'off' ignores PSF quality."
        ),
    )
    parser.add_argument(
        "--baseline-psf-followup-cell-ids",
        type=str,
        default=DEFAULT_BASELINE_PSF_FOLLOWUP_CELL_IDS,
        help=(
            "Comma-separated ridge cell ids accepted into the baseline despite PSF-quality warnings. "
            "Used only with --psf-quality-mode=annotate."
        ),
    )
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--write-configs", action="store_true", default=False)
    parser.add_argument("--prepare-cache", action="store_true", default=False)
    parser.add_argument(
        "--reuse-candidate-ledger",
        action="store_true",
        default=False,
        help="Reuse --candidate-ledger-csv rows and regenerate only selector flags/configs.",
    )
    parser.add_argument("--overwrite-filtered-cache", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-files-per-source-bin", type=int, default=None)
    parser.add_argument("--min-baseline-mc-count", type=int, default=1000)
    parser.add_argument("--baseline-ridge-min-peak-fraction", type=float, default=DEFAULT_BASELINE_RIDGE_MIN_PEAK_FRACTION)
    parser.add_argument("--write-diagnostics", action="store_true", default=False)
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


def interval_mask(values: np.ndarray, label: str) -> np.ndarray:
    low, high = parse_interval(label)
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values >= float(low)
    if high is not None:
        mask &= values < float(high)
    return mask


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


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int_set(value: str) -> set[int]:
    out: set[int] = set()
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


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


def compute_psf_quality_flags(path: Path, *, require_quality: bool) -> Dict[int, bool]:
    if not require_quality:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"--require-psf-quality was set but PSF summary is missing: {path}")
    flags: Dict[int, bool] = {}
    for row in read_csv(path):
        cell_id = int(row["cell_id"])
        valid_events = int(float(row.get("valid_events") or 0))
        positive_weight_events = int(float(row.get("positive_baseline_weight_events") or 0))
        angle_warning = truthy(row.get("angle_check_warning", ""))
        flags[cell_id] = bool(valid_events > 0 and positive_weight_events > 0 and not angle_warning)
    return flags


def source_nhit_bin(target_nhit: str) -> str:
    if target_nhit == TARGET_LOW_NHIT:
        return SOURCE_LOW_NHIT
    if target_nhit == TARGET_HIGH_NHIT:
        return SOURCE_HIGH_NHIT
    return target_nhit


def source_pred_bin(target_pred: str) -> str:
    if target_pred in {"[2,2.5)", "[2.5,3)"}:
        return "[2,3)"
    if target_pred in {"[5,6)", ">=6"}:
        return SOURCE_GE5_PRED
    return target_pred


def needs_filter(target_nhit: str, target_pred: str) -> bool:
    return (
        target_nhit in {TARGET_LOW_NHIT, TARGET_HIGH_NHIT}
        or target_pred in {"[2,2.5)", "[2.5,3)", "[5,6)", ">=6"}
    )


def source_dir(source_root: Path, target_nhit: str, target_pred: str) -> Path:
    return source_root / f"nhit_{sanitize_label(source_nhit_bin(target_nhit))}" / f"predE_{sanitize_label(source_pred_bin(target_pred))}"


def target_dir(target_root: Path, target_nhit: str, target_pred: str) -> Path:
    return target_root / f"nhit_{sanitize_label(target_nhit)}" / f"predE_{sanitize_label(target_pred)}"


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


def split_file(task: Dict[str, object]) -> Dict[str, object]:
    source = Path(str(task["source"]))
    target = Path(str(task["target"]))
    tree_name = str(task["tree_name"])
    target_nhit = str(task["target_nhit"])
    target_pred = str(task["target_pred"])
    overwrite = bool(task["overwrite"])

    if target.exists() and not overwrite:
        root_file, tree = open_tree(target, tree_name)
        try:
            kept = int(tree.num_entries)
        finally:
            root_file.close()
        return {
            "status": "exists",
            "source": str(source),
            "target": str(target),
            "target_nhit": target_nhit,
            "target_pred": target_pred,
            "input": kept,
            "kept": kept,
        }

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
    mask = np.isfinite(nv) & np.isfinite(loge_pred)
    mask &= interval_mask(nv, target_nhit)
    mask &= interval_mask(loge_pred, target_pred)
    kept = int(np.count_nonzero(mask))
    if kept <= 0:
        if target.exists() and overwrite:
            target.unlink()
        return {
            "status": "empty",
            "source": str(source),
            "target": str(target),
            "target_nhit": target_nhit,
            "target_pred": target_pred,
            "input": int(nv.size),
            "kept": 0,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value)[mask] for name, value in arrays.items()}
    payload["nhit_bin"] = np.asarray([target_nhit] * kept, dtype=object)
    payload["predE_bin"] = np.asarray([target_pred] * kept, dtype=object)
    import uproot

    with uproot.recreate(target) as output:
        output[tree_name] = payload
    return {
        "status": "written",
        "source": str(source),
        "target": str(target),
        "target_nhit": target_nhit,
        "target_pred": target_pred,
        "input": int(nv.size),
        "kept": kept,
    }


def source_count_lookup(source_counts_csv: Path) -> Dict[Tuple[str, str], int]:
    lookup: Dict[Tuple[str, str], int] = {}
    for row in read_csv(source_counts_csv):
        lookup[(row["nhit_bin"], row["predE_bin"])] = int(row.get("count") or 0)
    return lookup


def estimated_count(target_nhit: str, target_pred: str, source_counts: Dict[Tuple[str, str], int]) -> int:
    if needs_filter(target_nhit, target_pred):
        return 0
    return int(source_counts.get((source_nhit_bin(target_nhit), source_pred_bin(target_pred)), 0))


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
        count = (
            int(exact_counts.get((nhit_bin, pred_bin), 0))
            if exact_counts is not None
            else estimated_count(nhit_bin, pred_bin, source_counts)
        )
        source_pool = "filtered_existing_cache" if needs_filter(nhit_bin, pred_bin) else "existing_cache_symlink"
        role = "candidate"
        reason = "v3 HAWC-style candidate grid cell"
        rows.append(
            {
                "cell_id": cell_id,
                "nhit_bin": nhit_bin,
                "predE_bin": pred_bin,
                "mc_count": count,
                "candidate_version": V3_CANDIDATE_VERSION,
                "selection_version": V3_CANDIDATE_VERSION,
                "selection_reason": reason,
                "raw_ledger_version": V3_CANDIDATE_VERSION,
                "cell_role": role,
                "role_reason": reason,
                "source_pool": source_pool,
                "source_nhit_bin": source_nhit_bin(nhit_bin),
                "source_predE_bin": source_pred_bin(pred_bin),
            }
        )
    return rows


def compute_central_flags(rows: Sequence[Dict[str, object]], central_fraction: float) -> Dict[int, bool]:
    flags: Dict[int, bool] = {}
    tail = 0.5 * (1.0 - float(central_fraction))
    by_nhit: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)
    for nhit, items in by_nhit.items():
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


def compute_mc_ridge_flags(
    rows: Sequence[Dict[str, object]],
    *,
    central_flags: Dict[int, bool],
    min_baseline_mc_count: int,
    min_peak_fraction: float,
) -> Dict[int, Tuple[bool, float]]:
    flags: Dict[int, Tuple[bool, float]] = {}
    by_nhit: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)

    for nhit, items in by_nhit.items():
        counts = np.asarray([int(row.get("mc_count") or 0) for row in items], dtype=np.float64)
        peak = float(np.max(counts)) if counts.size else 0.0
        for row, count_value in zip(items, counts):
            cell_id = int(row["cell_id"])
            pred = str(row["predE_bin"])
            count = int(count_value)
            peak_fraction = float(count_value / peak) if peak > 0.0 else 0.0
            count_ok = count >= int(min_baseline_mc_count)
            central = bool(central_flags.get(cell_id, False))
            # The [5,6) bin is deliberately wide and is kept as a baseline
            # high-energy bin only in the highest Nhit row; elsewhere it stays
            # in the high-energy probe selector.
            high_energy_allowed = pred != "[5,6)" or nhit == TARGET_HIGH_NHIT
            on_ridge = central and count_ok and high_energy_allowed and pred != ">=6" and peak_fraction >= float(min_peak_fraction)
            flags[cell_id] = (bool(on_ridge), peak_fraction)
    return flags


def selector_rows(
    rows: Sequence[Dict[str, object]],
    *,
    version: str,
    mode: str,
    min_baseline_mc_count: int,
    central_flags: Dict[int, bool],
    ridge_flags: Dict[int, Tuple[bool, float]],
    psf_quality_flags: Dict[int, bool],
    psf_quality_mode: str,
    psf_followup_cell_ids: set[int],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        cell_id = int(row["cell_id"])
        nhit = str(row["nhit_bin"])
        pred = str(row["predE_bin"])
        count = int(row.get("mc_count") or 0)
        central = bool(central_flags.get(cell_id, False))
        on_ridge, ridge_peak_fraction = ridge_flags.get(cell_id, (False, 0.0))
        psf_quality = bool(psf_quality_flags.get(cell_id, True)) if psf_quality_mode in {"annotate", "strict"} else True
        count_ok = count >= int(min_baseline_mc_count)
        high_probe = pred in {"[5,6)", ">=6"} or nhit == TARGET_HIGH_NHIT
        if mode == "baseline":
            include = central and on_ridge and count_ok and pred != ">=6"
            if psf_quality_mode == "strict":
                include = include and psf_quality
            elif psf_quality_mode == "annotate" and not psf_quality:
                include = include and cell_id in psf_followup_cell_ids
            reason = (
                "central99 MC-occupancy-ridge prefit cell"
                if include and psf_quality
                else "central99 MC-occupancy-ridge prefit cell; PSF requires follow-up"
                if include
                else "excluded by central99/MC-ridge/statistics/high-energy prefit rule"
            )
        elif mode == "systematics":
            include = central and count_ok and pred != ">=6"
            reason = "central99 count-qualified expanded systematics cell" if include else "excluded from expanded systematics selector"
        elif mode == "high_energy":
            include = (central and count_ok and high_probe) or (mode == "high_energy" and pred == ">=6" and count > 0)
            reason = "high-energy probe selector cell" if include else "excluded from high-energy probe selector"
        else:
            raise ValueError(f"Unknown selector mode: {mode}")
        out.append(
            {
                "cell_id": cell_id,
                "include": int(include),
                "subset_version": version,
                "subset_reason": reason,
                "nhit_bin": nhit,
                "predE_bin": pred,
                "mc_count": count,
                "central99_flag": int(central),
                "physical_ridge_flag": int(on_ridge),
                "ridge_peak_fraction": ridge_peak_fraction,
                "psf_quality_flag": int(psf_quality),
                "cell_role": "baseline_fit" if include and mode == "baseline" else ("probe" if include else "excluded"),
                "exclusion_source": "" if include else "MC_prefit_selector",
            }
        )
    return out


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
    tasks: List[Dict[str, object]] = []
    status_counts: Dict[str, int] = {}
    direct_links: List[Dict[str, str]] = []
    exact_counts: Dict[Tuple[str, str], int] = {(str(row["nhit_bin"]), str(row["predE_bin"])): 0 for row in rows}

    for row in rows:
        nhit = str(row["nhit_bin"])
        pred = str(row["predE_bin"])
        src_dir = source_dir(source_root, nhit, pred)
        dst_dir = target_dir(target_root, nhit, pred)
        if not src_dir.exists():
            status_counts["missing_source_dir"] = status_counts.get("missing_source_dir", 0) + 1
            continue
        if not needs_filter(nhit, pred):
            status = symlink_dir(src_dir, dst_dir)
            status_counts[status] = status_counts.get(status, 0) + 1
            direct_links.append({"target_nhit": nhit, "target_pred": pred, "source": str(src_dir), "target": str(dst_dir), "status": status})
            continue
        files = sorted(src_dir.glob("*.root"))
        if max_files_per_source_bin is not None:
            files = files[: int(max_files_per_source_bin)]
        for source_file in files:
            tasks.append(
                {
                    "source": str(source_file),
                    "target": str(dst_dir / source_file.name),
                    "tree_name": tree_name,
                    "target_nhit": nhit,
                    "target_pred": pred,
                    "overwrite": overwrite_filtered,
                }
            )

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
            consume(split_file(task))
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"[v3-cache] {idx}/{len(tasks)} filtered files | kept={kept_events:,}", flush=True)
    else:
        print(f"[v3-cache] filtering {len(tasks)} files with {workers} workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(split_file, task) for task in tasks]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                consume(future.result())
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"[v3-cache] {idx}/{len(tasks)} filtered files | kept={kept_events:,}", flush=True)

    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "filtered_tasks": len(tasks),
        "direct_links": direct_links,
        "status_counts": status_counts,
        "filtered_input_events": int(input_events),
        "filtered_kept_events": int(kept_events),
        "exact_counts_filtered_only": {f"{k[0]}__{k[1]}": int(v) for k, v in exact_counts.items()},
        "elapsed_seconds": time.perf_counter() - start,
        "_exact_counts": exact_counts,
    }


def count_target_cache(rows: Sequence[Dict[str, object]], target_root: Path, tree_name: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        nhit = str(row["nhit_bin"])
        pred = str(row["predE_bin"])
        total = 0
        directory = target_dir(target_root, nhit, pred)
        if directory.exists():
            for path in sorted(directory.glob("*.root")):
                root_file, tree = open_tree(path, tree_name)
                try:
                    total += int(tree.num_entries)
                finally:
                    root_file.close()
        counts[(nhit, pred)] = total
    return counts


def html_escape(value: object) -> str:
    import html

    return html.escape(str(value))


def write_diagnostics_html(
    path: Path,
    rows: Sequence[Dict[str, object]],
    selectors: Dict[str, Sequence[Dict[str, object]]],
    central_flags: Dict[int, bool],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selector_includes = {
        name: {int(row["cell_id"]) for row in selector if int(row.get("include") or 0) == 1}
        for name, selector in selectors.items()
    }
    by_key = {(str(row["nhit_bin"]), str(row["predE_bin"])): row for row in rows}
    matrix_rows: List[str] = []
    for nhit in NHIT_BINS:
        cells = [f"<th>{html_escape(nhit)}</th>"]
        for pred in PRED_BINS:
            row = by_key[(nhit, pred)]
            cell_id = int(row["cell_id"])
            classes = ["cell"]
            if cell_id in selector_includes.get("baseline", set()):
                classes.append("baseline")
            elif cell_id in selector_includes.get("high_energy", set()):
                classes.append("probe")
            elif not central_flags.get(cell_id, False):
                classes.append("excluded")
            count = int(row.get("mc_count") or 0)
            cells.append(
                f'<td class="{" ".join(classes)}">'
                f'<strong>{cell_id}</strong><br><span>{html_escape(pred)}</span><br><code>{count:,}</code></td>'
            )
        matrix_rows.append("<tr>" + "".join(cells) + "</tr>")

    counts_by_selector = {
        name: sum(1 for row in selector if int(row.get("include") or 0) == 1)
        for name, selector in selectors.items()
    }
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v3 Cell Selection Diagnostics</title>
<style>
body {{ margin:0; background:#f7f8f9; color:#182027; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.55; }}
main {{ max-width:1280px; margin:0 auto; padding:36px 20px 60px; }}
h1 {{ margin:0 0 10px; font-size:34px; }}
.lead {{ color:#53606a; max-width:920px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.metric {{ border:1px solid #d7dee3; background:white; border-radius:8px; padding:14px; }}
.label {{ color:#66727c; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
.value {{ font-size:26px; font-weight:750; margin-top:6px; }}
.matrix-wrap {{ overflow-x:auto; border:1px solid #d7dee3; border-radius:8px; background:white; }}
table {{ border-collapse:collapse; min-width:1180px; width:100%; font-size:12px; }}
th,td {{ border:1px solid #d7dee3; padding:8px; vertical-align:top; text-align:center; }}
th {{ background:#eef2f4; position:sticky; left:0; z-index:1; }}
td.cell {{ background:#f4f5f6; }}
td.baseline {{ background:#d7f0df; }}
td.probe {{ background:#fff2c6; }}
td.excluded {{ background:#f0d5d8; color:#6d2028; }}
code {{ background:#edf1f3; border-radius:4px; padding:1px 4px; }}
.legend span {{ display:inline-block; margin-right:14px; }}
.swatch {{ width:12px; height:12px; border:1px solid #aab4bc; vertical-align:-1px; }}
</style>
</head>
<body><main>
<h1>Crab SED v3 Cell Selection Diagnostics</h1>
<p class="lead">Candidate grid starts at Nhit [125,200), uses the v3 mixed predicted-energy bins, and marks the frozen prefit selector without using Crab on-source excess or fit residuals.</p>
<div class="metrics">
  <div class="metric"><div class="label">candidate</div><div class="value">{len(rows)}</div></div>
  <div class="metric"><div class="label">baseline</div><div class="value">{counts_by_selector.get('baseline', 0)}</div></div>
  <div class="metric"><div class="label">systematics</div><div class="value">{counts_by_selector.get('systematics', 0)}</div></div>
  <div class="metric"><div class="label">high-energy probes</div><div class="value">{counts_by_selector.get('high_energy', 0)}</div></div>
</div>
<p class="legend">
  <span><span class="swatch" style="background:#d7f0df"></span> baseline</span>
  <span><span class="swatch" style="background:#fff2c6"></span> high-energy probe</span>
  <span><span class="swatch" style="background:#f0d5d8"></span> outside central-99 / low-stat</span>
</p>
<div class="matrix-wrap"><table>
<thead><tr><th>Nhit / predE</th>{''.join(f'<th>{html_escape(pred)}</th>' for pred in PRED_BINS)}</tr></thead>
<tbody>{''.join(matrix_rows)}</tbody>
</table></div>
</main></body></html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    payload = {key: value for key, value in payload.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    psf_quality_mode = "strict" if bool(args.require_psf_quality) else str(args.psf_quality_mode)
    psf_followup_cell_ids = parse_int_set(str(args.baseline_psf_followup_cell_ids))

    source_root = Path(args.source_binned_root).resolve()
    target_root = Path(args.target_binned_root).resolve()
    source_counts_csv = Path(args.source_bin_counts_csv).resolve()
    if args.reuse_candidate_ledger:
        rows = read_csv(Path(args.candidate_ledger_csv).resolve())
    else:
        rows = build_candidate_rows(source_counts_csv=source_counts_csv)

    cache_manifest: Optional[Dict[str, object]] = None
    exact_counts: Optional[Dict[Tuple[str, str], int]] = None
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
        for row in rows:
            key = (str(row["nhit_bin"]), str(row["predE_bin"]))
            if not needs_filter(key[0], key[1]):
                exact_counts[key] = int(row.get("mc_count") or 0)
        cache_manifest["exact_counts"] = {f"{k[0]}__{k[1]}": int(v) for k, v in exact_counts.items()}
        write_manifest(target_root / "summary" / "v3_prepare_manifest.json", cache_manifest)
        rows = build_candidate_rows(source_counts_csv=source_counts_csv, exact_counts=exact_counts)

    central_flags = compute_central_flags(rows, central_fraction=0.99)
    psf_quality_flags = compute_psf_quality_flags(
        Path(args.psf_summary_csv).resolve(),
        require_quality=psf_quality_mode in {"annotate", "strict"},
    )
    ridge_flags = compute_mc_ridge_flags(
        rows,
        central_flags=central_flags,
        min_baseline_mc_count=int(args.min_baseline_mc_count),
        min_peak_fraction=float(args.baseline_ridge_min_peak_fraction),
    )
    baseline = selector_rows(
        rows,
        version=V3_BASELINE_VERSION,
        mode="baseline",
        min_baseline_mc_count=int(args.min_baseline_mc_count),
        central_flags=central_flags,
        ridge_flags=ridge_flags,
        psf_quality_flags=psf_quality_flags,
        psf_quality_mode=psf_quality_mode,
        psf_followup_cell_ids=psf_followup_cell_ids,
    )
    systematics = selector_rows(
        rows,
        version=V3_SYSTEMATICS_VERSION,
        mode="systematics",
        min_baseline_mc_count=int(args.min_baseline_mc_count),
        central_flags=central_flags,
        ridge_flags=ridge_flags,
        psf_quality_flags=psf_quality_flags,
        psf_quality_mode=psf_quality_mode,
        psf_followup_cell_ids=psf_followup_cell_ids,
    )
    high_energy = selector_rows(
        rows,
        version=V3_HIGH_ENERGY_VERSION,
        mode="high_energy",
        min_baseline_mc_count=int(args.min_baseline_mc_count),
        central_flags=central_flags,
        ridge_flags=ridge_flags,
        psf_quality_flags=psf_quality_flags,
        psf_quality_mode=psf_quality_mode,
        psf_followup_cell_ids=psf_followup_cell_ids,
    )

    for row in rows:
        cell_id = int(row["cell_id"])
        row["central99_flag"] = int(central_flags.get(cell_id, False))
        baseline_row = baseline[cell_id - 1]
        if int(baseline_row["include"]) == 1:
            row["cell_role"] = "baseline_fit"
            row["role_reason"] = baseline_row["subset_reason"]
        elif str(row["predE_bin"]) in {"[5,6)", ">=6"}:
            row["cell_role"] = "diagnostic_high_energy_probe"
            row["role_reason"] = "high-energy low-stat/upper-limit probe; not default baseline"
        elif not central_flags.get(cell_id, False):
            row["cell_role"] = "diagnostic_response_tail"
            row["role_reason"] = "outside MC central-99 reconstructed-energy population"
        elif int(row.get("mc_count") or 0) < int(args.min_baseline_mc_count):
            row["cell_role"] = "diagnostic_low_stat"
            row["role_reason"] = "below prefit MC-count threshold"
        else:
            row["cell_role"] = "systematics_probe"
            row["role_reason"] = "central-99 count-qualified probe outside MC occupancy ridge"
        row["selection_reason"] = row["role_reason"]

    if args.write_configs:
        ledger_fields = [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "mc_count",
            "candidate_version",
            "central99_flag",
            "selection_version",
            "selection_reason",
            "raw_ledger_version",
            "cell_role",
            "role_reason",
            "source_pool",
            "source_nhit_bin",
            "source_predE_bin",
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
            "psf_quality_flag",
            "cell_role",
            "exclusion_source",
        ]
        write_csv(Path(args.candidate_ledger_csv).resolve(), rows, ledger_fields)
        write_csv(Path(args.baseline_selector_csv).resolve(), baseline, selector_fields)
        write_csv(Path(args.systematics_selector_csv).resolve(), systematics, selector_fields)
        write_csv(Path(args.high_energy_selector_csv).resolve(), high_energy, selector_fields)

    if args.write_diagnostics:
        write_diagnostics_html(
            Path(args.diagnostics_html).resolve(),
            rows,
            {"baseline": baseline, "systematics": systematics, "high_energy": high_energy},
            central_flags,
        )

    summary = {
        "candidate_cells": len(rows),
        "baseline_cells": sum(1 for row in baseline if int(row["include"]) == 1),
        "systematics_cells": sum(1 for row in systematics if int(row["include"]) == 1),
        "high_energy_probe_cells": sum(1 for row in high_energy if int(row["include"]) == 1),
        "psf_quality_mode": psf_quality_mode,
        "baseline_psf_followup_cell_ids": sorted(psf_followup_cell_ids),
        "source_binned_root": str(source_root),
        "target_binned_root": str(target_root),
        "cache_prepared": bool(args.prepare_cache),
        "cache_manifest": None if cache_manifest is None else {key: value for key, value in cache_manifest.items() if not key.startswith("_")},
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
