#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apply.simulation_all_bin import sanitize_label


DEFAULT_BIN_COUNTS = "apply/summary_selectedcuts/bin_counts.csv"
DEFAULT_V1_SELECTION = "apply/config/cell_selection_v1.csv"
DEFAULT_V2_SELECTION = "apply/config/cell_selection_v2.csv"
DEFAULT_V2P1_SELECTION = "apply/config/cell_selection_v2p1.csv"
DEFAULT_RAW_LEDGER = "apply/config/cell_ledger_v2_raw65.csv"
DEFAULT_BASELINE_SELECTOR = "apply/config/cell_selector_v2_baseline26.csv"
DEFAULT_PROBE_SELECTOR = "apply/config/cell_selector_v2_transition_probes.csv"
DEFAULT_SOURCE_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v1"
DEFAULT_TARGET_BINNED_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v2_raw65"

V2_HIGH_NHIT_BIN = "[2000,3000)"
OLD_HIGH_NHIT_BIN = ">=2000"
RAW_LEDGER_VERSION = "v2_raw65"
BASELINE_VERSION = "v2_baseline26"
PROBE_VERSION = "v2_transition_probes"
BASELINE_100_200_PREDS = {"[2,3)", "[3,3.25)", "[3.25,3.5)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare v2 Crab SED inputs: raw65 ledger, baseline/probe selectors, "
            "and optional non-overlapping [2000,3000) MC cache derived from the old >=2000 cache."
        )
    )
    parser.add_argument("--bin-counts-csv", type=str, default=DEFAULT_BIN_COUNTS)
    parser.add_argument("--v1-selection-csv", type=str, default=DEFAULT_V1_SELECTION)
    parser.add_argument("--v2-selection-csv", type=str, default=DEFAULT_V2_SELECTION)
    parser.add_argument("--v2p1-selection-csv", type=str, default=DEFAULT_V2P1_SELECTION)
    parser.add_argument("--raw-ledger-csv", type=str, default=DEFAULT_RAW_LEDGER)
    parser.add_argument("--baseline-selector-csv", type=str, default=DEFAULT_BASELINE_SELECTOR)
    parser.add_argument("--probe-selector-csv", type=str, default=DEFAULT_PROBE_SELECTOR)
    parser.add_argument("--source-binned-root", type=str, default=DEFAULT_SOURCE_BINNED_ROOT)
    parser.add_argument("--target-binned-root", type=str, default=DEFAULT_TARGET_BINNED_ROOT)
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--write-configs", action="store_true", default=False)
    parser.add_argument("--prepare-cache", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--overwrite-high-cache", action="store_true", default=False)
    parser.add_argument("--max-high-files", type=int, default=None)
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def key_for(row: Dict[str, str]) -> Tuple[str, str]:
    return str(row["nhit_bin"]), str(row["predE_bin"])


def selection_map(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {key_for(row): row for row in rows}


def build_raw_ledger(
    *,
    bin_counts: Sequence[Dict[str, str]],
    v1_rows: Sequence[Dict[str, str]],
    v2_rows: Sequence[Dict[str, str]],
    v2p1_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    v1_by_key = selection_map(v1_rows)
    v2_by_key = selection_map(v2_rows)
    v2p1_by_key = selection_map(v2p1_rows)
    v2p1_keys = set(v2p1_by_key)
    baseline_keys = set(v2p1_keys)
    baseline_keys.update(("[100,200)", pred) for pred in BASELINE_100_200_PREDS)

    raw_rows: List[Dict[str, str]] = []
    for row in bin_counts:
        if row.get("statistics_level") != "acceptable":
            continue
        nhit = row["nhit_bin"]
        if nhit == OLD_HIGH_NHIT_BIN:
            converted = dict(row)
            converted["nhit_bin"] = V2_HIGH_NHIT_BIN
            converted["formal_nhit_bin"] = "True"
            raw_rows.append(converted)
        else:
            raw_rows.append(dict(row))

    raw_rows = sorted(raw_rows, key=lambda row: (interval_key(row["nhit_bin"]), interval_key(row["predE_bin"])))
    if len(raw_rows) != 65:
        raise ValueError(f"Expected 65 raw cells, got {len(raw_rows)}")

    ledger: List[Dict[str, object]] = []
    for idx, row in enumerate(raw_rows, start=1):
        nhit = row["nhit_bin"]
        pred = row["predE_bin"]
        key = (nhit, pred)
        source_pool = "v2_high_nhit_split" if nhit == V2_HIGH_NHIT_BIN else "old_acceptable_pool"
        if key in baseline_keys:
            role = "baseline_fit"
            role_reason = "prefit frozen baseline26 selector cell"
        elif nhit in {"[30,60)", "[60,100)"}:
            role = "diagnostic_legacy_low_nhit"
            role_reason = "legacy low-Nhit acceptable cell kept only for diagnostics"
        elif nhit == "[100,200)" and pred == "[3.5,3.75)":
            role = "transition_probe"
            role_reason = "[100,200) transition cell excluded from baseline; reserved as probe"
        elif nhit == V2_HIGH_NHIT_BIN:
            role = "diagnostic_low_stat_high_energy"
            role_reason = "high-Nhit boundary cell kept in raw ledger but excluded from baseline"
        else:
            role = "diagnostic_response_tail"
            role_reason = "acceptable-pool response/background tail kept for diagnostics"

        v1 = v1_by_key.get(key)
        v2 = v2_by_key.get(key)
        v2p1 = v2p1_by_key.get(key)
        ledger.append(
            {
                "cell_id": idx,
                "nhit_bin": nhit,
                "predE_bin": pred,
                "mc_count": int(row.get("count") or row.get("mc_count") or 0),
                "formal_nhit_bin": row.get("formal_nhit_bin", ""),
                "statistics_level": row.get("statistics_level", ""),
                "selection_version": RAW_LEDGER_VERSION,
                "selection_reason": role_reason,
                "raw_ledger_version": RAW_LEDGER_VERSION,
                "cell_role": role,
                "role_reason": role_reason,
                "source_pool": source_pool,
                "source_cell_id_v1": v1.get("cell_id", "") if v1 else "",
                "source_cell_id_v2": v2.get("cell_id", "") if v2 else "",
                "source_cell_id_v2p1": v2p1.get("cell_id", "") if v2p1 else "",
                "crab_roi_events_v2": v2p1.get("crab_roi_events_v2", "") if v2p1 else "",
            }
        )
    return ledger


def build_selector_rows(ledger: Sequence[Dict[str, object]], *, mode: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in ledger:
        role = str(row["cell_role"])
        if mode == "baseline":
            include = role == "baseline_fit"
            version = BASELINE_VERSION
            reason = "included in frozen v2 baseline26" if include else f"excluded from baseline26: {role}"
        elif mode == "probe":
            include = role in {"baseline_fit", "transition_probe", "diagnostic_low_stat_high_energy"}
            version = PROBE_VERSION
            reason = "included in baseline+transition/high-energy probe" if include else f"excluded from probe: {role}"
        else:
            raise ValueError(f"Unknown selector mode: {mode}")
        rows.append(
            {
                "cell_id": row["cell_id"],
                "include": int(include),
                "subset_version": version,
                "subset_reason": reason,
                "nhit_bin": row["nhit_bin"],
                "predE_bin": row["predE_bin"],
                "cell_role": role,
                "raw_ledger_version": row["raw_ledger_version"],
            }
        )
    included = sum(1 for row in rows if int(row["include"]) == 1)
    if mode == "baseline" and included != 26:
        raise ValueError(f"Expected 26 baseline cells, got {included}")
    return rows


def ensure_low_nhit_symlinks(ledger: Sequence[Dict[str, object]], source_root: Path, target_root: Path) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    labels = sorted(
        {str(row["nhit_bin"]) for row in ledger if str(row["nhit_bin"]) != V2_HIGH_NHIT_BIN},
        key=interval_key,
    )
    target_root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        source = source_root / f"nhit_{sanitize_label(label)}"
        target = target_root / f"nhit_{sanitize_label(label)}"
        if not source.exists():
            raise FileNotFoundError(f"Missing source binned directory: {source}")
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                status = "existing_symlink"
            else:
                raise FileExistsError(f"Refusing to replace existing target directory: {target}")
        else:
            target.symlink_to(source)
            status = "created_symlink"
        summaries.append({"nhit_bin": label, "source": str(source), "target": str(target), "status": status})
    return summaries


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


def split_high_file(kwargs: Dict[str, object]) -> Dict[str, object]:
    source = Path(str(kwargs["source"]))
    target = Path(str(kwargs["target"]))
    tree_name = str(kwargs["tree_name"])
    overwrite = bool(kwargs["overwrite"])

    if target.exists() and not overwrite:
        return {"source": str(source), "target": str(target), "status": "exists", "input": 0, "kept": 0}
    root_file, tree = open_tree(source, tree_name)
    try:
        arrays = tree.arrays(list(tree.keys()), library="np")
    finally:
        root_file.close()

    if "nv" not in arrays:
        raise KeyError(f"{source} is missing required branch nv")
    nv = np.asarray(arrays["nv"])
    mask = np.isfinite(nv) & (nv >= 2000) & (nv < 3000)
    kept = int(np.count_nonzero(mask))
    if kept <= 0:
        if target.exists() and overwrite:
            target.unlink()
        return {"source": str(source), "target": str(target), "status": "empty", "input": int(nv.size), "kept": 0}

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        payload[name] = arr[mask]
    payload["nhit_bin"] = np.asarray([V2_HIGH_NHIT_BIN] * kept, dtype=object)
    with uproot.recreate(target) as output:
        output[tree_name] = payload
    return {"source": str(source), "target": str(target), "status": "written", "input": int(nv.size), "kept": kept}


def high_pred_bins(ledger: Sequence[Dict[str, object]]) -> List[str]:
    return sorted(
        {str(row["predE_bin"]) for row in ledger if str(row["nhit_bin"]) == V2_HIGH_NHIT_BIN},
        key=interval_key,
    )


def prepare_high_cache(
    *,
    ledger: Sequence[Dict[str, object]],
    source_root: Path,
    target_root: Path,
    tree_name: str,
    workers: int,
    progress_every: int,
    overwrite: bool,
    max_files: Optional[int],
) -> Dict[str, object]:
    source_high = source_root / f"nhit_{sanitize_label(OLD_HIGH_NHIT_BIN)}"
    target_high = target_root / f"nhit_{sanitize_label(V2_HIGH_NHIT_BIN)}"
    if not source_high.exists():
        raise FileNotFoundError(f"Missing source high-Nhit cache: {source_high}")
    if target_high.exists() and overwrite:
        if target_high.is_symlink() or target_high.is_file():
            target_high.unlink()
        else:
            shutil.rmtree(target_high)

    tasks: List[Dict[str, object]] = []
    for pred in high_pred_bins(ledger):
        source_pred = source_high / f"predE_{sanitize_label(pred)}"
        if not source_pred.exists():
            raise FileNotFoundError(f"Missing source high-Nhit predE directory: {source_pred}")
        files = sorted(source_pred.glob("*.root"))
        if max_files is not None:
            files = files[:max_files]
        target_pred = target_high / f"predE_{sanitize_label(pred)}"
        for source in files:
            tasks.append(
                {
                    "source": str(source),
                    "target": str(target_pred / source.name),
                    "tree_name": tree_name,
                    "overwrite": overwrite,
                }
            )

    start = time.perf_counter()
    status_counts: Dict[str, int] = {}
    kept_by_pred = {pred: 0 for pred in high_pred_bins(ledger)}
    input_events = 0
    kept_events = 0
    workers = max(1, int(workers))
    progress_every = max(1, int(progress_every))

    def consume(result: Dict[str, object]) -> None:
        nonlocal input_events, kept_events
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        input_events += int(result["input"])
        kept_events += int(result["kept"])
        target = Path(str(result["target"]))
        pred_label = target.parent.name.removeprefix("predE_")
        for pred in kept_by_pred:
            if pred_label == sanitize_label(pred):
                kept_by_pred[pred] += int(result["kept"])
                break

    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            consume(split_high_file(task))
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"[split-high] {idx}/{len(tasks)} files | kept={kept_events}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(split_high_file, task) for task in tasks]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                consume(future.result())
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"[split-high] {idx}/{len(tasks)} files | kept={kept_events}", flush=True)

    return {
        "source_high_dir": str(source_high),
        "target_high_dir": str(target_high),
        "tasks": len(tasks),
        "status_counts": status_counts,
        "input_events": int(input_events),
        "kept_events": int(kept_events),
        "kept_by_predE_bin": kept_by_pred,
        "elapsed_seconds": time.perf_counter() - start,
    }


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    if not args.write_configs and not args.prepare_cache:
        raise ValueError("Nothing to do: pass --write-configs and/or --prepare-cache")

    bin_counts = read_csv(Path(args.bin_counts_csv))
    v1_rows = read_csv(Path(args.v1_selection_csv))
    v2_rows = read_csv(Path(args.v2_selection_csv))
    v2p1_rows = read_csv(Path(args.v2p1_selection_csv))
    ledger = build_raw_ledger(bin_counts=bin_counts, v1_rows=v1_rows, v2_rows=v2_rows, v2p1_rows=v2p1_rows)
    baseline = build_selector_rows(ledger, mode="baseline")
    probes = build_selector_rows(ledger, mode="probe")

    if args.write_configs:
        ledger_fields = [
            "cell_id",
            "nhit_bin",
            "predE_bin",
            "mc_count",
            "formal_nhit_bin",
            "statistics_level",
            "selection_version",
            "selection_reason",
            "raw_ledger_version",
            "cell_role",
            "role_reason",
            "source_pool",
            "source_cell_id_v1",
            "source_cell_id_v2",
            "source_cell_id_v2p1",
            "crab_roi_events_v2",
        ]
        selector_fields = [
            "cell_id",
            "include",
            "subset_version",
            "subset_reason",
            "nhit_bin",
            "predE_bin",
            "cell_role",
            "raw_ledger_version",
        ]
        write_csv(Path(args.raw_ledger_csv), ledger, ledger_fields)
        write_csv(Path(args.baseline_selector_csv), baseline, selector_fields)
        write_csv(Path(args.probe_selector_csv), probes, selector_fields)
        print(f"Wrote {args.raw_ledger_csv}", flush=True)
        print(f"Wrote {args.baseline_selector_csv}", flush=True)
        print(f"Wrote {args.probe_selector_csv}", flush=True)

    if args.prepare_cache:
        source_root = Path(args.source_binned_root)
        target_root = Path(args.target_binned_root)
        symlinks = ensure_low_nhit_symlinks(ledger, source_root, target_root)
        high_summary = prepare_high_cache(
            ledger=ledger,
            source_root=source_root,
            target_root=target_root,
            tree_name=str(args.tree_name),
            workers=int(args.workers),
            progress_every=int(args.progress_every),
            overwrite=bool(args.overwrite_high_cache),
            max_files=args.max_high_files,
        )
        manifest = {
            "description": "v2 raw65 binned MC cache built from v1 cache plus filtered [2000,3000) high-Nhit split.",
            "raw_ledger_csv": str(Path(args.raw_ledger_csv).resolve()),
            "source_binned_root": str(source_root.resolve()),
            "target_binned_root": str(target_root.resolve()),
            "low_nhit_symlinks": symlinks,
            "high_nhit_split": high_summary,
            "created_at_unix": time.time(),
        }
        manifest_path = target_root / "summary" / "v2_raw65_cache_manifest.json"
        write_manifest(manifest_path, manifest)
        print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
