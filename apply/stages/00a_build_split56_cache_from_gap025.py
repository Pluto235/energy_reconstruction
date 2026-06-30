#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time
from typing import Dict, Iterable, List, Tuple

from importlib.machinery import SourceFileLoader


stage00 = SourceFileLoader(
    "stage00_v5_predbin",
    str(Path(__file__).resolve().with_name("00_prepare_v5_predbin_ablation.py")),
).load_module()


NHIT_BINS = stage00.NHIT_BINS
SPLIT56_PRED_BINS = stage00.pred_bins_for_strategy("split56")
GAP025_ROOT = Path("/mnt/mydisk/WCDA_simulation_binned_response_v5_predbin_gap025")
TARGET_ROOT = Path("/mnt/mydisk/WCDA_simulation_binned_response_v4_split56_ridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v4 split56 MC cache from an existing gap025 cache without re-filtering v1 ROOT files."
    )
    parser.add_argument("--source-root", type=Path, default=GAP025_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def sanitize(label: str) -> str:
    return stage00.sanitize_label(label)


def pred_interval(label: str) -> Tuple[float | None, float | None]:
    return stage00.parse_interval(label)


def split56_sources(pred_bin: str) -> List[str]:
    if pred_bin == ">=6":
        return [">=6"]
    low, high = pred_interval(pred_bin)
    if low is None or high is None:
        return [pred_bin]
    out: List[str] = []
    value = float(low)
    while value < float(high) - 1.0e-9:
        nxt = min(float(high), value + 0.25)
        out.append(stage00.interval_label(value, nxt))
        value = nxt
    return out


def read_counts(path: Path) -> Dict[Tuple[str, str], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {(row["nhit_bin"], row["predE_bin"]): int(row.get("count") or 0) for row in csv.DictReader(handle)}


def write_counts(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["nhit_bin", "predE_bin", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def link_file(source: Path, target: Path, *, overwrite: bool) -> str:
    if target.exists() or target.is_symlink():
        if not overwrite:
            return "exists"
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    return "linked"


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    source_counts = read_counts(source_root / "summary" / "bin_counts.csv")
    target_counts: List[dict[str, object]] = []
    manifest_records: List[dict[str, object]] = []
    status_counts: Dict[str, int] = {}
    start = time.perf_counter()

    for nhit in NHIT_BINS:
        nhit_dir = target_root / f"nhit_{sanitize(nhit)}"
        for pred in SPLIT56_PRED_BINS:
            target_dir = nhit_dir / f"predE_{sanitize(pred)}"
            sources = split56_sources(pred)
            count = int(sum(source_counts.get((nhit, source_pred), 0) for source_pred in sources))
            target_counts.append({"nhit_bin": nhit, "predE_bin": pred, "count": count})

            if target_dir.exists() and not args.overwrite:
                status_counts["target_exists"] = status_counts.get("target_exists", 0) + 1
                manifest_records.append(
                    {"nhit_bin": nhit, "predE_bin": pred, "sources": sources, "status": "target_exists", "count": count}
                )
                continue
            if target_dir.exists() or target_dir.is_symlink():
                if target_dir.is_dir() and not target_dir.is_symlink():
                    shutil.rmtree(target_dir)
                else:
                    target_dir.unlink()
            target_dir.mkdir(parents=True, exist_ok=True)

            linked = 0
            missing_dirs: List[str] = []
            for source_pred in sources:
                source_dir = source_root / f"nhit_{sanitize(nhit)}" / f"predE_{sanitize(source_pred)}"
                if not source_dir.exists():
                    missing_dirs.append(str(source_dir))
                    continue
                for source_file in sorted(source_dir.glob("*.root")):
                    target_name = source_file.name if len(sources) == 1 else f"{sanitize(source_pred)}__{source_file.name}"
                    status = link_file(source_file, target_dir / target_name, overwrite=args.overwrite)
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if status in {"linked", "exists"}:
                        linked += 1

            status = "missing_source_dir" if missing_dirs else "linked_files"
            status_counts[status] = status_counts.get(status, 0) + 1
            manifest_records.append(
                {
                    "nhit_bin": nhit,
                    "predE_bin": pred,
                    "sources": sources,
                    "target_dir": str(target_dir),
                    "count": count,
                    "linked_files": linked,
                    "missing_source_dirs": missing_dirs,
                    "status": status,
                }
            )

    write_counts(target_root / "summary" / "bin_counts.csv", target_counts)
    payload = {
        "strategy": "split56",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_strategy": "gap025",
        "predE_bins": SPLIT56_PRED_BINS,
        "nhit_bins": NHIT_BINS,
        "status_counts": status_counts,
        "records": manifest_records,
        "elapsed_seconds": time.perf_counter() - start,
        "note": "Target split56 cache is composed from gap025 bin files by symlink; no event-level re-filtering from v1 is performed.",
    }
    manifest_path = target_root / "summary" / "v4_split56_from_gap025_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {target_root / 'summary' / 'bin_counts.csv'}")
    print(f"Wrote {manifest_path}")
    print(json.dumps(status_counts, sort_keys=True))


if __name__ == "__main__":
    main()
