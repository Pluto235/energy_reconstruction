#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_SELECTOR = "apply/config/cell_selector_v3_nhit_only.csv"
DEFAULT_SOURCE_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate"
DEFAULT_TARGET_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v3_nhit_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PredE-blind Nhit-only MC cache by linking all predE child "
            "ROOT files under each selected Nhit bin into predE_all."
        )
    )
    parser.add_argument("--selector-csv", type=str, default=DEFAULT_SELECTOR)
    parser.add_argument("--source-root", type=str, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=str, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--metadata-name", type=str, default="v3_nhit_only_cache_metadata.json")
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


def read_selector(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    selected = [row for row in rows if str(row.get("include", "1")).strip() not in {"0", "false", "False"}]
    if not selected:
        raise ValueError(f"No included rows found in {path}")
    for row in selected:
        if str(row.get("predE_bin", "")).strip().lower() not in {"all", "*"}:
            raise ValueError(f"Nhit-only selector row is not predE-blind: {row}")
    return selected


def clear_target(path: Path, *, overwrite: bool, dry_run: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"Target directory already exists; pass --overwrite to replace it: {path}")
    if dry_run:
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_file(source: Path, target: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def build_cache(args: argparse.Namespace) -> Dict[str, object]:
    selector_csv = Path(args.selector_csv).resolve()
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    rows = read_selector(selector_csv)

    if not source_root.exists():
        raise FileNotFoundError(f"Source binned root does not exist: {source_root}")
    clear_target(target_root, overwrite=bool(args.overwrite), dry_run=bool(args.dry_run))
    if not args.dry_run:
        target_root.mkdir(parents=True, exist_ok=True)

    cell_summaries: List[Dict[str, object]] = []
    total_links = 0
    for row in rows:
        nhit_bin = str(row["nhit_bin"])
        source_nhit_dir = source_root / f"nhit_{sanitize_label(nhit_bin)}"
        target_cell_dir = target_root / f"nhit_{sanitize_label(nhit_bin)}" / "predE_all"
        if not source_nhit_dir.exists():
            raise FileNotFoundError(f"Missing source Nhit directory for {nhit_bin}: {source_nhit_dir}")
        if not args.dry_run:
            target_cell_dir.mkdir(parents=True, exist_ok=True)

        pred_dirs = sorted(path for path in source_nhit_dir.iterdir() if path.is_dir() and path.name.startswith("predE_"))
        if not pred_dirs:
            raise FileNotFoundError(f"No predE_* child directories under {source_nhit_dir}")

        links_for_cell = 0
        source_files_for_cell = 0
        for pred_dir in pred_dirs:
            root_files = sorted(path for path in pred_dir.iterdir() if path.is_file() and path.suffix == ".root")
            source_files_for_cell += len(root_files)
            for source_file in root_files:
                target_name = f"{pred_dir.name}__{source_file.name}"
                target_path = target_cell_dir / target_name
                if target_path.exists() or target_path.is_symlink():
                    raise FileExistsError(f"Target link collision: {target_path}")
                link_file(source_file, target_path, dry_run=bool(args.dry_run))
                links_for_cell += 1

        total_links += links_for_cell
        cell_summaries.append(
            {
                "cell_id": int(row["cell_id"]),
                "nhit_bin": nhit_bin,
                "predE_bin": "all",
                "source_nhit_dir": str(source_nhit_dir),
                "target_cell_dir": str(target_cell_dir),
                "source_predE_dirs": len(pred_dirs),
                "source_root_files": int(source_files_for_cell),
                "linked_root_files": int(links_for_cell),
            }
        )

    metadata = {
        "description": "PredE-blind Nhit-only symlink cache for Stage A/B control runs.",
        "created_at_unix": int(time.time()),
        "selector_csv": str(selector_csv),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "dry_run": bool(args.dry_run),
        "cells": cell_summaries,
        "total_linked_root_files": int(total_links),
        "naming": "target links use predE_<bin>__<source basename> to avoid cross-bin basename collisions.",
    }
    if not args.dry_run:
        metadata_path = target_root / str(args.metadata_name)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        metadata["metadata_path"] = str(metadata_path)
    return metadata


def main() -> None:
    metadata = build_cache(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
