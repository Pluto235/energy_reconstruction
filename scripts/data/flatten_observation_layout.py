#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


LEGACY_LAYOUT_PREFIX = "obs_filtered_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten legacy observation batches such as "
            "/mnt/mydisk/WCDA_observation/obs_filtered_20220101_20220110/0101 "
            "into /mnt/mydisk/WCDA_observation/0101."
        )
    )
    parser.add_argument("--input-root", type=str, default="/mnt/mydisk/WCDA_observation")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files and remove empty legacy directories. Default is dry-run.",
    )
    return parser.parse_args()


def is_day_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) == 4 and path.name.isdigit()


def build_plan(input_root: Path) -> Dict[str, object]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    legacy_dirs = sorted(
        path for path in input_root.iterdir() if path.is_dir() and path.name.startswith(LEGACY_LAYOUT_PREFIX)
    )

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "legacy_dirs": [str(path) for path in legacy_dirs],
        "legacy_dir_count": len(legacy_dirs),
        "day_dir_count": 0,
        "root_file_count": 0,
        "move_count": 0,
        "remove_dir_count": 0,
        "unexpected_entries": [],
        "conflicts": [],
        "moves": [],
        "remove_dirs": [],
    }

    for legacy_dir in legacy_dirs:
        for child in sorted(legacy_dir.iterdir()):
            if not is_day_dir(child):
                summary["unexpected_entries"].append(str(child))
                continue

            target_day_dir = input_root / child.name
            root_files = sorted(path for path in child.iterdir() if path.is_file() and path.suffix.lower() == ".root")
            move_entries: List[Dict[str, str]] = []

            for source_entry in sorted(child.iterdir()):
                target_entry = target_day_dir / source_entry.name
                if target_entry.exists():
                    summary["conflicts"].append(
                        {
                            "source": str(source_entry),
                            "target": str(target_entry),
                        }
                    )
                else:
                    move_entries.append(
                        {
                            "source": str(source_entry),
                            "target": str(target_entry),
                        }
                    )

            summary["day_dir_count"] += 1
            summary["root_file_count"] += len(root_files)
            summary["move_count"] += len(move_entries)
            summary["moves"].append(
                {
                    "source_day_dir": str(child),
                    "target_day_dir": str(target_day_dir),
                    "root_file_count": len(root_files),
                    "entry_count": len(move_entries),
                    "entries": move_entries,
                }
            )

        summary["remove_dirs"].append(str(legacy_dir))
        summary["remove_dir_count"] += 1

    return summary


def apply_plan(summary: Dict[str, object]) -> Dict[str, object]:
    moved_entries = 0
    removed_dirs = 0

    for move_group in summary["moves"]:
        target_day_dir = Path(move_group["target_day_dir"])
        target_day_dir.mkdir(parents=True, exist_ok=True)
        for entry in move_group["entries"]:
            shutil.move(entry["source"], entry["target"])
            moved_entries += 1

        source_day_dir = Path(move_group["source_day_dir"])
        if source_day_dir.exists() and not any(source_day_dir.iterdir()):
            source_day_dir.rmdir()
            removed_dirs += 1

    for legacy_dir_str in summary["remove_dirs"]:
        legacy_dir = Path(legacy_dir_str)
        if legacy_dir.exists() and not any(legacy_dir.iterdir()):
            legacy_dir.rmdir()
            removed_dirs += 1

    summary["applied"] = True
    summary["moved_entries"] = moved_entries
    summary["removed_dirs"] = removed_dirs
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    summary = build_plan(input_root)
    summary["mode"] = "apply" if args.apply else "dry-run"
    summary["applied"] = False

    if summary["unexpected_entries"]:
        raise RuntimeError(
            "Found unexpected entries under legacy observation batches: "
            + ", ".join(summary["unexpected_entries"][:5])
        )
    if summary["conflicts"]:
        first = summary["conflicts"][0]
        raise RuntimeError(
            "Refusing to flatten observation layout because of existing targets. "
            f"First conflict: {first['source']} -> {first['target']}"
        )

    if args.apply:
        summary = apply_plan(summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
