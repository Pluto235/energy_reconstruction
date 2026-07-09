#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate observation eval ROOT files against recovered-time friend ROOT files for one MMDD prefix."
    )
    parser.add_argument("--obs-root", required=True, help="Observation eval root, e.g. /mnt/mydisk/WCDA_observation_eval_64748")
    parser.add_argument(
        "--time-root",
        required=True,
        help="Recovered-time root, e.g. /mnt/mydisk/WCDA_observation_eval_64748/recovered_time",
    )
    parser.add_argument("--day-prefix", required=True, help="MMDD prefix such as 03 or 0315")
    parser.add_argument("--obs-tree-name", default="t_eventout")
    parser.add_argument("--time-tree-name", default="t_recovered_time")
    parser.add_argument("--summary-json", default="", help="Optional path to write validation summary JSON.")
    parser.add_argument("--print-missing", type=int, default=20)
    return parser.parse_args()


def iter_obs_files(obs_root: Path, day_prefix: str) -> List[Path]:
    day_dirs = sorted(path for path in obs_root.iterdir() if path.is_dir() and path.name.startswith(day_prefix))
    return [path for day_dir in day_dirs for path in sorted(day_dir.glob("Esg*.root"))]


def iter_time_files(time_root: Path, day_prefix: str) -> List[Path]:
    day_dirs = sorted(path for path in time_root.iterdir() if path.is_dir() and path.name.startswith(day_prefix))
    return [path for day_dir in day_dirs for path in sorted(day_dir.glob("Esg*.time.root"))]


def expected_obs_rel_from_time(time_path: Path, time_root: Path) -> Path:
    rel = time_path.relative_to(time_root)
    return rel.with_name(rel.name.replace(".time.root", ".root"))


def tree_entries(path: Path, tree_name: str) -> int:
    with uproot.open(path) as root_file:
        if tree_name in root_file:
            return int(root_file[tree_name].num_entries)
        versioned = f"{tree_name};1"
        if versioned in root_file:
            return int(root_file[versioned].num_entries)
    raise KeyError(f"{path} does not contain tree {tree_name!r}")


def load_time_sidecar(path: Path) -> Optional[Dict[str, object]]:
    sidecar = path.with_suffix(path.suffix + ".summary.json")
    if not sidecar.exists():
        return None
    with sidecar.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nonzero_summary_value(summary: Dict[str, object], keys: Iterable[str]) -> int:
    total = 0
    for key in keys:
        value = summary.get(key, 0)
        if isinstance(value, bool):
            total += int(value)
        elif isinstance(value, (int, float)):
            total += int(value)
    return total


def main() -> None:
    args = parse_args()
    obs_root = Path(args.obs_root)
    time_root = Path(args.time_root)
    obs_files = iter_obs_files(obs_root, args.day_prefix)
    time_files = iter_time_files(time_root, args.day_prefix)
    obs_rel_paths = {path.relative_to(obs_root) for path in obs_files}
    expected_obs_rel_paths = {expected_obs_rel_from_time(path, time_root) for path in time_files}

    missing_obs = sorted(str(path) for path in expected_obs_rel_paths - obs_rel_paths)
    missing_time: List[str] = []
    unreadable: List[str] = []
    entry_mismatches: List[Dict[str, object]] = []
    sidecar_missing: List[str] = []
    sidecar_bad_files: List[str] = []
    sidecar_bad_total = 0
    total_obs_entries = 0
    total_time_entries = 0

    for obs_path in obs_files:
        rel = obs_path.relative_to(obs_root)
        time_path = time_root / rel.parent / f"{obs_path.stem}.time.root"
        if not time_path.exists():
            missing_time.append(str(rel))
            continue
        try:
            obs_entries = tree_entries(obs_path, args.obs_tree_name)
            time_entries = tree_entries(time_path, args.time_tree_name)
        except Exception as exc:  # noqa: BLE001 - validation should report all file-level failures.
            unreadable.append(f"{rel}: {exc}")
            continue

        total_obs_entries += obs_entries
        total_time_entries += time_entries
        if obs_entries != time_entries:
            entry_mismatches.append(
                {
                    "file": str(rel),
                    "obs_entries": obs_entries,
                    "time_entries": time_entries,
                }
            )

        sidecar = load_time_sidecar(time_path)
        if sidecar is None:
            sidecar_missing.append(str(time_path.relative_to(time_root)))
            continue
        bad_value = nonzero_summary_value(
            sidecar,
            (
                "missing_reduced",
                "missing_leaf",
                "entry_out_of_range",
                "event_mismatch",
                "bad_eval_entry",
            ),
        )
        sidecar_bad_total += bad_value
        if bad_value:
            sidecar_bad_files.append(str(time_path.relative_to(time_root)))

    summary = {
        "obs_root": str(obs_root),
        "time_root": str(time_root),
        "day_prefix": args.day_prefix,
        "obs_files": len(obs_files),
        "time_files": len(time_files),
        "obs_files_missing": len(missing_obs),
        "time_files_missing": len(missing_time),
        "unreadable_files": len(unreadable),
        "entry_mismatch_files": len(entry_mismatches),
        "obs_entries": total_obs_entries,
        "time_entries": total_time_entries,
        "time_sidecar_missing": len(sidecar_missing),
        "time_sidecar_bad_files": len(sidecar_bad_files),
        "time_sidecar_bad_total": sidecar_bad_total,
        "missing_obs_examples": missing_obs[: args.print_missing],
        "missing_time_examples": missing_time[: args.print_missing],
        "unreadable_examples": unreadable[: args.print_missing],
        "entry_mismatch_examples": entry_mismatches[: args.print_missing],
        "time_sidecar_missing_examples": sidecar_missing[: args.print_missing],
        "time_sidecar_bad_examples": sidecar_bad_files[: args.print_missing],
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if missing_obs or missing_time or unreadable or entry_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
