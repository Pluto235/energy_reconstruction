#!/usr/bin/env python3
"""Build a common-GTI subset of a Stage C obs_events dataset.

Stage D (04_background.py) and Stage E (05_signal.py) take no event-level
filtering flag; they read parquet events and `source_files.csv` straight out
of `--stage-c-dir`. To rerun them on the common-GTI intersection (v6 sorted
GTI ∩ Pass5-accepted four-hour chunks) without touching those scripts, this
tool builds a second, self-contained "stage_c_dir" clone that only contains
the accepted hours:

    <output-dir>/obs_events/...          (symlinks to the accepted parquet parts)
    <output-dir>/source_files.csv        (filtered rows, same schema as Stage C)
    <output-dir>/obs_events_metadata.json (copy of the original, live-time patched)

Acceptance is decided per hour (Pass5's four-hour DI chunks partition the
observation into whole hours), so filtering the original Stage C
`source_files.csv` -- which is a 1:1 mapping of source_file_id to one hourly
ROOT/parquet file -- to accepted hours is exact; no intra-file/row-level MJD
splitting is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-c-dir",
        type=Path,
        required=True,
        help="Original Stage C run dir containing obs_events/, source_files.csv, obs_events_metadata.json",
    )
    parser.add_argument(
        "--common-gti-tsv",
        type=Path,
        required=True,
        help="apply/tools/pass5_covariance/common_gti/common_gti.tsv (has an 'hour' column, YYYYMMDD_HH)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


def hour_key(relative_path: str) -> str:
    # "0101/Esg20220101_00.root" -> "20220101_00"
    name = Path(relative_path).stem
    return name[len("Esg"):] if name.startswith("Esg") else name


def main() -> None:
    args = parse_args()
    stage_c_dir = args.stage_c_dir.resolve()
    obs_events_src = stage_c_dir / "obs_events"
    source_files_src = stage_c_dir / "source_files.csv"
    metadata_src = stage_c_dir / "obs_events_metadata.json"
    for path in (obs_events_src, source_files_src):
        if not path.exists():
            raise FileNotFoundError(path)

    with args.common_gti_tsv.open(newline="", encoding="utf-8") as handle:
        accepted_hours = {row["hour"] for row in csv.DictReader(handle, delimiter="\t")}
    if not accepted_hours:
        raise ValueError(f"No accepted hours found in {args.common_gti_tsv}")

    output_dir = args.output_dir.resolve()
    obs_events_dst = output_dir / "obs_events"
    obs_events_dst.mkdir(parents=True, exist_ok=True)

    with source_files_src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept_rows = [
        row
        for row in rows
        if row.get("status") == "processed" and hour_key(row["relative_path"]) in accepted_hours
    ]
    if not kept_rows:
        raise ValueError("No source_files.csv rows matched the accepted-hour set; check inputs.")

    linked = 0
    for row in kept_rows:
        relative_parquet = Path(row["parquet_relative_path"])
        src = stage_c_dir / relative_parquet
        if not src.is_file():
            raise FileNotFoundError(f"Missing parquet part referenced by source_files.csv: {src}")
        dst = output_dir / relative_parquet
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if args.link_mode == "symlink":
            os.symlink(src, dst)
        else:
            dst.write_bytes(src.read_bytes())
        linked += 1

    with (output_dir / "source_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    total_live_time_seconds = sum(float(row["rough_live_time_seconds"]) for row in kept_rows)

    metadata: dict = {}
    if metadata_src.is_file():
        metadata = json.loads(metadata_src.read_text(encoding="utf-8"))
    metadata.setdefault("live_time_basis", {})
    if isinstance(metadata.get("live_time_basis"), dict):
        metadata["live_time_basis"]["rough_live_time_seconds_sum_files"] = total_live_time_seconds
    metadata["common_gti_filter"] = {
        "source_stage_c_dir": str(stage_c_dir),
        "common_gti_tsv": str(args.common_gti_tsv.resolve()),
        "accepted_hour_count": len(accepted_hours),
        "kept_source_file_count": len(kept_rows),
        "kept_total_rough_live_time_seconds": total_live_time_seconds,
        "kept_total_rough_live_time_days": total_live_time_seconds / 86400.0,
    }
    (output_dir / "obs_events_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "accepted_hour_count": len(accepted_hours),
                "source_files_total_rows": len(rows),
                "source_files_kept_rows": len(kept_rows),
                "parquet_parts_linked_this_run": linked,
                "kept_total_rough_live_time_days": total_live_time_seconds / 86400.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
