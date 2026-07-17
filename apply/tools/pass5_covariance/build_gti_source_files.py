#!/usr/bin/env python3
"""Expand Stage C source_files.csv into one exact exposure row per sorted GTI."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-files", type=Path, required=True)
    parser.add_argument("--gti-tsv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    return parser.parse_args()


def hour_from_relative_path(relative_path: str) -> str:
    return Path(relative_path).name.removeprefix("Esg").removesuffix(".root")


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)

    with args.source_files.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_by_hour = {
            hour_from_relative_path(row["relative_path"]): row
            for row in reader
            if row.get("status") == "processed"
        }

    intervals: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.gti_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            intervals[row["hour"]].append(row)

    missing_source_hours = sorted(set(intervals) - set(source_by_hour))
    missing_gti_hours = sorted(set(source_by_hour) - set(intervals))
    if missing_source_hours or missing_gti_hours:
        raise RuntimeError(
            f"GTI/source hour mismatch: missing_source={len(missing_source_hours)} "
            f"missing_gti={len(missing_gti_hours)}"
        )

    extra_fields = ["parent_source_file_id", "gti_interval_index"]
    output_fields = fieldnames + [field for field in extra_fields if field not in fieldnames]
    output_rows: list[dict[str, str]] = []
    total_live_seconds = 0.0
    for hour in sorted(source_by_hour):
        source = source_by_hour[hour]
        for interval in sorted(intervals[hour], key=lambda row: int(row["interval_index"])):
            start = float(interval["start_mjd"])
            stop = float(interval["stop_mjd"])
            duration = max(0.0, (stop - start) * 86400.0)
            row = dict(source)
            row["parent_source_file_id"] = source["source_file_id"]
            row["gti_interval_index"] = interval["interval_index"]
            row["source_file_id"] = str(len(output_rows))
            row["matched_mjd_min"] = f"{start:.15f}"
            row["matched_mjd_max"] = f"{stop:.15f}"
            row["matched_span_seconds"] = f"{duration:.12f}"
            row["matched_gap_count"] = "0"
            row["matched_gap_seconds"] = "0"
            row["rough_live_time_seconds"] = f"{duration:.12f}"
            row["selected_mjd_min"] = f"{start:.15f}"
            row["selected_mjd_max"] = f"{stop:.15f}"
            output_rows.append(row)
            total_live_seconds += duration

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "selection": "one Stage F exposure row per sorted v6 recovered-time GTI",
        "source_files": str(args.source_files),
        "gti_tsv": str(args.gti_tsv),
        "source_hour_count": len(source_by_hour),
        "output_interval_count": len(output_rows),
        "total_live_seconds": total_live_seconds,
        "total_live_days": total_live_seconds / 86400.0,
        "output_csv": str(args.output_csv),
    }
    with args.manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
