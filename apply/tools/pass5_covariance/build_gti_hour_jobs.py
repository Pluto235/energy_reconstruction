#!/usr/bin/env python3
"""Build IHEP hourly Pass5 jobs that apply the exact v6 recovered-time GTIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_EVENT_ROOT = Path(
    "/eos/user/h/hushicong/WCDA/8_All_sky_survey/data/Cod_FullArray/Data"
)
DEFAULT_XROOTD_PREFIX = "root://eos01.ihep.ac.cn/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-files", type=Path, required=True)
    parser.add_argument("--gti-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--xrootd-prefix", default=DEFAULT_XROOTD_PREFIX)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--stop", default="2022-06-30")
    return parser.parse_args()


def hour_from_relative_path(relative_path: str) -> str:
    return Path(relative_path).name.removeprefix("Esg").removesuffix(".root")


def event_path(event_root: Path, hour: str) -> Path:
    date, hour_of_day = hour.split("_")
    return event_root / date[:4] / date[4:] / f"{date}_{int(hour_of_day)}_event.root"


def xrootd_url(prefix: str, path: Path) -> str:
    return f"{prefix}{path}" if prefix else str(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    scratch_dir = args.scratch_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    list_dir = output_dir / "event_lists"
    gti_dir = output_dir / "gti_files"
    list_dir.mkdir(parents=True, exist_ok=True)
    gti_dir.mkdir(parents=True, exist_ok=True)

    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with args.gti_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            intervals[row["hour"]].append((float(row["start_mjd"]), float(row["stop_mjd"])))

    start_hour = args.start.replace("-", "") + "_00"
    stop_hour = args.stop.replace("-", "") + "_23"
    with args.source_files.open(newline="", encoding="utf-8") as handle:
        source_rows = [row for row in csv.DictReader(handle) if row.get("status") == "processed"]
    selected = []
    for row in source_rows:
        hour = hour_from_relative_path(row["relative_path"])
        if start_hour <= hour <= stop_hour:
            selected.append((hour, row))
    selected.sort(key=lambda item: item[0])

    missing_events: list[str] = []
    missing_gti: list[str] = []
    jobs: list[dict[str, object]] = []
    total_gti_seconds = 0.0
    for hour, row in selected:
        source_event = event_path(args.event_root, hour)
        if not source_event.is_file():
            missing_events.append(str(source_event))
            continue
        hour_intervals = sorted(intervals.get(hour, []))
        if not hour_intervals:
            missing_gti.append(hour)
            continue

        date, hour_of_day = hour.split("_")
        label = f"{date}_{int(hour_of_day)}"
        event_list = list_dir / f"{label}.list"
        gti_file = gti_dir / f"{label}.gti"
        with event_list.open("w", encoding="utf-8") as handle:
            handle.write(xrootd_url(args.xrootd_prefix, source_event) + "\n")
        with gti_file.open("w", encoding="utf-8") as handle:
            for interval_start, interval_stop in hour_intervals:
                handle.write(f"{interval_start:.15f} {interval_stop:.15f}\n")

        hour_output_dir = scratch_dir / date[:4] / date[4:]
        output_prefix = hour_output_dir / label
        duration = sum(max(0.0, stop - start) * 86400.0 for start, stop in hour_intervals)
        jobs.append(
            {
                "index": len(jobs),
                "hour": hour,
                "event_list": str(event_list),
                "gti_file": str(gti_file),
                "output_acc": f"{output_prefix}_acc.root",
                "output_bkg": f"{output_prefix}_bkg.root",
                "gti_duration_seconds": duration,
                "source_file_id": int(row["source_file_id"]),
            }
        )
        total_gti_seconds += duration

    with (output_dir / "jobs.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        columns = [
            "index",
            "hour",
            "event_list",
            "gti_file",
            "output_acc",
            "output_bkg",
            "gti_duration_seconds",
            "source_file_id",
        ]
        writer.writerow(columns)
        for job in jobs:
            writer.writerow([job[column] for column in columns])

    manifest = {
        "selection": "Pass5 event-level reconstruction masked by exact v6 recovered-time GTIs",
        "source_files": str(args.source_files),
        "gti_tsv": str(args.gti_tsv),
        "event_root": str(args.event_root),
        "xrootd_prefix": args.xrootd_prefix,
        "start": args.start,
        "stop": args.stop,
        "selected_hour_count": len(selected),
        "job_count": len(jobs),
        "missing_event_count": len(missing_events),
        "missing_gti_count": len(missing_gti),
        "gti_duration_seconds": total_gti_seconds,
        "gti_duration_days": total_gti_seconds / 86400.0,
        "scratch_dir": str(scratch_dir),
        "missing_events": missing_events,
        "missing_gti_hours": missing_gti,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps({key: value for key, value in manifest.items() if not key.startswith("missing_")}, indent=2))
    if missing_events or missing_gti:
        raise SystemExit("Cannot build complete GTI-masked Pass5 hourly sample")


if __name__ == "__main__":
    main()
