#!/usr/bin/env python3
"""Extract GTI-mask diagnostics from the completed Pass5 hourly job logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DURATION_RE = re.compile(r"GTI intervals = (\d+), duration = ([0-9.eE+-]+) seconds")
HOUR_RE = re.compile(r"/(\d{8})_(\d{1,2})_event\.root")
HISTOGRAM_RE = re.compile(
    r"GTI histogram counts before = ([0-9.eE+-]+), after = ([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_by_hour: dict[str, dict[str, object]] = {}
    for path in sorted(args.log_dir.glob("run_gti_hour.sh.out.*.*")):
        text = path.read_text(encoding="utf-8", errors="replace")
        duration = DURATION_RE.search(text)
        hour = HOUR_RE.search(text)
        histogram = HISTOGRAM_RE.search(text)
        if duration is None or hour is None or histogram is None:
            continue
        before = float(histogram.group(1))
        after = float(histogram.group(2))
        hour_key = f"{hour.group(1)}_{int(hour.group(2)):02d}"
        record: dict[str, object] = {
            "hour": hour_key,
            "gti_interval_count": int(duration.group(1)),
            "gti_duration_seconds": float(duration.group(2)),
            "histogram_counts_before": before,
            "histogram_counts_after": after,
            "histogram_retained_fraction": after / before if before > 0.0 else 0.0,
            "log_path": str(path),
        }
        prior = records_by_hour.get(hour_key)
        if prior is not None:
            comparable = {
                key: value for key, value in record.items() if key != "log_path"
            }
            prior_comparable = {
                key: value for key, value in prior.items() if key != "log_path"
            }
            if comparable != prior_comparable:
                raise ValueError(f"conflicting completed logs for {hour_key}")
            if path.stat().st_mtime <= Path(str(prior["log_path"])).stat().st_mtime:
                continue
        records_by_hour[hour_key] = record

    records = [records_by_hour[hour] for hour in sorted(records_by_hour)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    zeroed = sum(float(record["histogram_counts_after"]) == 0.0 for record in records)
    print(f"PASS5_HOUR_LOG_AUDIT hours={len(records)} zeroed_histograms={zeroed}")


if __name__ == "__main__":
    main()
