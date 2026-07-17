#!/usr/bin/env python3
"""Reconstruct the exact Stage C recovered-time GTIs for the v6 sample."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-files", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--tree", default="t_recovered_time")
    parser.add_argument("--match-status", type=int, default=0)
    parser.add_argument("--gap-threshold-sec", type=float, default=60.0)
    parser.add_argument("--entries-per-chunk", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--stop", default="2022-06-30")
    return parser.parse_args()


def hour_from_relative_path(relative_path: str) -> str:
    name = Path(relative_path).name
    if not name.startswith("Esg") or not name.endswith(".root"):
        raise ValueError(f"Unexpected v6 observation filename: {name}")
    return name.removeprefix("Esg").removesuffix(".root")


def matched_mjd_chunks(
    path: Path,
    tree_name: str,
    match_status: int,
    entries_per_chunk: int,
) -> Iterator[np.ndarray]:
    with uproot.open(path) as root_file:
        tree = root_file[tree_name]
        for arrays in tree.iterate(
            ["mjd", "match_status"],
            step_size=entries_per_chunk,
            library="np",
        ):
            mjd = np.asarray(arrays["mjd"], dtype=np.float64)
            status = np.asarray(arrays["match_status"])
            selected = mjd[(status == match_status) & np.isfinite(mjd)]
            if selected.size:
                yield selected


def intervals_for_file(
    path: Path,
    tree_name: str,
    match_status: int,
    gap_threshold_sec: float,
    entries_per_chunk: int,
) -> tuple[list[tuple[float, float]], int, int, float]:
    threshold_days = gap_threshold_sec / 86400.0
    chunks: list[np.ndarray] = []
    negative_steps = 0
    previous: float | None = None

    for values in matched_mjd_chunks(path, tree_name, match_status, entries_per_chunk):
        if previous is not None and float(values[0]) < previous:
            negative_steps += 1
        negative_steps += int(np.count_nonzero(np.diff(values) < 0))
        previous = float(values[-1])
        chunks.append(values)

    if not chunks:
        return [], negative_steps, 0, 0.0

    ordered = np.sort(np.concatenate(chunks))
    differences = np.diff(ordered)
    gap_indices = np.flatnonzero(differences > threshold_days)
    segment_starts = np.concatenate((np.asarray([0]), gap_indices + 1))
    segment_stops = np.concatenate((gap_indices + 1, np.asarray([ordered.size])))
    intervals = [
        (float(ordered[start]), float(ordered[stop - 1]))
        for start, stop in zip(segment_starts, segment_stops)
    ]
    gap_seconds = float(differences[gap_indices].sum() * 86400.0)
    return intervals, negative_steps, int(gap_indices.size), gap_seconds


def process_row(task: tuple[str, dict[str, str], argparse.Namespace]) -> dict[str, object]:
    hour, row, args = task
    time_path = Path(row["time_path"])
    intervals, negative_steps, sorted_gap_count, sorted_gap_seconds = intervals_for_file(
        time_path,
        args.tree,
        args.match_status,
        args.gap_threshold_sec,
        args.entries_per_chunk,
    )
    duration = sum(max(0.0, stop - start) * 86400.0 for start, stop in intervals)
    expected = float(row["rough_live_time_seconds"])
    if not intervals and float(row["matched_span_seconds"]) > 0:
        raise RuntimeError(f"No matched GTI intervals for {hour}: {time_path}")
    return {
        "source_file_id": int(row["source_file_id"]),
        "hour": hour,
        "time_path": str(time_path),
        "intervals": intervals,
        "interval_count": len(intervals),
        "duration_seconds": duration,
        "sorted_gap_count": sorted_gap_count,
        "sorted_gap_seconds": sorted_gap_seconds,
        "original_order_negative_step_count": negative_steps,
        "source_files_rough_live_time_seconds": expected,
        "difference_from_source_rough_seconds": duration - expected,
    }


def main() -> None:
    args = parse_args()
    if args.gap_threshold_sec <= 0:
        raise ValueError("--gap-threshold-sec must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)

    with args.source_files.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "processed"]

    start_hour = args.start.replace("-", "") + "_00"
    stop_hour = args.stop.replace("-", "") + "_23"
    selected_rows = []
    for row in rows:
        hour = hour_from_relative_path(row["relative_path"])
        if start_hour <= hour <= stop_hour:
            selected_rows.append((hour, row))
    selected_rows.sort(key=lambda item: item[0])

    records: list[dict[str, object]] = []
    output_rows: list[list[object]] = []
    tasks = [(hour, row, args) for hour, row in selected_rows]
    if args.workers == 1:
        records = [process_row(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_row, task): task[0] for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if completed % 100 == 0 or completed == len(futures):
                    print(f"GTI_PROGRESS {completed}/{len(futures)}", flush=True)
    records.sort(key=lambda record: str(record["hour"]))

    for record in records:
        intervals = record.pop("intervals")
        assert isinstance(intervals, list)
        for interval_index, (interval_start, interval_stop) in enumerate(intervals):
            output_rows.append(
                [
                    record["source_file_id"],
                    record["hour"],
                    interval_index,
                    f"{interval_start:.15f}",
                    f"{interval_stop:.15f}",
                    f"{max(0.0, interval_stop - interval_start) * 86400.0:.12f}",
                    record["time_path"],
                ]
            )

    total_duration = sum(float(record["duration_seconds"]) for record in records)
    total_expected = sum(float(record["source_files_rough_live_time_seconds"]) for record in records)
    total_negative_steps = sum(int(record["original_order_negative_step_count"]) for record in records)
    total_sorted_gaps = sum(int(record["sorted_gap_count"]) for record in records)
    total_sorted_gap_seconds = sum(float(record["sorted_gap_seconds"]) for record in records)
    max_file_difference = max(
        (abs(float(record["difference_from_source_rough_seconds"])) for record in records),
        default=0.0,
    )

    with args.output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "source_file_id",
                "hour",
                "interval_index",
                "start_mjd",
                "stop_mjd",
                "duration_seconds",
                "time_path",
            ]
        )
        writer.writerows(output_rows)

    manifest = {
        "source_files": str(args.source_files),
        "tree": args.tree,
        "match_status": args.match_status,
        "gap_threshold_sec": args.gap_threshold_sec,
        "selection_start": args.start,
        "selection_stop": args.stop,
        "processed_hour_count": len(selected_rows),
        "interval_count": len(output_rows),
        "sorted_gti_live_time_seconds": total_duration,
        "sorted_gti_live_time_days": total_duration / 86400.0,
        "sorted_gap_count": total_sorted_gaps,
        "sorted_gap_seconds": total_sorted_gap_seconds,
        "source_files_rough_live_time_seconds": total_expected,
        "source_files_rough_live_time_days": total_expected / 86400.0,
        "difference_from_source_rough_seconds": total_duration - total_expected,
        "max_abs_file_difference_from_source_rough_seconds": max_file_difference,
        "original_order_negative_mjd_step_count": total_negative_steps,
        "workers": args.workers,
        "files": records,
    }
    with args.manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
