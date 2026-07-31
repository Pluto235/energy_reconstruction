#!/usr/bin/env python3
"""Build the terminal v6/Pass5 common-GTI manifest from final recovery state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance"),
    )
    parser.add_argument("--strict-hour-manifest", type=Path)
    parser.add_argument("--strict-recovery-manifest", type=Path)
    parser.add_argument("--v6-gti-tsv", type=Path)
    parser.add_argument("--v6-gti-source-files", type=Path)
    parser.add_argument("--accepted-maps", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hour_key_from_iso(iso: str) -> str:
    date_part, time_part = iso.split("T")
    return f"{date_part.replace('-', '')}_{time_part[:2]}"


def nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_unique(values: list[str], label: str) -> set[str]:
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"{label} has {len(values) - len(unique)} duplicate entries")
    return unique


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    strict_hour_manifest = args.strict_hour_manifest or (
        run_dir / "strict_hour_selection" / "strict_hour_manifest.json"
    )
    strict_recovery_manifest = args.strict_recovery_manifest or (
        run_dir / "strict_recovery" / "strict_recovery_manifest.json"
    )
    v6_gti_tsv = args.v6_gti_tsv or (run_dir / "v6_gti_output" / "v6_sorted_gti.tsv")
    v6_gti_source_files = args.v6_gti_source_files or (
        run_dir / "v6_gti_output" / "v6_sorted_gti_source_files.csv"
    )
    accepted_maps_path = args.accepted_maps or (
        run_dir / "strict_recovery" / "accepted_maps.list"
    )
    output_dir = args.output_dir or (run_dir / "common_gti")
    output_dir.mkdir(parents=True, exist_ok=True)

    hour_manifest = json.loads(strict_hour_manifest.read_text(encoding="utf-8"))
    recovery = json.loads(strict_recovery_manifest.read_text(encoding="utf-8"))
    jobs = {str(job["label"]): job for job in hour_manifest["jobs"]}
    if len(jobs) != int(hour_manifest["job_count"]):
        raise ValueError("strict hour manifest has duplicate labels")

    records = recovery.get("records") or []
    if len(records) != int(recovery.get("job_count", -1)) or len(records) != len(jobs):
        raise ValueError("strict recovery records do not cover the strict hour manifest")
    recovery_by_label = {str(record["label"]): record for record in records}
    if len(recovery_by_label) != len(records) or set(recovery_by_label) != set(jobs):
        raise ValueError("strict recovery labels are duplicate or differ from strict hour labels")

    recovery_remaining = {
        "recovery_job_count": int(recovery.get("recovery_job_count", -1)),
        "step2_recovery_job_count": int(recovery.get("step2_recovery_job_count", -1)),
        "j2000_recovery_job_count": int(recovery.get("j2000_recovery_job_count", -1)),
    }
    if any(recovery_remaining.values()):
        raise ValueError(f"strict recovery is not terminal: {recovery_remaining}")

    accepted_records = [record for record in records if record["status"] == "accepted"]
    rejected_records = [record for record in records if record["status"] != "accepted"]
    accepted_labels = sorted(str(record["label"]) for record in accepted_records)
    rejected_labels = sorted(str(record["label"]) for record in rejected_records)
    if len(accepted_records) != int(recovery["accepted_map_count"]):
        raise ValueError("accepted record count differs from accepted_map_count")
    if len(rejected_records) != int(recovery["rejected_job_count"]):
        raise ValueError("rejected record count differs from rejected_job_count")

    accepted_map_lines = nonempty_lines(accepted_maps_path)
    accepted_map_set = require_unique(accepted_map_lines, "accepted_maps.list")
    recovery_map_lines = [
        str(record.get("output_j2000_uri") or record.get("output_j2000") or "")
        for record in accepted_records
    ]
    recovery_map_set = require_unique(recovery_map_lines, "strict recovery accepted map paths")
    if "" in recovery_map_set:
        raise ValueError("an accepted recovery record has no J2000 map path")
    if accepted_map_set != recovery_map_set:
        only_list = sorted(accepted_map_set - recovery_map_set)
        only_manifest = sorted(recovery_map_set - accepted_map_set)
        raise ValueError(
            "accepted map sets differ: "
            f"only accepted_maps.list={only_list[:3]}, only recovery manifest={only_manifest[:3]}"
        )

    accepted_hour_keys: set[str] = set()
    for label in accepted_labels:
        for iso in jobs[label]["central_available_hours"]:
            key = hour_key_from_iso(iso)
            if key in accepted_hour_keys:
                raise ValueError(f"accepted jobs claim the same central hour: {key}")
            accepted_hour_keys.add(key)

    with v6_gti_tsv.open(newline="", encoding="utf-8") as handle:
        gti_rows = list(csv.DictReader(handle, delimiter="\t"))
    included_gti_rows = [row for row in gti_rows if row["hour"] in accepted_hour_keys]
    excluded_gti_rows = [row for row in gti_rows if row["hour"] not in accepted_hour_keys]
    missing_hours = accepted_hour_keys - {row["hour"] for row in gti_rows}
    if missing_hours:
        raise ValueError(f"{len(missing_hours)} accepted hours have no v6 GTI")
    common_live_time_seconds = sum(float(row["duration_seconds"]) for row in included_gti_rows)

    with v6_gti_source_files.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    def source_hour_key(row: dict[str, str]) -> str:
        stem = Path(row["relative_path"]).stem
        return stem[len("Esg"):]

    included_source_rows = [
        row
        for row in source_rows
        if row["status"] == "processed" and source_hour_key(row) in accepted_hour_keys
    ]
    included_source_file_ids = sorted(
        {int(row["parent_source_file_id"]) for row in included_source_rows}
    )
    included_gti_source_row_ids = sorted(int(row["source_file_id"]) for row in included_source_rows)
    if len(included_source_file_ids) != len(accepted_hour_keys):
        raise ValueError(
            f"{len(included_source_file_ids)} unique source files != "
            f"{len(accepted_hour_keys)} accepted hours"
        )
    if len(included_source_rows) != len(included_gti_rows):
        raise ValueError(
            f"{len(included_source_rows)} GTI source rows != {len(included_gti_rows)} GTIs"
        )

    accepted_maps_match = (
        len(accepted_map_lines) == len(accepted_records)
        and accepted_map_set == recovery_map_set
    )
    manifest = {
        "description": (
            "Terminal common-GTI intersection of the v6 sorted recovered-time GTIs "
            "and the final official Pass5 accepted four-hour DI chunks."
        ),
        "run_dir": str(run_dir),
        "job_count": len(jobs),
        "strict_recovery_status_counts": recovery["status_counts"],
        "strict_recovery_remaining": recovery_remaining,
        "rejected_job_count": len(rejected_records),
        "accepted_job_count": len(accepted_records),
        "accepted_hour_count": len(accepted_hour_keys),
        "v6_sorted_gti_interval_count_total": len(gti_rows),
        "common_gti_interval_count": len(included_gti_rows),
        "excluded_gti_interval_count": len(excluded_gti_rows),
        "common_gti_live_time_seconds": common_live_time_seconds,
        "common_gti_live_time_days": common_live_time_seconds / 86400.0,
        "included_source_file_count": len(included_source_file_ids),
        "included_gti_source_row_count": len(included_source_rows),
        "accepted_maps_list_current_line_count": len(accepted_map_lines),
        "accepted_maps_list_unique_count": len(accepted_map_set),
        "accepted_maps_list_matches_accepted_job_count": accepted_maps_match,
        "accepted_maps_list_matches_strict_recovery_manifest": accepted_maps_match,
        "terminal_self_consistency_note": (
            "Final recovery has zero remaining work; accepted_maps.list is unique and "
            "exactly matches every accepted record's J2000 URI."
        ),
        "provenance": {
            "strict_hour_manifest": {
                "path": str(strict_hour_manifest),
                "sha256": sha256(strict_hour_manifest),
            },
            "strict_recovery_manifest": {
                "path": str(strict_recovery_manifest),
                "sha256": sha256(strict_recovery_manifest),
            },
            "accepted_maps_list": {
                "path": str(accepted_maps_path),
                "sha256": sha256(accepted_maps_path),
            },
            "v6_gti_tsv": {"path": str(v6_gti_tsv), "sha256": sha256(v6_gti_tsv)},
            "v6_gti_source_files": {
                "path": str(v6_gti_source_files),
                "sha256": sha256(v6_gti_source_files),
            },
        },
        "accepted_labels": accepted_labels,
        "rejected_labels": rejected_labels,
        "accepted_map_uris": accepted_map_lines,
        "included_source_file_ids": included_source_file_ids,
        "included_gti_source_row_ids": included_gti_source_row_ids,
    }

    manifest_path = output_dir / "common_gti_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "common_gti.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["source_file_id", "hour", "interval_index", "start_mjd", "stop_mjd", "duration_seconds"]
        )
        for row in included_gti_rows:
            writer.writerow(
                [
                    row["source_file_id"],
                    row["hour"],
                    row["interval_index"],
                    row["start_mjd"],
                    row["stop_mjd"],
                    row["duration_seconds"],
                ]
            )
    with (output_dir / "included_source_file_ids.txt").open("w", encoding="utf-8") as handle:
        for source_file_id in included_source_file_ids:
            handle.write(f"{source_file_id}\n")

    summary = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "accepted_labels",
            "rejected_labels",
            "accepted_map_uris",
            "included_source_file_ids",
            "included_gti_source_row_ids",
        }
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
