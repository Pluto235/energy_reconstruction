#!/usr/bin/env python3
"""Audit strict DI products and build an EOS J2000 recovery queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_RUN_DIR = Path("/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance")
DEFAULT_SOURCE_MAP_ROOT = Path(
    "/scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_map_chunks"
)
DEFAULT_SOURCE_HOUR_ROOT = Path(
    "/scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_hours"
)
DEFAULT_TARGET_HOUR_ROOT = Path(
    "/eos/user/l/liushijie/pass5_crab_v6_sorted_gti/"
    "pass5_crab_v6_sorted_gti_hours"
)
DEFAULT_TARGET_MAP_ROOT = Path(
    "/eos/user/l/liushijie/pass5_crab_v6_sorted_gti/"
    "pass5_crab_v6_sorted_gti_map_chunks"
)
DEFAULT_XROOTD_PREFIX = "root://eos01.ihep.ac.cn/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--source-map-root", type=Path, default=DEFAULT_SOURCE_MAP_ROOT)
    parser.add_argument("--target-map-root", type=Path, default=DEFAULT_TARGET_MAP_ROOT)
    parser.add_argument("--source-hour-root", type=Path, default=DEFAULT_SOURCE_HOUR_ROOT)
    parser.add_argument("--target-hour-root", type=Path, default=DEFAULT_TARGET_HOUR_ROOT)
    parser.add_argument("--xrootd-prefix", default=DEFAULT_XROOTD_PREFIX)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def relocated(path: str, source_root: Path, target_root: Path) -> Path:
    source = Path(path)
    try:
        relative = source.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"Map path is outside source root: {source}") from error
    return target_root / relative


def classify(error_text: str, output_text: str, bkg: Path, j2000: Path) -> str:
    if j2000.is_file() and j2000.stat().st_size > 0:
        return "accepted"
    bkg_write_failed = any(
        "error writing to file" in line and f"/{bkg.name}" in line
        for line in error_text.splitlines()
    )
    if bkg_write_failed or "Missing Step2 output" in error_text:
        return "recover_step2"
    if bkg.is_file() and bkg.stat().st_size > 0:
        return "recover_j2000"
    if "Acceptance correction factor" in error_text:
        return "rejected_acceptance"
    if "Background correction factor" in error_text:
        return "rejected_background"
    if "CorrectAcceptance" in output_text:
        return "recover_step2"
    if "CorrectAcceptance" not in output_text:
        return "rejected_no_central_events"
    return "recover_step2"


def xrootd_uri(path: Path, prefix: str) -> str:
    return f"{prefix.rstrip('/')}/{path}"


def resolve_job_path(run_dir: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else run_dir / candidate


def write_relocated_hour_list(
    source: Path,
    output: Path,
    source_hour_root: Path,
    target_hour_root: Path,
    xrootd_prefix: str,
) -> None:
    lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        path = Path(raw_line)
        try:
            relative = path.relative_to(source_hour_root)
        except ValueError:
            lines.append(raw_line)
        else:
            lines.append(xrootd_uri(target_hour_root / relative, xrootd_prefix))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    jobs_path = args.jobs or args.run_dir / "strict_hour_selection" / "jobs.tsv"
    output_dir = args.output_dir or args.run_dir / "strict_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with jobs_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            original_index = int(row["index"])
            bkg = relocated(row["output_bkg"], args.source_map_root, args.target_map_root)
            j2000 = relocated(row["output_j2000"], args.source_map_root, args.target_map_root)
            error_path = args.run_dir / f"run_strict_map_chunk.sh.err.87259359.{original_index}"
            output_path = args.run_dir / f"run_strict_map_chunk.sh.out.87259359.{original_index}"
            error_text = error_path.read_text(errors="replace") if error_path.is_file() else ""
            output_text = output_path.read_text(errors="replace") if output_path.is_file() else ""
            status = classify(error_text, output_text, bkg, j2000)
            records.append(
                {
                    "original_index": original_index,
                    "label": row["label"],
                    "list_path": str(resolve_job_path(args.run_dir, row["list"])),
                    "status": status,
                    "output_bkg": str(bkg),
                    "output_j2000": str(j2000),
                    "output_bkg_uri": xrootd_uri(bkg, args.xrootd_prefix),
                    "output_j2000_uri": xrootd_uri(j2000, args.xrootd_prefix),
                    "bkg_bytes": bkg.stat().st_size if bkg.is_file() else 0,
                    "j2000_bytes": j2000.stat().st_size if j2000.is_file() else 0,
                }
            )

    recovery = [record for record in records if record["status"] == "recover_j2000"]
    step2_recovery = [record for record in records if record["status"] == "recover_step2"]
    accepted = [record for record in records if record["status"] == "accepted"]
    rejected = [
        record
        for record in records
        if record["status"] not in {"accepted", "recover_j2000", "recover_step2"}
    ]

    with (output_dir / "recovery_jobs.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["recovery_index", "original_index", "label", "output_bkg", "output_j2000"]
        )
        for recovery_index, record in enumerate(recovery):
            writer.writerow(
                [
                    recovery_index,
                    record["original_index"],
                    record["label"],
                    record["output_bkg_uri"],
                    record["output_j2000_uri"],
                ]
            )

    step2_list_dir = output_dir / "step2_hour_lists"
    with (output_dir / "step2_recovery_jobs.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "recovery_index",
                "original_index",
                "label",
                "list_path",
                "output_bkg",
                "output_j2000",
            ]
        )
        for recovery_index, record in enumerate(step2_recovery):
            relocated_list = step2_list_dir / f"{record['label']}.list"
            write_relocated_hour_list(
                Path(str(record["list_path"])),
                relocated_list,
                args.source_hour_root,
                args.target_hour_root,
                args.xrootd_prefix,
            )
            writer.writerow(
                [
                    recovery_index,
                    record["original_index"],
                    record["label"],
                    relocated_list,
                    record["output_bkg_uri"],
                    record["output_j2000_uri"],
                ]
            )

    with (output_dir / "accepted_maps.list").open("w", encoding="utf-8") as handle:
        for record in accepted:
            handle.write(f"{record['output_j2000_uri']}\n")

    with (output_dir / "rejected_jobs.tsv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "original_index",
            "label",
            "status",
            "output_bkg",
            "output_j2000",
            "output_bkg_uri",
            "output_j2000_uri",
            "bkg_bytes",
            "j2000_bytes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: record[key] for key in fieldnames} for record in rejected)

    counts = Counter(str(record["status"]) for record in records)
    manifest = {
        "job_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "accepted_map_count": len(accepted),
        "j2000_recovery_job_count": len(recovery),
        "step2_recovery_job_count": len(step2_recovery),
        "recovery_job_count": len(recovery) + len(step2_recovery),
        "rejected_job_count": len(rejected),
        "source_jobs": str(jobs_path),
        "target_map_root": str(args.target_map_root),
        "records": records,
    }
    with (output_dir / "strict_recovery_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
