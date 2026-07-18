#!/usr/bin/env python3
"""Map the v6 hourly observation selection onto Pass5 DI intermediates."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/eos/user/h/hushicong/WCDA/8_All_sky_survey/data/Cod_FullArray/"
    "DI_mask/WCDA_v0/"
    "PincOpt_7bins_nq03_ge200_test_v2_timebin1_4hours_bkg10hours_le2000_bkgJnow"
)
DEFAULT_XROOTD_PREFIX = "root://eos01.ihep.ac.cn/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-files", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--gti-manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--xrootd-prefix", default=DEFAULT_XROOTD_PREFIX)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--stop", default="2022-06-30")
    return parser.parse_args()


def selected_hour(relative_path: str) -> datetime:
    filename = Path(relative_path).name
    return datetime.strptime(filename, "Esg%Y%m%d_%H.root")


def prefix_for(data_root: Path, hour: datetime) -> Path:
    return data_root / hour.strftime("%Y") / hour.strftime("%m%d") / hour.strftime("%Y%m%d_%-H")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    list_dir = args.output_dir / "hour_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[datetime, dict[str, str]] = {}
    with args.source_files.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "processed":
                continue
            selected[selected_hour(row["relative_path"])] = row

    start = datetime.strptime(args.start, "%Y-%m-%d")
    stop = datetime.strptime(args.stop, "%Y-%m-%d")
    selected_in_window = {hour for hour in selected if start <= hour < stop + timedelta(days=1)}

    available: set[datetime] = set()
    missing: list[dict[str, object]] = []
    for hour in sorted(selected_in_window):
        prefix = prefix_for(args.data_root, hour)
        acc = Path(f"{prefix}_acc.root")
        bkg = Path(f"{prefix}_bkg.root")
        if acc.is_file() and bkg.is_file():
            available.add(hour)
        else:
            row = selected[hour]
            missing.append(
                {
                    "hour": hour.isoformat(),
                    "prefix": str(prefix),
                    "acc_exists": acc.is_file(),
                    "bkg_exists": bkg.is_file(),
                    "v6_rough_live_time_seconds": float(row["rough_live_time_seconds"]),
                }
            )

    jobs: list[dict[str, object]] = []
    day = start
    while day <= stop:
        for segment in range(6):
            central_start = day + timedelta(hours=4 * segment)
            context = [central_start + timedelta(hours=offset) for offset in range(-3, 7)]
            central = context[3:7]
            central_available = [hour for hour in central if hour in available]
            if not central_available:
                continue

            label = f"{day:%Y%m%d}_{segment}"
            list_path = list_dir / f"{label}.list"
            with list_path.open("w", encoding="utf-8") as handle:
                for hour in context:
                    if hour in available:
                        handle.write(f"{args.xrootd_prefix}{prefix_for(args.data_root, hour)}\n")
                    else:
                        handle.write(f"/nonexistent/pass5_v6_hour_mask/{hour:%Y%m%d_%H}\n")

            output_bkg = args.scratch_dir / f"{label}_BKG.root"
            output_j2000 = args.scratch_dir / f"{label}_BKG_J2000.root"
            jobs.append(
                {
                    "index": len(jobs),
                    "label": label,
                    "list": str(list_path),
                    "output_bkg": str(output_bkg),
                    "output_j2000": str(output_j2000),
                    "central_selected_hours": [hour.isoformat() for hour in central if hour in selected_in_window],
                    "central_available_hours": [hour.isoformat() for hour in central_available],
                    "context_available_hours": [hour.isoformat() for hour in context if hour in available],
                }
            )
        day += timedelta(days=1)

    jobs_tsv = args.output_dir / "jobs.tsv"
    with jobs_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["index", "label", "list", "output_bkg", "output_j2000"])
        for job in jobs:
            writer.writerow([job[key] for key in ("index", "label", "list", "output_bkg", "output_j2000")])

    total_rough_live = sum(float(selected[hour]["rough_live_time_seconds"]) for hour in selected_in_window)
    missing_rough_live = sum(float(item["v6_rough_live_time_seconds"]) for item in missing)
    gti_audit = None
    if args.gti_manifest:
        with args.gti_manifest.open(encoding="utf-8") as handle:
            gti_audit = json.load(handle)
        if int(gti_audit["processed_hour_count"]) != len(selected_in_window):
            raise ValueError(
                "GTI manifest hour count does not match selected source files: "
                f"{gti_audit['processed_hour_count']} != {len(selected_in_window)}"
            )
    manifest = {
        "selection": "v6 processed hours rebuilt from Pass5 events after exact sorted-GTI masking",
        "start": args.start,
        "stop": args.stop,
        "v6_source_files": str(args.source_files),
        "pass5_data_root": str(args.data_root),
        "pass5_xrootd_prefix": args.xrootd_prefix,
        "v6_selected_hour_count": len(selected_in_window),
        "pass5_available_selected_hour_count": len(available),
        "missing_selected_hour_count": len(missing),
        "v6_rough_live_time_seconds": total_rough_live,
        "v6_rough_live_time_days": total_rough_live / 86400.0,
        "sorted_gti_manifest": str(args.gti_manifest) if args.gti_manifest else None,
        "sorted_gti_live_time_seconds": (
            float(gti_audit["sorted_gti_live_time_seconds"]) if gti_audit else None
        ),
        "sorted_gti_live_time_days": (
            float(gti_audit["sorted_gti_live_time_days"]) if gti_audit else None
        ),
        "missing_v6_rough_live_time_seconds": missing_rough_live,
        "job_count": len(jobs),
        "missing_selected_hours": missing,
        "jobs": jobs,
    }
    with (args.output_dir / "strict_hour_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps({key: manifest[key] for key in manifest if key not in {"jobs", "missing_selected_hours"}}, indent=2))


if __name__ == "__main__":
    main()
