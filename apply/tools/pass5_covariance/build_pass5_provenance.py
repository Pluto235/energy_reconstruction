#!/usr/bin/env python3
"""Record file-level provenance for the completed common-GTI Pass5 fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--merge-log",
        type=Path,
        default=None,
        help="defaults to <run-dir>/merge_strict_map.sh.out.87689726.0",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    fit_dir = run_dir / "common_gti_fit"
    paths = {
        "strict_recovery_manifest": run_dir / "strict_recovery" / "strict_recovery_manifest.json",
        "accepted_maps_list": run_dir / "strict_recovery" / "accepted_maps.list",
        "merged_map": run_dir / "pass5_v6_common_gti_map.root",
        "strict_live_days": fit_dir / "strict_pass5_live_days.txt",
        "data_config": fit_dir / "data_config.yaml",
        "data_root": fit_dir / "data.root",
        "covariance_fit": fit_dir / "covariance_fit.yaml",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    merge_log = args.merge_log or (run_dir / "merge_strict_map.sh.out.87689726.0")
    data_config = yaml.safe_load(paths["data_config"].read_text(encoding="utf-8"))
    covariance_fit = yaml.safe_load(paths["covariance_fit"].read_text(encoding="utf-8"))
    accepted_maps = [
        line.strip()
        for line in paths["accepted_maps_list"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recovery_manifest = json.loads(
        paths["strict_recovery_manifest"].read_text(encoding="utf-8")
    )
    eos_root = str(recovery_manifest["target_map_root"])
    xrootd_environment = os.environ.copy()
    xrootd_environment.pop("LD_LIBRARY_PATH", None)
    listing = subprocess.run(
        ["/usr/bin/xrdfs", "eos01.ihep.ac.cn", "ls", "-l", eos_root],
        check=True,
        capture_output=True,
        text=True,
        env=xrootd_environment,
    ).stdout.splitlines()
    eos_nonempty_j2000_paths = {
        fields[-1]
        for line in listing
        if len(fields := line.split()) >= 7
        and int(fields[3]) > 0
        and fields[-1].endswith("_BKG_J2000.root")
    }
    accepted_map_paths = {
        path.removeprefix("root://eos01.ihep.ac.cn/") for path in accepted_maps
    }
    embedded_data_path = Path(str(data_config["data_save_path"]))
    embedded_covariance_data_path = Path(str(covariance_fit["selection"]["all_sky_map"]))
    live_days = float(paths["strict_live_days"].read_text(encoding="utf-8").strip())
    merge_tail = merge_log.read_text(encoding="utf-8", errors="replace").splitlines()[-1]
    expected_merge_tail = (
        f"STRICT_MAP_MERGE_COMPLETE maps={len(accepted_maps)} "
        f"output={paths['merged_map']}"
    )
    mtimes = [paths[key].stat().st_mtime for key in ("merged_map", "data_config", "data_root", "covariance_fit")]

    payload = {
        "description": "File-level provenance audit for the final common-GTI official Pass5 fit.",
        "files": {key: file_record(path) for key, path in paths.items()},
        "merge": {
            "log_path": str(merge_log),
            "terminal_line": merge_tail,
            "expected_terminal_line": expected_merge_tail,
            "terminal_line_matches": merge_tail == expected_merge_tail,
            "input_map_line_count": len(accepted_maps),
            "input_map_unique_count": len(set(accepted_maps)),
        },
        "eos_map_set": {
            "root": eos_root,
            "nonempty_j2000_count": len(eos_nonempty_j2000_paths),
            "accepted_map_unique_count": len(accepted_map_paths),
            "matches_accepted_maps_list": eos_nonempty_j2000_paths == accepted_map_paths,
            "only_eos": sorted(eos_nonempty_j2000_paths - accepted_map_paths),
            "only_accepted_maps_list": sorted(accepted_map_paths - eos_nonempty_j2000_paths),
        },
        "data_config": {
            "data_read_path": data_config["data_read_path"],
            "data_read_path_matches_merged_map": data_config["data_read_path"]
            == [str(paths["merged_map"])],
            "live_time_days": float(data_config["live_time"][0]),
            "live_time_matches_header": abs(float(data_config["live_time"][0]) - live_days) < 1e-12,
            "embedded_data_save_path": str(embedded_data_path),
            "embedded_data_save_path_exists": embedded_data_path.is_file(),
            "actual_data_root": str(paths["data_root"]),
        },
        "covariance_fit": {
            "embedded_all_sky_map": str(embedded_covariance_data_path),
            "embedded_all_sky_map_exists": embedded_covariance_data_path.is_file(),
            "actual_data_root": str(paths["data_root"]),
            "covariance_status": int(covariance_fit["output_option"]["gtlike"]["covariance_status"]),
            "pivot_tev": float(
                covariance_fit["source_dict"]["J0534+2200"]["sed_model"]["E_0"]
            ),
        },
        "chronology": {
            "merged_map_then_data_config_then_data_root_then_covariance_fit": all(
                left <= right for left, right in zip(mtimes, mtimes[1:])
            )
        },
        "path_provenance_status": "relocated_with_stale_embedded_paths",
        "path_provenance_note": (
            "data_config.yaml and covariance_fit.yaml retain common_gti_fit_interactive "
            "output paths that no longer exist. The products are present under common_gti_fit "
            "with a monotonic generation chronology, but the embedded path chain is not "
            "self-contained and must not be described as fully path-clean provenance."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
