#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


EXPECTED_RUN_DIR = "/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670"
EXPECTED_OUTPUT_ROOT = "/mnt/mydisk/WCDA_simulation_binned_response_v6_64670"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the v6 _64670 MC binned cache before downstream apply stages.")
    parser.add_argument("--binned-root", type=str, required=True)
    parser.add_argument("--expected-files", type=int, default=10000)
    parser.add_argument("--expected-run-dir", type=str, default=EXPECTED_RUN_DIR)
    parser.add_argument("--expected-output-root", type=str, default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--require-provenance", action="store_true", default=False)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_bin_counts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def find_bad_strings(payload: object, needles: Iterable[str]) -> List[str]:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return [needle for needle in needles if needle in text]


def main() -> None:
    args = parse_args()
    root = Path(args.binned_root).resolve()
    summary_dir = root / "summary"
    run_summary_path = summary_dir / "run_summary.json"
    bin_counts_path = summary_dir / "bin_counts.csv"
    failures: List[str] = []

    if not root.exists():
        failures.append(f"missing binned root: {root}")
    if not run_summary_path.exists():
        failures.append(f"missing run summary: {run_summary_path}")
    if not bin_counts_path.exists():
        failures.append(f"missing bin counts CSV: {bin_counts_path}")
    if failures:
        raise SystemExit("\n".join(failures))

    run_summary = load_json(run_summary_path)
    bin_rows = read_bin_counts(bin_counts_path)
    processed_files = int(run_summary.get("processed_files") or 0)
    if processed_files != int(args.expected_files):
        failures.append(f"processed_files={processed_files}, expected {int(args.expected_files)}")

    inferred_events = int(run_summary.get("inferred_events") or 0)
    total_events = int(run_summary.get("total_events") or 0)
    if total_events <= 0:
        failures.append("total_events is not positive")
    if inferred_events <= 0:
        failures.append("inferred_events is not positive")

    if len(bin_rows) < 80:
        failures.append(f"bin_counts.csv has only {len(bin_rows)} rows; expected the full formal grid")
    positive_bins = sum(1 for row in bin_rows if int(float(row.get("count") or 0)) > 0)
    if positive_bins <= 0:
        failures.append("bin_counts.csv has no positive-count bins")

    metadata = run_summary.get("run_metadata")
    if isinstance(metadata, dict):
        run_dir = str(metadata.get("run_dir") or "")
        output_root = str(metadata.get("output_root") or "")
        checkpoint = str(metadata.get("checkpoint_path") or "")
        if "_64670" not in run_dir:
            failures.append(f"run_metadata.run_dir does not reference _64670: {run_dir}")
        if "_64670" not in checkpoint:
            failures.append(f"run_metadata.checkpoint_path does not reference _64670: {checkpoint}")
        if output_root and Path(output_root).resolve() != root:
            failures.append(f"run_metadata.output_root={output_root} does not match {root}")
        expected_run = str(Path(args.expected_run_dir).resolve())
        if run_dir and str(Path(run_dir).resolve()) != expected_run:
            failures.append(f"run_metadata.run_dir={run_dir} does not match expected {expected_run}")
    elif args.require_provenance:
        failures.append("run_summary.json is missing run_metadata provenance")

    bad = find_bad_strings(run_summary, ["theta_recoxy_position_embed_midenergy_8666", "no_core_cut_2724"])
    if bad:
        failures.append(f"run_summary.json contains obsolete model references: {', '.join(bad)}")

    if failures:
        raise SystemExit("v6 MC cache validation failed:\n- " + "\n- ".join(failures))

    print(
        json.dumps(
            {
                "status": "passed",
                "binned_root": str(root),
                "processed_files": processed_files,
                "total_events": total_events,
                "inferred_events": inferred_events,
                "positive_bins": positive_bins,
                "has_run_metadata": isinstance(metadata, dict),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
