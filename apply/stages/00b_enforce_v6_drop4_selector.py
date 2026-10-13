#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


INCLUDED_CELL_IDS = {1, 2, 3, 14, 15, 16, 26, 27, 28, 29, 30, 40, 41, 42, 52, 53, 54, 55, 65, 66, 67, 68, 69, 81, 82, 83}
DROP4_EXCLUDED_CELL_IDS = {4, 17, 39, 43}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce the v4 drop4 cell contract on a v6 _64670 prefit selector.")
    parser.add_argument("--input-selector-csv", type=str, required=True)
    parser.add_argument("--output-selector-csv", type=str, required=True)
    parser.add_argument("--candidate-ledger-csv", type=str, required=True)
    parser.add_argument("--subset-version", type=str, default="v6_64670_drop4_psfborrow")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    input_rows, input_fields = read_csv(Path(args.input_selector_csv).resolve())
    ledger_rows, _ = read_csv(Path(args.candidate_ledger_csv).resolve())
    ledger_by_id = {int(row["cell_id"]): row for row in ledger_rows}
    rows_by_id = {int(row["cell_id"]): row for row in input_rows}
    expected_ids = set(range(1, 85))
    missing = sorted(expected_ids - set(rows_by_id))
    if missing:
        raise SystemExit(f"input selector is missing cell ids: {missing}")

    output_rows: List[Dict[str, object]] = []
    for cell_id in sorted(rows_by_id):
        row: Dict[str, object] = dict(rows_by_id[cell_id])
        ledger = ledger_by_id.get(cell_id, {})
        for field in ("nhit_bin", "predE_bin", "mc_count", "central99_flag"):
            if field in ledger:
                row[field] = ledger[field]
        row["subset_version"] = str(args.subset_version)
        if cell_id in INCLUDED_CELL_IDS:
            row["include"] = 1
            row["subset_reason"] = "v4 drop4 preserved fit cell; v6 _64670 MC prefit fields refreshed"
            row["cell_role"] = "drop4_fit"
            row["exclusion_source"] = ""
        elif cell_id in DROP4_EXCLUDED_CELL_IDS:
            row["include"] = 0
            prior = str(row.get("subset_reason") or "central99 MC-occupancy-ridge prefit cell")
            if "v4 cell-selection-bias drop4 control" not in prior:
                prior = prior.rstrip("; ") + "; excluded for v4 cell-selection-bias drop4 control"
            row["subset_reason"] = prior
            row["cell_role"] = "drop4_control_excluded"
            row["exclusion_source"] = "v4_drop4_bias_control"
        else:
            row["include"] = 0
            row["cell_role"] = "excluded"
            row["exclusion_source"] = row.get("exclusion_source") or "MC_prefit_selector"
            if truthy(rows_by_id[cell_id].get("include")):
                row["subset_reason"] = "excluded to preserve v4 drop4 cell-selection contract for v6 comparability"
        output_rows.append(row)

    included = [int(row["cell_id"]) for row in output_rows if truthy(row.get("include"))]
    drop4 = [
        int(row["cell_id"])
        for row in output_rows
        if int(row["cell_id"]) in DROP4_EXCLUDED_CELL_IDS and not truthy(row.get("include"))
    ]
    if included != sorted(INCLUDED_CELL_IDS):
        raise SystemExit(f"included cells do not match v4 contract: {included}")
    if drop4 != sorted(DROP4_EXCLUDED_CELL_IDS):
        raise SystemExit(f"drop4 excluded controls do not match v4 contract: {drop4}")

    extra_fields = ["psf_systematic_variant", "psf_borrowed_from", "psf_borrow_method"]
    fieldnames = list(input_fields)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(Path(args.output_selector_csv).resolve(), output_rows, fieldnames)

    print(
        json.dumps(
            {
                "status": "passed",
                "output_selector_csv": str(Path(args.output_selector_csv).resolve()),
                "subset_version": str(args.subset_version),
                "included_cell_ids": included,
                "drop4_excluded_cell_ids": drop4,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
