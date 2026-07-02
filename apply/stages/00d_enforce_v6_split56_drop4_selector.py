#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OLD_SPLIT_SOURCE_KEY = ("[2000,3000)", "[5,6)")
NEW_SPLIT_KEYS = (("[2000,3000)", "[5,5.5)"), ("[2000,3000)", "[5.5,6)"))
EXPECTED_CANDIDATE_CELLS = 91
EXPECTED_FIT_CELLS = 27
DEFAULT_MIN_MC_COUNT = 1000
SPLIT_CHILD_MIN_MC_COUNT = 500
MIN_RIDGE_PEAK_FRACTION = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce the v6 _64670 split56 baselinev4 selector contract by bin labels, not old cell ids."
    )
    parser.add_argument("--input-selector-csv", type=Path, required=True)
    parser.add_argument("--output-selector-csv", type=Path, required=True)
    parser.add_argument("--candidate-ledger-csv", type=Path, required=True)
    parser.add_argument("--baseline-selector-csv", type=Path, default=Path("apply/config/cell_selector_v6_drop4_psfborrow.csv"))
    parser.add_argument("--subset-version", type=str, default="v6_64670_split56_drop4_psfborrow")
    parser.add_argument("--expected-candidate-cells", type=int, default=EXPECTED_CANDIDATE_CELLS)
    parser.add_argument("--expected-fit-cells", type=int, default=EXPECTED_FIT_CELLS)
    parser.add_argument("--default-min-mc-count", type=int, default=DEFAULT_MIN_MC_COUNT)
    parser.add_argument("--split-child-min-mc-count", type=int, default=SPLIT_CHILD_MIN_MC_COUNT)
    parser.add_argument("--min-ridge-peak-fraction", type=float, default=MIN_RIDGE_PEAK_FRACTION)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def bin_key(row: Dict[str, str]) -> Tuple[str, str]:
    return str(row["nhit_bin"]), str(row["predE_bin"])


def main() -> None:
    args = parse_args()
    input_rows, input_fields = read_csv(resolve(args.input_selector_csv))
    ledger_rows, _ = read_csv(resolve(args.candidate_ledger_csv))
    baseline_rows, _ = read_csv(resolve(args.baseline_selector_csv))
    failures: List[str] = []

    if len(input_rows) != int(args.expected_candidate_cells):
        failures.append(f"input selector has {len(input_rows)} rows, expected {int(args.expected_candidate_cells)}")
    if len(ledger_rows) != int(args.expected_candidate_cells):
        failures.append(f"candidate ledger has {len(ledger_rows)} rows, expected {int(args.expected_candidate_cells)}")

    rows_by_key = {bin_key(row): row for row in input_rows}
    ledger_by_key = {bin_key(row): row for row in ledger_rows}
    baseline_include_keys = {bin_key(row) for row in baseline_rows if truthy(row.get("include"))}
    baseline_exclude_by_key = {bin_key(row): row for row in baseline_rows if not truthy(row.get("include"))}
    if OLD_SPLIT_SOURCE_KEY not in baseline_include_keys:
        failures.append(f"baseline selector does not include source split cell {OLD_SPLIT_SOURCE_KEY}")

    desired_include_keys = set(baseline_include_keys)
    desired_include_keys.discard(OLD_SPLIT_SOURCE_KEY)
    desired_include_keys.update(NEW_SPLIT_KEYS)

    missing = sorted(desired_include_keys - set(rows_by_key))
    if missing:
        failures.append(f"split56 selector is missing desired include bins: {missing}")

    split_gate_failures: List[Dict[str, object]] = []
    split_gate_records: Dict[Tuple[str, str], Dict[str, object]] = {}
    for key in NEW_SPLIT_KEYS:
        row = rows_by_key.get(key)
        if row is None:
            continue
        count = int(float(row.get("mc_count") or 0))
        ridge_peak_fraction = float(row.get("ridge_peak_fraction") or 0.0)
        effective_min_mc_count = int(args.split_child_min_mc_count)
        gates = {
            "central99_flag": truthy(row.get("central99_flag")),
            f"mc_count_ge_{effective_min_mc_count}": count >= effective_min_mc_count,
            f"ridge_peak_fraction_ge_{float(args.min_ridge_peak_fraction):g}": ridge_peak_fraction >= float(args.min_ridge_peak_fraction),
            "not_tail": row.get("predE_bin") != ">=6",
        }
        split_gate_records[key] = {
            "cell_id": int(row["cell_id"]),
            "nhit_bin": key[0],
            "predE_bin": key[1],
            "mc_count": count,
            "default_min_mc_count": int(args.default_min_mc_count),
            "effective_min_mc_count": effective_min_mc_count,
            "ridge_peak_fraction": ridge_peak_fraction,
            "original_prefit_physical_ridge_flag": truthy(row.get("physical_ridge_flag")),
            "gates": gates,
            "uses_count_gate_exception": count < int(args.default_min_mc_count),
        }
        if not all(gates.values()):
            split_gate_failures.append({"bin": key, "cell_id": row.get("cell_id"), "gates": gates, "row": row})
    if split_gate_failures:
        failures.append(
            "new split child bin failed quality gate(s): " + json.dumps(split_gate_failures, indent=2, sort_keys=True)
        )

    if failures:
        raise SystemExit("v6 split56 selector enforcement failed:\n- " + "\n- ".join(failures))

    output_rows: List[Dict[str, object]] = []
    for row in sorted(input_rows, key=lambda item: int(item["cell_id"])):
        key = bin_key(row)
        out: Dict[str, object] = dict(row)
        ledger = ledger_by_key.get(key, {})
        for field in (
            "nhit_bin",
            "predE_bin",
            "mc_count",
            "central99_flag",
            "fit_predE_range_flag",
            "tail_bin_flag",
        ):
            if field in ledger:
                out[field] = ledger[field]
        out["subset_version"] = str(args.subset_version)
        if key in desired_include_keys:
            out["include"] = 1
            if key in NEW_SPLIT_KEYS:
                gate = split_gate_records[key]
                out["physical_ridge_flag"] = 1
                out["split_child_gate_exception"] = int(bool(gate["uses_count_gate_exception"]))
                out["split_child_effective_min_mc_count"] = gate["effective_min_mc_count"]
                out["split_child_default_min_mc_count"] = gate["default_min_mc_count"]
                out["split_child_original_prefit_physical_ridge_flag"] = int(bool(gate["original_prefit_physical_ridge_flag"]))
                out["subset_reason"] = (
                    "split56 child of v6 baselinev4 [2000,3000) x [5,6) fit cell; "
                    "central99/ridge-fraction gates passed with explicit split-child count threshold"
                )
                out["cell_role"] = "split56_drop4_fit"
            else:
                out["split_child_gate_exception"] = 0
                out["split_child_effective_min_mc_count"] = ""
                out["split_child_default_min_mc_count"] = ""
                out["split_child_original_prefit_physical_ridge_flag"] = ""
                out["subset_reason"] = "v6 baselinev4 drop4 fit cell preserved by (nhit_bin,predE_bin)"
                out["cell_role"] = "drop4_fit"
            out["exclusion_source"] = ""
        else:
            out["include"] = 0
            out["split_child_gate_exception"] = 0
            out["split_child_effective_min_mc_count"] = ""
            out["split_child_default_min_mc_count"] = ""
            out["split_child_original_prefit_physical_ridge_flag"] = ""
            if key == OLD_SPLIT_SOURCE_KEY:
                out["subset_reason"] = "replaced by split56 child bins"
                out["cell_role"] = "superseded_by_split56"
                out["exclusion_source"] = "split56_rebin"
            elif row.get("predE_bin") == ">=6":
                out["subset_reason"] = ">=6 tail retained in candidate cache but excluded from fit"
                out["cell_role"] = "diagnostic_tail"
                out["exclusion_source"] = "fit_predE_range"
            elif key in baseline_exclude_by_key:
                base = baseline_exclude_by_key[key]
                out["subset_reason"] = base.get("subset_reason") or row.get("subset_reason") or "excluded by v6 baselinev4 drop4 contract"
                out["cell_role"] = base.get("cell_role") or "excluded"
                out["exclusion_source"] = base.get("exclusion_source") or row.get("exclusion_source") or "MC_prefit_selector"
            elif str(row.get("predE_bin")) in {"[5,5.5)", "[5.5,6)"}:
                out["subset_reason"] = "split56 high-energy probe excluded outside preserved highest-Nhit baseline split"
                out["cell_role"] = "high_energy_probe"
                out["exclusion_source"] = "high_energy_probe"
            else:
                out["subset_reason"] = "excluded to preserve v6 baselinev4 drop4 cell-selection contract after split56 rebin"
                out["cell_role"] = row.get("cell_role") or "excluded"
                out["exclusion_source"] = row.get("exclusion_source") or "MC_prefit_selector"
        output_rows.append(out)

    included = [int(row["cell_id"]) for row in output_rows if truthy(row.get("include"))]
    included_keys = [bin_key(row) for row in output_rows if truthy(row.get("include"))]
    if len(included) != int(args.expected_fit_cells):
        raise SystemExit(f"included fit cells={len(included)}, expected {int(args.expected_fit_cells)}: {included_keys}")
    if set(included_keys) != desired_include_keys:
        raise SystemExit(f"included bin-key set mismatch: {included_keys}")

    extra_fields = [
        "fit_predE_range_flag",
        "tail_bin_flag",
        "split_child_gate_exception",
        "split_child_effective_min_mc_count",
        "split_child_default_min_mc_count",
        "split_child_original_prefit_physical_ridge_flag",
        "psf_systematic_variant",
        "psf_borrowed_from",
        "psf_borrow_method",
    ]
    fieldnames = list(input_fields)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(resolve(args.output_selector_csv), output_rows, fieldnames)

    print(
        json.dumps(
            {
                "status": "passed",
                "output_selector_csv": str(resolve(args.output_selector_csv)),
                "subset_version": str(args.subset_version),
                "candidate_cells": len(output_rows),
                "included_cell_ids": included,
                "included_bin_keys": [list(key) for key in included_keys],
                "split_child_cells": [
                    split_gate_records[key]
                    for key in NEW_SPLIT_KEYS
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
