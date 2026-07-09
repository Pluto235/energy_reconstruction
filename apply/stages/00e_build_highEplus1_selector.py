#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PRED_BINS = [
    "[2,2.5)",
    "[2.5,3)",
    "[3,3.25)",
    "[3.25,3.5)",
    "[3.5,3.75)",
    "[3.75,4.0)",
    "[4.0,4.25)",
    "[4.25,4.5)",
    "[4.5,4.75)",
    "[4.75,5.0)",
    "[5,5.5)",
    "[5.5,6)",
    ">=6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v6 highEplus1 fit selector: preserve MC-ridge cells, "
            "then test only the next higher predE bin in each Nhit band."
        )
    )
    parser.add_argument("--prefit-selector-csv", type=Path, required=True)
    parser.add_argument("--candidate-ledger-csv", type=Path, required=True)
    parser.add_argument("--stage-b-summary-csv", type=Path, required=True)
    parser.add_argument("--output-selector-csv", type=Path, required=True)
    parser.add_argument("--decision-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--subset-version", type=str, default="v6_64748_nhit100_highEplus1_split56")
    parser.add_argument("--min-mc-count", type=int, default=1000)
    parser.add_argument("--min-psf-effective-events", type=float, default=200.0)
    parser.add_argument("--min-core-fit-effective-events", type=float, default=200.0)
    parser.add_argument("--max-theta-missing-mass", type=float, default=0.10)
    parser.add_argument("--allow-containment-warning", action="store_true", default=False)
    parser.add_argument("--allow-angle-warning", action="store_true", default=False)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def pred_index(label: str) -> int:
    try:
        return PRED_BINS.index(str(label))
    except ValueError as exc:
        raise ValueError(f"Unexpected predE bin label: {label!r}") from exc


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row["nhit_bin"]), str(row["predE_bin"])


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def psf_quality(
    row: Dict[str, str] | None,
    *,
    min_psf_effective_events: float,
    min_core_fit_effective_events: float,
    max_theta_missing_mass: float,
    allow_containment_warning: bool,
    allow_angle_warning: bool,
) -> tuple[bool, str]:
    reasons: List[str] = []
    if row is None:
        return False, "missing_stage_b_summary"

    effective_events = as_float(row.get("effective_events"), default=0.0)
    core_fit_effective_events = as_float(row.get("core_fit_effective_events"), default=0.0)
    valid_events = as_int(row.get("valid_events"), default=0)
    missing_mass = as_float(row.get("theta_missing_crab_probability_mass"), default=1.0)
    containment_warning = truthy(row.get("containment_warning"))
    angle_warning = truthy(row.get("angle_check_warning"))

    if valid_events <= 0:
        reasons.append("valid_events_le_0")
    if effective_events < float(min_psf_effective_events):
        reasons.append(f"effective_events_lt_{float(min_psf_effective_events):g}")
    if core_fit_effective_events < float(min_core_fit_effective_events):
        reasons.append(f"core_fit_effective_events_lt_{float(min_core_fit_effective_events):g}")
    if missing_mass > float(max_theta_missing_mass):
        reasons.append(f"theta_missing_mass_gt_{float(max_theta_missing_mass):g}")
    if containment_warning and not allow_containment_warning:
        reasons.append("containment_warning")
    if angle_warning and not allow_angle_warning:
        reasons.append("angle_check_warning")
    return not reasons, ";".join(reasons) if reasons else "pass"


def main() -> None:
    args = parse_args()
    prefit_rows, prefit_fields = read_csv(resolve(args.prefit_selector_csv))
    ledger_rows, _ = read_csv(resolve(args.candidate_ledger_csv))
    psf_rows, _ = read_csv(resolve(args.stage_b_summary_csv))
    if len(prefit_rows) != len(ledger_rows):
        raise SystemExit(f"prefit selector rows={len(prefit_rows)} but ledger rows={len(ledger_rows)}")

    ledger_by_key = {row_key(row): row for row in ledger_rows}
    psf_by_key = {row_key(row): row for row in psf_rows}
    rows_by_key = {row_key(row): row for row in prefit_rows}
    if set(rows_by_key) != set(ledger_by_key):
        missing_from_prefit = sorted(set(ledger_by_key) - set(rows_by_key))
        missing_from_ledger = sorted(set(rows_by_key) - set(ledger_by_key))
        raise SystemExit(
            "candidate ledger / prefit key mismatch: "
            f"missing_from_prefit={missing_from_prefit}, missing_from_ledger={missing_from_ledger}"
        )

    by_nhit: Dict[str, List[Dict[str, str]]] = {}
    for row in prefit_rows:
        by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)

    base_keys = {
        row_key(row)
        for row in prefit_rows
        if truthy(row.get("include")) and str(row.get("predE_bin")) != ">=6"
    }
    extension_keys: set[Tuple[str, str]] = set()
    decision_rows: List[Dict[str, Any]] = []

    for nhit, rows in sorted(by_nhit.items(), key=lambda item: as_float(item[0].strip("[)").split(",", 1)[0].replace(">=", "999999"), 0.0)):
        base_for_nhit = sorted(
            [row for row in rows if row_key(row) in base_keys],
            key=lambda row: pred_index(str(row["predE_bin"])),
        )
        if not base_for_nhit:
            decision_rows.append(
                {
                    "nhit_bin": nhit,
                    "status": "no_original_ridge_cell",
                    "candidate_cell_id": "",
                    "candidate_predE_bin": "",
                    "mc_count": "",
                    "psf_quality_flag": 0,
                    "psf_quality_reasons": "no_base_ridge",
                }
            )
            continue
        highest = max(pred_index(str(row["predE_bin"])) for row in base_for_nhit)
        next_index = highest + 1
        if next_index >= len(PRED_BINS):
            decision_rows.append(
                {
                    "nhit_bin": nhit,
                    "status": "no_higher_predE_bin",
                    "candidate_cell_id": "",
                    "candidate_predE_bin": "",
                    "mc_count": "",
                    "psf_quality_flag": 0,
                    "psf_quality_reasons": "already_at_tail",
                }
            )
            continue
        candidate_key = (nhit, PRED_BINS[next_index])
        candidate = rows_by_key.get(candidate_key)
        if candidate is None:
            decision_rows.append(
                {
                    "nhit_bin": nhit,
                    "status": "candidate_missing_from_grid",
                    "candidate_cell_id": "",
                    "candidate_predE_bin": PRED_BINS[next_index],
                    "mc_count": "",
                    "psf_quality_flag": 0,
                    "psf_quality_reasons": "missing_candidate_row",
                }
            )
            continue
        if candidate_key[1] == ">=6":
            decision_rows.append(
                {
                    "nhit_bin": nhit,
                    "status": "diagnostic_tail_only",
                    "candidate_cell_id": candidate.get("cell_id"),
                    "candidate_predE_bin": candidate_key[1],
                    "mc_count": candidate.get("mc_count"),
                    "psf_quality_flag": 0,
                    "psf_quality_reasons": "tail_ge6_excluded_from_main_fit",
                }
            )
            continue

        mc_count = as_int(candidate.get("mc_count"), default=0)
        mc_pass = mc_count >= int(args.min_mc_count)
        psf_pass, psf_reasons = psf_quality(
            psf_by_key.get(candidate_key),
            min_psf_effective_events=float(args.min_psf_effective_events),
            min_core_fit_effective_events=float(args.min_core_fit_effective_events),
            max_theta_missing_mass=float(args.max_theta_missing_mass),
            allow_containment_warning=bool(args.allow_containment_warning),
            allow_angle_warning=bool(args.allow_angle_warning),
        )
        if mc_pass and psf_pass:
            extension_keys.add(candidate_key)
            status = "included_highEplus1"
        else:
            status = "rejected_highEplus1_probe"
        reasons = []
        if not mc_pass:
            reasons.append(f"mc_count_lt_{int(args.min_mc_count)}")
        if not psf_pass:
            reasons.append(psf_reasons)
        decision_rows.append(
            {
                "nhit_bin": nhit,
                "status": status,
                "candidate_cell_id": candidate.get("cell_id"),
                "candidate_predE_bin": candidate_key[1],
                "mc_count": mc_count,
                "psf_quality_flag": int(psf_pass),
                "psf_quality_reasons": ";".join(reason for reason in reasons if reason) or "pass",
            }
        )

    output_rows: List[Dict[str, Any]] = []
    for row in sorted(prefit_rows, key=lambda item: as_int(item.get("cell_id"), 0)):
        key = row_key(row)
        out: Dict[str, Any] = dict(row)
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

        psf_pass, psf_reasons = psf_quality(
            psf_by_key.get(key),
            min_psf_effective_events=float(args.min_psf_effective_events),
            min_core_fit_effective_events=float(args.min_core_fit_effective_events),
            max_theta_missing_mass=float(args.max_theta_missing_mass),
            allow_containment_warning=bool(args.allow_containment_warning),
            allow_angle_warning=bool(args.allow_angle_warning),
        )
        original_ridge = key in base_keys
        extension_candidate = any(
            decision.get("candidate_cell_id") == row.get("cell_id")
            for decision in decision_rows
            if str(decision.get("status", "")).endswith("highEplus1")
            or str(decision.get("status", "")).endswith("probe")
        )
        extension_included = key in extension_keys
        tail = str(row.get("predE_bin")) == ">=6"

        out["subset_version"] = str(args.subset_version)
        out["original_ridge_fit_flag"] = int(original_ridge)
        out["highEplus1_candidate_flag"] = int(extension_candidate)
        out["highEplus1_included_flag"] = int(extension_included)
        out["highEplus1_rejected_flag"] = int(extension_candidate and not extension_included)
        out["psf_quality_flag"] = int(psf_pass)
        out["psf_quality_reasons"] = psf_reasons

        if original_ridge:
            out["include"] = 1
            out["subset_reason"] = "original MC-ridge fit cell preserved"
            out["cell_role"] = "mc_ridge_fit"
            out["exclusion_source"] = ""
        elif extension_included:
            out["include"] = 1
            out["subset_reason"] = "one-bin high-predE extension passed MC statistics and Stage B PSF quality"
            out["cell_role"] = "highEplus1_fit"
            out["exclusion_source"] = ""
            out["physical_ridge_flag"] = out.get("physical_ridge_flag") or 0
        elif extension_candidate:
            matching = [decision for decision in decision_rows if decision.get("candidate_cell_id") == row.get("cell_id")]
            reason = str(matching[0].get("psf_quality_reasons", "highEplus1_gate_failed")) if matching else "highEplus1_gate_failed"
            out["include"] = 0
            out["subset_reason"] = f"one-bin high-predE probe rejected from fit: {reason}"
            out["cell_role"] = "highEplus1_probe_rejected"
            out["exclusion_source"] = reason
        elif tail:
            out["include"] = 0
            out["subset_reason"] = ">=6 diagnostic tail retained in Stage A-E products but excluded from Stage F/G main fit"
            out["cell_role"] = "diagnostic_tail"
            out["exclusion_source"] = "fit_predE_range"
        else:
            out["include"] = 0
            out["subset_reason"] = "not on original MC ridge and not the one-bin high-predE extension"
            out["cell_role"] = "excluded"
            out["exclusion_source"] = "MC_ridge_highEplus1_selector"
        output_rows.append(out)

    included_tail = [row for row in output_rows if truthy(row.get("include")) and str(row.get("predE_bin")) == ">=6"]
    if included_tail:
        raise SystemExit(f"selector illegally includes >=6 diagnostic tail rows: {included_tail}")

    included_rows = [row for row in output_rows if truthy(row.get("include"))]
    extension_included_rows = [row for row in output_rows if truthy(row.get("highEplus1_included_flag"))]
    extension_rejected_rows = [row for row in output_rows if truthy(row.get("highEplus1_rejected_flag"))]
    tail_rows = [row for row in output_rows if str(row.get("predE_bin")) == ">=6"]

    extra_fields = [
        "original_ridge_fit_flag",
        "highEplus1_candidate_flag",
        "highEplus1_included_flag",
        "highEplus1_rejected_flag",
        "psf_quality_reasons",
    ]
    fieldnames = list(prefit_fields)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(resolve(args.output_selector_csv), output_rows, fieldnames)
    write_csv(
        resolve(args.decision_csv),
        decision_rows,
        [
            "nhit_bin",
            "status",
            "candidate_cell_id",
            "candidate_predE_bin",
            "mc_count",
            "psf_quality_flag",
            "psf_quality_reasons",
        ],
    )

    metadata = {
        "selector": str(resolve(args.output_selector_csv)),
        "prefit_selector_csv": str(resolve(args.prefit_selector_csv)),
        "candidate_ledger_csv": str(resolve(args.candidate_ledger_csv)),
        "stage_b_summary_csv": str(resolve(args.stage_b_summary_csv)),
        "subset_version": str(args.subset_version),
        "rule": "preserve original MC-ridge fit cells; add at most one adjacent higher predE bin per Nhit; never extend lower; exclude >=6 tail from Stage F/G",
        "thresholds": {
            "min_mc_count": int(args.min_mc_count),
            "min_psf_effective_events": float(args.min_psf_effective_events),
            "min_core_fit_effective_events": float(args.min_core_fit_effective_events),
            "max_theta_missing_mass": float(args.max_theta_missing_mass),
            "allow_containment_warning": bool(args.allow_containment_warning),
            "allow_angle_warning": bool(args.allow_angle_warning),
        },
        "candidate_cells": len(output_rows),
        "included_cells": len(included_rows),
        "original_ridge_fit_cells": sum(1 for row in output_rows if truthy(row.get("original_ridge_fit_flag"))),
        "highEplus1_included_cells": len(extension_included_rows),
        "highEplus1_rejected_cells": len(extension_rejected_rows),
        "diagnostic_tail_cells": len(tail_rows),
        "included_cell_ids": [as_int(row.get("cell_id")) for row in included_rows],
        "highEplus1_included": [
            {"cell_id": as_int(row.get("cell_id")), "nhit_bin": row.get("nhit_bin"), "predE_bin": row.get("predE_bin")}
            for row in extension_included_rows
        ],
        "highEplus1_rejected": [
            {
                "cell_id": as_int(row.get("cell_id")),
                "nhit_bin": row.get("nhit_bin"),
                "predE_bin": row.get("predE_bin"),
                "reason": row.get("exclusion_source"),
            }
            for row in extension_rejected_rows
        ],
        "tail_policy": ">=6 diagnostic only; no Stage F/G include rows",
        "decisions": decision_rows,
    }
    resolve(args.metadata_json).parent.mkdir(parents=True, exist_ok=True)
    resolve(args.metadata_json).write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(metadata), indent=2))


if __name__ == "__main__":
    main()
