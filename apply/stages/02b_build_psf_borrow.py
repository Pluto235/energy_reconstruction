#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_DIR = "apply/output/stage_b_v3_candidate/current"
DEFAULT_OUTPUT_DIR = "apply/output/stage_b_v3_candidate_psfborrow"
DEFAULT_SELECTOR_IN = "apply/config/cell_selector_v3_baseline.csv"
DEFAULT_SELECTOR_OUT = "apply/config/cell_selector_v3_baseline_psfborrow.csv"
RAYLEIGH_OPT_RADIUS_FACTOR = 1.58
RAYLEIGH_OPT_CONTAINMENT = 1.0 - math.exp(-0.5 * RAYLEIGH_OPT_RADIUS_FACTOR**2)

BORROW_SPECS = {
    39: {
        "sources": [40],
        "weights": [1.0],
        "method": "nearest_neighbor_borrow",
        "reason": "cell 39 is a physical-ridge left-shoulder fallback; use adjacent cell 40 PSF.",
    },
    52: {
        "sources": [53, 54],
        "weights": [2.0 / 3.0, 1.0 / 3.0],
        "method": "nearest_neighbor_weighted_interpolation",
        "reason": "cell 52 is a physical-ridge left-shoulder fallback; use a 2:1 weighted 53/54 neighbor PSF.",
    },
    65: {
        "sources": [66, 67],
        "weights": [2.0 / 3.0, 1.0 / 3.0],
        "method": "nearest_neighbor_weighted_interpolation",
        "reason": "cell 65 is a physical-ridge left-shoulder fallback; use a 2:1 weighted 66/67 neighbor PSF.",
    },
}

BORROW_ARRAY_KEYS = [
    "sigma_rad",
    "sigma_deg",
    "sigma_mc_weight_deg",
    "sigma_unweighted_deg",
    "sigma_full_rayleigh_rad",
    "sigma_full_rayleigh_deg",
    "r_opt_rad",
    "r_opt_deg",
    "containment_r_opt",
    "r68_deg",
    "r90_deg",
    "r95_deg",
    "core_r68_deg",
    "core_r90_deg",
    "core_r95_deg",
]

ROW_PSF_KEYS = [
    "sigma_rad",
    "sigma_deg",
    "sigma_mc_weight_deg",
    "sigma_unweighted_deg",
    "sigma_full_rayleigh_rad",
    "sigma_full_rayleigh_deg",
    "sigma_full_mc_weight_deg",
    "sigma_full_unweighted_deg",
    "r_opt_rad",
    "r_opt_deg",
    "containment_r_opt",
    "containment_r_opt_core_fit_full_distribution",
    "containment_minus_expected",
    "r68_deg",
    "r90_deg",
    "r95_deg",
    "core_r68_deg",
    "core_r90_deg",
    "core_r95_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v3 PSF-borrowing systematic Stage B artifact without overwriting nominal v3."
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-npz-name", type=str, default="psf_v3_candidate.npz")
    parser.add_argument("--input-metadata-name", type=str, default="psf_v3_candidate_metadata.json")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default="v3_psfborrow_from_nominal")
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--npz-name", type=str, default="psf_v3_candidate.npz")
    parser.add_argument("--metadata-name", type=str, default="psf_v3_candidate_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="psf_v3_candidate_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="psf_v3_candidate_summary.md")
    parser.add_argument("--selector-in", type=str, default=DEFAULT_SELECTOR_IN)
    parser.add_argument("--selector-out", type=str, default=DEFAULT_SELECTOR_OUT)
    return parser.parse_args()


def sanitize_run_id(run_id: str) -> str:
    keep = []
    for ch in str(run_id):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    value = "".join(keep).strip("._-")
    if not value:
        raise ValueError("Empty run id after sanitization.")
    return value


def prepare_run_dir(output_root: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_root / "runs" / sanitize_run_id(run_id)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def promote_current(output_root: Path, run_dir: Path) -> None:
    for name in ["current", "latest"]:
        link = output_root / name
        if link.is_symlink() or link.exists():
            if not link.is_symlink():
                raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
            link.unlink()
        os.symlink(run_dir.resolve(), link)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def original_missing_mass(row: Dict[str, object]) -> float | None:
    candidates: List[str] = []
    warnings = row.get("warnings")
    if isinstance(warnings, list):
        candidates.extend(str(value) for value in warnings)
    theta_reweight = row.get("theta_reweight")
    if isinstance(theta_reweight, dict):
        candidates.append(str(theta_reweight.get("reason", "")))
    for text in candidates:
        match = re.search(r"theta_missing_crab_probability_mass:([0-9.eE+-]+)", text)
        if match:
            value = finite_float(match.group(1))
            if value is not None:
                return value
    return finite_float(row.get("theta_missing_crab_probability_mass"))


def weighted_value(values: Sequence[object], weights: Sequence[float]) -> float | None:
    nums = [finite_float(value) for value in values]
    if any(value is None for value in nums):
        return None
    return float(sum(float(v) * w for v, w in zip(nums, weights)))


def load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def update_numeric_array(
    arrays: Dict[str, np.ndarray],
    key: str,
    target_index: int,
    source_indices: Sequence[int],
    weights: Sequence[float],
) -> float:
    values = np.asarray([arrays[key][idx] for idx in source_indices], dtype=np.float64)
    value = float(np.sum(values * np.asarray(weights, dtype=np.float64)))
    arrays[key][target_index] = np.asarray(value, dtype=arrays[key].dtype)
    return value


def source_record(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "cell_id": row.get("cell_id"),
        "nhit_bin": row.get("nhit_bin"),
        "predE_bin": row.get("predE_bin"),
        "theta_missing_crab_probability_mass": original_missing_mass(row),
        "effective_events": row.get("effective_events"),
        "core_fit_effective_events": row.get("core_fit_effective_events"),
        "sigma_deg": row.get("sigma_deg"),
        "r_opt_deg": row.get("r_opt_deg"),
        "containment_r_opt": row.get("containment_r_opt"),
    }


def append_warning(row: Dict[str, object], warning: str) -> None:
    warnings = row.get("warnings")
    if isinstance(warnings, list):
        if warning not in warnings:
            warnings.append(warning)
    else:
        row["warnings"] = [warning]


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "input_files",
        "events",
        "logE_range_events",
        "valid_events",
        "positive_baseline_weight_events",
        "sumw_baseline",
        "effective_events",
        "core_fit_events",
        "core_fit_effective_events",
        "core_fit_weight_fraction",
        "tail_weight_fraction_above_core_fit",
        "theta_missing_crab_probability_mass",
        "sigma_deg",
        "sigma_mc_weight_deg",
        "sigma_unweighted_deg",
        "sigma_full_rayleigh_deg",
        "r_opt_deg",
        "containment_r_opt",
        "r68_deg",
        "r90_deg",
        "r95_deg",
        "containment_warning",
        "angle_check_warning",
        "psf_borrowed",
        "borrowed_from",
        "borrow_method",
        "borrow_weights",
        "original_theta_missing_crab_probability_mass",
        "original_effective_events",
        "original_core_fit_effective_events",
        "original_sigma_deg",
        "original_r_opt_deg",
        "original_containment_r_opt",
        "borrowed_sigma_deg",
        "borrowed_r_opt_deg",
        "borrowed_containment_r_opt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary_md(path: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    records = metadata.get("psf_borrowing", {}).get("records", [])  # type: ignore[union-attr]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stage B v3 PSF Borrowing Systematic Summary\n\n")
        f.write("- Variant: `v3_psfborrow`\n")
        f.write(f"- Source Stage B: `{metadata['source_stage_b']['npz']}`\n")
        f.write(f"- Output NPZ: `{metadata['outputs']['npz']}`\n")
        f.write("- Nominal v3 Stage B is not overwritten; this run is a PSF systematic.\n\n")
        f.write("## Borrowed PSF Cells\n\n")
        f.write(
            "| cell | method | borrowed from | original missing mass | original Neff | "
            "borrowed sigma deg | borrowed r_opt deg | borrowed containment |\n"
        )
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for record in records:
            f.write(
                f"| {record['target_cell_id']} | {record['method']} | "
                f"{','.join(str(v) for v in record['borrowed_from'])} | "
                f"{record['original']['theta_missing_crab_probability_mass']} | "
                f"{record['original']['effective_events']} | "
                f"{record['borrowed']['sigma_deg']:.6g} | "
                f"{record['borrowed']['r_opt_deg']:.6g} | "
                f"{record['borrowed']['containment_r_opt']:.6g} |\n"
            )
        f.write("\n## Full Cell Table\n\n")
        f.write(
            "| cell | Nhit bin | predE bin | Neff | missing mass | sigma deg | r_opt deg | "
            "containment | borrowed |\n"
        )
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            f.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{float(row.get('effective_events') or 0.0):.6g} | "
                f"{float(row.get('theta_missing_crab_probability_mass') or 0.0):.6g} | "
                f"{float(row.get('sigma_deg') or 0.0):.6g} | "
                f"{float(row.get('r_opt_deg') or 0.0):.6g} | "
                f"{float(row.get('containment_r_opt') or 0.0):.6g} | "
                f"{row.get('borrowed_from', '')} |\n"
            )


def build_selector(selector_in: Path, selector_out: Path, records: Sequence[Dict[str, object]]) -> None:
    with selector_in.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        base_fields = list(reader.fieldnames or [])
    extra_fields = ["psf_systematic_variant", "psf_borrowed_from", "psf_borrow_method"]
    fields = base_fields + [field for field in extra_fields if field not in base_fields]
    record_by_cell = {int(record["target_cell_id"]): record for record in records}

    for row in rows:
        cell_id = int(row["cell_id"])
        row["subset_version"] = "v3_baseline_psfborrow"
        if cell_id in record_by_cell:
            record = record_by_cell[cell_id]
            row["include"] = "1"
            row["psf_quality_flag"] = "1"
            row["cell_role"] = "baseline_fit"
            row["exclusion_source"] = ""
            row["subset_reason"] = (
                "central99 MC-occupancy-ridge prefit cell; PSF borrowed for v3_psfborrow systematic "
                f"from {','.join(str(v) for v in record['borrowed_from'])}"
            )
            row["psf_systematic_variant"] = "v3_psfborrow"
            row["psf_borrowed_from"] = ",".join(str(v) for v in record["borrowed_from"])
            row["psf_borrow_method"] = str(record["method"])
        else:
            row.setdefault("psf_systematic_variant", "")
            row.setdefault("psf_borrowed_from", "")
            row.setdefault("psf_borrow_method", "")
        if cell_id in {79, 80}:
            row["include"] = "0"
            row["cell_role"] = "excluded"
            row["exclusion_source"] = "MC_prefit_selector"
            row["subset_reason"] = (
                "excluded by high-Nhit edge low-stat/PSF-untrusted prefit rule in v3_psfborrow"
            )

    included = [int(row["cell_id"]) for row in rows if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}]
    for required in [39, 52, 65]:
        if required not in included:
            raise ValueError(f"psfborrow selector did not include repaired cell {required}")
    for excluded in [79, 80]:
        if excluded in included:
            raise ValueError(f"psfborrow selector unexpectedly includes high-Nhit edge cell {excluded}")

    selector_out.parent.mkdir(parents=True, exist_ok=True)
    with selector_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    input_dir = Path(args.input_dir).resolve()
    input_npz = input_dir / args.input_npz_name
    input_metadata = input_dir / args.input_metadata_name
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = prepare_run_dir(output_root, args.run_id, overwrite=bool(args.overwrite_run_dir))

    if not input_npz.exists():
        raise FileNotFoundError(f"Input PSF NPZ does not exist: {input_npz}")
    if not input_metadata.exists():
        raise FileNotFoundError(f"Input PSF metadata does not exist: {input_metadata}")

    arrays = load_npz_arrays(input_npz)
    metadata = json.loads(input_metadata.read_text(encoding="utf-8"))
    cell_ids = np.asarray(arrays["cell_id"], dtype=np.int64)
    index_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids.tolist())}
    for target, spec in BORROW_SPECS.items():
        if target not in index_by_cell:
            raise ValueError(f"Target cell {target} is absent from {input_npz}")
        for source in spec["sources"]:
            if int(source) not in index_by_cell:
                raise ValueError(f"Source cell {source} for target {target} is absent from {input_npz}")

    original_cells = metadata.get("cells") if isinstance(metadata.get("cells"), list) else []
    row_by_cell = {int(row["cell_id"]): deepcopy(row) for row in original_cells if isinstance(row, dict)}
    rows = [deepcopy(row_by_cell[int(cell_id)]) for cell_id in cell_ids.tolist()]
    row_pos_by_cell = {int(row["cell_id"]): idx for idx, row in enumerate(rows)}

    records: List[Dict[str, object]] = []
    for target, spec in BORROW_SPECS.items():
        target_index = index_by_cell[target]
        source_indices = [index_by_cell[int(source)] for source in spec["sources"]]
        weights = [float(weight) for weight in spec["weights"]]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError(f"Invalid non-positive borrow weights for cell {target}: {weights}")
        weights = [weight / weight_sum for weight in weights]

        original_row = row_by_cell[target]
        source_rows = [row_by_cell[int(source)] for source in spec["sources"]]
        borrowed_values: Dict[str, object] = {}
        for key in BORROW_ARRAY_KEYS:
            borrowed_values[key] = update_numeric_array(arrays, key, target_index, source_indices, weights)
        if "profile_density" in arrays:
            profile = np.zeros_like(arrays["profile_density"][target_index], dtype=np.float64)
            for source_index, weight in zip(source_indices, weights):
                profile += np.asarray(arrays["profile_density"][source_index], dtype=np.float64) * weight
            arrays["profile_density"][target_index] = profile.astype(arrays["profile_density"].dtype)

        target_row = rows[row_pos_by_cell[target]]
        target_row["psf_borrowed"] = True
        target_row["borrowed_from"] = ",".join(str(source) for source in spec["sources"])
        target_row["borrow_method"] = str(spec["method"])
        target_row["borrow_weights"] = ",".join(f"{weight:.8g}" for weight in weights)
        target_row["psf_systematic_variant"] = "v3_psfborrow"
        target_row["active_psf_source"] = "borrowed_neighbor_psf"
        actual_original_missing = original_missing_mass(original_row)
        target_row["theta_missing_crab_probability_mass"] = actual_original_missing
        arrays["theta_missing_crab_probability_mass"][target_index] = np.asarray(
            actual_original_missing if actual_original_missing is not None else 1.0,
            dtype=arrays["theta_missing_crab_probability_mass"].dtype,
        )
        target_row["original_theta_missing_crab_probability_mass"] = actual_original_missing
        target_row["original_effective_events"] = original_row.get("effective_events")
        target_row["original_core_fit_effective_events"] = original_row.get("core_fit_effective_events")
        target_row["original_sigma_deg"] = original_row.get("sigma_deg")
        target_row["original_r_opt_deg"] = original_row.get("r_opt_deg")
        target_row["original_containment_r_opt"] = original_row.get("containment_r_opt")
        target_row["borrowed_sigma_deg"] = borrowed_values["sigma_deg"]
        target_row["borrowed_r_opt_deg"] = borrowed_values["r_opt_deg"]
        target_row["borrowed_containment_r_opt"] = borrowed_values["containment_r_opt"]
        target_row["borrow_source_cells"] = [source_record(row) for row in source_rows]
        for key in ROW_PSF_KEYS:
            if key in target_row:
                value = weighted_value([row.get(key) for row in source_rows], weights)
                if value is not None:
                    target_row[key] = value
        target_row["r_opt_factor"] = float(RAYLEIGH_OPT_RADIUS_FACTOR)
        target_row["rayleigh_expected_containment_r_opt"] = float(RAYLEIGH_OPT_CONTAINMENT)
        containment = finite_float(target_row.get("containment_r_opt"))
        target_row["containment_minus_expected"] = (
            float(containment - RAYLEIGH_OPT_CONTAINMENT) if containment is not None else None
        )
        tolerance = (
            metadata.get("quality_thresholds", {}).get("containment_warning_tolerance", 0.12)
            if isinstance(metadata.get("quality_thresholds"), dict)
            else 0.12
        )
        target_row["containment_warning"] = (
            abs(float(target_row["containment_minus_expected"])) > float(tolerance)
            if target_row.get("containment_minus_expected") is not None
            else True
        )
        target_row["psf_quality_flag"] = "borrowed"
        append_warning(target_row, f"psf_borrowed_from:{target_row['borrowed_from']}")

        record = {
            "target_cell_id": int(target),
            "method": str(spec["method"]),
            "reason": str(spec["reason"]),
            "borrowed_from": [int(source) for source in spec["sources"]],
            "weights": {str(source): float(weight) for source, weight in zip(spec["sources"], weights)},
            "original": source_record(original_row),
            "borrowed": {
                "sigma_deg": float(borrowed_values["sigma_deg"]),
                "r_opt_deg": float(borrowed_values["r_opt_deg"]),
                "containment_r_opt": float(borrowed_values["containment_r_opt"]),
                "sigma_rad": float(borrowed_values["sigma_rad"]),
                "r_opt_rad": float(borrowed_values["r_opt_rad"]),
            },
            "sources": [source_record(row) for row in source_rows],
        }
        records.append(record)

    for row in rows:
        row.setdefault("psf_borrowed", False)
        row.setdefault("borrowed_from", "")
        row.setdefault("borrow_method", "")
        row.setdefault("borrow_weights", "")

    npz_path = run_dir / args.npz_name
    np.savez_compressed(npz_path, **arrays)

    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    source_stage_b = {
        "npz": str(input_npz),
        "metadata_json": str(input_metadata),
        "run_id": metadata.get("run_id"),
        "slurm_job_id": metadata.get("slurm_job_id"),
    }
    out_metadata = deepcopy(metadata)
    out_metadata.update(
        {
            "description": "Stage B v3 PSF borrowing/interpolation systematic; not the nominal v3 PSF.",
            "run_id": sanitize_run_id(args.run_id),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_stage_b": source_stage_b,
            "psf_systematic_variant": "v3_psfborrow",
            "nominal_v3_stage_b_overwritten": False,
            "output_root": str(output_root),
            "output_dir": str(run_dir),
            "current_dir": str(output_root / "current"),
            "latest": str(output_root / "latest"),
            "cells": rows,
            "psf_borrowing": {
                "status": "applied",
                "variant": "v3_psfborrow",
                "active_parameter_policy": (
                    "Replace active sigma/r_opt/containment and radial-profile PSF shape for fallback cells; "
                    "preserve original target-cell support diagnostics as original_* audit fields."
                ),
                "target_cells": sorted(BORROW_SPECS),
                "records": records,
            },
            "elapsed_seconds": float(time.perf_counter() - start),
            "promotion": {
                "promote_current": not bool(args.no_promote_current),
                "status": "pending",
            },
            "outputs": {
                "npz": str(npz_path),
                "metadata_json": str(metadata_path),
                "summary_csv": str(summary_csv_path),
                "summary_md": str(summary_md_path),
                "selector_csv": str(Path(args.selector_out).resolve()),
            },
        }
    )
    warning_rows = out_metadata.get("warning_rows") if isinstance(out_metadata.get("warning_rows"), list) else []
    warning_by_cell = {int(row.get("cell_id")): dict(row) for row in warning_rows if isinstance(row, dict) and row.get("cell_id") is not None}
    for record in records:
        cell_id = int(record["target_cell_id"])
        warning = warning_by_cell.setdefault(cell_id, {"cell_id": cell_id})
        warning["psf_borrowed"] = True
        warning["borrowed_from"] = ",".join(str(v) for v in record["borrowed_from"])
        warning["missing_crab_probability_mass"] = record["original"].get("theta_missing_crab_probability_mass")  # type: ignore[union-attr]
    out_metadata["warning_rows"] = list(warning_by_cell.values())

    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, out_metadata, rows)
    build_selector(Path(args.selector_in).resolve(), Path(args.selector_out).resolve(), records)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(out_metadata), f, indent=2)

    if not args.no_promote_current:
        promote_current(output_root, run_dir)
        out_metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        out_metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        out_metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(json_ready(out_metadata), f, indent=2)
    else:
        out_metadata["promotion"]["status"] = "skipped_no_promote_current"  # type: ignore[index]
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(json_ready(out_metadata), f, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {Path(args.selector_out).resolve()}")
    if not args.no_promote_current:
        print(f"Promoted {output_root / 'current'}")


if __name__ == "__main__":
    main()
