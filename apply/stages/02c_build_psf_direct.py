#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from copy import deepcopy
from typing import Dict, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_DIR = "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal"
DEFAULT_DIRECT_DIR = (
    "apply/output/stage_b_v3_candidate_direct_ownpsf_focus/runs/"
    "v3_stage_b_direct_ownpsf_focus_39_52_65"
)
DEFAULT_OUTPUT_DIR = "apply/output/stage_b_v3_candidate_directpsf"
DEFAULT_SELECTOR_IN = "apply/config/cell_selector_v3_baseline_psfborrow.csv"
DEFAULT_SELECTOR_OUT = "apply/config/cell_selector_v3_baseline_directpsf.csv"
DIRECT_CELL_IDS = (39, 52, 65)

ARRAY_KEYS = [
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
    "effective_events",
    "core_fit_effective_events",
    "core_fit_weight_fraction",
    "tail_weight_fraction_above_core_fit",
    "theta_missing_crab_probability_mass",
    "events",
    "sumw_baseline",
]

ROW_KEYS = [
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
    "core_r68_deg",
    "core_r90_deg",
    "core_r95_deg",
    "containment_warning",
    "angle_check_warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full v3 Stage B PSF artifact using direct own-cell PSFs for selected cells."
    )
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--base-npz-name", type=str, default="psf_v3_candidate.npz")
    parser.add_argument("--base-metadata-name", type=str, default="psf_v3_candidate_metadata.json")
    parser.add_argument("--direct-dir", type=str, default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--direct-npz-name", type=str, default="psf_v3_candidate_direct_ownpsf_focus.npz")
    parser.add_argument("--direct-metadata-name", type=str, default="psf_v3_candidate_direct_ownpsf_focus_metadata.json")
    parser.add_argument("--direct-summary-csv-name", type=str, default="psf_v3_candidate_direct_ownpsf_focus_summary.csv")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default="v3_directpsf_from_psfborrow")
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--npz-name", type=str, default="psf_v3_candidate.npz")
    parser.add_argument("--metadata-name", type=str, default="psf_v3_candidate_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="psf_v3_candidate_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="psf_v3_candidate_summary.md")
    parser.add_argument("--selector-in", type=str, default=DEFAULT_SELECTOR_IN)
    parser.add_argument("--selector-out", type=str, default=DEFAULT_SELECTOR_OUT)
    parser.add_argument("--direct-cell-ids", type=str, default=",".join(str(v) for v in DIRECT_CELL_IDS))
    return parser.parse_args()


def path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def sanitize_run_id(run_id: str) -> str:
    keep = []
    for ch in str(run_id):
        keep.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
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


def load_npz_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_cell_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def append_warning(row: Dict[str, object], warning: str) -> None:
    warnings = row.get("warnings")
    if isinstance(warnings, list):
        if warning not in warnings:
            warnings.append(warning)
    else:
        row["warnings"] = [warning]


def write_summary_csv(path_: Path, rows: Sequence[Dict[str, object]]) -> None:
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
        "psf_direct_own_cell",
        "direct_source_run",
        "previous_psf_source",
        "previous_sigma_deg",
        "previous_r_opt_deg",
        "previous_containment_r_opt",
        "direct_sigma_deg",
        "direct_r_opt_deg",
        "direct_containment_r_opt",
    ]
    with path_.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary_md(path_: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    direct = metadata.get("psf_direct_own_cell", {}) if isinstance(metadata.get("psf_direct_own_cell"), dict) else {}
    records = direct.get("records", []) if isinstance(direct.get("records"), list) else []
    with path_.open("w", encoding="utf-8") as handle:
        handle.write("# Stage B v3 Direct Own-cell PSF Summary\n\n")
        handle.write("- Variant: `v3_directpsf`\n")
        handle.write(f"- Base Stage B: `{metadata['source_stage_b']['npz']}`\n")
        handle.write(f"- Direct focus Stage B: `{metadata['direct_focus_stage_b']['npz']}`\n")
        handle.write(f"- Output NPZ: `{metadata['outputs']['npz']}`\n\n")
        handle.write("## Direct Replaced Cells\n\n")
        handle.write(
            "| cell | previous source | previous sigma | direct sigma | previous r_opt | direct r_opt | "
            "missing mass | direct Neff |\n"
        )
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for record in records:
            handle.write(
                f"| {record['cell_id']} | {record['previous_psf_source']} | "
                f"{record['previous']['sigma_deg']:.6g} | {record['direct']['sigma_deg']:.6g} | "
                f"{record['previous']['r_opt_deg']:.6g} | {record['direct']['r_opt_deg']:.6g} | "
                f"{record['direct']['theta_missing_crab_probability_mass']:.6g} | "
                f"{record['direct']['effective_events']:.6g} |\n"
            )
        handle.write("\n## Full Cell Table\n\n")
        handle.write("| cell | Nhit bin | predE bin | Neff | missing mass | sigma deg | r_opt deg | containment | direct |\n")
        handle.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{float(row.get('effective_events') or 0.0):.6g} | "
                f"{float(row.get('theta_missing_crab_probability_mass') or 0.0):.6g} | "
                f"{float(row.get('sigma_deg') or 0.0):.6g} | "
                f"{float(row.get('r_opt_deg') or 0.0):.6g} | "
                f"{float(row.get('containment_r_opt') or 0.0):.6g} | "
                f"{row.get('psf_direct_own_cell', False)} |\n"
            )


def build_selector(selector_in: Path, selector_out: Path, direct_cell_ids: Sequence[int]) -> None:
    with selector_in.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        base_fields = list(reader.fieldnames or [])
    extra_fields = ["psf_systematic_variant", "psf_direct_own_cell", "psf_borrowed_from", "psf_borrow_method"]
    fields = base_fields + [field for field in extra_fields if field not in base_fields]
    direct_set = {int(v) for v in direct_cell_ids}
    for row in rows:
        cell_id = int(row["cell_id"])
        row["subset_version"] = "v3_baseline_directpsf"
        if cell_id in direct_set:
            row["include"] = "1"
            row["psf_quality_flag"] = "1"
            row["cell_role"] = "baseline_fit"
            row["exclusion_source"] = ""
            row["subset_reason"] = (
                "central99 MC-occupancy-ridge prefit cell; direct own-cell PSF used despite "
                "incomplete theta support"
            )
            row["psf_systematic_variant"] = "v3_directpsf"
            row["psf_direct_own_cell"] = "1"
            row["psf_borrowed_from"] = ""
            row["psf_borrow_method"] = ""
        else:
            row["psf_direct_own_cell"] = "0"
            row.setdefault("psf_systematic_variant", "")
            row.setdefault("psf_borrowed_from", "")
            row.setdefault("psf_borrow_method", "")
    included = [
        int(row["cell_id"])
        for row in rows
        if str(row.get("include", "")).strip().lower() in {"1", "true", "yes", "y", "include"}
    ]
    for required in direct_set:
        if required not in included:
            raise ValueError(f"directpsf selector did not include direct cell {required}")
    selector_out.parent.mkdir(parents=True, exist_ok=True)
    with selector_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_dir = path(args.base_dir)
    direct_dir = path(args.direct_dir)
    output_root = path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = prepare_run_dir(output_root, args.run_id, overwrite=bool(args.overwrite_run_dir))

    base_npz = base_dir / args.base_npz_name
    base_metadata_path = base_dir / args.base_metadata_name
    direct_npz = direct_dir / args.direct_npz_name
    direct_metadata_path = direct_dir / args.direct_metadata_name
    direct_summary_csv = direct_dir / args.direct_summary_csv_name
    for required in [base_npz, base_metadata_path, direct_npz, direct_metadata_path, direct_summary_csv]:
        if not required.exists():
            raise FileNotFoundError(f"Required input does not exist: {required}")

    arrays = load_npz_arrays(base_npz)
    direct_arrays = load_npz_arrays(direct_npz)
    metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    direct_metadata = json.loads(direct_metadata_path.read_text(encoding="utf-8"))
    direct_rows = {int(row["cell_id"]): row for row in read_csv_rows(direct_summary_csv)}
    direct_cell_ids = parse_cell_ids(args.direct_cell_ids)
    base_ids = np.asarray(arrays["cell_id"], dtype=np.int64)
    direct_ids = np.asarray(direct_arrays["cell_id"], dtype=np.int64)
    base_index = {int(cell_id): idx for idx, cell_id in enumerate(base_ids.tolist())}
    direct_index = {int(cell_id): idx for idx, cell_id in enumerate(direct_ids.tolist())}
    for cell_id in direct_cell_ids:
        if cell_id not in base_index:
            raise ValueError(f"Cell {cell_id} missing from base PSF")
        if cell_id not in direct_index:
            raise ValueError(f"Cell {cell_id} missing from direct focus PSF")
        if cell_id not in direct_rows:
            raise ValueError(f"Cell {cell_id} missing from direct summary CSV")

    source_cells = metadata.get("cells") if isinstance(metadata.get("cells"), list) else []
    row_by_cell = {int(row["cell_id"]): deepcopy(row) for row in source_cells if isinstance(row, dict)}
    rows = [deepcopy(row_by_cell[int(cell_id)]) for cell_id in base_ids.tolist()]
    row_index = {int(row["cell_id"]): idx for idx, row in enumerate(rows)}
    records = []

    for cell_id in direct_cell_ids:
        bi = base_index[cell_id]
        di = direct_index[cell_id]
        previous = {
            "sigma_deg": float(arrays["sigma_deg"][bi]),
            "r_opt_deg": float(arrays["r_opt_deg"][bi]),
            "containment_r_opt": float(arrays["containment_r_opt"][bi]),
            "source": str(rows[row_index[cell_id]].get("borrowed_from") or "base"),
        }
        for key in ARRAY_KEYS:
            if key in arrays and key in direct_arrays:
                arrays[key][bi] = np.asarray(direct_arrays[key][di], dtype=arrays[key].dtype)
        if "profile_density" in arrays and "profile_density" in direct_arrays:
            arrays["profile_density"][bi] = direct_arrays["profile_density"][di].astype(arrays["profile_density"].dtype)

        row = rows[row_index[cell_id]]
        direct_row = direct_rows[cell_id]
        for key in ROW_KEYS:
            if key in direct_row:
                row[key] = direct_row[key]
        row["psf_borrowed"] = False
        row["borrowed_from"] = ""
        row["borrow_method"] = ""
        row["borrow_weights"] = ""
        row["psf_direct_own_cell"] = True
        row["direct_source_run"] = direct_metadata.get("run_id", "")
        row["active_psf_source"] = "direct_own_cell_psf"
        row["psf_systematic_variant"] = "v3_directpsf"
        row["previous_psf_source"] = previous["source"]
        row["previous_sigma_deg"] = previous["sigma_deg"]
        row["previous_r_opt_deg"] = previous["r_opt_deg"]
        row["previous_containment_r_opt"] = previous["containment_r_opt"]
        row["direct_sigma_deg"] = float(direct_row["sigma_deg"])
        row["direct_r_opt_deg"] = float(direct_row["r_opt_deg"])
        row["direct_containment_r_opt"] = float(direct_row["containment_r_opt"])
        append_warning(row, "direct_own_cell_psf_used_despite_incomplete_theta_support")
        records.append(
            {
                "cell_id": int(cell_id),
                "previous_psf_source": previous["source"],
                "previous": previous,
                "direct": {
                    "sigma_deg": float(direct_row["sigma_deg"]),
                    "r_opt_deg": float(direct_row["r_opt_deg"]),
                    "containment_r_opt": float(direct_row["containment_r_opt"]),
                    "effective_events": float(direct_row["effective_events"]),
                    "theta_missing_crab_probability_mass": float(direct_row["theta_missing_crab_probability_mass"]),
                },
            }
        )

    for row in rows:
        row.setdefault("psf_direct_own_cell", False)
        row.setdefault("psf_borrowed", False)
        row.setdefault("borrowed_from", "")
        row.setdefault("borrow_method", "")
        row.setdefault("borrow_weights", "")

    npz_path = run_dir / args.npz_name
    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    selector_out = path(args.selector_out)
    np.savez_compressed(npz_path, **arrays)

    out_metadata = deepcopy(metadata)
    out_metadata.update(
        {
            "description": "Stage B v3 direct-own-cell PSF artifact for cells 39/52/65.",
            "run_id": sanitize_run_id(args.run_id),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_stage_b": {
                "npz": str(base_npz),
                "metadata_json": str(base_metadata_path),
                "run_id": metadata.get("run_id"),
                "variant": metadata.get("psf_systematic_variant", "base"),
            },
            "direct_focus_stage_b": {
                "npz": str(direct_npz),
                "metadata_json": str(direct_metadata_path),
                "summary_csv": str(direct_summary_csv),
                "run_id": direct_metadata.get("run_id"),
            },
            "psf_systematic_variant": "v3_directpsf",
            "nominal_v3_stage_b_overwritten": False,
            "output_root": str(output_root),
            "output_dir": str(run_dir),
            "current_dir": str(output_root / "current"),
            "latest": str(output_root / "latest"),
            "cells": rows,
            "psf_direct_own_cell": {
                "status": "applied",
                "variant": "v3_directpsf",
                "target_cells": direct_cell_ids,
                "records": records,
                "policy": (
                    "Replace active sigma/r_opt/containment and radial-profile PSF shape for cells "
                    "39/52/65 with their direct own-cell calculation; keep the original missing-theta "
                    "diagnostic as an audit field."
                ),
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
                "selector_csv": str(selector_out),
            },
        }
    )
    warning_rows = out_metadata.get("warning_rows") if isinstance(out_metadata.get("warning_rows"), list) else []
    warning_by_cell = {
        int(row.get("cell_id")): dict(row)
        for row in warning_rows
        if isinstance(row, dict) and row.get("cell_id") is not None
    }
    for record in records:
        cell_id = int(record["cell_id"])
        warning = warning_by_cell.setdefault(cell_id, {"cell_id": cell_id})
        warning["psf_direct_own_cell"] = True
        warning["missing_crab_probability_mass"] = record["direct"]["theta_missing_crab_probability_mass"]
        warning["effective_events"] = record["direct"]["effective_events"]
    out_metadata["warning_rows"] = list(warning_by_cell.values())

    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, out_metadata, rows)
    build_selector(path(args.selector_in), selector_out, direct_cell_ids)
    metadata_path.write_text(json.dumps(json_ready(out_metadata), indent=2) + "\n", encoding="utf-8")

    if not args.no_promote_current:
        promote_current(output_root, run_dir)
        out_metadata["promotion"]["status"] = "promoted"
        out_metadata["promotion"]["current_dir"] = str(output_root / "current")
        out_metadata["promotion"]["latest"] = str(output_root / "latest")
        metadata_path.write_text(json.dumps(json_ready(out_metadata), indent=2) + "\n", encoding="utf-8")
    else:
        out_metadata["promotion"]["status"] = "skipped_no_promote_current"
        metadata_path.write_text(json.dumps(json_ready(out_metadata), indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {npz_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {selector_out}")
    if not args.no_promote_current:
        print(f"Promoted {output_root / 'current'}")


if __name__ == "__main__":
    main()
