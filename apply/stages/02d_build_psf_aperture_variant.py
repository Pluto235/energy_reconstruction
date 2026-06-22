#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_DIR = "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal"
DEFAULT_OUTPUT_DIR = "apply/output/stage_b_v4_aperture_variants"
APERTURE_SOURCE_ARRAY = {
    "r68": "r68_deg",
    "r90": "r90_deg",
    "r95": "r95_deg",
    "core_r68": "core_r68_deg",
    "core_r90": "core_r90_deg",
    "core_r95": "core_r95_deg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Stage B PSF contract variant with an empirical containment aperture."
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-npz-name", type=str, default="psf_v3_candidate.npz")
    parser.add_argument("--input-metadata-name", type=str, default="psf_v3_candidate_metadata.json")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default="v4_r68_from_psfborrow")
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--npz-name", type=str, default="psf_v4_r68_aperture.npz")
    parser.add_argument("--metadata-name", type=str, default="psf_v4_r68_aperture_metadata.json")
    parser.add_argument("--summary-csv-name", type=str, default="psf_v4_r68_aperture_summary.csv")
    parser.add_argument("--summary-md-name", type=str, default="psf_v4_r68_aperture_summary.md")
    parser.add_argument(
        "--aperture-source",
        choices=sorted(APERTURE_SOURCE_ARRAY),
        default="r68",
        help="Existing Stage B empirical radius array to use as the new r_opt_deg.",
    )
    parser.add_argument(
        "--containment-fraction",
        type=float,
        default=0.68,
        help="Containment fraction recorded as containment_r_opt for the new aperture.",
    )
    parser.add_argument(
        "--fallback-to-original",
        action="store_true",
        default=False,
        help="Use original r_opt/containment for cells whose requested empirical radius is invalid.",
    )
    return parser.parse_args()


def path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def sanitize_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not value:
        raise ValueError("--run-id cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("--run-id may only contain letters, digits, dots, underscores, and hyphens")
    if value in {".", ".."}:
        raise ValueError(f"Invalid --run-id: {value!r}")
    return value


def prepare_run_dir(output_root: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_root / "runs" / sanitize_run_id(run_id)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def replace_path_atomic(target: Path, replacement: Path) -> None:
    backup = target.with_name(f".{target.name}.old")
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    if target.exists() or target.is_symlink():
        target.replace(backup)
    replacement.replace(target)
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def symlink_atomic(link_path: Path, target: Path) -> None:
    tmp = link_path.with_name(f".{link_path.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    replace_path_atomic(link_path, tmp)


def promote_current(output_root: Path, run_dir: Path) -> None:
    for name in ["current", "latest"]:
        try:
            symlink_atomic(output_root / name, run_dir)
        except OSError:
            marker = output_root / name
            marker.write_text(str(run_dir) + "\n", encoding="utf-8")


def load_npz_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_json(path_: Path) -> Dict[str, object]:
    if not path_.exists():
        return {}
    with path_.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path_: Path, payload: Dict[str, object]) -> None:
    with path_.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)


def cell_label(arrays: Dict[str, np.ndarray], key: str, idx: int) -> str:
    if key not in arrays:
        return ""
    return str(np.asarray(arrays[key])[idx])


def write_summary_csv(path_: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "original_r_opt_deg",
        "new_r_opt_deg",
        "new_over_original_r_opt",
        "original_containment_r_opt",
        "new_containment_r_opt",
        "sigma_deg",
        "aperture_source",
        "aperture_source_radius_deg",
        "warning",
    ]
    with path_.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary_md(path_: Path, metadata: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    with path_.open("w", encoding="utf-8") as handle:
        handle.write("# Stage B PSF Aperture Variant Summary\n\n")
        handle.write(f"- Run id: `{metadata['run_id']}`\n")
        handle.write(f"- Aperture source: `{metadata['aperture_variant']['aperture_source']}`\n")
        handle.write(f"- Containment fraction: `{metadata['aperture_variant']['containment_fraction']}`\n")
        handle.write(f"- Input NPZ: `{metadata['inputs']['input_npz']}`\n")
        handle.write(f"- Output NPZ: `{metadata['outputs']['npz']}`\n\n")
        handle.write("| cell | Nhit | predE | old r_opt | new r_opt | ratio | old containment | new containment | warning |\n")
        handle.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['cell_id']} | {row['nhit_bin']} | {row['predE_bin']} | "
                f"{float(row['original_r_opt_deg']):.6g} | {float(row['new_r_opt_deg']):.6g} | "
                f"{float(row['new_over_original_r_opt']):.6g} | "
                f"{float(row['original_containment_r_opt']):.6g} | {float(row['new_containment_r_opt']):.6g} | "
                f"{row.get('warning', '')} |\n"
            )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    input_dir = path(args.input_dir)
    input_npz = input_dir / args.input_npz_name
    input_metadata = input_dir / args.input_metadata_name
    if not input_npz.exists():
        raise FileNotFoundError(input_npz)
    if not (0.0 < float(args.containment_fraction) <= 1.0):
        raise ValueError("--containment-fraction must be in (0, 1]")

    arrays = load_npz_arrays(input_npz)
    source_key = APERTURE_SOURCE_ARRAY[args.aperture_source]
    required = {"cell_id", "r_opt_deg", "r_opt_rad", "containment_r_opt", "sigma_deg", source_key}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"{input_npz} is missing required arrays: {sorted(missing)}")

    source_radius = np.asarray(arrays[source_key], dtype=np.float64)
    original_r = np.asarray(arrays["r_opt_deg"], dtype=np.float64)
    original_containment = np.asarray(arrays["containment_r_opt"], dtype=np.float64)
    valid = np.isfinite(source_radius) & (source_radius > 0.0)
    if not np.all(valid) and not bool(args.fallback_to_original):
        invalid_ids = [int(cid) for cid, ok in zip(arrays["cell_id"], valid) if not bool(ok)]
        raise ValueError(f"Invalid {args.aperture_source} for cells: {invalid_ids}")

    new_r = source_radius.copy()
    new_containment = np.full_like(original_containment, float(args.containment_fraction), dtype=np.float64)
    warnings = np.full(source_radius.shape, "", dtype=object)
    if not np.all(valid):
        new_r[~valid] = original_r[~valid]
        new_containment[~valid] = original_containment[~valid]
        warnings[~valid] = f"invalid_{args.aperture_source}_fallback_to_original"

    output_root = path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_id(args.run_id)
    run_dir = prepare_run_dir(output_root, run_id, overwrite=bool(args.overwrite_run_dir))
    npz_path = run_dir / args.npz_name
    metadata_path = run_dir / args.metadata_name
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name

    payload = {key: value.copy() for key, value in arrays.items()}
    payload["aperture_variant_original_r_opt_deg"] = original_r.astype(np.float32)
    payload["aperture_variant_original_r_opt_rad"] = np.asarray(arrays["r_opt_rad"], dtype=np.float64).astype(np.float32)
    payload["aperture_variant_original_containment_r_opt"] = original_containment.astype(np.float32)
    payload["aperture_variant_source_radius_deg"] = source_radius.astype(np.float32)
    payload["aperture_variant_warning"] = warnings.astype("U96")
    payload["r_opt_deg"] = new_r.astype(np.float32)
    payload["r_opt_rad"] = np.radians(new_r).astype(np.float32)
    payload["containment_r_opt"] = new_containment.astype(np.float32)
    payload["aperture_variant_name"] = np.asarray([args.run_id], dtype="U96")
    payload["aperture_variant_source"] = np.asarray([args.aperture_source], dtype="U32")
    payload["aperture_variant_containment_fraction"] = np.asarray([float(args.containment_fraction)], dtype=np.float32)
    np.savez_compressed(npz_path, **payload)

    rows = []
    for idx, cell_id in enumerate(np.asarray(arrays["cell_id"], dtype=np.int64)):
        ratio = float(new_r[idx] / original_r[idx]) if original_r[idx] > 0.0 else float("nan")
        rows.append(
            {
                "cell_id": int(cell_id),
                "nhit_bin": cell_label(arrays, "nhit_bin", idx),
                "predE_bin": cell_label(arrays, "predE_bin", idx),
                "original_r_opt_deg": float(original_r[idx]),
                "new_r_opt_deg": float(new_r[idx]),
                "new_over_original_r_opt": ratio,
                "original_containment_r_opt": float(original_containment[idx]),
                "new_containment_r_opt": float(new_containment[idx]),
                "sigma_deg": float(np.asarray(arrays["sigma_deg"], dtype=np.float64)[idx]),
                "aperture_source": args.aperture_source,
                "aperture_source_radius_deg": float(source_radius[idx]),
                "warning": str(warnings[idx]),
            }
        )

    input_meta = load_json(input_metadata)
    metadata: Dict[str, object] = {
        "description": "Stage B PSF contract variant using an empirical containment aperture as r_opt.",
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "input_dir": str(input_dir),
            "input_npz": str(input_npz),
            "input_metadata": str(input_metadata) if input_metadata.exists() else None,
            "input_run_id": input_meta.get("run_id"),
        },
        "aperture_variant": {
            "aperture_source": args.aperture_source,
            "aperture_source_array": source_key,
            "new_r_opt_definition": f"r_opt_deg := {args.aperture_source}_deg",
            "containment_fraction": float(args.containment_fraction),
            "new_containment_definition": "constant empirical containment fraction for Stage F/G response scaling",
            "fallback_to_original": bool(args.fallback_to_original),
            "invalid_radius_cells": [int(cid) for cid, ok in zip(arrays["cell_id"], valid) if not bool(ok)],
            "note": (
                "This does not refit the PSF. It changes the aperture contract consumed by Stage D/E/F/G "
                "so that on-region counts, B_on, and response containment are evaluated consistently."
            ),
        },
        "summary": {
            "n_cells": int(len(rows)),
            "median_new_over_original_r_opt": float(np.nanmedian(new_r / original_r)),
            "min_new_over_original_r_opt": float(np.nanmin(new_r / original_r)),
            "max_new_over_original_r_opt": float(np.nanmax(new_r / original_r)),
            "mean_original_containment": float(np.nanmean(original_containment)),
            "new_containment": float(args.containment_fraction),
        },
        "outputs": {
            "npz": str(npz_path),
            "metadata_json": str(metadata_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
        },
        "processing": {
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
    }
    write_summary_csv(summary_csv_path, rows)
    write_summary_md(summary_md_path, metadata, rows)
    write_json(metadata_path, metadata)

    if not args.no_promote_current:
        promote_current(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        write_json(metadata_path, metadata)
    else:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]
        write_json(metadata_path, metadata)

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
