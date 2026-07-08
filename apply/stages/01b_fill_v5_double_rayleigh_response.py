#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill zero cells in the v5 double-Rayleigh aperture-conditioned response "
            "from the Rayleigh response, scaled by the Stage B radial-profile containment ratio."
        )
    )
    parser.add_argument("--target-response-npz", required=True)
    parser.add_argument("--target-response-metadata", required=True)
    parser.add_argument("--source-response-npz", required=True)
    parser.add_argument("--source-response-metadata", required=True)
    parser.add_argument("--target-psf-npz", required=True)
    parser.add_argument("--source-psf-npz", required=True)
    parser.add_argument("--min-source-aeff-sum", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_npz(path_: Path) -> Dict[str, np.ndarray]:
    with np.load(path_, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write_npz(path_: Path, arrays: Dict[str, np.ndarray]) -> None:
    tmp = path_.with_suffix(path_.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path_)


def profile_density_containment(profile_density: np.ndarray, profile_edges_deg: np.ndarray, radius_deg: float) -> float:
    edges = np.asarray(profile_edges_deg, dtype=np.float64)
    density = np.asarray(profile_density, dtype=np.float64)
    if edges.size != density.size + 1:
        return float("nan")
    r = float(radius_deg)
    if not np.isfinite(r) or r < 0.0:
        return float("nan")
    widths = np.diff(edges)
    mass = np.clip(np.where(np.isfinite(density), density, 0.0), 0.0, None) * widths
    total = float(np.sum(mass))
    if total <= 0.0:
        return float("nan")
    contained = float(np.sum(mass[edges[1:] <= r]))
    partial = np.nonzero((edges[:-1] < r) & (r < edges[1:]))[0]
    if partial.size:
        idx = int(partial[0])
        width = float(widths[idx])
        if width > 0.0:
            contained += float(mass[idx]) * (r - float(edges[idx])) / width
    return float(min(max(contained / total, 0.0), 1.0))


def index_by_cell(arrays: Dict[str, np.ndarray]) -> Dict[int, int]:
    return {int(cell_id): idx for idx, cell_id in enumerate(np.asarray(arrays["cell_id"], dtype=np.int64))}


def main() -> None:
    args = parse_args()
    target_response_path = path(args.target_response_npz)
    target_meta_path = path(args.target_response_metadata)
    source_response_path = path(args.source_response_npz)
    source_meta_path = path(args.source_response_metadata)
    target_psf_path = path(args.target_psf_npz)
    source_psf_path = path(args.source_psf_npz)

    target = load_npz(target_response_path)
    source = load_npz(source_response_path)
    target_psf = load_npz(target_psf_path)
    source_psf = load_npz(source_psf_path)

    target_by_cell = index_by_cell(target)
    source_by_cell = index_by_cell(source)
    target_psf_by_cell = index_by_cell(target_psf)
    source_psf_by_cell = index_by_cell(source_psf)

    target_aeff = np.asarray(target["a_eff"], dtype=np.float64)
    source_aeff = np.asarray(source["a_eff"], dtype=np.float64)
    profile_edges = np.asarray(target_psf["profile_edges_deg"], dtype=np.float64)
    target_profiles = np.asarray(target_psf["profile_density"], dtype=np.float64)
    source_profiles = np.asarray(source_psf["profile_density"], dtype=np.float64)
    target_r = np.asarray(target_psf["r_opt_deg"], dtype=np.float64)
    source_r = np.asarray(source_psf["r_opt_deg"], dtype=np.float64)

    fillable_keys = [
        "eta",
        "eta_count",
        "eta_conditional",
        "eta_conditional_count",
        "a_eff",
        "numerator_sumw",
        "numerator_count",
    ]
    filled: List[Dict[str, object]] = []
    for cell_id, tidx in target_by_cell.items():
        sidx = source_by_cell.get(cell_id)
        tpidx = target_psf_by_cell.get(cell_id)
        spidx = source_psf_by_cell.get(cell_id)
        if sidx is None or tpidx is None or spidx is None:
            continue
        target_sum = float(np.nansum(target_aeff[tidx]))
        source_sum = float(np.nansum(source_aeff[sidx]))
        if target_sum > 0.0 or source_sum <= float(args.min_source_aeff_sum):
            continue
        c_target = profile_density_containment(target_profiles[tpidx], profile_edges, float(target_r[tpidx]))
        c_source = profile_density_containment(source_profiles[spidx], np.asarray(source_psf["profile_edges_deg"], dtype=np.float64), float(source_r[spidx]))
        if not (np.isfinite(c_target) and np.isfinite(c_source) and c_target > 0.0 and c_source > 0.0):
            continue
        scale = float(c_target / c_source)
        if not (math.isfinite(scale) and scale > 0.0):
            continue
        for key in fillable_keys:
            if key in target and key in source:
                target[key][tidx] = np.asarray(source[key])[sidx] * scale
        filled.append(
            {
                "cell_id": int(cell_id),
                "target_index": int(tidx),
                "source_index": int(sidx),
                "source_a_eff_sum": source_sum,
                "target_containment": c_target,
                "source_containment": c_source,
                "scale": scale,
            }
        )

    if not args.dry_run and filled:
        write_npz(target_response_path, target)

    metadata = json.loads(target_meta_path.read_text(encoding="utf-8"))
    metadata["zero_response_fill_from_rayleigh"] = {
        "mode": "source_response_scaled_by_profile_containment_ratio",
        "source_response_npz": str(source_response_path),
        "source_response_metadata": str(source_meta_path),
        "target_psf_npz": str(target_psf_path),
        "source_psf_npz": str(source_psf_path),
        "filled_cells": filled,
        "dry_run": bool(args.dry_run),
        "note": (
            "Applied only to cells whose newly rebuilt double-Rayleigh response had zero effective area "
            "while the existing Rayleigh aperture-conditioned response was nonzero. The scale is "
            "C_double(r_opt_double) / C_rayleigh(r_opt_rayleigh) from the Stage B radial profile."
        ),
    }
    if not args.dry_run:
        target_meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Filled {len(filled)} zero-response cells in {target_response_path}")
    if filled:
        print("Cells: " + ",".join(str(row["cell_id"]) for row in filled))


if __name__ == "__main__":
    main()
