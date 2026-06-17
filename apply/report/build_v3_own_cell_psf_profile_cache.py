#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diagnostic-only own-cell PSF radial profile cache for borrowed v3 cells."
    )
    parser.add_argument("--stage-b-metadata", type=str, default="apply/output/stage_b_v3_candidate/current/psf_v3_candidate_metadata.json")
    parser.add_argument("--cell-ledger-csv", type=str, default="apply/config/cell_ledger_v3_candidate.csv")
    parser.add_argument("--cell-ids", type=str, default="39,52,65")
    parser.add_argument("--output-npz", type=str, default="apply/report/assets/v3-psfborrow/v3_own_cell_psf_profile_cache.npz")
    parser.add_argument("--tree-name", type=str, default="")
    parser.add_argument("--weight-branch", type=str, default="mc_weight")
    parser.add_argument("--batch-size", type=str, default="200 MB")
    return parser.parse_args()


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def parse_interval(label: str) -> Tuple[Optional[float], Optional[float]]:
    text = label.strip()
    if text.startswith("[") and text.endswith(")"):
        low, high = text[1:-1].split(",", 1)
        return float(low), float(high)
    if text.startswith(">="):
        return float(text[2:]), None
    raise ValueError(f"Unsupported interval label: {label}")


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def sanitize_label(label: str) -> str:
    return (
        str(label)
        .replace(">=", "ge_")
        .replace("<", "lt_")
        .replace("[", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def load_cells(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    rows.sort(key=lambda row: (interval_key(row["nhit_bin"]), interval_key(row["predE_bin"]), int(row["cell_id"])))
    return rows


def tree_path(path: Path, tree_name: str) -> str:
    with uproot.open(path) as root_file:
        if tree_name in root_file:
            return f"{path}:{tree_name}"
        versioned = f"{tree_name};1"
        if versioned in root_file:
            return f"{path}:{versioned}"
    raise KeyError(f"{path} does not contain tree {tree_name!r}")


def build_profile(
    *,
    cell: Dict[str, object],
    binned_root: Path,
    tree_name_value: str,
    weight_branch: str,
    loge_min: float,
    loge_max: float,
    profile_edges_deg: np.ndarray,
    theta_edges_deg: np.ndarray,
    batch_size: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    cell_dir = binned_root / f"nhit_{sanitize_label(str(cell['nhit_bin']))}" / f"predE_{sanitize_label(str(cell['predE_bin']))}"
    files = sorted(cell_dir.glob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files found for cell {cell['cell_id']} in {cell_dir}")

    hist = np.zeros(len(profile_edges_deg) - 1, dtype=np.float64)
    theta_hist = np.zeros(len(theta_edges_deg) - 1, dtype=np.float64)
    events = 0
    used_events = 0
    input_files = 0
    branches = ["mc_dangle", "mc_theta", "mc_energy", weight_branch]
    for file_index, path in enumerate(files, start=1):
        input_files += 1
        source = tree_path(path, tree_name_value)
        for arrays in uproot.iterate(source, branches, library="np", step_size=batch_size):
            dangle = np.asarray(arrays["mc_dangle"], dtype=np.float64)
            mc_theta = np.asarray(arrays["mc_theta"], dtype=np.float64)
            mc_energy = np.asarray(arrays["mc_energy"], dtype=np.float64)
            weight = np.asarray(arrays[weight_branch], dtype=np.float64)
            events += int(dangle.size)
            loge_true = np.log10(mc_energy, where=mc_energy > 0.0, out=np.full_like(mc_energy, np.nan, dtype=np.float64))
            valid = (
                np.isfinite(dangle)
                & (dangle >= 0.0)
                & np.isfinite(loge_true)
                & (loge_true >= loge_min)
                & (loge_true < loge_max)
                & np.isfinite(weight)
                & (weight > 0.0)
            )
            used_events += int(np.count_nonzero(valid))
            if np.any(valid):
                r_deg = np.degrees(dangle[valid])
                values, _ = np.histogram(r_deg, bins=profile_edges_deg, weights=weight[valid])
                hist += values
                theta_values, _ = np.histogram(np.degrees(mc_theta[valid]), bins=theta_edges_deg, weights=weight[valid])
                theta_hist += theta_values
        if file_index % 1000 == 0 or file_index == len(files):
            print(f"cell {cell['cell_id']}: read {file_index}/{len(files)} files, used={used_events}", flush=True)

    total = float(np.sum(hist))
    widths = np.diff(profile_edges_deg)
    density = hist / (total * widths) if total > 0.0 else hist
    theta_total = float(np.sum(theta_hist))
    theta_probability = theta_hist / theta_total if theta_total > 0.0 else theta_hist
    return density.astype(np.float32), theta_probability.astype(np.float32), {
        "cell_id": int(cell["cell_id"]),
        "nhit_bin": str(cell["nhit_bin"]),
        "predE_bin": str(cell["predE_bin"]),
        "input_files": input_files,
        "events": events,
        "used_events": used_events,
        "status": "ok" if total > 0.0 else "empty",
        "sumw": total,
    }


def main() -> None:
    args = parse_args()
    stage_b_metadata_path = abs_path(args.stage_b_metadata)
    stage_b = json.loads(stage_b_metadata_path.read_text(encoding="utf-8"))
    binned_root = Path(str(stage_b["binned_root"]))
    tree_name_value = args.tree_name or str(stage_b.get("tree_name") or "t_eventout")
    loge_filter = stage_b.get("logE_true_filter") if isinstance(stage_b.get("logE_true_filter"), dict) else {}
    loge_min = float(loge_filter.get("min_inclusive", 2.0))
    loge_max = float(loge_filter.get("max_exclusive", 6.0))
    profile_edges_deg = np.asarray(stage_b.get("profile_edges_deg") or np.arange(0.0, 5.0 + 0.05, 0.05), dtype=np.float64)
    if profile_edges_deg.ndim != 1 or profile_edges_deg.size < 2:
        profile_edges_deg = np.arange(0.0, 5.0 + 0.05, 0.05, dtype=np.float64)
    theta_edges_deg = np.asarray(stage_b.get("theta_edges_deg") or np.arange(0.0, 50.0 + 1.0, 1.0), dtype=np.float64)
    if theta_edges_deg.ndim != 1 or theta_edges_deg.size < 2:
        theta_edges_deg = np.arange(0.0, 50.0 + 1.0, 1.0, dtype=np.float64)

    requested_ids = [int(value.strip()) for value in args.cell_ids.split(",") if value.strip()]
    cells = {int(row["cell_id"]): row for row in load_cells(abs_path(args.cell_ledger_csv))}
    profiles = []
    theta_probabilities = []
    summaries = []
    for cell_id in requested_ids:
        if cell_id not in cells:
            raise KeyError(f"Cell {cell_id} missing from {args.cell_ledger_csv}")
        profile, theta_probability, summary = build_profile(
            cell=cells[cell_id],
            binned_root=binned_root,
            tree_name_value=tree_name_value,
            weight_branch=args.weight_branch,
            loge_min=loge_min,
            loge_max=loge_max,
            profile_edges_deg=profile_edges_deg,
            theta_edges_deg=theta_edges_deg,
            batch_size=args.batch_size,
        )
        profiles.append(profile)
        theta_probabilities.append(theta_probability)
        summaries.append(summary)

    output_npz = abs_path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        cell_id=np.asarray(requested_ids, dtype=np.int32),
        profile_edges_deg=profile_edges_deg.astype(np.float32),
        profile_density=np.vstack(profiles).astype(np.float32),
        theta_edges_deg=theta_edges_deg.astype(np.float32),
        mc_theta_probability=np.vstack(theta_probabilities).astype(np.float32),
        status=np.asarray([row["status"] for row in summaries], dtype="U32"),
        events=np.asarray([row["events"] for row in summaries], dtype=np.int64),
        used_events=np.asarray([row["used_events"] for row in summaries], dtype=np.int64),
        input_files=np.asarray([row["input_files"] for row in summaries], dtype=np.int32),
        sumw=np.asarray([row["sumw"] for row in summaries], dtype=np.float64),
    )
    summary_path = output_npz.with_suffix(".json")
    summary_path.write_text(json.dumps({"profiles": summaries}, indent=2), encoding="utf-8")
    print(f"Wrote {output_npz}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
