#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_B_METADATA = (
    "apply/output/stage_b_v6_64748_nhit100_highEplus1_split56/"
    "runs/v6_64748_nhit100_highEplus1_split56_stage_b_psf/"
    "psf_v6_64748_nhit100_highEplus1_split56_metadata.json"
)
DEFAULT_OUTPUT_NPZ = (
    "apply/report/assets/v6-64748-nhit100-highEplus1/"
    "v6_64748_nhit100_highEplus1_split56_stage_b_raw_theta_profiles.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete per-cell raw MC theta-profile cache from Stage B metadata, "
            "rescanning only cells whose fallback rows did not preserve the profile."
        )
    )
    parser.add_argument("--stage-b-metadata", default=DEFAULT_STAGE_B_METADATA)
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--tree-name", default="")
    parser.add_argument("--weight-branch", default="mc_weight")
    parser.add_argument("--batch-files", type=int, default=250)
    parser.add_argument("--step-size", default="200 MB")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def saved_theta_probability(row: dict[str, Any], n_bins: int) -> np.ndarray | None:
    theta_reweight = row.get("theta_reweight")
    if not isinstance(theta_reweight, dict):
        return None
    raw = theta_reweight.get("mc_theta_probability")
    if raw is None:
        return None
    probability = np.asarray(raw, dtype=np.float64)
    if probability.shape != (n_bins,) or not np.all(np.isfinite(probability)):
        return None
    total = float(np.sum(probability))
    if total <= 0.0:
        return None
    return probability / total


def scan_cell_theta(
    row: dict[str, Any],
    *,
    tree_name: str,
    weight_branch: str,
    theta_edges_deg: np.ndarray,
    loge_min: float,
    loge_max: float,
    batch_files: int,
    step_size: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cell_id = int(row["cell_id"])
    input_dir = Path(str(row["input_dir"]))
    files = sorted(input_dir.glob("*.root")) if input_dir.is_dir() else []
    theta_sumw = np.zeros(theta_edges_deg.size - 1, dtype=np.float64)
    raw_events = 0
    valid_events = 0
    branches = ["mc_dangle", "mc_theta", "mc_energy", weight_branch]

    batch_files = max(1, int(batch_files))
    for start in range(0, len(files), batch_files):
        chunk = files[start : start + batch_files]
        sources = [f"{path}:{tree_name}" for path in chunk]
        for arrays in uproot.iterate(sources, branches, library="np", step_size=step_size):
            dangle = np.asarray(arrays["mc_dangle"], dtype=np.float64)
            theta_deg = np.degrees(np.asarray(arrays["mc_theta"], dtype=np.float64))
            energy = np.asarray(arrays["mc_energy"], dtype=np.float64)
            weight = np.asarray(arrays[weight_branch], dtype=np.float64)
            raw_events += int(dangle.size)
            loge = np.log10(
                energy,
                where=energy > 0.0,
                out=np.full_like(energy, np.nan, dtype=np.float64),
            )
            valid = (
                np.isfinite(dangle)
                & (dangle >= 0.0)
                & np.isfinite(theta_deg)
                & np.isfinite(loge)
                & (loge >= loge_min)
                & (loge < loge_max)
                & np.isfinite(weight)
                & (weight > 0.0)
            )
            valid_events += int(np.count_nonzero(valid))
            if np.any(valid):
                hist, _ = np.histogram(theta_deg[valid], bins=theta_edges_deg, weights=weight[valid])
                theta_sumw += hist
        print(
            f"cell {cell_id}: files {min(start + batch_files, len(files))}/{len(files)}, "
            f"valid events={valid_events}",
            flush=True,
        )

    total = float(np.sum(theta_sumw))
    probability = theta_sumw / total if total > 0.0 else theta_sumw
    summary = {
        "cell_id": cell_id,
        "nhit_bin": str(row["nhit_bin"]),
        "predE_bin": str(row["predE_bin"]),
        "source": "rescanned_binned_mc" if files else "no_input_files",
        "input_dir": str(input_dir),
        "input_files": len(files),
        "raw_events": raw_events,
        "valid_events": valid_events,
        "sumw": total,
    }
    return probability, summary


def main() -> None:
    args = parse_args()
    metadata_path = resolve(args.stage_b_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = list(metadata.get("cells") or [])
    if not rows:
        raise ValueError(f"No Stage B cells found in {metadata_path}")

    theta_edges_deg = np.asarray(metadata["theta_edges_deg"], dtype=np.float64)
    crab_probability = np.asarray(metadata["crab_theta_probability"], dtype=np.float64)
    if theta_edges_deg.size != crab_probability.size + 1:
        raise ValueError("Stage B theta edges and Crab probability have incompatible shapes")
    loge_filter = metadata.get("logE_true_filter") or {}
    loge_min = float(loge_filter.get("min_inclusive", 2.0))
    loge_max = float(loge_filter.get("max_exclusive", 6.0))
    tree_name = str(args.tree_name or metadata.get("tree_name") or "t_eventout")

    profiles: list[np.ndarray] = []
    missing_masses: list[float] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        probability = saved_theta_probability(row, crab_probability.size)
        if probability is not None:
            summary = {
                "cell_id": int(row["cell_id"]),
                "nhit_bin": str(row["nhit_bin"]),
                "predE_bin": str(row["predE_bin"]),
                "source": "stage_b_metadata",
                "input_dir": str(row.get("input_dir") or ""),
                "input_files": int(row.get("input_files") or 0),
                "raw_events": int(row.get("events") or 0),
                "valid_events": int(row.get("valid_events") or 0),
                "sumw": float(row.get("sumw_mc_weight") or 0.0),
            }
        else:
            probability, summary = scan_cell_theta(
                row,
                tree_name=tree_name,
                weight_branch=str(args.weight_branch),
                theta_edges_deg=theta_edges_deg,
                loge_min=loge_min,
                loge_max=loge_max,
                batch_files=int(args.batch_files),
                step_size=str(args.step_size),
            )

        missing = (crab_probability > 0.0) & ~(probability > 0.0)
        missing_mass = float(np.sum(crab_probability[missing]))
        summary["theta_missing_crab_probability_mass"] = missing_mass
        profiles.append(probability.astype(np.float32))
        missing_masses.append(missing_mass)
        summaries.append(summary)
        print(
            f"cell {summary['cell_id']}: source={summary['source']} "
            f"missing_mass={missing_mass:.6f}",
            flush=True,
        )

    output_npz = resolve(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        cell_id=np.asarray([int(row["cell_id"]) for row in rows], dtype=np.int32),
        nhit_bin=np.asarray([str(row["nhit_bin"]) for row in rows], dtype="U32"),
        predE_bin=np.asarray([str(row["predE_bin"]) for row in rows], dtype="U32"),
        theta_edges_deg=theta_edges_deg.astype(np.float32),
        crab_theta_probability=crab_probability.astype(np.float32),
        mc_theta_probability=np.vstack(profiles).astype(np.float32),
        theta_missing_crab_probability_mass=np.asarray(missing_masses, dtype=np.float32),
        source=np.asarray([str(row["source"]) for row in summaries], dtype="U32"),
        input_files=np.asarray([int(row["input_files"]) for row in summaries], dtype=np.int32),
        raw_events=np.asarray([int(row["raw_events"]) for row in summaries], dtype=np.int64),
        valid_events=np.asarray([int(row["valid_events"]) for row in summaries], dtype=np.int64),
        sumw=np.asarray([float(row["sumw"]) for row in summaries], dtype=np.float64),
    )
    output_json = output_npz.with_suffix(".json")
    output_json.write_text(
        json.dumps(
            {
                "description": "Complete raw mc_weight-normalized theta profiles for all v6 Stage B cells.",
                "stage_b_metadata": str(metadata_path),
                "tree_name": tree_name,
                "weight_branch": str(args.weight_branch),
                "logE_true_range": [loge_min, loge_max],
                "profiles": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_npz}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
