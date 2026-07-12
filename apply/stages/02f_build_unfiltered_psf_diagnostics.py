#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unfiltered diagnostic radial MC profiles by cell.")
    parser.add_argument("--binned-root", required=True)
    parser.add_argument("--cell-selection-csv", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--tree-name", default="t_eventout")
    parser.add_argument("--weight-branch", default="mc_weight")
    parser.add_argument("--profile-max-deg", type=float, default=5.0)
    parser.add_argument("--profile-bin-width-deg", type=float, default=0.05)
    parser.add_argument("--core-fit-max-deg", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    return (
        label.replace(">=", "ge_")
        .replace("<", "lt_")
        .replace("[", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def load_cells(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "cell_id": int(row["cell_id"]),
            "nhit_bin": row["nhit_bin"],
            "predE_bin": row["predE_bin"],
        }
        for row in rows
    ]


def tree_arrays(path: Path, tree_name: str, weight_branch: str) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as root_file:
        tree = root_file[tree_name] if tree_name in root_file else root_file[f"{tree_name};1"]
        arrays = tree.arrays(["mc_dangle", weight_branch], library="np")
    return np.asarray(arrays["mc_dangle"], dtype=np.float64), np.asarray(arrays[weight_branch], dtype=np.float64)


def effective_events(weights: np.ndarray) -> float:
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return 0.0
    sumw = float(np.sum(weights[valid]))
    sumw2 = float(np.sum(np.square(weights[valid])))
    return (sumw * sumw) / sumw2 if sumw2 > 0.0 else 0.0


def profile_density(r_deg: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    valid = np.isfinite(r_deg) & (r_deg >= 0.0) & np.isfinite(weights) & (weights > 0.0)
    hist, _ = np.histogram(r_deg[valid], bins=edges, weights=weights[valid])
    total = float(np.sum(hist))
    return hist / (total * np.diff(edges)) if total > 0.0 else np.zeros(edges.size - 1, dtype=np.float64)


def process_cell(
    cell: dict[str, Any],
    binned_root: str,
    tree_name: str,
    weight_branch: str,
    edges: np.ndarray,
    core_fit_max_deg: float,
) -> tuple[dict[str, Any], np.ndarray]:
    cell_dir = (
        Path(binned_root)
        / f"nhit_{sanitize_label(str(cell['nhit_bin']))}"
        / f"predE_{sanitize_label(str(cell['predE_bin']))}"
    )
    files = sorted(cell_dir.glob("*.root")) if cell_dir.exists() else []
    dangle_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for path in files:
        dangle, weight = tree_arrays(path, tree_name, weight_branch)
        dangle_chunks.append(dangle)
        weight_chunks.append(weight)

    dangle = np.concatenate(dangle_chunks) if dangle_chunks else np.asarray([], dtype=np.float64)
    weights = np.concatenate(weight_chunks) if weight_chunks else np.asarray([], dtype=np.float64)
    valid = np.isfinite(dangle) & (dangle >= 0.0) & np.isfinite(weights) & (weights > 0.0)
    core = valid & (dangle <= math.radians(core_fit_max_deg))
    core_sumw = float(np.sum(weights[core]))
    sigma_rad = (
        math.sqrt(float(np.sum(weights[core] * np.square(dangle[core]))) / (2.0 * core_sumw))
        if core_sumw > 0.0
        else float("nan")
    )
    density = profile_density(np.degrees(dangle), weights, edges)
    status = "ok" if np.isfinite(sigma_rad) and sigma_rad > 0.0 and np.sum(density) > 0.0 else "no_fit_support"
    if not files:
        status = "no_mc_files"
    elif dangle.size == 0:
        status = "no_mc_events"
    row = {
        **cell,
        "input_files": len(files),
        "events": int(dangle.size),
        "valid_weighted_events": int(np.count_nonzero(valid)),
        "effective_events": effective_events(np.where(valid, weights, 0.0)),
        "core_fit_events": int(np.count_nonzero(core)),
        "core_fit_effective_events": effective_events(np.where(core, weights, 0.0)),
        "sigma_deg": math.degrees(sigma_rad) if np.isfinite(sigma_rad) else float("nan"),
        "r_opt_deg": 1.58 * math.degrees(sigma_rad) if np.isfinite(sigma_rad) else float("nan"),
        "status": status,
    }
    return row, density


def main() -> None:
    args = parse_args()
    cells = load_cells(Path(args.cell_selection_csv))
    edges = np.arange(
        0.0,
        float(args.profile_max_deg) + 0.5 * float(args.profile_bin_width_deg),
        float(args.profile_bin_width_deg),
        dtype=np.float64,
    )
    results: dict[int, tuple[dict[str, Any], np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                process_cell,
                cell,
                args.binned_root,
                args.tree_name,
                args.weight_branch,
                edges,
                float(args.core_fit_max_deg),
            ): int(cell["cell_id"])
            for cell in cells
        }
        for done, future in enumerate(as_completed(futures), start=1):
            cell_id = futures[future]
            row, density = future.result()
            results[cell_id] = (row, density)
            print(
                f"[{done}/{len(cells)}] cell={cell_id} events={row['events']} "
                f"sigma={row['sigma_deg']:.5g} status={row['status']}",
                flush=True,
            )

    ordered = [results[int(cell["cell_id"])] for cell in cells]
    rows = [item[0] for item in ordered]
    profiles = np.vstack([item[1] for item in ordered]).astype(np.float32)
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        cell_id=np.asarray([row["cell_id"] for row in rows], dtype=np.int32),
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        predE_bin=np.asarray([row["predE_bin"] for row in rows], dtype="U32"),
        profile_edges_deg=edges.astype(np.float32),
        profile_density=profiles,
        sigma_deg=np.asarray([row["sigma_deg"] for row in rows], dtype=np.float32),
        r_opt_deg=np.asarray([row["r_opt_deg"] for row in rows], dtype=np.float32),
        events=np.asarray([row["events"] for row in rows], dtype=np.int64),
        effective_events=np.asarray([row["effective_events"] for row in rows], dtype=np.float32),
        core_fit_effective_events=np.asarray([row["core_fit_effective_events"] for row in rows], dtype=np.float32),
        status=np.asarray([row["status"] for row in rows], dtype="U32"),
    )

    output_csv = Path(args.output_summary_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_id = {int(row["cell_id"]): row for row in rows}
    for required in (75, 90):
        row = by_id[required]
        if row["status"] != "ok":
            raise SystemExit(f"Diagnostic profile validation failed for cell {required}: {row}")
    print(f"Wrote {output_npz} and {output_csv}")


if __name__ == "__main__":
    main()
