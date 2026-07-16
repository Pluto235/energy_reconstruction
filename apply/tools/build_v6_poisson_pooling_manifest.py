#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apply.stages.poisson_roi_background import (  # noqa: E402
    PoissonSurfaceFitError,
    fit_profiled_poisson_surface,
    quadratic_rectangle_basis_integrals,
)


EXCLUDED_TAIL_CELL_IDS = (13, 26, 39, 52, 65, 78, 91)
INDEPENDENT_QUADRATIC_MIN_COUNT = 20_000
POOLED_QUADRATIC_MIN_COUNT = 10_000
SHAPE_CONTRIBUTOR_MIN_COUNT = 100
N_AZIMUTH_SECTORS = 8


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def write_self_hashed_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["manifest_sha256"] = manifest_payload_sha256(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="ascii")
    reloaded = json.loads(path.read_text(encoding="ascii"))
    actual = manifest_payload_sha256(reloaded)
    if actual != reloaded.get("manifest_sha256"):
        raise RuntimeError(f"Manifest self-hash mismatch after reload: {actual}")
    return reloaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_interval(label: str) -> tuple[float | None, float | None]:
    value = label.strip()
    if value.startswith("[") and value.endswith(")"):
        low, high = value[1:-1].split(",", 1)
        return float(low), float(high)
    if value.startswith(">="):
        return float(value[2:]), None
    if value.startswith("<"):
        return None, float(value[1:])
    raise ValueError(f"Unsupported interval: {label}")


def interval_center(label: str) -> float:
    low, high = parse_interval(label)
    if low is None or high is None:
        raise ValueError(f"A finite interval is required: {label}")
    return 0.5 * (low + high)


def _nearest_same_nhit_ids(
    target_id: int,
    donor_ids: Sequence[int],
    pred_e_center: Mapping[int, float],
) -> list[int]:
    target_center = float(pred_e_center[target_id])
    return sorted(
        (int(cell_id) for cell_id in donor_ids),
        key=lambda cell_id: (
            abs(float(pred_e_center[cell_id]) - target_center),
            float(pred_e_center[cell_id]),
            cell_id,
        ),
    )


def build_pooling_manifest(
    counts: Mapping[int, int],
    pred_e_center: Mapping[int, float],
    *,
    target_ids: Sequence[int],
    nhit_by_cell: Mapping[int, str] | None = None,
    excluded_tail_cell_ids: Sequence[int] = EXCLUDED_TAIL_CELL_IDS,
    independent_min_count: int = INDEPENDENT_QUADRATIC_MIN_COUNT,
    pooled_min_count: int = POOLED_QUADRATIC_MIN_COUNT,
    shape_min_count: int = SHAPE_CONTRIBUTOR_MIN_COUNT,
) -> dict[int, dict[str, Any]]:
    excluded = {int(cell_id) for cell_id in excluded_tail_cell_ids}
    donor_universe = sorted(int(cell_id) for cell_id in counts if int(cell_id) not in excluded)
    result: dict[int, dict[str, Any]] = {}
    for target_id_raw in target_ids:
        target_id = int(target_id_raw)
        if target_id not in donor_universe:
            raise ValueError(f"Target cell {target_id} is not in the donor universe")
        target_count = int(counts[target_id])
        if nhit_by_cell is None:
            same_nhit = donor_universe
        else:
            target_nhit = nhit_by_cell[target_id]
            same_nhit = [cell_id for cell_id in donor_universe if nhit_by_cell[cell_id] == target_nhit]
        if target_count >= int(independent_min_count):
            donors = [target_id]
            mode = "independent"
            pooled_count = target_count
        else:
            donors = []
            pooled_count = 0
            for donor_id in _nearest_same_nhit_ids(target_id, same_nhit, pred_e_center):
                donors.append(donor_id)
                pooled_count += int(counts[donor_id])
                if pooled_count >= int(pooled_min_count):
                    break
            mode = "pooled" if pooled_count >= int(pooled_min_count) else "shared_plane_fallback"
        result[target_id] = {
            "continuous_annulus_count": target_count,
            "mode": mode,
            "surface_order": 2 if mode in {"independent", "pooled"} else 1,
            "donor_cell_ids": sorted(donors),
            "pooled_continuous_annulus_count": int(pooled_count),
            "shape_contributor": target_count >= int(shape_min_count),
            "shape_contributor_by_donor": {
                str(donor_id): int(counts[donor_id]) >= int(shape_min_count) for donor_id in sorted(donors)
            },
        }
    return result


def crab_tangent_xy(
    ra_deg: np.ndarray, dec_deg: np.ndarray, source_ra_deg: float, source_dec_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dra = ((np.asarray(ra_deg, dtype=np.float64) - float(source_ra_deg) + 180.0) % 360.0) - 180.0
    x = dra * math.cos(math.radians(float(source_dec_deg)))
    y = np.asarray(dec_deg, dtype=np.float64) - float(source_dec_deg)
    return x, y, np.hypot(x, y)


def annulus_placement(
    r_opt_deg: np.ndarray,
    *,
    default_inner_deg: float = 1.5,
    width_deg: float = 2.0,
    source_mask_min_deg: float = 1.5,
    source_mask_r_opt_factor: float = 2.0,
    source_mask_margin_deg: float = 0.2,
    max_inner_deg: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    source_mask = np.maximum(float(source_mask_min_deg), float(source_mask_r_opt_factor) * r_opt_deg)
    shifted = source_mask + float(source_mask_margin_deg)
    inner = np.full(r_opt_deg.shape, float(default_inner_deg), dtype=np.float64)
    needs_shift = shifted > float(default_inner_deg)
    inner[needs_shift] = np.minimum(np.ceil(shifted[needs_shift] * 2.0) / 2.0, float(max_inner_deg))
    return inner, inner + float(width_deg)


def make_grid_edges(step_deg: float, offset_fraction: float = 0.0, radius_deg: float = 8.0) -> np.ndarray:
    step = float(step_deg)
    offset = step * float(offset_fraction)
    start = -float(radius_deg) + offset
    while start > -float(radius_deg):
        start -= step
    stop = float(radius_deg) + step
    count = int(math.ceil((stop - start) / step))
    return start + step * np.arange(count + 1, dtype=np.float64)


def rectangle_basis_grid(edges: np.ndarray) -> np.ndarray:
    n = edges.size - 1
    basis = np.empty((n, n, 6), dtype=np.float64)
    for y_idx in range(n):
        for x_idx in range(n):
            basis[y_idx, x_idx] = quadratic_rectangle_basis_integrals(
                float(edges[x_idx]), float(edges[x_idx + 1]), float(edges[y_idx]), float(edges[y_idx + 1])
            )
    return basis


def scan_continuous_annulus_maps(
    obs_events_dir: Path,
    donor_cell_ids: Sequence[int],
    annulus_inner_by_cell: Mapping[int, float],
    annulus_outer_by_cell: Mapping[int, float],
    *,
    grid_step_deg: float = 0.1,
    source_ra_deg: float = 83.63,
    source_dec_deg: float = 22.01,
    batch_size: int = 500_000,
    print_every: int = 10,
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    donor_ids = tuple(sorted(int(cell_id) for cell_id in donor_cell_ids))
    donor_set = set(donor_ids)
    edges = make_grid_edges(grid_step_deg)
    n = edges.size - 1
    maps = {cell_id: np.zeros((n, n), dtype=np.int64) for cell_id in donor_ids}
    sector_maps = {cell_id: np.zeros((N_AZIMUTH_SECTORS, n, n), dtype=np.int64) for cell_id in donor_ids}
    continuous_counts = {cell_id: 0 for cell_id in donor_ids}
    dataset = ds.dataset(obs_events_dir, format="parquet", partitioning="hive")
    scanner = dataset.scanner(columns=["ra_mean_deg", "dec_mean_deg", "cell_id"], batch_size=int(batch_size))
    for batch_index, batch in enumerate(scanner.to_batches(), start=1):
        ra = np.asarray(batch.column("ra_mean_deg"), dtype=np.float64)
        dec = np.asarray(batch.column("dec_mean_deg"), dtype=np.float64)
        cell = np.asarray(batch.column("cell_id"), dtype=np.int64)
        x, y, rho = crab_tangent_xy(ra, dec, source_ra_deg, source_dec_deg)
        finite = np.isfinite(x) & np.isfinite(y)
        for cell_id in np.unique(cell[finite]):
            cid = int(cell_id)
            if cid not in donor_set:
                continue
            selected = finite & (cell == cid)
            selected &= rho >= float(annulus_inner_by_cell[cid])
            selected &= rho < float(annulus_outer_by_cell[cid])
            selected &= rho < 6.0
            continuous_counts[cid] += int(np.count_nonzero(selected))
            if not np.any(selected):
                continue
            hist = np.histogram2d(y[selected], x[selected], bins=(edges, edges))[0].astype(np.int64)
            maps[cid] += hist
            angle = np.mod(np.arctan2(y[selected], x[selected]), 2.0 * math.pi)
            sector = np.minimum((angle * N_AZIMUTH_SECTORS / (2.0 * math.pi)).astype(np.int64), 7)
            for sector_id in range(N_AZIMUTH_SECTORS):
                in_sector = sector == sector_id
                if np.any(in_sector):
                    sector_maps[cid][sector_id] += np.histogram2d(
                        y[selected][in_sector], x[selected][in_sector], bins=(edges, edges)
                    )[0].astype(np.int64)
        if print_every > 0 and batch_index % int(print_every) == 0:
            print(f"[manifest batch {batch_index}] annulus events={sum(continuous_counts.values()):,}", flush=True)
    return continuous_counts, maps, sector_maps, edges


def poisson_deviance_score(counts: np.ndarray, expected_probability: np.ndarray) -> float:
    z = np.asarray(counts, dtype=np.float64)
    p = np.asarray(expected_probability, dtype=np.float64)
    if z.shape != p.shape or np.any(p <= 0.0) or not np.all(np.isfinite(p)):
        return float("inf")
    total = float(z.sum())
    if total <= 0.0:
        return 0.0
    mu = total * p / float(p.sum())
    positive = z > 0.0
    terms = np.zeros(z.shape, dtype=np.float64)
    terms[positive] = z[positive] * np.log(z[positive] / mu[positive]) - (z[positive] - mu[positive])
    terms[~positive] = mu[~positive]
    return float(2.0 * terms.sum())


def choose_one_standard_error(candidate_scores: Mapping[int, Sequence[float]]) -> tuple[int, dict[str, Any]]:
    summary: dict[str, Any] = {}
    for order in sorted(candidate_scores):
        scores = np.asarray(candidate_scores[order], dtype=np.float64)
        finite = scores[np.isfinite(scores)]
        mean = float(np.mean(finite)) if finite.size else float("inf")
        se = float(np.std(finite, ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else 0.0
        summary[str(order)] = {"fold_scores": scores.tolist(), "mean": mean, "standard_error": se}
    best_order = min(candidate_scores, key=lambda order: summary[str(order)]["mean"])
    threshold = float(summary[str(best_order)]["mean"] + summary[str(best_order)]["standard_error"])
    eligible = [order for order in sorted(candidate_scores) if summary[str(order)]["mean"] <= threshold]
    selected = int(min(eligible)) if eligible else int(best_order)
    return selected, {"candidates": summary, "best_order": int(best_order), "one_se_threshold": threshold}


def _score_fit(
    coefficients: np.ndarray,
    pixel_basis: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
) -> float:
    shape = np.asarray(pixel_basis, dtype=np.float64) @ np.asarray(coefficients, dtype=np.float64)
    selected_shape = shape[np.asarray(mask, dtype=bool)]
    return poisson_deviance_score(np.asarray(counts)[np.asarray(mask, dtype=bool)], selected_shape)


def cross_validate_orders(
    donor_cell_ids: Sequence[int],
    counts_map_by_cell: Mapping[int, np.ndarray],
    sector_maps_by_cell: Mapping[int, np.ndarray],
    pixel_basis_grid: np.ndarray,
    annulus_mask_by_cell: Mapping[int, np.ndarray],
    shape_contributor_by_cell: Mapping[int, bool],
    *,
    positivity_radius_deg: float = 6.0,
) -> tuple[int, dict[str, Any]]:
    donors = tuple(int(cell_id) for cell_id in donor_cell_ids)
    basis = {cell_id: pixel_basis_grid.reshape(-1, 6) for cell_id in donors}
    masks = {cell_id: np.asarray(annulus_mask_by_cell[cell_id], dtype=bool).reshape(-1) for cell_id in donors}
    candidate_scores: dict[int, list[float]] = {0: [], 1: [], 2: []}
    failures: dict[str, list[str]] = {"0": [], "1": [], "2": []}
    for order in (0, 1, 2):
        for sector_id in range(N_AZIMUTH_SECTORS):
            training_counts = {
                cell_id: (np.asarray(counts_map_by_cell[cell_id]) - np.asarray(sector_maps_by_cell[cell_id])[sector_id]).reshape(-1)
                for cell_id in donors
            }
            try:
                fit = fit_profiled_poisson_surface(
                    training_counts,
                    basis,
                    masks,
                    donors,
                    order,
                    positivity_radius_deg,
                    dict(shape_contributor_by_cell),
                )
                score = sum(
                    _score_fit(
                        fit.shape_coefficients,
                        basis[cell_id],
                        masks[cell_id],
                        np.asarray(sector_maps_by_cell[cell_id])[sector_id].reshape(-1),
                    )
                    for cell_id in donors
                )
            except (PoissonSurfaceFitError, ValueError) as exc:
                score = float("inf")
                failures[str(order)].append(f"sector_{sector_id}:{exc}")
            candidate_scores[order].append(float(score))
        if len(donors) > 1:
            for held_id in donors:
                training_donors = tuple(cell_id for cell_id in donors if cell_id != held_id)
                try:
                    fit = fit_profiled_poisson_surface(
                        {cell_id: np.asarray(counts_map_by_cell[cell_id]).reshape(-1) for cell_id in training_donors},
                        {cell_id: basis[cell_id] for cell_id in training_donors},
                        {cell_id: masks[cell_id] for cell_id in training_donors},
                        training_donors,
                        order,
                        positivity_radius_deg,
                        {cell_id: bool(shape_contributor_by_cell[cell_id]) for cell_id in training_donors},
                    )
                    score = _score_fit(
                        fit.shape_coefficients,
                        basis[held_id],
                        masks[held_id],
                        np.asarray(counts_map_by_cell[held_id]).reshape(-1),
                    )
                except (PoissonSurfaceFitError, ValueError) as exc:
                    score = float("inf")
                    failures[str(order)].append(f"donor_{held_id}:{exc}")
                candidate_scores[order].append(float(score))
    selected, evidence = choose_one_standard_error(candidate_scores)
    evidence["failures"] = failures
    evidence["fold_contract"] = {
        "azimuth_sectors": N_AZIMUTH_SECTORS,
        "sector_zero_angle_rad": 0.0,
        "leave_one_donor_cell_out": len(donors) > 1,
    }
    return selected, evidence


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _require_baseline_sha(value: str) -> str:
    sha = value.strip()
    if not sha:
        raise SystemExit("ANALYTIC_BON_BASELINE_SHA is required")
    status = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT)
    if status.returncode != 0:
        raise SystemExit(f"Analytic B_on baseline {sha} is not an ancestor of HEAD")
    return sha


def parse_args() -> argparse.Namespace:
    production = REPO_ROOT.parent / "energy_reconstruction"
    parser = argparse.ArgumentParser(description="Freeze v6 Poisson background pooling and CV topology.")
    parser.add_argument("--stage-c-dir", type=Path, default=production / "apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear")
    parser.add_argument("--psf-npz", type=Path, default=production / "apply/output/stage_b_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh/runs/v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_stage_b_psf/psf_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh.npz")
    parser.add_argument("--psf-metadata", type=Path, default=production / "apply/output/stage_b_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh/runs/v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_stage_b_psf/psf_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_metadata.json")
    parser.add_argument("--selector", type=Path, default=REPO_ROOT / "apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv")
    parser.add_argument("--source-table", type=Path, default=REPO_ROOT / "apply/config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json")
    parser.add_argument("--analytic-baseline-sha", default=os.environ.get("ANALYTIC_BON_BASELINE_SHA", ""))
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_sha = _require_baseline_sha(args.analytic_baseline_sha)
    source_rows = _read_csv(args.source_table.resolve())
    selector_rows = _read_csv(args.selector.resolve())
    source_by_id = {int(row["cell_id"]): row for row in source_rows}
    donor_ids = sorted(cell_id for cell_id, row in source_by_id.items() if int(row.get("tail_bin_flag", "0")) == 0)
    target_ids = sorted(int(row["cell_id"]) for row in selector_rows if row.get("include") == "1")
    if len(source_rows) != 91 or len(donor_ids) != 84 or len(target_ids) != 44:
        raise SystemExit(f"Expected 91 source / 84 donor / 44 target cells; got {len(source_rows)}/{len(donor_ids)}/{len(target_ids)}")
    if tuple(sorted(set(source_by_id) - set(donor_ids))) != EXCLUDED_TAIL_CELL_IDS:
        raise SystemExit("The excluded tail ids do not match the frozen seven-cell contract")

    psf_path = args.psf_npz.resolve()
    with np.load(psf_path, allow_pickle=False) as psf:
        psf_ids = np.asarray(psf["cell_id"], dtype=np.int64)
        r_opt = np.asarray(psf["r_opt_deg"], dtype=np.float64)
    r_opt_by_id = {int(cell_id): float(r_opt[index]) for index, cell_id in enumerate(psf_ids)}
    if set(donor_ids) - set(r_opt_by_id):
        raise SystemExit("Stage B PSF is missing donor cells")
    annulus_inner, annulus_outer = annulus_placement(np.asarray([r_opt_by_id[cell_id] for cell_id in donor_ids]))
    inner_by_id = dict(zip(donor_ids, annulus_inner.tolist()))
    outer_by_id = dict(zip(donor_ids, annulus_outer.tolist()))
    continuous, maps, sector_maps, edges = scan_continuous_annulus_maps(
        args.stage_c_dir.resolve() / "obs_events",
        donor_ids,
        inner_by_id,
        outer_by_id,
        batch_size=args.batch_size,
        print_every=args.print_every,
    )
    pred_center = {cell_id: interval_center(source_by_id[cell_id]["predE_bin"]) for cell_id in donor_ids}
    nhit = {cell_id: source_by_id[cell_id]["nhit_bin"] for cell_id in donor_ids}
    cells = build_pooling_manifest(continuous, pred_center, target_ids=target_ids, nhit_by_cell=nhit)
    basis = rectangle_basis_grid(edges)
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, x_centers)
    rho = np.hypot(x_grid, y_grid)
    mask_by_id = {
        cell_id: (rho >= inner_by_id[cell_id]) & (rho < outer_by_id[cell_id]) & (rho < 6.0)
        for cell_id in donor_ids
    }
    group_cv: dict[tuple[int, ...], tuple[int, dict[str, Any]]] = {}
    for target_id in target_ids:
        row = cells[target_id]
        donors = tuple(int(cell_id) for cell_id in row["donor_cell_ids"])
        if donors not in group_cv:
            group_cv[donors] = cross_validate_orders(
                donors,
                maps,
                sector_maps,
                basis,
                mask_by_id,
                {cell_id: continuous[cell_id] >= SHAPE_CONTRIBUTOR_MIN_COUNT for cell_id in donors},
            )
        selected_order, cv = group_cv[donors]
        if row["mode"] == "independent":
            row["surface_order"] = 2 if int(selected_order) == 2 else 1
            if int(selected_order) != 2:
                row["mode"] = "independent_plane_cv_fallback"
        else:
            row["surface_order"] = 2 if (
                int(selected_order) == 2
                and int(row["pooled_continuous_annulus_count"]) >= POOLED_QUADRATIC_MIN_COUNT
            ) else 1
            if row["surface_order"] == 1:
                row["mode"] = "shared_plane_fallback"
        row["cross_validation"] = cv
        row["annulus_inner_deg"] = float(inner_by_id[target_id])
        row["annulus_outer_deg"] = float(outer_by_id[target_id])
        row["r_opt_deg"] = float(r_opt_by_id[target_id])

    execution_assignments: dict[str, dict[str, Any]] = {}
    for cell_id in donor_ids:
        containing = [
            target_id for target_id in target_ids if cell_id in cells[target_id]["donor_cell_ids"]
        ]
        candidates = containing or [
            target_id for target_id in target_ids if nhit[target_id] == nhit[cell_id]
        ]
        if not candidates:
            raise SystemExit(f"No frozen execution pool is available for donor cell {cell_id}")
        target_id = min(
            candidates,
            key=lambda candidate: (
                abs(pred_center[candidate] - pred_center[cell_id]),
                pred_center[candidate],
                candidate,
            ),
        )
        execution_assignments[str(cell_id)] = {
            "pool_target_cell_id": int(target_id),
            "donor_cell_ids": list(cells[target_id]["donor_cell_ids"]),
            "surface_order": int(cells[target_id]["surface_order"]),
        }

    stage_c_dir = args.stage_c_dir.resolve()
    code_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled",
        "analytic_bon_baseline_sha": baseline_sha,
        "implementation_commit_sha": _git_sha(),
        "target_cell_ids": target_ids,
        "donor_universe_cell_ids": donor_ids,
        "excluded_tail_cell_ids": list(EXCLUDED_TAIL_CELL_IDS),
        "continuous_annulus_counts": {str(cell_id): int(continuous[cell_id]) for cell_id in donor_ids},
        "thresholds": {
            "independent_quadratic_min_continuous_annulus_count": INDEPENDENT_QUADRATIC_MIN_COUNT,
            "pooled_quadratic_min_continuous_annulus_count": POOLED_QUADRATIC_MIN_COUNT,
            "shape_contributor_min_continuous_annulus_count": SHAPE_CONTRIBUTOR_MIN_COUNT,
        },
        "grid_contract": {"nominal_step_deg": 0.1, "nominal_offset_fraction": [0.0, 0.0]},
        "provenance": {
            "selector": {"path": str(args.selector.resolve()), "sha256": sha256_file(args.selector.resolve())},
            "source_table": {"path": str(args.source_table.resolve()), "sha256": sha256_file(args.source_table.resolve())},
            "psf_npz": {"path": str(psf_path), "sha256": sha256_file(psf_path)},
            "psf_metadata": {"path": str(args.psf_metadata.resolve()), "sha256": sha256_file(args.psf_metadata.resolve())},
            "stage_c_metadata": {"path": str(stage_c_dir / "obs_events_metadata.json"), "sha256": sha256_file(stage_c_dir / "obs_events_metadata.json")},
            "stage_c_source_files": {"path": str(stage_c_dir / "source_files.csv"), "sha256": sha256_file(stage_c_dir / "source_files.csv")},
            "manifest_builder": {"path": str(code_path), "sha256": sha256_file(code_path)},
        },
        "cells": {str(cell_id): cells[cell_id] for cell_id in target_ids},
        "execution_cell_assignments": execution_assignments,
    }
    written = write_self_hashed_manifest(args.output.resolve(), payload)
    print(f"Wrote {args.output.resolve()} ({written['manifest_sha256']})", flush=True)


if __name__ == "__main__":
    main()
