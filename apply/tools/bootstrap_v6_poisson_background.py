#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_STAGE_D_NPZ = "apply/output/stage_d/current/background_v1.npz"
DEFAULT_MANIFEST = (
    "apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_"
    "double_rayleigh_poisson.json"
)
DEFAULT_SEED = 64748
DEFAULT_REPLICATES = 1000
DEFAULT_WORKERS = 32
POSITIVITY_RADIUS_DEG = 6.0


_WORKER_CONTEXT: Optional[Dict[str, object]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parametric bootstrap covariance for the nominal v6 pooled-Poisson background."
    )
    parser.add_argument("--stage-d-npz", default=DEFAULT_STAGE_D_NPZ)
    parser.add_argument(
        "--stage-e-npz",
        required=True,
        help="Nominal Stage E signal NPZ providing event-level N_on in manifest target order.",
    )
    parser.add_argument("--pooling-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Defaults to the output NPZ name with a .json suffix.",
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--positivity-radius-deg", type=float, default=POSITIVITY_RADIUS_DEG)
    parser.add_argument(
        "--allow-smoke-failures",
        action="store_true",
        help="Debug only and accepted only for at most 100 requested replicates.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def implementation_commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def rng_for_replicate(base_seed: int, replicate_index: int) -> np.random.Generator:
    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    seed_sequence = np.random.SeedSequence([int(base_seed), int(replicate_index)])
    return np.random.default_rng(seed_sequence)


def rectangle_basis_grid(x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x_edges[:-1], dtype=np.float64)[None, :]
    x1 = np.asarray(x_edges[1:], dtype=np.float64)[None, :]
    y0 = np.asarray(y_edges[:-1], dtype=np.float64)[:, None]
    y1 = np.asarray(y_edges[1:], dtype=np.float64)[:, None]
    dx = x1 - x0
    dy = y1 - y0
    return np.stack(
        np.broadcast_arrays(
            dx * dy,
            0.5 * (x1 * x1 - x0 * x0) * dy,
            dx * 0.5 * (y1 * y1 - y0 * y0),
            (x1**3 - x0**3) * dy / 3.0,
            0.25 * (x1 * x1 - x0 * x0) * (y1 * y1 - y0 * y0),
            dx * (y1**3 - y0**3) / 3.0,
        ),
        axis=-1,
    )


def centered_disk_polynomial_integral(coefficients: np.ndarray, radius_deg: float) -> float:
    coefficients = np.pad(
        np.asarray(coefficients, dtype=np.float64),
        (0, max(0, 6 - len(coefficients))),
    )[:6]
    radius = float(radius_deg)
    return float(
        math.pi * radius * radius * coefficients[0]
        + math.pi * radius**4 * (coefficients[3] + coefficients[5]) / 4.0
    )


def centered_annulus_polynomial_integral(
    coefficients: np.ndarray,
    inner_radius_deg: float,
    outer_radius_deg: float,
) -> float:
    return centered_disk_polynomial_integral(
        coefficients, outer_radius_deg
    ) - centered_disk_polynomial_integral(coefficients, inner_radius_deg)


def _required_array(
    data: Mapping[str, np.ndarray],
    name: str,
    *,
    source: str = "Stage D",
) -> np.ndarray:
    if name not in data:
        raise ValueError(f"{source} NPZ is missing required array {name!r}")
    return np.asarray(data[name])


def _one_value_by_cell(
    data: Mapping[str, np.ndarray],
    name: str,
    cell_ids: np.ndarray,
) -> Dict[int, float]:
    values = np.asarray(_required_array(data, name), dtype=np.float64)
    if values.shape != cell_ids.shape:
        raise ValueError(f"Stage D {name} shape {values.shape} does not match cell_id shape {cell_ids.shape}")
    return {int(cell_id): float(value) for cell_id, value in zip(cell_ids, values)}


def prepare_context(
    stage_d: Mapping[str, np.ndarray],
    stage_e: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    from apply.stages.poisson_roi_background import fit_profiled_poisson_surface  # noqa: F401

    cell_ids = np.asarray(_required_array(stage_d, "cell_id"), dtype=np.int64)
    if cell_ids.ndim != 1 or len(np.unique(cell_ids)) != cell_ids.size:
        raise ValueError("Stage D cell_id must be a one-dimensional array of unique ids")
    index_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    target_ids = np.asarray(manifest.get("target_cell_ids", []), dtype=np.int64)
    if target_ids.shape != (44,) or len(np.unique(target_ids)) != 44:
        raise ValueError("Pooling manifest target_cell_ids must contain exactly 44 unique cells in fit order")
    missing_targets = [int(cell_id) for cell_id in target_ids if int(cell_id) not in index_by_cell]
    if missing_targets:
        raise ValueError(f"Stage D NPZ is missing target cells: {missing_targets}")

    stage_e_cell_ids_raw = _required_array(stage_e, "cell_id", source="Stage E")
    if not np.issubdtype(stage_e_cell_ids_raw.dtype, np.integer):
        raise ValueError("Stage E cell_id must use an integer dtype")
    stage_e_cell_ids = np.asarray(stage_e_cell_ids_raw, dtype=np.int64)
    if not np.array_equal(stage_e_cell_ids, target_ids):
        raise ValueError("Stage E cell_id must exactly match pooling manifest target_cell_ids order")

    stage_e_n_on_raw = _required_array(stage_e, "N_on", source="Stage E")
    if not np.issubdtype(stage_e_n_on_raw.dtype, np.integer):
        raise ValueError("Stage E N_on must use an integer dtype")
    stage_e_n_on = np.asarray(stage_e_n_on_raw, dtype=np.int64)
    if stage_e_n_on.shape != target_ids.shape or np.any(stage_e_n_on < 0):
        raise ValueError("Stage E N_on must contain exactly 44 non-negative integer counts")

    cells_manifest = manifest.get("cells")
    if not isinstance(cells_manifest, dict):
        raise ValueError("Pooling manifest is missing the cells object")
    donor_universe = [int(value) for value in manifest.get("donor_universe_cell_ids", [])]
    if len(donor_universe) != 84 or len(set(donor_universe)) != 84:
        raise ValueError(
            "Pooling manifest donor_universe_cell_ids must contain exactly 84 unique non-tail cells"
        )
    donor_universe_set = set(donor_universe)
    if any(int(cell_id) not in donor_universe_set for cell_id in target_ids):
        raise ValueError("Every target cell must belong to the frozen donor universe")
    missing_donors = [cell_id for cell_id in donor_universe if cell_id not in index_by_cell]
    if missing_donors:
        raise ValueError(f"Stage D NPZ is missing donor-universe cells: {missing_donors}")

    counts_map = np.asarray(_required_array(stage_d, "counts_map"), dtype=np.int64)
    expected_map = np.asarray(_required_array(stage_d, "background_map"), dtype=np.float64)
    training_mask = np.asarray(_required_array(stage_d, "training_mask"), dtype=bool)
    expected_shape = (cell_ids.size,) + counts_map.shape[1:]
    if counts_map.ndim != 3 or counts_map.shape != expected_shape or expected_map.shape != expected_shape:
        raise ValueError("Stage D counts_map/background_map must both have shape (cell, y, x)")
    if training_mask.shape != expected_shape:
        raise ValueError("Stage D training_mask shape must match counts_map")
    x_edges = np.asarray(_required_array(stage_d, "x_edges_deg"), dtype=np.float64)
    y_edges = np.asarray(_required_array(stage_d, "y_edges_deg"), dtype=np.float64)
    if counts_map.shape[2] != x_edges.size - 1 or counts_map.shape[1] != y_edges.size - 1:
        raise ValueError("Stage D grid edges do not match counts_map dimensions")
    pixel_basis = rectangle_basis_grid(x_edges, y_edges).reshape(-1, 6)

    r_opt = _one_value_by_cell(stage_d, "r_opt_deg", cell_ids)
    annulus_inner = _one_value_by_cell(stage_d, "annulus_inner_deg", cell_ids)
    annulus_outer = _one_value_by_cell(stage_d, "annulus_outer_deg", cell_ids)
    b_on_nominal = _one_value_by_cell(stage_d, "B_on", cell_ids)
    aligned_stage_d_b_on = np.asarray(
        [b_on_nominal[int(cell_id)] for cell_id in target_ids],
        dtype=np.float64,
    )
    stage_e_b_on = np.asarray(
        _required_array(stage_e, "B_on", source="Stage E"),
        dtype=np.float64,
    )
    if stage_e_b_on.shape != target_ids.shape or not np.all(np.isfinite(stage_e_b_on)):
        raise ValueError("Stage E B_on must contain exactly 44 finite values")
    if not np.array_equal(stage_e_b_on, aligned_stage_d_b_on):
        raise ValueError("Stage E B_on must exactly match Stage D B_on in manifest target order")

    target_specs: Dict[int, Dict[str, object]] = {}
    all_needed_cells: set[int] = set()
    for target_id in target_ids:
        raw = cells_manifest.get(str(int(target_id)))
        if not isinstance(raw, dict):
            raise ValueError(f"Pooling manifest has no cell specification for target {int(target_id)}")
        donors = tuple(int(value) for value in raw.get("donor_cell_ids", []))
        if int(target_id) not in donors:
            raise ValueError(f"Target {int(target_id)} must be included in its frozen donor list")
        if not donors or any(donor not in index_by_cell for donor in donors):
            raise ValueError(f"Target {int(target_id)} has invalid donor_cell_ids {list(donors)}")
        order = int(raw.get("surface_order", -1))
        if order not in {0, 1, 2}:
            raise ValueError(f"Target {int(target_id)} has invalid surface_order {order}")
        target_specs[int(target_id)] = {"donors": donors, "order": order}
        all_needed_cells.update(donors)

    shape_contributor: Dict[int, bool] = {}
    continuous_counts = manifest.get("continuous_annulus_counts")
    if not isinstance(continuous_counts, dict):
        raise ValueError("Pooling manifest is missing continuous_annulus_counts")
    for cell_id in sorted(all_needed_cells):
        if str(cell_id) not in continuous_counts:
            raise ValueError(f"Pooling manifest has no continuous count for donor cell {cell_id}")
        shape_contributor[cell_id] = int(continuous_counts[str(cell_id)]) >= 100

    expected_by_cell: Dict[int, np.ndarray] = {}
    for cell_id in sorted(all_needed_cells):
        cell_index = index_by_cell[cell_id]
        expectation = expected_map[cell_index]
        mask = training_mask[cell_index]
        if np.any(~np.isfinite(expectation[mask])) or np.any(expectation[mask] < 0.0):
            raise ValueError(
                f"Stage D background_map has invalid Poisson expectations in the training mask for cell {cell_id}"
            )
        expected_by_cell[cell_id] = np.where(mask, expectation, 0.0).reshape(-1)

    return {
        "target_ids": target_ids,
        "target_specs": target_specs,
        "expected_by_cell": expected_by_cell,
        "pixel_basis_by_cell": {cell_id: pixel_basis for cell_id in sorted(all_needed_cells)},
        "annulus_mask_by_cell": {
            cell_id: training_mask[index_by_cell[cell_id]].reshape(-1)
            for cell_id in sorted(all_needed_cells)
        },
        "shape_contributor_by_cell": shape_contributor,
        "r_opt_by_cell": r_opt,
        "annulus_inner_by_cell": annulus_inner,
        "annulus_outer_by_cell": annulus_outer,
        "B_on_nominal": aligned_stage_d_b_on,
        "N_on": stage_e_n_on,
    }


def _initialize_worker(context: Dict[str, object]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def bootstrap_replicate(
    replicate_index: int,
    base_seed: int,
    context: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    from apply.stages.poisson_roi_background import fit_profiled_poisson_surface

    active = context if context is not None else _WORKER_CONTEXT
    if active is None:
        raise RuntimeError("Bootstrap worker context was not initialized")
    rng = rng_for_replicate(base_seed, replicate_index)
    expected_by_cell = active["expected_by_cell"]
    assert isinstance(expected_by_cell, dict)
    counts_by_cell = {
        int(cell_id): rng.poisson(np.asarray(expectation, dtype=np.float64))
        for cell_id, expectation in expected_by_cell.items()
    }
    target_ids = np.asarray(active["target_ids"], dtype=np.int64)
    target_specs = active["target_specs"]
    assert isinstance(target_specs, dict)
    fit_cache: Dict[Tuple[Tuple[int, ...], int], object] = {}
    sample = np.empty(target_ids.size, dtype=np.float64)
    for output_index, target_id_raw in enumerate(target_ids):
        target_id = int(target_id_raw)
        spec = target_specs[target_id]
        assert isinstance(spec, dict)
        donors = tuple(int(value) for value in spec["donors"])
        order = int(spec["order"])
        cache_key = (donors, order)
        if cache_key not in fit_cache:
            fit_cache[cache_key] = fit_profiled_poisson_surface(
                counts_by_cell=counts_by_cell,
                pixel_basis_by_cell=active["pixel_basis_by_cell"],
                annulus_mask_by_cell=active["annulus_mask_by_cell"],
                donor_cell_ids=donors,
                order=order,
                positivity_radius_deg=float(active["positivity_radius_deg"]),
                shape_contributor_by_cell=active["shape_contributor_by_cell"],
            )
        fit = fit_cache[cache_key]
        coefficients = np.asarray(fit.shape_coefficients, dtype=np.float64)
        normalization = float(fit.annulus_normalizations[target_id])
        disk_integral = centered_disk_polynomial_integral(
            coefficients, float(active["r_opt_by_cell"][target_id])
        )
        annulus_integral = centered_annulus_polynomial_integral(
            coefficients,
            float(active["annulus_inner_by_cell"][target_id]),
            min(float(active["annulus_outer_by_cell"][target_id]), float(active["positivity_radius_deg"])),
        )
        if not math.isfinite(annulus_integral) or annulus_integral <= 0.0:
            raise RuntimeError(
                f"Replicate {replicate_index} target {target_id} has invalid annulus integral {annulus_integral}"
            )
        sample[output_index] = normalization * disk_integral / annulus_integral
    if not np.all(np.isfinite(sample)) or np.any(sample <= 0.0):
        raise RuntimeError(f"Replicate {replicate_index} produced invalid B_on values")
    return sample


def _worker_task(replicate_index: int, base_seed: int) -> Tuple[int, Optional[np.ndarray], Optional[str]]:
    try:
        return replicate_index, bootstrap_replicate(replicate_index, base_seed), None
    except Exception as exc:
        return replicate_index, None, f"{type(exc).__name__}: {exc}"


def run_bootstrap(
    context: Dict[str, object],
    *,
    replicates: int,
    seed: int,
    workers: int,
    allow_failures: bool,
) -> Tuple[np.ndarray, Dict[int, str]]:
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if workers < 1:
        raise ValueError("workers must be positive")
    samples_by_index: Dict[int, np.ndarray] = {}
    failures: Dict[int, str] = {}
    if workers == 1:
        _initialize_worker(context)
        results = (_worker_task(index, seed) for index in range(replicates))
        for index, sample, error in results:
            if error is not None or sample is None:
                failures[index] = error or "unknown bootstrap failure"
            else:
                samples_by_index[index] = sample
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(context,),
        ) as executor:
            futures = {
                executor.submit(_worker_task, index, seed): index for index in range(replicates)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, sample, error = future.result()
                if error is not None or sample is None:
                    failures[index] = error or "unknown bootstrap failure"
                else:
                    samples_by_index[index] = sample
                if completed % max(1, min(100, replicates // 10)) == 0 or completed == replicates:
                    print(f"Bootstrap progress: {completed}/{replicates}", flush=True)
    if failures and not allow_failures:
        first_index = min(failures)
        raise RuntimeError(
            f"{len(failures)} bootstrap refits failed; first failure replicate {first_index}: {failures[first_index]}"
        )
    if len(samples_by_index) < 2:
        raise RuntimeError("Fewer than two successful bootstrap replicates remain")
    ordered = np.stack([samples_by_index[index] for index in sorted(samples_by_index)], axis=0)
    return ordered, failures


def covariance_monte_carlo_standard_error(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("samples must have shape (replicate, cell) with at least two replicates")
    centered = values - np.mean(values, axis=0, keepdims=True)
    products = centered[:, :, None] * centered[:, None, :]
    return np.std(products, axis=0, ddof=1) / math.sqrt(values.shape[0])


def main() -> None:
    args = parse_args()
    if args.allow_smoke_failures and args.replicates > 100:
        raise ValueError("--allow-smoke-failures is restricted to at most 100 requested replicates")
    if args.positivity_radius_deg <= 0.0:
        raise ValueError("--positivity-radius-deg must be positive")

    stage_d_path = Path(args.stage_d_npz).resolve()
    stage_e_path = Path(args.stage_e_npz).resolve()
    manifest_path = Path(args.pooling_manifest).resolve()
    output_path = Path(args.output_npz).resolve()
    metadata_path = (
        Path(args.metadata_json).resolve()
        if args.metadata_json
        else output_path.with_suffix(".json")
    )
    if not stage_d_path.exists():
        raise FileNotFoundError(f"Stage D NPZ does not exist: {stage_d_path}")
    if not stage_e_path.exists():
        raise FileNotFoundError(f"Stage E NPZ does not exist: {stage_e_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pooling manifest does not exist: {manifest_path}")
    manifest = load_json(manifest_path)
    with np.load(stage_d_path, allow_pickle=False) as data:
        stage_d = {name: data[name].copy() for name in data.files}
    with np.load(stage_e_path, allow_pickle=False) as data:
        stage_e = {name: data[name].copy() for name in data.files}
    context = prepare_context(stage_d, stage_e, manifest)
    context["positivity_radius_deg"] = float(args.positivity_radius_deg)

    samples, failures = run_bootstrap(
        context,
        replicates=int(args.replicates),
        seed=int(args.seed),
        workers=int(args.workers),
        allow_failures=bool(args.allow_smoke_failures),
    )
    mean = np.mean(samples, axis=0)
    background_covariance = np.cov(samples, rowvar=False, ddof=1)
    n_on = np.asarray(context["N_on"], dtype=np.int64)
    excess_covariance = np.diag(n_on.astype(np.float64)) + background_covariance
    eigenvalues = np.linalg.eigvalsh(excess_covariance)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise RuntimeError(
            f"Bootstrap excess covariance is not positive definite; minimum eigenvalue={eigenvalues[0]:.6g}"
        )
    covariance_mcse = covariance_monte_carlo_standard_error(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cell_id=np.asarray(context["target_ids"], dtype=np.int32),
        B_on_nominal=np.asarray(context["B_on_nominal"], dtype=np.float64),
        N_on=n_on.astype(np.int64),
        B_on_bootstrap_mean=mean.astype(np.float64),
        B_on_bootstrap_samples=samples.astype(np.float64),
        B_on_covariance=background_covariance.astype(np.float64),
        excess_covariance=excess_covariance.astype(np.float64),
    )
    manifest_sha = str(manifest.get("manifest_sha256") or sha256_file(manifest_path))
    metadata = {
        "description": "Nominal pooled-Poisson Stage D bootstrap background covariance with Stage E event-level N_on.",
        "inputs": {
            "stage_d_npz": str(stage_d_path),
            "stage_e_npz": str(stage_e_path),
            "pooling_manifest": str(manifest_path),
        },
        "outputs": {"npz": str(output_path), "metadata_json": str(metadata_path)},
        "bootstrap_count_requested": int(args.replicates),
        "bootstrap_count_completed": int(samples.shape[0]),
        "seed": int(args.seed),
        "seed_splitting": "numpy SeedSequence([base_seed, replicate_index])",
        "workers": int(args.workers),
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": sha256_file(manifest_path),
        "stage_d_sha256": sha256_file(stage_d_path),
        "stage_e_sha256": sha256_file(stage_e_path),
        "implementation_commit_sha": implementation_commit_sha(),
        "cell_id": np.asarray(context["target_ids"], dtype=np.int64).tolist(),
        "excess_covariance_eigenvalues": eigenvalues.tolist(),
        "excess_covariance_condition_number": float(np.linalg.cond(excess_covariance)),
        "B_on_covariance_monte_carlo_standard_error": covariance_mcse.tolist(),
        "refit_failure_count": len(failures),
        "refit_failures": {str(index): message for index, message in sorted(failures.items())},
        "production_complete": bool(samples.shape[0] == args.replicates and not failures),
    }
    write_json(metadata_path, metadata)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
