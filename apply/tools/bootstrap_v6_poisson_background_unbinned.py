#!/usr/bin/env python
"""Bootstrap covariance for the v6 UNBINNED pooled-Poisson background.

Unbinned analogue of ``bootstrap_v6_poisson_background.py``. For each replicate:

- order-2 cells: draw ``M ~ Poisson(N_ann)`` OFF events, resample M observed OFF-event
  coordinates with replacement, refit the shared shape with
  ``fit_profiled_poisson_surface_unbinned``, and recompute B_on from the analytic centered
  integrals. This captures both the level (M) and the curvature (tau) uncertainty.
- order<=1 cells: B_on is linear in N_ann with a shape-independent geometric factor, so draw
  ``N_ann' ~ Poisson(N_ann)`` and scale ``B_on' = B_on_nominal * N_ann'/N_ann`` (no refit).

Output schema matches the binned tool (44x44 ``B_on_covariance`` and
``excess_covariance = diag(N_on) + B_on_covariance``) so ``06_fit.py`` consumes it unchanged.

The OFF-event coordinates come from an NPZ dumped by ``04_background.py --dump-off-events-npz``
during the nominal unbinned Stage-D run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = (
    "apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_"
    "double_rayleigh_poisson.json"
)
DEFAULT_SEED = 64748
DEFAULT_REPLICATES = 1000
DEFAULT_WORKERS = 20
POSITIVITY_RADIUS_DEG = 6.0

_WORKER_CONTEXT: Optional[Dict[str, object]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap covariance for the nominal v6 UNBINNED pooled-Poisson background."
    )
    parser.add_argument("--stage-d-npz", required=True)
    parser.add_argument("--stage-e-npz", required=True,
                        help="Nominal Stage E signal NPZ providing event-level N_on in manifest target order.")
    parser.add_argument("--off-events-npz", required=True,
                        help="OFF-event dump from 04_background.py --dump-off-events-npz (unbinned nominal).")
    parser.add_argument("--pooling-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--positivity-radius-deg", type=float, default=POSITIVITY_RADIUS_DEG)
    parser.add_argument("--allow-smoke-failures", action="store_true",
                        help="Debug only; accepted only for at most 100 requested replicates.")
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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def rng_for_replicate(base_seed: int, replicate_index: int) -> np.random.Generator:
    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(replicate_index)]))


def centered_disk_polynomial_integral(coefficients: np.ndarray, radius_deg: float) -> float:
    coefficients = np.pad(np.asarray(coefficients, dtype=np.float64), (0, max(0, 6 - len(coefficients))))[:6]
    radius = float(radius_deg)
    return float(
        math.pi * radius * radius * coefficients[0]
        + math.pi * radius**4 * (coefficients[3] + coefficients[5]) / 4.0
    )


def centered_annulus_polynomial_integral(coefficients: np.ndarray, inner_radius_deg: float, outer_radius_deg: float) -> float:
    return centered_disk_polynomial_integral(coefficients, outer_radius_deg) - centered_disk_polynomial_integral(
        coefficients, inner_radius_deg
    )


def _required_array(data: Mapping[str, np.ndarray], name: str, *, source: str = "Stage D") -> np.ndarray:
    if name not in data:
        raise ValueError(f"{source} NPZ is missing required array {name!r}")
    return np.asarray(data[name])


def _one_value_by_cell(data: Mapping[str, np.ndarray], name: str, cell_ids: np.ndarray) -> Dict[int, float]:
    values = np.asarray(_required_array(data, name), dtype=np.float64)
    if values.shape != cell_ids.shape:
        raise ValueError(f"Stage D {name} shape {values.shape} does not match cell_id shape {cell_ids.shape}")
    return {int(cell_id): float(value) for cell_id, value in zip(cell_ids, values)}


def prepare_context(
    stage_d: Mapping[str, np.ndarray],
    stage_e: Mapping[str, np.ndarray],
    off_events: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    from apply.stages.poisson_roi_background import fit_profiled_poisson_surface_unbinned  # noqa: F401

    cell_ids = np.asarray(_required_array(stage_d, "cell_id"), dtype=np.int64)
    if cell_ids.ndim != 1 or len(np.unique(cell_ids)) != cell_ids.size:
        raise ValueError("Stage D cell_id must be a one-dimensional array of unique ids")
    index_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}

    target_ids = np.asarray(manifest.get("target_cell_ids", []), dtype=np.int64)
    if target_ids.shape != (44,) or len(np.unique(target_ids)) != 44:
        raise ValueError("Pooling manifest target_cell_ids must contain exactly 44 unique cells in fit order")
    if any(int(cell_id) not in index_by_cell for cell_id in target_ids):
        raise ValueError("Stage D NPZ is missing some target cells")

    stage_e_cell_ids = np.asarray(_required_array(stage_e, "cell_id", source="Stage E"), dtype=np.int64)
    if not np.array_equal(stage_e_cell_ids, target_ids):
        raise ValueError("Stage E cell_id must exactly match pooling manifest target_cell_ids order")
    stage_e_n_on = np.asarray(_required_array(stage_e, "N_on", source="Stage E"), dtype=np.int64)
    if stage_e_n_on.shape != target_ids.shape or np.any(stage_e_n_on < 0):
        raise ValueError("Stage E N_on must contain exactly 44 non-negative integer counts")

    cells_manifest = manifest.get("cells")
    if not isinstance(cells_manifest, dict):
        raise ValueError("Pooling manifest is missing the cells object")

    r_opt = _one_value_by_cell(stage_d, "r_opt_deg", cell_ids)
    annulus_inner = _one_value_by_cell(stage_d, "annulus_inner_deg", cell_ids)
    annulus_outer = _one_value_by_cell(stage_d, "annulus_outer_deg", cell_ids)
    n_ann = _one_value_by_cell(stage_d, "continuous_annulus_counts", cell_ids)
    b_on_nominal_by_cell = _one_value_by_cell(stage_d, "B_on", cell_ids)
    aligned_b_on = np.asarray([b_on_nominal_by_cell[int(c)] for c in target_ids], dtype=np.float64)

    stage_e_b_on = np.asarray(_required_array(stage_e, "B_on", source="Stage E"), dtype=np.float64)
    if not np.array_equal(stage_e_b_on, aligned_b_on):
        raise ValueError("Stage E B_on must exactly match Stage D B_on in manifest target order")

    # OFF events (order-2 donor cells only).
    events_by_cell: Dict[int, np.ndarray] = {}
    dumped_ids = set(int(v) for v in np.asarray(off_events.get("cell_ids", np.empty(0)), dtype=np.int64))
    for cid in dumped_ids:
        key = f"events_{cid}"
        if key not in off_events:
            raise ValueError(f"OFF-event dump is missing array {key!r}")
        events_by_cell[cid] = np.asarray(off_events[key], dtype=np.float64)

    continuous_counts = manifest.get("continuous_annulus_counts")
    if not isinstance(continuous_counts, dict):
        raise ValueError("Pooling manifest is missing continuous_annulus_counts")

    target_specs: Dict[int, Dict[str, object]] = {}
    for target_id in target_ids:
        raw = cells_manifest.get(str(int(target_id)))
        if not isinstance(raw, dict):
            raise ValueError(f"Pooling manifest has no cell specification for target {int(target_id)}")
        donors = tuple(int(v) for v in raw.get("donor_cell_ids", []))
        order = int(raw.get("surface_order", -1))
        if order not in {0, 1, 2}:
            raise ValueError(f"Target {int(target_id)} has invalid surface_order {order}")
        if order == 2:
            for donor in donors:
                if donor not in events_by_cell:
                    raise ValueError(f"OFF-event dump is missing order-2 donor cell {donor} for target {int(target_id)}")
        target_specs[int(target_id)] = {"donors": donors, "order": order}

    shape_contributor = {
        cid: int(continuous_counts[str(cid)]) >= 100
        for cid in events_by_cell
        if str(cid) in continuous_counts
    }

    return {
        "target_ids": target_ids,
        "target_specs": target_specs,
        "events_by_cell": events_by_cell,
        "shape_contributor_by_cell": shape_contributor,
        "r_opt_by_cell": r_opt,
        "annulus_inner_by_cell": annulus_inner,
        "annulus_outer_by_cell": annulus_outer,
        "n_ann_by_cell": n_ann,
        "B_on_nominal": aligned_b_on,
        "N_on": stage_e_n_on,
    }


def _initialize_worker(context: Dict[str, object]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def bootstrap_replicate(replicate_index: int, base_seed: int, context: Optional[Dict[str, object]] = None) -> np.ndarray:
    from apply.stages.poisson_roi_background import fit_profiled_poisson_surface_unbinned

    active = context if context is not None else _WORKER_CONTEXT
    if active is None:
        raise RuntimeError("Bootstrap worker context was not initialized")
    rng = rng_for_replicate(base_seed, replicate_index)
    events_by_cell = active["events_by_cell"]
    target_ids = np.asarray(active["target_ids"], dtype=np.int64)
    target_specs = active["target_specs"]
    radius = float(active["positivity_radius_deg"])
    r_opt = active["r_opt_by_cell"]
    inner = active["annulus_inner_by_cell"]
    outer = active["annulus_outer_by_cell"]
    n_ann = active["n_ann_by_cell"]
    b_on_nom_by_cell = {int(c): float(b) for c, b in zip(target_ids, np.asarray(active["B_on_nominal"], dtype=np.float64))}

    sample = np.empty(target_ids.size, dtype=np.float64)
    for output_index, target_id_raw in enumerate(target_ids):
        target_id = int(target_id_raw)
        spec = target_specs[target_id]
        donors = tuple(int(v) for v in spec["donors"])
        order = int(spec["order"])
        if order == 2:
            resampled: Dict[int, np.ndarray] = {}
            bounds: Dict[int, Tuple[float, float]] = {}
            for donor in donors:
                coords = np.asarray(events_by_cell[donor], dtype=np.float64)
                n_d = coords.shape[0]
                m_d = int(rng.poisson(float(n_d)))
                if m_d <= 0:
                    m_d = 1  # guard: keep the fit well-posed
                idx = rng.integers(0, n_d, size=m_d)
                resampled[donor] = coords[idx]
                bounds[donor] = (float(inner[donor]), min(float(outer[donor]), radius))
            fit = fit_profiled_poisson_surface_unbinned(
                resampled, bounds, donors, order, radius, active["shape_contributor_by_cell"]
            )
            coefficients = np.asarray(fit.shape_coefficients, dtype=np.float64)
            normalization = float(fit.annulus_normalizations[target_id])
            disk_integral = centered_disk_polynomial_integral(coefficients, float(r_opt[target_id]))
            annulus_integral = centered_annulus_polynomial_integral(
                coefficients, float(inner[target_id]), min(float(outer[target_id]), radius)
            )
            if not math.isfinite(annulus_integral) or annulus_integral <= 0.0:
                raise RuntimeError(f"Replicate {replicate_index} target {target_id} invalid annulus integral")
            sample[output_index] = normalization * disk_integral / annulus_integral
        else:
            n_nom = float(n_ann[target_id])
            if n_nom <= 0.0:
                sample[output_index] = 0.0
                continue
            g = b_on_nom_by_cell[target_id] / n_nom  # shape-independent geometric factor
            sample[output_index] = float(rng.poisson(n_nom)) * g
    if not np.all(np.isfinite(sample)) or np.any(sample < 0.0):
        raise RuntimeError(f"Replicate {replicate_index} produced invalid B_on values")
    return sample


def _worker_task(replicate_index: int, base_seed: int) -> Tuple[int, Optional[np.ndarray], Optional[str]]:
    try:
        return replicate_index, bootstrap_replicate(replicate_index, base_seed), None
    except Exception as exc:  # noqa: BLE001
        return replicate_index, None, f"{type(exc).__name__}: {exc}"


def run_bootstrap(context: Dict[str, object], *, replicates: int, seed: int, workers: int, allow_failures: bool):
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if workers < 1:
        raise ValueError("workers must be positive")
    samples_by_index: Dict[int, np.ndarray] = {}
    failures: Dict[int, str] = {}
    if workers == 1:
        _initialize_worker(context)
        for index in range(replicates):
            _, sample, error = _worker_task(index, seed)
            if error is not None or sample is None:
                failures[index] = error or "unknown bootstrap failure"
            else:
                samples_by_index[index] = sample
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker, initargs=(context,)) as executor:
            futures = {executor.submit(_worker_task, index, seed): index for index in range(replicates)}
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
    off_events_path = Path(args.off_events_npz).resolve()
    manifest_path = Path(args.pooling_manifest).resolve()
    output_path = Path(args.output_npz).resolve()
    metadata_path = Path(args.metadata_json).resolve() if args.metadata_json else output_path.with_suffix(".json")
    for label, path in (("Stage D", stage_d_path), ("Stage E", stage_e_path),
                        ("OFF events", off_events_path), ("Manifest", manifest_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} input does not exist: {path}")

    manifest = load_json(manifest_path)
    with np.load(stage_d_path, allow_pickle=False) as data:
        stage_d = {name: data[name].copy() for name in data.files}
    with np.load(stage_e_path, allow_pickle=False) as data:
        stage_e = {name: data[name].copy() for name in data.files}
    with np.load(off_events_path, allow_pickle=False) as data:
        off_events = {name: data[name].copy() for name in data.files}

    context = prepare_context(stage_d, stage_e, off_events, manifest)
    context["positivity_radius_deg"] = float(args.positivity_radius_deg)

    samples, failures = run_bootstrap(
        context, replicates=int(args.replicates), seed=int(args.seed),
        workers=int(args.workers), allow_failures=bool(args.allow_smoke_failures),
    )
    mean = np.mean(samples, axis=0)
    background_covariance = np.cov(samples, rowvar=False, ddof=1)
    n_on = np.asarray(context["N_on"], dtype=np.int64)
    excess_covariance = np.diag(n_on.astype(np.float64)) + background_covariance
    eigenvalues = np.linalg.eigvalsh(excess_covariance)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise RuntimeError(f"Bootstrap excess covariance is not positive definite; min eigenvalue={eigenvalues[0]:.6g}")
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
        "description": "Nominal UNBINNED pooled-Poisson Stage D bootstrap background covariance (event resample + unbinned refit for order-2; analytic Poisson for order<=1).",
        "inputs": {
            "stage_d_npz": str(stage_d_path),
            "stage_e_npz": str(stage_e_path),
            "off_events_npz": str(off_events_path),
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
        "off_events_sha256": sha256_file(off_events_path),
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
