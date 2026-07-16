#!/usr/bin/env python3
"""Aggregate and validate the registered 12-branch Poisson grid experiment.

The numerical checks in this module are deliberately independent of the
production directory layout.  Production jobs provide a JSON branch registry;
tests and review tools can pass the same records in memory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BRANCH_IDS = tuple(
    f"h{step}_{phase}"
    for step in ("005", "010", "020")
    for phase in ("x0_y0", "xh_y0", "x0_yh", "xh_yh")
)
NOMINAL_BRANCH_ID = "h010_x0_y0"
EXCLUDED_TAIL_CELL_IDS = (13, 26, 39, 52, 65, 78, 91)
DONOR_UNIVERSE_CELL_IDS = tuple(cell_id for cell_id in range(1, 92) if cell_id not in EXCLUDED_TAIL_CELL_IDS)
PROVENANCE_KEYS = (
    "pooling_manifest_sha256",
    "selector_sha256",
    "psf_sha256",
    "stage_c_sha256",
    "response_sha256",
    "analytic_bon_baseline_sha",
    "code_sha",
)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: Any
    limit: Any
    evidence: Any

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


class GridConvergenceError(ValueError):
    """Raised when branch inputs violate the registered experiment contract."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise GridConvergenceError(f"{name} must contain finite values")
    return array


def evaluate_cell_gate(all_sigma_units: Sequence[float]) -> GateResult:
    values = _finite_array(all_sigma_units, "all_sigma_units")
    maximum = float(np.max(np.abs(values)))
    return GateResult(
        "background_all_cells_half_sigma",
        maximum <= 0.5,
        maximum,
        {"operator": "<=", "value": 0.5},
        {"cell_count": int(values.size), "sigma_units": values},
    )


def evaluate_cell_coverage_gate(all_sigma_units: Sequence[float]) -> GateResult:
    values = np.abs(_finite_array(all_sigma_units, "all_sigma_units"))
    fraction = float(np.mean(values <= 0.25))
    return GateResult(
        "background_cells_quarter_sigma_coverage",
        fraction >= 0.9,
        fraction,
        {"operator": ">=", "value": 0.9, "per_cell_limit": 0.25},
        {"passing_cells": int(np.count_nonzero(values <= 0.25)), "cell_count": int(values.size)},
    )


def evaluate_beta_gate(delta_beta: float, sigma_beta: float) -> GateResult:
    delta = abs(float(delta_beta))
    sigma = float(sigma_beta)
    limit = 0.25 * sigma
    passed = math.isfinite(delta) and math.isfinite(sigma) and sigma > 0.0 and delta <= limit
    return GateResult(
        "beta_stability",
        passed,
        delta,
        {"operator": "<=", "value": limit, "sigma_beta_nominal": sigma, "fraction": 0.25},
        {"delta_beta": float(delta_beta)},
    )


def evaluate_pull_gate(delta_pull: float) -> GateResult:
    observed = abs(float(delta_pull))
    return GateResult(
        "pull_stability",
        math.isfinite(observed) and observed <= 0.5,
        observed,
        {"operator": "<=", "value": 0.5},
        {"delta_pull": float(delta_pull)},
    )


def evaluate_mahalanobis_gate(distance: float) -> GateResult:
    observed = float(distance)
    return GateResult(
        "fit_parameter_mahalanobis",
        math.isfinite(observed) and observed <= 0.5,
        observed,
        {"operator": "<=", "value": 0.5},
        {},
    )


def strict_branch_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    ids = [str(record.get("branch_id", "")) for record in records]
    duplicate_ids = sorted({branch_id for branch_id in ids if ids.count(branch_id) > 1})
    missing = sorted(set(BRANCH_IDS) - set(ids))
    unexpected = sorted(set(ids) - set(BRANCH_IDS))
    if duplicate_ids or missing or unexpected or len(ids) != len(BRANCH_IDS):
        raise GridConvergenceError(
            f"Expected exactly the 12 registered branch ids; missing={missing}, "
            f"duplicate={duplicate_ids}, unexpected={unexpected}"
        )
    return {branch_id: record for branch_id, record in zip(ids, records)}


def provenance_gate(records_by_id: Mapping[str, Mapping[str, Any]]) -> GateResult:
    values: dict[str, dict[str, str | None]] = {}
    missing: dict[str, list[str]] = {}
    mismatched: dict[str, dict[str, str | None]] = {}
    malformed: dict[str, dict[str, str | None]] = {}
    for key in PROVENANCE_KEYS:
        values[key] = {}
        for branch_id in BRANCH_IDS:
            provenance = records_by_id[branch_id].get("provenance") or {}
            value = provenance.get(key) if isinstance(provenance, Mapping) else None
            text = str(value).strip() if value is not None else ""
            values[key][branch_id] = text or None
            if not text:
                missing.setdefault(branch_id, []).append(key)
        distinct = {value for value in values[key].values() if value is not None}
        if len(distinct) != 1:
            mismatched[key] = values[key]
        expected_lengths = {64} if key.endswith("sha256") else {40, 64}
        invalid = {
            branch_id: value
            for branch_id, value in values[key].items()
            if value is not None and (len(value) not in expected_lengths or re.fullmatch(r"[0-9a-f]+", value) is None)
        }
        if invalid:
            malformed[key] = invalid
    passed = not missing and not mismatched and not malformed
    return GateResult(
        "shared_provenance_identical",
        passed,
        {key: sorted({value for value in branch_values.values() if value is not None}) for key, branch_values in values.items()},
        {"required_keys": list(PROVENANCE_KEYS), "distinct_values_per_key": 1},
        {"missing": missing, "mismatched": mismatched, "malformed": malformed, "values": values},
    )


def _aligned(record: Mapping[str, Any], key: str, nominal_ids: np.ndarray) -> np.ndarray:
    cell_ids = np.asarray(record["cell_ids"], dtype=np.int64)
    values = np.asarray(record[key])
    if cell_ids.ndim != 1 or values.shape[0] != cell_ids.size:
        raise GridConvergenceError(f"{record.get('branch_id')}: {key} is not aligned with cell_ids")
    if len(set(cell_ids.tolist())) != cell_ids.size:
        raise GridConvergenceError(f"{record.get('branch_id')}: duplicate cell ids")
    index = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    missing = [int(cell_id) for cell_id in nominal_ids if int(cell_id) not in index]
    extra = sorted(set(cell_ids.tolist()) - set(nominal_ids.tolist()))
    if missing or extra:
        raise GridConvergenceError(
            f"{record.get('branch_id')}: selected-cell contract differs; missing={missing}, extra={extra}"
        )
    return np.asarray([values[index[int(cell_id)]] for cell_id in nominal_ids])


def _mahalanobis(delta: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (delta.size, delta.size) or not np.all(np.isfinite(covariance)):
        return float("inf")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0.0:
        return float("inf")
    return float(math.sqrt(max(0.0, delta @ np.linalg.solve(covariance, delta))))


def evaluate_grid_convergence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate every registered numerical and provenance gate."""
    by_id = strict_branch_records(records)
    nominal = by_id[NOMINAL_BRANCH_ID]
    nominal_ids = np.asarray(nominal["cell_ids"], dtype=np.int64)
    if nominal_ids.size != 44 or len(set(nominal_ids.tolist())) != 44:
        raise GridConvergenceError(f"Nominal branch must contain exactly 44 unique cells, got {nominal_ids.size}")

    b_nominal = _aligned(nominal, "b_on", nominal_ids).astype(np.float64)
    sigma = _aligned(nominal, "sigma_excess", nominal_ids).astype(np.float64)
    pull_nominal = _aligned(nominal, "pull", nominal_ids).astype(np.float64)
    n_on_nominal = _aligned(nominal, "n_on", nominal_ids)
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise GridConvergenceError("Nominal sigma_excess values must be finite and positive")

    b_by_branch: dict[str, np.ndarray] = {}
    pull_by_branch: dict[str, np.ndarray] = {}
    fit_parameters: dict[str, np.ndarray] = {}
    for branch_id in BRANCH_IDS:
        record = by_id[branch_id]
        b_by_branch[branch_id] = _aligned(record, "b_on", nominal_ids).astype(np.float64)
        pull_by_branch[branch_id] = _aligned(record, "pull", nominal_ids).astype(np.float64)
        parameters = np.asarray(record["fit_parameters"], dtype=np.float64)
        if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
            raise GridConvergenceError(f"{branch_id}: fit_parameters must be finite [log10_phi0, alpha, beta]")
        fit_parameters[branch_id] = parameters

    all_b_delta_units = {
        branch_id: np.abs(values - b_nominal) / sigma for branch_id, values in b_by_branch.items()
    }
    b_envelope_units = np.max(np.stack(list(all_b_delta_units.values())), axis=0)

    phase_rms_by_resolution: dict[str, np.ndarray] = {}
    for step in ("005", "010", "020"):
        phase_values = np.stack([b_by_branch[f"h{step}_{phase}"] for phase in ("x0_y0", "xh_y0", "x0_yh", "xh_yh")])
        phase_rms_by_resolution[f"h{step}"] = np.sqrt(np.mean((phase_values - np.mean(phase_values, axis=0)) ** 2, axis=0)) / sigma
    phase_rms_envelope = np.max(np.stack(list(phase_rms_by_resolution.values())), axis=0)

    pull_delta = {branch_id: values - pull_nominal for branch_id, values in pull_by_branch.items()}
    pull_envelope = np.max(np.abs(np.stack(list(pull_delta.values()))), axis=0)
    migrations = []
    nominal_above = np.abs(pull_nominal) >= 5.0
    for branch_id, values in pull_by_branch.items():
        changed = np.flatnonzero((np.abs(values) >= 5.0) != nominal_above)
        migrations.extend({"branch_id": branch_id, "cell_id": int(nominal_ids[idx])} for idx in changed)

    nominal_covariance = np.asarray(nominal["fit_covariance"], dtype=np.float64)
    distances = {
        branch_id: _mahalanobis(parameters - fit_parameters[NOMINAL_BRANCH_ID], nominal_covariance)
        for branch_id, parameters in fit_parameters.items()
    }
    sigma_beta = math.sqrt(float(nominal_covariance[2, 2])) if nominal_covariance.shape == (3, 3) and nominal_covariance[2, 2] > 0.0 else float("nan")
    beta_deltas = {branch_id: abs(float(parameters[2] - fit_parameters[NOMINAL_BRANCH_ID][2])) for branch_id, parameters in fit_parameters.items()}

    d_valid_failures: list[dict[str, Any]] = []
    d_positive_failures: list[dict[str, Any]] = []
    f_valid_failures: list[str] = []
    e_contract_failures: list[str] = []
    n_on_mismatches: list[str] = []
    provenance_verification_failures: dict[str, Any] = {}
    for branch_id, record in by_id.items():
        d_cell_ids = np.asarray(record["stage_d_cell_ids"], dtype=np.int64)
        d_valid = np.asarray(record["stage_d_valid"], dtype=bool)
        d_positive = np.asarray(record["stage_d_positive"], dtype=bool)
        if d_cell_ids.shape != (84,) or set(d_cell_ids.tolist()) != set(DONOR_UNIVERSE_CELL_IDS):
            raise GridConvergenceError(f"{branch_id}: Stage D must contain the exact registered 84-cell donor universe")
        if d_valid.shape != (84,) or d_positive.shape != (84,):
            raise GridConvergenceError(f"{branch_id}: Stage D validity and positivity arrays must have length 84")
        d_valid_failures.extend({"branch_id": branch_id, "cell_id": int(cell_id)} for cell_id in d_cell_ids[~d_valid])
        d_positive_failures.extend({"branch_id": branch_id, "cell_id": int(cell_id)} for cell_id in d_cell_ids[~d_positive])
        if record.get("stage_f_valid") is not True:
            f_valid_failures.append(branch_id)
        if record.get("stage_e_contract_valid") is not True:
            e_contract_failures.append(branch_id)
        if not np.array_equal(_aligned(record, "n_on", nominal_ids), n_on_nominal):
            n_on_mismatches.append(branch_id)
        verification = record.get("provenance_verification") or {}
        if not isinstance(verification, Mapping) or verification.get("passed") is not True:
            provenance_verification_failures[branch_id] = verification

    cell_gate = evaluate_cell_gate(b_envelope_units)
    coverage_gate = evaluate_cell_coverage_gate(b_envelope_units)
    phase_all = evaluate_cell_gate(phase_rms_envelope)
    phase_all = GateResult("phase_rms_all_cells_half_sigma", phase_all.passed, phase_all.observed, phase_all.limit, phase_all.evidence)
    phase_coverage = evaluate_cell_coverage_gate(phase_rms_envelope)
    phase_coverage = GateResult("phase_rms_cells_quarter_sigma_coverage", phase_coverage.passed, phase_coverage.observed, phase_coverage.limit, phase_coverage.evidence)
    mahalanobis_gate = evaluate_mahalanobis_gate(max(distances.values()))
    mahalanobis_gate = GateResult(mahalanobis_gate.name, mahalanobis_gate.passed, mahalanobis_gate.observed, mahalanobis_gate.limit, distances)
    beta_gate = evaluate_beta_gate(max(beta_deltas.values()), sigma_beta)
    beta_gate = GateResult(beta_gate.name, beta_gate.passed, beta_gate.observed, beta_gate.limit, beta_deltas)
    pull_gate = evaluate_pull_gate(float(np.max(pull_envelope)))
    pull_gate = GateResult(pull_gate.name, pull_gate.passed, pull_gate.observed, pull_gate.limit, {"per_cell": dict(zip(nominal_ids.tolist(), pull_envelope.tolist()))})

    checks = [
        GateResult("registered_branch_ids", True, list(BRANCH_IDS), {"exact_ids": list(BRANCH_IDS), "count": 12}, {}),
        provenance_gate(by_id),
        GateResult("artifact_provenance_verified", not provenance_verification_failures, len(provenance_verification_failures), {"operator": "==", "value": 0}, provenance_verification_failures),
        cell_gate,
        coverage_gate,
        phase_all,
        phase_coverage,
        mahalanobis_gate,
        beta_gate,
        pull_gate,
        GateResult("no_pull_five_sigma_migration", not migrations, len(migrations), {"operator": "==", "value": 0}, migrations),
        GateResult("all_stage_d_fits_valid", not d_valid_failures, len(d_valid_failures), {"operator": "==", "value": 0}, d_valid_failures),
        GateResult("all_stage_d_surfaces_positive", not d_positive_failures, len(d_positive_failures), {"operator": "==", "value": 0}, d_positive_failures),
        GateResult("all_stage_f_fits_valid", not f_valid_failures, len(f_valid_failures), {"operator": "==", "value": 0}, f_valid_failures),
        GateResult("stage_e_background_matches_stage_d", not e_contract_failures, len(e_contract_failures), {"operator": "==", "value": 0}, e_contract_failures),
        GateResult("event_level_n_on_identical", not n_on_mismatches, len(n_on_mismatches), {"operator": "==", "value": 0}, n_on_mismatches),
    ]
    branch_rows = []
    for branch_id in BRANCH_IDS:
        params = fit_parameters[branch_id]
        branch_rows.append({
            "branch_id": branch_id,
            "fit_parameters": {"log10_phi0": params[0], "phi0": 10.0 ** params[0], "alpha": params[1], "beta": params[2]},
            "mahalanobis_from_nominal": distances[branch_id],
            "delta_beta": beta_deltas[branch_id],
            "max_abs_delta_pull": float(np.max(np.abs(pull_delta[branch_id]))),
            "max_abs_delta_b_on_sigma": float(np.max(all_b_delta_units[branch_id])),
            "provenance": by_id[branch_id]["provenance"],
        })
    cell_rows = []
    for idx, cell_id in enumerate(nominal_ids):
        cell_rows.append({
            "cell_id": int(cell_id),
            "sigma_excess_nominal": sigma[idx],
            "B_on_nominal": b_nominal[idx],
            "B_on_envelope_sigma": b_envelope_units[idx],
            "phase_rms_envelope_sigma": phase_rms_envelope[idx],
            "pull_nominal": pull_nominal[idx],
            "pull_envelope": pull_envelope[idx],
            "B_on_by_branch": {branch_id: values[idx] for branch_id, values in b_by_branch.items()},
            "pull_by_branch": {branch_id: values[idx] for branch_id, values in pull_by_branch.items()},
        })
    return _json_ready({
        "schema_version": 1,
        "experiment": "v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled",
        "nominal_branch_id": NOMINAL_BRANCH_ID,
        "passed": all(check.passed for check in checks),
        "checks": [check.to_dict() for check in checks],
        "branches": branch_rows,
        "cells": cell_rows,
        "phase_rms_sigma_by_resolution": phase_rms_by_resolution,
    })


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata_input_path(metadata: Mapping[str, Any], key: str, metadata_path: Path) -> Path:
    inputs = metadata.get("inputs") or {}
    raw = inputs.get(key) if isinstance(inputs, Mapping) else None
    if not raw:
        raise GridConvergenceError(f"{metadata_path}: metadata lacks inputs.{key}")
    path = Path(str(raw))
    path = path if path.is_absolute() else (metadata_path.parent / path).resolve()
    if not path.exists() or path.stat().st_size == 0:
        raise GridConvergenceError(f"{metadata_path}: inputs.{key} is missing or empty: {path}")
    return path


def _expected_grid(branch_id: str) -> tuple[float, float, float]:
    step = {"005": 0.05, "010": 0.1, "020": 0.2}[branch_id[1:4]]
    phase = branch_id[5:]
    x_fraction = 0.5 if phase in {"xh_y0", "xh_yh"} else 0.0
    y_fraction = 0.5 if phase in {"x0_yh", "xh_yh"} else 0.0
    return step, x_fraction, y_fraction


def verify_artifact_provenance(
    branch_id: str,
    claimed: Mapping[str, Any],
    d_metadata: Mapping[str, Any],
    d_metadata_path: Path,
    f_metadata: Mapping[str, Any],
    f_metadata_path: Path,
) -> dict[str, Any]:
    """Recompute all file-backed provenance and the registered grid phase."""
    manifest_path = _metadata_input_path(d_metadata, "pooling_manifest", d_metadata_path)
    response_path = _metadata_input_path(f_metadata, "response_npz", f_metadata_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    declared_manifest_sha = str(unsigned.pop("manifest_sha256", ""))
    actual_manifest_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    manifest_provenance = manifest.get("provenance") or {}

    def manifest_artifact(name: str) -> tuple[Path, str]:
        entry = manifest_provenance.get(name) if isinstance(manifest_provenance, Mapping) else None
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise GridConvergenceError(f"{manifest_path}: manifest lacks provenance.{name} path/SHA")
        path = Path(str(entry["path"]))
        path = path if path.is_absolute() else (manifest_path.parent / path).resolve()
        if not path.exists() or path.stat().st_size == 0:
            raise GridConvergenceError(f"{manifest_path}: provenance.{name} is missing or empty: {path}")
        actual_sha = _sha256(path)
        if actual_sha != str(entry["sha256"]):
            raise GridConvergenceError(f"{manifest_path}: provenance.{name} SHA mismatch")
        return path, actual_sha

    selector_path, selector_sha = manifest_artifact("selector")
    psf_path, psf_sha = manifest_artifact("psf_npz")
    stage_c_path, stage_c_sha = manifest_artifact("stage_c_metadata")
    model = d_metadata.get("background_model") or {}
    grid = d_metadata.get("grid") or {}
    step, x_fraction, y_fraction = _expected_grid(branch_id)
    actual = {
        "pooling_manifest_sha256": actual_manifest_sha,
        "selector_sha256": selector_sha,
        "psf_sha256": psf_sha,
        "stage_c_sha256": stage_c_sha,
        "response_sha256": _sha256(response_path),
        "analytic_bon_baseline_sha": str(model.get("analytic_bon_baseline_sha", "")),
        "code_sha": str(manifest.get("implementation_commit_sha", "")),
    }
    mismatches = {
        key: {"claimed": str(claimed.get(key, "")), "actual": value}
        for key, value in actual.items()
        if str(claimed.get(key, "")) != value
    }
    grid_ok = (
        math.isclose(float(grid.get("grid_step_deg", float("nan"))), step, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(grid.get("offset_x_fraction", float("nan"))), x_fraction, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(grid.get("offset_y_fraction", float("nan"))), y_fraction, rel_tol=0.0, abs_tol=1e-12)
    )
    contract_ok = (
        declared_manifest_sha == actual_manifest_sha
        and model.get("pooling_manifest_sha256") == actual_manifest_sha
        and model.get("fit_statistic") == "poisson"
        and grid_ok
        and not mismatches
    )
    return {
        "passed": contract_ok,
        "actual": actual,
        "mismatches": mismatches,
        "manifest_self_hash_matches": declared_manifest_sha == actual_manifest_sha,
        "grid_matches_branch_id": grid_ok,
        "paths": {
            "pooling_manifest": str(manifest_path), "selector": str(selector_path), "psf": str(psf_path),
            "stage_c_metadata": str(stage_c_path), "response": str(response_path),
        },
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def _first(data: Mapping[str, Any], names: Iterable[str], context: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise GridConvergenceError(f"{context} is missing one of {list(names)}")


def _fit_payload(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    fits = metadata.get("fits") or {}
    for name in ("logpar_conservative", "logpar"):
        value = fits.get(name) if isinstance(fits, Mapping) else None
        if isinstance(value, Mapping):
            return value
    raise GridConvergenceError("Stage F metadata lacks fits.logpar_conservative")


def load_branch_record(spec: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    """Load one production branch from its explicit Stage D/E/F registry entry."""
    branch_id = str(spec.get("branch_id", ""))

    def path_for(key: str) -> Path:
        raw = spec.get(key)
        if not raw:
            raise GridConvergenceError(f"{branch_id}: missing registry path {key}")
        path = Path(str(raw))
        return path if path.is_absolute() else (base_dir / path).resolve()

    d_path, d_meta_path, e_path, f_path, f_meta_path = map(path_for, ("stage_d_npz", "stage_d_metadata", "stage_e_npz", "stage_f_npz", "stage_f_metadata"))
    for path in (d_path, d_meta_path, e_path, f_path, f_meta_path):
        if not path.exists() or path.stat().st_size == 0:
            raise GridConvergenceError(f"{branch_id}: missing or empty production artifact {path}")
    d, e, f = _load_npz(d_path), _load_npz(e_path), _load_npz(f_path)
    d_meta = json.loads(d_meta_path.read_text(encoding="utf-8"))
    f_meta = json.loads(f_meta_path.read_text(encoding="utf-8"))
    fit = _fit_payload(f_meta)
    parameters = fit.get("fit_parameters") or {}
    if not parameters:
        physical = fit.get("parameters") or {}
        phi0 = float(physical["phi0"])
        parameters = {"log10_phi0": math.log10(phi0), "alpha": physical["alpha"], "beta": physical["beta"]}
    cell_ids = np.asarray(_first(f, ("cell_id",), f"{branch_id} Stage F"), dtype=np.int64)

    def aligned_from(data: Mapping[str, Any], names: Sequence[str], context: str) -> np.ndarray:
        source_ids = np.asarray(_first(data, ("cell_id",), context), dtype=np.int64)
        values = np.asarray(_first(data, names, context))
        index = {int(cell_id): idx for idx, cell_id in enumerate(source_ids)}
        try:
            return np.asarray([values[index[int(cell_id)]] for cell_id in cell_ids])
        except KeyError as exc:
            raise GridConvergenceError(f"{context} lacks selected cell {exc.args[0]}") from exc

    d_cell_ids = np.asarray(_first(d, ("cell_id",), f"{branch_id} Stage D"), dtype=np.int64)
    d_valid_raw = np.asarray(_first(d, ("surface_fit_success", "poisson_fit_success"), f"{branch_id} Stage D"))
    if "on_aperture_surface_min" in d:
        d_positive_raw = np.asarray(d["on_aperture_surface_min"]) > 0.0
    else:
        d_positive_raw = np.asarray(_first(d, ("positive_minimum",), f"{branch_id} Stage D")) > 0.0
    covariance = np.asarray(fit.get("covariance"), dtype=np.float64)
    d_b_on = aligned_from(d, ("B_on",), f"{branch_id} Stage D")
    e_b_on = aligned_from(e, ("B_on",), f"{branch_id} Stage E")
    provenance = dict(spec.get("provenance") or {})
    provenance_verification = verify_artifact_provenance(
        branch_id, provenance, d_meta, d_meta_path, f_meta, f_meta_path
    )
    return {
        "branch_id": branch_id,
        "cell_ids": cell_ids,
        "b_on": d_b_on,
        "n_on": aligned_from(e, ("N_on",), f"{branch_id} Stage E"),
        "sigma_excess": aligned_from(f, ("excess_err_conservative", "excess_error_conservative", "conservative_error", "sigma_excess", "excess_error"), f"{branch_id} Stage F"),
        "pull": aligned_from(f, ("logpar_conservative_pull", "logpar_pull"), f"{branch_id} Stage F"),
        "fit_parameters": np.asarray([parameters["log10_phi0"], parameters["alpha"], parameters["beta"]], dtype=np.float64),
        "fit_covariance": covariance,
        "stage_d_cell_ids": d_cell_ids,
        "stage_d_valid": d_valid_raw.astype(bool),
        "stage_d_positive": d_positive_raw.astype(bool),
        "stage_f_valid": fit.get("valid") is True,
        "stage_e_contract_valid": bool(np.array_equal(d_b_on, e_b_on)),
        "provenance": provenance,
        "provenance_verification": provenance_verification,
        "paths": {"stage_d_npz": str(d_path), "stage_d_metadata": str(d_meta_path), "stage_e_npz": str(e_path), "stage_f_npz": str(f_path), "stage_f_metadata": str(f_meta_path)},
        "artifact_sha256": {"stage_d_npz": _sha256(d_path), "stage_d_metadata": _sha256(d_meta_path), "stage_e_npz": _sha256(e_path), "stage_f_npz": _sha256(f_path), "stage_f_metadata": _sha256(f_meta_path)},
        "model_tier": spec.get("model_tier"),
        "donor_cell_ids": spec.get("donor_cell_ids"),
        "cv_deviance": spec.get("cv_deviance"),
    }


def load_registry(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("branches") if isinstance(payload, Mapping) else None
    if not isinstance(specs, list):
        raise GridConvergenceError("Branch registry must be an object with a branches list")
    strict_branch_records(specs)
    records = [load_branch_record(spec, path.parent) for spec in specs]
    return records, dict(payload)


def write_diagnostic_plots(payload: Mapping[str, Any], output_dir: Path) -> list[str]:
    """Write the registered diagnostic plot families from an aggregate payload."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise GridConvergenceError("matplotlib is required when --plot-dir is requested") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = payload["cells"]
    branches = payload["branches"]
    cell_ids = np.asarray([row["cell_id"] for row in cells])
    figures: list[tuple[str, Any]] = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cell_ids, [row["B_on_envelope_sigma"] for row in cells], "o-", label="resolution envelope")
    ax.plot(cell_ids, [row["phase_rms_envelope_sigma"] for row in cells], "s-", label="phase RMS envelope")
    ax.axhline(0.5, color="tab:red", linestyle="--"); ax.axhline(0.25, color="0.4", linestyle=":")
    ax.set(xlabel="cell id", ylabel="sigma(excess) units"); ax.legend(); figures.append(("b_on_grid_envelopes.png", fig))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cell_ids, [row["pull_envelope"] for row in cells], "o-")
    ax.axhline(0.5, color="tab:red", linestyle="--"); ax.set(xlabel="cell id", ylabel="max |delta pull|")
    figures.append(("pull_envelopes.png", fig))

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for index, name in enumerate(("phi0", "alpha", "beta")):
        axes[index].plot(range(12), [row["fit_parameters"][name] for row in branches], "o-"); axes[index].set_ylabel(name)
    axes[-1].set_xticks(range(12), [row["branch_id"] for row in branches], rotation=45, ha="right")
    figures.append(("fit_parameter_branches.png", fig))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(12), [row["mahalanobis_from_nominal"] for row in branches]); ax.axhline(0.5, color="tab:red", linestyle="--")
    ax.set_xticks(range(12), [row["branch_id"] for row in branches], rotation=45, ha="right"); ax.set_ylabel("Mahalanobis distance")
    figures.append(("fit_parameter_mahalanobis.png", fig))

    diagnostics = payload.get("diagnostics") or {}
    pooling_rows = diagnostics.get("pooling_rows") if isinstance(diagnostics, Mapping) else None
    if not isinstance(pooling_rows, list) or not pooling_rows:
        raise GridConvergenceError("Plot production requires diagnostics.pooling_rows from the frozen manifest")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    pool_cells = [int(row["cell_id"]) for row in pooling_rows]
    axes[0].step(pool_cells, [int(row["surface_order"]) for row in pooling_rows], where="mid")
    axes[0].set_ylabel("surface order")
    axes[1].bar(pool_cells, [len(row["donor_cell_ids"]) for row in pooling_rows])
    axes[1].set(xlabel="cell id", ylabel="donor count")
    figures.append(("model_tier_donor_map.png", fig))

    covariance = np.asarray(diagnostics.get("B_on_covariance"), dtype=np.float64)
    if covariance.shape != (44, 44) or not np.all(np.isfinite(covariance)):
        raise GridConvergenceError("Plot production requires a finite 44x44 B_on covariance")
    diagonal = np.sqrt(np.diag(covariance))
    correlation = np.divide(
        covariance,
        np.outer(diagonal, diagonal),
        out=np.zeros_like(covariance),
        where=np.outer(diagonal, diagonal) > 0.0,
    )
    for name, matrix, label, bounds in (
        ("background_covariance_heatmap.png", covariance, "B_on covariance", None),
        ("background_correlation_heatmap.png", correlation, "B_on correlation", (-1.0, 1.0)),
    ):
        fig, ax = plt.subplots(figsize=(7, 6))
        kwargs = {} if bounds is None else {"vmin": bounds[0], "vmax": bounds[1]}
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", **kwargs)
        fig.colorbar(image, ax=ax, label=label); ax.set(xlabel="cell index", ylabel="cell index")
        figures.append((name, fig))

    sed = diagnostics.get("sed_ratio") if isinstance(diagnostics, Mapping) else None
    if not isinstance(sed, Mapping):
        raise GridConvergenceError("Plot production requires diagnostics.sed_ratio")
    energy = np.asarray(sed.get("energy_tev"), dtype=np.float64)
    ratio = np.asarray(sed.get("ratio"), dtype=np.float64)
    if energy.ndim != 1 or ratio.shape != energy.shape or energy.size == 0 or np.any(energy <= 0.0):
        raise GridConvergenceError("Legacy-versus-new SED ratio arrays are invalid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(energy, ratio, "o-"); ax.axhline(1.0, color="0.3", linestyle="--")
    ax.set(xlabel="energy [TeV]", ylabel="Poisson pooled / legacy SED")
    figures.append(("legacy_vs_new_sed_ratio.png", fig))

    for filename, fig in figures:
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)
    cv_path = output_dir / "cv_deviance_table.csv"
    with cv_path.open("w", encoding="utf-8") as handle:
        handle.write("cell_id,surface_order,donor_cell_ids,cv_deviance\n")
        for row in pooling_rows:
            donors = ";".join(str(value) for value in row["donor_cell_ids"])
            cv = json.dumps(row.get("cv_deviance"), sort_keys=True, separators=(";", ":"))
            handle.write(f"{int(row['cell_id'])},{int(row['surface_order'])},{donors},{cv}\n")
    return [filename for filename, _ in figures] + [cv_path.name]


def _resolve_auxiliary_path(registry_path: Path, value: Any, name: str) -> Path:
    if not value:
        raise GridConvergenceError(f"Registry diagnostics must declare {name}")
    path = Path(str(value))
    path = path if path.is_absolute() else (registry_path.parent / path).resolve()
    if not path.exists() or path.stat().st_size == 0:
        raise GridConvergenceError(f"Registry diagnostic {name} is missing or empty: {path}")
    return path


def load_plot_diagnostics(registry: Mapping[str, Any], registry_path: Path) -> dict[str, Any]:
    """Load the manifest, bootstrap covariance, and SED pair needed by plots."""
    configured = registry.get("diagnostics") or {}
    if not isinstance(configured, Mapping):
        raise GridConvergenceError("Registry diagnostics must be an object")
    manifest_path = _resolve_auxiliary_path(registry_path, configured.get("pooling_manifest"), "pooling_manifest")
    bootstrap_path = _resolve_auxiliary_path(registry_path, configured.get("bootstrap_npz"), "bootstrap_npz")
    legacy_path = _resolve_auxiliary_path(registry_path, configured.get("legacy_sed_npz"), "legacy_sed_npz")
    nominal_path = _resolve_auxiliary_path(registry_path, configured.get("nominal_sed_npz"), "nominal_sed_npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_ids = [int(value) for value in manifest.get("target_cell_ids", [])]
    cells = manifest.get("cells") or {}
    pooling_rows = []
    for cell_id in target_ids:
        row = cells.get(str(cell_id)) if isinstance(cells, Mapping) else None
        if not isinstance(row, Mapping):
            raise GridConvergenceError(f"Pooling manifest lacks target cell {cell_id}")
        pooling_rows.append({
            "cell_id": cell_id,
            "surface_order": int(row["surface_order"]),
            "donor_cell_ids": [int(value) for value in row["donor_cell_ids"]],
            "cv_deviance": row.get("cv_scores") or row.get("validation_scores") or row.get("cv_deviance"),
        })
    with np.load(bootstrap_path, allow_pickle=False) as handle:
        covariance = np.asarray(handle["B_on_covariance"], dtype=np.float64)

    def sed_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as handle:
            energy = np.asarray(_first(handle, ("energy_tev", "energy_center_tev", "e_ref_tev"), str(path)), dtype=np.float64)
            flux = np.asarray(_first(handle, ("flux", "dnde", "flux_per_tev_cm2_s", "e2_dnde"), str(path)), dtype=np.float64)
        return energy, flux

    legacy_energy, legacy_flux = sed_arrays(legacy_path)
    nominal_energy, nominal_flux = sed_arrays(nominal_path)
    if nominal_energy.shape != nominal_flux.shape or legacy_energy.shape != legacy_flux.shape:
        raise GridConvergenceError("SED energy and flux arrays have inconsistent shapes")
    if np.array_equal(nominal_energy, legacy_energy):
        legacy_at_nominal = legacy_flux
    else:
        if np.any(nominal_energy <= 0.0) or np.any(legacy_energy <= 0.0) or np.any(legacy_flux <= 0.0):
            raise GridConvergenceError("SED interpolation requires positive energies and legacy flux")
        legacy_at_nominal = np.exp(np.interp(np.log(nominal_energy), np.log(legacy_energy), np.log(legacy_flux)))
    ratio = np.divide(nominal_flux, legacy_at_nominal, out=np.full_like(nominal_flux, np.nan), where=legacy_at_nominal != 0.0)
    if not np.all(np.isfinite(ratio)):
        raise GridConvergenceError("Legacy-versus-new SED ratio contains non-finite values")
    return {
        "pooling_rows": pooling_rows,
        "B_on_covariance": covariance,
        "sed_ratio": {"energy_tev": nominal_energy, "ratio": ratio},
        "sources": {
            "pooling_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "bootstrap_npz": {"path": str(bootstrap_path), "sha256": _sha256(bootstrap_path)},
            "legacy_sed_npz": {"path": str(legacy_path), "sha256": _sha256(legacy_path)},
            "nominal_sed_npz": {"path": str(nominal_path), "sha256": _sha256(nominal_path)},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches-json", type=Path, required=True, help="Explicit 12-branch production registry")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, registry = load_registry(args.branches_json.resolve())
    payload = evaluate_grid_convergence(records)
    payload["registry_path"] = str(args.branches_json.resolve())
    payload["registry_sha256"] = _sha256(args.branches_json.resolve())
    payload["slurm_jobs"] = registry.get("slurm_jobs")
    if args.plot_dir is not None:
        payload["diagnostics"] = _json_ready(load_plot_diagnostics(registry, args.branches_json.resolve()))
        payload["plots"] = write_diagnostic_plots(payload, args.plot_dir.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    print(f"Grid convergence {'passed' if payload['passed'] else 'failed'}: {args.output_json}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
