#!/usr/bin/env python3
"""Validate preflight, computation, report, and Slurm contracts for Poisson v6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from apply.report.build_v6_poisson_grid_convergence import (
    BRANCH_IDS,
    NOMINAL_BRANCH_ID,
    PROVENANCE_KEYS,
    GridConvergenceError,
    evaluate_grid_convergence,
    load_registry,
    provenance_gate,
    strict_branch_records,
)


EXPERIMENT_ID = "v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled"
DEFAULT_OUTPUT_ROOT = SCRIPT_REPO_ROOT / f"apply/output/{EXPERIMENT_ID}"
DEFAULT_BRANCHES = DEFAULT_OUTPUT_ROOT / "branches.json"
DEFAULT_VALIDATION = SCRIPT_REPO_ROOT / "scheme_R_double_rayleigh_poisson_pooled_validation.json"
DEFAULT_CONVERGENCE = SCRIPT_REPO_ROOT / "scheme_R_double_rayleigh_poisson_grid_convergence.json"
DEFAULT_BOOTSTRAP = SCRIPT_REPO_ROOT / "scheme_R_double_rayleigh_poisson_background_covariance.npz"
DEFAULT_BOOTSTRAP_METADATA = SCRIPT_REPO_ROOT / "scheme_R_double_rayleigh_poisson_background_covariance.json"
DEFAULT_REPORT = SCRIPT_REPO_ROOT / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_pooled_report.html"


def check(name: str, passed: bool, observed: Any, limit: Any, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "limit": limit, "evidence": evidence}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def preflight_checks(registry_path: Path, repo_root: Path, baseline_sha: str | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    exists = registry_path.exists() and registry_path.stat().st_size > 0
    checks.append(check("branch_registry_exists", exists, str(registry_path), {"exists": True, "nonempty": True}, {}))
    if not exists:
        return checks
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        specs = registry.get("branches") if isinstance(registry, Mapping) else None
        if not isinstance(specs, list):
            raise GridConvergenceError("registry has no branches list")
        by_id = strict_branch_records(specs)
        checks.append(check("registered_branch_ids", True, [item["branch_id"] for item in specs], list(BRANCH_IDS), {}))
        provenance = provenance_gate(by_id).to_dict()
        checks.append(provenance)
    except (OSError, json.JSONDecodeError, GridConvergenceError, KeyError) as exc:
        checks.append(check("registered_branch_ids", False, None, list(BRANCH_IDS), {"error": str(exc)}))
        return checks

    baseline = (baseline_sha or os.environ.get("ANALYTIC_BON_BASELINE_SHA") or "").strip()
    checks.append(check("analytic_baseline_sha_supplied", bool(baseline), baseline or None, {"nonempty": True}, {"source": "argument or ANALYTIC_BON_BASELINE_SHA"}))
    if baseline:
        exists_result = _git(repo_root, "cat-file", "-e", f"{baseline}^{{commit}}")
        ancestor_result = _git(repo_root, "merge-base", "--is-ancestor", baseline, "HEAD")
        checks.append(check("analytic_baseline_is_ancestor", exists_result.returncode == 0 and ancestor_result.returncode == 0, baseline, {"ancestor_of": "HEAD"}, {"cat_file_stderr": exists_result.stderr.strip(), "merge_base_stderr": ancestor_result.stderr.strip()}))
        recorded = {str((spec.get("provenance") or {}).get("analytic_bon_baseline_sha", "")) for spec in specs}
        checks.append(check("analytic_baseline_matches_registry", recorded == {baseline}, sorted(recorded), [baseline], {}))
    head_result = _git(repo_root, "rev-parse", "HEAD")
    head = head_result.stdout.strip() if head_result.returncode == 0 else ""
    recorded_code = {str((spec.get("provenance") or {}).get("code_sha", "")) for spec in specs}
    checks.append(check("code_sha_matches_head", bool(head) and recorded_code == {head}, sorted(recorded_code), [head] if head else {"valid_git_head": True}, {"git_stderr": head_result.stderr.strip()}))

    required_paths = ("stage_d_npz", "stage_d_metadata", "stage_e_npz", "stage_e_metadata", "stage_f_npz", "stage_f_metadata")
    absent = {branch_id: [key for key in required_paths if not by_id[branch_id].get(key)] for branch_id in BRANCH_IDS}
    absent = {branch_id: keys for branch_id, keys in absent.items() if keys}
    checks.append(check("branch_artifact_paths_declared", not absent, absent, {"missing": {}}, {"required_path_keys": list(required_paths)}))
    return checks


def _sacct_state(job_id: str) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "--format=State", "--parsable2"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return "", str(exc)
    states = [line.strip().split("+", 1)[0] for line in result.stdout.splitlines() if line.strip()]
    unique = sorted(set(states))
    return (unique[0] if len(unique) == 1 else ",".join(unique), result.stderr.strip())


def slurm_checks(
    registry: Mapping[str, Any],
    *,
    query_sacct: bool = False,
    current_finalizer_job_id: str | None = None,
) -> list[dict[str, Any]]:
    jobs = registry.get("slurm_jobs")
    if isinstance(jobs, Mapping):
        rows = [{"role": role, **(value if isinstance(value, Mapping) else {"state": value})} for role, value in jobs.items()]
    elif isinstance(jobs, list):
        rows = list(jobs)
    else:
        rows = []
    expanded = []
    for row in rows:
        tasks = row.get("tasks") if isinstance(row, Mapping) else None
        if isinstance(tasks, Mapping):
            for branch_id, task in tasks.items():
                detail = task if isinstance(task, Mapping) else {"state": task}
                expanded.append({"role": str(branch_id), **detail})
        else:
            expanded.append(row)
    states: dict[str, str] = {}
    query_errors: dict[str, str] = {}
    job_ids: dict[str, str | None] = {}
    for idx, row in enumerate(expanded):
        role = str(row.get("branch_id") or row.get("role") or row.get("job_id") or idx)
        job_id = str(row.get("job_id", "")).strip()
        state = str(row.get("state", "")).upper()
        if query_sacct and job_id:
            state, error = _sacct_state(job_id)
            state = state.upper()
            if error:
                query_errors[role] = error
        states[role] = state
        job_ids[role] = job_id or None
    required_roles = {"manifest", "bootstrap", "finalizer", *BRANCH_IDS}
    missing_roles = sorted(required_roles - set(states))
    current_finalizer_allowed = bool(
        current_finalizer_job_id
        and job_ids.get("finalizer") == str(current_finalizer_job_id)
        and states.get("finalizer") in {"RUNNING", "COMPLETING"}
    )
    incomplete = {
        name: state
        for name, state in states.items()
        if state != "COMPLETED" and not (name == "finalizer" and current_finalizer_allowed)
    }
    return [check("all_slurm_jobs_completed", not missing_roles and not incomplete and not query_errors, states, {"every_state": "COMPLETED", "required_roles": sorted(required_roles), "current_finalizer_may_be_running_inside_itself": True}, {"job_ids": job_ids, "missing_roles": missing_roles, "incomplete": incomplete, "query_errors": query_errors, "current_finalizer_exemption_used": current_finalizer_allowed})]


def bootstrap_checks(
    npz_path: Path | None,
    metadata_path: Path | None,
    *,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate the registered nominal-only 1,000-replicate covariance artifact."""
    checks: list[dict[str, Any]] = []
    paths_exist = bool(
        npz_path is not None
        and metadata_path is not None
        and npz_path.exists()
        and npz_path.stat().st_size > 0
        and metadata_path.exists()
        and metadata_path.stat().st_size > 0
    )
    checks.append(check("bootstrap_artifacts_exist", paths_exist, {"npz": str(npz_path) if npz_path else None, "metadata": str(metadata_path) if metadata_path else None}, {"exists": True, "nonempty": True}, {}))
    if not paths_exist or npz_path is None or metadata_path is None:
        return checks
    try:
        with np.load(npz_path, allow_pickle=False) as handle:
            required = {"cell_id", "B_on_nominal", "N_on", "B_on_bootstrap_mean", "B_on_bootstrap_samples", "B_on_covariance", "excess_covariance"}
            missing = sorted(required - set(handle.files))
            if missing:
                raise ValueError(f"missing arrays: {missing}")
            cell_ids = np.asarray(handle["cell_id"], dtype=np.int64)
            b_on_nominal = np.asarray(handle["B_on_nominal"], dtype=np.float64)
            n_on_raw = np.asarray(handle["N_on"])
            n_on = np.asarray(n_on_raw, dtype=np.int64)
            samples = np.asarray(handle["B_on_bootstrap_samples"], dtype=np.float64)
            b_covariance = np.asarray(handle["B_on_covariance"], dtype=np.float64)
            excess_covariance = np.asarray(handle["excess_covariance"], dtype=np.float64)
            stored_mean = np.asarray(handle["B_on_bootstrap_mean"], dtype=np.float64)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape_ok = cell_ids.shape == (44,) and len(set(cell_ids.tolist())) == 44 and b_on_nominal.shape == (44,) and n_on.shape == (44,) and samples.shape == (1000, 44) and b_covariance.shape == (44, 44) and excess_covariance.shape == (44, 44)
        checks.append(check("bootstrap_registered_shape", shape_ok, {"cell_id": list(cell_ids.shape), "B_on_nominal": list(b_on_nominal.shape), "N_on": list(n_on.shape), "samples": list(samples.shape), "B_on_covariance": list(b_covariance.shape), "excess_covariance": list(excess_covariance.shape)}, {"cell_id": [44], "vectors": [44], "samples": [1000, 44], "covariances": [44, 44]}, {}))
        n_on_integer = bool(np.issubdtype(n_on_raw.dtype, np.integer))
        finite = all(np.all(np.isfinite(value)) for value in (b_on_nominal, samples, b_covariance, excess_covariance, stored_mean)) and n_on_integer and bool(np.all(n_on >= 0))
        symmetric = np.allclose(b_covariance, b_covariance.T, rtol=1e-12, atol=1e-12) and np.allclose(excess_covariance, excess_covariance.T, rtol=1e-12, atol=1e-12)
        recomputed_mean = np.mean(samples, axis=0)
        recomputed_covariance = np.cov(samples, rowvar=False, ddof=1)
        reproduction = np.allclose(stored_mean, recomputed_mean, rtol=1e-12, atol=1e-12) and np.allclose(b_covariance, recomputed_covariance, rtol=1e-10, atol=1e-10)
        expected_excess_covariance = np.diag(n_on.astype(np.float64)) + b_covariance if shape_ok else np.empty((0, 0))
        excess_reproduction = shape_ok and np.allclose(excess_covariance, expected_excess_covariance, rtol=1e-12, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(excess_covariance) if excess_covariance.shape == (44, 44) and finite and symmetric else np.asarray([float("nan")])
        checks.append(check("bootstrap_covariance_integrity", finite and symmetric and reproduction and excess_reproduction and bool(np.all(eigenvalues > 0.0)), {"finite": finite, "symmetric": symmetric, "sample_reproduced": reproduction, "excess_formula_reproduced": excess_reproduction, "minimum_excess_eigenvalue": float(np.min(eigenvalues))}, {"finite": True, "symmetric": True, "sample_reproduction": True, "excess_covariance_formula": "diag(N_on) + B_on_covariance", "minimum_eigenvalue_operator": "> 0"}, {"npz_sha256": _sha256(npz_path)}))
        production = metadata.get("bootstrap_count_requested") == 1000 and metadata.get("bootstrap_count_completed") == 1000 and metadata.get("refit_failure_count") == 0 and metadata.get("production_complete") is True and metadata.get("seed") == 64748
        checks.append(check("bootstrap_production_complete", production, {key: metadata.get(key) for key in ("bootstrap_count_requested", "bootstrap_count_completed", "refit_failure_count", "production_complete", "seed")}, {"bootstrap_count_requested": 1000, "bootstrap_count_completed": 1000, "refit_failure_count": 0, "production_complete": True, "seed": 64748}, {"metadata_sha256": _sha256(metadata_path)}))

        if registry_path is not None:
            registry_path = registry_path.resolve()
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            specs = strict_branch_records(registry.get("branches") or [])
            nominal = specs[NOMINAL_BRANCH_ID]

            def registry_path_for(key: str) -> Path:
                raw = nominal.get(key)
                if not raw:
                    raise ValueError(f"Nominal registry branch lacks {key}")
                path = Path(str(raw))
                return (path if path.is_absolute() else registry_path.parent / path).resolve()

            stage_d_path = registry_path_for("stage_d_npz")
            stage_d_metadata_path = registry_path_for("stage_d_metadata")
            stage_e_path = registry_path_for("stage_e_npz")
            stage_f_metadata_path = registry_path_for("stage_f_metadata")
            stage_d_metadata = json.loads(stage_d_metadata_path.read_text(encoding="utf-8"))
            stage_f_metadata = json.loads(stage_f_metadata_path.read_text(encoding="utf-8"))

            def metadata_path_for(payload: Mapping[str, Any], key: str, anchor: Path) -> Path:
                inputs = payload.get("inputs") or {}
                raw = inputs.get(key) if isinstance(inputs, Mapping) else None
                if not raw:
                    raise ValueError(f"{anchor} lacks inputs.{key}")
                path = Path(str(raw))
                return (path if path.is_absolute() else anchor.parent / path).resolve()

            manifest_path = metadata_path_for(stage_d_metadata, "pooling_manifest", stage_d_metadata_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_target_ids = np.asarray(manifest.get("target_cell_ids", []), dtype=np.int64)
            with np.load(stage_d_path, allow_pickle=False) as handle:
                stage_d_ids = np.asarray(handle["cell_id"], dtype=np.int64)
                stage_d_b_on = np.asarray(handle["B_on"], dtype=np.float64)
            with np.load(stage_e_path, allow_pickle=False) as handle:
                stage_e_ids_raw = np.asarray(handle["cell_id"])
                stage_e_ids = np.asarray(stage_e_ids_raw, dtype=np.int64)
                stage_e_n_on_raw = np.asarray(handle["N_on"])
                stage_e_n_on = np.asarray(stage_e_n_on_raw, dtype=np.int64)
                stage_e_b_on = np.asarray(handle["B_on"], dtype=np.float64)
            stage_d_index = {int(cell_id): index for index, cell_id in enumerate(stage_d_ids)}
            aligned_stage_d_b_on = np.asarray([stage_d_b_on[stage_d_index[int(cell_id)]] for cell_id in cell_ids])
            alignment_ok = (
                np.array_equal(cell_ids, manifest_target_ids)
                and np.array_equal(cell_ids, stage_e_ids)
                and np.issubdtype(stage_e_ids_raw.dtype, np.integer)
                and np.issubdtype(stage_e_n_on_raw.dtype, np.integer)
                and np.all(stage_e_n_on >= 0)
                and np.array_equal(n_on, stage_e_n_on)
                and np.allclose(b_on_nominal, aligned_stage_d_b_on, rtol=1e-12, atol=1e-12)
                and np.allclose(stage_e_b_on, aligned_stage_d_b_on, rtol=1e-12, atol=1e-12)
            )
            checks.append(check("bootstrap_nominal_alignment", alignment_ok, {"cell_id": cell_ids.tolist(), "manifest_target_cell_id": manifest_target_ids.tolist(), "stage_e_cell_id": stage_e_ids.tolist(), "stage_e_cell_id_integer": bool(np.issubdtype(stage_e_ids_raw.dtype, np.integer)), "stage_e_N_on_non_negative_integer": bool(np.issubdtype(stage_e_n_on_raw.dtype, np.integer) and np.all(stage_e_n_on >= 0)), "N_on_matches_stage_e": bool(np.array_equal(n_on, stage_e_n_on)), "B_on_nominal_matches_stage_d": bool(np.allclose(b_on_nominal, aligned_stage_d_b_on, rtol=1e-12, atol=1e-12)), "stage_e_B_on_matches_stage_d": bool(np.allclose(stage_e_b_on, aligned_stage_d_b_on, rtol=1e-12, atol=1e-12))}, {"exact_cell_order": True, "stage_e_cell_id_integer": True, "stage_e_N_on_non_negative_integer": True, "N_on_matches_stage_e": True, "B_on_nominal_matches_stage_d": True, "stage_e_B_on_matches_stage_d": True}, {}))

            metadata_stage_d = metadata_path_for(metadata, "stage_d_npz", metadata_path)
            metadata_inputs = metadata.get("inputs") or {}
            metadata_stage_e = (
                metadata_path_for(metadata, "stage_e_npz", metadata_path)
                if isinstance(metadata_inputs, Mapping) and metadata_inputs.get("stage_e_npz")
                else None
            )
            metadata_manifest = metadata_path_for(metadata, "pooling_manifest", metadata_path)
            implementation_sha = str(metadata.get("implementation_commit_sha", ""))
            manifest_implementation_sha = str(manifest.get("implementation_commit_sha", ""))
            registry_implementation_sha = str((nominal.get("provenance") or {}).get("code_sha", ""))
            provenance_ok = (
                metadata_stage_d == stage_d_path
                and metadata_stage_e == stage_e_path
                and metadata_manifest == manifest_path
                and metadata.get("stage_d_sha256") == _sha256(stage_d_path)
                and metadata.get("stage_e_sha256") == _sha256(stage_e_path)
                and metadata.get("manifest_sha256") == manifest.get("manifest_sha256")
                and metadata.get("manifest_file_sha256") == _sha256(manifest_path)
                and metadata.get("cell_id") == cell_ids.tolist()
                and bool(implementation_sha)
                and implementation_sha == manifest_implementation_sha
                and implementation_sha == registry_implementation_sha
            )
            checks.append(check("bootstrap_provenance_linked", provenance_ok, {"stage_d_path_matches": metadata_stage_d == stage_d_path, "stage_e_path_matches": metadata_stage_e == stage_e_path, "manifest_path_matches": metadata_manifest == manifest_path, "stage_d_sha_matches": metadata.get("stage_d_sha256") == _sha256(stage_d_path), "stage_e_sha_matches": metadata.get("stage_e_sha256") == _sha256(stage_e_path), "manifest_self_hash_matches": metadata.get("manifest_sha256") == manifest.get("manifest_sha256"), "manifest_file_sha_matches": metadata.get("manifest_file_sha256") == _sha256(manifest_path), "metadata_cell_id_matches": metadata.get("cell_id") == cell_ids.tolist(), "implementation_commit_matches_manifest": implementation_sha == manifest_implementation_sha, "implementation_commit_matches_registry": implementation_sha == registry_implementation_sha}, {"all": True}, {}))

            fits = stage_f_metadata.get("fits") or {}
            covariance_fit_names = ("pl_background_covariance", "logpar_background_covariance")
            covariance_fits_valid = isinstance(fits, Mapping) and all(isinstance(fits.get(name), Mapping) and fits[name].get("valid") is True for name in covariance_fit_names)
            preferred = stage_f_metadata.get("preferred_fit") or {}
            diagnostic = ((stage_f_metadata.get("statistic") or {}).get("background_covariance_diagnostic") or {})
            covariance_input_path = metadata_path_for(stage_f_metadata, "excess_covariance_npz", stage_f_metadata_path)
            stage_f_linked = (
                covariance_input_path == npz_path.resolve()
                and diagnostic.get("path") == str(npz_path.resolve())
                and diagnostic.get("sha256") == _sha256(npz_path)
                and covariance_fits_valid
                and preferred.get("error_mode") == "conservative"
            )
            checks.append(check("bootstrap_stage_f_covariance_diagnostic", stage_f_linked, {"covariance_input_path_matches": covariance_input_path == npz_path.resolve(), "covariance_sha_matches": diagnostic.get("sha256") == _sha256(npz_path), "covariance_fits_valid": covariance_fits_valid, "preferred_error_mode": preferred.get("error_mode")}, {"covariance_input_and_sha_match": True, "covariance_fits_valid": True, "preferred_error_mode": "conservative"}, {}))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        checks.append(check("bootstrap_covariance_integrity", False, None, {"valid": True}, {"error": str(exc)}))
    return checks


def report_checks(report: Path | None, convergence: Path | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    convergence_payload: dict[str, Any] = {}
    if convergence is not None and convergence.exists():
        convergence_payload = json.loads(convergence.read_text(encoding="utf-8"))
    convergence_ok = bool(convergence_payload) and convergence_payload.get("passed") is True
    checks.append(check("grid_convergence_artifact_passed", convergence_ok, convergence_payload.get("passed") if convergence_payload else None, True, {"path": str(convergence) if convergence else None, "sha256": _sha256(convergence) if convergence and convergence.exists() else None}))
    if report is None:
        return checks
    exists = report.exists() and report.stat().st_size > 0
    text = report.read_text(encoding="utf-8") if exists else ""
    required = ("Poisson", "Pooling", "Grid Convergence", "Covariance", "Legacy")
    missing = [token for token in required if token not in text]
    checks.append(check("final_report_contract", exists and not missing, {"path": str(report), "missing_sections": missing}, {"required_sections": list(required)}, {"sha256": _sha256(report) if exists else None}))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches-json", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--convergence-json", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument("--bootstrap-npz", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--bootstrap-metadata", type=Path, default=DEFAULT_BOOTSTRAP_METADATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_REPO_ROOT)
    parser.add_argument("--analytic-baseline-sha")
    parser.add_argument("--phase", choices=("preflight", "computation", "bootstrap", "report", "slurm", "all"), default="all")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase = "preflight" if args.preflight_only else args.phase
    registry_path = args.branches_json.resolve()
    checks = preflight_checks(registry_path, args.repo_root.resolve(), args.analytic_baseline_sha)
    preflight_passed = all(row["passed"] for row in checks)
    registry: dict[str, Any] = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    convergence_payload: dict[str, Any] | None = None
    if phase in ("computation", "all") and preflight_passed:
        try:
            records, _ = load_registry(registry_path)
            convergence_payload = evaluate_grid_convergence(records)
            checks.extend(convergence_payload["checks"])
        except (OSError, KeyError, ValueError, GridConvergenceError) as exc:
            checks.append(check("grid_convergence_computation", False, None, {"passed": True}, {"error": str(exc)}))
    if phase in ("slurm", "all"):
        checks.extend(
            slurm_checks(
                registry,
                query_sacct=True,
                current_finalizer_job_id=os.environ.get("SLURM_JOB_ID"),
            )
        )
    if phase in ("bootstrap", "all"):
        checks.extend(bootstrap_checks(args.bootstrap_npz, args.bootstrap_metadata, registry_path=registry_path))
    if phase == "report" or args.require_report or phase == "all":
        checks.extend(report_checks(args.report, args.convergence_json))

    passed = bool(checks) and all(row["passed"] for row in checks)
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "phase": phase,
        "passed": passed,
        "checks": checks,
        "branch_registry": str(registry_path),
        "branch_registry_sha256": _sha256(registry_path) if registry_path.exists() else None,
        "convergence": convergence_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Validation {'passed' if passed else 'failed'}: {args.output_json}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
