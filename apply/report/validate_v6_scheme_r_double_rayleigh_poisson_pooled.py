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

    required_paths = ("stage_d_npz", "stage_d_metadata", "stage_e_npz", "stage_f_npz", "stage_f_metadata")
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


def slurm_checks(registry: Mapping[str, Any], *, query_sacct: bool = False) -> list[dict[str, Any]]:
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
    required_roles = {"manifest", "bootstrap", *BRANCH_IDS}
    missing_roles = sorted(required_roles - set(states))
    incomplete = {name: state for name, state in states.items() if state != "COMPLETED"}
    return [check("all_slurm_jobs_completed", not missing_roles and not incomplete and not query_errors, states, {"every_state": "COMPLETED", "required_roles": sorted(required_roles)}, {"job_ids": job_ids, "missing_roles": missing_roles, "incomplete": incomplete, "query_errors": query_errors})]


def bootstrap_checks(npz_path: Path | None, metadata_path: Path | None) -> list[dict[str, Any]]:
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
            required = {"cell_id", "B_on_nominal", "B_on_bootstrap_mean", "B_on_bootstrap_samples", "B_on_covariance", "excess_covariance"}
            missing = sorted(required - set(handle.files))
            if missing:
                raise ValueError(f"missing arrays: {missing}")
            cell_ids = np.asarray(handle["cell_id"], dtype=np.int64)
            samples = np.asarray(handle["B_on_bootstrap_samples"], dtype=np.float64)
            b_covariance = np.asarray(handle["B_on_covariance"], dtype=np.float64)
            excess_covariance = np.asarray(handle["excess_covariance"], dtype=np.float64)
            stored_mean = np.asarray(handle["B_on_bootstrap_mean"], dtype=np.float64)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape_ok = cell_ids.shape == (44,) and len(set(cell_ids.tolist())) == 44 and samples.shape == (1000, 44) and b_covariance.shape == (44, 44) and excess_covariance.shape == (44, 44)
        checks.append(check("bootstrap_registered_shape", shape_ok, {"cell_id": list(cell_ids.shape), "samples": list(samples.shape), "B_on_covariance": list(b_covariance.shape), "excess_covariance": list(excess_covariance.shape)}, {"cell_id": [44], "samples": [1000, 44], "covariances": [44, 44]}, {}))
        finite = all(np.all(np.isfinite(value)) for value in (samples, b_covariance, excess_covariance, stored_mean))
        symmetric = np.allclose(b_covariance, b_covariance.T, rtol=1e-12, atol=1e-12) and np.allclose(excess_covariance, excess_covariance.T, rtol=1e-12, atol=1e-12)
        recomputed_mean = np.mean(samples, axis=0)
        recomputed_covariance = np.cov(samples, rowvar=False, ddof=1)
        reproduction = np.allclose(stored_mean, recomputed_mean, rtol=1e-12, atol=1e-12) and np.allclose(b_covariance, recomputed_covariance, rtol=1e-10, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(excess_covariance) if excess_covariance.shape == (44, 44) and finite and symmetric else np.asarray([float("nan")])
        checks.append(check("bootstrap_covariance_integrity", finite and symmetric and reproduction and bool(np.all(eigenvalues > 0.0)), {"finite": finite, "symmetric": symmetric, "reproduced": reproduction, "minimum_excess_eigenvalue": float(np.min(eigenvalues))}, {"finite": True, "symmetric": True, "sample_reproduction": True, "minimum_eigenvalue_operator": "> 0"}, {"npz_sha256": _sha256(npz_path)}))
        production = metadata.get("bootstrap_count_requested") == 1000 and metadata.get("bootstrap_count_completed") == 1000 and metadata.get("refit_failure_count") == 0 and metadata.get("production_complete") is True
        checks.append(check("bootstrap_production_complete", production, {key: metadata.get(key) for key in ("bootstrap_count_requested", "bootstrap_count_completed", "refit_failure_count", "production_complete")}, {"bootstrap_count_requested": 1000, "bootstrap_count_completed": 1000, "refit_failure_count": 0, "production_complete": True}, {"metadata_sha256": _sha256(metadata_path)}))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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
        checks.extend(slurm_checks(registry, query_sacct=True))
    if phase in ("bootstrap", "all"):
        checks.extend(bootstrap_checks(args.bootstrap_npz, args.bootstrap_metadata))
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
