from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from apply.report.build_v6_poisson_grid_convergence import (
    BRANCH_IDS,
    DONOR_UNIVERSE_CELL_IDS,
    GridConvergenceError,
    evaluate_beta_gate,
    evaluate_cell_gate,
    evaluate_grid_convergence,
    evaluate_pull_gate,
    load_registry,
    strict_branch_records,
)
from apply.report.validate_v6_scheme_r_double_rayleigh_poisson_pooled import (
    bootstrap_checks,
    preflight_checks,
    report_checks,
    slurm_checks,
)


PROVENANCE = {
    "pooling_manifest_sha256": "1" * 64,
    "selector_sha256": "2" * 64,
    "psf_sha256": "3" * 64,
    "stage_c_sha256": "4" * 64,
    "response_sha256": "5" * 64,
    "analytic_bon_baseline_sha": "6" * 40,
    "code_sha": "7" * 40,
}


def synthetic_records() -> list[dict[str, object]]:
    ids = np.asarray(DONOR_UNIVERSE_CELL_IDS[:44], dtype=np.int64)
    donor_ids = np.asarray(DONOR_UNIVERSE_CELL_IDS, dtype=np.int64)
    records = []
    covariance = np.diag([0.04, 0.04, 0.0004])
    for index, branch_id in enumerate(BRANCH_IDS):
        offset = (index - 4) * 0.002
        records.append({
            "branch_id": branch_id,
            "cell_ids": ids.copy(),
            "b_on": np.full(44, 100.0 + offset),
            "n_on": np.full(44, 110, dtype=np.int64),
            "sigma_excess": np.full(44, 2.0),
            "pull": np.full(44, 1.0 + offset),
            "fit_parameters": np.asarray([-11.7 + offset * 0.01, 2.7, 0.12 + offset * 0.001]),
            "fit_covariance": covariance.copy(),
            "stage_d_cell_ids": donor_ids.copy(),
            "stage_d_valid": np.ones(84, dtype=bool),
            "stage_d_positive": np.ones(84, dtype=bool),
            "stage_f_valid": True,
            "stage_e_contract_valid": True,
            "provenance": dict(PROVENANCE),
            "provenance_verification": {"passed": True},
        })
    return records


class RegisteredGateTests(unittest.TestCase):
    def test_registered_examples(self) -> None:
        self.assertTrue(evaluate_cell_gate(all_sigma_units=[0.2] * 44).passed)
        self.assertFalse(evaluate_cell_gate(all_sigma_units=[0.2] * 43 + [0.51]).passed)
        self.assertFalse(evaluate_beta_gate(delta_beta=0.0031, sigma_beta=0.0121).passed)
        self.assertFalse(evaluate_pull_gate(delta_pull=0.51).passed)

    def test_all_structured_checks_have_exact_fields(self) -> None:
        payload = evaluate_grid_convergence(synthetic_records())
        self.assertTrue(payload["passed"])
        for item in payload["checks"]:
            self.assertEqual(set(item), {"name", "passed", "observed", "limit", "evidence"})

    def test_rejects_missing_duplicate_and_unexpected_branches(self) -> None:
        records = synthetic_records()
        for malformed in (records[:-1], records + [records[0]], [dict(records[0], branch_id="surprise")] + records[1:]):
            with self.assertRaises(GridConvergenceError):
                strict_branch_records(malformed)

    def test_rejects_provenance_mismatch(self) -> None:
        records = synthetic_records()
        records[-1]["provenance"] = dict(PROVENANCE, response_sha256="different")
        payload = evaluate_grid_convergence(records)
        self.assertFalse(payload["passed"])
        self.assertFalse(next(row for row in payload["checks"] if row["name"] == "shared_provenance_identical")["passed"])

    def test_every_registered_numerical_gate_can_fail(self) -> None:
        mutations = {
            "background_all_cells_half_sigma": lambda records: records[-1]["b_on"].__setitem__(0, 101.1),
            "background_cells_quarter_sigma_coverage": lambda records: records[-1].__setitem__("b_on", np.r_[np.full(5, 100.6), np.full(39, 100.2)]),
            "phase_rms_all_cells_half_sigma": lambda records: records[1]["b_on"].__setitem__(0, 102.5),
            "phase_rms_cells_quarter_sigma_coverage": lambda records: records[1]["b_on"].__setitem__(slice(0, 5), np.full(5, 101.4)),
            "fit_parameter_mahalanobis": lambda records: records[-1].__setitem__("fit_parameters", np.asarray([-11.0, 2.7, 0.12])),
            "beta_stability": lambda records: records[-1].__setitem__("fit_parameters", np.asarray([-11.7, 2.7, 0.126])),
            "pull_stability": lambda records: records[-1]["pull"].__setitem__(0, 1.51),
            "no_pull_five_sigma_migration": lambda records: records[-1]["pull"].__setitem__(0, 5.1),
            "all_stage_d_fits_valid": lambda records: records[-1]["stage_d_valid"].__setitem__(0, False),
            "all_stage_d_surfaces_positive": lambda records: records[-1]["stage_d_positive"].__setitem__(0, False),
            "all_stage_f_fits_valid": lambda records: records[-1].__setitem__("stage_f_valid", False),
            "stage_e_background_matches_stage_d": lambda records: records[-1].__setitem__("stage_e_contract_valid", False),
            "event_level_n_on_identical": lambda records: records[-1]["n_on"].__setitem__(0, 111),
            "artifact_provenance_verified": lambda records: records[-1].__setitem__("provenance_verification", {"passed": False}),
        }
        for gate_name, mutate in mutations.items():
            with self.subTest(gate=gate_name):
                records = synthetic_records()
                mutate(records)
                payload = evaluate_grid_convergence(records)
                gate = next(row for row in payload["checks"] if row["name"] == gate_name)
                self.assertFalse(gate["passed"])


class RegistryAndValidatorTests(unittest.TestCase):
    def _write_branch_files(self, root: Path, branch_id: str) -> dict[str, object]:
        ids = np.asarray(DONOR_UNIVERSE_CELL_IDS[:44], dtype=np.int64)
        donor_ids = np.asarray(DONOR_UNIVERSE_CELL_IDS, dtype=np.int64)
        d = root / f"{branch_id}_d.npz"
        e = root / f"{branch_id}_e.npz"
        f = root / f"{branch_id}_f.npz"
        d_meta = root / f"{branch_id}_d.json"
        meta = root / f"{branch_id}_f.json"
        selector = root / "selector.csv"
        psf = root / "psf.npz"
        stage_c = root / "stage_c.json"
        response = root / "response.npz"
        selector.write_text("cell_id,include\n", encoding="utf-8")
        if not psf.exists():
            np.savez(psf, value=np.asarray([1]))
            np.savez(response, value=np.asarray([2]))
            stage_c.write_text("{}\n", encoding="utf-8")
        manifest = root / "manifest.json"
        if not manifest.exists():
            unsigned = {
                "analytic_bon_baseline_sha": PROVENANCE["analytic_bon_baseline_sha"],
                "implementation_commit_sha": PROVENANCE["code_sha"],
                "target_cell_ids": ids.tolist(),
                "provenance": {
                    "selector": {"path": selector.name, "sha256": hashlib.sha256(selector.read_bytes()).hexdigest()},
                    "psf_npz": {"path": psf.name, "sha256": hashlib.sha256(psf.read_bytes()).hexdigest()},
                    "stage_c_metadata": {"path": stage_c.name, "sha256": hashlib.sha256(stage_c.read_bytes()).hexdigest()},
                },
            }
            manifest_sha = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
            manifest.write_text(json.dumps({**unsigned, "manifest_sha256": manifest_sha}), encoding="utf-8")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        provenance = {
            **PROVENANCE,
            "pooling_manifest_sha256": manifest_payload["manifest_sha256"],
            "selector_sha256": hashlib.sha256(selector.read_bytes()).hexdigest(),
            "psf_sha256": hashlib.sha256(psf.read_bytes()).hexdigest(),
            "stage_c_sha256": hashlib.sha256(stage_c.read_bytes()).hexdigest(),
            "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
        }
        step = {"005": 0.05, "010": 0.1, "020": 0.2}[branch_id[1:4]]
        phase = branch_id[5:]
        x_fraction = 0.5 if phase in {"xh_y0", "xh_yh"} else 0.0
        y_fraction = 0.5 if phase in {"x0_yh", "xh_yh"} else 0.0
        np.savez(d, cell_id=donor_ids, B_on=np.full(84, 100.0), surface_fit_success=np.ones(84, dtype=bool), on_aperture_surface_min=np.ones(84))
        np.savez(e, cell_id=ids, N_on=np.full(44, 110), B_on=np.full(44, 100.0), excess=np.full(44, 10.0))
        np.savez(f, cell_id=ids, excess_error_conservative=np.full(44, 2.0), logpar_conservative_pull=np.full(44, 1.0))
        d_meta.write_text(json.dumps({
            "inputs": {"pooling_manifest": manifest.name, "cell_selection_csv": selector.name, "psf_npz": psf.name, "stage_c_metadata_json": stage_c.name},
            "background_model": {"fit_statistic": "poisson", "pooling_manifest_sha256": provenance["pooling_manifest_sha256"], "analytic_bon_baseline_sha": provenance["analytic_bon_baseline_sha"]},
            "grid": {"grid_step_deg": step, "offset_x_fraction": x_fraction, "offset_y_fraction": y_fraction},
        }), encoding="utf-8")
        meta.write_text(json.dumps({"inputs": {"response_npz": response.name}, "fits": {"logpar_conservative": {"valid": True, "fit_parameters": {"log10_phi0": -11.7, "alpha": 2.7, "beta": 0.12}, "covariance": np.diag([0.04, 0.04, 0.0004]).tolist()}}}), encoding="utf-8")
        return {"branch_id": branch_id, "stage_d_npz": d.name, "stage_d_metadata": d_meta.name, "stage_e_npz": e.name, "stage_f_npz": f.name, "stage_f_metadata": meta.name, "provenance": provenance}

    def test_registry_loads_synthetic_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = [self._write_branch_files(root, branch_id) for branch_id in BRANCH_IDS]
            registry = root / "branches.json"
            registry.write_text(json.dumps({"branches": specs}), encoding="utf-8")
            records, _ = load_registry(registry)
            self.assertTrue(evaluate_grid_convergence(records)["passed"])

    def test_preflight_does_not_read_production_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            specs = [{"branch_id": branch_id, "stage_d_npz": "future-d.npz", "stage_d_metadata": "future-d.json", "stage_e_npz": "future-e.npz", "stage_f_npz": "future-f.npz", "stage_f_metadata": "future-f.json", "provenance": dict(PROVENANCE, analytic_bon_baseline_sha=head, code_sha=head)} for branch_id in BRANCH_IDS]
            registry = root / "branches.json"
            registry.write_text(json.dumps({"branches": specs}), encoding="utf-8")
            checks = preflight_checks(registry, Path.cwd(), head)
            self.assertTrue(all(row["passed"] for row in checks))

    def test_slurm_contract_requires_completed(self) -> None:
        completed = {"manifest": {"state": "COMPLETED"}, "bootstrap": {"state": "COMPLETED"}}
        completed.update({branch_id: {"state": "COMPLETED"} for branch_id in BRANCH_IDS})
        passed = slurm_checks({"slurm_jobs": completed})
        failed = slurm_checks({"slurm_jobs": {**completed, BRANCH_IDS[-1]: {"state": "RUNNING"}}})
        self.assertTrue(passed[0]["passed"])
        self.assertFalse(failed[0]["passed"])

    def test_bootstrap_contract_recomputes_covariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(64748)
            samples = rng.normal(size=(1000, 44))
            b_covariance = np.cov(samples, rowvar=False, ddof=1)
            excess_covariance = b_covariance + np.eye(44)
            npz = root / "bootstrap.npz"
            meta = root / "bootstrap.json"
            np.savez(
                npz,
                cell_id=np.arange(1, 45),
                B_on_nominal=np.zeros(44),
                B_on_bootstrap_mean=np.mean(samples, axis=0),
                B_on_bootstrap_samples=samples,
                B_on_covariance=b_covariance,
                excess_covariance=excess_covariance,
            )
            meta.write_text(json.dumps({"bootstrap_count_requested": 1000, "bootstrap_count_completed": 1000, "refit_failure_count": 0, "production_complete": True}), encoding="utf-8")
            self.assertTrue(all(row["passed"] for row in bootstrap_checks(npz, meta)))
            with np.load(npz) as handle:
                arrays = {name: handle[name] for name in handle.files}
            arrays["B_on_covariance"] = arrays["B_on_covariance"].copy()
            arrays["B_on_covariance"][0, 0] += 1.0
            np.savez(npz, **arrays)
            self.assertFalse(all(row["passed"] for row in bootstrap_checks(npz, meta)))

    def test_report_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            convergence = root / "convergence.json"
            convergence.write_text(json.dumps({"passed": True}), encoding="utf-8")
            report = root / "report.html"
            report.write_text("Poisson Pooling Grid Convergence Covariance Legacy", encoding="utf-8")
            self.assertTrue(all(row["passed"] for row in report_checks(report, convergence)))


if __name__ == "__main__":
    unittest.main()
