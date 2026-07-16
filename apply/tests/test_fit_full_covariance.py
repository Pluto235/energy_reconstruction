from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from apply.tools.bootstrap_v6_poisson_background import (
    bootstrap_replicate,
    prepare_context,
    rectangle_basis_grid,
    rng_for_replicate,
)


MODULE_PATH = Path(__file__).resolve().parents[1] / "stages/06_fit.py"
SPEC = importlib.util.spec_from_file_location("stage06_fit_full_covariance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stage06 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage06
SPEC.loader.exec_module(stage06)


class CellSubsetAlignmentTests(unittest.TestCase):
    def test_response_superset_is_aligned_to_selected_signal_order(self) -> None:
        response = {
            "cell_id": np.asarray([1, 2, 3, 4, 5]),
            "nhit_bin": np.asarray(["a", "b", "c", "d", "e"]),
            "a_eff": np.arange(10).reshape(5, 2),
            "energy_edges": np.asarray([0.0, 1.0, 2.0]),
        }
        signal = {
            "cell_id": np.asarray([4, 2]),
            "nhit_bin": np.asarray(["d", "b"]),
            "N_on": np.asarray([40.0, 20.0]),
        }
        subset = {
            "included_cell_ids": [4, 2],
            "excluded_cell_ids": [1, 3, 5],
        }
        aligned_response, aligned_signal, metadata = stage06.apply_cell_subset(
            response, signal, subset
        )
        np.testing.assert_array_equal(aligned_response["cell_id"], [4, 2])
        np.testing.assert_array_equal(aligned_response["a_eff"], [[6, 7], [2, 3]])
        np.testing.assert_array_equal(aligned_signal["cell_id"], [4, 2])
        np.testing.assert_array_equal(aligned_response["energy_edges"], [0.0, 1.0, 2.0])
        self.assertEqual(metadata["n_input_response_cells"], 5)


class GeneralizedChi2Tests(unittest.TestCase):
    def test_matches_linear_solve_reference(self) -> None:
        residual = np.asarray([1.25, -0.75, 2.0], dtype=np.float64)
        covariance = np.asarray(
            [[4.0, 0.8, 0.2], [0.8, 2.5, -0.3], [0.2, -0.3, 3.0]],
            dtype=np.float64,
        )
        actual = stage06.generalized_chi2(residual, covariance)
        expected = float(residual @ np.linalg.solve(covariance, residual))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_diagonal_covariance_reproduces_scalar_error_objective(self) -> None:
        residual = np.asarray([3.0, -4.0, 1.5, 8.0], dtype=np.float64)
        errors = np.asarray([2.0, 5.0, 0.75, 4.0], dtype=np.float64)
        actual = stage06.generalized_chi2(residual, np.diag(errors * errors))
        expected = float(np.sum((residual / errors) ** 2))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_non_positive_definite_covariance_has_clear_error(self) -> None:
        covariance = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            stage06.generalized_chi2(np.asarray([1.0, -1.0]), covariance)


class CovarianceArtifactTests(unittest.TestCase):
    def test_load_requires_exact_cell_order(self) -> None:
        cell_ids = np.arange(1, 45, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "covariance.npz"
            reversed_ids = cell_ids[::-1]
            np.savez_compressed(
                path,
                cell_id=reversed_ids,
                excess_covariance=np.eye(44, dtype=np.float64),
            )
            with self.assertRaisesRegex(ValueError, "order does not exactly match"):
                stage06.load_excess_covariance(path, cell_ids)

    def test_load_accepts_strict_positive_definite_contract(self) -> None:
        cell_ids = np.arange(101, 145, dtype=np.int64)
        covariance = np.diag(np.linspace(2.0, 5.0, 44))
        covariance[0, 1] = covariance[1, 0] = 0.2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "covariance.npz"
            np.savez_compressed(path, cell_id=cell_ids, excess_covariance=covariance)
            actual, metadata = stage06.load_excess_covariance(path, cell_ids)
        np.testing.assert_array_equal(actual, covariance)
        self.assertEqual(metadata["n_cells"], 44)
        self.assertGreater(metadata["minimum_eigenvalue"], 0.0)

    def test_load_rejects_wrong_cell_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "covariance.npz"
            np.savez_compressed(
                path,
                cell_id=np.arange(4, dtype=np.int64),
                excess_covariance=np.eye(4),
            )
            with self.assertRaisesRegex(ValueError, "exactly 44"):
                stage06.load_excess_covariance(path, np.arange(4, dtype=np.int64))


class CovarianceFitTests(unittest.TestCase):
    def test_covariance_diagnostics_do_not_change_preferred_fit(self) -> None:
        fits = {
            "pl_conservative": SimpleNamespace(valid=True, chi2=20.0),
            "logpar_conservative": SimpleNamespace(valid=True, chi2=15.0),
            "pl_background_covariance": SimpleNamespace(valid=True, chi2=1.0),
            "logpar_background_covariance": SimpleNamespace(valid=True, chi2=100.0),
        }
        preferred = stage06.choose_preferred_fit(fits)
        self.assertEqual(preferred["model"], "logpar")
        self.assertEqual(preferred["error_mode"], "conservative")
        self.assertEqual(preferred["delta_chi2_pl_minus_logpar"], 5.0)

    def test_invalid_covariance_diagnostics_fail_production_quality(self) -> None:
        fits = {
            "pl_conservative": SimpleNamespace(valid=True),
            "logpar_conservative": SimpleNamespace(valid=True),
            "pl_background_covariance": SimpleNamespace(valid=False),
            "logpar_background_covariance": SimpleNamespace(valid=False),
        }
        quality = stage06.fit_quality(fits, {"status": "passed"})
        self.assertFalse(quality["stage_f_current_promotable"])
        self.assertFalse(quality["background_covariance_fits_valid"])

    def test_fit_stores_marginal_pulls_and_whitened_residuals(self) -> None:
        n_cells = 6
        a_eff = np.linspace(0.8, 1.3, n_cells, dtype=np.float64)[:, None, None]
        containment = np.ones(n_cells, dtype=np.float64)
        exposure = np.asarray([2.5e5], dtype=np.float64)
        loge_edges = np.asarray([3.0, 3.3], dtype=np.float64)
        truth = stage06.model_counts(
            a_eff,
            containment,
            exposure,
            loge_edges,
            model_name="pl",
            params={"phi0": 2.0e-12, "gamma": 2.7},
            pivot_tev=3.0,
            quadrature_points=32,
        )
        observed = truth + np.asarray([2.0, -1.0, 1.5, -0.5, 0.75, -1.25])
        covariance = np.diag(np.linspace(4.0, 9.0, n_cells))
        covariance += 0.15 * np.ones((n_cells, n_cells), dtype=np.float64)
        errors = np.sqrt(np.diag(covariance))
        result = stage06.fit_model(
            model_name="pl",
            error_mode="background_covariance",
            observed=observed,
            errors=errors,
            full_covariance=covariance,
            a_eff_m2=a_eff,
            containment=containment,
            theta_exposure_sec=exposure,
            loge_edges=loge_edges,
            pivot_tev=3.0,
            quadrature_points=32,
            start_gamma=2.7,
            start_phi0=2.0e-12,
        )
        np.testing.assert_allclose(result.pull, result.residual / errors, rtol=0.0, atol=1.0e-12)
        expected_whitened = np.linalg.solve(np.linalg.cholesky(covariance), result.residual)
        np.testing.assert_allclose(result.whitened_residual, expected_whitened, rtol=0.0, atol=1.0e-12)
        self.assertAlmostEqual(result.chi2, float(expected_whitened @ expected_whitened), places=9)


class BootstrapDeterminismTests(unittest.TestCase):
    @staticmethod
    def _nominal_context_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
        donor_ids = np.arange(1, 85, dtype=np.int64)
        target_ids = donor_ids[:44]
        b_on = np.linspace(100.0, 183.0, donor_ids.size)
        counts_map = np.full((donor_ids.size, 2, 2), 250, dtype=np.int64)
        stage_d = {
            "cell_id": donor_ids,
            "counts_map": counts_map,
            "background_map": np.full(counts_map.shape, 250.0),
            "training_mask": np.ones(counts_map.shape, dtype=bool),
            "on_mask": np.ones(counts_map.shape, dtype=bool),
            "x_edges_deg": np.asarray([-1.0, 0.0, 1.0]),
            "y_edges_deg": np.asarray([-1.0, 0.0, 1.0]),
            "r_opt_deg": np.full(donor_ids.size, 0.5),
            "annulus_inner_deg": np.full(donor_ids.size, 0.6),
            "annulus_outer_deg": np.full(donor_ids.size, 0.9),
            "B_on": b_on,
        }
        event_n_on = np.arange(10_001, 10_045, dtype=np.int64)
        stage_e = {
            "cell_id": target_ids,
            "N_on": event_n_on,
            "B_on": b_on[: target_ids.size],
        }
        manifest: dict[str, object] = {
            "target_cell_ids": target_ids.tolist(),
            "donor_universe_cell_ids": donor_ids.tolist(),
            "continuous_annulus_counts": {str(cell_id): 1_000 for cell_id in donor_ids},
            "cells": {
                str(cell_id): {"donor_cell_ids": [int(cell_id)], "surface_order": 0}
                for cell_id in target_ids
            },
        }
        return stage_d, stage_e, manifest

    def test_context_uses_event_level_stage_e_n_on_not_pixelized_stage_d_counts(self) -> None:
        stage_d, stage_e, manifest = self._nominal_context_inputs()
        pixelized_n_on = np.sum(stage_d["counts_map"], axis=(1, 2))[:44]
        self.assertFalse(np.array_equal(pixelized_n_on, stage_e["N_on"]))

        context = prepare_context(stage_d, stage_e, manifest)

        np.testing.assert_array_equal(context["N_on"], stage_e["N_on"])
        self.assertFalse(np.array_equal(context["N_on"], pixelized_n_on))

    def test_context_rejects_non_integer_stage_e_n_on(self) -> None:
        stage_d, stage_e, manifest = self._nominal_context_inputs()
        stage_e["N_on"] = stage_e["N_on"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "N_on must use an integer dtype"):
            prepare_context(stage_d, stage_e, manifest)

    def test_context_rejects_stage_e_b_on_not_aligned_to_stage_d(self) -> None:
        stage_d, stage_e, manifest = self._nominal_context_inputs()
        stage_e["B_on"] = stage_e["B_on"].copy()
        stage_e["B_on"][3] += 0.5
        with self.assertRaisesRegex(ValueError, "B_on must exactly match Stage D"):
            prepare_context(stage_d, stage_e, manifest)

    def test_replicate_index_seed_splitting_is_stable(self) -> None:
        first = rng_for_replicate(64748, 17).poisson(20.0, size=20)
        repeated = rng_for_replicate(64748, 17).poisson(20.0, size=20)
        neighbor = rng_for_replicate(64748, 18).poisson(20.0, size=20)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, neighbor))

    def test_same_replicate_is_identical_with_frozen_constant_surface(self) -> None:
        edges = np.linspace(-2.0, 2.0, 9)
        basis = rectangle_basis_grid(edges, edges).reshape(-1, 6)
        context = {
            "target_ids": np.asarray([1], dtype=np.int64),
            "target_specs": {1: {"donors": (1,), "order": 0}},
            "expected_by_cell": {1: np.full(64, 500.0)},
            "pixel_basis_by_cell": {1: basis},
            "annulus_mask_by_cell": {1: np.ones(64, dtype=bool)},
            "shape_contributor_by_cell": {1: True},
            "r_opt_by_cell": {1: 0.8},
            "annulus_inner_by_cell": {1: 1.0},
            "annulus_outer_by_cell": {1: 2.0},
            "positivity_radius_deg": 6.0,
        }
        first = bootstrap_replicate(23, 64748, context)
        repeated = bootstrap_replicate(23, 64748, context)
        np.testing.assert_array_equal(first, repeated)


if __name__ == "__main__":
    unittest.main()
