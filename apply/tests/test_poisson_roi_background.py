from __future__ import annotations

import unittest

import numpy as np

from apply.stages.poisson_roi_background import (
    PoissonSurfaceFitError,
    fit_profiled_poisson_surface,
    quadratic_rectangle_basis_integrals,
)


class ExactPixelIntegralTests(unittest.TestCase):
    def test_rectangle_basis_matches_closed_form(self) -> None:
        x0, x1, y0, y1 = -1.37, -1.17, 2.41, 2.51
        expected = np.array(
            [
                (x1 - x0) * (y1 - y0),
                0.5 * (x1**2 - x0**2) * (y1 - y0),
                (x1 - x0) * 0.5 * (y1**2 - y0**2),
                (x1**3 - x0**3) * (y1 - y0) / 3.0,
                0.25 * (x1**2 - x0**2) * (y1**2 - y0**2),
                (x1 - x0) * (y1**3 - y0**3) / 3.0,
            ]
        )
        actual = quadratic_rectangle_basis_integrals(x0, x1, y0, y1)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)

    def test_four_children_sum_to_parent_for_every_basis_term(self) -> None:
        parent = quadratic_rectangle_basis_integrals(-0.2, 0.0, 0.3, 0.5)
        children = sum(
            (
                quadratic_rectangle_basis_integrals(x0, x0 + 0.1, y0, y0 + 0.1)
                for x0 in (-0.2, -0.1)
                for y0 in (0.3, 0.4)
            ),
            np.zeros(6),
        )
        np.testing.assert_allclose(children, parent, rtol=0.0, atol=1.0e-14)


def synthetic_fit_inputs(
    truth: np.ndarray,
    *,
    seed: int = 64748,
    total_per_cell: tuple[int, ...] = (500_000, 350_000),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    edges = np.linspace(-5.5, 5.5, 23)
    rows = []
    centers = []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        for x0, x1 in zip(edges[:-1], edges[1:]):
            center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
            if np.hypot(*center) < 5.5:
                rows.append(quadratic_rectangle_basis_integrals(x0, x1, y0, y1))
                centers.append(center)
    basis = np.asarray(rows)
    shape = basis @ truth
    if np.any(shape <= 0.0):
        raise ValueError("Synthetic truth must be positive on every pixel")
    probabilities = shape / shape.sum()
    rng = np.random.default_rng(seed)
    counts = {idx + 1: rng.multinomial(total, probabilities).astype(np.float64) for idx, total in enumerate(total_per_cell)}
    bases = {cell_id: basis.copy() for cell_id in counts}
    masks = {cell_id: np.ones(basis.shape[0], dtype=bool) for cell_id in counts}
    return counts, bases, masks


class ProfiledPoissonSurfaceTests(unittest.TestCase):
    def assert_recovers(self, truth: np.ndarray, order: int) -> None:
        counts, bases, masks = synthetic_fit_inputs(truth)
        result = fit_profiled_poisson_surface(
            counts,
            bases,
            masks,
            donor_cell_ids=(1, 2),
            order=order,
            positivity_radius_deg=6.0,
            shape_contributor_by_cell={1: True, 2: True},
        )
        self.assertGreater(result.positive_minimum, 0.0)
        np.testing.assert_allclose(result.shape_coefficients, truth, rtol=0.08, atol=0.01)
        self.assertEqual(result.annulus_normalizations, {1: 500_000.0, 2: 350_000.0})
        self.assertTrue(result.optimizer_status["success"])

    def test_recovers_constant(self) -> None:
        self.assert_recovers(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0)

    def test_recovers_plane(self) -> None:
        self.assert_recovers(np.asarray([1.0, 0.035, -0.025, 0.0, 0.0, 0.0]), 1)

    def test_recovers_quadratic(self) -> None:
        self.assert_recovers(np.asarray([1.0, 0.025, -0.018, 0.006, -0.003, 0.004]), 2)

    def test_non_contributor_keeps_normalization_but_not_shape(self) -> None:
        truth = np.asarray([1.0, 0.03, -0.02, 0.004, 0.0, 0.003])
        counts, bases, masks = synthetic_fit_inputs(truth, total_per_cell=(400_000, 42))
        counts[2] = np.roll(counts[2], 20)
        result = fit_profiled_poisson_surface(
            counts,
            bases,
            masks,
            (1, 2),
            2,
            6.0,
            {1: True, 2: False},
        )
        self.assertEqual(result.annulus_normalizations[2], 42.0)
        np.testing.assert_allclose(result.shape_coefficients, truth, rtol=0.08, atol=0.01)

    def test_negative_unconstrained_curvature_is_forced_positive(self) -> None:
        # This training region ends at rho=3, so its unconstrained continuation is negative by rho=6.
        truth = np.asarray([1.0, 0.0, 0.0, -0.055, 0.0, -0.045])
        counts, bases, masks = synthetic_fit_inputs(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), total_per_cell=(700_000,))
        centers = np.column_stack([bases[1][:, 1] / bases[1][:, 0], bases[1][:, 2] / bases[1][:, 0]])
        masks[1] = np.hypot(centers[:, 0], centers[:, 1]) < 3.0
        training_shape = bases[1][masks[1]] @ truth
        self.assertTrue(np.all(training_shape > 0.0))
        generated = np.zeros_like(counts[1])
        generated[masks[1]] = np.random.default_rng(64748).multinomial(
            700_000, training_shape / training_shape.sum()
        )
        counts[1] = generated
        result = fit_profiled_poisson_surface(counts, bases, masks, (1,), 2, 6.0, {1: True})
        self.assertGreater(result.positive_minimum, 0.0)
        self.assertGreaterEqual(result.optimizer_status["cutting_plane_constraints_added"], 1)

    def test_impossible_input_raises_typed_error(self) -> None:
        basis = np.asarray([quadratic_rectangle_basis_integrals(0.0, 0.1, 0.0, 0.1)])
        with self.assertRaises(PoissonSurfaceFitError):
            fit_profiled_poisson_surface(
                {1: np.asarray([3.0])},
                {1: basis},
                {1: np.asarray([True])},
                (1,),
                2,
                6.0,
                {1: False},
            )


if __name__ == "__main__":
    unittest.main()
