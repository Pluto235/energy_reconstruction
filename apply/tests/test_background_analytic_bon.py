from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "stages/04_background.py"
SPEC = importlib.util.spec_from_file_location("stage04_background", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stage04 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage04
SPEC.loader.exec_module(stage04)

SIGNAL_MODULE_PATH = Path(__file__).resolve().parents[1] / "stages/05_signal.py"
SIGNAL_SPEC = importlib.util.spec_from_file_location("stage05_signal", SIGNAL_MODULE_PATH)
assert SIGNAL_SPEC is not None and SIGNAL_SPEC.loader is not None
stage05 = importlib.util.module_from_spec(SIGNAL_SPEC)
sys.modules[SIGNAL_SPEC.name] = stage05
SIGNAL_SPEC.loader.exec_module(stage05)


def gauss_disk_integral(coefficients: np.ndarray, radius: float, step: float, scale: float) -> float:
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(96)
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(128)
    rho = 0.5 * radius * (radial_nodes + 1.0)
    theta = math.pi * (angular_nodes + 1.0)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    c = np.pad(np.asarray(coefficients, dtype=np.float64), (0, 6 - len(coefficients)))
    values = c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y
    weights = 0.5 * radius * radial_weights[:, None] * math.pi * angular_weights[None, :]
    return float(scale * np.sum(values * rr * weights) / (step * step))


class AnalyticBackgroundIntegralTests(unittest.TestCase):
    def test_constant_surface_uses_true_disk_area(self) -> None:
        value = stage04.integrate_centered_disk_polynomial(
            np.asarray([7.5]), radius_deg=0.8, grid_step_deg=0.1, scale=1.2
        )
        self.assertAlmostEqual(value, 1.2 * 7.5 * math.pi * 0.8**2 / 0.1**2, places=10)

    def test_linear_and_xy_terms_cancel_by_circle_symmetry(self) -> None:
        base = stage04.integrate_centered_disk_polynomial(
            np.asarray([10.0, 0.0, 0.0, 2.0, 0.0, 3.0]), 1.1, 0.2, 0.75
        )
        asymmetric = stage04.integrate_centered_disk_polynomial(
            np.asarray([10.0, 14.0, -9.0, 2.0, 17.0, 3.0]), 1.1, 0.2, 0.75
        )
        self.assertAlmostEqual(base, asymmetric, places=10)

    def test_quadratic_formula_matches_independent_gauss_integral(self) -> None:
        coefficients = np.asarray([717.5, -1.1, 18.4, -1.2, -0.08, -2.8])
        actual = stage04.integrate_centered_disk_polynomial(coefficients, 0.85318446, 0.1, 1.0014)
        expected = gauss_disk_integral(coefficients, 0.85318446, 0.1, 1.0014)
        self.assertTrue(np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-8))

    def test_minimum_detects_interior_and_boundary_values(self) -> None:
        interior = stage04.minimum_quadratic_on_centered_disk(
            np.asarray([5.0, -2.0, 0.0, 1.0, 0.0, 1.0]), 2.0
        )
        boundary = stage04.minimum_quadratic_on_centered_disk(
            np.asarray([5.0, -2.0, 0.0, -1.0, 0.0, 0.0]), 2.0
        )
        self.assertAlmostEqual(interior, 4.0, places=10)
        self.assertAlmostEqual(boundary, -3.0, places=9)

    def test_analytic_selection_rejects_nonpositive_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-positive"):
            stage04.select_on_aperture_background(
                pixel_center_b_on=90.0,
                analytic_b_on=95.0,
                analytic_surface_min=-0.1,
                method="analytic-quadratic",
                cell_id=1,
            )

    def test_pixel_center_is_backward_compatible_default(self) -> None:
        selected = stage04.select_on_aperture_background(
            pixel_center_b_on=90.0,
            analytic_b_on=95.0,
            analytic_surface_min=-0.1,
            method="pixel-center",
            cell_id=1,
        )
        self.assertEqual(selected, 90.0)
        with mock.patch("sys.argv", ["04_background.py"]):
            self.assertEqual(stage04.parse_args().on_aperture_integration, "pixel-center")

    def test_load_cells_can_filter_to_included_selector_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.csv"
            path.write_text(
                "cell_id,nhit_bin,predE_bin,include\n"
                "1,\"[100,200)\",\"[2,2.5)\",1\n"
                "2,\"[100,200)\",\"[2.5,3)\",0\n",
                encoding="utf-8",
            )
            cells = stage04.load_cells(path, included_only=True)
        self.assertEqual([cell.cell_id for cell in cells], [1])

    def test_stage_e_uses_the_same_included_selector_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.csv"
            path.write_text(
                "cell_id,nhit_bin,predE_bin,include\n"
                "1,\"[100,200)\",\"[2,2.5)\",1\n"
                "2,\"[100,200)\",\"[2.5,3)\",0\n",
                encoding="utf-8",
            )
            cells = stage05.load_cells(path, included_only=True)
        self.assertEqual([cell.cell_id for cell in cells], [1])


if __name__ == "__main__":
    unittest.main()
