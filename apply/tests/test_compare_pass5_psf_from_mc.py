import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "compare_pass5_psf_from_mc.py"
SPEC = importlib.util.spec_from_file_location("compare_pass5_psf_from_mc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Pass5PsfComparisonTest(unittest.TestCase):
    def test_pass5_bin_boundaries_and_cuts(self):
        nv = np.asarray([29, 30, 59, 60, 99, 100, 799, 800, 1999, 2000])
        pinc = np.zeros_like(nv, dtype=np.float64)
        fitstat = np.zeros_like(nv)
        theta = np.zeros_like(nv, dtype=np.float64)
        dangle = np.zeros_like(nv, dtype=np.float64)
        rmds = np.zeros_like(nv, dtype=np.float64)
        dcedge = np.full_like(nv, 100.0, dtype=np.float64)
        result = MODULE.assign_pass5_bins(
            nv,
            pinc,
            fitstat,
            theta,
            dangle,
            rmds,
            dcedge,
            fitstat_equals=0,
            theta_max_deg=50.0,
            pincness_calibration=1.0,
            rmds_max=20.0,
            dcedge_min=None,
        )
        self.assertEqual(result.tolist(), [-1, 0, 0, 1, 1, 2, 5, 6, 6, -1])

        pinc[1] = MODULE.PASS5_PINC_MAX[0]
        result = MODULE.assign_pass5_bins(
            nv,
            pinc,
            fitstat,
            theta,
            dangle,
            rmds,
            dcedge,
            fitstat_equals=0,
            theta_max_deg=50.0,
            pincness_calibration=1.0,
            rmds_max=20.0,
            dcedge_min=None,
        )
        self.assertEqual(int(result[1]), -1)

    def test_double_rayleigh_fit_recovers_exact_profile(self):
        edges = np.linspace(0.0, 6.0, 601)
        truth = (0.72, 0.21, 0.63)
        shell = MODULE.model_shell_mass(edges, *truth) * 1.0e7
        fit = MODULE.fit_double_rayleigh(edges, shell)
        self.assertTrue(fit.success)
        self.assertAlmostEqual(fit.a_core, truth[0], delta=2.0e-3)
        self.assertAlmostEqual(fit.sigma_core_deg, truth[1], delta=2.0e-3)
        self.assertAlmostEqual(fit.sigma_tail_deg, truth[2], delta=2.0e-3)

    def test_theta_fold_uses_target_conditional_mixture(self):
        theta_weight = np.asarray(
            [
                [8.0, 2.0, 0.0],
                [0.0, 5.0, 5.0],
            ]
        )
        theta_weight2 = theta_weight.copy()
        target = np.asarray([0.25, 0.75])
        profile, _, metadata = MODULE.fold_crab_profile(theta_weight, theta_weight2, target)
        normalized = profile / profile.sum()
        expected = 0.25 * np.asarray([0.8, 0.2, 0.0]) + 0.75 * np.asarray([0.0, 0.5, 0.5])
        np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=1.0e-12)
        self.assertTrue(math.isclose(float(metadata["missing_target_probability_mass"]), 0.0))

    def test_full_sample_quantile_accounts_for_right_overflow(self):
        edges = np.asarray([0.0, 1.0, 2.0])
        shell = np.asarray([60.0, 20.0])
        result = MODULE.histogram_quantile_with_right_overflow(edges, shell, 20.0, 0.68)
        self.assertAlmostEqual(result, 1.4)
        result = MODULE.histogram_quantile_with_right_overflow(edges, shell, 20.0, 0.9)
        self.assertTrue(math.isnan(result))


if __name__ == "__main__":
    unittest.main()
