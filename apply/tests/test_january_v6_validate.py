import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("validation", Path(__file__).parents[1] / "tools/january_v6_validate.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ResponseContractTest(unittest.TestCase):
    def test_diagnostic_json_must_match_raw_arrays_and_c_count(self):
        ids = np.arange(1, 92)
        data = {"cell_id": ids, "N_on": ids.copy(), "B_on_analytic": ids.astype(float), "on_aperture_surface_min": np.ones(91), "annulus_counts": ids.copy(), "fit_rank": np.full(91, 6), "fit_condition_number": np.ones(91)}
        rows = [{"cell_id": i, "N_on": i, "raw_analytic_B_on": float(i), "surface_min": 1., "annulus_counts": i} for i in range(1, 45)]
        diagnostic = {"input_rows": 123, "cells": rows}
        validation.check_diagnostic(diagnostic, data, list(range(1, 45)), 123)
        rows[0]["N_on"] += 1
        with self.assertRaisesRegex(ValueError, "N_on"):
            validation.check_diagnostic(diagnostic, data, list(range(1, 45)), 123)
        rows[0]["N_on"] -= 1
        with self.assertRaisesRegex(ValueError, "input rows"):
            validation.check_diagnostic(diagnostic, data, list(range(1, 45)), 124)

    def test_manifest_cannot_change_frozen_selected_cells(self):
        rows = [{"cell_id": str(i), "include": "1"} for i in range(1, 45)]
        selected = list(range(1, 45))
        validation.check_selected_cells(selected, rows)
        with self.assertRaisesRegex(ValueError, "44-cell"):
            validation.check_selected_cells(selected[:-1] + [62], rows)

    def test_reject_double_containment_and_changed_denominator(self):
        base = {"a_eff": np.array([1., 10.]), "denominator_sumw": np.array([100.]), "denominator_count": np.array([200]), "cell_id": np.array([1, 2]), "logE_true_edges": np.array([2., 3.]), "theta_true_edges_deg": np.array([0., 50.]), "s0_m2": np.array(2e5)}
        fixed = {k: v.copy() for k, v in base.items()}
        fixed["a_eff"] *= validation.TARGET
        validation.check_response(base, fixed)
        fixed["a_eff"] *= validation.TARGET
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validation.check_response(base, fixed)
        fixed["a_eff"] = base["a_eff"] * validation.TARGET
        fixed["denominator_sumw"] *= .5
        with self.assertRaisesRegex(ValueError, "denominator_sumw"):
            validation.check_response(base, fixed)


if __name__ == "__main__":
    unittest.main()
