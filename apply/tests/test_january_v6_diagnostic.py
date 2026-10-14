import importlib.util
from pathlib import Path
import unittest
import math

spec = importlib.util.spec_from_file_location("diagnostic", Path(__file__).parents[1] / "tools/january_v6_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class BackgroundFailureTest(unittest.TestCase):
    def test_scientific_failure_stays_invalid(self):
        failures = []
        def invalid(**kwargs):
            raise ValueError("Cell 62 has a non-positive fitted background inside the analytic on aperture: minimum=-1")
        result = diagnostic.record_aperture(invalid, failures, cell_id=62, analytic_required=True)
        self.assertTrue(math.isnan(result))
        self.assertEqual(failures[0]["cell_id"], 62)
        self.assertEqual(failures[0]["status"], "INVALID_BACKGROUND")

    def test_valid_value_and_unexpected_error_preserved(self):
        self.assertEqual(diagnostic.record_aperture(lambda **k: 3., [], cell_id=1), 3.)
        def broken(**kwargs):
            raise ValueError("unexpected coding error")
        with self.assertRaisesRegex(ValueError, "unexpected coding error"):
            diagnostic.record_aperture(broken, [], cell_id=1)


if __name__ == "__main__":
    unittest.main()
