import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("january", Path(__file__).parents[1] / "tools/january_v6_inputs.py")
january = importlib.util.module_from_spec(spec)
spec.loader.exec_module(january)


class FrozenInputsTest(unittest.TestCase):
    def test_source_binding_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.root"
            path.write_bytes(b"original")
            binding = january.file_stat(path)
            january.check_stat(path, binding)
            path.write_bytes(b"changed content")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                january.check_stat(path, binding)

    def test_inventory_rejects_extra_missing_and_wrong_month(self):
        expected = ["0101/Esg20220101_00.root", "0131/Esg20220131_00.root"]
        january.check_inventory(expected, expected, 2)
        for actual in [expected[:1], expected + ["0101/Esg20220101_01.root"], ["0201/Esg20220201_00.root", expected[1]]]:
            with self.assertRaises(ValueError):
                january.check_inventory(actual, expected, 2)


if __name__ == "__main__":
    unittest.main()
