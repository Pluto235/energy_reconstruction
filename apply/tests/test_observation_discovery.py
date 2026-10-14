import tempfile
import unittest
from pathlib import Path

from apply.apply_observation_energy import discover_root_files, filter_root_files_by_day_prefix


class ObservationDiscoveryTest(unittest.TestCase):
    def test_month_scan_ignores_time_sidecars_and_other_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = ["0101/first.root", "0102/second.root", "0201/third.root"]
            for name in valid + ["recovered_time/0101/first.time.root",
                                 "notes/example.root", "loose.root",
                                 "0101/nested/example.root"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            files, base = discover_root_files(str(root), None)
            self.assertEqual(files, [str(root / name) for name in valid])
            january, days = filter_root_files_by_day_prefix(files, input_root=base, day_prefix="01")
            self.assertEqual(january, [str(root / name) for name in valid[:2]])
            self.assertEqual(days, ["0101", "0102"])
            self.assertEqual(discover_root_files(str(root), 1)[0], files[:1])

    def test_day_scan_is_direct(self):
        with tempfile.TemporaryDirectory() as td:
            day = Path(td) / "0101"
            (day / "nested").mkdir(parents=True)
            source = day / "events.root"
            source.touch()
            (day / "nested/sidecar.root").touch()
            self.assertEqual(discover_root_files(str(day), None), ([str(source)], str(day)))

    def test_single_file_input_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "0101/events.root"
            source.parent.mkdir()
            source.touch()
            self.assertEqual(discover_root_files(str(source), None), ([str(source)], str(source)))


if __name__ == "__main__":
    unittest.main()
