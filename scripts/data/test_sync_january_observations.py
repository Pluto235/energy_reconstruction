from pathlib import Path
import hashlib
import tempfile
import unittest

from sync_january_observations import verify


class TransferAuditTests(unittest.TestCase):
    def test_accepts_expected_file_and_rejects_corruption_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "Esg20220101_00.root"
            (root / name).write_bytes(b"rootdata")
            report = {"day": "20220101", "file_count": 1, "bytes": 8,
                      "files": [{"name": name, "output": {"bytes": 8,
                                "sha256": hashlib.sha256(b"rootdata").hexdigest()}}]}
            verify(report, root)
            (root / name).write_bytes(b"rootxxxx")
            with self.assertRaises(RuntimeError):
                verify(report, root)
            (root / name).write_bytes(b"wrongdat")
            with self.assertRaises(RuntimeError):
                verify(report, root)
            report["files"][0]["name"] = "../escape.root"
            with self.assertRaises(ValueError):
                verify(report, root)


if __name__ == "__main__":
    unittest.main()
