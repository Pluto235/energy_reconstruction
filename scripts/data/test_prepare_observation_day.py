"""Tiny ROOT fixture checking the historical reduction and validation contract."""
from array import array
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import ROOT

from prepare_observation_day import KEEP, MACRO, inspect


class ObservationReductionTests(unittest.TestCase):
    def test_cut_branches_header_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = [Path(directory) / name for name in ("source.root", "output.root")]
            file = ROOT.TFile(str(source), "RECREATE")
            tree = ROOT.TTree("t_eventout", "events")
            buffers = {name: array("f", [0.0]) for name in KEEP | {"vflag", "vnpe", "vqsamp"}}
            for name, buffer in buffers.items():
                tree.Branch(name, buffer, f"{name}/F")
            for index, value in enumerate([0.3, 1.099, 1.1, 1.3, float("nan")]):
                buffers["pincness"][0] = value
                buffers["ievent"][0] = index
                tree.Fill()
            header = ROOT.TTree("t_eventout_h", "header")
            header.Branch("run", buffers["irun"], "run/F")
            header.Fill()
            header.Fill()
            file.Write()
            file.Close()
            self.assertEqual(len(KEEP), 30)
            self.assertTrue({"vx", "vy", "vq", "vt", "vidmc", "ievent"}.issubset(KEEP))
            ROOT.gROOT.ProcessLine(f".L {MACRO}")
            self.assertEqual(ROOT.filter_observed_pincness_loop_scoped(str(source), str(output), 1.1), 0)
            self.assertEqual(inspect(source)["selected_entries"], 2)
            self.assertEqual(inspect(output, 2, 2)["entries"], 2)
            with self.assertRaises(RuntimeError):
                inspect(output, 3, 2)
            with self.assertRaises(RuntimeError):
                inspect(Path(directory) / "missing.root")
            # A full filesystem must fail, even when Fill only writes buffered baskets at close.
            result = subprocess.run([sys.executable, "-c", f'''
import ROOT, resource, signal
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
ROOT.gROOT.ProcessLine({f'.L {MACRO}'!r})
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
rc = ROOT.filter_observed_pincness_loop_scoped({str(source)!r}, {str(Path(directory) / 'limited.root')!r}, 1.1)
raise SystemExit(0 if rc != 0 else 1)
'''], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
