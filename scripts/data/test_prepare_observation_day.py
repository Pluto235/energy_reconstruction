"""Tiny ROOT fixture checking the historical reduction and validation contract."""
from array import array
from pathlib import Path
import tempfile
import subprocess
import sys
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ROOT

from prepare_observation_day import KEEP, MACRO, inspect, main, prepare_one
from observation_state import DAYS, MONTH, StagingQueue, atomic_json


class ObservationReductionTests(unittest.TestCase):
    def test_cleanup_rejects_nonstaging_output_root(self):
        with patch.object(sys, "argv", ["prepare", DAYS[0], "unused", "/eos/raw", "unused", "unused"]):
            with self.assertRaisesRegex(ValueError, "temporary staging"):
                main()

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
            self.assertTrue(hasattr(ROOT, "observation_filter_stats"), "One-scan metrics are missing")
            stats = ROOT.observation_filter_stats()
            self.assertEqual((stats.input_entries, stats.selected_entries, stats.header_entries), (5, 2, 2))
            self.assertEqual(inspect(source)["selected_entries"], 2)
            self.assertEqual(inspect(output, 2, 2)["entries"], 2)
            for cache in (0, 32, 128):
                candidate = Path(directory) / f"cache{cache}.root"
                self.assertEqual(ROOT.filter_observed_pincness_loop_scoped(str(source), str(candidate), 1.1, cache), 0)
                self.assertEqual(inspect(candidate, 2, 2)["entries"], 2)
                left, right = ROOT.TFile.Open(str(output)), ROOT.TFile.Open(str(candidate))
                self.assertTrue(ROOT.same_observation_tree(left.Get("t_eventout"), right.Get("t_eventout")))
                events = right.Get("t_eventout")
                events.GetEntry(0); self.assertEqual(events.ievent, 0)
                events.GetEntry(1); self.assertEqual(events.ievent, 1)
                left.Close(); right.Close()
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
            stage, work = Path(directory) / 'stage', Path(directory) / 'work'
            stage.mkdir(); (work / 'audit').mkdir(parents=True)
            (work / 'audit_files').mkdir(); (work / 'acks').mkdir()
            named_source = Path(directory) / f'Esg{DAYS[4]}_00.root'
            os.link(source, named_source)
            args = SimpleNamespace(output_root=stage, audit_dir=work/'audit',
                                   archive_root=Path(directory)/'no_archive', day=DAYS[4])
            config=dict(capacity_bytes=2**20,initial_reservation_bytes=2**20,cache_mb=32,full_read=True)
            atomic_json(work/'queue_config.json',config)
            queue=StagingQueue(stage,work)
            record=prepare_one(str(named_source),args,config,queue)
            self.assertNotIn('sha256',record['output'])
            self.assertFalse((work/'acks'/(record['name']+'.ack')).exists())
            self.assertEqual(queue.usage(),record['output']['bytes'])
            retry_source = Path(directory) / f'Esg{DAYS[4]}_01.root'
            os.link(source, retry_source)
            config.update(capacity_bytes=65536,initial_reservation_bytes=4096)
            atomic_json(work/'queue_config.json',config)
            with patch('prepare_observation_day.time.sleep', return_value=None):
                retried = prepare_one(str(retry_source),args,config,queue)
            self.assertEqual(retried['output']['entries'],2)
            self.assertFalse(list(stage.glob(MONTH[4:]+'??/*.partial')))
            self.assertLessEqual(queue.usage(),config['capacity_bytes'])


if __name__ == "__main__":
    unittest.main()
