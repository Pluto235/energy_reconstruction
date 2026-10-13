import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import torch
import uproot

from apply.apply_observation_energy import process_one_file


class ObservationRootWritingTest(unittest.TestCase):
    def check_roundtrip(self, all_rejected):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "0101" / "events.root"
            source.parent.mkdir()
            hits = ak.Array([list(range(20)), list(range(20)), list(range(20))])
            data = {
                "vidmc": hits, "vx": hits * 1., "vy": hits * 1.,
                "vt": hits * 1., "vq": hits * 1. + 1.,
                "theta": np.full(3, .2), "xc": np.zeros(3), "yc": np.zeros(3),
                "pincness": np.full(3, 2.) if all_rejected else np.array([.1, 2., .3]),
                "fitstat": np.zeros(3, dtype=np.int32), "dcedge": np.full(3, 30.),
                "ievent": np.array([10, 11, 12], dtype=np.int32),
            }
            header = {"filename": np.array(["original.root"], dtype=object),
                      "texts": np.array(["header must survive"], dtype=object)}
            with uproot.recreate(source) as f:
                f.mktree("t_eventout", data)
                f.mktree("t_eventout_h", header)
            with patch("apply.apply_observation_energy.infer_prepared",
                       side_effect=lambda prepared, **kw: np.full(len(prepared["survivor_event_indices"]), 3., dtype=np.float32)):
                result = process_one_file(
                    str(source), model=None, device=torch.device("cpu"),
                    tree_name="t_eventout", header_tree_name="t_eventout_h",
                    input_root=str(root), output_root=str(root / "result"),
                    max_points=500, sample_mode="weighted_q", norm_mode="per_event",
                    seed=42, batch_size=32, cut_pinc_max=1.1, cut_fitstat_equals=0,
                    cut_theta_max_deg=50., cut_dcedge_min=20., core_scale=(130., 110.),
                    reader_workers=0, max_events_per_file=None, step_size="1 kB",
                )
            expected = [] if all_rejected else [10, 12]
            self.assertEqual(result["written_events"], len(expected))
            with uproot.open(result["out_path"]) as f:
                self.assertEqual(f["t_eventout"].classname, "TTree")
                self.assertEqual(f["t_eventout_h"].classname, "TTree")
                self.assertEqual(f["t_eventout_h"]["texts"].array(library="np").tolist(),
                                 ["header must survive"])
                t = f["t_eventout"]
                self.assertEqual(t["ievent"].array(library="np").tolist(), expected)
                self.assertFalse({"vx", "vy", "vt", "vq"} & set(t.keys()))
                np.testing.assert_array_equal(t["ml_logE_pred"].array(library="np"),
                                              np.full(len(expected), 3.))
                np.testing.assert_array_equal(t["ml_energy_pred"].array(library="np"),
                                              np.full(len(expected), 1000.))

    def test_object_header_and_jagged_events(self):
        self.check_roundtrip(False)

    def test_empty_selection_keeps_typed_trees(self):
        self.check_roundtrip(True)


if __name__ == "__main__":
    unittest.main()
