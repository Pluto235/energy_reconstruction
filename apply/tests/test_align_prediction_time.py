import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

spec = importlib.util.spec_from_file_location('align', Path(__file__).parents[1] / 'tools/align_prediction_time.py')
align = importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(align)


class AlignmentTest(unittest.TestCase):
    def test_filtered_and_reordered_identity(self):
        original = {k: np.array([1, 2, 3], dtype=np.int32) for k in ('irun', 'ies', 'iseq', 'ievent')}
        prediction = {k: v[[2, 0]] for k, v in original.items()}
        self.assertTrue(hasattr(align, 'map_entries'), 'identity mapping tool is missing')
        np.testing.assert_array_equal(align.map_entries(original, prediction), [2, 0])
        for field in original:
            bad = dict(prediction, **{field: np.array([99, 1], dtype=np.int32)})
            with self.assertRaisesRegex(ValueError, 'Missing'):
                align.map_entries(original, bad)
        for source, pred in (({k: v[[0, 0, 2]] for k, v in original.items()}, prediction),
                             (original, {k: v[[0, 0]] for k, v in original.items()})):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                align.map_entries(source, pred)

    def test_root_fixture_status_precision_and_no_overwrite(self):
        self.assertTrue(hasattr(align, 'align_file'), 'ROOT alignment tool is missing')
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for d in ('obs/0101', 'pred/0101', 'time/0101', 'obs_receipts', 'time_receipts', 'empty_receipts'):
                (base / d).mkdir(parents=True)
            rel = '0101/Esg20220101_00.root'
            raw = {k: np.array([1, 2, 3], dtype=np.int32) for k in align.KEYS}
            times = {'mjd': np.array([59580.123456789, 59580.2, np.nan]),
                     'ra_mean_deg': np.array([1., 2., 3.]), 'dec_mean_deg': np.array([4., 5., 6.]),
                     'match_status': np.array([0, 0, 3], dtype=np.int32),
                     'tai_sec': np.array([1640995201, 1640995202, 1640995203], dtype=np.int64),
                     'unix_subsec': np.array([0.123456789012345, -0.0, 0.999999999999999]),
                     'extra_field': np.array([1., 2., 3.], dtype=np.float32)}
            with uproot.recreate(base / 'obs' / rel) as f:
                f.mktree('t_eventout', raw)
            with uproot.recreate(base / 'pred' / rel) as f:
                f.mktree('t_eventout', {k: v[[0, 2]] for k, v in raw.items()})
            time = base / 'time/0101/Esg20220101_00.time.root'
            with uproot.recreate(time) as f:
                f.mktree('t_recovered_time', times)
                f.mktree('t_alignment', dict(raw, entry_index=np.arange(3, dtype=np.int64)))
                f['recovery_request_id'] = 'fixture-request'
                f['recovery_code_commit'] = 'fixture-code'
            summary = time.with_suffix('.summary.json')
            exceptions = time.with_suffix('.exceptions.jsonl')
            summary.write_text(json.dumps({'entries': 3, 'status': [2, 0, 0, 1, 0, 0, 0]}))
            exceptions.write_text('{}\n')
            obs_receipt = dict(name=Path(rel).name, record_id='fixture-record', entries=3,
                               bytes=(base / 'obs' / rel).stat().st_size)
            (base / 'obs_receipts' / (Path(rel).name + '.json')).write_text(json.dumps(obs_receipt))
            time_receipt = dict(obs_receipt, request_id='fixture-request', git_commit='fixture-code',
                                summary=json.loads(summary.read_text()),
                                output_stat=[align.file_stat(p) for p in (time, summary, exceptions)])
            (base / 'time_receipts' / (Path(rel).name + '.json')).write_text(json.dumps(time_receipt))
            config = dict(original_root=str(base / 'obs'), prediction_root=str(base / 'pred'),
                          time_root=str(base / 'time'), output_root=str(base / 'out'),
                          observation_receipts=str(base / 'obs_receipts'),
                          time_receipts=[str(base / 'empty_receipts'), str(base / 'time_receipts')])
            receipt = align.align_file(rel, config)
            self.assertEqual(receipt['prediction_entries'], 2)
            self.assertEqual(receipt['prediction_nonzero_status'], 1)
            with uproot.open(base / 'out/0101/Esg20220101_00.time.root') as f:
                self.assertEqual(f['t_recovered_time'].classname, 'TTree')
                for k, values in times.items():
                    actual = f['t_recovered_time'][k].array(library='np')
                    self.assertEqual(actual.dtype, values.dtype)
                    self.assertEqual(actual.tobytes(), values[[0, 2]].tobytes())
            with self.assertRaises(FileExistsError):
                align.align_file(rel, config)
            self.assertEqual(align.align_file(rel, config, verify_only=True)['prediction_entries'], 2)
            with uproot.update(time) as f:
                f.mktree('t_recovered_time', {k: v[:2] for k, v in times.items()})
            with self.assertRaisesRegex(ValueError, 'changed|count'):
                align.align_file(rel, dict(config, output_root=str(base / 'bad')))


if __name__ == '__main__':
    unittest.main()
