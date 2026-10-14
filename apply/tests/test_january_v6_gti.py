import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'january_gti', Path(__file__).parents[1] / 'tools/january_v6_gti.py')
gti = importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(gti)


class JanuaryGTITest(unittest.TestCase):
    def test_sort_then_split_preserves_gap60_and_singletons(self):
        self.assertTrue(hasattr(gti, 'intervals_from_mjd'), 'GTI builder is missing')
        # MJD zero makes the exact 60-second boundary representable for this fixture.
        values = np.array([200., 10., 0., 70., 200.]) / 86400.
        intervals, audit = gti.intervals_from_mjd(values)
        np.testing.assert_allclose(np.array(intervals) * 86400., [[0., 70.], [200., 200.]])
        self.assertEqual(audit['sorted_gap_count'], 1)
        self.assertEqual(audit['duplicate_time_count'], 1)
        self.assertEqual(audit['original_order_negative_step_count'], 2)
        self.assertEqual(audit.get('interval_count'), 2)
        self.assertAlmostEqual(audit['duration_seconds'], 70.)

    def test_merge_removes_overlap_without_filling_positive_boundary_gaps(self):
        self.assertTrue(hasattr(gti, 'merge_intervals'), 'GTI merge is missing')
        self.assertEqual(gti.merge_intervals([(8., 10.), (1., 4.), (3., 8.), (20., 20.),
                                             (10.5, 12.), (10.5, 12.)]),
                         [(1., 10.), (10.5, 12.)])
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            gti.merge_intervals([(2., 1.)])

    def times(self):
        unix = np.array([1640966421, 0, 1640966431], dtype=np.int64)
        sub = np.array([.123456789, 0., .987654321])
        return {'unix_sec': unix, 'unix_subsec': sub, 'tai_sec': unix + [37, 0, 37],
                'tai_subsec': sub.copy(), 'mjd': 40587. + (unix + sub) / 86400.,
                'match_status': np.array([0, 1, 0], dtype=np.int32)}

    def test_times_retain_nonzero_status_but_reject_invalid_status0(self):
        self.assertTrue(hasattr(gti, 'validate_times'), 'Time validation is missing')
        values = self.times()
        values['mjd'][1] = 0.
        result = gti.validate_times(values)
        self.assertEqual(result['status'], [2, 1, 0, 0, 0, 0, 0])
        self.assertEqual(result['valid_entries'], 2)
        for field, bad in [('mjd', np.nan), ('unix_subsec', 1.), ('tai_sec', 1),
                           ('tai_subsec', .9), ('match_status', 7)]:
            changed = {k: v.copy() for k, v in values.items()}
            changed[field][0] = bad
            with self.assertRaisesRegex(ValueError, 'Invalid|inconsistent'):
                gti.validate_times(changed)
        changed = {k: v.copy() for k, v in values.items()}
        changed['mjd'][0] += 1.e-9  # 86 microseconds exceeds rounding tolerance.
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            gti.validate_times(changed)

    def test_acceptance_binding_is_not_invented_from_current_file(self):
        self.assertTrue(hasattr(gti, 'load_tasks'), 'Source binding is missing')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = '0101/Esg20220101_00.root'
            time_path = root / 'raw_time' / Path(rel).with_suffix('.time.root')
            accepted = {'ok': True, 'relative_path': rel, 'original_entries': 3,
                        'prediction_entries': 2, 'original_status': [2, 1, 0, 0, 0, 0, 0],
                        'sources': {str(time_path): {'stat': [11, 22, 33], 'uuid': 'accepted'}}}
            verify_path = root / 'verify_66080.json'
            verify_path.write_text(json.dumps({'ok': True, 'config': {'time_root': str(root / 'raw_time')},
                                              'results': [accepted]}))
            manifest = {'result': 'PASS', 'time_label': '202201', 'original_time_root': str(root / 'raw_time'),
                        'file_count': 1, 'original_entries': 3, 'prediction_entries': 2,
                        'original_status': accepted['original_status'],
                        'contract': {'gti_gap_seconds': 60},
                        'files': [{k: accepted[k] for k in ('relative_path', 'original_entries', 'prediction_entries')}],
                        'hashes': {str(verify_path): gti.sha256(verify_path)}}
            manifest_path = root / 'analysis_manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            _, tasks = gti.load_tasks(manifest_path)
            self.assertEqual(tasks[0]['time_path'], str(time_path))
            self.assertEqual(tasks[0]['accepted']['sources'][str(time_path)]['stat'], [11, 22, 33])
            manifest['files'][0]['relative_path'] = '../wrong.root'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'inventory|January'):
                gti.load_tasks(manifest_path)
            manifest['files'][0]['relative_path'] = rel
            manifest_path.write_text(json.dumps(manifest))
            verify_path.write_text(verify_path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'hash'):
                gti.load_tasks(manifest_path)

    def test_source_change_and_existing_output_fail(self):
        self.assertTrue(hasattr(gti, 'check_binding'), 'Source checking is missing')
        self.assertTrue(hasattr(gti, 'run'), 'GTI entrypoint is missing')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.json'
            source.write_text('{}')
            binding = {'stat': gti.file_stat(source), 'sha256': gti.sha256(source)}
            gti.check_binding(source, binding)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                gti.check_binding(source, binding)
            with self.assertRaises(FileExistsError):
                gti.run(root / 'unused.json', root, workers=1)
            with self.assertRaisesRegex(ValueError, 'workers'):
                gti.run(root / 'unused.json', root / 'output', workers=17)

    def test_smoke_file_cap_must_be_explicit_and_positive(self):
        self.assertTrue('max_files' in inspect.signature(gti.run).parameters,
                        'An explicit smoke scope is missing')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, 'max-files'):
                gti.run(root / 'unused.json', root / 'output', workers=1, max_files=0)


if __name__ == '__main__':
    unittest.main()
