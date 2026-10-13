"""Small checks for migration receipts and byte reservations; no ROOT or large I/O."""
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest

try:
    import observation_state as state
except ImportError:
    state = None


def reserve_in_child(args):
    root, work, name = args
    return state.StagingQueue(Path(root), Path(work)).try_reserve(name, 100)


class ObservationStateTests(unittest.TestCase):
    def test_february_snapshot_isolated_dates_paths_and_partial_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['observation_state.py', 'filter_observed_pincness_loop_scoped.C', 'sync_january_observations.py']:
                shutil.copy(Path(__file__).with_name(name), root / name)
            (root / 'run_config.json').write_text(json.dumps(
                dict(month='202202', run_date='20261014', expected_files=600)))
            result = subprocess.run([sys.executable, '-c', '''
from pathlib import Path
import observation_state as s
from sync_january_observations import validate_manifests, STATE, SOURCE
assert s.MONTH == '202202', 'snapshot must select February'
assert len(s.DAYS) == 28 and s.DAYS[-1] == '20220228'
assert str(s.STAGING).endswith('WCDA_obs_202202_trimmed_20261014')
assert str(STATE).endswith('2026-10-14/obs-february') and SOURCE == str(s.STAGING)
for name in ['Esg20220229_00.root', 'Esg20220101_00.root', '../Esg20220201_00.root']:
    try: s.file_path(Path('.'), name)
    except ValueError: pass
    else: raise AssertionError(name)
assert s.file_path(Path('.'), 'Esg20220228_23.root').parent.name == '0228'
stage=Path('stage'); work=Path('control'); (stage/'0228').mkdir(parents=True); work.mkdir()
(stage/'0228'/'Esg20220228_23.root.partial').write_bytes(b'x'*30)
s.atomic_json(work/'queue_config.json', {'capacity_bytes':100})
q=s.StagingQueue(stage,work)
assert q.usage()==30
assert not q.try_reserve('Esg20220228_22.root',80)
manifests={day:[f'Esg{day}_{h:02d}.root' for h in range(22 if n<12 else 21)] for n,day in enumerate(s.DAYS)}
assert len(validate_manifests(manifests))==600
manifests[s.DAYS[-1]][0]=manifests[s.DAYS[0]][0]
try: validate_manifests(manifests)
except ValueError: pass
else: raise AssertionError('duplicate or cross-date file accepted')
try: validate_manifests({'20220201':['Esg20220201_00.root']})
except ValueError: pass
else: raise AssertionError('incomplete month accepted')
'''], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setUp(self):
        self.assertIsNotNone(state, 'Shared receipt/byte-queue implementation is missing')

    def test_legacy_and_new_receipts_do_not_invent_hashes(self):
        legacy = {'name': f'Esg{state.MONTH}05_00.root', 'output': {'bytes': 10, 'entries': 2,
                  'header_entries': 1, 'sha256': 'a' * 64}}
        self.assertTrue(state.ack_matches(legacy, 'a' * 64))
        self.assertFalse(state.ack_matches(legacy, 'b' * 64))
        new = {'name': legacy['name'], 'receipt_id': 'b' * 32,
               'output': {'bytes': 10, 'entries': 2, 'header_entries': 1}}
        receipt = state.make_receipt(new)
        self.assertNotIn('sha256', json.dumps(receipt))
        self.assertTrue(state.ack_matches(new, json.dumps(receipt)))
        receipt['bytes'] += 1
        self.assertFalse(state.ack_matches(new, json.dumps(receipt)))
        self.assertFalse(state.ack_matches(new, 'a' * 64))
        new['output']['sha256'] = 'a' * 64
        self.assertFalse(state.ack_matches(new, 'a' * 64), 'A legacy ack must not confirm a new receipt ID')

    def test_reservations_include_partials_and_pending_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work = Path(directory) / 'staging', Path(directory) / 'state'
            day = state.MONTH[4:] + '05'
            (root / day).mkdir(parents=True)
            work.mkdir()
            state.atomic_json(work / 'queue_config.json', {'capacity_bytes': 200})
            queue = state.StagingQueue(root, work)
            names = [f'Esg{state.MONTH}05_{n:02d}.root' for n in range(3)]
            (root / day / (names[0] + '.partial')).write_bytes(b'x' * 30)
            self.assertTrue(queue.try_reserve(names[0], 60))
            self.assertTrue(queue.try_reserve(names[1], 100))
            self.assertFalse(queue.try_reserve(names[2], 50))
            self.assertEqual(queue.usage(), 160)
            (root / day / (names[0] + '.partial')).rename(root / day / names[0])
            queue.release(names[0])
            self.assertEqual(queue.usage(), 130)
            with self.assertRaises(ValueError):
                queue.try_reserve('../escape.root', 1)

    def test_concurrent_reservations_never_overcommit(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work = Path(directory) / 'staging', Path(directory) / 'state'
            root.mkdir(); work.mkdir()
            state.atomic_json(work / 'queue_config.json', {'capacity_bytes': 350})
            args = [(str(root), str(work), f'Esg{state.MONTH}05_{n:02d}.root') for n in range(8)]
            with multiprocessing.get_context('fork').Pool(8) as pool:
                results = pool.map(reserve_in_child, args)
            self.assertEqual(sum(results), 3)
            self.assertEqual(state.StagingQueue(root, work).usage(), 300)


if __name__ == '__main__':
    unittest.main()
