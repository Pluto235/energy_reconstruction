"""Small checks for migration receipts and byte reservations; no ROOT or large I/O."""
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

try:
    import observation_state as state
except ImportError:
    state = None


def reserve_in_child(args):
    root, work, name = args
    return state.StagingQueue(Path(root), Path(work)).try_reserve(name, 100)


class ObservationStateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(state, 'Shared receipt/byte-queue implementation is missing')

    def test_legacy_and_new_receipts_do_not_invent_hashes(self):
        legacy = {'name': 'Esg20220105_00.root', 'output': {'bytes': 10, 'entries': 2,
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
            (root / '0105').mkdir(parents=True)
            work.mkdir()
            state.atomic_json(work / 'queue_config.json', {'capacity_bytes': 200})
            queue = state.StagingQueue(root, work)
            names = [f'Esg20220105_{n:02d}.root' for n in range(3)]
            (root / '0105' / (names[0] + '.partial')).write_bytes(b'x' * 30)
            self.assertTrue(queue.try_reserve(names[0], 60))
            self.assertTrue(queue.try_reserve(names[1], 100))
            self.assertFalse(queue.try_reserve(names[2], 50))
            self.assertEqual(queue.usage(), 160)
            (root / '0105' / (names[0] + '.partial')).rename(root / '0105' / names[0])
            queue.release(names[0])
            self.assertEqual(queue.usage(), 130)
            with self.assertRaises(ValueError):
                queue.try_reserve('../escape.root', 1)

    def test_concurrent_reservations_never_overcommit(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work = Path(directory) / 'staging', Path(directory) / 'state'
            root.mkdir(); work.mkdir()
            state.atomic_json(work / 'queue_config.json', {'capacity_bytes': 350})
            args = [(str(root), str(work), f'Esg20220105_{n:02d}.root') for n in range(8)]
            with multiprocessing.get_context('fork').Pool(8) as pool:
                results = pool.map(reserve_in_child, args)
            self.assertEqual(sum(results), 3)
            self.assertEqual(state.StagingQueue(root, work).usage(), 300)


if __name__ == '__main__':
    unittest.main()
