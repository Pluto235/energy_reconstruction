"""Small regressions; ROOT fixture tests run inside the IHEP batch environment."""
import importlib.util
import os
import resource
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('sync_observation_time', HERE / 'sync_observation_time.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(control=str(self.root), eos_root='/eos/user/l/liushijie/time_test',
                           eos_server='root://eos01.ihep.ac.cn', queue_capacity=1024,
                           allowed_names=['Esg20220101_00.root'], config_id='a'*32, git_commit='b'*40)
        for d in ('requests', 'ready', 'acks', 'file_locks', 'states'):
            (self.root/d).mkdir()
        self.request = dict(name='Esg20220101_00.root', request_id='c'*32,
                            record_id='v2:'+'d'*32, config_id='a'*32,
                            git_commit='b'*40, entries=4, bytes=100)

    def test_receipt_identity_is_immutable(self):
        m = self.module
        m.put_request(self.config, self.request)
        self.assertEqual(m.put_request(self.config, self.request), self.request)
        changed = dict(self.request, record_id='v2:'+'e'*32)
        with self.assertRaises(ValueError):
            m.put_request(self.config, changed)
        self.assertEqual(m.read_json(self.root/'requests'/('Esg20220101_00.root.json')), self.request)

    def test_reservations_include_untracked_partial_and_survive_restart(self):
        m = self.module
        with patch.object(m, 'eos_listing', return_value={'untracked.partial': 300}):
            self.assertTrue(m.reserve(self.config, self.request, 700))
            self.assertFalse(m.reserve(self.config, dict(self.request, request_id='f'*32), 800))
            self.assertEqual(m.queue_usage(self.config), 1000)
        self.assertTrue((self.root/'queue_ledger.json').exists())

    def test_real_xrdfs_listing_counts_size_column_and_skips_directories(self):
        listing = ('drwxr-x--- user group 59 2026-10-14 02:55:00 /eos/time/0101\n'
                   '-rw-r--r-- user group 42 2026-10-14 02:55:00 /eos/time/0101/file.root\n'
                   '-rw-r--r-- user group 17 2026-10-14 02:55:00 /eos/time/0101/file.partial\n')
        with patch.object(self.module, 'eos_command', return_value=listing):
            self.assertEqual(self.module.eos_listing(self.config),
                             {'/eos/time/0101/file.root': 42, '/eos/time/0101/file.partial': 17})

    def test_unknown_xrdfs_listing_cannot_undercount_queue(self):
        with patch.object(self.module, 'eos_command', return_value='unrecognized listing format'):
            with self.assertRaises(ValueError):
                self.module.queue_usage(self.config)

    def test_cleanup_requires_exact_receipt(self):
        m = self.module
        m.put_request(self.config, self.request)
        with patch.object(m, 'eos_remove') as remove:
            with self.assertRaises(ValueError):
                m.acknowledge(self.config, dict(self.request, request_id='e'*32))
            remove.assert_not_called()

    def test_ack_is_durable_before_cleanup_and_retry_finishes(self):
        m = self.module
        m.put_request(self.config, self.request)
        m.atomic_json(self.root/'ready'/(self.request['name']+'.json'), dict(self.request, products={}))
        with patch.object(m, 'eos_remove', side_effect=OSError('transfer interrupted')):
            with self.assertRaises(OSError):
                m.acknowledge(self.config, self.request)
        self.assertTrue((self.root/'acks'/(self.request['name']+'.json')).exists())
        with patch.object(m, 'eos_remove'):
            m.acknowledge(self.config, self.request)
        self.assertEqual(m.read_json(self.root/'states'/(self.request['name']+'.json'))['state'], 'cleaned')

    def test_foreign_cleanup_path_rejected(self):
        m = self.module
        with self.assertRaises(ValueError):
            m.product_paths(self.config, dict(self.request, request_id='../../old_archive'))

    def test_concurrent_reservations_do_not_oversubscribe(self):
        m = self.module
        second = dict(self.request, request_id='f'*32)
        with patch.object(m, 'eos_listing', return_value={}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda req: m.reserve(self.config, req, 700), [self.request, second]))
            self.assertEqual(sorted(results), [False, True])
            self.assertEqual(m.queue_usage(self.config), 700)

    def test_receiver_rejects_foreign_product_before_download(self):
        m = self.module
        ready = dict(self.request, products={k:dict(path='/other_project/file',bytes=1) for k in ('root','summary','exceptions')})
        with patch.object(m, 'transfer') as download:
            with self.assertRaises(ValueError):
                m.receive_one(self.config,ready)
            download.assert_not_called()

    def test_interrupted_upload_does_not_publish_or_cleanup(self):
        m = self.module
        local = {key: self.root / key for key in ('root', 'summary', 'exceptions')}
        for path in local.values():
            path.write_bytes(b'preserve on transfer failure')
        with patch.object(m, 'eos_command') as command, patch.object(m, 'xrdcopy', side_effect=OSError('upload interrupted')):
            with self.assertRaisesRegex(OSError, 'upload interrupted'):
                m.upload(self.config, self.request, local)
        self.assertEqual([call.args[1] for call in command.call_args_list], ['mkdir'])
        self.assertTrue(all(path.exists() for path in local.values()))

    def test_lost_ack_reply_completes_pilot_without_monthly_marker(self):
        m=self.module
        config=dict(self.config,state=str(self.root/'eto'))
        state=Path(config['state'])
        m.atomic_json(state/'receipts'/(self.request['name']+'.json'),dict(self.request,summary={'status':[4,0,0,0,0,0,0]}))
        def remote(config,action,payload=None):
            if action=='ready':
                return [self.request]
            if action=='status':
                return dict(cleaned=[self.request['name']],queued_bytes=0)
            raise AssertionError('unexpected remote action')
        with patch.object(m,'request_receipts'),patch.object(m,'remote',side_effect=remote),patch.object(m,'receive_one',side_effect=OSError('ACK reply lost')),patch.object(m.time,'sleep',side_effect=AssertionError('durable cleanup should complete without another poll')):
            m.receive_loop(config,{self.request['name']})
        self.assertFalse((state/'completed.json').exists())
        result=m.read_json(state/'pilot_completed.json')
        self.assertEqual(result['files'],1)
        self.assertEqual(result['errors'],{})
        self.assertEqual(result['names'],[self.request['name']])


@unittest.skipUnless(os.environ.get('OBS_TIME_BINARY'), 'ROOT checks run in HTCondor')
class ROOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ['OBS_TIME_BINARY'])
        if not cls.binary.is_file():
            raise AssertionError('Recovery executable has not been implemented')
        directory = os.environ.get('OBS_TIME_ROOT_FIXTURES')
        if directory:
            cls.fixtures = Path(directory)
        else:
            temporary = tempfile.TemporaryDirectory()
            cls.addClassCleanup(temporary.cleanup)
            cls.fixtures = Path(temporary.name)
        subprocess.run([str(cls.binary), '--self-test', str(cls.fixtures)], check=True, timeout=120)

    def setUp(self):
        spec = importlib.util.spec_from_file_location('sync_observation_time', HERE / 'sync_observation_time.py')
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.config = dict(control=str(root), eos_root='/eos/user/l/liushijie/time_test',
                           eos_server='root://eos01.ihep.ac.cn',queue_capacity=1024,
                           allowed_names=['Esg20220101_00.root'],config_id='a'*32,git_commit='b'*40,
                           observation_root=str(root/'observed'),receipt_root=str(root/'obsstate/receipts'),
                           state=str(root/'state'),destination=str(root/'destination'))
        name = 'Esg20220101_00.root'
        observed = root/'observed/0101'/name
        observed.parent.mkdir(parents=True)
        shutil.copyfile(self.fixtures/'observed/0101'/name,observed)
        self.request = dict(name=name, request_id='c'*32,record_id='v2:'+'d'*32,
                            config_id='a'*32,git_commit='b'*40,entries=4,bytes=observed.stat().st_size)
        for directory in ('requests','receipts','incoming'):
            (root/'state'/directory).mkdir(parents=True)
        self.m.atomic_json(root/'state/requests'/(name+'.json'),self.request)
        self.m.atomic_json(Path(self.config['receipt_root'])/(name+'.json'),self.request)
        self.local = dict(root=self.fixtures/'hybrid.root',summary=self.fixtures/'hybrid.root.summary.json',
                          exceptions=self.fixtures/'hybrid.root.exceptions.jsonl')
        paths = self.m.product_paths(self.config,self.request)
        self.mapping = {paths[k]:v for k,v in self.local.items()}
        self.ready = dict(self.request, products={k:dict(path=paths[k],bytes=v.stat().st_size) for k,v in self.local.items()})

    def download(self, config, source, target):
        shutil.copyfile(self.mapping[source],target)

    def test_full_alignment_accepts_real_root_fixture(self):
        summary = self.m.verify_file(self.config,self.request,self.local['root'],self.local['summary'],self.local['exceptions'])
        self.assertEqual(summary['status'],[4,0,0,0,0,0,0])
        self.assertEqual(summary['reused_time'],2)

    def test_same_count_wrong_event_order_rejected(self):
        import uproot
        observed = Path(self.config['observation_root'])/'0101'/self.request['name']
        with uproot.open(observed) as f:
            arrays = f['t_eventout'].arrays(library='np')
        arrays = {k:v[[1,0,2,3]] for k,v in arrays.items()}
        with uproot.recreate(observed) as f:
            f.mktree('t_eventout',{k:v.dtype for k,v in arrays.items()})
            f['t_eventout'].extend(arrays)
        self.request['bytes'] = observed.stat().st_size
        self.m.atomic_json(Path(self.config['receipt_root'])/(self.request['name']+'.json'),self.request)
        with self.assertRaisesRegex(ValueError,'sequence mismatch'):
            self.m.verify_file(self.config,self.request,self.local['root'],self.local['summary'],self.local['exceptions'])

    def test_root_write_failure_is_not_reported_as_success(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree,directory)
        output = directory/'failed.root'
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE,(1024,1024))
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result = subprocess.run([str(self.binary),'--input',str(self.fixtures/'Esg20220101_00.root'),
                                 '--output',str(output),'--expected-entries','4','--request-id','f'*32,
                                 '--git-commit','b'*40,'--reduced-root',str(self.fixtures/'reduced')],
                                preexec_fn=limits,capture_output=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(Path(str(output)+'.summary.json').exists())

    def test_truncated_root_read_fails(self):
        directory=Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree,directory)
        path=directory/'truncated.root'
        data=(self.fixtures/'hybrid.root').read_bytes()
        path.write_bytes(data[:-200])
        result=subprocess.run([str(self.binary),'--verify',str(path)],capture_output=True,timeout=30)
        self.assertNotEqual(result.returncode,0)

    def test_restart_after_publish_before_receipt(self):
        original = self.m.atomic_json
        receipt_path = Path(self.config['state'])/'receipts'/(self.request['name']+'.json')
        def interrupt(path,value):
            if Path(path)==receipt_path:
                raise OSError('interrupted before durable receipt')
            return original(path,value)
        with patch.object(self.m,'transfer',side_effect=self.download),patch.object(self.m,'atomic_json',side_effect=interrupt),patch.object(self.m,'remote') as ack:
            with self.assertRaises(OSError):
                self.m.receive_one(self.config,self.ready)
            ack.assert_not_called()
        self.assertFalse(receipt_path.exists())
        with patch.object(self.m,'transfer',side_effect=AssertionError('unexpected recopy')),patch.object(self.m,'remote',return_value={'cleaned':True}):
            receipt,cleaned=self.m.receive_one(self.config,self.ready)
        self.assertTrue(cleaned)
        self.assertTrue(receipt_path.exists())

    def test_restart_after_receipt_before_remote_ack(self):
        with patch.object(self.m,'transfer',side_effect=self.download),patch.object(self.m,'remote',side_effect=OSError('SSH interrupted')):
            with self.assertRaises(OSError):
                self.m.receive_one(self.config,self.ready)
        with patch.object(self.m,'transfer',side_effect=AssertionError('unexpected recopy')),patch.object(self.m,'verify_file',side_effect=AssertionError('unexpected repeated full validation')),patch.object(self.m,'remote',return_value={'cleaned':True}):
            receipt,cleaned=self.m.receive_one(self.config,self.ready)
        self.assertTrue(cleaned)


if __name__ == '__main__':
    unittest.main()
