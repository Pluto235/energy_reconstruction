"""Check both transport command paths without network access."""
import unittest
from unittest.mock import patch
import relay

class Transport(unittest.TestCase):
    def test_existing_alias(self):
        relay.ON_ETO=False
        with patch('relay.subprocess.check_output',return_value='ok') as call:
            self.assertEqual(relay.remote('print(1)'),'ok')
            self.assertEqual(call.call_args[0][0][-2],'ETO')
    def test_resume_cleanup_requires_matching_hash(self):
        import tempfile, hashlib, json, subprocess
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d)/'stage';stage.mkdir();run=Path(d)/'run';package=run/'files'/'sample';package.mkdir(parents=True)
            source=stage/'sample.root';source.write_bytes(b'valid')
            (package/'receipt.json').write_text(json.dumps({'summary_sha256':hashlib.sha256(b'valid').hexdigest()}))
            with patch.object(relay,'STAGE',str(stage)),patch.object(relay,'ON_ETO',True):
                relay.cleanup_staged(['sample.root'],str(run));self.assertFalse(source.exists())
                source.write_bytes(b'changed')
                with self.assertRaises(subprocess.CalledProcessError):relay.cleanup_staged(['sample.root'],str(run))
                self.assertTrue(source.exists())

    def test_local_control_on_eto(self):
        relay.ON_ETO=True
        with patch('relay.subprocess.check_output',return_value='ok') as call:
            relay.remote('print(1)')
            self.assertEqual(call.call_args[0][0],['python3','-c','print(1)'])
        relay.ON_ETO=False

if __name__=='__main__':unittest.main()
