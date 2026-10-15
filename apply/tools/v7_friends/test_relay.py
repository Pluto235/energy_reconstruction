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
    def test_local_control_on_eto(self):
        relay.ON_ETO=True
        with patch('relay.subprocess.check_output',return_value='ok') as call:
            relay.remote('print(1)')
            self.assertEqual(call.call_args[0][0],['python3','-c','print(1)'])
        relay.ON_ETO=False

if __name__=='__main__':unittest.main()
