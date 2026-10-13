from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from observation_state import DAYS, KEEP
from sync_january_observations import validate_report, verify


class TransferAuditTests(unittest.TestCase):
    def test_metadata_size_paths_and_legacy_identity_without_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); name = f'Esg{DAYS[0]}_00.root'
            (root / name).write_bytes(b'rootdata')
            record = dict(name=name, receipt_id='b'*32, source=dict(selected_entries=2,header_entries=1),
                          output=dict(bytes=8, entries=2, header_entries=1))
            report = dict(day=DAYS[0], file_count=1, bytes=8, files=[record])
            metadata=dict(entries=2,header_entries=1,branches=KEEP,branch_entries=[2]*30,header_branches=['run'])
            with patch('sync_january_observations.read_metadata', return_value=metadata):
                verify(report, root)
                record.pop('receipt_id'); record['output']['sha256']='a'*64
                verify(report, root)
                metadata['branches']=KEEP-{'vx'}
                with self.assertRaises(RuntimeError): verify(report, root)
                metadata['branches']=KEEP
                (root/name).write_bytes(b'root')
                with self.assertRaises(RuntimeError): verify(report, root)
                record['name']='../escape.root'
                with self.assertRaises(ValueError): validate_report(report)


if __name__ == '__main__':
    unittest.main()
