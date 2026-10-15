"""Small resume/immutability test for adopting validated packages into a new version."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from friends import dump, process, sha256, stamp
import test_publication
import release_native

class Release(unittest.TestCase):
    def test_copy_revalidate_and_resume_without_touching_old(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);item,stage=test_publication.Publication().fixture(base);old=base/'old';new=base/'new'
            self.assertEqual(process(item,old,base,stage),'verified')
            ep=base/item['name'].replace('000000000','000000001');shutil.copyfile(base/item['name'],ep)
            sp=stage/item['summary_name'].replace('000000000','000000001');shutil.copyfile(stage/item['summary_name'],sp)
            other=dict(name=ep.name,summary_name=sp.name,**stamp(ep))
            self.assertEqual(process(other,old,base,stage),'verified')
            package=old/'files'/ep.name.removesuffix('_eventout.root');rp=package/'receipt.json'
            r=json.loads(rp.read_text());(package/r['friend_name']).unlink()
            r.update(status='quarantined',failures=['nhit_difference_requires_definition_or_alignment_review'],readback_verified=False)
            dump(rp,r);previous={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}
            inv=base/'inventory.json';dump(inv,dict(input_root=str(base),files=[item,other]))
            argv=['release','--old-run',str(old),'--new-run',str(new),'--inventory',str(inv),'--stage',str(stage)]
            with patch.dict(os.environ,SLURM_JOB_ID='synthetic'),patch('sys.argv',argv):
                release_native.main();release_native.main()
            self.assertEqual(previous,{str(p):sha256(p) for p in old.rglob('*') if p.is_file()})
            self.assertEqual(len(list(new.rglob('*.v7friend'))),2)
            lineage=json.loads((new/'lineage.json').read_text())
            self.assertEqual(lineage['copied_verified_files'],1);self.assertEqual(lineage['revalidated_files'],1)

if __name__=='__main__':unittest.main()
