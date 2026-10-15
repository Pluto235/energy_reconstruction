"""Synthetic ROOT end-to-end checks, including user chi2 exception and non-7 rows."""
import json
import tempfile
import unittest
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from friends import COMMON, NATIVE, process, stamp

class Publication(unittest.TestCase):
    def fixture(self, base, bad=False, boundary=False, hit_mismatch=False):
        ep=base/'Egr000000000_01_n0010_eventout.root';stage=base/'stage';stage.mkdir();sp=stage/'Egr000000000_01_n0010.root'
        ints={'fitstat','istationcore','mc_pid','nfit','nfit0'}
        e={k:np.array([1,2],dtype='i4') if k in ints else np.array([.1,.2],dtype='f8') for k in COMMON}
        if boundary:e['rmds'][0]=20.000000077668222
        s={k:v[[1,0]].astype('i4' if k in ints else 'f4') for k,v in e.items()}
        s['chi2']=np.array([12345,67890],dtype='f4')
        if bad:s['rmds'][0]=np.nextafter(s['rmds'][0],np.float32(np.inf))
        for k in NATIVE:s[k]=np.array([2,1],dtype='i4')
        s['poolused']=np.array([5,7],dtype='i4')
        s['mc_nq03t30s0']=np.array([999,999],dtype='i4')
        with uproot.recreate(ep) as f:
            schema={k:v.dtype for k,v in e.items()};schema.update(vqsamp='var * float32',vt='var * float32')
            f.mktree('t_eventout',schema)
            f['t_eventout'].extend(dict(e,vqsamp=ak.Array([[1.]*3,[1.]*6]),vt=ak.Array([[31. if hit_mismatch else 0.]*3,[0.]*6])))
        with uproot.recreate(sp) as f:
            f.mktree('wcdaevent',{k:v.dtype for k,v in s.items()});f['wcdaevent'].extend(s)
        return dict(name=ep.name,summary_name=sp.name,**stamp(ep)),stage
    def test_complete_reordered_native_non7(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);item,stage=self.fixture(base);run=base/'run'
            self.assertEqual(process(item,run,base,stage),'verified')
            directory=run/'files'/'Egr000000000_01_n0010'
            r=json.loads((directory/'receipt.json').read_text());self.assertTrue(r['readback_verified'])
            with uproot.open(directory/r['friend_name']) as f:
                a=f['v7_friend'].arrays(library='np')
                np.testing.assert_array_equal(a['v7_summary_entry'],[1,0])
                np.testing.assert_array_equal(a['v7_poolused'],[7,5])
                np.testing.assert_array_equal(a['v7_nhit_authoritative'],[3,6])
    def test_delivery_scopes_pending_and_non7(self):
        from unittest.mock import patch
        from friends import summarize, dump
        import deliver
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);item,stage=self.fixture(base);run=base/'run'
            process(item,run,base,stage)
            missing=dict(item,name='Egr000000001_01_n0010_eventout.root')
            inv=[item,missing];summarize(run,inv)
            dump(base/'inventory.json',dict(files=inv,input_root=str(base)))
            dump(run/'contract.json',{})
            dump(run/'final_audit.json',dict(verified_files=1,verified_events=2,quarantine_verified_files=0,failures=[]))
            with patch('sys.argv',['deliver','--run',str(run),'--inventory',str(base/'inventory.json')]):deliver.main()
            out=json.loads((run/'delivery_summary.json').read_text())
            self.assertEqual(out['pending_files'],1)
            self.assertEqual(out['published_verified']['poolused_events'],{'5':1,'7':1})
            self.assertFalse(out['all_target_poolused_proven_7'])
            self.assertEqual(out['published_verified']['nhit_equal_events'],0)
            self.assertEqual(out['published_verified']['nhit_not_compared_events'],2)
            self.assertIsNone(out['published_verified']['nhit_agreement'])

    def test_native_only_and_rounding_rmds_boundary_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);item,stage=self.fixture(base,boundary=True,hit_mismatch=True);run=base/'run'
            self.assertEqual(process(item,run,base,stage),'verified')
            r=json.loads(next(run.rglob('receipt.json')).read_text())
            self.assertNotIn('nhit_mismatches',r)
            self.assertEqual(r['boundaries']['rmds_le20']['accepted_rounding_flips'],1)
            with uproot.open(next(run.rglob('*.v7friend'))) as f:
                np.testing.assert_array_equal(f['v7_friend']['v7_nhit_authoritative'].array(library='np'),[3,6])
            with uproot.open(base/item['name']) as f:
                self.assertEqual(f['t_eventout']['rmds'].array(library='np')[0],20.000000077668222)

    def test_independent_field_failure_quarantines_file(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);item,stage=self.fixture(base,bad=True);run=base/'run'
            self.assertEqual(process(item,run,base,stage),'quarantined')
            self.assertFalse(list(run.rglob('*.v7friend')))
            r=json.loads(next(run.rglob('receipt.json')).read_text())
            self.assertIn('common_field:rmds',r['failures'])

if __name__=='__main__':unittest.main()
