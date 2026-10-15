"""Independent published-file readback; run under Slurm after the batch."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import uproot
from friends import NATIVE, sha256, stamp, native_sum, dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    inv=json.loads(args.inventory.read_text()); rows=[]
    for item in inv['files']:
        stem=item['name'].removesuffix('_eventout.root'); directory=args.run/'files'/stem
        q=dict(file=item['name'],status='failed')
        try:
            r=json.loads((directory/'receipt.json').read_text());ep=Path(inv['input_root'])/item['name']
            assert stamp(ep)==r['source_stat'],'original source stat changed'
            if r['status']!='verified':
                assert not list(directory.glob('*.v7friend')),'quarantined file published'
                q['status']='quarantine_verified'
            else:
                fpath=directory/r['friend_name'];assert sha256(fpath)==r['friend_sha256'],'friend hash changed'
                with uproot.open(fpath) as f:
                    t=f['v7_friend'];a=t.arrays(library='np');n=r['events'];assert t.num_entries==n
                    assert set(a)==set(r['field_coverage'])
                    np.testing.assert_array_equal(a['v7_eventout_entry'],np.arange(n))
                    assert len(np.unique(a['v7_summary_entry']))==n
                    assert np.all((a['v7_summary_entry']>=0)&(a['v7_summary_entry']<r['summary_events']))
                    assert np.all(a['v7_match_status']==1)
                    native={k:a['v7_'+k] for k in NATIVE}
                    for k,v in native.items():assert str(v.dtype)==r['native_ranges'][k]['dtype']
                    np.testing.assert_array_equal(native_sum(native),a['v7_nhit_authoritative'])
                    meta=json.loads(str(f['v7_metadata']))
                    for k in ('event_sha256','summary_sha256','code_sha256'):assert meta[k]==r[k]
                q.update(status='verified',events=n)
        except Exception as exc:q['error']=repr(exc)
        rows.append(q)
    out=dict(job_id=os.environ['SLURM_JOB_ID'],target_files=len(rows),verified_files=sum(x['status']=='verified' for x in rows),
             quarantine_verified_files=sum(x['status']=='quarantine_verified' for x in rows),failures=[x for x in rows if x['status']=='failed'],
             verified_events=sum(x.get('events',0) for x in rows),source_verification='all original stat identities unchanged; SHA256 recorded during processing',files=rows)
    dump(args.run/'final_audit.json',out);print(json.dumps({k:v for k,v in out.items() if k!='files'}))
    if out['failures']:raise SystemExit(1)

if __name__=='__main__':main()
