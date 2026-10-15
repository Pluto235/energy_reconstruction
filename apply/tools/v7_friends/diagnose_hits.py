"""Inspect raw threshold-adjacent hits in quarantined Nhit mismatch events; never republish."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import uproot
from friends import dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    cases=[]
    for receipt in sorted((args.run/'files').glob('*/receipt.json')):
        r=json.loads(receipt.read_text())
        if not r.get('nhit_mismatches'):continue
        with uproot.open(r['source_path']) as f:
            for ex in r['nhit_examples']:
                i=ex['event_entry'];a=f['t_eventout'].arrays(['vqsamp','vt'],entry_start=i,entry_stop=i+1,library='np')
                q,t=a['vqsamp'][0],a['vt'][0]
                q0=np.asarray(.3,dtype=q.dtype);t0=np.asarray(30,dtype=t.dtype)
                nearq=(q>=np.nextafter(q0,-np.inf,dtype=q.dtype))&(q<=np.nextafter(q0,np.inf,dtype=q.dtype))
                neart=(np.abs(t)>=np.nextafter(t0,-np.inf,dtype=t.dtype))&(np.abs(t)<=np.nextafter(t0,np.inf,dtype=t.dtype))
                ids=np.flatnonzero(nearq|neart)
                cases.append(dict(file=r['file'],**ex,hit_dtype=dict(q=str(q.dtype),t=str(t.dtype)),
                    q_at_threshold_and_time_pass=int(np.sum((q==q0)&(np.abs(t)<=30))),
                    t_at_threshold_and_q_pass=int(np.sum((np.abs(t)==t0)&(q>=.3))),
                    threshold_adjacent_hits=[dict(hit_index=int(j),q=float(q[j]),t=float(t[j]),passes=bool(q[j]>=q0 and abs(t[j])<=t0)) for j in ids[:30]]))
    dump(args.run/'hit_threshold_diagnostics.json',dict(job_id=os.environ['SLURM_JOB_ID'],cases=cases,
        scope='Up to five recorded mismatch events per failed file; up to 30 threshold-adjacent hits per event',
        qualification='Adjacency is diagnostic evidence only; no upstream precision reconstruction or mismatch waiver. Files remain quarantined.'))
    print(json.dumps(dict(cases=len(cases),with_exact_q_boundary=sum(x['q_at_threshold_and_time_pass']>0 for x in cases),with_exact_t_boundary=sum(x['t_at_threshold_and_q_pass']>0 for x in cases))))

if __name__=='__main__':main()
