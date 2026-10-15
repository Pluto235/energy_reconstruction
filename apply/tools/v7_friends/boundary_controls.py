"""Scan published MC files for controls contradicting a strict time-boundary hypothesis."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from friends import dump, sha256, stamp


def inspect_control(receipt):
    r=json.loads(receipt.read_text())
    if r['status'] != 'verified':
        return None
    assert stamp(Path(r['source_path'])) == r['source_stat']
    with uproot.open(r['source_path']) as f:
        a=f['t_eventout'].arrays(['vt','vqsamp'],library='ak')
        boundary=ak.to_numpy(ak.sum((a['vqsamp']>=.3)&(abs(a['vt'])==30),axis=1))
        indices=np.flatnonzero(boundary)
        cases=[]
        if len(indices):
            friends=list(receipt.parent.glob('*.v7friend'))
            assert len(friends)==1
            with uproot.open(friends[0]) as ff:
                native=ff['v7_friend'].arrays(['v7_nhit_authoritative','v7_eventout_entry','v7_summary_entry'],library='np')
            assert np.array_equal(native['v7_eventout_entry'],np.arange(len(boundary)))
            for i in indices:
                q,t=a['vqsamp'][i],a['vt'][i]
                inclusive=int(ak.sum((q>=.3)&(abs(t)<=30)))
                strict=int(ak.sum((q>=.3)&(abs(t)<30)))
                cases.append(dict(event_entry=int(i),summary_entry=int(native['v7_summary_entry'][i]),
                    native=int(native['v7_nhit_authoritative'][i]),inclusive=inclusive,strict=strict,
                    boundary_times=ak.to_list(t[(q>=.3)&(abs(t)==30)])))
    return dict(file=r['file'],events=len(boundary),cases=cases)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    receipts=sorted((args.run/'files').glob('*/receipt.json'))
    before={str(p):sha256(p) for p in receipts}
    files=events=0;cases=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(inspect_control,receipts,chunksize=10):
            if r is None:continue
            files+=1;events+=r['events']
            cases.extend(dict(file=r['file'],**c) for c in r['cases'])
            if files%1000==0:print(f'files={files}, boundary_events={len(cases)}',flush=True)
    assert before=={str(p):sha256(p) for p in receipts}
    out=dict(job_id=os.environ['SLURM_JOB_ID'],code_sha256=sha256(__file__),files=files,events=events,cases=cases,
        receipts_unchanged=True,inclusive_mismatches=sum(c['native']!=c['inclusive'] for c in cases),
        strict_mismatches=sum(c['native']!=c['strict'] for c in cases))
    assert files==9981 and events==38448238
    dump(args.out,out);print(json.dumps({k:v for k,v in out.items() if k!='cases'}))

if __name__=='__main__':main()
