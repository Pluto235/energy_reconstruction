"""Slurm-only creation/readback of aligned scalar test friends and timing summary."""
import json
import os
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot
from benchmark import STATE, INPUT, NAMES, align, check_candidate
from common import KEYS, atomic_json, digest_column, require_slurm, sha256, sync_dir


def main():
    require_slurm()
    if not (STATE/'complete.json').exists():raise RuntimeError('paired benchmark is not complete')
    results=[]
    for name in NAMES:
        started=time.monotonic()
        checked=check_candidate(STATE,name)
        with uproot.open(INPUT/name[7:11]/name) as f:target=f['t_eventout'].arrays(KEYS,library='np')
        with uproot.open(STATE/(name+'.candidate.root')) as f:
            candidate=f['t_nhit'].arrays(library='np');segments=f['t_segments'].arrays()
        order,surplus=align(target,candidate)
        if np.any(order<0):raise ValueError('missing target')
        data={k:v[order] for k,v in candidate.items()}
        data['input_entry']=np.arange(len(order),dtype='i8')
        output=STATE/(name+'.aligned.root')
        if output.exists():raise FileExistsError(output)
        with uproot.recreate(output,compression=uproot.ZLIB(4)) as f:
            f.mktree('t_nhit',{k:'int64' if k in ['input_entry','ihep_entry'] else 'int32' for k in data})
            f['t_nhit'].extend(data)
            f.mktree('t_segments',{k:'int32' for k in ['segment_index']+KEYS[:3]+['poolused']} |
                    {k:'string' for k in ['path','uuid','source_filename','header_sha256']})
            if len(segments):f['t_segments'].extend(segments)
            f['provenance']=json.dumps(dict(benchmark_only=True,production_accepted=False,
                candidate_sha256=sha256(STATE/(name+'.candidate.root')),
                target=str(INPUT/name[7:11]/name),selection='pincness < 1.1',charge_saved=False))
        with uproot.open(output) as f:
            actual=f['t_nhit'].arrays(library='np')
            if set(actual)!=set(data) or 'vqsamp' in actual:raise ValueError('aligned schema')
            for k in data:
                if digest_column(ak.Array(actual[k]))!=digest_column(ak.Array(data[k])):raise ValueError(k)
            if f['t_segments'].arrays().to_list()!=segments.to_list():raise ValueError('aligned segment readback')
        with output.open('rb') as f:os.fsync(f.fileno())
        sync_dir(STATE)
        digest=sha256(output);eto_seconds=time.monotonic()-started
        comparison=json.loads((STATE/(name+'.comparison.json')).read_text())
        old=comparison['baseline_source'];new=comparison['candidate_source'];tx=comparison['transfer_seconds']
        baseline_total=old['elapsed_seconds']+tx['baseline.ready.json']+tx['baseline.friend.root']+comparison['baseline_validation']['verify_seconds']
        candidate_total=new['elapsed_seconds']+tx['.candidate.ready.json']+tx['.candidate.root']+eto_seconds
        row=dict(name=name,events=checked['events'],raw_entries=new['raw_entries'],
            baseline_source_seconds=old['elapsed_seconds'],candidate_source_seconds=new['elapsed_seconds'],
            source_speedup=old['elapsed_seconds']/new['elapsed_seconds'],
            baseline_eto_seconds=comparison['baseline_validation']['verify_seconds'],candidate_eto_seconds=eto_seconds,
            baseline_total_seconds=baseline_total,candidate_total_seconds=candidate_total,
            processing_speedup=baseline_total/candidate_total,baseline_bytes=old['bytes'],
            candidate_transfer_bytes=new['bytes'],candidate_aligned_bytes=output.stat().st_size,
            candidate_aligned_sha256=digest,matching=checked,surplus=surplus,source_phases=new['root'] |
            dict(header_seconds=new['header_seconds'],readback_seconds=new['readback_seconds']))
        atomic_json(STATE/(name+'.final.json'),row);results.append(row)
        print('ALIGNED',name,row['events'],row['candidate_aligned_bytes'],eto_seconds,flush=True)
    aggregate=dict(files=len(results),events=sum(r['events'] for r in results),results=results,
        source_speedup=sum(r['baseline_source_seconds'] for r in results)/sum(r['candidate_source_seconds'] for r in results),
        processing_speedup=sum(r['baseline_total_seconds'] for r in results)/sum(r['candidate_total_seconds'] for r in results),
        baseline_bytes=sum(r['baseline_bytes'] for r in results),aligned_bytes=sum(r['candidate_aligned_bytes'] for r in results),
        excluded='scheduler/startup delay and controller polling; no monthly concurrency-throughput test',
        finalizer_sha256=sha256(__file__),production_changed=False)
    atomic_json(STATE/'summary.json',aggregate)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
