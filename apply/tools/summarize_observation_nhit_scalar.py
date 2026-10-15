"""Independent Slurm final audit/report for a finished mixed-schema Nhit run."""
import argparse
import csv
import re
from collections import Counter
import json
from pathlib import Path
import sys
import shlex
import time

sys.path.insert(0,str(Path(__file__).resolve().parent/'observation_nhit_scalar'))
import numpy as np
import uproot
from common import KEYS, EDGES, atomic_json, require_slurm, sha256, stamp
from validate import SCALARS, check_scalars


def audit(state):
    require_slurm()
    if not (state/'finished.json').exists():
        raise RuntimeError('Receiver has not finished all remaining files')
    from transport import command, SSH, REMOTE
    config=json.loads((state/'config.json').read_text())
    command(['rsync','-rt','--partial','--timeout=180','--exclude=*.root','-e',shlex.join(SSH),
             REMOTE+':'+config['stage']+'/',str(state/'source_archive')+'/'])
    manifest=json.loads((state/'manifest.json').read_text())
    gate=json.loads((state/'gate.json').read_text())
    totals={kind:dict(files=0,events=0,bytes=0) for kind in ['legacy_with_charge','scalar_only']}
    bins=Counter();pools=Counter();boundaries=Counter();failures=[];surplus=0;segments=0;rows=[]
    for task in manifest['records']:
        name=task['name'];p=state/'receipts'/(name+'.json')
        if stamp(task['target'])!=task['input']:raise ValueError('INPUT_CHANGED '+name)
        if not p.exists():
            failed=state/'production'/name/'failed.json'
            failures.append(dict(name=name,expected_events=task['input']['entries'],
                failure=json.loads(failed.read_text()) if failed.exists() else 'NO_TERMINAL_FAILURE_RECORD'))
            continue
        rec=json.loads(p.read_text());target=Path(rec['path'])
        if sha256(target)!=rec['sha256'] or target.stat().st_size!=rec['bytes']:
            raise ValueError('published checksum changed '+name)
        with uproot.open(target) as f:
            data=f['t_nhit'].arrays(SCALARS,library='np');seg=f['t_segments'].arrays().to_list()
            kind='legacy_with_charge' if 'vqsamp' in f['t_nhit'] else 'scalar_only'
            if kind!=rec['kind']:raise ValueError('schema category '+name)
        check_scalars(data,seg,task['source_entries'])
        with uproot.open(task['target']) as f:
            expected=f['t_eventout'].arrays(KEYS+['nv'],library='np')
        if len(data['nv'])!=task['input']['entries']:raise ValueError('event count '+name)
        for k,v in expected.items():
            if not np.array_equal(v,data[k]):raise ValueError('published target alignment '+name+' '+k)
        total=totals[kind];total['files']+=1;total['events']+=len(data['nv']);total['bytes']+=target.stat().st_size
        bin_values,bin_counts=np.unique(np.searchsorted(EDGES,data['nhit'],side='right')-1,return_counts=True)
        bins.update({str(int(k)):int(v) for k,v in zip(bin_values,bin_counts)})
        values,counts=np.unique(data['poolused'],return_counts=True)
        pools.update({str(int(k)):int(v) for k,v in zip(values,counts)})
        boundaries.update({str(e):int(np.count_nonzero(data['nhit']==e)) for e in EDGES})
        surplus+=rec['validation'].get('surplus',0);segments+=len(seg)
        rows.append(dict(name=name,kind=kind,events=len(data['nv']),bytes=target.stat().st_size,
                         sha256=rec['sha256'],receipt=str(p),receipt_sha256=sha256(p)))
    generated=[json.loads(p.read_text()) for p in (state/'receipts').glob('*.json') if not json.loads(p.read_text())['inherited']]
    sessions={}
    for p in sorted((state/'source_sessions').glob('*.json'),key=lambda p:p.name.endswith('.done.json')):
        row=json.loads(p.read_text());sessions[row['id']]=row
    attempts=list(sessions.values())
    submissions=[json.loads(p.read_text()) for p in (state/'jobs').glob('production-*.json')
        if not p.name.endswith(('.output.json','.policy.json'))]
    first=min([r['at'] for r in submissions],default=None)
    last=json.loads((state/'finished.json').read_text())['finished_at']
    wall=last-first if first and last else None
    history=[json.loads(s) for s in (state/'staging_history.jsonl').read_text().splitlines()]
    source_history=[json.loads(s) for s in (state/'source_quota_history.jsonl').read_text().splitlines()]
    result=dict(status='complete' if len(rows)==681 else 'incomplete',files=len(rows),total_files=681,
        events=sum(t['events'] for t in totals.values()),total_events=manifest['total_events'],categories=totals,
        actual_mixed_bytes=sum(t['bytes'] for t in totals.values()),matching_coverage_accepted=1,
        monthly_event_coverage=sum(t['events'] for t in totals.values())/manifest['total_events'],
        accepted_duplicates=0,accepted_missing=0,surplus_excluded=surplus,failures=failures,
        bins=dict(bins),poolused=dict(pools),nhit_boundaries=dict(boundaries),segments=segments,
        pilot=gate,production_wall_seconds=wall,production_wall_scope='First production submission through receiver final checksum audit; includes queueing and restarts, excludes earlier inherited production and this independent audit',production_files_per_hour=len(generated)*3600/wall if wall else None,
        stage_peak_observed_bytes=max([s['bytes'] for s in history]+[s['stage_bytes'] for s in source_history]),
        stage_peak_note='Apparent bytes observed by receiver polling and per-file source reservations; not an exact filesystem high-water counter; personal allocated quota checked independently',
        source_budget_bytes=2*1024**3,source_sessions=attempts,files_catalog=rows,final_audited_at=time.time(),
        audit_code_sha256=sha256(__file__),
        generated_code_versions=dict(Counter(r['source']['provenance']['git_commit'] for r in generated)),
        nhit_validation_scope='Exact independent comparison on three full pilots plus inherited original receipts. New production Nhit is source-calculated, digest/readback verified; ETO does not recompute charge or compare all 30 branches.',
        source_charge_boundary_hits=sum(r['source']['root']['charge_boundary'] for r in generated),
        source_time_boundary_hits=sum(r['source']['root']['time_boundary'] for r in generated),
        source_boundary_scope='All PINC candidates for successfully accepted new files; any surplus is included in these source counters')
    atomic_json(state/'final_audit.json',result)
    with (state/'file_catalog.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','kind','events','bytes','sha256','receipt','receipt_sha256'])
        writer.writeheader();writer.writerows(rows)
    with (state/'failed_files.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','expected_events','required_header','error'])
        writer.writeheader()
        for item in failures:
            error=item['failure'].get('error','') if isinstance(item['failure'],dict) else item['failure']
            match=re.search(r'/eos/lhaaso/raw/wcda/2022/\d{4}/.*?\.reduced\.root',error)
            writer.writerow(dict(name=item['name'],expected_events=item['expected_events'],required_header=match[0] if match else '',error=error))
    print(json.dumps({k:v for k,v in result.items() if k not in ['files_catalog','source_sessions','pilot']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);a=p.parse_args();audit(a.state)
