"""Independent Slurm final audit/report for a finished mixed-schema Nhit run."""
import argparse
import importlib.util
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
    repair = None
    repair_policy = None
    if (state/'repair.json').exists():
        binding=json.loads((state/'repair.json').read_text());repair=Path(binding['state'])
        if sha256(repair/'config.json')!=binding['config_sha256'] or sha256(repair/'finished.json')!=binding['finished_sha256']:
            raise ValueError('repair binding changed')
        repair_config=json.loads((repair/'config.json').read_text())
        sys.path.insert(0,str(Path(__file__).resolve().parent/'observation_nhit_repair'))
        from policy import load_policy
        repair_policy=load_policy(repair,repair_config)
        spec=importlib.util.spec_from_file_location('repair_validate',Path(__file__).resolve().parent/'observation_nhit_repair/validate.py')
        repair_validate=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair_validate)
    manifest=json.loads((state/'manifest.json').read_text())
    gate=json.loads((state/'gate.json').read_text())
    totals={kind:dict(files=0,events=0,bytes=0) for kind in ['legacy_with_charge','scalar_only']}
    bins=Counter();pools=Counter();boundaries=Counter();failures=[];surplus=0;segments=0;rows=[]
    manual_rows=[];manual_events=0;repaired_names=[]
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
        if repair and (repair/'receipts'/(name+'.json')).exists():
            repaired_names.append(name)
            if sha256(p)!=sha256(repair/'receipts'/(name+'.json')):
                raise ValueError('merged repair receipt changed')
            repair_validate.check_scalars(data,seg,task['source_entries'],task=task,policy=repair_policy)
            source=rec['source'];prov=source['provenance']
            if (prov['code']!=repair_config['code'] or prov['git_commit']!=repair_config['git_commit'] or
                prov['contract']!=repair_config['contract'] or prov['run_id']!=repair_config['run_id'] or
                prov['repair_policy']!=repair_policy['binding'] or prov['target_stamp']!=task['input'] or
                prov['target_path']!=task['target'] or source['segments']!=seg):
                raise ValueError('repaired provenance mismatch')
            with uproot.open(target) as f:
                expected_prov=dict(prov,candidate_sha256=rec['candidate_sha256'],surplus_excluded=rec['validation']['surplus'],aligned=True)
                if json.loads(str(f['provenance']))!=expected_prov:
                    raise ValueError('embedded repair provenance changed')
            for row in seg:
                if row['poolused_source']=='user_confirmed_override':
                    count=int(np.count_nonzero(data['segment_index']==row['segment_index']))
                    manual_events+=count;manual_rows.append(dict(name=name,events=count,segment=row))
        else:
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
    if repair:
        for p in sorted((repair/'source_sessions').glob('*.json'),key=lambda p:p.name.endswith('.done.json')):
            row=json.loads(p.read_text());sessions[row['id']]=row
    attempts=list(sessions.values())
    timings=[]
    for r in sorted(generated,key=lambda r:r['name']):
        source=r['source'];session=source['session'];phases=source['root']
        row=dict(name=r['name'],events=r['entries'],source_candidates=source['entries'],
                 session_id=session['id'],source_host=session['host'],
                 source_seconds=source['elapsed_seconds'],
                 scan_seconds=phases['scan_seconds'],hit_seconds=phases['hit_seconds'],
                 header_seconds=source['header_seconds'],write_seconds=source['write_seconds'],
                 readback_seconds=source['readback_seconds'],transfer_seconds=r['transfer_seconds'],
                 eto_seconds=r['eto_seconds'],
                 python_root_startup_seconds=session['startup_seconds'],
                 process_startup_seconds=session.get('process_startup_seconds'),
                 startup_scope='Shared once per persistent session; do not sum per file',
                 git_commit=source['provenance']['git_commit'])
        row['measured_processing_seconds']=sum(row[k] for k in ['source_seconds','transfer_seconds','eto_seconds'])
        timings.append(row)
    operation_history=dict(file_attempts=[],checkpoint_restarts=[],transport_failures=[],header_retries=[])
    for p in sorted((state/'source_archive').glob('production/*/attempts.json')):
        operation_history['file_attempts'].append(dict(path=str(p),records=json.loads(p.read_text())))
    for p in sorted((state/'source_archive').glob('production/*/restart-*.json')):
        operation_history['checkpoint_restarts'].append(dict(path=str(p),record=json.loads(p.read_text())))
    for p in sorted(state.glob('controller-*.log')):
        for number,line in enumerate(p.read_text(errors='replace').splitlines(),1):
            if line.startswith('TRANSPORT_RETRY '):
                operation_history['transport_failures'].append(dict(path=str(p),line=number,record=json.loads(line.split(' ',1)[1])))
    for r in generated:
        for segment in r['source']['segments']:
            if segment.get('attempts',1)>1:
                operation_history['header_retries'].append(dict(name=r['name'],segment=segment))
    operation_history['resolved_failures']=[json.loads(p.read_text()) for p in sorted((state/'resolutions').glob('*.json'))]
    if repair:
        operation_history['repair_state']=str(repair)
        for p in sorted((repair/'source_archive').glob('production/*/attempts.json')):
            operation_history['file_attempts'].append(dict(path=str(p),records=json.loads(p.read_text())))
        for p in sorted(repair.glob('controller-*.log')):
            for number,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if line.startswith('TRANSPORT_RETRY '):
                    operation_history['transport_failures'].append(dict(path=str(p),line=number,record=json.loads(line.split(' ',1)[1])))
    operation_history['scope']='Canonical production source archive and receiver logs, including successful recoveries; diagnostic header-probe setup failures are separately retained under audit/header_probe. Failed-file attempt lists also include non-retryable data failures.'
    atomic_json(state/'operation_history.json',operation_history)
    submissions=[json.loads(p.read_text()) for p in (state/'jobs').glob('production-*.json')
        if not p.name.endswith(('.output.json','.policy.json'))]
    first=min([r['at'] for r in submissions],default=None)
    last=json.loads((state/'finished.json').read_text())['finished_at']
    if repair:last=max(last,json.loads((repair/'finished.json').read_text())['finished_at'])
    wall=last-first if first and last else None
    history=[json.loads(s) for s in (state/'staging_history.jsonl').read_text().splitlines()]
    source_history=[json.loads(s) for s in (state/'source_quota_history.jsonl').read_text().splitlines()]
    if repair:
        retained=repair_config['retained_original_stage_bytes']
        history += [dict(bytes=json.loads(line)['bytes']+retained) for line in (repair/'staging_history.jsonl').read_text().splitlines()]
        source_history += [dict(stage_bytes=json.loads(line)['stage_bytes']+retained) for line in (repair/'source_quota_history.jsonl').read_text().splitlines()]
    result=dict(status='complete' if len(rows)==681 else 'incomplete',files=len(rows),total_files=681,
        events=sum(t['events'] for t in totals.values()),total_events=manifest['total_events'],categories=totals,
        actual_mixed_bytes=sum(t['bytes'] for t in totals.values()),matching_coverage_accepted=1,
        monthly_event_coverage=sum(t['events'] for t in totals.values())/manifest['total_events'],
        accepted_duplicates=0,accepted_missing=0,surplus_excluded=surplus,failures=failures,
        bins=dict(bins),poolused=dict(pools),nhit_boundaries=dict(boundaries),segments=segments,
        pilot=gate,production_wall_seconds=wall,production_wall_scope='First production submission through final repair acceptance when applicable; includes queueing, restarts, scheduled-check gaps and repair preparation, excludes earlier inherited production and this independent audit',production_files_per_hour=len(generated)*3600/wall if wall else None,
        stage_peak_observed_bytes=max([s['bytes'] for s in history]+[s['stage_bytes'] for s in source_history]),
        stage_peak_note='Apparent bytes observed by receiver polling and per-file source reservations; not an exact filesystem high-water counter; personal allocated quota checked independently',
        source_budget_bytes=2*1024**3,source_sessions=attempts,files_catalog=rows,final_audited_at=time.time(),
        timing_catalog=str(state/'file_timings.csv'),operation_history=str(state/'operation_history.json'),
        measured_file_processing_seconds=sum(r['measured_processing_seconds'] for r in timings),
        timing_scope='Source processing + transfer + ETO alignment/readback per accepted new file; excludes startup, queueing, waiting for receiver polls and failed/restarted work. Actual wall throughput is reported separately.',
        generated_over_4096_candidates=sum(r['source']['root']['over_4096'] for r in generated),
        audit_code_sha256=sha256(__file__),
        generated_code_versions=dict(Counter(r['source']['provenance']['git_commit'] for r in generated)),
        nhit_validation_scope='Exact independent comparison on three full pilots plus inherited original receipts. New production Nhit is source-calculated, digest/readback verified; ETO does not recompute charge or compare all 30 branches.',
        source_charge_boundary_hits=sum(r['source']['root']['charge_boundary'] for r in generated),
        source_time_boundary_hits=sum(r['source']['root']['time_boundary'] for r in generated),
        user_confirmed_overrides=dict(files=len(manual_rows),segments=len(manual_rows),events=manual_events,records=manual_rows),
        header_verified_segments=segments-len(manual_rows),header_verified_events=sum(t['events'] for t in totals.values())-manual_events,
        repaired_files=repaired_names,
        source_boundary_scope='All PINC candidates for successfully accepted new files; any surplus is included in these source counters')
    if len(rows)==681 and (result['events']!=69279369 or (repair and (len(manual_rows)!=2 or len(repaired_names)!=4))):
        raise ValueError('full-month event/exception coverage incomplete')
    atomic_json(state/'final_audit.json',result)
    with (state/'file_catalog.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','kind','events','bytes','sha256','receipt','receipt_sha256'])
        writer.writeheader();writer.writerows(rows)
    if timings:
        with (state/'file_timings.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(timings[0]))
            writer.writeheader();writer.writerows(timings)
    with (state/'failed_files.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','expected_events','required_header','error'])
        writer.writeheader()
        for item in failures:
            error=item['failure'].get('error','') if isinstance(item['failure'],dict) else item['failure']
            match=re.search(r'/eos/lhaaso/raw/wcda/2022/\d{4}/.*?\.reduced\.root',error)
            writer.writerow(dict(name=item['name'],expected_events=item['expected_events'],required_header=match[0] if match else '',error=error))
    if result['files']==681 and result['events']==69279369 and not failures:
        if repair and (len(manual_rows)!=2 or len(repaired_names)!=4):
            raise ValueError('repair/exception coverage incomplete')
        atomic_json(state/'summary.json',dict(result,verified_files=result['files'],verified_events=result['events']))
        atomic_json(state/'failures.json',[])
        atomic_json(state/'completed.json',dict(status='complete',files=681,events=69279369,
            user_confirmed_override_segments=len(manual_rows),user_confirmed_override_events=manual_events,
            final_audit_sha256=sha256(state/'final_audit.json'),file_catalog_sha256=sha256(state/'file_catalog.csv'),
            audit_code_sha256=sha256(__file__),completed_at=time.time()))
    print(json.dumps({k:v for k,v in result.items() if k not in ['files_catalog','source_sessions','pilot']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);a=p.parse_args();audit(a.state)
