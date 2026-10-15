"""Slurm receiver: gated pilot, persistent workers, durable receipt/ack publication."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import traceback

from common import atomic_json, require_slurm, sha256, stamp
from prepare import BASE, OLD, PILOT, freeze, prepare
from source import CONTRACT
from transport import SSH, REMOTE, quota, remote, remote_py, transfer
from validate import accept

HERE=Path(__file__).resolve().parent
EXCLUDED=['lhws183.ihep.ac.cn','bws0995.ihep.ac.cn']
BASELINE=dict(zip(PILOT,[213.4,216.3,102.2]))


def snapshot(stage):
    return json.loads(remote_py('''import json,subprocess
from pathlib import Path
p=Path(%r)
raw=subprocess.check_output(['condor_q','-name','scheduler@schedd11.ihep.ac.cn','-constraint','Owner == "liushijie"','-json','-attributes','ClusterId,ProcId,Cmd,Args,Owner,JobStatus,RemoteHost,CurrentHosts,ExitCode,HoldReason,JobStartDate,EnteredCurrentStatus'],text=True)
jobs=[r for r in json.loads(raw.strip() or '[]') if r.get('Cmd','').startswith(str(p)+'/')]
print(json.dumps(dict(jobs=jobs,ready=[str(f.relative_to(p)) for f in p.glob('*/*/ready.json')],
failed=[str(f.relative_to(p)) for f in p.glob('*/*/failed.json')],sessions=[json.loads(f.read_text()) for f in (p/'sessions').glob('*.done.json')],
stage_bytes=int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]))))
''' % stage))


def submit(state, stage, phase, shard, generation):
    label=f'{phase}-{shard}-{generation}'
    record=state/'jobs'/(label+'.json')
    if record.exists():
        prior=json.loads(record.read_text())
        if 'cluster' not in prior:
            # Reconcile an interrupted reply by exact Cmd and arguments; never blind resubmit.
            found=[j for j in snapshot(stage)['jobs'] if j.get('Args')==f'{stage} {phase} {shard}' and j['JobStatus'] not in [3,4]]
            if len(found)!=1:
                raise RuntimeError('ambiguous prior submission '+label)
            prior['cluster']=found[0]['ClusterId'];atomic_json(record,prior)
        return prior['cluster']
    cmd=('source /afs/ihep.ac.cn/soft/common/sysgroup/hep_job/setup_hepjob.sh; cd '+shlex.quote(stage)+
         '; hep_sub ./worker.sh -argu '+shlex.quote(stage)+f' {phase} {shard} -g lhaaso -os AlmaLinux9 -m 4000 -cpu 1 -schedd schedd11'+
         (' -wt short' if phase=='pilot' else '')+
         ' -o '+label+'.out -e '+label+'.err')
    atomic_json(record,dict(label=label,status='submission_pending',at=time.time(),phase=phase,shard=shard))
    output=remote(cmd,attempts=1)
    atomic_json(state/'jobs'/(label+'.output.json'),dict(output=output))
    match=re.search(r'cluster\s+(\d+)',output,re.I)
    if not match:
        raise RuntimeError('submission ambiguous: '+output)
    cluster=int(match[1])
    atomic_json(record,dict(label=label,status='submitted',at=time.time(),phase=phase,shard=shard,cluster=cluster))
    policy=remote_py('''import json,subprocess
base=['-name','scheduler@schedd11.ihep.ac.cn'];job=%r
raw=subprocess.check_output(['condor_q']+base+[job,'-json','-attributes','Owner,Cmd,Args,Requirements,JobStatus,RemoteHost'],text=True)
a=json.loads(raw)[0];assert a['Owner']=='liushijie' and a['Cmd'].startswith(%r+'/')
old=a['Requirements'];assert old.startswith('/Expr(') and old.endswith(')/')
new='('+old[6:-2]+')'+''.join(' && (TARGET.Machine != '+json.dumps(h)+')' for h in %r)
subprocess.run(['condor_qedit']+base+[job,'Requirements',new],check=True,stdout=subprocess.PIPE)
if any(h in a.get('RemoteHost','') for h in %r):
    subprocess.run(['condor_vacate_job']+base+['-fast',job],check=True,stdout=subprocess.PIPE)
print(json.dumps(dict(original=old,updated=new,job=job)))
''' % (str(cluster)+'.0',stage,EXCLUDED,EXCLUDED))
    atomic_json(state/'jobs'/(label+'.policy.json'),json.loads(policy))
    print('SUBMITTED',label,cluster,flush=True)
    return cluster


def acknowledge(stage, phase, name, receipt):
    remote_py('''import json,os
from pathlib import Path
p=Path(%r)/%r/%r
r=json.loads((p/'ready.json').read_text());a=json.loads(%r)
assert r['sha256']==a['candidate_sha256'] and a['accepted_entries']<=r['entries']
tmp=p/'ack.partial';tmp.write_text(json.dumps(a));f=tmp.open('rb');os.fsync(f.fileno());f.close();os.replace(tmp,p/'ack.json')
fd=os.open(str(p),os.O_DIRECTORY);os.fsync(fd);os.close(fd)
(p/'candidate.root').unlink(missing_ok=True)
''' % (stage,phase,name,json.dumps(dict(candidate_sha256=receipt['candidate_sha256'],
    accepted_entries=receipt['entries'],final_sha256=receipt['sha256'],eto_path=receipt['path'],accepted_at=receipt['accepted_at']))))


def receive(state, stage, phase, task, config):
    name=task['name'];folder=state/phase/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():
        receipt=json.loads((folder/'receipt.json').read_text())
        if sha256(receipt['path'])!=receipt['sha256']:
            raise ValueError('published target changed before source ack')
    else:
        start=time.monotonic()
        for part in ['ready.json','candidate.root']:
            transfer(REMOTE+':'+stage+'/'+phase+'/'+name+'/'+part,str(folder/part))
        tx=time.monotonic()-start
        reference=BASE/'full'/name[7:11]/name.replace('.root','.nhit.root') if phase=='pilot' else None
        target=state/'pilot_results'/name.replace('.root','.nhit.root') if phase=='pilot' else BASE/'full'/name[7:11]/name.replace('.root','.nhit.root')
        receipt=accept(state,folder,task,config,target,reference)
        receipt['transfer_seconds']=tx;atomic_json(folder/'receipt.json',receipt)
    if phase=='production':
        atomic_json(state/'receipts'/(name+'.json'),receipt)
    acknowledge(stage,phase,name,receipt)
    atomic_json(folder/'cleaned.json',dict(at=time.time(),candidate_sha256=receipt['candidate_sha256']))
    (folder/'candidate.root').unlink(missing_ok=True)
    print('ACCEPTED',phase,name,receipt['entries'],flush=True)


def poll(state, stage, config, phase, tasks):
    snap=snapshot(stage)
    atomic_json(state/'scheduler.json',snap)
    with (state/'staging_history.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=time.time(),bytes=snap['stage_bytes']))+'\n')
    q=quota();atomic_json(state/'quota.json',q)
    byname={r['name']:r for r in tasks}
    for rel in snap['failed']:
        ph,name,_=rel.split('/')
        if ph!=phase:continue
        dst=state/ph/name;dst.mkdir(parents=True,exist_ok=True)
        if not (dst/'failed.json').exists():
            transfer(REMOTE+':'+stage+'/'+rel,str(dst/'failed.json'))
    for rel in snap['ready']:
        ph,name,_=rel.split('/')
        if ph!=phase or name not in byname:continue
        folder=state/ph/name
        if (folder/'cleaned.json').exists() or (folder/'failed.json').exists():continue
        try:
            receive(state,stage,phase,byname[name],config)
        except Exception as e:
            if 'INPUT_CHANGED' in str(e):
                remote_py('import json; from pathlib import Path; Path('+repr(stage+'/STOP_EXPANSION.json')+').write_text('+repr(json.dumps(dict(error=str(e),at=time.time())))+')')
                raise
            atomic_json(folder/'failed.json',dict(name=name,error=repr(e),traceback=traceback.format_exc(),at=time.time(),phase='ETO'))
            print('QUARANTINED',phase,name,repr(e),flush=True)
    done={n for n in byname if (state/phase/n/'cleaned.json').exists()}
    failed={n for n in byname if (state/phase/n/'failed.json').exists()}
    return snap,done,failed


def ensure_workers(state,stage,phase,shards,snap,pending):
    for shard in shards:
        if shard not in pending:continue
        active=[j for j in snap['jobs'] if j.get('Args')==f'{stage} {phase} {shard}' and j['JobStatus'] not in [3,4]]
        if len(active)>1:raise RuntimeError('duplicate submitted shard')
        if active:
            j=active[0]
            if j['JobStatus']==5:
                raise RuntimeError('worker held; inspect '+str(j))
            continue
        previous=[p for p in (state/'jobs').glob(f'{phase}-{shard}-*.json') if not p.name.endswith(('.output.json','.policy.json'))]
        if len(previous)>=8:raise RuntimeError('worker restart budget exceeded')
        submit(state,stage,phase,shard,len(previous))


def summarize(state, config):
    receipts=[json.loads(p.read_text()) for p in (state/'receipts').glob('*.json')]
    bins=Counter();pools=Counter();bounds=Counter()
    for r in receipts:
        bins.update(r['validation']['bins']);pools.update(r['validation']['poolused']);bounds.update(r['validation']['boundaries'])
    failures=[json.loads(p.read_text()) for p in (state/'production').glob('*/failed.json')]
    inherited=[r for r in receipts if r['inherited']];new=[r for r in receipts if not r['inherited']]
    stages=[json.loads(s) for s in (state/'staging_history.jsonl').read_text().splitlines()] if (state/'staging_history.jsonl').exists() else []
    result=dict(status='complete' if len(receipts)==681 else 'incomplete',total_files=681,verified_files=len(receipts),
        verified_events=sum(r['entries'] for r in receipts),expected_events=69279369,
        inherited=dict(files=len(inherited),events=sum(r['entries'] for r in inherited),bytes=sum(r['bytes'] for r in inherited)),
        generated=dict(files=len(new),events=sum(r['entries'] for r in new),bytes=sum(r['bytes'] for r in new)),
        actual_mixed_bytes=sum(r['bytes'] for r in receipts),bins=dict(bins),poolused=dict(pools),boundaries=dict(bounds),
        segments=sum(r['validation']['segments'] for r in receipts),
        surplus=sum(r['validation'].get('surplus',0) for r in receipts),accepted_duplicates=0,accepted_missing=0,
        failures=failures,peak_observed_staging_bytes=max([s['bytes'] for s in stages],default=0),
        contract=CONTRACT,git_commit=config['git_commit'],started_at=config['started_at'],updated_at=time.time(),
        validation_limit='No ETO original-charge recomputation or 30-common-branch comparison for scalar outputs; pilot exact comparison only certifies its 3 files')
    atomic_json(state/'summary.json',result)
    atomic_json(state/'failures.json',failures)
    return result


def run(state):
    require_slurm();state.mkdir(parents=True,exist_ok=True)
    lock=(state/'receiver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Also block simultaneous independent scalar runs publishing the same universe.
    global_lock=(BASE/'scalar_receiver.lock').open('a');fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (state/'adoption.json').exists():prepare(state)
    else:freeze(state)
    files=[p for p in HERE.iterdir() if p.suffix in ['.py','.C','.sh','.sbatch']]
    code={p.name:sha256(p) for p in files}
    stage='/scratchfs2/lhaaso/liushijie/WCDA_nhit_'+state.name
    if (state/'config.json').exists():
        config=json.loads((state/'config.json').read_text())
        if config['code']!=code or config['contract']!=CONTRACT:raise ValueError('frozen run code changed')
    else:
        config=dict(run_id=state.name,stage=stage,code=code,contract=CONTRACT,started_at=time.time(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            stage_budget_bytes=2*1024**3,header_concurrency=4,max_workers=4,excluded_machines=EXCLUDED)
        atomic_json(state/'config.json',config)
    q=quota();atomic_json(state/'initial_quota.json',q)
    if min(q['soft'],q['hard'])-q['used']<config['stage_budget_bytes']+512*1024**2:
        raise RuntimeError('insufficient fresh personal quota for configured budget')
    # Stale legacy receiver cannot restart: paused guard + no source jobs. Record this proof.
    raw=remote_py('''import json,subprocess
raw=subprocess.check_output(['condor_q','-name','scheduler@schedd11.ihep.ac.cn','-constraint','Owner == "liushijie"','-json','-attributes','Cmd,Args,JobStatus,Owner,ClusterId'],text=True)
a=[j for j in json.loads(raw.strip() or '[]') if 'WCDA_nhit_20261015_january_v5/' in j.get('Cmd','') and j['JobStatus'] not in [3,4]]
assert not a,a
print(json.dumps(dict(legacy_active_jobs=a)))
''')
    atomic_json(state/'audit'/'legacy_inactive.json',json.loads(raw))
    remote('mkdir -p '+shlex.quote(stage)+'/sessions')
    snapshot_dir=state/'code';snapshot_dir.mkdir(exist_ok=True)
    for p in files:
        dest=snapshot_dir/p.name
        if not dest.exists():shutil.copyfile(p,dest)
        if sha256(dest)!=code[p.name]:raise ValueError('local code snapshot changed')
        transfer(str(dest),REMOTE+':'+stage+'/'+p.name)
    for name in ['tasks.json','config.json']:
        transfer(str(state/name),REMOTE+':'+stage+'/'+name)
    remote('chmod 755 '+shlex.quote(stage+'/worker.sh'))
    tasks=json.loads((state/'tasks.json').read_text())
    if not (state/'gate.json').exists():
        while True:
            snap,done,failed=poll(state,stage,config,'pilot',tasks['pilot'])
            if failed:raise RuntimeError('pilot data acceptance failed: '+repr(failed))
            if len(done)==3:break
            ensure_workers(state,stage,'pilot',[0],snap,{0})
            time.sleep(15)
        # Wait for first job exit; then actually restart it to check all three checkpoints.
        while not any(s['phase']=='pilot' and len(s['processed'])==3 for s in snap['sessions']):
            time.sleep(10);snap=snapshot(stage)
        if not (state/'resume_probe.json').exists():
            active=[j for j in snap['jobs'] if j['JobStatus'] not in [3,4]]
            while active:
                time.sleep(10);snap=snapshot(stage);active=[j for j in snap['jobs'] if j['JobStatus'] not in [3,4]]
            submit(state,stage,'pilot',0,1)
            while True:
                snap=snapshot(stage)
                resumed=[s for s in snap['sessions'] if s['phase']=='pilot' and len(s['skipped'])==3 and not s['processed']]
                if resumed:
                    atomic_json(state/'resume_probe.json',resumed[-1]);break
                time.sleep(15)
        rows=[]
        for n in PILOT:
            r=json.loads((state/'pilot'/n/'receipt.json').read_text())
            rows.append(dict(name=n,baseline_source_seconds=BASELINE[n],source_seconds=r['source']['elapsed_seconds'],
                            startup_seconds=r['source']['session']['startup_seconds'],transfer_seconds=r['transfer_seconds'],
                            eto_seconds=r['eto_seconds'],source_phases=r['source']['root'] |
                            {k:r['source'][k] for k in ['header_seconds','write_seconds','readback_seconds']},comparison=r['comparison']))
        speedup=sum(BASELINE.values())/sum(r['source_seconds'] for r in rows)
        if speedup<=1:raise RuntimeError('no aggregate speedup against previous scalar baseline; inspect/tune before expansion')
        freeze(state)
        atomic_json(state/'gate.json',dict(passed=True,source_speedup=speedup,results=rows,
                                         baseline='previous PINC scalar implementation; measured 213.4,216.3,102.2 s',at=time.time()))
        print('GATE_PASSED',speedup,flush=True)
    while True:
        snap,done,failed=poll(state,stage,config,'production',tasks['production'])
        result=summarize(state,config)
        atomic_json(state/'progress.json',dict(accepted=result['verified_files'],failed=len(failed),remaining=len(tasks['production'])-len(done|failed),at=time.time()))
        if len(done|failed)==len(tasks['production']):break
        pending={i for i in range(4) if any(t['name'] not in done|failed for t in tasks['production'][i::4])}
        ensure_workers(state,stage,'production',range(4),snap,pending)
        time.sleep(20)
    freeze(state)
    for p in (state/'receipts').glob('*.json'):
        r=json.loads(p.read_text())
        if sha256(r['path'])!=r['sha256']:raise ValueError('final published checksum changed')
    transfer(REMOTE+':'+stage+'/sessions/',str(state/'source_sessions')+'/')
    transfer(REMOTE+':'+stage+'/quota_history.jsonl',str(state/'source_quota_history.jsonl'))
    result=summarize(state,config)
    atomic_json(state/'finished.json',dict(result=result,finished_at=time.time(),all_tasks_terminal=True))
    if result['verified_files']==681 and result['verified_events']==69279369:
        atomic_json(state/'completed.json',dict(files=681,events=69279369,at=time.time(),summary_sha256=sha256(state/'summary.json')))
    print('FINISHED',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id');a=p.parse_args()
    state=BASE/'manifests'/a.run_id
    try:run(state)
    except Exception as e:
        atomic_json(state/'controller_error.json',dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()))
        raise
