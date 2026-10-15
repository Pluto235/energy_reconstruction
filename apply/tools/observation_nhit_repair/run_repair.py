"""Isolated, resumable four-file recovery after the original run is terminal."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import traceback

from common import atomic_json, require_slurm, sha256, sync_dir
from control import snapshot, poll, ensure_workers
from prepare import BASE, freeze
from policy import AUTH_SHA, SCOPES, RECOVERED, load_policy
from source import CONTRACT
from transport import quota, remote, transfer, REMOTE

HERE = Path(__file__).resolve().parent


def run(parent, state):
    require_slurm()
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/'receiver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock = (BASE/'scalar_receiver.lock').open('a'); fcntl.flock(global_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    original = json.loads((parent/'config.json').read_text())
    if not json.loads((parent/'finished.json').read_text())['all_tasks_terminal']:
        raise RuntimeError('original production is not terminal')
    old = snapshot(original['stage'])
    atomic_json(state/'original_scheduler.json', old)
    if any(j['JobStatus'] not in [3,4] for j in old['jobs']):
        raise RuntimeError('original source or investigation still active')
    freeze(parent)
    names = set(SCOPES)|RECOVERED
    manifest = json.loads((parent/'manifest.json').read_text())
    tasks = [r for r in manifest['records'] if r['name'] in names]
    if len(tasks) != 4: raise ValueError('repair scope differs')
    archive = parent/'audit'/'initial_terminal'
    if not archive.exists():
        archive.mkdir()
        for name in ['finished.json','summary.json','final_audit.json','file_catalog.csv',
                     'file_timings.csv','operation_history.json','failed_files.csv','failures.json','progress.json']:
            shutil.copyfile(parent/name, archive/name)
    for task in tasks:
        p = parent/'receipts'/(task['name']+'.json')
        if p.exists() and not (state/'receipts'/p.name).exists():
            raise ValueError('already confirmed file is outside repair ownership: '+task['name'])
    files = sorted(p for p in HERE.iterdir() if p.suffix in ['.py','.C','.sh'])
    code = {p.name:sha256(p) for p in files}
    q = quota(); atomic_json(state/'initial_quota.json', q)
    stage = '/scratchfs2/lhaaso/liushijie/WCDA_nhit_'+state.name
    if not (state/'config.json').exists():
        auth = parent/'authorizations/observation_nhit_poolused_overrides_20261015.json'
        if sha256(auth) != AUTH_SHA: raise ValueError('authorization changed')
        evidence = parent/'audit/header_fullmonth/source/result.json'
        expected = json.loads((parent/'audit/header_fullmonth/completed.json').read_text())['result_sha256']
        if sha256(evidence) != expected: raise ValueError('header evidence changed')
        shutil.copyfile(auth, state/'authorization.json')
        shutil.copyfile(evidence, state/'header_evidence.json')
        atomic_json(state/'tasks.json', dict(pilot=[], production=tasks))
        config = dict(run_id=state.name,stage=stage,code=code,contract=CONTRACT,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            parent_state=str(parent),parent_manifest_sha256=sha256(parent/'manifest.json'),
            tasks_sha256=sha256(state/'tasks.json'),authorization_sha256=AUTH_SHA,
            header_evidence_sha256=expected,started_at=time.time(),max_workers=4,header_concurrency=4,
            retained_original_stage_bytes=old['stage_bytes'],stage_budget_bytes=2*1024**3-old['stage_bytes'])
        atomic_json(state/'config.json', config)
    config = json.loads((state/'config.json').read_text())
    if config['code'] != code or config['contract'] != CONTRACT:
        raise ValueError('repair execution code changed')
    if config['tasks_sha256'] != sha256(state/'tasks.json') or config['parent_manifest_sha256'] != sha256(parent/'manifest.json'):
        raise ValueError('repair input bindings changed')
    load_policy(state,config)
    if min(q['soft'],q['hard'])-q['used'] < 2*1024**3+512*1024**2:
        raise RuntimeError('insufficient personal quota for repair')
    # This is also the fresh connection check from the actual Slurm environment.
    atomic_json(state/'ssh_selfcheck.json',dict(output=remote('hostname; id -un'),at=time.time()))
    code_dir = state/'code'; code_dir.mkdir(exist_ok=True)
    for p in files:
        dest=code_dir/p.name
        if not dest.exists(): shutil.copyfile(p,dest)
        if sha256(dest)!=code[p.name]: raise ValueError('repair snapshot changed')
    result = subprocess.run([os.sys.executable,'-B','-m','unittest','selftest','test_policy'],cwd=code_dir,text=True,capture_output=True)
    atomic_json(state/'selftests.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,at=time.time()))
    if result.returncode: raise RuntimeError(result.stderr)
    # Deploy once; never overwrite an active source package on resumption.
    if not (state/'deployed.json').exists():
        remote('mkdir -p '+shlex.quote(stage)+'/sessions')
        for p in files: transfer(str(code_dir/p.name), REMOTE+':'+stage+'/'+p.name)
        for name in ['config.json','tasks.json','authorization.json','header_evidence.json']:
            transfer(str(state/name),REMOTE+':'+stage+'/'+name)
        remote('chmod 755 '+shlex.quote(stage+'/worker.sh'))
        atomic_json(state/'deployed.json',dict(at=time.time(),stage=stage,code=code))
    while True:
        snap,done,failed = poll(state,stage,config,'production',tasks)
        atomic_json(state/'progress.json',dict(accepted=len(done),failed=len(failed),remaining=4-len(done)-len(failed),at=time.time()))
        if len(done|failed)==4 and not any(j['JobStatus'] not in [3,4] for j in snap['jobs']): break
        pending={i for i,t in enumerate(tasks) if t['name'] not in done|failed}
        ensure_workers(state,stage,'production',range(4),snap,pending)
        time.sleep(20)
    freeze(parent)
    # Only receipt-confirmed results enter the primary catalogue. Old failures stay intact.
    for task in tasks:
        name=task['name']; p=state/'receipts'/(name+'.json')
        if not p.exists(): continue
        rec=json.loads(p.read_text())
        if sha256(rec['path'])!=rec['sha256']: raise ValueError('repair output changed')
        dst=parent/'receipts'/p.name
        if dst.exists():
            if sha256(dst)!=sha256(p): raise ValueError('existing primary receipt conflict')
        else: os.link(p,dst); sync_dir(dst.parent)
        failure=parent/'production'/name/'failed.json'
        atomic_json(parent/'resolutions'/(name+'.json'),dict(name=name,status='resolved',
            original_failure=str(failure),original_failure_sha256=sha256(failure),repair_state=str(state),
            receipt=str(dst),receipt_sha256=sha256(dst),authorization_sha256=AUTH_SHA if name in SCOPES else None,
            basis='user_confirmed_override_for_one_segment' if name in SCOPES else 'independently_recovered_exact_original_header',at=time.time()))
    transfer(REMOTE+':'+stage+'/sessions/',str(state/'source_sessions')+'/')
    transfer(REMOTE+':'+stage+'/quota_history.jsonl',str(state/'source_quota_history.jsonl'))
    from transport import command,SSH
    command(['rsync','-rt','--partial','--timeout=180','--exclude=*.root','-e',shlex.join(SSH),REMOTE+':'+stage+'/',str(state/'source_archive')+'/'])
    atomic_json(state/'final_quota.json',quota())
    atomic_json(state/'finished.json',dict(accepted=sorted(done),failed=sorted(failed),finished_at=time.time(),all_tasks_terminal=True))
    atomic_json(parent/'repair.json',dict(state=str(state),config_sha256=sha256(state/'config.json'),finished_sha256=sha256(state/'finished.json')))
    if set(SCOPES)<=done:
        atomic_json(parent/'authorizations/poolused_override_status.json',dict(status='accepted_with_user_confirmed_overrides',
            authorization_file=str(state/'authorization.json'),sha256=AUTH_SHA,repair_state=str(state),scope=sorted(SCOPES),at=time.time()))
    atomic_json(parent/'progress.json',dict(accepted=len(list((parent/'receipts').glob('*.json'))),failed=len(failed),remaining=0,at=time.time(),repair_state=str(state)))
    print('REPAIR_TERMINAL',len(done),len(failed),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('state',type=Path);a=p.parse_args()
    try:run(a.parent,a.state)
    except Exception as e:
        atomic_json(a.state/'controller_error.json',dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()));raise
