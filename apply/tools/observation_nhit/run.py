"""Resumable bounded IHEP HTCondor -> ETO Slurm January friend receiver."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter

import numpy as np
import uproot

from common import CONTRACT, KEYS, atomic_json, require_slurm, sha256, stamp, sync_dir, unique_keys
from verify import verify

BASE = Path('/mnt/newdisk/shijie/WCDA_obs_friends/nhit')
INPUT = Path('/mnt/newdisk/shijie/WCDA_obs')
REPO = Path('/mnt/newdisk/shijie/energy')
OLD = REPO / '.recovery/2026-10-13/obs-january'
REMOTE = 'liushijie@lxlogin.ihep.ac.cn'
SSH = ['ssh', '-S', str(OLD/'ihep.sock'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30']
PILOT = [f'Esg202201{d}_00.root' for d in ['01', '16', '31']]
GIB = 1024**3
EXCLUDED_MACHINE = 'lhws183.ihep.ac.cn'


def command(args, timeout=1200):
    for attempt in range(3):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if attempt == 2:
                raise RuntimeError(str(e) + ' ' + str(getattr(e, 'stderr', ''))[-2000:]) from e
            time.sleep(5 * (attempt+1))


def remote(code):
    return command(SSH + [REMOTE, code])


def remote_py(code):
    return remote('python3 -c ' + shlex.quote(code))


def transfer(src, dst):
    command(['rsync', '-rt', '--partial', '--timeout=300', '-e', shlex.join(SSH), src, dst])


def quota():
    raw = remote('lfs quota -u liushijie /scratchfs2')
    m = re.search(r'^\s*/scratchfs2\s+(\d+)\*?\s+(\d+)\s+(\d+)\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)', raw, re.M)
    if not m:
        raise RuntimeError('Cannot parse current personal quota: ' + raw)
    used, soft, hard, files, fsoft, fhard = map(int, m.groups())
    if files + 3000 >= min(fsoft, fhard):
        raise RuntimeError('Insufficient inode quota for this run')
    return dict(used=used*1024, soft=soft*1024, hard=hard*1024, files=files, checked_at=time.time(), raw=raw)


def freeze(state):
    names = sorted(p.name for p in (OLD/'receipts').glob('Esg202201*.root.json'))
    names = [n[:-5] for n in names]
    actual = sorted(p.name for p in INPUT.glob('01??/Esg202201*.root'))
    if names != actual or len(names) != 681 or len({n[7:11] for n in names}) != 31:
        raise ValueError('restoration receipt/current 681-file manifest mismatch')
    records = []
    for name in names:
        path = INPUT/name[7:11]/name
        s = stamp(path)
        r = json.loads((OLD/'receipts'/(name+'.json')).read_text())
        if s['bytes'] != r['bytes'] or s['entries'] != r['entries']:
            raise ValueError('restoration count/size mismatch: ' + name)
        records.append(dict(name=name, input=s, source_entries=r['record']['source']['entries']))
    manifest = dict(contract=CONTRACT, records=records)
    p = state/'manifest.json'
    if p.exists():
        if json.loads(p.read_text()) != manifest:
            raise ValueError('frozen manifest changed; expansion forbidden')
    else:
        atomic_json(p, manifest)
    return records


def archive_samples():
    src = REPO/'apply/output/v7_nhit_audit_20261015_ihep'
    mapping = []
    for path in sorted(src.rglob('*')):
        if not path.is_file() or path.suffix not in {'.root', '.json'}:
            continue
        day = path.name[7:11] if path.name.startswith('Esg') else 'audit'
        out = BASE/'sample512'/day/path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copyfile(path, out)
        digest = sha256(path)
        if sha256(out) != digest:
            raise ValueError('sample archival checksum mismatch')
        mapping.append(dict(source=str(path), destination=str(out), sha256=digest))
    atomic_json(BASE/'sample512'/'catalog.json', dict(scope='512 events per file; NOT full files', files=mapping))


def prepare(state, row, code, stage):
    name = row['name']; path = INPUT/name[7:11]/name
    if stamp(path) != row['input']:
        raise ValueError('input changed: ' + name)
    with uproot.open(path) as f:
        a = f['t_eventout'].arrays(KEYS, library='np')
    unique_keys(a)
    request = state/(name+'.requests.npz')
    np.savez_compressed(request, **a)
    reservation = row['input']['bytes'] + request.stat().st_size + 64*1024**2
    task = dict(row, contract=CONTRACT, code=code,
                source='/eos/lhaaso/ai/wcda/raw/2022/'+name[7:11]+'/'+name,
                request_sha256=sha256(request), file_limit=reservation-request.stat().st_size-32*1024**2,
                reservation=reservation)
    atomic_json(state/(name+'.task.json'), task)
    return task


def submit(state, row, task, stage):
    name = row['name']
    for suffix in ['.requests.npz', '.task.json']:
        transfer(str(state/(name+suffix)), REMOTE+':'+stage+'/'+name+suffix)
    # Submission must not be retried blindly: an interrupted reply can hide a live job.
    cmd = ('source /afs/ihep.ac.cn/soft/common/sysgroup/hep_job/setup_hepjob.sh; cd '+shlex.quote(stage)+
           '; hep_sub ./worker.sh -argu '+shlex.quote(stage)+' '+shlex.quote(name)+
           ' -g lhaaso -os AlmaLinux9 -m 4000 -cpu 1 -schedd schedd11 -o '+name+'.out -e '+name+'.err')
    atomic_json(state/(name+'.submitted.json'), dict(submitted_at=time.time(), status='submission_pending',
                                                  reservation=task['reservation']))
    p = subprocess.run(SSH+[REMOTE, cmd], text=True, capture_output=True, timeout=120)
    atomic_json(state/(name+'.submission_output.json'), dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
    if p.returncode:
        raise RuntimeError('HTCondor submission failed or ambiguous; inspect submission_output: '+name)
    m = re.search(r'cluster\s+(\d+)', p.stdout, re.I)
    if not m:
        raise RuntimeError('Cannot determine submitted cluster: '+p.stdout)
    cluster = int(m[1])
    adjust = f'''import json,subprocess
base=['-name','scheduler@schedd11.ihep.ac.cn']; job={str(cluster)+'.0'!r}
a=json.loads(subprocess.check_output(['condor_q']+base+[job,'-json','-attributes','Owner,Cmd,Requirements,JobStatus,RemoteHost'],text=True))[0]
assert a['Owner']=='liushijie' and a['Cmd'].startswith({stage!r}+'/')
old=a['Requirements']; assert old.startswith('/Expr(') and old.endswith(')/')
new='('+old[6:-2]+') && (TARGET.Machine != '+json.dumps({EXCLUDED_MACHINE!r})+')'
subprocess.run(['condor_qedit']+base+[job,'Requirements',new],check=True)
if a.get('JobStatus')==2 and {EXCLUDED_MACHINE!r} in a.get('RemoteHost',''):
    subprocess.run(['condor_hold']+base+[job],check=True)
    raise RuntimeError('Excluded node already claimed job; held for inspection')
print(json.dumps(dict(cluster={cluster},excluded_machine={EXCLUDED_MACHINE!r})))
'''
    adjusted = remote_py(adjust)
    atomic_json(state/(name+'.scheduler_policy.json'), dict(output=adjusted, excluded_machine=EXCLUDED_MACHINE))
    atomic_json(state/(name+'.submitted.json'), dict(submitted_at=time.time(), cluster=int(m[1]),
                                                   reservation=task['reservation'], status='submitted'))


def cleanup(stage, name, receipt):
    code = (f'import json,os; from pathlib import Path; p=Path({stage!r}); n={name!r}; '
            f'r=json.loads((p/(n+".ready.json")).read_text()); a=json.loads({json.dumps(receipt)!r}); '
            'assert r["sha256"]==a["sha256"] and r["entries"]==a["entries"]; '
            't=p/(n+".ack.partial"); t.write_text(json.dumps(a)); '
            'f=t.open("rb"); os.fsync(f.fileno()); f.close(); os.replace(t,p/(n+".ack.json")); '
            'fd=os.open(str(p),os.O_DIRECTORY); os.fsync(fd); os.close(fd); '
            '[(p/(n+s)).unlink(missing_ok=True) for s in [".friend.root",".requests.npz"]]')
    remote_py(code)


def receive(state, row, stage):
    name = row['name']; recpath = state/(name+'.ready.json')
    transfer(REMOTE+':'+stage+'/'+name+'.ready.json', str(recpath))
    record = json.loads(recpath.read_text())
    task = json.loads((state/(name+'.task.json')).read_text())
    if (record['name'] != name or record['request_sha256'] != task['request_sha256']
            or record['code'] != task['code']):
        raise ValueError('source receipt does not belong to frozen task: '+name)
    target = BASE/'full'/name[7:11]/name.replace('.root', '.nhit.root')
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = state/(name+'.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if not target.exists() or sha256(target) != receipt['sha256']:
            raise ValueError('published target changed: '+name)
        cleanup(stage, name, receipt)
        return
    incoming = state/(name+'.incoming.root')
    if not target.exists():
        transfer(REMOTE+':'+stage+'/'+name+'.friend.root', str(incoming))
    candidate = target if target.exists() else incoming
    report = verify(INPUT/name[7:11]/name, candidate, record, row['input'])
    with candidate.open('rb') as f:
        os.fsync(f.fileno())
    if not target.exists():
        # Hard-link publication refuses to replace any existing target, including on a restart.
        os.link(incoming, target); sync_dir(target.parent); incoming.unlink()
    receipt = dict(name=name, path=str(target), sha256=record['sha256'], bytes=record['bytes'],
                   entries=record['entries'], validation=report)
    atomic_json(receipt_path, receipt)
    cleanup(stage, name, receipt)
    atomic_json(state/(name+'.cleaned.json'), dict(cleaned_at=time.time()))
    (state/(name+'.requests.npz')).unlink(missing_ok=True)
    print('VERIFIED', name, record['entries'], flush=True)


def summarize(state, done, failures):
    bins = Counter(); pools = Counter(); boundaries = Counter(); events = 0; nsegments = 0
    for name in sorted(done):
        r = json.loads((state/(name+'.receipt.json')).read_text())
        v = r['validation']; events += r['entries']; nsegments += v['segments']
        bins.update(v['bins']); pools.update(v['poolused'])
        boundaries.update({k:n for k,n in v.items() if k.startswith('events_nhit_') or k.startswith('hits_exact_')})
    atomic_json(state/'summary.json', dict(status='complete' if len(done)==681 else 'incomplete',
        verified_files=len(done), failed_files=len(failures), total_files=681, verified_events=events,
        event_coverage='only validated files; pending/failed files are NOT verified',
        nhit_equal_fraction=1 if events else None, bin_mismatch_fraction=0 if events else None,
        delta_histogram={'0':events}, bins=dict(bins), poolused=dict(pools), segments=nsegments,
        boundary_counts=dict(boundaries), boundary_note='zero means not covered in actual data',
        failures=sorted(failures), contract=CONTRACT, updated_at=time.time()))


def run(args):
    require_slurm()
    state = BASE/'manifests'/args.run_id
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/'receiver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage = '/scratchfs2/lhaaso/liushijie/WCDA_nhit_'+args.run_id
    source_code = Path(__file__).resolve().parent
    code = {p.name: sha256(p) for p in source_code.glob('*.py')}
    config = dict(stage=stage, code=code, contract=CONTRACT, max_concurrency=4,
                  excluded_machine=EXCLUDED_MACHINE)
    cfg = state/'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('run code/config changed; preserve frozen run')
    atomic_json(cfg, config)
    remote('mkdir -p '+shlex.quote(stage))
    q = quota(); atomic_json(state/'quota.json', q)
    capacity = min(4*GIB, q['soft']-q['used']-GIB, q['hard']-q['used']-6*GIB)
    if capacity <= 0:
        raise RuntimeError('No safe personal staging quota')
    for name in ['common.py', 'extract.py']:
        transfer(str(source_code/name), REMOTE+':'+stage+'/'+name)
    worker = state/'worker.sh'
    worker.write_text('#!/bin/bash\nset -eo pipefail\nsource /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh\nexec python3 "'+stage+'/extract.py" "$@"\n')
    worker.chmod(0o755); transfer(str(worker), REMOTE+':'+stage+'/worker.sh')
    remote('chmod 755 '+shlex.quote(stage+'/worker.sh'))
    rows = freeze(state); archive_samples()
    byname = {r['name']: r for r in rows}
    ordered = PILOT + [r['name'] for r in rows if r['name'] not in PILOT]
    while True:
        for receipt_path in state.glob('*.receipt.json'):
            name = receipt_path.name[:-13]
            if not (state/(name+'.cleaned.json')).exists():
                receipt = json.loads(receipt_path.read_text())
                if sha256(receipt['path']) != receipt['sha256']:
                    raise ValueError('published target changed before cleanup: '+name)
                cleanup(stage, name, receipt)
                atomic_json(state/(name+'.cleaned.json'), dict(cleaned_at=time.time()))
        failures = {p.name[:-12] for p in state.glob('*.failed.json')}
        done = {p.name[:-13] for p in state.glob('*.receipt.json')}
        active = {p.name[:-15]: json.loads(p.read_text()) for p in state.glob('*.submitted.json')
                  if p.name[:-15] not in done | failures}
        listing = json.loads(remote_py(f'import os,json; print(json.dumps(os.listdir({stage!r})))'))
        for name, sub in list(active.items()):
            if name+'.failed.json' in listing:
                transfer(REMOTE+':'+stage+'/'+name+'.failed.json', str(state/(name+'.failed.json')))
                continue
            if name+'.ready.json' in listing:
                try:
                    receive(state, byname[name], stage)
                except ValueError as e:
                    atomic_json(state/(name+'.failed.json'), dict(name=name, error=str(e), stage='ETO verification'))
                continue
            if time.time()-sub['submitted_at'] > 20*3600:
                raise RuntimeError('HTCondor task has no result after 20h: '+name)
        done = {p.name[:-13] for p in state.glob('*.receipt.json')}
        failures = {p.name[:-12] for p in state.glob('*.failed.json')}
        active = {n:s for n,s in active.items() if n not in done | failures}
        pilot_ok = set(PILOT) <= done
        atomic_json(state/'progress.json', dict(verified=len(done), failed=len(failures), active=list(active),
                                               pilot_verified=pilot_ok, total=681, updated_at=time.time()))
        atomic_json(state/'failures.json', [json.loads((state/(n+'.failed.json')).read_text()) for n in sorted(failures)])
        summarize(state, done, failures)
        if failures & set(PILOT):
            raise RuntimeError('Full-file pilot failed; automatic monthly expansion disabled')
        if len(done | failures) == 681:
            if not failures:
                freeze(state)
                atomic_json(state/'completed.json', dict(files=681, days=31, completed_at=time.time(), contract=CONTRACT))
            return
        limit = 4 if pilot_ok else 1
        pending = [n for n in ordered if n not in done | failures | set(active) and (pilot_ok or n in PILOT)]
        if len(active) < limit and pending:
            q = quota(); atomic_json(state/'quota.json', q)
            reserved = sum(s['reservation'] for s in active.values())
            # Failed artifacts remain reserved until reviewed; never silently delete evidence.
            reserved += sum(json.loads((state/(n+'.task.json')).read_text())['reservation']
                            for n in failures if (state/(n+'.task.json')).exists())
            available = min(capacity-reserved, q['soft']-q['used']-GIB, q['hard']-q['used']-6*GIB)
            name = pending[0]; row = byname[name]
            estimate = row['input']['bytes'] + row['input']['entries']*16 + 64*1024**2
            if estimate <= available:
                try:
                    task = prepare(state, row, code, stage)
                except ValueError as e:
                    atomic_json(state/(name+'.failed.json'), dict(name=name, error=str(e), stage='ETO preparation'))
                    continue
                submit(state, row, task, stage)
                print('SUBMITTED', name, flush=True)
                continue
        time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-id', required=True)
    args = p.parse_args()
    if not re.fullmatch(r'20261015_[a-z0-9_]+', args.run_id):
        p.error('run-id must be isolated under 20261015_<name>')
    run(args)
