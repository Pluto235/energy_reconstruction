"""ETO Slurm timer: queue one check in the existing chat every 30 minutes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def save(path, value):
    partial=path.with_suffix('.partial')
    with partial.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(partial,path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def run(state, check_only=False):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this timer in ETO Slurm')
    work=state/'audit'/'heartbeat';work.mkdir(parents=True,exist_ok=True)
    lock=(work/'timer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(work/'config.json')
    assert cfg['interval_seconds']==1800
    if check_only:
        p=subprocess.run([cfg['codex'],'app-server','daemon','version'],text=True,capture_output=True,check=True,timeout=30)
        daemon=json.loads(p.stdout)
        if daemon['status']!='running':raise RuntimeError('ETO Codex daemon is not running')
        save(work/'slurm_preflight.json',dict(job=os.environ['SLURM_JOB_ID'],daemon=daemon,
             progress=read(state/'progress.json'),at=time.time()))
        print('SLURM_TIMER_PREFLIGHT_OK',flush=True);return
    save(work/'running.json',dict(job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),at=time.time()))
    next_at=cfg['first_check_at']
    while True:
        if (work/'stop.json').exists():
            save(work/'stopped.json',dict(at=time.time(),reason=read(work/'stop.json')));return
        delay=next_at-time.time()
        if delay>0:time.sleep(delay)
        if (work/'stop.json').exists():continue
        pending=read(work/'pending.json');handled=read(work/'handled.json')
        if pending and (not handled or handled.get('tick_id')!=pending['tick_id']):
            print('PRIOR_CHECK_PENDING',pending['tick_id'],flush=True)
            next_at=time.time()+cfg['interval_seconds'];continue
        tick=dict(tick_id=uuid.uuid4().hex,at=time.time(),progress=read(state/'progress.json'))
        text=(work/'prompt.txt').read_text()+'\n本次 tick_id: '+tick['tick_id']+'\n当前快照: '+json.dumps(tick['progress'],ensure_ascii=False)
        # Persist before the request: an ambiguous timeout must never blindly duplicate.
        save(work/'pending.json',tick)
        try:
            p=subprocess.run([cfg['codex'],'queue','--thread',cfg['thread_id'],'--message',text],
                             text=True,capture_output=True,timeout=60)
            tick.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
            if p.returncode:raise RuntimeError(p.stderr)
            tick['queued']=True
        except Exception as e:
            tick['queue_error']=repr(e)
        save(work/(tick['tick_id']+'.json'),tick)
        print('CHECK_QUEUED' if tick.get('queued') else 'QUEUE_NEEDS_INSPECTION',tick['tick_id'],flush=True)
        next_at=time.time()+cfg['interval_seconds']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();run(a.state,a.check_only)
