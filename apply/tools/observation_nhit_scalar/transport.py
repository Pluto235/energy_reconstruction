"""ETO-only control transport; never depends on an agent or persistent socket."""
import json
import re
import shlex
import subprocess
import time

REMOTE = 'ihep-work'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'RemoteCommand=none', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
       '-o', 'IdentityAgent=none', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=2']


def command(args, timeout=300, attempts=3):
    history = []
    for attempt in range(attempts):
        try:
            p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
            if p.returncode:
                raise RuntimeError(p.stderr[-3000:] + p.stdout[-1000:])
            return p.stdout
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            history.append(dict(attempt=attempt+1, error=str(e), at=time.time()))
            print('TRANSPORT_RETRY', json.dumps(history[-1]), flush=True)
            if attempt+1 == attempts:
                raise RuntimeError(json.dumps(history)) from e
            time.sleep(3*(attempt+1))


def remote(code, attempts=3):
    return command(SSH+[REMOTE, code], attempts=attempts)


def remote_py(code, attempts=3):
    return remote('python3 -c '+shlex.quote(code), attempts)


def transfer(src, dst):
    return command(['rsync', '-rt', '--partial', '--timeout=180', '-e', shlex.join(SSH), src, dst])


def parse_quota(raw):
    m = re.search(r'^\s*/scratchfs2\s+(\d+)\*?\s+(\d+)\s+(\d+)\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)', raw, re.M)
    if not m:
        raise RuntimeError('Cannot parse personal quota: '+raw)
    used, soft, hard, files, fsoft, fhard = map(int, m.groups())
    return dict(used=used*1024, soft=soft*1024, hard=hard*1024, files=files,
                files_soft=fsoft, files_hard=fhard, checked_at=time.time(), raw=raw)


def quota():
    return parse_quota(remote('lfs quota -u liushijie /scratchfs2'))
