#!/usr/bin/env python3
"""Independent, receipt-bound WCDA time recovery; EOS is never treated as POSIX."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import filecmp
import json
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import tempfile
import threading
import time
import uuid

from observation_state import atomic_json, record_id, file_path

IDENTITY = ('name', 'request_id', 'record_id', 'config_id', 'git_commit', 'entries', 'bytes')
TIME_TYPES = dict(tai_sec='int64', tai_subsec='float64', unix_sec='int64', unix_subsec='float64',
                  mjd='float64', mjdns4='int32', ra_mean='float64', dec_mean='float64',
                  ra_mean_deg='float64', dec_mean_deg='float64', match_status='int32')
ALIGN_TYPES = dict(entry_index='int64', irun='int32', ies='int32', iseq='int32',
                   ievent='int32', theta='float64', phi='float64')


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_request(config, request):
    name = request['name']
    if name not in config['allowed_names']:
        raise ValueError('Request outside the original manifests')
    file_path(Path('/tmp/wcda_time_name_validation'), name)
    if not re.fullmatch('[a-f0-9]{32}', request['request_id']):
        raise ValueError('Invalid request identity')
    if (request['config_id'] != config['config_id'] or request['git_commit'] != config['git_commit']
            or not re.fullmatch(r'(v2:[a-f0-9]{32}|legacy-sha256:[a-f0-9]{64})', request['record_id'])
            or type(request['entries']) is not int or request['entries'] < 0
            or type(request['bytes']) is not int or request['bytes'] <= 0):
        raise ValueError('Invalid or stale request')
    return {k: request[k] for k in IDENTITY}


def control_path(config, kind, name):
    if name not in config['allowed_names']:
        raise ValueError('Unknown observation name')
    path = Path(config['control']) / kind / (name + '.json')
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Unexpected control symlink')
    return path


def ensure_control_budget(config):
    used = sum(p.stat().st_blocks * 512 for p in Path(config['control']).rglob('*') if p.is_file())
    if used > 63 * 1024**2:
        raise RuntimeError('Control directory reached 63 MiB; reserve final MiB for confirmations')


def put_request(config, request):
    identity = validate_request(config, request)
    path = control_path(config, 'requests', request['name'])
    with task_lock(config,request['name']) as acquired:
        if not acquired:
            raise BlockingIOError('Task is busy; request preserved and retry required')
        if path.exists():
            existing = read_json(path)
            if validate_request(config, existing) != identity:
                raise ValueError('Input/code identity changed; previous request preserved')
            return existing
        ensure_control_budget(config)
        atomic_json(path, request)
    return request


def product_paths(config, request):
    validate_request(config, request)
    root = config['eos_root'].rstrip('/')
    if not re.fullmatch(r'/eos/user/l/liushijie/[A-Za-z0-9_/-]+', root):
        raise ValueError('Unexpected task EOS root')
    base = root + '/' + request['name'][7:11] + '/' + request['name'][:-5] + '.' + request['request_id']
    return dict(root=base + '.time.root', summary=base + '.summary.json', exceptions=base + '.exceptions.jsonl')


def eos_command(config, *args):
    result = subprocess.run(['xrdfs', config['eos_server'], *args], text=True, capture_output=True, timeout=60)
    if result.returncode:
        if '[3011]' in result.stderr or 'No such file' in result.stderr:
            raise FileNotFoundError(result.stderr[-2048:])
        raise OSError(result.stderr[-2048:])
    return result.stdout


def eos_listing(config):
    text = eos_command(config, 'ls', '-l', '-R', config['eos_root'])
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('d'):
            continue
        if len(fields) != 7 or not fields[0].startswith('-') or not fields[3].isdigit():
            raise ValueError('Unrecognized xrdfs listing; refusing to undercount queue')
        result[fields[-1]] = int(fields[3])
    return result


@contextmanager
def queue_lock(config):
    with (Path(config['control']) / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@contextmanager
def task_lock(config, name):
    path = Path(config['control']) / 'file_locks' / (name + '.lock')
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def ledger_path(config):
    return Path(config['control']) / 'queue_ledger.json'


def _usage(config, ledger):
    by_request, other = {}, 0
    for path, size in eos_listing(config).items():
        match = re.search(r'\.([a-f0-9]{32})\.', Path(path).name)
        if match and match[1] in ledger:
            by_request[match[1]] = by_request.get(match[1], 0) + size
        else:
            other += size
    return other + sum(max(value['bytes'], by_request.get(key, 0)) for key, value in ledger.items())


def queue_usage(config):
    with queue_lock(config):
        ledger = read_json(ledger_path(config)) if ledger_path(config).exists() else {}
        return _usage(config, ledger)


def reserve(config, request, amount):
    validate_request(config, request)
    if amount <= 0 or amount > config['queue_capacity']:
        return False
    with queue_lock(config):
        ledger = read_json(ledger_path(config)) if ledger_path(config).exists() else {}
        ledger.pop(request['request_id'], None)
        projected = dict(ledger)
        projected[request['request_id']] = dict(bytes=amount, name=request['name'], at=time.time())
        if _usage(config, projected) > config['queue_capacity']:
            return False
        atomic_json(ledger_path(config), projected)
        return True


def release(config, request):
    with queue_lock(config):
        ledger = read_json(ledger_path(config)) if ledger_path(config).exists() else {}
        ledger.pop(request['request_id'], None)
        atomic_json(ledger_path(config), ledger)


def eos_remove(config, request):
    for path in product_paths(config, request).values():
        for candidate in (path, path + '.partial'):
            try:
                eos_command(config, 'rm', candidate)
            except FileNotFoundError:
                pass


def acknowledge(config, receipt):
    identity = validate_request(config, receipt)
    name = receipt['name']
    request = read_json(control_path(config, 'requests', name))
    if validate_request(config, request) != identity:
        raise ValueError('Cleanup requires the exact durable request/receipt')
    ready = read_json(control_path(config, 'ready', name))
    if validate_request(config, ready) != identity:
        raise ValueError('Cleanup requires the exact durable request/receipt')
    with task_lock(config, name) as acquired:
        if not acquired:
            return dict(cleaned=False, busy=True)
        atomic_json(control_path(config, 'acks', name), receipt)
        eos_remove(config, request)
        release(config, request)
        atomic_json(control_path(config, 'states', name), dict(request, state='cleaned', at=time.time()))
    return dict(cleaned=True)


def ssh_args(config):
    return ['ssh', '-S', config['ssh_socket'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']


def remote(config, action, payload=None):
    encoded = base64.b64encode(json.dumps(payload or {}).encode()).decode()
    command = shlex.join(['python3', config['control'] + '/code/sync_observation_time.py',
                          '--config', config['remote_config'], 'remote', action, encoded])
    raw = subprocess.check_output(ssh_args(config) + [config['ssh_host'], command], text=True, timeout=180)
    return json.loads(raw)


def transfer(config, source, target):
    """XRootD streams over the existing SSH channel; no ETO XRootD install required."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.parent.is_symlink():
        raise ValueError('Unexpected incoming symlink')
    command = shlex.join(['xrdcp', '--nopbar', '--silent', config['eos_server'] + '/' + source, '-'])
    with target.open('wb') as out:
        result = subprocess.run(ssh_args(config) + [config['ssh_host'], command], stdout=out,
                                stderr=subprocess.PIPE, timeout=1200)
        if result.returncode:
            raise OSError(result.stderr.decode(errors='replace')[-2048:])
        out.flush()
        os.fsync(out.fileno())


def verify_file(config, request, root_path, summary_path, exceptions_path):
    import numpy as np
    import uproot
    validate_request(config, request)
    observation = file_path(Path(config['observation_root']), request['name'])
    receipt = read_json(Path(config['receipt_root']) / (request['name'] + '.json'))
    if receipt['record_id'] != request['record_id'] or receipt['entries'] != request['entries'] or receipt['bytes'] != request['bytes']:
        raise ValueError('Observation receipt identity changed')
    before = observation.stat()
    if before.st_size != request['bytes'] or observation.is_symlink():
        raise ValueError('Observation path or size changed')
    summary = read_json(summary_path)
    counters = np.zeros(7, dtype=np.int64)
    audit_failures, audit_fallback = 0, 0
    with uproot.open(root_path) as f, uproot.open(observation) as obs, Path(exceptions_path).open() as errors:
        if f.file.fEND > Path(root_path).stat().st_size:
            raise ValueError('Truncated recovered ROOT')
        if (f['recovery_request_id'].member('fTitle') != request['request_id']
                or f['recovery_code_commit'].member('fTitle') != request['git_commit']):
            raise ValueError('Recovered file belongs to another request/code snapshot')
        t, a, main = f['t_recovered_time'], f['t_alignment'], obs['t_eventout']
        for tree, schema in ((t, TIME_TYPES), (a, ALIGN_TYPES)):
            if tree.classname != 'TTree' or set(tree.keys(recursive=False)) != set(schema) or tree.num_entries != request['entries']:
                raise ValueError('Recovered ROOT schema/count mismatch')
            for key, dtype in schema.items():
                branch = tree[key]
                if branch.num_entries != request['entries'] or branch.interpretation.numpy_dtype != np.dtype(dtype):
                    raise ValueError('Wrong recovered branch type/count: ' + key)
        if main.num_entries != request['entries']:
            raise ValueError('Observation event count changed')
        audit = next(errors, None)
        audit = json.loads(audit) if audit is not None else None
        previous_audit = -1
        for start in range(0, request['entries'], 65536):
            stop = min(start + 65536, request['entries'])
            alignment = a.arrays(entry_start=start, entry_stop=stop, library='np')
            original = main.arrays(list(ALIGN_TYPES)[1:], entry_start=start, entry_stop=stop, library='np')
            values = t.arrays(entry_start=start, entry_stop=stop, library='np')
            if not np.array_equal(alignment['entry_index'], np.arange(start, stop, dtype=np.int64)):
                raise ValueError('Non-contiguous alignment indices')
            for key in list(ALIGN_TYPES)[1:]:
                expected, actual = original[key], alignment[key]
                if expected.dtype != actual.dtype:
                    raise ValueError('Observation scalar type changed: ' + key)
                if key in ('theta', 'phi'):
                    expected, actual = expected.view(np.uint64), actual.view(np.uint64)
                if not np.array_equal(expected, actual):
                    raise ValueError('Observation sequence mismatch: ' + key)
            status = values['match_status']
            if np.any((status < 0) | (status > 6)):
                raise ValueError('Unknown recovery status')
            counters += np.bincount(status, minlength=7)
            timed = (status == 0) | (status == 6)
            for key in ('tai_subsec', 'unix_subsec', 'mjd'):
                if not np.all(np.isfinite(values[key][timed])):
                    raise ValueError('Nonfinite successful time')
            if (np.any(values['tai_sec'][timed] <= 0) or np.any(values['unix_sec'][timed] <= 0)
                    or np.any(values['mjd'][timed] <= 0)
                    or np.any((values['tai_subsec'][timed] < 0) | (values['tai_subsec'][timed] >= 1))
                    or np.any((values['unix_subsec'][timed] < 0) | (values['unix_subsec'][timed] >= 1))):
                raise ValueError('Invalid successful time fields')
            good = status == 0
            for key in ('ra_mean', 'dec_mean', 'ra_mean_deg', 'dec_mean_deg'):
                if not np.all(np.isfinite(values[key][good])):
                    raise ValueError('Nonfinite successful coordinates')
            if (np.any((values['ra_mean'][good] < 0) | (values['ra_mean'][good] >= 2*np.pi))
                    or np.any(np.abs(values['dec_mean'][good]) > np.pi/2)
                    or not np.array_equal(values['ra_mean_deg'][good], values['ra_mean'][good]*(180/np.pi))
                    or not np.array_equal(values['dec_mean_deg'][good], values['dec_mean'][good]*(180/np.pi))):
                raise ValueError('Invalid coordinate units/range')
            failure_indices = set((np.flatnonzero(status) + start).tolist())
            seen_failures = set()
            while audit is not None and audit['entry_index'] < stop:
                idx = audit['entry_index']
                if type(idx) is not int or idx < start or idx <= previous_audit:
                    raise ValueError('Exception audit order mismatch')
                previous_audit = idx
                j = idx - start
                if audit['status'] != int(status[j]) or any(audit[k] != int(alignment[k][j]) for k in ('irun','ies','iseq','ievent')):
                    raise ValueError('Exception audit refers to another event')
                for key in ('tai_sec','tai_subsec','unix_sec','unix_subsec','mjd'):
                    if audit[key] != values[key][j]:
                        raise ValueError('Exception audit lost available time')
                if audit.get('ies_fallback'):
                    audit_fallback += 1
                if audit['status']:
                    seen_failures.add(idx)
                    audit_failures += 1
                elif audit['reason'] == 'ies_fallback' and audit.get('ies_fallback'):
                    pass
                else:
                    raise ValueError('Unexpected successful event in exception audit')
                line = next(errors, None)
                audit = json.loads(line) if line is not None else None
            if failure_indices != seen_failures:
                raise ValueError('Missing per-event failure audit')
        if audit is not None:
            raise ValueError('Exception index outside target sequence')
    if (summary['entries'] != request['entries'] or summary['status'] != counters.tolist()
            or summary['reused_time'] + summary['new_time'] != int(counters[0] + counters[6])
            or summary['reused_coords'] + summary['new_coords'] != int(counters[0])
            or audit_failures != int(sum(counters[1:])) or audit_fallback != summary['ies_fallback']):
        raise ValueError('Recovery accounting mismatch')
    after = observation.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Observation changed during verification')
    current = read_json(Path(config['receipt_root']) / (request['name'] + '.json'))
    if current['record_id'] != request['record_id']:
        raise ValueError('Receipt changed during verification')
    return summary


def final_paths(config, request):
    root = Path(config['destination']) / request['name'][7:11]
    base = request['name'][:-5]
    return dict(root=root/(base+'.time.root'), summary=root/(base+'.time.summary.json'),
                exceptions=root/(base+'.time.exceptions.jsonl'))


def receive_one(config, ready):
    validate_request(config, ready)
    expected_products = product_paths(config, ready)
    if set(ready['products']) != set(expected_products) or any(ready['products'][k]['path'] != v or type(ready['products'][k]['bytes']) is not int or ready['products'][k]['bytes'] < 0 for k,v in expected_products.items()):
        raise ValueError('Unexpected ready product paths/sizes')
    state = Path(config['state'])
    request = read_json(state/'requests'/(ready['name']+'.json'))
    if validate_request(config, request) != validate_request(config, ready):
        raise ValueError('Ready product belongs to another request')
    receipt_path = state/'receipts'/(ready['name']+'.json')
    final = final_paths(config, request)
    previous = read_json(receipt_path) if receipt_path.exists() else None
    if previous and validate_request(config, previous) != validate_request(config, request):
        raise ValueError('Confirmed output identity changed; preserved')
    if previous:
        if any(p.is_symlink() or not p.is_file() for p in final.values()):
            raise ValueError('Confirmed output is missing or symlinked; preserved')
        actual = [list((p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)) for p in final.values()]
        if previous.get('output_stat') != actual:
            raise ValueError('Confirmed output changed; preserved')
        current = read_json(Path(config['receipt_root'])/(request['name']+'.json'))
        if current['record_id'] != request['record_id']:
            raise ValueError('Observation identity changed after recovery')
        return previous, remote(config, 'ack', previous)['cleaned']
    # A crash after rename but before receipt is resumed only after complete revalidation.
    candidates = all(p.exists() for p in final.values())
    if candidates:
        if any(p.is_symlink() for p in final.values()):
            raise ValueError('Unexpected final symlink')
        summary = verify_file(config, request, final['root'], final['summary'], final['exceptions'])
    else:
        incoming = state/'incoming'/request['request_id']
        incoming.mkdir(parents=True, exist_ok=True)
        if incoming.resolve() != incoming.absolute():
            raise ValueError('Unexpected incoming directory symlink')
        local = {k: incoming/Path(v).name for k,v in product_paths(config, request).items()}
        for key, target in local.items():
            transfer(config, ready['products'][key]['path'], target)
            if target.stat().st_size != ready['products'][key]['bytes']:
                raise ValueError('Transferred byte count mismatch')
        summary = verify_file(config, request, local['root'], local['summary'], local['exceptions'])
        final['root'].parent.mkdir(parents=True, exist_ok=True)
        if final['root'].parent.resolve() != final['root'].parent.absolute():
            raise ValueError('Unexpected final directory symlink')
        for key in ('summary','exceptions','root'):
            if final[key].is_symlink():
                raise ValueError('Unexpected final symlink')
            if final[key].exists():
                if final[key].stat().st_size != local[key].stat().st_size or not filecmp.cmp(final[key], local[key], shallow=False):
                    raise ValueError('Unconfirmed conflicting target preserved: '+str(final[key]))
            else:
                with local[key].open('rb') as file:
                    os.fsync(file.fileno())
                # link provides atomic publication without replacing an existing target.
                os.link(local[key], final[key])
            local[key].unlink()
        fd = os.open(str(final['root'].parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    receipt = previous or dict(request, summary=summary, confirmed_at=time.time(), verification='full-event-alignment+ROOT+bytes',
                               output_stat=[list((p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)) for p in final.values()])
    if not previous:
        atomic_json(receipt_path, receipt)
    cleaned = remote(config, 'ack', receipt)['cleaned']
    return receipt, cleaned


def request_receipts(config, limit_names=None):
    state = Path(config['state'])
    for path in sorted(Path(config['receipt_root']).glob('*.json')):
        receipt = read_json(path)
        name = receipt['name']
        if name not in config['allowed_names'] or (limit_names and name not in limit_names):
            continue
        if receipt['record_id'] != record_id(receipt['record']):
            raise ValueError('Observation receipt/record identity mismatch')
        local = state/'requests'/(name+'.json')
        if local.exists():
            request = read_json(local)
            validate_request(config, request)
            if any(request[k] != receipt[k] for k in ('record_id','entries','bytes')):
                raise ValueError('Observation identity changed: '+name)
        else:
            request = dict(name=name, request_id=uuid.uuid4().hex, record_id=receipt['record_id'],
                           entries=receipt['entries'], bytes=receipt['bytes'], config_id=config['config_id'],
                           git_commit=config['git_commit'])
            validate_request(config, request)
            atomic_json(local, request)
        if not (state/'requested'/(name+'.json')).exists():
            remote(config, 'request', request)
            atomic_json(state/'requested'/(name+'.json'), request)


def receive_loop(config, once_names=None):
    state = Path(config['state'])
    for d in ('requests','requested','receipts','incoming'):
        (state/d).mkdir(parents=True, exist_ok=True)
    errors, retry_at = {}, {}
    history = []
    previous_mode = 2
    started = time.time()
    while time.time()-started < 48*3600:
        try:
            request_receipts(config, once_names)
            ready = remote(config, 'ready')
            errors.pop('receiver',None)
            for item in ready:
                name = item['name']
                if once_names and name not in once_names:
                    continue
                if time.time() < retry_at.get(name, 0):
                    continue
                try:
                    receive_one(config, item)
                    errors.pop(name, None)
                except Exception as exc:
                    errors[name] = str(exc)
                    retry_at[name] = time.time()+30
            receipts = [read_json(p) for p in (state/'receipts').glob('*.json')]
            if once_names:
                receipts = [r for r in receipts if r['name'] in once_names]
            status = remote(config, 'status')
            cleaned = set(status['cleaned'])
            completed = set(r['name'] for r in receipts)
            expected = set(once_names or config['allowed_names'])
            for name in completed & cleaned:
                errors.pop(name,None)
            counters = [sum(r['summary']['status'][i] for r in receipts) for i in range(7)]
            progress = dict(scope='pilot' if once_names else 'month',names=sorted(expected),files=len(completed), cleaned=len(cleaned & expected), expected=len(expected),
                            status=counters, errors=errors, source=status, updated_at=time.time())
            atomic_json(state/'progress.json', progress)
            if expected <= completed and expected <= cleaned and not errors and (once_names or status['queued_bytes']==0):
                atomic_json(state/('pilot_completed.json' if once_names else 'completed.json'), progress)
                return
            # Backpressure is based on observed bytes, not file sizes or Running job counts.
            obs_progress = read_json(Path(config['receipt_root']).parent/'progress.json')
            now = time.time()
            history.append((now, obs_progress['bytes']))
            history = [x for x in history if now-x[0] <= 3600]
            obs_done = obs_progress['files'] == len(config['allowed_names'])
            baseline = config.get('observation_baseline_bytes_per_second', 0)
            mode = previous_mode
            if obs_done:
                mode = 2
            elif baseline and len(history)>2 and now-history[0][0]>=1800:
                def speed(lo, hi):
                    points = [x for x in history if lo<=x[0]<=hi]
                    return (points[-1][1]-points[0][1])/(points[-1][0]-points[0][0]) if len(points)>1 else baseline
                slow = all(speed(now-900*(i+1),now-900*i)<.8*baseline for i in (0,1))
                if slow:
                    mode = max(0, previous_mode-1)
                    history = [(now, obs_progress['bytes'])]
            if mode != previous_mode:
                remote(config, 'concurrency', {'active_workers':mode})
                previous_mode = mode
        except Exception as exc:
            errors['receiver'] = str(exc)
            atomic_json(state/'receiver_error.json', dict(error=str(exc), at=time.time()))
        time.sleep(10)
    raise TimeoutError('Time receiver reached 48-hour deadline; unconfirmed outputs retained')


def legacy_manifest(config, day):
    path = Path(config['control'])/'legacy'/(day+'.tsv')
    if path.exists():
        return path
    lines = []
    root = Path(config['old_root'])
    for main in sorted((root/day[4:]).glob('Esg'+day+'_*.root')):
        friend = root/'recovered_time'/day[4:]/(main.stem+'.time.root')
        summary_path = Path(str(friend)+'.summary.json')
        if not friend.exists() or not summary_path.exists():
            continue
        summary = read_json(summary_path)
        # Legacy summaries have no per-row fallback provenance: reject the whole affected file.
        reusable = int(summary['ies_fallback']==0 and summary['entries']>=0
                       and str(friend) not in config.get('excluded_legacy_friends', {}))
        lines.append(f'{main}\t{friend}\t{reusable}\t0\n')
    temporary = path.with_suffix('.tsv.partial')
    with temporary.open('w') as out:
        out.writelines(lines)
        out.flush();os.fsync(out.fileno())
    temporary.replace(path)
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def reduced_manifest(config, day):
    directory=config['reduced_root'].rstrip('/')+'/'+day[4:]
    listing=eos_command(config,'ls',directory)
    pattern=re.compile(r'ES\.([0-9]+)\.WCDA_EVENT\..*\.es-([0-9]+)\.[0-9]{14}\.([0-9]+)\.reduced\.root$')
    lines=[]
    for path in sorted(listing.splitlines()):
        if not path.startswith(directory+'/'):
            raise ValueError('Reduced listing escaped its day directory')
        match=pattern.fullmatch(Path(path).name)
        if match:
            lines.append(day+'\t'+'\t'.join(match.groups())+'\t'+config['eos_server']+'/'+path+'\n')
    if not lines:
        raise ValueError('No current reduced files for '+day)
    path=Path(config['control'])/'legacy'/('reduced_'+day+'.tsv')
    temporary=path.with_suffix('.partial')
    with temporary.open('w') as out:
        out.writelines(lines);out.flush();os.fsync(out.fileno())
    temporary.replace(path)
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


class RecoveryProcess:
    """Keep only the current day's legacy index in one ROOT process."""
    def __init__(self, config, day, scratch):
        binary = os.environ['OBS_TIME_BINARY']
        self.reduced_manifest = reduced_manifest(config, day)
        args = [binary, '--serve', '--legacy-manifest', str(legacy_manifest(config, day)),
                '--reduced-root', config['reduced_root'], '--reduced-manifest', str(self.reduced_manifest)]
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (4*1024**3, 4*1024**3))
        self.tail = bytearray()
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, preexec_fn=limits)
        def collect():
            while True:
                chunk = self.process.stderr.read(4096)
                if not chunk:
                    break
                self.tail.extend(chunk)
                if len(self.tail)>32768:
                    del self.tail[:-32768]
        self.reader = threading.Thread(target=collect, daemon=True)
        self.reader.start()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Legacy index process failed: '+self.tail.decode(errors='replace'))
            if line.strip()==b'READY':
                break

    def run(self, config, request, output):
        raw = config['raw_root'].rstrip('/')+'/'+request['name'][7:11]+'/'+request['name']
        fields = [raw, str(output), str(request['entries']), request['request_id'], request['git_commit']]
        if any(re.search(r'\s', field) for field in fields):
            raise ValueError('Unexpected whitespace in recovery input')
        self.process.stdin.write((' '.join(fields)+'\n').encode())
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                self.reader.join(timeout=2)
                raise RuntimeError('ROOT recovery failed: '+self.tail.decode(errors='replace'))
            if line.startswith(b'RESULT '):
                if line.decode().strip()[7:] != str(output):
                    raise ValueError('ROOT process returned a different task')
                return

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.reader.join(timeout=2)


def xrdcopy(source, target):
    result = subprocess.run(['xrdcp', '--nopbar', '--silent', '--force', source, target],
                            capture_output=True, timeout=1200)
    if result.returncode:
        raise OSError(result.stderr.decode(errors='replace')[-2048:])


def upload(config, request, local):
    import filecmp
    remote_paths = product_paths(config, request)
    eos_command(config, 'mkdir', '-p', str(Path(remote_paths['root']).parent))
    for key in ('summary', 'exceptions', 'root'):
        remote_path = remote_paths[key]
        xrdcopy(str(local[key]), config['eos_server']+'/'+remote_path+'.partial')
        readback = local[key].with_name(local[key].name+'.readback')
        xrdcopy(config['eos_server']+'/'+remote_path+'.partial', str(readback))
        if key=='root':
            subprocess.run([os.environ['OBS_TIME_BINARY'], '--verify', str(readback)],
                           check=True, timeout=300, stdout=subprocess.DEVNULL)
        elif not filecmp.cmp(local[key], readback, shallow=False):
            raise ValueError('EOS audit content changed')
        if local[key].stat().st_size != readback.stat().st_size:
            raise ValueError('EOS uploaded size changed')
        readback.unlink()
        eos_command(config, 'mv', remote_path+'.partial', remote_path)
    return {key:dict(path=remote_paths[key], bytes=local[key].stat().st_size) for key in local}


def worker(config, index, workers, once_name=None):
    if not 0<=index<workers or workers not in (1,2):
        raise ValueError('Only the validated one/two-worker modes are supported')
    scratch = Path(os.environ['_CONDOR_SCRATCH_DIR'])/'wcda_time'
    scratch.mkdir(exist_ok=True)
    engine, day = None, None
    deadline = time.time()+12*3600
    try:
        while time.time()<deadline:
            ensure_control_budget(config)
            mode_path = Path(config['control'])/'concurrency.json'
            active = read_json(mode_path)['active_workers'] if mode_path.exists() else 0
            if index>=active:
                time.sleep(10)
                continue
            did_work = False
            for request_path in sorted((Path(config['control'])/'requests').glob('*.json')):
                # A priority change must stop claims within the current scan.
                if mode_path.exists() and index>=read_json(mode_path)['active_workers']:
                    break
                request = read_json(request_path)
                validate_request(config, request)
                name = request['name']
                if (once_name and name!=once_name) or (int(name[9:11])-1)%workers!=index:
                    continue
                if control_path(config,'ready',name).exists():
                    if once_name:
                        return
                    continue
                state_path = control_path(config,'states',name)
                old_state = read_json(state_path) if state_path.exists() else {}
                if old_state.get('attempts',0)>=3:
                    continue
                with task_lock(config, name) as acquired:
                    if not acquired:
                        continue
                    if control_path(config,'acks',name).exists():
                        raise ValueError('Acknowledgment exists without durable ready record')
                    amount = request['entries']*1024 + 16*1024**2
                    if amount>2*1024**3:
                        raise ValueError('Task exceeds validated 4 GiB scratch budget including readback')
                    if not reserve(config, request, amount):
                        continue
                    did_work = True
                    attempts = old_state.get('attempts',0)+1
                    started = time.time()
                    atomic_json(state_path,dict(request,state='running',attempts=attempts,started_at=started,worker=index))
                    directory = scratch/request['request_id']
                    try:
                        directory.mkdir(exist_ok=True)
                        eos_remove(config,request)
                        target_day = name[3:11]
                        if day!=target_day or engine is None:
                            if engine:
                                engine.close()
                            engine = RecoveryProcess(config,target_day,scratch)
                            day = target_day
                        output = directory/(name[:-5]+'.time.root.partial')
                        for path in directory.iterdir():
                            if path.is_file():
                                path.unlink()
                        compute_started = time.time()
                        engine.run(config,request,output)
                        compute_seconds = time.time()-compute_started
                        local = dict(root=output,summary=Path(str(output)+'.summary.json'),
                                     exceptions=Path(str(output)+'.exceptions.jsonl'))
                        if sum(p.stat().st_size for p in local.values())>amount:
                            raise ValueError('Actual products exceed reserved queue bytes')
                        summary = read_json(local['summary'])
                        reference_seconds = None
                        if name in config.get('reference_names', []):
                            reference_started = time.time()
                            reference_input = directory/name
                            os.link(output, reference_input)
                            reference_output = directory/'reference.time.root'
                            result = subprocess.run([os.environ['OBS_TIME_REFERENCE'], '--input', str(reference_input),
                                                     '--tree', 't_alignment', '--with-ra-dec', '--reduced-root', config['reduced_root'],
                                                     '--reduced-manifest', str(engine.reduced_manifest),
                                                     '--output', str(reference_output)], capture_output=True, timeout=7200)
                            if result.returncode:
                                raise RuntimeError('Independent reference failed: '+result.stderr.decode(errors='replace')[-2048:])
                            subprocess.run([os.environ['OBS_TIME_BINARY'], '--compare', str(output), '--reference', str(reference_output)],
                                           check=True, timeout=300)
                            reference_seconds = time.time()-reference_started
                            reference_input.unlink();reference_output.unlink()
                            Path(str(reference_output)+'.summary.json').unlink()
                        upload_started = time.time()
                        products = upload(config,request,local)
                        ready = dict(request,products=products,summary=summary,
                                     compute_seconds=compute_seconds,reference_seconds=reference_seconds,upload_seconds=time.time()-upload_started,
                                     elapsed_seconds=time.time()-started,prepared_at=time.time(),attempts=attempts)
                        atomic_json(control_path(config,'ready',name),ready)
                        atomic_json(state_path,dict(ready,state='ready'))
                        for path in directory.iterdir():
                            path.unlink()
                        directory.rmdir()
                        print(json.dumps(dict(event='READY',name=name,summary=summary)),flush=True)
                        if once_name:
                            return
                    except Exception as exc:
                        if engine:
                            tail = bytes(engine.tail).decode(errors='replace')[-32768:]
                            engine.close();engine=None
                        else:
                            tail = ''
                        (Path(config['control'])/'logs'/(name+f'.attempt{attempts}.tail')).write_text(tail[-32768:])
                        atomic_json(state_path,dict(request,state='retry' if attempts<3 else 'failed',
                                                  attempts=attempts,error=str(exc)[-4096:],at=time.time()))
                        print(json.dumps(dict(event='FAILED',name=name,attempt=attempts,error=str(exc)[-1024:])),flush=True)
                        # Reservation and task-owned partials remain accounted until retry or acknowledgment.
                        if once_name and attempts>=3:
                            raise
            if not did_work:
                time.sleep(10)
            cleaned = [read_json(p).get('state')=='cleaned' for p in (Path(config['control'])/'states').glob('*.json')]
            if len(cleaned)==len(config['allowed_names']) and all(cleaned):
                return
    finally:
        if engine:
            engine.close()


def source_status(config):
    states = [read_json(p) for p in (Path(config['control'])/'states').glob('*.json')]
    return dict(cleaned=[s['name'] for s in states if s['state']=='cleaned'],
                failures=[dict(name=s['name'],error=s.get('error'),attempts=s.get('attempts')) for s in states if s['state']=='failed'],
                running=[dict(name=s['name'],started_at=s.get('started_at'),worker=s.get('worker')) for s in states if s['state']=='running'],
                queued_bytes=queue_usage(config),capacity_bytes=config['queue_capacity'])


def remote_action(config, action, payload):
    if action=='request':
        return put_request(config,payload)
    if action=='ack':
        return acknowledge(config,payload)
    if action=='ready':
        ready=[]
        for path in sorted((Path(config['control'])/'ready').glob('*.json')):
            item=read_json(path)
            state=control_path(config,'states',item['name'])
            if not state.exists() or read_json(state).get('state')!='cleaned':
                ready.append(item)
        return ready
    if action=='status':
        return source_status(config)
    if action=='concurrency':
        count=payload['active_workers']
        if type(count) is not int or not 0<=count<=2:
            raise ValueError('Invalid concurrency')
        atomic_json(Path(config['control'])/'concurrency.json',dict(active_workers=count,at=time.time()))
        return dict(active_workers=count)
    raise ValueError('Unknown remote action')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    sub=parser.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('remote');r.add_argument('action');r.add_argument('payload')
    w=sub.add_parser('worker');w.add_argument('--index',type=int,default=0);w.add_argument('--workers',type=int,default=1);w.add_argument('--once')
    recv=sub.add_parser('receive');recv.add_argument('--names',nargs='+')
    sub.add_parser('status')
    args=parser.parse_args()
    config=read_json(args.config)
    if args.mode=='remote':
        payload=json.loads(base64.b64decode(args.payload))
        print(json.dumps(remote_action(config,args.action,payload)))
    elif args.mode=='worker':
        worker(config,args.index,args.workers,args.once)
    elif args.mode=='receive':
        Path(config['state']).mkdir(parents=True,exist_ok=True)
        with (Path(config['state'])/'receiver.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            receive_loop(config,set(args.names) if args.names else None)
    else:
        print(json.dumps(source_status(config)))


if __name__=='__main__':
    main()
