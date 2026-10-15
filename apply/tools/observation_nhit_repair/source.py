"""Persistent HTCondor source worker, scalar PINC candidates, bounded staging."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback
import uuid

import awkward as ak
import numpy as np
import uproot
from common import KEYS, atomic_json, digest_column, sha256, source_uri, stamp, sync_dir, unique_keys
from transport import parse_quota

CONTRACT = dict(version=3, selection='pincness < 1.1',
    nhit='sum((vqsamp >= float32(0.3)) & (abs(vt) < 30))',
    edges=[30,60,100,200,300,500,800,2000], charge_saved=False, vidmc_read=False,
    poolused='identity-checked unique reduced wcdah; two digest-bound user-confirmed missing-segment exceptions; no pool cut',
    eto_validation='unique keys, nv, source position and segment mapping; no charge recomputation or 30-column comparison')
LIMIT = 128*1024**2
RESERVE = 2*LIMIT+16*1024**2


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def header_specs(headers, day, wanted):
    seen, specs = set(), []
    for h in headers:
        name = h['filename']
        m = re.fullmatch(r'ES\.(\d+)\.WCDA_EVENT\..+\.es-(\d+)\.\d{14}\.(\d+)\.samp\.dt\.root', name)
        if not m:
            raise ValueError('unrecognized segment filename: '+name)
        key = tuple(map(int, m.groups()))
        if key in seen:
            raise ValueError('duplicate segment header: '+str(key))
        seen.add(key)
        if h['infiles'] != name.replace('.samp.dt.root', '.samp.root'):
            raise ValueError('segment filename/infiles conflict: '+name)
        if key in wanted:
            specs.append(dict(key=key, header=h, path='/eos/lhaaso/raw/wcda/2022/'+day+'/'+name.replace('.samp.dt.root','.reduced.root')))
    if wanted-seen:
        raise ValueError('missing segment headers: '+repr(sorted(wanted-seen)))
    return specs


def validate_header(key, data, entries):
    if entries != 1:
        raise ValueError('segment wcdah is not unique')
    if tuple(int(data[k][0]) for k in KEYS[:3]) != tuple(key):
        raise ValueError('segment identity mismatch')
    pool = int(data['poolused'][0])
    if not 0 <= pool <= 7:
        raise ValueError('invalid pool mask')
    return pool


class HeaderTransportExhausted(RuntimeError):
    """Three header attempts consumed; must not be retried by the file loop."""


def is_transient_file_error(error):
    if isinstance(error, HeaderTransportExhausted):
        return False
    text=str(error).lower()
    return '3011' not in text and any(s in text for s in ['timeout','timed out','connection','read failure','read failed'])


def read_header(spec):
    for attempt in range(3):
        try:
            with uproot.open(source_uri(spec['path']), timeout=60,
                             handler=uproot.source.xrootd.MultithreadedXRootDSource, num_workers=1) as f:
                if sum(k.split(';')[0] == 'wcdah' for k in f.keys()) != 1:
                    raise ValueError('non-unique wcdah key')
                t = f['wcdah']
                a = t.arrays(KEYS[:3]+['poolused'], library='np')
                if any(a[k].dtype.kind not in 'iu' for k in a):
                    raise ValueError('non-integer header identity/pool')
                pool = validate_header(spec['key'], a, t.num_entries)
                return dict(irun=spec['key'][0], ies=spec['key'][1], iseq=spec['key'][2], poolused=pool,
                            path=spec['path'], uuid=str(f.file.uuid), source_filename=spec['header']['filename'],
                            header_sha256=json_sha(spec['header']), reduced_header_sha256=json_sha({k:v.tolist() for k,v in a.items()}),
                            attempts=attempt+1)
        except OSError as e:
            if '3011' in str(e) or 'No such file' in str(e):
                raise
            if attempt==2:
                raise HeaderTransportExhausted(str(e)) from e
            print('HEADER_RETRY', spec['path'], attempt+1, repr(e), flush=True)
            time.sleep(2*(attempt+1))


def segment_rows(headers, day, wanted, task, policy):
    specs = header_specs(headers, day, wanted)
    from policy import override_row, check_rows
    def read(spec):
        row = override_row(spec, task, policy)
        if row is None:
            row = dict(read_header(spec), poolused_source='reduced_header',
                       poolused_authorization_sha256='', reduced_header_missing=0)
        return row
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(read, specs))
    check_rows(rows, task, policy)
    for i, row in enumerate(rows):
        row['segment_index'] = i
    return rows


def write_scalar(path, tree, data, rows, provenance):
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as f:
        f.mktree(tree, {k:'int64' if k in ['input_entry','ihep_entry'] else 'int32' for k in data})
        if len(data['nv']):
            f[tree].extend(data)
        f.mktree('t_segments', {k:'int32' for k in ['segment_index']+KEYS[:3]+['poolused','attempts','reduced_header_missing']} |
            {k:'string' for k in ['path','uuid','source_filename','header_sha256','reduced_header_sha256','poolused_source','poolused_authorization_sha256']})
        if rows:
            f['t_segments'].extend(ak.Array(rows))
        f['provenance'] = json.dumps(provenance, sort_keys=True)


def readback(path, tree, data, rows, provenance):
    with uproot.open(path) as f:
        actual = f[tree].arrays(library='np')
        if set(actual) != set(data) or f[tree].num_entries != len(data['nv']):
            raise ValueError('readback schema/count')
        for k in data:
            if actual[k].dtype != data[k].dtype or not np.array_equal(actual[k], data[k]):
                raise ValueError('readback scalar mismatch: '+k)
        if f['t_segments'].arrays().to_list() != rows:
            raise ValueError('readback segment table')
        if json.loads(str(f['provenance'])) != provenance:
            raise ValueError('readback provenance')
    with Path(path).open('rb') as f:
        os.fsync(f.fileno())


def extract(stage, folder, task, config, ROOT, session):
    start = time.monotonic(); name = task['name']
    from policy import load_policy, authorization
    policy = load_policy(stage, config)
    authorization(task, policy)
    src = '/eos/lhaaso/ai/wcda/raw/2022/'+name[7:11]+'/'+name
    before = stamp(src)
    if before['entries'] != task['source_entries'] or before['header_sha256'] != task['input']['header_sha256']:
        raise ValueError('source metadata differs from frozen input')
    tmp = folder/'hits.partial.root'
    stats = ROOT.nhit_benchmark_filter(source_uri(src), str(tmp))
    with uproot.open(tmp) as f:
        data = f['t_candidate'].arrays(library='np')
    unique_keys(data)
    if len(data['nv']) != stats.selected:
        raise ValueError('PINC selection count')
    then = time.monotonic()
    with uproot.open(source_uri(src), timeout=60) as f:
        headers = f['t_eventout_h'].arrays().to_list()
    wanted = set(zip(*(data[k].tolist() for k in KEYS[:3])))
    rows = segment_rows(headers, name[7:11], wanted, task, policy)
    header_seconds = time.monotonic()-then
    mapping = {tuple(r[k] for k in KEYS[:3]):r for r in rows}
    indices = [mapping[k]['segment_index'] for k in zip(*(data[k].tolist() for k in KEYS[:3]))]
    data['segment_index'] = np.asarray(indices, dtype='i4')
    data['poolused'] = np.asarray([rows[i]['poolused'] for i in indices], dtype='i4')
    prov = dict(contract=CONTRACT, source_path=src, source_stamp=before, target_path=task['target'],
                target_stamp=task['input'], code=config['code'], git_commit=config['git_commit'], run_id=config['run_id'], repair_policy=policy['binding'])
    out = folder/'candidate.partial.root'
    then = time.monotonic(); write_scalar(out, 't_nhit', data, rows, prov)
    write_seconds = time.monotonic()-then+float(stats.write_seconds)
    then = time.monotonic(); readback(out, 't_nhit', data, rows, prov)
    readback_seconds = time.monotonic()-then
    if stamp(src) != before:
        raise ValueError('source changed during extraction')
    os.rename(out, folder/'candidate.root'); sync_dir(folder); tmp.unlink()
    record = dict(name=name, entries=len(data['nv']), raw_entries=int(stats.raw), source=before,
                  bytes=(folder/'candidate.root').stat().st_size, sha256=sha256(folder/'candidate.root'),
                  digests={k:digest_column(ak.Array(v)) for k,v in data.items()}, segments=rows, provenance=prov,
                  elapsed_seconds=time.monotonic()-start, header_seconds=header_seconds,
                  write_seconds=write_seconds, readback_seconds=readback_seconds,
                  session=session, finished_at=time.time(), source_readback=True,
                  root={k:int(getattr(stats,k)) for k in ['scan_bytes','hit_bytes','scan_calls','hit_calls','charge_boundary','time_boundary','over_4096']} |
                       {k:float(getattr(stats,k)) for k in ['scan_seconds','hit_seconds']})
    atomic_json(folder/'ready.json', record)
    print('READY',name,record['elapsed_seconds'],flush=True)


def stage_usage(stage):
    return int(subprocess.check_output(['du','-sb',str(stage)],text=True).split()[0])


def reserve(stage, folder, config):
    while True:
        if (stage/'STOP_EXPANSION.json').exists():
            raise RuntimeError('Input identity changed; source expansion stopped')
        with (stage/'budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            usage = stage_usage(stage)
            active = sum(json.loads(p.read_text())['bytes'] for p in stage.glob('*/*/reservation.json'))
            q = parse_quota(subprocess.check_output(['lfs','quota','-u','liushijie','/scratchfs2'],text=True,timeout=30))
            sample = dict(q, stage_bytes=usage, reserved_bytes=active, at=time.time())
            with (stage/'quota_history.jsonl').open('a') as f:
                f.write(json.dumps(sample)+'\n')
            if (usage+active+RESERVE <= config['stage_budget_bytes'] and
                min(q['soft'],q['hard'])-q['used']-active-RESERVE > 512*1024**2 and
                q['files']+2000 < min(q['files_soft'],q['files_hard'])):
                atomic_json(folder/'reservation.json',dict(bytes=RESERVE,at=time.time()))
                return
        print('BACKPRESSURE', usage, active, flush=True)
        time.sleep(30)


def worker(stage, phase, shard):
    if not os.environ.get('_CONDOR_JOB_AD'):
        raise RuntimeError('Source computation requires HTCondor')
    started = time.monotonic()
    config = json.loads((stage/'config.json').read_text())
    for name, digest in config['code'].items():
        if sha256(stage/name) != digest:
            raise ValueError('frozen source code changed: '+name)
    lock = (stage/(phase+'-'+str(shard)+'.lock')).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('DUPLICATE_WORKER_REJECTED',phase,shard,flush=True); return
    # A second process attempts this exact lock; the kernel must reject it.
    probe = subprocess.run([sys.executable,'-c',
        'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',lock.name],capture_output=True)
    if probe.returncode==0:
        raise RuntimeError('duplicate worker lock probe unexpectedly succeeded')
    session = dict(id=uuid.uuid4().hex, pid=os.getpid(), host=os.uname().nodename,
                   phase=phase, shard=shard, started_at=time.time(), duplicate_guard_verified=True)
    resource.setrlimit(resource.RLIMIT_FSIZE,(LIMIT,LIMIT))
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not ROOT.gInterpreter.Declare((stage/'pinc_only.C').read_text()):
        raise RuntimeError('ROOT compilation failed')
    ROOT.nhit_benchmark_selfcheck()
    from selftest import root_selftest
    checkdir=stage/'sessions'/session['id'];checkdir.mkdir()
    session['root_selftest']=root_selftest(ROOT,checkdir);checkdir.rmdir()
    session['startup_seconds'] = time.monotonic()-started
    session['process_startup_seconds'] = time.time()-float(os.environ.get('NHIT_WORKER_BOOT_TIME',session['started_at']))
    atomic_json(stage/'sessions'/(session['id']+'.json'),session)
    if sha256(stage/'tasks.json') != config['tasks_sha256']:
        raise ValueError('frozen repair tasks changed')
    all_tasks = json.loads((stage/'tasks.json').read_text())
    tasks = all_tasks['pilot'] if phase=='pilot' else all_tasks['production'][shard::4]
    processed, skipped = [], []
    for task in tasks:
        if (stage/'STOP_EXPANSION.json').exists() or time.monotonic()-started > 8*3600:
            break  # Checkpoints allow a fresh job before the site's long walltime.
        folder = stage/phase/task['name']; folder.mkdir(parents=True,exist_ok=True)
        if any((folder/n).exists() for n in ['ack.json','ready.json','failed.json']):
            skipped.append(task['name']); continue
        # Exclusive shard lock owns these leftovers; ready/ack never enter this path.
        leftovers=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in folder.iterdir() if p.name in ['hits.partial.root','candidate.partial.root','candidate.root','reservation.json']]
        if leftovers:
            atomic_json(folder/('restart-'+session['id']+'.json'),dict(session=session,leftovers=leftovers,action='remove only unconfirmed intermediates before retry'))
        for suffix in ['hits.partial.root','candidate.partial.root','candidate.root','reservation.json']:
            (folder/suffix).unlink(missing_ok=True)
        reserve(stage,folder,config)
        print('START',task['name'],flush=True)
        attempts=[]
        for attempt in range(3):
            try:
                extract(stage,folder,task,config,ROOT,session)
                processed.append(task['name']); break
            except Exception as e:
                failure=dict(name=task['name'],error=repr(e),traceback=traceback.format_exc(),
                             attempt=attempt+1,at=time.time(),session=session)
                attempts.append(failure); atomic_json(folder/'attempts.json',attempts)
                print('FILE_ERROR',json.dumps(failure),flush=True)
                # Data errors and missing headers are never guessed or retried.
                transient=is_transient_file_error(e)
                for n in ['hits.partial.root','candidate.partial.root','candidate.root']:
                    (folder/n).unlink(missing_ok=True)
                if not transient or attempt==2:
                    atomic_json(folder/'failed.json',failure); break
                time.sleep(3*(attempt+1))
        (folder/'reservation.json').unlink(missing_ok=True)
    atomic_json(stage/'sessions'/(session['id']+'.done.json'),dict(session,processed=processed,skipped=skipped,
        elapsed_seconds=time.monotonic()-started,finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('phase',choices=['pilot','production']);p.add_argument('shard',type=int)
    a=p.parse_args();worker(a.stage,a.phase,a.shard)
