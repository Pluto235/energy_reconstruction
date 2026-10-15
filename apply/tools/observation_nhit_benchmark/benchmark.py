"""Isolated, paired full-file benchmark. Never publishes production friends."""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'snapshot' if (HERE/'snapshot').exists() else HERE.parent/'observation_nhit'))
import awkward as ak
import numpy as np
import uproot
from common import KEYS, atomic_json, digest_column, histogram, keybytes, require_slurm, sha256, source_uri, stamp, unique_keys
from extract import extract, segments
from run import quota, remote, remote_py, transfer, SSH, REMOTE
from verify import verify

NAMES = ['Esg20220101_00.root', 'Esg20220116_00.root', 'Esg20220131_00.root']
STATE = Path('/mnt/newdisk/shijie/WCDA_obs_friends/nhit/benchmarks/20261015_pinc_only_v1')
STAGE = '/scratchfs2/lhaaso/liushijie/WCDA_nhit_bench_20261015_pinc_only_v1'
MANIFEST = Path('/mnt/newdisk/shijie/WCDA_obs_friends/nhit/manifests/20261015_january_v5')
INPUT = Path('/mnt/newdisk/shijie/WCDA_obs')


def align(target, candidate):
    """Missing entries use -1; duplicates always reject, including surplus candidates."""
    t, c = unique_keys(target), unique_keys(candidate)
    order = np.argsort(c)
    pos = np.searchsorted(c[order], t)
    rows = np.full(len(t), -1, dtype='i8')
    if len(c):
        bounded = np.minimum(pos, len(c)-1)
        hit = (pos < len(c)) & (c[order[bounded]] == t)
        rows[hit] = order[pos[hit]]
    return rows, len(c)-int(np.count_nonzero(rows >= 0))


def selftest():
    def keys(values):
        return {k: np.asarray(values if k == 'ievent' else [0]*len(values), dtype='i4') for k in KEYS}
    rows, extra = align(keys([3,1,8]), keys([1,9,3]))
    assert rows.tolist() == [2,0,-1] and extra == 1
    assert align(keys([]),keys([]))[0].size == 0
    assert align(keys([1]),keys([]))[0].tolist() == [-1]
    for a,b in [(keys([1,1]),keys([1])),(keys([1]),keys([1,1]))]:
        try: align(a,b)
        except ValueError: pass
        else: raise AssertionError('duplicate accepted')
    print('alignment self-check passed',flush=True)


def candidate(work, name, ROOT):
    started = time.monotonic()
    task = json.loads((work/'baseline'/(name+'.task.json')).read_text())
    src = task['source']; before = stamp(src)
    if before['entries'] != task['source_entries'] or before['header_sha256'] != task['input']['header_sha256']:
        raise ValueError('source metadata differs from frozen input')
    tmp = work/(name+'.pinc.partial.root')
    stats = ROOT.nhit_benchmark_filter(source_uri(src), str(tmp))
    with uproot.open(tmp) as f:
        data = f['t_candidate'].arrays(library='np')
    if len(data['nv']) != stats.selected:
        raise ValueError('PINC result count mismatch')
    wanted = set(zip(*(data[k].tolist() for k in KEYS[:3])))
    then = time.monotonic()
    with uproot.open(source_uri(src),timeout=60) as f:
        mapping, rows = segments(f,name[7:11],wanted)
    header_seconds = time.monotonic()-then
    selected = [mapping[k] for k in zip(*(data[k].tolist() for k in KEYS[:3]))]
    data.update(poolused=np.array([r['poolused'] for r in selected],dtype='i4'),
                segment_index=np.array([r['segment_index'] for r in selected],dtype='i4'))
    out = work/(name+'.candidate.root')
    if out.exists(): raise FileExistsError(out)
    with uproot.recreate(out,compression=uproot.ZLIB(4)) as f:
        f.mktree('t_nhit',{k:'int64' if k=='ihep_entry' else 'int32' for k in data})
        f['t_nhit'].extend(data)
        f.mktree('t_segments',{k:'int32' for k in ['segment_index']+KEYS[:3]+['poolused']} |
                 {k:'string' for k in ['path','uuid','source_filename','header_sha256']})
        if rows: f['t_segments'].extend(ak.Array(rows))
        f['provenance']=json.dumps(dict(selection='pincness < 1.1',source=src,source_stamp=before,
                                        benchmark_only=True,charge_saved=False),sort_keys=True)
    digests = {k:digest_column(ak.Array(v)) for k,v in data.items()}
    then=time.monotonic()
    with uproot.open(out) as f:
        read=f['t_nhit'].arrays(library='np')
        assert set(read)==set(data) and 'vqsamp' not in read
        if f['t_segments'].arrays().to_list()!=rows:raise ValueError('segment readback')
        for k,v in read.items():
            if digest_column(ak.Array(v))!=digests[k]:raise ValueError('scalar readback '+k)
    readback_seconds=time.monotonic()-then
    if stamp(src)!=before:raise ValueError('source changed')
    with out.open('rb') as f:os.fsync(f.fileno())
    record=dict(name=name,entries=len(data['nv']),raw_entries=int(stats.raw),source=before,
        bytes=out.stat().st_size,sha256=sha256(out),digests=digests,segments=rows,
        elapsed_seconds=time.monotonic()-started,header_seconds=header_seconds,
        readback_seconds=readback_seconds,
        root={k:int(getattr(stats,k)) for k in ['scan_bytes','hit_bytes','scan_calls','hit_calls']} |
             {k:float(getattr(stats,k)) for k in ['scan_seconds','hit_seconds']},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(work/(name+'.candidate.ready.json'),record)
    tmp.unlink()
    print('CANDIDATE_DONE',name,record['elapsed_seconds'],flush=True)


def source(work):
    resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
    selftest(); then=time.monotonic()
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not ROOT.gInterpreter.Declare((work/'pinc_only.C').read_text()):raise RuntimeError('ROOT compile failed')
    ROOT.nhit_benchmark_selfcheck()
    atomic_json(work/'setup.json',dict(root_setup_seconds=time.monotonic()-then,
                                      host=os.uname().nodename,started_at=time.time()))
    for i,name in enumerate(NAMES):
        modes=['baseline','candidate'] if i%2==0 else ['candidate','baseline']
        for mode in modes:
            print('BEGIN',mode,name,time.time(),flush=True)
            if mode=='baseline':extract(work/'baseline',name)
            else:candidate(work,name,ROOT)
    atomic_json(work/'source.complete.json',dict(finished_at=time.time()))


def check_candidate(work,name):
    rec=json.loads((work/(name+'.candidate.ready.json')).read_text())
    path=work/(name+'.candidate.root')
    if path.stat().st_size!=rec['bytes'] or sha256(path)!=rec['sha256']:raise ValueError('candidate transfer checksum')
    then=time.monotonic()
    inp=INPUT/name[7:11]/name
    frozen=json.loads((work/'baseline'/(name+'.task.json')).read_text())['input']
    if stamp(inp)!=frozen:raise ValueError('input changed')
    with uproot.open(inp) as f:a=f['t_eventout'].arrays(KEYS+['nv'],library='np')
    with uproot.open(path) as f:
        b=f['t_nhit'].arrays(library='np'); seg=f['t_segments'].arrays().to_list()
    if 'vqsamp' in b or seg!=rec['segments']:raise ValueError('candidate schema/header')
    for k,d in rec['digests'].items():
        if digest_column(ak.Array(b[k]))!=d:raise ValueError('candidate content '+k)
    rows,extra=align(a,b);missing=np.flatnonzero(rows<0)
    if len(missing):raise ValueError('missing target events: '+str(missing[:10].tolist()))
    with uproot.open(work/'baseline'/(name+'.friend.root')) as f:
        ref=f['t_nhit'].arrays(KEYS+['nv','nhit','poolused','ihep_entry'],library='np')
    for k in ref:
        if not np.array_equal(ref[k],b[k][rows]):raise ValueError('baseline mismatch '+k)
    if not np.array_equal(a['nv'],b['nv'][rows]) or not np.array_equal(keybytes(a),keybytes(ref)):
        raise ValueError('target identity/nv mismatch')
    if len(np.unique(b['ihep_entry']))!=len(b['ihep_entry']):raise ValueError('duplicate source entry')
    ids=b['segment_index']
    if np.any(ids<0) or np.any(ids>=len(seg)):raise ValueError('invalid segment index')
    for k in KEYS[:3]+['poolused']:
        if not np.array_equal(np.array([r[k] for r in seg])[ids],b[k]):raise ValueError('segment mapping '+k)
    if stamp(inp)!=frozen:raise ValueError('input changed after comparison')
    return dict(events=len(rows),candidate_events=len(b['nv']),surplus=extra,missing=0,duplicates=0,
                nhit_equal_fraction=1,bin_mismatch_fraction=0,eto_seconds=time.monotonic()-then,
                equality='all target keys/nv/nhit/poolused/ihep_entry against freshly rerun current extractor',
                charge_transferred=False,production_accepted=False)


def controller():
    require_slurm();STATE.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(HERE/'benchmark.py',STATE/'benchmark.py');shutil.copyfile(HERE/'pinc_only.C',STATE/'pinc_only.C')
    snapshot=STATE/'snapshot';snapshot.mkdir()
    for f in (HERE.parent/'observation_nhit').glob('*.py'):shutil.copyfile(f,snapshot/f.name)
    (STATE/'baseline').mkdir()
    q=quota();atomic_json(STATE/'quota.json',q)
    # Reserve separately from the production receiver's maximum 4 GiB staging budget.
    if q['soft']-q['used']<5.5*1024**3 or q['hard']-q['used']<10.5*1024**3:
        raise RuntimeError('insufficient quota for bounded 512 MiB benchmark plus production reserves')
    frozen=json.loads((MANIFEST/'manifest.json').read_text());byname={r['name']:r for r in frozen['records']}
    atomic_json(STATE/'manifest.json',dict(records=[byname[n] for n in NAMES],
        source_code={f.name:sha256(f) for f in snapshot.glob('*.py')},benchmark_code=sha256(HERE/'benchmark.py'),
        orders=[['baseline','candidate'],['candidate','baseline'],['baseline','candidate']],
        note='One independent HTCondor benchmark worker, no production mutation/publication; cached I/O order is not controlled'))
    from run import prepare
    for name in NAMES:
        task=prepare(STATE/'baseline',byname[name],{f.name:sha256(f) for f in snapshot.glob('*.py')},STAGE)
        task['file_limit']=256*1024**2
        atomic_json(STATE/'baseline'/(name+'.task.json'),task)
    remote('mkdir '+shlex.quote(STAGE))
    transfer(str(STATE)+'/',REMOTE+':'+STAGE+'/')
    worker=STATE/'worker.sh'
    worker.write_text('#!/bin/bash\nset -eo pipefail\necho BENCH_WORKER_START $(date -Is)\nsource /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexec python3 -u -B '+STAGE+'/benchmark.py --source '+STAGE+'\n')
    worker.chmod(0o755);transfer(str(worker),REMOTE+':'+STAGE+'/worker.sh')
    cmd=('source /afs/ihep.ac.cn/soft/common/sysgroup/hep_job/setup_hepjob.sh; cd '+shlex.quote(STAGE)+
         '; hep_sub ./worker.sh -g lhaaso -os AlmaLinux9 -m 4000 -cpu 1 -schedd schedd11 -o bench.out -e bench.err')
    atomic_json(STATE/'submission.json',dict(status='submission_pending',time=time.time()))
    proc=subprocess.run(SSH+[REMOTE,cmd],text=True,capture_output=True,timeout=120)
    atomic_json(STATE/'submission_output.json',dict(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode))
    if proc.returncode:raise RuntimeError('ambiguous benchmark submission; inspect before retry')
    import re
    m=re.search(r'cluster\s+(\d+)',proc.stdout,re.I)
    if not m:raise RuntimeError('unknown cluster; inspect before retry')
    cluster=int(m[1]);atomic_json(STATE/'submission.json',dict(cluster=cluster,time=time.time(),status='submitted'))
    remote_py(f'''import json,subprocess
b=['-name','scheduler@schedd11.ihep.ac.cn'];j={str(cluster)+'.0'!r}
a=json.loads(subprocess.check_output(['condor_q']+b+[j,'-json','-attributes','Owner,Cmd,Requirements'],text=True))[0]
assert a['Owner']=='liushijie' and a['Cmd'].startswith({STAGE!r}+'/')
r=a['Requirements'];assert r.startswith('/Expr(') and r.endswith(')/')
r='('+r[6:-2]+') && (TARGET.Machine != "lhws183.ihep.ac.cn") && (TARGET.Machine != "bws0995.ihep.ac.cn")'
subprocess.run(['condor_qedit']+b+[j,'Requirements',r],check=True)
''')
    print('BENCH_SUBMITTED',cluster,flush=True)
    complete=set()
    while len(complete)<len(NAMES):
        listing=json.loads(remote_py(f'import os,json; print(json.dumps(os.listdir({STAGE!r})))'))
        for name in NAMES:
            if name in complete or name+'.candidate.ready.json' not in listing:continue
            ready=remote_py(f'import os;print(int(os.path.exists({STAGE!r}+"/baseline/"+{name!r}+".ready.json")))').strip()
            if ready!='1':continue
            timings={}
            for suffix in ['.candidate.ready.json','.candidate.root']:
                then=time.monotonic();transfer(REMOTE+':'+STAGE+'/'+name+suffix,str(STATE/(name+suffix)))
                timings[suffix]=time.monotonic()-then
            for suffix in ['.ready.json','.friend.root']:
                then=time.monotonic();transfer(REMOTE+':'+STAGE+'/baseline/'+name+suffix,str(STATE/'baseline'/(name+suffix)))
                timings['baseline'+suffix]=time.monotonic()-then
            rec=json.loads((STATE/'baseline'/(name+'.ready.json')).read_text())
            ref=verify(INPUT/name[7:11]/name,STATE/'baseline'/(name+'.friend.root'),rec,byname[name]['input'])
            report=check_candidate(STATE,name)
            atomic_json(STATE/(name+'.comparison.json'),dict(candidate=report,baseline_validation=ref,
                         transfer_seconds=timings,candidate_source=json.loads((STATE/(name+'.candidate.ready.json')).read_text()),
                         baseline_source={k:rec[k] for k in ['elapsed_seconds','bytes','entries','peak_rss_kib']}))
            complete.add(name);print('BENCH_VERIFIED',name,flush=True)
        if len(complete)<len(NAMES):time.sleep(30)
    transfer(REMOTE+':'+STAGE+'/setup.json',str(STATE/'setup.json'))
    atomic_json(STATE/'complete.json',dict(files=NAMES,finished_at=time.time(),production_changed=False))
    print('BENCH_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.selftest:selftest()
    elif args.source:source(args.source)
    else:controller()
