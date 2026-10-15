"""Slurm controller for the final HTCondor missing-header investigation."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent/'observation_nhit_scalar'))
from common import atomic_json,require_slurm,sha256
from transport import remote,remote_py,quota,transfer,REMOTE


def run(state):
    require_slurm()
    if not (state/'finished.json').exists():raise RuntimeError('Production is not terminal')
    work=state/'audit'/'header_fullmonth';work.mkdir(parents=True,exist_ok=True)
    lock=(work/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((state/'config.json').read_text());stage=cfg['stage']+'/header_audit'
    tasks={r['name']:r for r in json.loads((state/'manifest.json').read_text())['records']}
    specs=[]
    for f in (state/'production').glob('*/failed.json'):
        rec=json.loads(f.read_text())
        m=re.search(r'/eos/lhaaso/raw/wcda/2022/\d{4}/[^;\n]+?\.reduced\.root',rec['error'])
        if m:specs.append(dict(name=rec['name'],required_path=m[0],target=tasks[rec['name']]['target'],target_stamp=tasks[rec['name']]['input']))
    specs.sort(key=lambda r:r['name'])
    atomic_json(work/'specs.json',specs)
    if not specs:
        atomic_json(work/'completed.json',dict(files=0,reason='No header failures'));return
    q=quota();atomic_json(work/'quota.json',q)
    usage=int(remote('du -sb '+shlex.quote(cfg['stage'])).split()[0])
    if min(q['soft'],q['hard'])-q['used']<528*1024**2 or usage+16*1024**2>cfg['stage_budget_bytes']:
        raise RuntimeError('No headroom for 16 MiB bounded metadata investigation')
    remote('mkdir -p '+shlex.quote(stage))
    script=Path(__file__).with_name('investigate_observation_nhit_headers.py')
    transfer(str(script),REMOTE+':'+stage+'/probe.py');transfer(str(work/'specs.json'),REMOTE+':'+stage+'/specs.json')
    worker=work/'worker.sh'
    worker.write_text('#!/bin/bash\nset -eo pipefail\ncase "$(hostname -s)" in lhws183|bws0995) exit 75;; esac\n'
        'ulimit -f 8192\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh\n'
        'exec timeout 1200 python3 -u -B '+stage+'/probe.py '+stage+'/specs.json '+stage+'/result.json\n')
    worker.chmod(0o755);transfer(str(worker),REMOTE+':'+stage+'/worker.sh');remote('chmod 755 '+stage+'/worker.sh')
    if not (work/'submitted.json').exists():
        atomic_json(work/'submitted.json',dict(status='pending',at=time.time()))
        output=remote('source /afs/ihep.ac.cn/soft/common/sysgroup/hep_job/setup_hepjob.sh; cd '+stage+
            '; hep_sub ./worker.sh -g lhaaso -os AlmaLinux9 -m 2000 -cpu 1 -wt short -schedd schedd11 -o probe.out -e probe.err',attempts=1)
        atomic_json(work/'submission_output.json',dict(output=output))
        m=re.search(r'cluster\s+(\d+)',output,re.I)
        if not m:raise RuntimeError('Ambiguous header probe submission')
        cluster=int(m[1]);atomic_json(work/'submitted.json',dict(cluster=cluster,at=time.time()))
        policy=remote_py('''import json,subprocess
b=['-name','scheduler@schedd11.ihep.ac.cn'];j=%r
a=json.loads(subprocess.check_output(['condor_q']+b+[j,'-json','-attributes','Owner,Cmd,Requirements'],text=True))[0]
assert a['Owner']=='liushijie' and a['Cmd']==%r+'/./worker.sh'
r=a['Requirements'];assert r.startswith('/Expr(') and r.endswith(')/')
n='('+r[6:-2]+') && (TARGET.Machine != "lhws183.ihep.ac.cn") && (TARGET.Machine != "bws0995.ihep.ac.cn")'
subprocess.run(['condor_qedit']+b+[j,'Requirements',n],check=True,stdout=subprocess.PIPE)
print(json.dumps(dict(job=j,original=r,updated=n)))
''' % (str(cluster)+'.0',stage))
        atomic_json(work/'scheduler_policy.json',json.loads(policy))
    submitted=json.loads((work/'submitted.json').read_text())
    if 'cluster' not in submitted:raise RuntimeError('Inspect ambiguous existing probe submission; never blindly duplicate')
    start=time.monotonic()
    while True:
        done=remote_py('from pathlib import Path; print(int(Path('+repr(stage+'/result.json')+').exists()))').strip()
        if done=='1':break
        if time.monotonic()-start>3600:raise RuntimeError('Header probe did not complete within one hour; inspect scheduler')
        time.sleep(20)
    source_digest=remote('sha256sum '+shlex.quote(stage+'/result.json')).split()[0]
    transfer(REMOTE+':'+stage+'/',str(work/'source')+'/')
    if sha256(work/'source/result.json')!=source_digest:
        raise ValueError('Header evidence transfer digest mismatch')
    result=json.loads((work/'source/result.json').read_text())
    if result['script_sha256']!=sha256(script) or result['specs_sha256']!=sha256(work/'specs.json'):
        raise ValueError('Header probe code/spec binding mismatch')
    atomic_json(work/'completed.json',dict(files=len(specs),result_sha256=sha256(work/'source/result.json'),
        reliable_exact_candidates=sum(bool(r.get('reliable_header') and r['same_segment_stem']) for s in result['specs'] for r in s['matches']),
        directory_errors=[r for r in result['directories'] if 'error' in r],at=time.time()))
    print((work/'completed.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);a=p.parse_args();run(a.state)
