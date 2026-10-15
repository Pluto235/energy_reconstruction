"""Slurm control for one bounded metadata-only HTCondor diagnosis."""
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
    work=state/'audit/header_io_v1';work.mkdir(parents=True,exist_ok=True)
    lock=(work/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stage='/scratchfs2/lhaaso/liushijie/WCDA_nhit_20261016_header_io_v1'
    result=state/'audit/header_fullmonth/source/result.json'
    expected=json.loads((state/'audit/header_fullmonth/completed.json').read_text())['result_sha256']
    if sha256(result)!=expected:raise ValueError('independent header evidence changed')
    spec=next(s for s in json.loads(result.read_text())['specs'] if s['name']=='Esg20220128_06.root')
    matches=[m for m in spec['matches'] if m.get('reliable_header') and m['same_segment_stem']]
    if len(matches)!=1:raise ValueError('no unique earlier header evidence')
    atomic_json(work/'spec.json',dict(name=spec['name'],expected=matches[0],target_stamp=spec['target_stamp'],evidence_sha256=expected))
    q=quota();atomic_json(work/'quota.json',q)
    if min(q['soft'],q['hard'])-q['used']<528*1024**2:raise RuntimeError('insufficient personal quota')
    script=Path(__file__).with_name('diagnose_observation_nhit_header_io.py')
    if not (work/'submitted.json').exists():
        remote('mkdir -p '+shlex.quote(stage))
        transfer(str(script),REMOTE+':'+stage+'/probe.py');transfer(str(work/'spec.json'),REMOTE+':'+stage+'/spec.json')
        worker=work/'worker.sh'
        worker.write_text('#!/bin/bash\nset -eo pipefail\ncase "$(hostname -s)" in lhws183|bws0995) exit 75;; esac\n'
            'ulimit -f 8192\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
            'source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh\n'
            'exec timeout 400 python3 -u -B '+stage+'/probe.py '+stage+'/spec.json '+stage+'/result.json\n')
        transfer(str(worker),REMOTE+':'+stage+'/worker.sh');remote('chmod 755 '+stage+'/worker.sh')
        atomic_json(work/'submitted.json',dict(status='pending',at=time.time(),stage=stage))
        output=remote('source /afs/ihep.ac.cn/soft/common/sysgroup/hep_job/setup_hepjob.sh; cd '+stage+
            '; hep_sub ./worker.sh -g lhaaso -os AlmaLinux9 -m 2000 -cpu 1 -wt short -schedd schedd11 -o probe.out -e probe.err',attempts=1)
        atomic_json(work/'submission_output.json',dict(output=output))
        m=re.search(r'cluster\s+(\d+)',output,re.I)
        if not m:raise RuntimeError('ambiguous diagnostic submission')
        cluster=int(m[1]);atomic_json(work/'submitted.json',dict(cluster=cluster,at=time.time(),stage=stage))
        record=remote_py('''import json,subprocess
b=['-name','scheduler@schedd11.ihep.ac.cn'];j=%r
a=json.loads(subprocess.check_output(['condor_q']+b+[j,'-json','-attributes','Owner,Cmd,Requirements'],text=True))[0]
assert a['Owner']=='liushijie' and a['Cmd']==%r+'/./worker.sh'
r=a['Requirements'];assert r.startswith('/Expr(') and r.endswith(')/')
n='('+r[6:-2]+') && (TARGET.Machine != "lhws183.ihep.ac.cn") && (TARGET.Machine != "bws0995.ihep.ac.cn")'
subprocess.run(['condor_qedit']+b+[j,'Requirements',n],check=True,stdout=subprocess.PIPE)
print(json.dumps(dict(job=j,original=r,updated=n)))
''' % (str(cluster)+'.0',stage))
        atomic_json(work/'scheduler_policy.json',json.loads(record))
    if 'cluster' not in json.loads((work/'submitted.json').read_text()):raise RuntimeError('inspect ambiguous diagnostic submission')
    start=time.monotonic()
    while remote_py('from pathlib import Path; print(int(Path('+repr(stage+'/result.json')+').exists()))').strip()!='1':
        if time.monotonic()-start>1800:raise RuntimeError('diagnostic timeout; inspect scheduler')
        time.sleep(20)
    digest=remote('sha256sum '+stage+'/result.json').split()[0]
    transfer(REMOTE+':'+stage+'/',str(work/'source')+'/')
    r=json.loads((work/'source/result.json').read_text())
    if sha256(work/'source/result.json')!=digest or r['script_sha256']!=sha256(script) or r['spec_sha256']!=sha256(work/'spec.json'):
        raise ValueError('diagnostic evidence binding mismatch')
    atomic_json(work/'completed.json',dict(at=time.time(),result_sha256=digest,stage=stage))
    print('DIAGNOSTIC_COMPLETE',digest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);a=p.parse_args();run(a.state)
