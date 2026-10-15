"""Slurm operational guard for frozen source jobs and site-native queue classes."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent/'observation_nhit_scalar'))
from common import atomic_json,require_slurm
from transport import remote_py


def supervise(state):
    require_slurm()
    stage=json.loads((state/'config.json').read_text())['stage']
    # Exact classes from site-generated jobs 3742194 (short) and 3739735 (default).
    # Fresh hep_clus confirms 30 min / 15 h; source checkpoints after 8 h.
    classes=dict(pilot=dict(AccountingGroup='COMMON.short.lhaaso.liushijie',
        AcctGroup='COMMON.short.lhaaso',WallTime='short',HepJob_Walltime='short',HepJob_VirtualGroup='short'),
        production=dict(AccountingGroup='LHAASO.lhaaso.default.liushijie',
        AcctGroup='LHAASO.lhaaso.default',WallTime='default',HepJob_Walltime='default',HepJob_VirtualGroup='lhaaso'))
    while not (state/'finished.json').exists():
        p=state/'controller_error.json'
        error=json.loads(p.read_text()) if p.exists() else None
        stop=bool(error and ('INPUT_CHANGED' in error['error'] or 'frozen manifest' in error['error']))
        record=remote_py('''import json,subprocess
b=['-name','scheduler@schedd11.ihep.ac.cn'];stage=%r;classes=%r;stop=%r
raw=subprocess.check_output(['condor_q']+b+['-constraint','Owner == "liushijie"','-json','-attributes','Owner,Cmd,Args,JobStatus,ClusterId,ProcId,NumJobStarts,Requirements,AccountingGroup,AcctGroup,WallTime,HepJob_Walltime,HepJob_VirtualGroup'],text=True)
jobs=[j for j in json.loads(raw.strip() or '[]') if j.get('Cmd','').startswith(stage+'/') and j['JobStatus'] in [1,2]]
actions=[]
for j in jobs:
 assert j['Owner']=='liushijie' and j['Cmd']==stage+'/./worker.sh'
 args=j['Args'].split();assert len(args)==3 and args[0]==stage and args[1] in classes and int(args[2]) in range(4)
 job=str(j['ClusterId'])+'.'+str(j['ProcId'])
 if stop:
  subprocess.run(['condor_hold']+b+[job,'-reason','Frozen observation input changed; expansion blocked'],check=True,stdout=subprocess.PIPE)
  actions.append(dict(job=job,action='input_change_hold',before=j));continue
 desired=classes[args[1]]
 if j['JobStatus']==1 and j.get('NumJobStarts',0)==0 and any(j.get(k)!=v for k,v in desired.items()):
  for k,v in desired.items():subprocess.run(['condor_qedit']+b+[job,k,json.dumps(v)],check=True,stdout=subprocess.PIPE)
  actions.append(dict(job=job,action='site_walltime_class',before=j,updated=desired,requirements_unchanged=True))
print(json.dumps(actions))
''' % (stage,classes,stop))
        actions=json.loads(record)
        if actions:
            atomic_json(state/'scheduler_policy_history'/(str(time.time_ns())+'.json'),dict(at=time.time(),actions=actions))
            print(record,flush=True)
        if stop:
            atomic_json(state/'input_change_hold.json',dict(error=error,actions=actions,at=time.time()))
            return
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);a=p.parse_args();supervise(a.state)
