"""Retire exactly the legacy Nhit jobs; preserve old manifests and all audit files."""
import json
from pathlib import Path
import subprocess
from transport import remote_py

STATE = Path('/mnt/newdisk/shijie/WCDA_obs_friends/nhit/manifests/20261015_january_v6_scalar/audit')
OLD = '/scratchfs2/lhaaso/liushijie/WCDA_nhit_20261015_january_v5'
code = '''import json, subprocess
stage = %r
base=['-name','scheduler@schedd11.ihep.ac.cn']
raw=subprocess.check_output(['condor_q']+base+['-constraint','Owner == "liushijie" && ClusterId >= 3739000','-json'],text=True)
rows=json.loads(raw.strip() or '[]')
jobs=[r for r in rows if r.get('Cmd','').startswith(stage+'/')]
for r in jobs:
    assert r['Owner']=='liushijie' and r['Cmd']==stage+'/./worker.sh'
    assert r['Args'].startswith(stage+' Esg202201') and r['Args'].endswith('.root')
    assert r['JobStatus']==5 and r.get('CurrentHosts',0)==0
print(json.dumps(jobs))
''' % OLD
before=json.loads(remote_py(code))
STATE.mkdir(parents=True,exist_ok=True)
if not (STATE/'legacy_condor_before.json').exists():
    (STATE/'legacy_condor_before.json').write_text(json.dumps(before,indent=2))
for row in before:
    job=str(row['ClusterId'])+'.'+str(row['ProcId'])
    action = remote_py('import subprocess; subprocess.run('+repr(['condor_rm','-name','scheduler@schedd11.ihep.ac.cn',job])+',check=True)', attempts=1)
    (STATE/('retire-'+job+'.txt')).write_text(action)
after=json.loads(remote_py(code))
(STATE/'legacy_condor_after.json').write_text(json.dumps(after,indent=2))
assert not after
print('retired', [r['ClusterId'] for r in before])
