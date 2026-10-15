"""Mac SSH relay: <=100 summaries staged per batch, no ROOT reads on Mac.
Uses existing ihep/ETO SSH aliases; resume requires the same immutable run/code.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

SSH=['ssh','-o','BatchMode=yes','-o','RemoteCommand=none','-o','RequestTTY=no','-o','ConnectTimeout=30','-o','ServerAliveInterval=30']
STAGE='/mnt/newdisk/shijie/WCDA_simulation/v7_friends/.staging/summary'
SOURCE='/eos/user/w/wcdarec/MC/eventout/Egr/data/T043g'
ON_ETO=False

def remote(code):
    return subprocess.check_output(['python3','-c',code] if ON_ETO else SSH+['ETO','python3 -c '+shlex.quote(code)],text=True)

def receipt_names(run):
    return set(json.loads(remote('import json;from pathlib import Path; print(json.dumps([p.parent.name+"_eventout.root" for p in Path('+repr(run)+').glob("files/*/receipt.json")]))')))

def main():
    global ON_ETO
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--on-eto',action='store_true');args=p.parse_args();ON_ETO=args.on_eto
    inv=json.loads(args.inventory.read_text());items=inv['files'];done=receipt_names(args.run)
    todo=[x for x in items if x['name'] not in done]
    for start in range(0,len(todo),100):
        batch=todo[start:start+100];names=[i['summary_name'] for i in batch]
        # A failed tar never gets a ready marker; retries overwrite only this bounded temporary batch.
        temp=STAGE+'.incoming'
        remote('from pathlib import Path;Path('+repr(temp)+').mkdir(parents=True,exist_ok=True)')
        pipes=[]
        for group in [names[i::4] for i in range(4)]:
            if not group: continue
            source=subprocess.Popen(SSH+(['-o','HostName=lxlogin004.ihep.ac.cn','-o','HostKeyAlias=lxlogin.ihep.ac.cn'] if ON_ETO else [])+['liushijie@lxlogin.ihep.ac.cn' if ON_ETO else 'ihep','tar -C '+shlex.quote(SOURCE)+' -cf - -- '+ ' '.join(map(shlex.quote,group))],stdout=subprocess.PIPE)
            dest=subprocess.Popen(['tar','-C',temp,'-xf','-'] if ON_ETO else SSH+['ETO','tar -C '+shlex.quote(temp)+' -xf -'],stdin=source.stdout)
            source.stdout.close();pipes.append((source,dest))
        codes=[(dest.wait(),source.wait()) for source,dest in pipes]
        if any(a or b for a,b in codes):raise RuntimeError(f'tar failed {codes}; staging left for retry')
        remote('''import hashlib,os
from pathlib import Path
names='''+repr(names)+'''
stage=Path('''+repr(STAGE)+''');temp=Path('''+repr(temp)+''')
for n in names:
 p=temp/n;dest=stage/n
 if not p.is_file() or p.stat().st_size==0:raise RuntimeError('empty transfer '+n)
 if dest.exists():
  if hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():raise RuntimeError('staged source changed '+n)
  p.unlink()
 else:os.rename(p,dest)
 (stage/(n+'.ready')).touch()
''')
        print(json.dumps(dict(staged=start+len(batch),remaining=len(todo)-start-len(batch))),flush=True)
        deadline=time.monotonic()+3600
        wanted={x['name'] for x in batch}
        while not wanted.issubset(receipt_names(args.run)):
            if time.monotonic()>deadline:raise TimeoutError('No completed batch after one hour; inspect Slurm')
            time.sleep(3)
        remote('''import json,hashlib
from pathlib import Path
stage=Path('''+repr(STAGE)+''');run=Path('''+repr(args.run)+''')
for n in '''+repr(names)+''':
 p=stage/n;r=json.loads((run/'files'/n.removesuffix('.root')/'receipt.json').read_text())
 if hashlib.sha256(p.read_bytes()).hexdigest()!=r.get('summary_sha256'):raise RuntimeError('receipt/hash missing; retain staging '+n)
 p.unlink();(stage/(n+'.ready')).unlink()
''')
        print(json.dumps(dict(receipted_and_cleaned=start+len(batch),total=len(todo))),flush=True)
    print('All inventory files have receipts; scientific success is in summary.json.',flush=True)

if __name__=='__main__':main()
