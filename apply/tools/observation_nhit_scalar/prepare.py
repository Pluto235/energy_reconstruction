"""Freeze the original input universe and adopt unchanged durable receipts in Slurm."""
import argparse
import json
from pathlib import Path
import time
from common import atomic_json, require_slurm, sha256, stamp
from validate import adopt

BASE=Path('/mnt/newdisk/shijie/WCDA_obs_friends/nhit')
INPUT=Path('/mnt/newdisk/shijie/WCDA_obs')
OLD=BASE/'manifests/20261015_january_v5'
RESTORE=Path('/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/receipts')
PILOT=['Esg20220101_00.root','Esg20220116_00.root','Esg20220131_00.root']


def freeze(state):
    require_slurm()
    old=json.loads((OLD/'manifest.json').read_text())
    actual=sorted(p.name for p in INPUT.glob('01??/Esg202201*.root'))
    restored=sorted(p.name[:-5] for p in RESTORE.glob('Esg202201*.root.json'))
    names=[r['name'] for r in old['records']]
    if actual!=restored or actual!=names or len(names)!=681 or len({n[7:11] for n in names})!=31:
        raise ValueError('INPUT_CHANGED original/current/legacy manifest universe')
    records=[]
    for row in old['records']:
        name=row['name']; target=INPUT/name[7:11]/name
        receipt=RESTORE/(name+'.json'); recovered=json.loads(receipt.read_text())
        if stamp(target)!=row['input'] or recovered['bytes']!=row['input']['bytes'] or recovered['entries']!=row['input']['entries']:
            raise ValueError('INPUT_CHANGED '+name)
        if recovered.get('record',{}).get('source',{}).get('entries',row['source_entries'])!=row['source_entries']:
            raise ValueError('original raw entry count mismatch')
        records.append(dict(row,target=str(target),restore_receipt=str(receipt),restore_receipt_sha256=sha256(receipt)))
    result=dict(records=records,old_manifest=str(OLD/'manifest.json'),old_manifest_sha256=sha256(OLD/'manifest.json'),
                total_events=sum(r['input']['entries'] for r in records),total_files=681,days=31)
    p=state/'manifest.json'
    if p.exists() and json.loads(p.read_text())!=result:
        raise ValueError('INPUT_CHANGED new snapshot')
    atomic_json(p,result)
    print('FROZEN',len(records),result['total_events'],flush=True)
    return records


def prepare(state):
    require_slurm(); records=freeze(state)
    adopted=[]
    for task in records:
        if (OLD/(task['name']+'.receipt.json')).exists():
            target=BASE/'full'/task['name'][7:11]/task['name'].replace('.root','.nhit.root')
            adopted.append(adopt(state,OLD,task,target))
            print('ADOPTED',task['name'],flush=True)
    atomic_json(state/'adoption.json',dict(files=len(adopted),events=sum(r['entries'] for r in adopted),
        bytes=sum(r['bytes'] for r in adopted),finished_at=time.time(),old_run=str(OLD)))
    atomic_json(state/'tasks.json',dict(pilot=[r for n in PILOT for r in records if r['name']==n],
        production=[r for r in records if not (state/'receipts'/(r['name']+'.json')).exists()]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',type=Path);a=p.parse_args();prepare(a.state)
