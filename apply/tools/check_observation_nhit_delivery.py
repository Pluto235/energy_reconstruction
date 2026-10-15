"""Slurm closure check: completion bindings, source acknowledgments and queues."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent/'observation_nhit_scalar'))
from common import atomic_json, require_slurm, sha256
from transport import quota, remote_py


def check(state):
    require_slurm()
    complete = json.loads((state/'completed.json').read_text())
    assert (complete['files'], complete['events']) == (681, 69279369)
    for key, filename in [('final_audit_sha256', 'final_audit.json'),
                          ('file_catalog_sha256', 'file_catalog.csv')]:
        assert complete[key] == sha256(state/filename), filename
    audit = json.loads((state/'final_audit.json').read_text())
    assert audit['status'] == 'complete' and not audit['failures']
    assert audit['user_confirmed_overrides']['events'] == 1803
    with (state/'failed_files.csv').open() as f:
        assert not list(csv.DictReader(f))
    bindings = json.loads((state/'repairs.json').read_text())
    states = [state] + [Path(b['state']) for b in bindings]
    configs = [json.loads((s/'config.json').read_text()) for s in states]
    stages = [c['stage'] for c in configs]
    before = quota()
    raw = remote_py('''import json,subprocess
from pathlib import Path
stages=%r
raw=subprocess.check_output(['condor_q','-name','scheduler@schedd11.ihep.ac.cn','-constraint','Owner == "liushijie"','-json','-attributes','ClusterId,ProcId,Owner,Cmd,Args,JobStatus'],text=True)
jobs=[j for j in json.loads(raw.strip() or '[]') if any(j.get('Cmd','').startswith(s+'/') for s in stages)]
inventory={}
for stage in stages:
 p=Path(stage)
 inventory[stage]=dict(acks={f.parent.name:json.loads(f.read_text()) for f in p.glob('production/*/ack.json')},
  root_files=[str(f.relative_to(p)) for f in p.rglob('*.root') if f.is_file()],
  partials=[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and ('.partial' in f.name or f.name=='reservation.json')],
  apparent_bytes=int(subprocess.check_output(['du','-sb',stage],text=True).split()[0]))
print(json.dumps(dict(jobs=jobs,stages=inventory)))
''' % stages)
    inventory = json.loads(raw)
    atomic_json(state/'audit/delivery_source_inventory.json', inventory)
    assert not inventory['jobs'], 'source jobs still present'
    for info in inventory['stages'].values():
        assert not info['root_files'] and not info['partials'], 'source queue not empty'
    by_run = {c['run_id']: c['stage'] for c in configs}
    count = 0
    for p in (state/'receipts').glob('*.json'):
        r = json.loads(p.read_text())
        if r['inherited']:
            continue
        stage = by_run[r['source']['provenance']['run_id']]
        ack = inventory['stages'][stage]['acks'][r['name']]
        assert ack['final_sha256'] == r['sha256'] and ack['candidate_sha256'] == r['candidate_sha256']
        assert ack['accepted_entries'] == r['entries'] and ack['eto_path'] == r['path']
        count += 1
    assert count == 626
    result = dict(status='passed', at=time.time(), new_receipts_acknowledged=count,
                  source_jobs=0, source_root_files=0, source_partials=0,
                  completed_sha256=sha256(state/'completed.json'),
                  source_inventory_sha256=sha256(state/'audit/delivery_source_inventory.json'),
                  quota_before=before, quota_after=quota(),
                  artifacts={n: sha256(state/n) for n in ['final_audit.json','file_catalog.csv',
                      'file_timings.csv','operation_history.json','failed_files.csv','repairs.json']})
    atomic_json(state/'audit/delivery_check.json', result)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('quota')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('state', type=Path)
    check(parser.parse_args().state)
