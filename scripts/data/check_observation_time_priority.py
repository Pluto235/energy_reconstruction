import json,sys,time
from pathlib import Path
state=Path('/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time-v3')
if (state/'monitor_pause.json').exists():
 print('MONITOR_PAUSED_BY_USER');sys.exit(0)
sys.path.insert(0,str(state/'deployments/de2c69083b0f9fefed4f7e1c05c64a13dd69993e/scripts/data'))
from sync_observation_time import atomic_json,read_json,remote
c=read_json(state/'config.json');baseline=read_json(state/'observation_priority_baseline.json')
override=state/'user_priority_override.json'
if override.exists():
 policy=read_json(override);desired=policy['active_workers']
 assert desired in (0,1,2) and policy['reason']=='January first; February explicitly paused by user'
 feb_pause=read_json(Path('/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-february/monitor_pause.json'))
 for process in feb_pause['processes']:
  status=Path('/proc')/str(process['pid'])/'status'
  assert not status.exists() or status.read_text().split('State:')[1].splitlines()[0].strip().startswith('T'), 'February receiver resumed unexpectedly'
 remote(c,'concurrency',{'active_workers':desired})
 entry=dict(at=time.time(),active_workers=desired,policy='user_january_first',changed=desired!=baseline['active_workers'])
 baseline.update(active_workers=desired);baseline['history'].append(entry)
 atomic_json(state/'observation_priority_baseline.json',baseline)
 print(json.dumps(entry));sys.exit(0)
feb=Path('/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-february')
p=read_json(feb/'progress.json');now=time.time();current=baseline['active_workers'];desired=current
rows=[read_json(x) for x in (feb/'receipts').glob('*.json')]
rates=[sum(r['bytes'] for r in rows if now-900*(i+1)<=r['confirmed_at']<now-900*i)/900 for i in (0,1)]
assert 0<=current<=2 and all(x>=0 for x in rates)
if p['files']==600:
 desired=2
elif now-baseline['last_throttle_at']>=1800 and baseline['baseline_bytes_per_second']>0 and all(r<.8*baseline['baseline_bytes_per_second'] for r in rates):
 desired=max(0,current-1)
if desired!=current:
 remote(c,'concurrency',{'active_workers':desired})
 baseline.update(active_workers=desired,last_throttle_at=now)
entry=dict(at=now,observation_files=p['files'],window_bytes_per_second=rates,active_workers=desired,changed=desired!=current)
baseline['history'].append(entry)
atomic_json(state/'observation_priority_baseline.json',baseline)
print(json.dumps(entry))
