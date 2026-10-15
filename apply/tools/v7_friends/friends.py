"""Exact MC summary matching and atomic, unfiltered friend publication (Slurm only)."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import uproot

MATCH_FIELDS = ('mc_energy', 'mc_dangle', 'mc_weight', 'mc_theta', 'theta', 'pincness')
NATIVE = ('poolused', 'nq03t30s0', 'nq03t30s1', 'nq03t30s2', 'irun', 'ies', 'iseq', 'ievent')
COMMON = ('ccindex','chi2','dcedge','dcedgepool','f5w','fitstat','istationcore',
          'mc_dangle','mc_dcore','mc_energy','mc_phi','mc_pid','mc_theta','mc_weight',
          'mc_xc','mc_yc','nfit','nfit0','phi','pincness','rmds','theta','xc','yc')
EDGES = [30,60,100,200,300,500,800,2000]
CONTRACT = dict(version=2, match_fields=MATCH_FIELDS, matching='exact float32 six-field bytes; all keys unique',
                identity_limit='Content correspondence is not independent physical event ID authentication',
                chi2_policy='diagnostic only, explicitly authorized by user 2026-10-15',
                float_policy='exact equality after float64-to-float32 cast; no tolerance',
                filters='none', nhit='native unprefixed nq03t30s0+1+2, checked int64 sum',
                tree='v7_friend', branch_prefix='v7_', extension='.v7friend', nhit_edges=EDGES,
                raw_hit_check='not performed or required; native Nhit authoritative per user 2026-10-15',
                upstream_time_rule='abs(vt)<30 per user; not independently established from stored float32 hits',
                rmds_boundary_policy='summary decision authoritative for rounding-only differences; preserve eventout')

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p, value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def stamp(p):
    s=p.stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)

def histogram(x):
    v,c=np.unique(x,return_counts=True)
    return {str(int(a)):int(b) for a,b in zip(v,c)}

def match_rows(events, summary):
    def keys(t):
        a=np.ascontiguousarray(np.column_stack([np.asarray(t[k],dtype='<f4') for k in MATCH_FIELDS]))
        return a.view('V24').ravel(), int(np.sum(~np.isfinite(a).all(axis=1)))
    e,ne=keys(events); s,ns=keys(summary)
    stats=dict(nonfinite_event_rows=ne,nonfinite_summary_rows=ns,
               event_duplicate_rows=len(e)-len(np.unique(e)), summary_duplicate_rows=len(s)-len(np.unique(s)),
               missing=0,ambiguous=0,reused=0)
    lookup={}
    for j,k in enumerate(s): lookup.setdefault(bytes(k),[]).append(j)
    matches=[lookup.get(bytes(k),[]) for k in e]
    stats['missing']=sum(not a for a in matches)
    stats['ambiguous']=sum(len(a)>1 for a in matches)
    unique=[a[0] for a in matches if len(a)==1]
    stats['reused']=len(unique)-len(set(unique))
    if any(stats.values()): return None,stats
    return np.asarray(unique,dtype='i8'),stats

def compare(x,y):
    if x.ndim!=1 or y.ndim!=1 or x.shape!=y.shape: raise ValueError('non-scalar or unequal common field')
    if x.dtype.kind in 'iu' and y.dtype.kind in 'iu':
        ok=x==y; finite=np.ones(len(x),dtype=bool)
    elif x.dtype.kind=='f' and y.dtype.kind=='f' and {x.dtype.itemsize,y.dtype.itemsize}<={4,8}:
        finite=np.isfinite(x)&np.isfinite(y)
        dtype=x.dtype if x.dtype.itemsize<=y.dtype.itemsize else y.dtype
        ok=finite&(x.astype(dtype)==y.astype(dtype))
    else: raise ValueError(f'unsupported common types {x.dtype} {y.dtype}')
    exact=x==y; d=x.astype('f8')-y.astype('f8')
    return dict(event_dtype=str(x.dtype),summary_dtype=str(y.dtype),exact=int(np.sum(exact&finite)),
                rounding_only=int(np.sum(ok&~exact)),outside_rounding=int(np.sum(~ok)),
                nonfinite=int(np.sum(~finite)),difference_quantiles=np.quantile(d[finite],[0,.25,.5,.75,1]).tolist() if finite.any() else [],
                bad_entries=np.flatnonzero(~ok).tolist())

def native_sum(s):
    cols=[s[f'nq03t30s{i}'] for i in range(3)]
    for a in cols:
        if a.dtype.kind not in 'iu' or a.ndim!=1 or np.any(a<0) or np.any(a>np.iinfo('i8').max):
            raise ValueError('invalid native integer counts')
    total=np.zeros(len(cols[0]),dtype='i8')
    for a in cols:
        a=a.astype('i8')
        if np.any(a>np.iinfo('i8').max-total): raise ValueError('int64 sum overflow')
        total+=a
    return total

def publish(target, writer):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError(target)
    tmp=Path(tempfile.mkdtemp(prefix='.'+target.name+'.',dir=target.parent))
    try:
        writer(tmp)
        if target.exists(): raise FileExistsError(target)
        os.rename(tmp,target)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)

def examples(indices, rows, x=None,y=None):
    result=[]
    for i in indices[:5]:
        r=dict(event_entry=int(i),summary_entry=int(rows[i]))
        if x is not None:
            r.update(event_value=float(x[i]) if np.isfinite(x[i]) else str(x[i]),
                     summary_value=float(y[i]) if np.isfinite(y[i]) else str(y[i]))
        result.append(r)
    return result

def process(item, run, base, stage):
    run,base,stage=map(Path,(run,base,stage)); name=item['name']; stem=name.removesuffix('_eventout.root')
    ep=base/name; sp=stage/item['summary_name']; target=run/'files'/stem
    if target.exists():
        r=json.loads((target/'receipt.json').read_text())
        if r['source_stat']!=stamp(ep) or r['code_sha256']!=sha256(__file__):
            raise RuntimeError('Existing receipt input/code mismatch: '+name)
        return r['status']
    r=dict(file=name,source_stat=stamp(ep),source_path=str(ep),summary_source='/eos/user/w/wcdarec/MC/eventout/Egr/data/T043g/'+sp.name,
           code_sha256=sha256(__file__),slurm_job=os.environ.get('SLURM_JOB_ID'),status='quarantined',events=None,
           failures=[],readback_verified=False,contract=CONTRACT)
    payload=None
    try:
        if any(r['source_stat'][k]!=item[k] for k in ('bytes','mtime_ns','device','inode')): raise ValueError('input changed since inventory')
        if ep.is_symlink(): raise ValueError('unexpected symlink')
        if not sp.exists(): raise FileNotFoundError('summary_source_missing')
        ss=stamp(sp); r['event_sha256']=sha256(ep);r['summary_sha256']=sha256(sp)
        with uproot.open(ep) as ef,uproot.open(sp) as sf:
            et,st=ef['t_eventout'],sf['wcdaevent'];r['events']=et.num_entries;r['summary_events']=st.num_entries
            r['schemas']={'event':et.typenames(),'summary':st.typenames()}
            actual=sorted(set(et.keys())&set(st.keys()))
            if actual!=sorted(COMMON): raise ValueError('common schema changed; needs review')
            r['direct_identity_fields_in_event']=[k for k in NATIVE[-4:] if k in et.keys()]
            if r['direct_identity_fields_in_event']: raise ValueError('new direct identity fields; needs review')
            e=et.arrays(list(COMMON),library='np'); s=st.arrays(list(COMMON)+list(NATIVE),library='np')
        rows,r['matching']=match_rows(e,s)
        if rows is None: raise ValueError('nonunique_or_missing_match')
        unused=np.setdiff1d(np.arange(st.num_entries),rows)
        r['unused_summary_entries']=unused.tolist()
        if len(unused):
            r['unused_summary_ids']=[{k:int(s[k][j]) for k in NATIVE[-4:]} for j in unused]
            r['failures'].append('unused_summary_origin_not_yet_explained')
        r['common_fields']={}
        for k in COMMON:
            x,y=e[k],s[k][rows]; c=compare(x,y); bad=c.pop('bad_entries'); c['examples']=examples(bad,rows,x,y)
            r['common_fields'][k]=c
            if bad and k!='chi2': r['failures'].append('common_field:'+k)
        r['boundaries']={}
        boundaries={'rmds_le20':('rmds',lambda a:a<=20),
                    'theta_le50deg':('theta',lambda a:a<=np.deg2rad(50)),
                    'theta_lt1e-5':('theta',lambda a:a<1e-5),
                    'theta_open_interval':('theta',lambda a:(a>.0247254)&(a<.0247256))}
        for label,(k,fn) in boundaries.items():
            x,y=e[k].astype('f8'),s[k][rows].astype('f8'); bad=np.flatnonzero(fn(x)!=fn(y))
            r['boundaries'][label]=dict(changed=len(bad),examples=examples(bad,rows,x,y))
            if label=='rmds_le20':
                r['boundaries'][label]['decision_source']='native summary; eventout unchanged'
                r['boundaries'][label]['accepted_rounding_flips']=len(bad) if r['common_fields'][k]['outside_rounding']==0 else 0
            elif len(bad):r['failures'].append('boundary:'+label)
        native={k:s[k][rows] for k in NATIVE};r['native_ranges']={}
        for k,a in native.items():
            if a.dtype.kind not in 'iu' or a.ndim!=1: raise ValueError('noninteger native '+k)
            r['native_ranges'][k]=dict(dtype=str(a.dtype),min=int(a.min()) if len(a) else None,max=int(a.max()) if len(a) else None)
        nh=native_sum(native);r['poolused']=histogram(native['poolused'])
        r['poolused_unexpected_examples']=examples(np.flatnonzero((native['poolused']<0)|(native['poolused']>7)),rows,native['poolused'],native['poolused'])
        r['nhit_validation']='native integer fields and overflow-checked sum; no raw-hit recomputation'
        r['nhit_bins']=[]
        for i,lo in enumerate(EDGES):
            hi=EDGES[i+1] if i+1<len(EDGES) else None;mask=(nh>=lo)&((nh<hi) if hi is not None else True)
            r['nhit_bins'].append(dict(lower=lo,upper=hi,events=int(mask.sum()),compared_events=0))
        r['nhit_below30']=dict(events=int(np.sum(nh<30)),compared_events=0)
        payload={f'v7_{k}':v for k,v in native.items()}
        payload.update(v7_nhit_authoritative=nh,v7_eventout_entry=np.arange(len(rows),dtype='i8'),
                       v7_summary_entry=rows,v7_match_status=np.ones(len(rows),dtype='u1'))
        if stamp(ep)!=r['source_stat'] or stamp(sp)!=ss: raise ValueError('input mutated during read')
        if not r['failures']: r['status']='verified'
    except Exception as exc:
        r['failures'].append(type(exc).__name__+': '+str(exc))
    def write(tmp):
        if r['status']=='verified':
            fp=tmp/(stem+'.v7friend')
            with uproot.recreate(fp) as f:
                f.mktree('v7_friend',{k:v.dtype for k,v in payload.items()});f['v7_friend'].extend(payload)
                f['v7_metadata']=json.dumps({k:r[k] for k in ('file','event_sha256','summary_sha256','code_sha256','contract')})
            with uproot.open(fp) as f:
                t=f['v7_friend'];a=t.arrays(library='np')
                if t.num_entries!=r['events'] or set(a)!=set(payload): raise ValueError('readback rows/branches')
                for k,v in payload.items():
                    if a[k].dtype!=v.dtype or not np.array_equal(a[k],v): raise ValueError('readback mismatch '+k)
                meta=json.loads(str(f['v7_metadata']))
                if meta['event_sha256']!=r['event_sha256']: raise ValueError('readback metadata')
            r['readback_verified']=True;r['friend_sha256']=sha256(fp);r['friend_name']=fp.name
            r['field_coverage']={k:len(v) for k,v in payload.items()}
        dump(tmp/'receipt.json',r)
    try: publish(target,write)
    except Exception as exc:
        if target.exists(): raise
        r['status']='quarantined';r['readback_verified']=False;r['failures'].append('publication: '+repr(exc))
        publish(target,lambda p:dump(p/'receipt.json',r))
    return r['status']

def summarize(run, inventory):
    totals=dict(target_files=len(inventory),target_events_known=0,verified_files=0,verified_events=0,
                quarantined_files=0,quarantined_events_known=0,pending_files=0,readback_verified_files=0,
                poolused_events={},poolused_files={},matching={},common_fields={},boundaries={},
                nhit_difference_native_minus_hits={},nhit_bins=[dict(lower=lo,upper=EDGES[i+1] if i+1<len(EDGES) else None,events=0,mismatches=0) for i,lo in enumerate(EDGES)],
                nhit_below30=dict(events=0,mismatches=0),failures=[],field_coverage={},
                diagnostic_scope='All fully matched processed files, including quarantined files. Published coverage separately counted.',contract=CONTRACT)
    def add(d,other):
        for k,v in other.items(): d[k]=d.get(k,0)+v
    for item in inventory:
        p=run/'files'/item['name'].removesuffix('_eventout.root')/'receipt.json'
        if not p.exists(): totals['pending_files']+=1;continue
        r=json.loads(p.read_text()); n=r['events'] or 0;totals['target_events_known']+=n
        if r['status']=='verified':
            totals['verified_files']+=1;totals['verified_events']+=n
            totals['readback_verified_files']+=int(r['readback_verified']);add(totals['field_coverage'],r['field_coverage'])
        else:
            totals['quarantined_files']+=1;totals['quarantined_events_known']+=n
            totals['failures'].append(dict(file=r['file'],events=r['events'],reasons=r['failures']))
        add(totals['matching'],r.get('matching',{}));add(totals['poolused_events'],r.get('poolused',{}));add(totals['poolused_files'],{k:1 for k in r.get('poolused',{})})
        add(totals['nhit_difference_native_minus_hits'],r.get('nhit_difference_native_minus_hits',{}))
        for k,v in r.get('common_fields',{}).items():
            d=totals['common_fields'].setdefault(k,dict(exact=0,rounding_only=0,outside_rounding=0,nonfinite=0,examples=[]))
            add(d,{x:v[x] for x in ('exact','rounding_only','outside_rounding','nonfinite')})
            if len(d['examples'])<5:d['examples']+=( [dict(file=r['file'],**x) for x in v['examples']][:5-len(d['examples'])])
        add(totals['boundaries'],{k:v['changed'] for k,v in r.get('boundaries',{}).items()})
        for a,b in zip(totals['nhit_bins'],r.get('nhit_bins',[])):add(a,{k:b.get(k,0) for k in ('events','mismatches')})
        add(totals['nhit_below30'],r.get('nhit_below30',{}))
    totals['all_target_poolused_proven_7']=(totals['verified_files']==totals['target_files'] and set(totals['poolused_events'])=={'7'})
    tmp=run/'summary.json.tmp';dump(tmp,totals);os.replace(tmp,run/'summary.json')
    return totals

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--once',action='store_true');p.add_argument('--summarize',action='store_true')
    args=p.parse_args(); inv=json.loads(args.inventory.read_text());items=inv['files'];base=inv['input_root']
    args.run.mkdir(parents=True,exist_ok=True)
    if args.summarize: summarize(args.run,items);return
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('ROOT processing must run in Slurm')
    import fcntl
    with (args.run/'writer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract_path=args.run/'contract.json';code=sha256(__file__)
        if contract_path.exists():
            if json.loads(contract_path.read_text())['code_sha256']!=code: raise RuntimeError('version changed; use new run directory')
        else:dump(contract_path,dict(contract=CONTRACT,code_sha256=code,git_commit=os.environ.get('V7_GIT_COMMIT','uncommitted'),inventory_sha256=sha256(args.inventory)))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while True:
                pending=[i for i in items if not (args.run/'files'/i['name'].removesuffix('_eventout.root')/'receipt.json').exists()]
                ready=[i for i in pending if (args.stage/(i['summary_name']+'.ready')).exists()]
                if ready:
                    results=list(pool.map(process,ready,[str(args.run)]*len(ready),[base]*len(ready),[str(args.stage)]*len(ready)))
                    print(json.dumps(dict(processed=len(ready),statuses=dict(Counter(results)))),flush=True)
                    summarize(args.run,items)
                if args.once or not pending:break
                if not ready: time.sleep(2)
        summary=summarize(args.run,items)
        print(json.dumps({k:summary[k] for k in ('verified_files','quarantined_files','pending_files')}),flush=True)

if __name__=='__main__':main()
