"""Slurm validation and no-replace publication of scalar friends."""
import json
import os
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot
from common import KEYS, EDGES, atomic_json, digest_column, histogram, sha256, stamp, sync_dir, unique_keys
from source import CONTRACT, readback, write_scalar

SCALARS = KEYS+['input_entry','ihep_entry','nv','nhit','poolused','segment_index']


def align(target, candidate):
    t, c = unique_keys(target), unique_keys(candidate)
    order = np.argsort(c); pos = np.searchsorted(c[order],t)
    rows = np.full(len(t),-1,dtype='i8')
    if len(c):
        hit = (pos<len(c)) & (c[order[np.minimum(pos,len(c)-1)]]==t)
        rows[hit] = order[pos[hit]]
    used = np.zeros(len(c),dtype=bool); used[rows[rows>=0]]=True
    return rows, np.flatnonzero(~used)


def check_scalars(data, segments, raw_entries, ordered_source=False):
    n = len(data['nv'])
    if any(len(v)!=n for v in data.values()):
        raise ValueError('scalar length mismatch')
    unique_keys(data)
    entry=data['ihep_entry']
    if len(np.unique(entry))!=n or np.any(entry<0) or np.any(entry>=raw_entries):
        raise ValueError('duplicate/out-of-bounds source entry')
    if ordered_source and np.any(np.diff(entry)<=0):
        raise ValueError('source PINC order changed')
    if np.any(data['nv']<0) or np.any(data['nhit']<0) or np.any(data['nhit']>data['nv']):
        raise ValueError('invalid hit counts')
    ids=data['segment_index']
    if np.any(ids<0) or np.any(ids>=len(segments)):
        raise ValueError('invalid segment index')
    if [r['segment_index'] for r in segments] != list(range(len(segments))):
        raise ValueError('segment table order')
    if len({tuple(r[k] for k in KEYS[:3]) for r in segments}) != len(segments):
        raise ValueError('duplicate segment identity')
    for r in segments:
        if not all(r.get(k) for k in ['path','uuid','source_filename','header_sha256']):
            raise ValueError('missing header provenance')
        if not 0<=r['poolused']<=7:
            raise ValueError('poolused mask')
    for k in KEYS[:3]+['poolused']:
        if not np.array_equal(np.array([r[k] for r in segments],dtype='i4')[ids],data[k]):
            raise ValueError('segment correspondence '+k)
    if 'input_entry' in data and not np.array_equal(data['input_entry'],np.arange(n,dtype='i8')):
        raise ValueError('non-contiguous input_entry')


def stats(data, segments):
    return dict(events=len(data['nv']),matching_coverage=1,duplicates=0,missing=0,
                bins=histogram(np.searchsorted(EDGES,data['nhit'],side='right')-1),
                poolused=histogram(data['poolused']),segments=len(segments),
                boundaries={str(e):int(np.count_nonzero(data['nhit']==e)) for e in EDGES},
                max_nv=int(data['nv'].max()) if len(data['nv']) else 0,
                events_over_4096=int(np.count_nonzero(data['nv']>4096)))


def adopt(state, old, task, target):
    name=task['name']; p=old/(name+'.receipt.json')
    r=json.loads(p.read_text()); ready=json.loads((old/(name+'.ready.json')).read_text())
    if str(target)!=r['path'] or target.stat().st_size!=r['bytes'] or sha256(target)!=r['sha256']:
        raise ValueError('legacy accepted file changed: '+name)
    if ready['input']!=task['input'] or ready['contract']['nhit']!=CONTRACT['nhit'] or ready['contract']['edges']!=EDGES:
        raise ValueError('legacy snapshot/algorithm differs: '+name)
    if ready['sha256']!=r['sha256'] or ready['entries']!=r['entries']:
        raise ValueError('legacy receipt/source mismatch')
    with uproot.open(target) as f:
        data=f['t_nhit'].arrays(SCALARS,library='np'); segments=f['t_segments'].arrays().to_list()
        prov=json.loads(str(f['provenance']))
        if prov['input']!=task['input'] or prov['code']!=ready['code'] or prov['contract']!=ready['contract']:
            raise ValueError('legacy embedded provenance differs')
        kind='legacy_with_charge' if 'vqsamp' in f['t_nhit'] else 'scalar_only'
    if segments!=ready['segments'] or len(data['nv'])!=task['input']['entries']:
        raise ValueError('legacy table/count differs')
    check_scalars(data,segments,task['source_entries'])
    with uproot.open(task['target']) as f:
        target_data=f['t_eventout'].arrays(KEYS+['nv'],library='np')
    for k in target_data:
        if not np.array_equal(target_data[k],data[k]):
            raise ValueError('legacy target alignment changed: '+k)
    rec=dict(r,kind=kind,inherited=True,original_receipt=str(p),original_receipt_sha256=sha256(p),
             input=task['input'],validation=stats(data,segments),
             validation_basis='original durable full-charge receipt + checksum/input/algorithm/scalar revalidation')
    atomic_json(state/'receipts'/(name+'.json'),rec)
    return rec


def accept(state, folder, task, config, target, reference=None):
    start=time.monotonic(); name=task['name']; incoming=folder/'candidate.root'
    rec=json.loads((folder/'ready.json').read_text())
    if rec['name']!=name or not rec['source_readback']:
        raise ValueError('candidate identity/readback')
    if incoming.stat().st_size!=rec['bytes'] or sha256(incoming)!=rec['sha256']:
        raise ValueError('transfer checksum')
    expected=dict(contract=CONTRACT,source_path='/eos/lhaaso/ai/wcda/raw/2022/'+name[7:11]+'/'+name,
                  source_stamp=rec['source'],target_path=task['target'],target_stamp=task['input'],
                  code=config['code'],git_commit=config['git_commit'],run_id=config['run_id'])
    if rec['provenance']!=expected or rec['source']['entries']!=task['source_entries'] or rec['source']['header_sha256']!=task['input']['header_sha256']:
        raise ValueError('candidate provenance mismatch')
    if stamp(task['target'])!=task['input']:
        raise ValueError('INPUT_CHANGED '+name)
    with uproot.open(task['target']) as f:
        a=f['t_eventout'].arrays(KEYS+['nv'],library='np')
        headers=f['t_eventout_h'].arrays().to_list()
    with uproot.open(incoming) as f:
        b=f['t_nhit'].arrays(library='np'); seg=f['t_segments'].arrays().to_list()
        if json.loads(str(f['provenance']))!=expected:
            raise ValueError('embedded candidate provenance')
    if set(b)!=set(SCALARS)-{'input_entry'} or seg!=rec['segments']:
        raise ValueError('candidate schema/table')
    if set(rec['digests'])!=set(b):
        raise ValueError('incomplete scalar digests')
    for k in b:
        if digest_column(ak.Array(b[k]))!=rec['digests'][k]:
            raise ValueError('scalar checksum '+k)
    check_scalars(b,seg,task['source_entries'],ordered_source=True)
    from source import header_specs, json_sha
    specs=header_specs(headers,name[7:11],{tuple(r[k] for k in KEYS[:3]) for r in seg})
    if len(specs)!=len(seg):
        raise ValueError('target header coverage')
    for spec,row in zip(specs,seg):
        if spec['path']!=row['path'] or json_sha(spec['header'])!=row['header_sha256']:
            raise ValueError('target segment header identity')
    order, extras=align(a,b); missing=np.flatnonzero(order<0)
    atomic_json(folder/'alignment.json',dict(target_events=len(order),candidate_events=len(b['nv']),missing=len(missing),
                                           surplus=len(extras),duplicates=0))
    if len(missing):
        np.savez_compressed(folder/'missing_keys.npz',**{k:a[k][missing] for k in KEYS},input_entry=missing)
        raise ValueError('missing target events; keys and provenance retained')
    if len(extras):
        np.savez_compressed(folder/'surplus_keys.npz',**{k:b[k][extras] for k in KEYS+['ihep_entry','segment_index']})
    data={k:v[order] for k,v in b.items()}; data['input_entry']=np.arange(len(order),dtype='i8')
    if not np.array_equal(data['nv'],a['nv']):
        raise ValueError('target nv mismatch')
    check_scalars(data,seg,task['source_entries'])
    comparison=None
    if reference is not None:
        with uproot.open(reference) as f:
            ref=f['t_nhit'].arrays(KEYS+['nv','nhit','poolused','ihep_entry'],library='np')
        for k in ref:
            if not np.array_equal(ref[k],data[k]):
                raise ValueError('accepted reference mismatch '+k)
        if len(extras):
            raise ValueError('pilot surplus is not zero')
        comparison=dict(compared_fields=list(ref),events=len(order),mismatches=0,bin_mismatches=0,
                        reference_path=str(reference),reference_sha256=sha256(reference),exact_fraction=1)
    provenance=dict(expected,candidate_sha256=rec['sha256'],surplus_excluded=len(extras),aligned=True)
    partial=folder/'aligned.partial.root'; partial.unlink(missing_ok=True)
    write_scalar(partial,'t_nhit',data,seg,provenance)
    readback(partial,'t_nhit',data,seg,provenance)
    if stamp(task['target'])!=task['input']:
        raise ValueError('INPUT_CHANGED after alignment '+name)
    digest=sha256(partial); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        # Crash recovery: ROOT UUID changes on a second write, so compare content/provenance.
        readback(target,'t_nhit',data,seg,provenance)
        digest=sha256(target)
    else:
        os.link(partial,target); sync_dir(target.parent)
    partial.unlink()
    receipt=dict(name=name,path=str(target),sha256=digest,bytes=target.stat().st_size,entries=len(order),
        candidate_sha256=rec['sha256'],input=task['input'],inherited=False,kind='scalar_only',
        validation=dict(stats(data,seg),surplus=len(extras)),comparison=comparison,
        source=rec,eto_seconds=time.monotonic()-start,accepted_at=time.time())
    atomic_json(folder/'receipt.json',receipt)
    return receipt
