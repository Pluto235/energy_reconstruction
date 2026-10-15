"""One HTCondor file task. No physics cuts; never edits original observations."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import time
import traceback

import awkward as ak
import numpy as np
import uproot

from common import (CONTRACT, KEYS, atomic_json, digest_column, histogram, lookup_chunk,
                    nhit, sha256, source_uri, stamp, sync_dir, unique_keys)


def segments(source, day, wanted):
    mapping = {}
    rows = []
    for h in source['t_eventout_h'].arrays().to_list():
        name = h['filename']
        m = re.fullmatch(r'ES\.(\d+)\.WCDA_EVENT\..+\.es-(\d+)\.\d{14}\.(\d+)\.samp\.dt\.root', name)
        if not m:
            raise ValueError('unrecognized segment filename: ' + name)
        key = tuple(map(int, m.groups()))
        if key in mapping:
            raise ValueError('duplicate segment header: ' + str(key))
        mapping[key] = None
        if key not in wanted:
            continue
        if h['infiles'] != name.replace('.samp.dt.root', '.samp.root'):
            raise ValueError('segment filename/infiles disagree: ' + name)
        path = '/eos/lhaaso/raw/wcda/2022/' + day + '/' + name.replace('.samp.dt.root', '.reduced.root')
        with uproot.open(source_uri(path), timeout=60) as f:
            t = f['wcdah']
            if t.num_entries != 1:
                raise ValueError('segment wcdah is not unique: ' + path)
            a = t.arrays(KEYS[:3] + ['poolused'], library='np')
            if tuple(int(a[k][0]) for k in KEYS[:3]) != key:
                raise ValueError('segment identity mismatch: ' + path)
            pool = int(a['poolused'][0])
            if not 0 <= pool <= 7:
                raise ValueError('unexpected poolused mask: ' + str(pool))
            row = dict(segment_index=len(rows), irun=key[0], ies=key[1], iseq=key[2],
                       poolused=pool, path=path, uuid=str(f.file.uuid), source_filename=name,
                       header_sha256=sha256_json(h))
        mapping[key] = row
        rows.append(row)
    missing = [k for k in wanted if mapping.get(k) is None]
    if missing:
        raise ValueError('missing segment headers: ' + str(missing[:10]))
    return mapping, rows


def sha256_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extract(work, name):
    start = time.time()
    print('START', name, flush=True)
    task = json.loads((work / (name + '.task.json')).read_text())
    if task['contract'] != CONTRACT:
        raise ValueError('contract mismatch')
    resource.setrlimit(resource.RLIMIT_FSIZE, (task['file_limit'], task['file_limit']))
    request = work / (name + '.requests.npz')
    if sha256(request) != task['request_sha256']:
        raise ValueError('request checksum mismatch')
    with np.load(request) as z:
        target = {k: z[k] for k in KEYS}
    keys = unique_keys(target)
    order = np.argsort(keys)
    entries = np.full(len(keys), -1, dtype=np.int64)
    seen = np.zeros(len(keys), dtype=bool)
    src = task['source']
    before = stamp(src)
    print('SOURCE_METADATA', name, before['entries'], flush=True)
    if before['entries'] != task['source_entries'] or before['header_sha256'] != task['input']['header_sha256']:
        raise ValueError('original source count/header differs from restoration')
    common = list(task['input']['types'])
    if any(before['types'].get(k) != task['input']['types'][k] for k in common):
        raise ValueError('common branch schema mismatch')
    output = work / (name + '.friend.root')
    partial = output.with_suffix('.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('stale output requires explicit recovery: ' + str(output))
    blocks = []
    # EOS rejects some fsspec vector reads; native bounded reads avoid that path.
    with uproot.open(source_uri(src), array_cache=None, timeout=60,
                     handler=uproot.source.xrootd.MultithreadedXRootDSource, num_workers=1) as source:
        tree = source['t_eventout']
        offset = 0
        for a in tree.iterate(KEYS, step_size=200000, library='np'):
            lookup_chunk(keys, order, a, offset, entries, seen)
            offset += len(a['irun'])
            if offset % 1000000 == 0:
                print('KEY_SCAN', name, offset, flush=True)
        if not np.all(seen):
            raise ValueError('missing source keys: ' + str(np.flatnonzero(~seen)[:10].tolist()))
        wanted = set(zip(*(target[k].tolist() for k in KEYS[:3])))
        segmap, rows = segments(source, name[7:11], wanted)
        print('SEGMENTS', name, len(rows), flush=True)
        with uproot.recreate(partial, compression=uproot.ZLIB(4)) as out:
            schema = {k: 'int32' for k in KEYS + ['nv', 'nhit', 'poolused', 'segment_index']}
            schema.update(input_entry='int64', ihep_entry='int64', vqsamp='var * float32')
            out.mktree('t_nhit', schema)
            out['provenance'] = json.dumps(dict(contract=CONTRACT, source=src, source_stamp=before,
                                               input=task['input'], code=task['code']), sort_keys=True)
            out.mktree('t_segments', {k: 'int32' for k in ['segment_index'] + KEYS[:3] + ['poolused']} |
                        {k: 'string' for k in ['path', 'uuid', 'source_filename', 'header_sha256']})
            if rows:
                # An Awkward record also initializes uproot 5.3's string writer correctly.
                out['t_segments'].extend(ak.Array(rows))
            lo = 0
            while lo < len(keys):
                hi = min(lo + 1000, len(keys))
                # Bound reads even if the selected event order changes radically.
                while hi - lo > 1 and np.ptp(entries[lo:hi]) > 100000:
                    hi = lo + (hi - lo) // 2
                first, last = int(entries[lo:hi].min()), int(entries[lo:hi].max()) + 1
                a = tree.arrays(common + ['vqsamp'], entry_start=first, entry_stop=last)
                a = a[entries[lo:hi] - first]
                if not np.array_equal(unique_keys(a), keys[lo:hi]):
                    raise ValueError('source changed during read')
                counts = nhit(a.vqsamp, a.vt, a.nv, a.vidmc)
                sr = [segmap[k] for k in zip(*(target[k][lo:hi].tolist() for k in KEYS[:3]))]
                data = {k: a[k] for k in KEYS + ['nv', 'vqsamp']}
                data.update(input_entry=np.arange(lo, hi, dtype=np.int64), ihep_entry=entries[lo:hi],
                            nhit=counts, poolused=np.array([r['poolused'] for r in sr], dtype='i4'),
                            segment_index=np.array([r['segment_index'] for r in sr], dtype='i4'))
                out['t_nhit'].extend(data)
                blocks.append(dict(start=lo, stop=hi,
                                   common={k: digest_column(a[k]) for k in common},
                                   friend={k: digest_column(ak.Array(v)) for k, v in data.items()}))
                lo = hi
                if len(blocks) % 20 == 0:
                    print(name, lo, '/', len(keys), flush=True)
    if stamp(src) != before:
        raise ValueError('source metadata changed during extraction')
    with uproot.open(partial) as f:
        if f['t_nhit'].num_entries != len(keys) or f['t_segments'].num_entries != len(rows):
            raise ValueError('source-side readback counts')
        for b in blocks:
            a = f['t_nhit'].arrays(entry_start=b['start'], entry_stop=b['stop'])
            for k, digest in b['friend'].items():
                if digest_column(a[k]) != digest:
                    raise ValueError('source-side readback mismatch: ' + k)
    with partial.open('rb') as f:
        os.fsync(f.fileno())
    os.rename(partial, output); sync_dir(work)
    record = dict(name=name, contract=CONTRACT, source=before, input=task['input'], segments=rows,
                  blocks=blocks, entries=len(keys), source_entries=before['entries'],
                  bytes=output.stat().st_size, sha256=sha256(output), request_sha256=task['request_sha256'],
                  code=task['code'], elapsed_seconds=time.time()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  unique_matches=len(keys), missing=0, duplicate_matches=0,
                  source_readback=True)
    atomic_json(work / (name + '.ready.json'), record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('work', type=Path); p.add_argument('name')
    args = p.parse_args()
    try:
        extract(args.work, args.name)
    except Exception as e:
        traceback.print_exc()
        atomic_json(args.work / (args.name + '.failed.json'),
                    dict(name=args.name, error=repr(e), traceback=traceback.format_exc()))
        raise
