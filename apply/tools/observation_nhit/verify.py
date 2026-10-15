"""Independent ETO validation before publication; runs inside the receiver Slurm job."""
from collections import Counter
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot

from common import CONTRACT, EDGES, KEYS, digest_column, histogram, nhit, sha256, stamp, unique_keys


def verify(input_path, friend, record, frozen):
    started = time.time()
    if record['contract'] != CONTRACT or record['input'] != frozen:
        raise ValueError('input/contract changed')
    if stamp(input_path) != frozen:
        raise ValueError('ETO input changed since manifest freeze')
    if Path(friend).stat().st_size != record['bytes'] or sha256(friend) != record['sha256']:
        raise ValueError('transfer size/checksum mismatch')
    stats = Counter(); pools = Counter(); bins = Counter(); segments_seen = set()
    all_keys = []; source_entries = []
    with uproot.open(input_path, array_cache=None) as src, uproot.open(friend, array_cache=None) as dst:
        t, f = src['t_eventout'], dst['t_nhit']
        if t.num_entries != f.num_entries or f.num_entries != record['entries']:
            raise ValueError('event count mismatch')
        seg = dst['t_segments'].arrays().to_list()
        if seg != record['segments']:
            raise ValueError('segment provenance readback mismatch')
        previous = 0
        for block in record['blocks']:
            lo, hi = block['start'], block['stop']
            if lo != previous or not lo < hi <= t.num_entries:
                raise ValueError('invalid block coverage')
            a = t.arrays(entry_start=lo, entry_stop=hi)
            b = f.arrays(entry_start=lo, entry_stop=hi)
            for k, expected in block['common'].items():
                if digest_column(a[k]) != expected:
                    key = {k: int(a[k][0]) for k in KEYS}
                    raise ValueError(f'common content differs: branch={k}, entries=[{lo},{hi}), first_key={key}')
            if set(block['common']) != set(frozen['types']):
                raise ValueError('incomplete common branch verification')
            for k, expected in block['friend'].items():
                if digest_column(b[k]) != expected:
                    raise ValueError('friend readback differs: ' + k)
            for k in KEYS + ['nv']:
                if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
                    raise ValueError('identity/nv mismatch: ' + k)
            if not np.array_equal(np.asarray(b.input_entry), np.arange(lo, hi)):
                raise ValueError('input_entry is not contiguous')
            counts = nhit(b.vqsamp, a.vt, a.nv, a.vidmc)
            delta = counts.astype('i8') - np.asarray(b.nhit, dtype='i8')
            if np.any(delta):
                j = int(np.flatnonzero(delta)[0])
                raise ValueError(f'Nhit mismatch at input_entry={lo+j}, delta={delta[j]}')
            ids = np.asarray(b.segment_index)
            if np.any(ids < 0) or np.any(ids >= len(seg)):
                raise ValueError('invalid segment index')
            for k in KEYS[:3] + ['poolused']:
                if not np.array_equal(np.asarray(b[k]), np.array([s[k] for s in seg])[ids]):
                    raise ValueError('segment/header identity mismatch: ' + k)
            bins.update(histogram(np.searchsorted(EDGES, counts, side='right') - 1))
            pools.update(histogram(b.poolused)); segments_seen.update(ids.tolist())
            stats['events'] += hi-lo
            stats['hits_exact_charge_0p3'] += int(np.sum(np.asarray(ak.flatten(b.vqsamp)) == np.float32(.3)))
            stats['hits_exact_abs_time_30'] += int(np.sum(abs(np.asarray(ak.flatten(a.vt))) == 30))
            for edge in EDGES:
                stats['events_nhit_' + str(edge)] += int(np.sum(counts == edge))
            all_keys.append(unique_keys(b)); source_entries.append(np.asarray(b.ihep_entry))
            previous = hi
        if previous != t.num_entries:
            raise ValueError('blocks do not cover whole file')
        keys = np.concatenate(all_keys) if all_keys else np.empty(0, dtype='V16')
        ent = np.concatenate(source_entries) if source_entries else np.empty(0, dtype='i8')
        if len(np.unique(keys)) != len(keys) or len(np.unique(ent)) != len(ent):
            raise ValueError('duplicate event identities or reused raw entries')
        if np.any(ent < 0) or np.any(ent >= record['source_entries']):
            raise ValueError('raw entry outside source')
        if segments_seen != set(range(len(seg))):
            raise ValueError('segment coverage mismatch')
    if stamp(input_path) != frozen:
        raise ValueError('input changed during verification')
    return dict(stats, field_coverage=1, matching_coverage=1, nhit_equal_fraction=1 if len(keys) else None,
                delta_histogram={'0': len(keys)}, bin_mismatch_fraction=0 if len(keys) else None,
                poolused=dict(pools), bins=dict(bins), segments=len(segments_seen),
                verified_at=time.time(), verify_seconds=time.time()-started,
                empty_sample=not len(keys), code_evidence=CONTRACT, data_evidence='full-file all retained branches and all events')
