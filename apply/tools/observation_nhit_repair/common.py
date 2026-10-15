"""January observation friend contract; shared by IHEP extraction and ETO checks."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import awkward as ak
import numpy as np
import uproot

KEYS = ['irun', 'ies', 'iseq', 'ievent']
EDGES = [30, 60, 100, 200, 300, 500, 800, 2000]
CONTRACT = dict(version=1, tree='t_nhit', keys=KEYS, edges=EDGES,
                nhit='sum((vqsamp >= float32(0.3)) & (abs(vt) < 30))',
                poolused='configuration from identity-checked reduced segment wcdah',
                filters='none; existing ETO event universe', common_comparison='exact canonical bytes')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def source_uri(path):
    # Compute-node EOS FUSE mounts can stall; use the site's authenticated endpoint.
    return 'root://eos01.ihep.ac.cn/' + str(path) if str(path).startswith('/eos/') else str(path)


def stamp(path):
    if str(path).startswith('/eos/'):
        from XRootD import client
        status, result = client.FileSystem('root://eos01.ihep.ac.cn').stat(str(path), timeout=60)
        if not status.ok:
            raise OSError(str(status))
        size, mtime = result.size, result.modtime * 10**9
    else:
        s = Path(path).stat()
        size, mtime = s.st_size, s.st_mtime_ns
    with uproot.open(source_uri(path), timeout=60) as f:
        t = f['t_eventout']
        return dict(bytes=size, mtime_ns=mtime, uuid=str(f.file.uuid),
                    entries=t.num_entries, types=t.typenames(),
                    header_sha256=hashlib.sha256(json.dumps(f['t_eventout_h'].arrays().to_list(),
                        sort_keys=True).encode()).hexdigest())


def keybytes(table):
    a = np.ascontiguousarray(np.column_stack([np.asarray(table[k], dtype='<i4') for k in KEYS]))
    return a.view('V16').ravel()


def unique_keys(table):
    keys = keybytes(table)
    if len(np.unique(keys)) != len(keys):
        raise ValueError('duplicate target event keys')
    return keys


def lookup_chunk(target_keys, order, source, offset, entries, seen):
    keys = keybytes(source)
    if not len(order):
        return
    pos = np.searchsorted(target_keys[order], keys)
    bounded = np.minimum(pos, len(order) - 1)
    hit = (pos < len(order)) & (target_keys[order[bounded]] == keys)
    src = np.flatnonzero(hit)
    dst = order[pos[hit]]
    if len(np.unique(dst)) != len(dst) or np.any(seen[dst]):
        raise ValueError('duplicate matching keys in IHEP source')
    entries[dst] = offset + src
    seen[dst] = True


def digest_column(x):
    """Canonical little-endian values plus per-event lengths (regular arrays included)."""
    h = hashlib.sha256()
    if x.ndim == 1:
        a = ak.to_numpy(x)
    else:
        h.update(np.asarray(ak.num(x, axis=1), dtype='<i8').tobytes())
        a = ak.to_numpy(ak.flatten(x, axis=1))
    a = np.asarray(a, dtype=a.dtype.newbyteorder('<'))
    h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def nhit(q, t, nv, ids):
    for a in (q, t, ids):
        if not np.array_equal(np.asarray(ak.num(a)), np.asarray(nv)):
            raise ValueError('hit array length differs from nv')
    if not bool(ak.all(np.isfinite(q))) or not bool(ak.all(np.isfinite(t))):
        raise ValueError('nonfinite charge/time')
    return np.asarray(ak.sum((q >= np.float32(0.3)) & (abs(t) < 30), axis=1), dtype=np.int32)


def histogram(x):
    v, n = np.unique(x, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(v, n)}


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('ETO processing must run through Slurm')
