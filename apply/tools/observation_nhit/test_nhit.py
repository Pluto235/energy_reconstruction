"""Small contract tests; no observational data or scheduler access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil

import awkward as ak
import numpy as np
import uproot

from common import CONTRACT, EDGES, KEYS, digest_column, lookup_chunk, nhit, sha256, stamp, unique_keys
from extract import segments
from verify import verify
import run


def table(ids):
    return {k: np.array(ids if k == 'ievent' else [1]*len(ids), dtype='i4') for k in KEYS}


class Checks(unittest.TestCase):
    def test_matching_order_missing_duplicates(self):
        keys = unique_keys(table([5, 1, 3])); order = np.argsort(keys)
        entries = np.full(3, -1, dtype='i8'); seen = np.zeros(3, dtype=bool)
        lookup_chunk(keys, order, table([1, 2, 3]), 0, entries, seen)
        self.assertEqual(seen.tolist(), [False, True, True])
        lookup_chunk(keys, order, table([4, 5]), 3, entries, seen)
        self.assertEqual(entries.tolist(), [4, 0, 2])
        with self.assertRaises(ValueError): lookup_chunk(keys, order, table([5]), 5, entries, seen)
        with self.assertRaises(ValueError): unique_keys(table([1, 1]))
        with self.assertRaises(ValueError):
            lookup_chunk(keys, order, table([1, 1]), 0, entries, np.zeros(3, dtype=bool))

    def test_boundaries_full_hits_and_invalid_arrays(self):
        q0 = np.float32(.3)
        q = ak.Array([np.array([np.nextafter(q0, np.float32(0)), q0, q0, q0, q0], dtype='f4')])
        t = ak.Array([np.array([0, 0, 30, -30, np.nextafter(np.float32(30), np.float32(0))], dtype='f4')])
        self.assertEqual(nhit(q, t, [5], ak.Array([[1]*5])).tolist(), [2])
        q = ak.Array([[1.]*2500, []]); t = ak.Array([[0.]*2500, []]); ids=ak.Array([[1]*2500, []])
        self.assertEqual(nhit(q, t, [2500, 0], ids).tolist(), [2500, 0])
        with self.assertRaises(ValueError): nhit(q, t, [1, 0], ids)
        with self.assertRaises(ValueError): nhit(ak.Array([[float('nan')]]), ak.Array([[0.]]), [1], ak.Array([[1]]))
        self.assertEqual(np.searchsorted(EDGES, EDGES, side='right').tolist(), list(range(1,9)))
        self.assertEqual(np.searchsorted(EDGES, [0,29,2001], side='right').tolist(), [0,0,8])

    def test_digest_preserves_event_boundaries(self):
        self.assertNotEqual(digest_column(ak.Array([[1,2],[]])), digest_column(ak.Array([[1],[2]])))

    def test_segment_missing_duplicate_and_bad_name(self):
        class Header:
            def __init__(self, rows): self.rows=rows
            def arrays(self): return ak.Array(self.rows)
        name='ES.1.WCDA_EVENT.foo.es-1.20220101000000.1.samp.dt.root'
        row=dict(filename=name, infiles=name.replace('.samp.dt.', '.samp.'), texts='', mtime='')
        with self.assertRaises(ValueError): segments({'t_eventout_h':Header([row,row])}, '0101', set())
        with self.assertRaises(ValueError): segments({'t_eventout_h':Header([row])}, '0101', {(2,1,1)})
        with self.assertRaises(ValueError): segments({'t_eventout_h':Header([dict(row,filename='bad')])}, '0101', set())

    def test_full_verifier_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); inp=d/'input.root'; friend=d/'friend.root'
            a=table([3,1]); a.update(nv=np.array([2,1],dtype='i4'),
                vt=ak.values_astype(ak.Array([[0,30],[0]]),np.float32),
                vidmc=ak.values_astype(ak.Array([[1,2],[3]]),np.int32))
            schema={k:'int32' for k in KEYS+['nv']} | dict(vt='var * float32',vidmc='var * int32')
            with uproot.recreate(inp) as f:
                f.mktree('t_eventout',schema); f['t_eventout'].extend(a)
                f.mktree('t_eventout_h',dict(filename='string')); f['t_eventout_h'].extend(dict(filename=['source']))
            frozen=stamp(inp)
            b={k:a[k] for k in KEYS+['nv']}
            b.update(vqsamp=ak.values_astype(ak.Array([[1,1],[1]]),np.float32),
                     nhit=np.array([1,1],dtype='i4'),poolused=np.array([0,0],dtype='i4'),
                     segment_index=np.array([0,0],dtype='i4'),input_entry=np.array([0,1],dtype='i8'),
                     ihep_entry=np.array([8,2],dtype='i8'))
            seg=dict(segment_index=0,irun=1,ies=1,iseq=1,poolused=0)
            with uproot.recreate(friend) as f:
                f.mktree('t_nhit',{k:'int32' for k in KEYS+['nv','nhit','poolused','segment_index']} |
                         dict(vqsamp='var * float32', input_entry='int64',ihep_entry='int64'))
                f['t_nhit'].extend(b)
                f.mktree('t_segments',{k:'int32' for k in seg}); f['t_segments'].extend({k:[v] for k,v in seg.items()})
            with uproot.open(inp) as f: common=f['t_eventout'].arrays()
            rec=dict(contract=CONTRACT,input=frozen,bytes=friend.stat().st_size,sha256=sha256(friend),
                     entries=2,source_entries=10,segments=[seg],blocks=[dict(start=0,stop=2,
                     common={k:digest_column(common[k]) for k in common.fields},
                     friend={k:digest_column(ak.Array(v)) for k,v in b.items()})])
            self.assertEqual(verify(inp,friend,rec,frozen)['nhit_equal_fraction'],1)
            rec['blocks'][0]['common']['nv']='bad'
            with self.assertRaises(ValueError): verify(inp,friend,rec,frozen)
            rec['sha256']='bad'
            with self.assertRaises(ValueError): verify(inp,friend,rec,frozen)

    def test_publication_failure_and_ack_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); state=d/'state'; state.mkdir(); name='Esg20220101_00.root'
            blob=d/'blob'; blob.write_bytes(b'root-test')
            record=dict(name=name, request_sha256='request', code={}, sha256=sha256(blob), bytes=9, entries=1)
            (state/(name+'.task.json')).write_text(json.dumps(dict(request_sha256='request',code={})))
            def transfer(src, dst):
                if src.endswith('.ready.json'): Path(dst).write_text(json.dumps(record))
                else: shutil.copyfile(blob,dst)
            target=d/'full/0101/Esg20220101_00.nhit.root'
            with patch.object(run,'BASE',d), patch.object(run,'transfer',transfer), patch.object(run,'cleanup') as cleanup:
                with patch.object(run,'verify',side_effect=ValueError('bad content')):
                    with self.assertRaises(ValueError): run.receive(state,dict(name=name,input={}),'/unused')
                self.assertFalse(target.exists()); cleanup.assert_not_called()
                cleanup.side_effect=RuntimeError('ack connection lost')
                with patch.object(run,'verify',return_value={}):
                    with self.assertRaises(RuntimeError): run.receive(state,dict(name=name,input={}),'/unused')
                self.assertEqual(target.read_bytes(),b'root-test')
                self.assertTrue((state/(name+'.receipt.json')).exists())
                cleanup.side_effect=None
                run.receive(state,dict(name=name,input={}),'/unused')
                target.write_bytes(b'changed')
                with self.assertRaises(ValueError): run.receive(state,dict(name=name,input={}),'/unused')


if __name__ == '__main__': unittest.main()
