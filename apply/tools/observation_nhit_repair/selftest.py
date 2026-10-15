"""Minimal adversarial contract checks, including real ROOT extraction when available."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
import uproot
from common import KEYS, EDGES
from source import HeaderTransportExhausted, is_transient_file_error, read_header, CONTRACT, header_specs, validate_header, write_scalar, readback
from validate import align, check_scalars


def keys(values):
    return {k:np.asarray(values if k=='ievent' else [0]*len(values),dtype='i4') for k in KEYS}


class ContractTests(unittest.TestCase):
    def test_retry_ceiling(self):
        self.assertTrue(is_transient_file_error(OSError('connection timed out')))
        self.assertFalse(is_transient_file_error(HeaderTransportExhausted('connection timed out')))
        self.assertFalse(is_transient_file_error(FileNotFoundError('[3011] connection: missing header')))
        with patch('source.uproot.open',side_effect=OSError('connection timed out')) as opened, patch('source.time.sleep'):
            with self.assertRaises(HeaderTransportExhausted) as error:
                read_header(dict(path='/test/timeout.root'))
            self.assertEqual(opened.call_count,3)
            self.assertFalse(is_transient_file_error(error.exception))

    def test_alignment(self):
        rows,extra=align(keys([3,1,8]),keys([1,9,3]))
        self.assertEqual(rows.tolist(),[2,0,-1]);self.assertEqual(extra.tolist(),[1])
        for a,b in [([1,1],[1]),([1],[1,1])]:
            with self.assertRaises(ValueError):align(keys(a),keys(b))
        self.assertEqual(align(keys([]),keys([]))[0].size,0)
        self.assertEqual(align(keys([1]),keys([]))[0].tolist(),[-1])

    def test_boundaries_and_full_hits(self):
        q=np.array([.3,np.nextafter(np.float32(.3),np.float32(0)),np.nextafter(np.float32(.3),np.float32(1))],dtype='f4')
        self.assertEqual((q>=np.float32(.3)).tolist(),[True,False,True])
        self.assertEqual((np.abs(np.array([30.,-30.,np.nextafter(30.,0.),np.nextafter(-30.,0.)]))<30).tolist(),[False,False,True,True])
        for i,e in enumerate(EDGES):
            self.assertEqual(np.searchsorted(EDGES,[e-1,e],side='right').tolist(),[i,i+1])
        self.assertEqual(np.count_nonzero(np.ones(100001)>=np.float32(.3)),100001)

    def test_header_failures_and_pool(self):
        name='ES.12.WCDA_EVENT.X.es-1.20220101000000.9.samp.dt.root'
        h=dict(filename=name,infiles=name.replace('.samp.dt.root','.samp.root'))
        specs=header_specs([h],'0101',{(12,1,9)})
        self.assertEqual(len(specs),1)
        for hs,wanted in [([h,h],{(12,1,9)}),([],{(12,1,9)}),([dict(h,infiles='wrong')],{(12,1,9)})]:
            with self.assertRaises(ValueError):header_specs(hs,'0101',wanted)
        a=dict(irun=[12],ies=[1],iseq=[9],poolused=[3])
        self.assertEqual(validate_header((12,1,9),a,1),3)
        with self.assertRaises(ValueError):validate_header((12,1,9),a,2)
        with self.assertRaises(ValueError):validate_header((12,1,8),a,1)

    def test_empty_and_pool_roundtrip(self):
        for count in [0,1]:
            data=keys(list(range(count)))
            data.update(nv=np.zeros(count,dtype='i4'),nhit=np.zeros(count,dtype='i4'),poolused=np.full(count,3,dtype='i4'),
                segment_index=np.zeros(count,dtype='i4'),ihep_entry=np.arange(count,dtype='i8'),input_entry=np.arange(count,dtype='i8'))
            rows=[dict(segment_index=0,irun=0,ies=0,iseq=0,poolused=3,attempts=1,path='p',uuid='u',source_filename='s',header_sha256='h',reduced_header_sha256='r',poolused_source='reduced_header',poolused_authorization_sha256='',reduced_header_missing=0,header_reader='uproot')] if count else []
            check_scalars(data,rows,max(1,count))
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'f.root';write_scalar(p,'t_nhit',data,rows,CONTRACT);readback(p,'t_nhit',data,rows,CONTRACT)


def root_selftest(ROOT, folder):
    """Exercise the deployed C++ reader with an empty event and 100001 stored hits."""
    src=folder/'selfcheck-input.root';out=folder/'selfcheck-output.root'
    down=np.nextafter(np.float32(.3),np.float32(0)); up=np.nextafter(np.float32(.3),np.float32(1))
    qs=[[],[.3,.3,.3,down,up,.3],np.ones(100001,dtype='f4'),[1.]]
    ts=[[],[0.,30.,-30.,0.,np.nextafter(30.,0.),np.nextafter(-30.,0.)],np.zeros(100001),[0.]]
    with uproot.recreate(src) as f:
        f.mktree('t_eventout',{**{k:'int32' for k in KEYS+['nv']},'pincness':'float64','vqsamp':'var * float32','vt':'var * float64'})
        data={k:np.arange(4,dtype='i4') if k=='ievent' else np.zeros(4,dtype='i4') for k in KEYS}
        data.update(nv=np.array([0,6,100001,1],dtype='i4'),pincness=np.array([0.,0.,0.,1.1]),
                    vqsamp=ak.values_astype(ak.Array(qs),'float32'),vt=ak.values_astype(ak.Array(ts),'float64'))
        f['t_eventout'].extend(data)
    ROOT.nhit_benchmark_filter(str(src),str(out))
    with uproot.open(out) as f:
        a=f['t_candidate'].arrays(library='np')
        assert a['nhit'].tolist()==[0,3,100001],a
        assert a['ihep_entry'].tolist()==[0,1,2]
    src.unlink();out.unlink()
    return dict(empty_event=True,charge_and_time_boundaries=True,over_model_limit_hits=100001,pinc_only=True)


if __name__=='__main__':unittest.main()
