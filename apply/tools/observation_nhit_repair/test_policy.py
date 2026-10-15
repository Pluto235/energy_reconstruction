"""Adversarial authorization checks; run in Slurm before deploying repair workers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from common import KEYS, sha256
from policy import AUTH_SHA, authorization, load_policy, override_row, check_rows
from source import segment_rows, write_scalar, readback
from validate import check_scalars

PARENT=Path('/mnt/newdisk/shijie/WCDA_obs_friends/nhit/manifests/20261015_january_v6_scalar')


class PolicyTests(unittest.TestCase):
    def setUp(self):
        # Configuration metadata only: no batch event reading in the fixture.
        self.tmp=tempfile.TemporaryDirectory(); self.folder=Path(self.tmp.name)
        for src,dst in [('authorizations/observation_nhit_poolused_overrides_20261015.json','authorization.json'),
                        ('audit/header_fullmonth/source/result.json','header_evidence.json')]:
            (self.folder/dst).write_bytes((PARENT/src).read_bytes())
        self.config=dict(authorization_sha256=AUTH_SHA,header_evidence_sha256=sha256(self.folder/'header_evidence.json'))
        self.policy=load_policy(self.folder,self.config)
        self.task=next(r for r in json.loads((PARENT/'manifest.json').read_text())['records'] if r['name']=='Esg20220102_12.root')
        r=self.policy['records'][self.task['name']]
        path=r['segment']['missing_reduced_path']; filename=Path(path).name.replace('.reduced.root','.samp.dt.root')
        self.spec=dict(key=(139579,1,3336),path=path,header=dict(filename=filename,infiles=filename.replace('.samp.dt.root','.samp.root')))
        self.row=dict(override_row(self.spec,self.task,self.policy),segment_index=0)

    def tearDown(self):self.tmp.cleanup()

    def test_digest_and_snapshot_binding(self):
        bad=copy.deepcopy(self.task);bad['input']['uuid']='changed'
        with self.assertRaises(ValueError):authorization(bad,self.policy)
        bad=copy.deepcopy(self.task);bad['target']='/other'
        with self.assertRaises(ValueError):authorization(bad,self.policy)
        (self.folder/'authorization.json').write_text('{}')
        with self.assertRaises(ValueError):load_policy(self.folder,self.config)

    def test_exact_exception_only(self):
        check_rows([self.row],self.task,self.policy)
        for field,value in [('uuid','fabricated'),('reduced_header_sha256','fake'),('poolused',3),
                            ('poolused_authorization_sha256','wrong'),('iseq',3337),('attempts',1),
                            ('reduced_header_missing',0),('path','other'),('poolused_source','reduced_header')]:
            bad=dict(self.row);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_rows([bad],self.task,self.policy)
        bad=dict(self.task,name='Esg20220128_06.root')
        with self.assertRaises(ValueError):check_rows([self.row],bad,self.policy)
        with self.assertRaises(ValueError):check_rows([],self.task,self.policy)

    def test_other_headers_still_required(self):
        other=copy.deepcopy(self.spec['header'])
        other['filename']=other['filename'].replace('.3336.', '.3337.')
        other['infiles']=other['infiles'].replace('.3336.', '.3337.')
        with patch('source.read_header',side_effect=FileNotFoundError('missing unapproved header')) as opened:
            with self.assertRaises(FileNotFoundError):
                segment_rows([self.spec['header'],other],'0102',{(139579,1,3336),(139579,1,3337)},self.task,self.policy)
            self.assertEqual(opened.call_count,1)
        with patch('source.read_header',side_effect=AssertionError('authorized row must not invent read')):
            rows=segment_rows([self.spec['header']],'0102',{(139579,1,3336)},self.task,self.policy)
            self.assertEqual(rows,[self.row])

    def test_complete_scalar_and_provenance_roundtrip(self):
        data={k:np.array([self.row[k]],dtype='i4') for k in KEYS[:3]}
        data.update(ievent=np.array([1],dtype='i4'),nv=np.array([3],dtype='i4'),nhit=np.array([2],dtype='i4'),
                    poolused=np.array([7],dtype='i4'),segment_index=np.array([0],dtype='i4'),
                    input_entry=np.array([0],dtype='i8'),ihep_entry=np.array([2],dtype='i8'))
        check_scalars(data,[self.row],10,task=self.task,policy=self.policy)
        with self.assertRaises(ValueError):check_scalars(data,[self.row],10)
        provenance=dict(repair_policy=self.policy['binding'],target_stamp=self.task['input'])
        path=self.folder/'roundtrip.root';write_scalar(path,'t_nhit',data,[self.row],provenance)
        readback(path,'t_nhit',data,[self.row],provenance)


if __name__=='__main__':unittest.main()
