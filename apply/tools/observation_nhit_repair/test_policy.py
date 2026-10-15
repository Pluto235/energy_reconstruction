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
        self.row=dict(override_row(self.spec,self.task,self.policy),segment_index=0,header_reader='user_attestation')

    def tearDown(self):self.tmp.cleanup()

    def test_native_evidence_conflicts(self):
        from native_header import validate_evidence
        expected=dict(uuid='real',header=dict(irun=[140739],ies=[2],iseq=[7581],poolused=[7]))
        actual=dict(expected,entries=1,header_keys=1,read_bytes=16,types={k:'Int_t' for k in expected['header']})
        validate_evidence(actual,expected)
        for field,value in [('uuid','different'),('entries',2),('header_keys',2),('read_bytes',0),
                            ('types',dict(actual['types'],poolused='Float_t')),
                            ('header',dict(expected['header'],ies=[1]))]:
            bad=dict(actual);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_evidence(bad,expected)

    def test_replica_binding_and_endpoint(self):
        from replica_header import NAME, EVIDENCE_SHA, validate_probe
        source=PARENT/'audit/header_io_v3/source/result.json'
        (self.folder/'replica_evidence.json').write_bytes(source.read_bytes())
        config=dict(self.config,replica_evidence_sha256=EVIDENCE_SHA)
        policy=load_policy(self.folder,config);route=policy['replica'];expected=policy['recovered'][NAME]
        proof=dict(uri=route['uri'],data_servers=[route['server']],evidence_sha256=EVIDENCE_SHA,
                   uuid=expected['uuid'],header=expected['header'],entries=1,header_keys=1,metadata_hashes=route['hashes'])
        actual=dict(path=route['uri'],opener='TNetXNGFile',native_server=route['server'],
                    replica_probe=proof,metadata_hashes=route['hashes'])
        validate_probe(actual,policy)
        for field,value in [('native_server',dict(host='eos40.ihep.ac.cn',port=1095)),
                            ('path',route['uri'].replace('40208','1300')),('opener','TFile')]:
            bad=copy.deepcopy(actual);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_probe(bad,policy)
        for location in ['metadata_hashes','replica_probe']:
            bad=copy.deepcopy(actual)
            hashes=bad['metadata_hashes'] if location=='metadata_hashes' else bad['replica_probe']['metadata_hashes']
            hashes['wcdah']['sha256']='0'*64
            with self.subTest(location=location),self.assertRaises(ValueError):validate_probe(bad,policy)
        (self.folder/'replica_evidence.json').write_text('{}')
        with self.assertRaises(ValueError):load_policy(self.folder,config)

    def test_replica_segment_roundtrip(self):
        from replica_header import NAME,EVIDENCE_SHA,ROW_FIELDS
        (self.folder/'replica_evidence.json').write_bytes((PARENT/'audit/header_io_v3/source/result.json').read_bytes())
        policy=load_policy(self.folder,dict(self.config,replica_evidence_sha256=EVIDENCE_SHA))
        task=next(r for r in json.loads((PARENT/'manifest.json').read_text())['records'] if r['name']==NAME)
        expected=policy['recovered'][NAME];route=policy['replica']
        from policy import json_sha
        row=dict(segment_index=0,path=expected['path'],uuid=expected['uuid'],source_filename='original',
                 header_sha256='original',reduced_header_sha256=json_sha(expected['header']),attempts=1,
                 poolused_source='reduced_header',poolused_authorization_sha256='',reduced_header_missing=0,
                 header_reader='native_ROOT',**{k:v[0] for k,v in expected['header'].items()},
                 **dict(zip(ROW_FIELDS,[route['uri'],'eos92.ihep.ac.cn:1095',EVIDENCE_SHA])))
        check_rows([row],task,policy)
        for key in ROW_FIELDS:
            bad=dict(row);bad[key]=''
            with self.subTest(key=key),self.assertRaises(ValueError):check_rows([bad],task,policy)
        data={k:np.array([row[k]],dtype='i4') for k in KEYS[:3]}
        data.update(ievent=np.array([1],dtype='i4'),nv=np.array([4],dtype='i4'),nhit=np.array([3],dtype='i4'),
                    poolused=np.array([7],dtype='i4'),segment_index=np.array([0],dtype='i4'),
                    input_entry=np.array([0],dtype='i8'),ihep_entry=np.array([2],dtype='i8'))
        path=self.folder/'pinned-roundtrip.root';provenance=dict(repair_policy=policy['binding'])
        write_scalar(path,'t_nhit',data,[row],provenance)
        readback(path,'t_nhit',data,[row],provenance)
        check_scalars(data,[row],10,task=task,policy=policy)

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
