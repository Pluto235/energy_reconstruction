"""Bounded HTCondor metadata diagnosis; never computes or publishes Nhit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

FIELDS=['irun','ies','iseq','poolused']


def endpoint(value):
    """Record only endpoint identity, never URL capabilities or credentials."""
    value=str(value)
    parsed=urlsplit(value if '://' in value else 'root://'+value)
    return dict(host=parsed.hostname,port=parsed.port)


def one(spec, mode):
    expected=spec['expected'];path=expected['path']
    uri=spec.get('read_uri',path if mode=='fuse_uproot' else 'root://eos01.ihep.ac.cn/'+path)
    result=dict(mode=mode,uri=uri,at=time.time(),host=os.uname().nodename)
    try:
        if mode in ['root_then_uproot','native_ROOT','native_direct']:
            import ROOT
            result['root_version']=str(ROOT.gROOT.GetVersion())
        if mode in ['native_ROOT','native_direct']:
            f=ROOT.TNetXNGFile(uri,'READ') if mode=='native_direct' else ROOT.TFile.Open(uri,'READ')
            if not f or f.IsZombie():raise ValueError('ROOT zombie')
            try:
                try:
                    actual=f.GetEndpointUrl()
                    result['data_server']=(dict(host=str(actual.GetHost()),port=int(actual.GetPort()))
                        if hasattr(actual,'GetHost') else endpoint(actual))
                except Exception as e:result['data_server_lookup_error']=type(e).__name__
                result['uuid']=str(f.GetUUID().AsString())
                if mode=='native_direct':
                    import ctypes
                    result['metadata_hashes']={}
                    for label,offset,size in [('wcdah',986671206,1647),('streamers',1105399178,6981)]:
                        buf=ctypes.create_string_buffer(size)
                        if f.ReadBuffer(buf,offset,size):raise ValueError('native metadata read error')
                        result['metadata_hashes'][label]=dict(sha256=hashlib.sha256(buf.raw).hexdigest(),bytes=size)
                result['header_keys']=sum(k.GetName()=='wcdah' for k in f.GetListOfKeys())
                t=f.Get('wcdah');result['entries']=int(t.GetEntries())
                result['read_bytes']=int(t.GetEntry(0))
                result['header']={k:[int(getattr(t,k))] for k in FIELDS}
                result['types']={k:str(t.GetLeaf(k).GetTypeName()) for k in FIELDS}
                result['reliable']=(result['header_keys']==1 and result['entries']==1 and result['read_bytes']>0 and
                    all(v in ['Int_t','UInt_t','Long64_t','ULong64_t'] for v in result['types'].values()) and
                    result['uuid']==expected['uuid'] and result['header']==expected['header'])
            finally:f.Close()
        else:
            import uproot
            result['uproot_version']=uproot.__version__
            with uproot.open(uri,timeout=40,handler=uproot.source.xrootd.MultithreadedXRootDSource,num_workers=1) if mode!='fuse_uproot' else uproot.open(uri) as f:
                result['uuid']=str(f.file.uuid);result['keys']=f.keys()
                try:
                    executor=f.file.source._executor
                    resources=([worker.resource for worker in executor.workers] if hasattr(executor,'workers') else [executor._resource])
                    result['data_servers']=[endpoint(resource.file.get_property('DataServer')) for resource in resources]
                except Exception as e:result['data_server_lookup_error']=repr(e)
                # Record the byte locations and content hashes of metadata, not hit data.
                key=f.key('wcdah')
                result['metadata_offsets']=dict(wcdah=int(key.fSeekKey),wcdah_bytes=int(key.fNbytes),streamers=int(f.file.fSeekInfo),streamer_bytes=int(f.file.fNbytesInfo))
                result['metadata_hashes']={}
                for name,offset,size in [('wcdah',key.fSeekKey,key.fNbytes),('streamers',f.file.fSeekInfo,f.file.fNbytesInfo)]:
                    if size<=0 or size>4*1024**2:continue
                    chunk=f.file.source.chunk(int(offset),int(offset+size));raw=bytes(chunk.raw_data)
                    result['metadata_hashes'][name]=dict(sha256=hashlib.sha256(raw).hexdigest(),first32=raw[:32].hex(),bytes=len(raw))
                t=f['wcdah'];a=t.arrays(FIELDS,library='np')
                result['entries']=t.num_entries;result['header']={k:v.tolist() for k,v in a.items()}
                result['types']={k:str(v.dtype) for k,v in a.items()}
                result['reliable']=(sum(k.split(';')[0]=='wcdah' for k in f.keys())==1 and t.num_entries==1 and
                    all(v.dtype.kind in 'iu' for v in a.values()) and result['uuid']==expected['uuid'] and result['header']==expected['header'])
    except Exception as e:result.update(error=repr(e),reliable=False)
    return result


def main(specpath, output):
    if not os.environ.get('_CONDOR_JOB_AD'):raise RuntimeError('HTCondor required')
    spec=json.loads(specpath.read_text());cases=[]
    routes = spec.get('routes', [dict(mode=m) for m in ['uproot','root_then_uproot','native_ROOT','fuse_uproot']])
    for route in routes:
        mode=route['mode']
        case_spec=specpath
        if 'read_uri' in route:
            case_spec=output.parent/('case-'+str(route['fsid'])+'-'+mode+'.json')
            case_spec.write_text(json.dumps(dict(spec,read_uri=route['read_uri'])))
        try:
            p=subprocess.run([sys.executable,'-B',__file__,str(case_spec),str(output),'--mode',mode],text=True,capture_output=True,timeout=75)
            cases.append(dict(route,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        except subprocess.TimeoutExpired as e:
            cases.append(dict(route,timeout=True,stdout=str(e.stdout),stderr=str(e.stderr)))
    try:
        p=subprocess.run(['xrdfs','root://eos01.ihep.ac.cn','locate','-d',spec['expected']['path']],text=True,capture_output=True,timeout=30)
        locations=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except Exception as e:locations=dict(error=repr(e))
    result=dict(cases=cases,locations=locations,host=os.uname().nodename,at=time.time(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),spec_sha256=hashlib.sha256(specpath.read_bytes()).hexdigest())
    tmp=output.with_suffix('.partial')
    with tmp.open('w') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('output',type=Path);p.add_argument('--mode');a=p.parse_args()
    if a.mode:print(json.dumps(one(json.loads(a.spec.read_text()),a.mode)))
    else:main(a.spec,a.output)
