"""HTCondor metadata-only search for identity-exact missing January headers.

This writes evidence, never invents poolused or publishes friends. Input is a list
of {name, required_path}; output includes negative as well as positive matches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import uproot
from XRootD import client

SEGMENT = re.compile(r'ES\.(\d+)\.WCDA_EVENT\..+\.es-(\d+)\.\d{14}\.(\d+)\.')
KEYS = ['irun', 'ies', 'iseq', 'poolused']


def inspect_header(path, wanted):
    result=dict(path=path)
    for attempt in range(3):
        try:
            with uproot.open('root://eos01.ihep.ac.cn/'+path, timeout=60,
                             handler=uproot.source.xrootd.MultithreadedXRootDSource, num_workers=1) as f:
                result.update(uuid=str(f.file.uuid),keys=f.keys(),attempts=attempt+1)
                if 'wcdah' not in f:
                    result['reliable_header']=False;return result
                t=f['wcdah'];a=t.arrays(KEYS,library='np')
                result.update(header_entries=t.num_entries,header={k:v.tolist() for k,v in a.items()},
                              header_types=t.typenames())
                result['exact_identity']=(t.num_entries==1 and
                    tuple(int(a[k][0]) for k in KEYS[:3])==tuple(wanted))
                result['reliable_header']=(result['exact_identity'] and
                    all(a[k].dtype.kind in 'iu' for k in KEYS) and 0<=int(a['poolused'][0])<=7)
                return result
        except Exception as e:
            result.setdefault('errors',[]).append(dict(attempt=attempt+1,error=repr(e)))
            if not isinstance(e,OSError) or '3011' in str(e) or attempt==2:
                return result
            time.sleep(2*(attempt+1))
    return result


def main(specs_path, output):
    if not os.environ.get('_CONDOR_JOB_AD'):
        raise RuntimeError('Header investigation requires HTCondor')
    specs=json.loads(specs_path.read_text())
    wanted={}
    for s in specs:
        path=s['required_path']
        if not re.fullmatch(r'/eos/lhaaso/raw/wcda/2022/01\d\d/ES\.[^/]+\.reduced\.root',path):
            raise ValueError('Unexpected requested header path')
        m=SEGMENT.match(Path(path).name)
        if not m:raise ValueError(path)
        s['identity']=list(map(int,m.groups()));s['matches']=[]
        wanted.setdefault((s['identity'][0],s['identity'][2]),[]).append(s)
    fs=client.FileSystem('root://eos01.ihep.ac.cn')
    directories=[]
    for day in range(1,32):
        path=f'/eos/lhaaso/raw/wcda/2022/01{day:02d}'
        status,entries=fs.dirlist(path,timeout=60)
        if not status.ok:
            directories.append(dict(path=path,error=str(status)));continue
        entries=list(entries);directories.append(dict(path=path,entries=len(entries)))
        for entry in entries:
            m=SEGMENT.match(entry.name)
            if not m:continue
            ident=tuple(map(int,m.groups()))
            for spec in wanted.get((ident[0],ident[2]),[]):
                candidate=path+'/'+entry.name
                row=inspect_header(candidate,spec['identity'])
                row['filename_identity']=list(ident)
                row['same_segment_stem']=entry.name.startswith(Path(spec['required_path']).name.removesuffix('.reduced.root')+'.')
                spec['matches'].append(row)
        print('LISTED',path,len(entries),flush=True)
    for spec in specs:
        raw='/eos/lhaaso/ai/wcda/raw/2022/'+spec['name'][7:11]+'/'+spec['name']
        try:
            with uproot.open('root://eos01.ihep.ac.cn/'+raw,timeout=60,
                             handler=uproot.source.xrootd.MultithreadedXRootDSource,num_workers=1) as f:
                spec['raw_metadata']=dict(path=raw,uuid=str(f.file.uuid),keys=f.keys(),
                    event_has_poolused='poolused' in f['t_eventout'],header_branches=f['t_eventout_h'].keys())
        except Exception as e:spec['raw_metadata']=dict(path=raw,error=repr(e))
    result=dict(specs=specs,directories=directories,host=os.uname().nodename,at=time.time(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All 31 January raw/wcda directories and failed AI file metadata; no inference, Nhit recomputation or publication')
    output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('specs',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.specs,a.output)
