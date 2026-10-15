"""Exact Jan28 replica route, bound to independent metadata observations."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

import uproot
from common import sha256

EVIDENCE_SHA = '601234660787675712483945250327c069a727c967183b1b249722cd2acd519e'
NAME = 'Esg20220128_06.root'
SERVER = {'host': 'eos92.ihep.ac.cn', 'port': 1095}
FIELDS = ['irun', 'ies', 'iseq', 'poolused']
ROW_FIELDS = ['header_read_uri', 'header_replica_server', 'header_replica_evidence_sha256']


def load_route(path, expected):
    if sha256(path) != EVIDENCE_SHA:
        raise ValueError('replica evidence digest mismatch')
    evidence = json.loads(path.read_text())
    good = [json.loads(case['stdout']) for case in evidence['cases']
            if case['mode'] == 'uproot' and case['fsid'] == 40208 and case['returncode'] == 0]
    uri = 'root://eos01.ihep.ac.cn/' + expected['path'] + '?eos.forcefsid=40208'
    if len(good) != 1:
        raise ValueError('replica evidence not unique')
    item = good[0]
    if (not item['reliable'] or item['uri'] != uri or item['data_servers'] != [SERVER] or
            item['uuid'] != expected['uuid'] or item['header'] != expected['header'] or item['entries'] != 1):
        raise ValueError('replica evidence identity/server mismatch')
    return dict(uri=uri, server=SERVER, evidence_sha256=EVIDENCE_SHA,
                offsets=item['metadata_offsets'], hashes=item['metadata_hashes'])


def check_hashes(hashes, route):
    if set(hashes) != {'wcdah', 'streamers'}:
        raise ValueError('replica metadata coverage mismatch')
    for label, observed in hashes.items():
        expected = route['hashes'][label]
        if observed['sha256'] != expected['sha256'] or observed['bytes'] != expected['bytes']:
            raise ValueError('replica metadata byte mismatch: ' + label)


def spans(route):
    offsets = route['offsets']
    return [('wcdah', offsets['wcdah'], offsets['wcdah_bytes']),
            ('streamers', offsets['streamers'], offsets['streamer_bytes'])]


def native_hashes(f, route):
    hashes = {}
    for label, offset, size in spans(route):
        if size <= 0 or size > 4 * 1024**2:
            raise ValueError('unexpected metadata size')
        buf = ctypes.create_string_buffer(size)
        if f.ReadBuffer(buf, offset, size):
            raise ValueError('native metadata read failed')
        hashes[label] = dict(sha256=hashlib.sha256(buf.raw).hexdigest(), bytes=size)
    check_hashes(hashes, route)
    return hashes


def read_pinned(expected, route, include_access=False):
    """Read only metadata and four integer header fields; no hit branches."""
    with uproot.open(route['uri'], timeout=60,
                     handler=uproot.source.xrootd.MultithreadedXRootDSource, num_workers=1) as f:
        workers = f.file.source._executor.workers
        servers = []
        for worker in workers:
            value = worker.resource.file.get_property('DataServer')
            parsed = urlsplit(value if '://' in value else 'root://' + value)
            servers.append(dict(host=parsed.hostname, port=parsed.port))
        if servers != [route['server']]:
            raise ValueError('pinned replica served by unexpected endpoint')
        hashes = {}
        for label, offset, size in spans(route):
            raw = bytes(f.file.source.chunk(offset, offset + size).raw_data)
            hashes[label] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        check_hashes(hashes, route)
        keys = sum(key.split(';')[0] == 'wcdah' for key in f.keys())
        tree = f['wcdah']
        values = tree.arrays(FIELDS, library='np')
        if (keys != 1 or tree.num_entries != 1 or any(v.dtype.kind not in 'iu' for v in values.values()) or
                str(f.file.uuid) != expected['uuid'] or
                {k: v.tolist() for k, v in values.items()} != expected['header']):
            raise ValueError('pinned replica header conflict')
        result = dict(uri=route['uri'], data_servers=servers, uuid=str(f.file.uuid),
                    header={k: v.tolist() for k, v in values.items()}, header_keys=keys,
                    entries=tree.num_entries, metadata_hashes=hashes, evidence_sha256=EVIDENCE_SHA)
        if include_access:
            # EOS supplies this short-lived URL. Pass it only through a private
            # child-process pipe to ROOT; never persist capabilities in evidence.
            access_uri = workers[0].resource.file.get_property('LastURL')
            parsed = urlsplit(access_uri)
            if parsed.scheme != 'root' or dict(host=parsed.hostname, port=parsed.port) != SERVER:
                raise ValueError('resolved access URL differs from verified endpoint')
            result['_access_uri'] = access_uri
        return result


def validate_probe(actual, policy):
    route = policy.get('replica')
    if route is None:
        return
    proof = actual['replica_probe']
    expected = policy['recovered'][NAME]
    if (actual['path'] != route['uri'] or actual['opener'] != 'TNetXNGFile' or
            actual['native_server'] != route['server'] or
            proof['uri'] != route['uri'] or proof['data_servers'] != [route['server']] or
            proof['evidence_sha256'] != EVIDENCE_SHA or proof['uuid'] != expected['uuid'] or
            proof['header'] != expected['header'] or proof['entries'] != 1 or proof['header_keys'] != 1):
        raise ValueError('pinned source preflight binding mismatch')
    check_hashes(actual['metadata_hashes'], route)
    check_hashes(proof['metadata_hashes'], route)


if __name__ == '__main__':
    # This child intentionally does not import ROOT: diagnostics showed that
    # the mixed-library process did not consistently honor the route request.
    try:
        from policy import load_policy
        stage = Path(sys.argv[1])
        policy = load_policy(stage, json.loads((stage/'config.json').read_text()))
        result = read_pinned(policy['recovered'][NAME], policy['replica'], include_access=True)
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__, 'error': 'pinned metadata preflight failed'}))
        sys.exit(1)
