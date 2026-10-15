"""Exact, digest-bound authorization for two missing segment headers only."""
import hashlib
import json
from pathlib import Path

from common import KEYS, sha256

AUTH_SHA = '2da7534d5a7d65cd36c4c9d889b35e43aa340b70aa62c41c819eb460ef09a5f8'
SCOPES = {'Esg20220102_12.root': (139579, 1, 3336),
          'Esg20220103_05.root': (139579, 2, 4700)}
RECOVERED = {'Esg20220123_00.root', 'Esg20220128_06.root'}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_policy(folder, config):
    auth = Path(folder) / 'authorization.json'
    evidence = Path(folder) / 'header_evidence.json'
    if sha256(auth) != AUTH_SHA or config['authorization_sha256'] != AUTH_SHA:
        raise ValueError('authorization digest mismatch')
    if sha256(evidence) != config['header_evidence_sha256']:
        raise ValueError('header investigation evidence changed')
    records = json.loads(auth.read_text())['records']
    if {r['name']: tuple(r['segment'][k] for k in KEYS[:3]) for r in records} != SCOPES:
        raise ValueError('authorization scope changed')
    recovered = {}
    for spec in json.loads(evidence.read_text())['specs']:
        if spec['name'] in RECOVERED:
            exact = [m for m in spec['matches'] if m.get('reliable_header') and m['same_segment_stem']]
            if len(exact) != 1 or exact[0]['path'] != spec['required_path']:
                raise ValueError('recovered header is not unique and exact')
            recovered[spec['name']] = exact[0]
    if set(recovered) != RECOVERED:
        raise ValueError('incomplete recovered header evidence')
    return dict(records={r['name']: r for r in records}, recovered=recovered,
                binding=dict(authorization_sha256=AUTH_SHA,
                             header_evidence_sha256=config['header_evidence_sha256']))


def authorization(task, policy):
    name = task['name']
    if name not in SCOPES and name not in RECOVERED:
        raise ValueError('file outside isolated repair scope')
    r = policy['records'].get(name)
    if r is not None:
        stamp = task['input']
        if (r['target_path'] != task['target'] or r['target_uuid'] != stamp['uuid'] or
                r['target_entries'] != stamp['entries'] or
                r['target_header_sha256'] != stamp['header_sha256'] or
                r['input_snapshot_sha256'] != json_sha(stamp)):
            raise ValueError('authorization target snapshot mismatch')
    return r


def override_row(spec, task, policy):
    r = authorization(task, policy)
    if r is None or tuple(spec['key']) != SCOPES[task['name']]:
        return None
    if spec['path'] != r['segment']['missing_reduced_path']:
        raise ValueError('authorized header path mismatch')
    return dict(zip(KEYS[:3], spec['key']), poolused=7, attempts=0,
                path=spec['path'], uuid='', source_filename=spec['header']['filename'],
                header_sha256=json_sha(spec['header']), reduced_header_sha256='',
                poolused_source='user_confirmed_override',
                poolused_authorization_sha256=AUTH_SHA, reduced_header_missing=1)


def check_rows(rows, task, policy):
    r = authorization(task, policy)
    manual = []
    for row in rows:
        if not all(row.get(k) for k in ['path', 'source_filename', 'header_sha256']):
            raise ValueError('missing segment identity provenance')
        reader=row.get('header_reader')
        if reader is not None:
            if reader not in ['uproot','native_ROOT','user_attestation']:
                raise ValueError('unknown header reader')
            if (reader=='user_attestation') != (row.get('poolused_source')=='user_confirmed_override'):
                raise ValueError('header reader/source conflict')
            if reader=='native_ROOT' and task['name']!='Esg20220128_06.root':
                raise ValueError('native reader outside diagnosed file')
        if row.get('poolused_source') == 'user_confirmed_override':
            if (r is None or tuple(row[k] for k in KEYS[:3]) != SCOPES[task['name']] or
                    row['path'] != r['segment']['missing_reduced_path'] or
                    row['poolused'] != 7 or row['uuid'] != '' or
                    row['reduced_header_sha256'] != '' or row['attempts'] != 0 or
                    row.get('reduced_header_missing') != 1 or
                    row.get('poolused_authorization_sha256') != AUTH_SHA):
                raise ValueError('invalid or out-of-scope manual header exception')
            manual.append(row)
        elif (row.get('poolused_source') != 'reduced_header' or
              not row.get('uuid') or not row.get('reduced_header_sha256') or
              row.get('reduced_header_missing') != 0 or
              row.get('poolused_authorization_sha256') != '' or
              not 1 <= row['attempts'] <= 3):
            raise ValueError('invalid verified header provenance')
    if len(manual) != int(r is not None):
        raise ValueError('authorized segment coverage mismatch')
    evidence = policy['recovered'].get(task['name'])
    if evidence:
        matches = [row for row in rows if row['path'] == evidence['path']]
        if (len(matches) != 1 or matches[0]['uuid'] != evidence['uuid'] or
                matches[0]['reduced_header_sha256'] != json_sha(evidence['header']) or
                matches[0]['poolused_source'] != 'reduced_header'):
            raise ValueError('recovered header differs from independent investigation')
