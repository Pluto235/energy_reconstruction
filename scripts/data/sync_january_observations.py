#!/usr/bin/env python3
"""Receive validated January files, persist ROOT/byte receipts, then reclaim staging."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from observation_state import KEEP, ack_matches, atomic_json, file_path, make_receipt, record_id

STATE = Path('/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january')
DESTINATION = Path('/mnt/newdisk/shijie/WCDA_obs')
REMOTE = 'liushijie@lxlogin.ihep.ac.cn'
WORK = '/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013'
SOURCE = '/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013'


def validate_report(report):
    day = report['day']
    if day not in {f'202201{n:02d}' for n in range(1, 32)}:
        raise ValueError(f'Unexpected day: {day}')
    names = [r['name'] for r in report['files']]
    if not names or len(names) != report['file_count'] or len(set(names)) != len(names):
        raise ValueError('Duplicate filenames or inconsistent audit count')
    for record in report['files']:
        output, source = record['output'], record['source']
        if (file_path(DESTINATION, record['name']).parent.name != day[4:] or output['bytes'] <= 0
                or output['entries'] != source['selected_entries']
                or output['header_entries'] != source['header_entries']):
            raise ValueError(f"Invalid audited file: {record['name']}")
        record_id(record)
    if sum(r['output']['bytes'] for r in report['files']) != report['bytes']:
        raise ValueError('Inconsistent byte total')


def read_metadata(path):
    import uproot
    with uproot.open(path) as file:
        tree, header = file['t_eventout'], file['t_eventout_h']
        if tree.classname != 'TTree' or header.classname != 'TTree' or file.file.fEND > path.stat().st_size:
            raise RuntimeError(f'Invalid or truncated ROOT trees: {path}')
        return dict(entries=tree.num_entries, header_entries=header.num_entries,
                    branches=set(tree.keys(recursive=False)),
                    branch_entries=[b.num_entries for b in tree.values(recursive=False)],
                    header_branches=sorted(header.keys(recursive=False)))


def verify(report, directory):
    validate_report(report)
    for record in report['files']:
        path = directory / record['name']
        if path.is_symlink() or path.stat().st_size != record['output']['bytes']:
            raise RuntimeError(f'Transferred size/path mismatch: {path}')
        with path.open('rb') as file:
            if file.read(4) != b'root':
                raise RuntimeError(f'Transferred ROOT signature mismatch: {path}')
        metadata = read_metadata(path)
        if (metadata['branches'] != KEEP or metadata['entries'] != record['output']['entries']
                or metadata['header_entries'] != record['output']['header_entries']
                or any(n != metadata['entries'] for n in metadata['branch_entries'])
                or ('header_branches' in record['source'] and metadata['header_branches'] != record['source']['header_branches'])):
            raise RuntimeError(f'Transferred ROOT metadata mismatch: {path}')


def acknowledge(ssh, record, receipt):
    code = ('import sys,json;sys.path.insert(0,' + repr(WORK + '/code') + '); '
            'from observation_state import WORK,atomic_json,ack_matches,cleanup_received; '
            'r=json.loads((WORK/"audit_files"/' + repr(record['name'] + '.json') + ').read_text())["files"][0]; '
            'a=WORK/"acks"/' + repr(record['name'] + '.ack') + '; '
            'receipt=json.loads(' + repr(json.dumps(receipt)) + '); '
            'assert ack_matches(r,json.dumps(receipt)),"Receipt identity changed"; '
            'atomic_json(a,receipt) if not (a.exists() and ack_matches(r,a.read_text())) else None; '
            'print(json.dumps({"cleaned":cleanup_received(r["name"])}))')
    raw = subprocess.check_output(ssh + [REMOTE, 'python3 -c ' + shlex.quote(code)], text=True)
    return json.loads(raw)['cleaned']


def receive(ssh, report, source_ack, source_pending=True):
    record = report['files'][0]
    name = record['name']
    target = DESTINATION / report['day'][4:]
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = STATE / 'receipts' / (name + '.json')
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    confirmed = previous is not None or (source_ack and ack_matches(record, source_ack))
    if previous and previous['record_id'] != record_id(record):
        raise RuntimeError(f'Confirmed target has a different receipt; preserved: {name}')
    transfer_seconds = 0.0
    before = time.monotonic()
    try:
        verify(report, target)
    except (OSError, RuntimeError, ValueError, KeyError):
        if confirmed:
            raise RuntimeError(f'Confirmed target failed metadata checks; preserved: {name}')
        incoming = STATE / 'incoming' / report['day'][4:]
        incoming.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        subprocess.run(['rsync', '-rt', '--ignore-times', '--partial', '--partial-dir=.rsync-partial',
                        '--timeout=1200', '-e', shlex.join(ssh),
                        f"{REMOTE}:{SOURCE}/{report['day'][4:]}/{name}", str(incoming) + '/'], check=True)
        transfer_seconds = time.monotonic() - start
        before = time.monotonic(); verify(report, incoming)
        with (incoming / name).open('rb') as file:
            os.fsync(file.fileno())
        if (target / name).exists():
            saved = STATE / 'preserved_unconfirmed' / (name + f'.{time.time_ns()}')
            saved.parent.mkdir(exist_ok=True); (target / name).replace(saved)
        (incoming / name).replace(target / name)
        fd = os.open(str(target), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    verify_seconds = time.monotonic() - before
    receipt = previous or make_receipt(record)
    if not previous:
        if source_ack and ack_matches(record, source_ack):
            receipt['verification'] = 'legacy-or-v2-ack+ROOT-metadata-and-bytes'
        receipt.update(record=record, transfer_seconds=transfer_seconds, verify_seconds=verify_seconds)
        atomic_json(receipt_path, receipt)
    start = time.monotonic()
    cleaned = not source_pending
    if source_pending or not (source_ack and ack_matches(record, source_ack)):
        cleaned = acknowledge(ssh, record, receipt)
    print('RECEIVED ' + json.dumps(dict(name=name, transfer_seconds=transfer_seconds,
          verify_seconds=verify_seconds, acknowledgment_cleanup_seconds=time.monotonic()-start,
          ready_to_ack_seconds=time.time()-record['prepared_at'] if 'prepared_at' in record else None,
          cleaned=cleaned)), flush=True)
    return record


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-S', str(STATE / 'ihep.sock'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30']
    command = '''from pathlib import Path
import json
w=Path(WORK_VALUE);root=Path(SOURCE_VALUE)
r={k:[json.loads(p.read_text()) for p in (w/d).glob('*.json')] for k,d in [('files','audit_files'),('days','audit')]}
r['acks']={p.stem:p.read_text() for p in (w/'acks').glob('*.ack')}
r['manifests']={p.stem:[u.rsplit('/',1)[-1] for u in p.read_text().splitlines()] for p in (w/'manifests').glob('202201??.txt')}
r['pending']=[p.name for p in root.glob('01??/Esg*.root*')]
import sys
sys.path.insert(0,str(w/'code'))
from observation_state import StagingQueue
r['queue']=StagingQueue().status()
print(json.dumps(r))'''.replace('WORK_VALUE', repr(WORK)).replace('SOURCE_VALUE', repr(SOURCE))
    received, completed, errors, retry_at = {}, {}, {}, {}
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        try:
            raw = subprocess.check_output(ssh + [REMOTE, 'python3 -c ' + shlex.quote(command)], text=True)
            data = json.loads(raw)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f'SOURCE_POLL_RETRY {exc}', flush=True); time.sleep(10); continue
        manifests = data['manifests']
        expected = {name for names in manifests.values() for name in names}
        if len(manifests) != 31 or len(expected) != 681 or sum(map(len, manifests.values())) != 681:
            raise ValueError('Source manifest snapshot must contain exactly 31 dates / 681 unique names')
        for report in sorted(data['files'], key=lambda r: r['files'][0].get('prepared_at', 0)):
            validate_report(report)
            if report['file_count'] != 1:
                raise ValueError('Per-file audit must contain exactly one file')
            record = report['files'][0]; name = record['name']
            if name not in expected:
                raise ValueError(f'Audit is outside the source manifests: {name}')
            if time.monotonic() < retry_at.get(name, 0):
                continue
            try:
                if name not in received:
                    received[name] = receive(ssh, report, data['acks'].get(name), name in data['pending'])
                elif name in data['pending'] or not ack_matches(record, data['acks'].get(name, '')):
                    receipt = json.loads((STATE/'receipts'/(name+'.json')).read_text())
                    acknowledge(ssh, record, receipt)
                errors.pop(name, None)
            except Exception as exc:
                errors[name] = str(exc); retry_at[name] = time.monotonic() + 30
                print(f'FILE_RETRY {name} {exc}', flush=True)
        for report in data['days']:
            validate_report(report); day = report['day']
            if {r['name'] for r in report['files']} != set(manifests[day]):
                raise ValueError(f'Day report differs from the original manifest: {day}')
            if all(received.get(r['name']) == r for r in report['files']):
                completed[day] = report
        atomic_json(STATE/'progress.json', dict(completed_days=sorted(completed), files=len(received),
                    bytes=sum(r['output']['bytes'] for r in received.values()), errors=errors,
                    queue=data['queue'], updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        if len(completed) == 31 and set(received) == expected and not data['pending']:
            atomic_json(STATE/'completed.json', list(completed.values()))
            print('JANUARY_SYNC_COMPLETE files=681', flush=True)
            return
        time.sleep(3)
    raise TimeoutError('January preparation did not complete within 36 hours')


if __name__ == '__main__':
    main()
