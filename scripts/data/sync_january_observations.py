#!/usr/bin/env python3
"""Receive audited January files; acknowledge checksums before IHEP removes staging."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

STATE = Path('/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january')
DESTINATION = Path('/mnt/newdisk/shijie/WCDA_obs')
REMOTE = 'liushijie@lxlogin.ihep.ac.cn'
WORK = '/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013'
SOURCE = '/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013'


def validate_report(report):
    day = report['day']
    if day not in {f'202201{n:02d}' for n in range(1, 32)}:
        raise ValueError(f'Unexpected day: {day}')
    names = [record['name'] for record in report['files']]
    if not names or len(names) != report['file_count'] or len(set(names)) != len(names):
        raise ValueError('Duplicate filenames or inconsistent audit count')
    for record in report['files']:
        if (not re.fullmatch(rf'Esg{day}_\d{{2}}\.root', record['name'])
                or record['output']['bytes'] <= 0
                or not re.fullmatch(r'[a-f0-9]{64}', record['output']['sha256'])):
            raise ValueError(f"Invalid audited file: {record['name']}")
    if sum(record['output']['bytes'] for record in report['files']) != report['bytes']:
        raise ValueError('Inconsistent byte total')


def verify(report, directory, checksums=True):
    validate_report(report)
    for record in report['files']:
        path = directory / record['name']
        if path.stat().st_size != record['output']['bytes']:
            raise RuntimeError(f'Transferred size mismatch: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as file:
            if file.read(4) != b'root':
                raise RuntimeError(f'Transferred ROOT signature mismatch: {path}')
            if checksums:
                file.seek(0)
                for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                    digest.update(block)
        if checksums and digest.hexdigest() != record['output']['sha256']:
            raise RuntimeError(f'Transferred SHA-256 mismatch: {path}')


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-S', str(STATE / 'ihep.sock'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30']
    command = ('import pathlib,json; w=pathlib.Path(' + repr(WORK) + '); '
               'print(json.dumps({k:[json.loads(p.read_text()) for p in (w/d).glob("*.json")] '
               'for k,d in [("files","audit_files"),("days","audit")]}))')
    completed, received = {}, {}
    deadline = time.monotonic() + 36 * 3600
    while len(completed) < 31:
        if time.monotonic() > deadline:
            raise TimeoutError('January preparation did not complete within 36 hours')
        raw = subprocess.check_output(ssh + [REMOTE, 'python3 -c ' + shlex.quote(command)], text=True)
        reports = json.loads(raw)
        for report in reports['files']:
            validate_report(report)
            if report['file_count'] != 1:
                raise ValueError('Per-file receipt must contain exactly one file')
            record = report['files'][0]
            name, digest = record['name'], record['output']['sha256']
            if received.get(name, {}).get('output', {}).get('sha256') == digest:
                continue
            target = DESTINATION / report['day'][4:]
            target.mkdir(exist_ok=True)
            try:
                verify(report, target)
            except (OSError, RuntimeError):
                subprocess.run([
                    'rsync', '-rt', '--ignore-times', '--partial', '--partial-dir=.rsync-partial', '--timeout=1200',
                    '-e', shlex.join(ssh), f"{REMOTE}:{SOURCE}/{report['day'][4:]}/{name}", str(target) + '/',
                ], check=True)
                verify(report, target)
            ack_code = ('import pathlib; p=pathlib.Path(' + repr(WORK + '/acks/' + name + '.ack') + '); '
                        't=p.with_suffix(".partial"); t.write_text(' + repr(digest + '\n') + '); t.replace(p)')
            subprocess.run(ssh + [REMOTE, 'python3 -c ' + shlex.quote(ack_code)], check=True)
            received[name] = record
            print(f'SHA256_RECEIVED {name} bytes={record["output"]["bytes"]}', flush=True)
        for report in reports['days']:
            validate_report(report)
            day = report['day']
            if day in completed:
                continue
            if all(received.get(r['name']) == r for r in report['files']):
                verify(report, DESTINATION / day[4:], checksums=False)
                completed[day] = report
                print(f'DAY_COMPLETE {day} files={report["file_count"]}', flush=True)
        progress = {
            'completed_days': sorted(completed), 'files': len(received),
            'bytes': sum(r['output']['bytes'] for r in received.values()),
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        }
        temporary = STATE / 'progress.json.partial'
        temporary.write_text(json.dumps(progress, indent=2) + '\n')
        temporary.replace(STATE / 'progress.json')
        if len(completed) < 31:
            time.sleep(15)
    if len(received) != 681:
        raise RuntimeError('January inventory differs from the 681-file source snapshot')
    (STATE / 'completed.json').write_text(json.dumps(list(completed.values()), indent=2) + '\n')
    print('JANUARY_SYNC_COMPLETE files=681', flush=True)


if __name__ == '__main__':
    main()
