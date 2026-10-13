"""Durable observation receipts and a shared byte budget for this restoration."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import time
import uuid

STAGING = Path('/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013')
WORK = Path('/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013')
KEEP = set(re.findall(r'"(\w+)"', Path(__file__).with_name('filter_observed_pincness_loop_scoped.C').read_text().split('return {', 1)[1].split('};', 1)[0]))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.partial')
    with temporary.open('w') as file:
        json.dump(value, file, indent=2); file.write('\n')
        file.flush(); os.fsync(file.fileno())
    temporary.replace(path)
    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def record_id(record):
    if re.fullmatch(r'[a-f0-9]{32}', record.get('receipt_id', '')):
        return 'v2:' + record['receipt_id']
    digest = record.get('output', {}).get('sha256', '')
    if re.fullmatch(r'[a-f0-9]{64}', digest):
        return 'legacy-sha256:' + digest
    raise ValueError('Missing receipt identity')


def make_receipt(record):
    return dict(version=2, record_id=record_id(record), name=record['name'],
                bytes=record['output']['bytes'], entries=record['output']['entries'],
                header_entries=record['output']['header_entries'],
                verification='ROOT-metadata-and-bytes', confirmed_at=time.time())


def ack_matches(record, text):
    digest = record.get('output', {}).get('sha256')
    if not record.get('receipt_id') and re.fullmatch(r'[a-f0-9]{64}', digest or '') and text.strip() == digest:
        return True
    try:
        ack = json.loads(text)
        return (ack['version'] == 2 and ack['record_id'] == record_id(record)
                and ack['name'] == record['name'] and ack['bytes'] == record['output']['bytes']
                and ack['entries'] == record['output']['entries']
                and ack['header_entries'] == record['output']['header_entries'])
    except (ValueError, KeyError, TypeError):
        return False


def file_path(root, name):
    if not re.fullmatch(r'Esg202201(?:0[1-9]|[12][0-9]|3[01])_(?:[01][0-9]|2[0-3])\.root', name):
        raise ValueError(f'Invalid observation name: {name}')
    root = Path(root)
    path = root / name[7:11] / name
    if path.parent.is_symlink() or path.is_symlink() or path.with_suffix('.root.partial').is_symlink():
        raise ValueError(f'Unexpected staging symlink: {path}')
    return path


@contextmanager
def file_lock(work, name, blocking=True):
    file_path(STAGING, name)
    path = Path(work) / 'file_locks' / (name + '.lock')
    path.parent.mkdir(exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class StagingQueue:
    def __init__(self, root=STAGING, work=WORK):
        self.root, self.work = Path(root), Path(work)
        self.ledger = self.work / 'queue_ledger.json'
        self.config = self.work / 'queue_config.json'

    @contextmanager
    def locked(self):
        with (self.work / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(self.ledger.read_text()) if self.ledger.exists() else {}
            yield ledger
            fcntl.flock(lock, fcntl.LOCK_UN)

    def _usage(self, ledger):
        actual = {}
        for path in self.root.glob('01??/Esg*.root*'):
            if path.is_symlink():
                raise ValueError(f'Unexpected queue symlink: {path}')
            name = path.name.split('.root', 1)[0] + '.root'
            actual[name] = actual.get(name, 0) + path.stat().st_size
        return sum(max(actual.get(n, 0), r['bytes']) for n, r in ledger.items()) + sum(
            size for n, size in actual.items() if n not in ledger)

    def usage(self):
        with self.locked() as ledger:
            return self._usage(ledger)

    def status(self):
        with self.locked() as ledger:
            config = json.loads(self.config.read_text())
            previous = json.loads((self.work / 'queue_status.json').read_text()) if (self.work / 'queue_status.json').exists() else {}
            used = self._usage(ledger)
            status = dict(used_bytes=used, capacity_bytes=config['capacity_bytes'], reservations=len(ledger),
                          peak_used_bytes=max(used, previous.get('peak_used_bytes', 0)), updated_at=time.time())
            atomic_json(self.work / 'queue_status.json', status)
            return status

    def try_reserve(self, name, amount):
        file_path(self.root, name)
        config = json.loads(self.config.read_text())
        if amount <= 0 or amount > config['capacity_bytes']:
            raise ValueError('File reservation exceeds staging capacity')
        with self.locked() as ledger:
            old = ledger.pop(name, None)
            used = self._usage(ledger)
            existing = sum(p.stat().st_size for p in file_path(self.root, name).parent.glob(name + '*'))
            projected = used - existing + max(existing, amount)
            if projected > config['capacity_bytes']:
                return False
            ledger[name] = {'bytes': amount, 'reserved_at': time.time()}
            atomic_json(self.ledger, ledger)
            previous = json.loads((self.work / 'queue_status.json').read_text()) if (self.work / 'queue_status.json').exists() else {}
            atomic_json(self.work / 'queue_status.json', dict(used_bytes=projected,
                        capacity_bytes=config['capacity_bytes'], reservations=len(ledger),
                        peak_used_bytes=max(projected, previous.get('peak_used_bytes', 0)), updated_at=time.time()))
            return True

    def release(self, name):
        with self.locked() as ledger:
            ledger.pop(name, None)
            atomic_json(self.ledger, ledger)


def cleanup_received(name):
    if STAGING.resolve() != STAGING:
        raise ValueError('Cleanup root is not the task staging tree')
    with file_lock(WORK, name, blocking=False) as acquired:
        if not acquired:
            return False
        report = json.loads((WORK / 'audit_files' / f'{name}.json').read_text())
        record = report['files'][0]
        if record['name'] != name or not ack_matches(record, (WORK / 'acks' / f'{name}.ack').read_text()):
            raise ValueError('Cleanup requires a matching durable acknowledgment')
        output = file_path(STAGING, name)
        # The old archive remains a separate hard link; only this run's alias is removed.
        with StagingQueue().locked() as ledger:
            output.unlink(missing_ok=True)
            ledger.pop(name, None)
            atomic_json(WORK / 'queue_ledger.json', ledger)
        return True
