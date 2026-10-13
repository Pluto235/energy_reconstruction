#!/usr/bin/env python3
"""Reuse or rebuild one day of the established pincness/branch-reduced inputs."""
import argparse
import errno
import os
import resource
import signal
import uuid
import json
from pathlib import Path
import time

import ROOT
from observation_state import DAYS, KEEP, STAGING, StagingQueue, ack_matches, atomic_json, file_lock, file_path

ROOT.gROOT.SetBatch(True)
MACRO = Path(__file__).with_name("filter_observed_pincness_loop_scoped.C")


def inspect(path, expected_entries=None, expected_headers=None, full_read=True):
    try:
        file = ROOT.TFile.Open(str(path), "READ")
    except OSError as exc:
        raise RuntimeError(f"Cannot open ROOT file: {path}") from exc
    if not file or file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered):
        if file:
            file.Close()
        raise RuntimeError(f"Invalid ROOT file: {path}")
    try:
        tree, header = file.Get("t_eventout"), file.Get("t_eventout_h")
        if not tree or not header:
            raise RuntimeError(f"Missing event/header tree: {path}")
        branches = {b.GetName() for b in tree.GetListOfBranches()}
        entries, headers = int(tree.GetEntries()), int(header.GetEntries())
        if expected_entries is not None:
            if branches != KEEP or entries != expected_entries or headers != expected_headers:
                raise RuntimeError(f"Branch/entry/header mismatch: {path}")
            if not ROOT.checked_observation_content(tree, full_read, True) or not ROOT.readable_observation_tree(header):
                raise RuntimeError(f"Unreadable retained data: {path}")
        elif not KEEP.issubset(branches):
            raise RuntimeError(f"Missing source branches: {path}")
        return {
            "entries": entries,
            "selected_entries": int(tree.GetEntries("pincness < 1.1")) if expected_entries is None else entries,
            "header_entries": headers,
            "header_branches": sorted(b.GetName() for b in header.GetListOfBranches()),
            "bytes": int(file.GetSize()),
        }
    finally:
        file.Close()


def filter_source(url, temporary, cache_mb):
    rc = ROOT.filter_observed_pincness_loop_scoped(url, str(temporary), 1.1, cache_mb)
    stats = ROOT.observation_filter_stats()
    metrics = {name: getattr(stats, name) for name in (
        "scan_seconds", "hit_seconds", "scan_bytes", "hit_bytes", "scan_calls", "hit_calls",
        "scan_cached_branches", "hit_cached_branches", "write_errno",
    )}
    source = {
        "entries": int(stats.input_entries), "selected_entries": int(stats.selected_entries),
        "header_entries": int(stats.header_entries), "bytes": int(stats.input_bytes),
        "header_branches": sorted(str(n) for n in stats.header_branches),
    }
    return rc, source, metrics


def publish_day(day, sources, audit_dir):
    paths = [audit_dir.parent / 'audit_files' / (u.rsplit('/', 1)[-1] + '.json') for u in sources]
    if not all(p.exists() for p in paths):
        return
    records = [json.loads(p.read_text())['files'][0] for p in paths]
    report = dict(day=day, cut='pincness < 1.1', kept_branches=sorted(KEEP),
                  file_count=len(records), bytes=sum(r['output']['bytes'] for r in records), files=records)
    atomic_json(audit_dir / f'{day}.json', report)


def prepare_one(url, args, config, queue):
    name = url.rsplit('/', 1)[-1]
    output = file_path(args.output_root, name)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_report = args.audit_dir.parent / 'audit_files' / f'{name}.json'
    ack = args.audit_dir.parent / 'acks' / f'{name}.ack'
    temporary = output.with_suffix('.root.partial')
    with file_lock(args.audit_dir.parent, name):
        if file_report.exists():
            record = json.loads(file_report.read_text())['files'][0]
            if record['name'] != name:
                raise ValueError(f'Audit filename mismatch: {name}')
            if ack.exists() and ack_matches(record, ack.read_text()):
                with queue.locked() as ledger:
                    output.unlink(missing_ok=True); ledger.pop(name, None)
                    atomic_json(queue.ledger, ledger)
                print(f'SKIP_CONFIRMED {name}', flush=True)
                return record
            if output.exists():
                print(f'SKIP_PENDING_TRANSFER {name}', flush=True)
                return record
            raise RuntimeError(f'Orphaned audit without output/ack; receiver must reconcile it first: {name}')
        with queue.locked() as ledger:
            temporary.unlink(missing_ok=True)
            ledger.pop(name, None); atomic_json(queue.ledger, ledger)
        archive = args.archive_root / args.day[4:] / name
        reservation = max(config['initial_reservation_bytes'], archive.stat().st_size if archive.is_file() else 0,
                          output.stat().st_size if output.exists() else 0)
        wait_seconds = 0.0
        failed_seconds = 0.0
        for attempt in range(3):
            waiting = time.monotonic()
            while not queue.try_reserve(name, reservation):
                if time.monotonic() - waiting > 6 * 3600:
                    raise TimeoutError(f'Staging queue remained full for six hours: {name}')
                time.sleep(2)
            wait_seconds += time.monotonic() - waiting
            started, cpu = time.monotonic(), time.process_time()
            metrics = {}
            reused = output.exists() or archive.is_file()
            try:
                if reused:
                    before = time.monotonic(); source = inspect(url)
                    metrics['scan_seconds'] = time.monotonic() - before
                    metrics['hit_seconds'] = 0.0
                    if not output.exists():
                        with queue.locked():
                            os.link(archive, output)
                    target = output
                else:
                    original_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
                    if original_limit[1] != resource.RLIM_INFINITY and reservation > original_limit[1]:
                        raise RuntimeError('Batch hard file-size limit is below the reservation')
                    try:
                        resource.setrlimit(resource.RLIMIT_FSIZE, (reservation, original_limit[1]))
                        rc, source, metrics = filter_source(url, temporary, config['cache_mb'])
                    finally:
                        resource.setrlimit(resource.RLIMIT_FSIZE, original_limit)
                    if rc:
                        error = OSError(metrics['write_errno'] or errno.EIO, f'ROOT filter status {rc}')
                        raise error
                    target = temporary
                before = time.monotonic()
                result = inspect(target, source['selected_entries'], source['header_entries'], config['full_read'])
                if result['header_branches'] != source['header_branches']:
                    raise RuntimeError(f'Header branch mismatch: {name}')
                metrics['validation_seconds'] = time.monotonic() - before
                metrics['validation_mode'] = 'full' if config['full_read'] else 'sample-2048'
                before = time.monotonic()
                with target.open('rb') as file:
                    os.fsync(file.fileno())
                metrics['fsync_seconds'] = time.monotonic() - before
                metrics['processing_seconds'] = time.monotonic() - started
                metrics['cpu_seconds'] = time.process_time() - cpu
                metrics['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                metrics['backpressure_seconds'] = wait_seconds
                metrics['failed_attempt_seconds'] = failed_seconds
                metrics['attempts'] = attempt + 1
                with queue.locked() as ledger:
                    if target != output:
                        temporary.replace(output)
                    ledger.pop(name, None); atomic_json(queue.ledger, ledger)
                record = dict(name=name, receipt_id=uuid.uuid4().hex, source=source, output=result,
                              reused=reused, prepared_at=time.time(), timings=metrics)
                atomic_json(file_report, dict(version=2, day=args.day, file_count=1, bytes=result['bytes'], files=[record]))
                print('PREPARED ' + json.dumps(record), flush=True)
                return record
            except Exception as exc:
                failed_seconds += time.monotonic() - started
                with queue.locked() as ledger:
                    temporary.unlink(missing_ok=True)
                    ledger.pop(name, None); atomic_json(queue.ledger, ledger)
                if output.exists():
                    raise
                if isinstance(exc, OSError) and exc.errno == errno.EFBIG:
                    reservation *= 2
                if reservation > config['capacity_bytes'] or attempt == 2:
                    raise
                print(f'RETRY {name} attempt={attempt+1} error={exc}', flush=True)
                time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('day')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('audit_dir', type=Path)
    parser.add_argument('archive_root', type=Path)
    parser.add_argument('--names', nargs='+', help='Benchmark a subset of this unchanged full-day manifest')
    args = parser.parse_args()
    if args.day not in DAYS:
        raise ValueError('Day is outside the configured observation month')
    if args.output_root.resolve() != STAGING or (args.output_root / args.day[4:]).is_symlink():
        raise ValueError("Receipt cleanup is restricted to this run's temporary staging tree")
    sources = args.manifest.read_text().splitlines()
    prefix = f'root://eos01.ihep.ac.cn//eos/lhaaso/ai/wcda/raw/2022/{args.day[4:]}/'
    if not sources or len(sources) != len(set(sources)) or any(
        not u.startswith(prefix) or file_path(args.output_root, u.rsplit('/', 1)[-1]).parent.name != args.day[4:]
        for u in sources
    ):
        raise ValueError('Invalid or empty input manifest')
    selected = set(args.names) if args.names else {u.rsplit('/', 1)[-1] for u in sources}
    if not selected.issubset({u.rsplit('/', 1)[-1] for u in sources}):
        raise ValueError('Requested filename is absent from the full source manifest')
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    for name in ['audit_files', 'acks']:
        (args.audit_dir.parent / name).mkdir(exist_ok=True)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    ROOT.gROOT.ProcessLine(f'.L {MACRO}')
    config = json.loads((args.audit_dir.parent / 'queue_config.json').read_text())
    queue = StagingQueue(args.output_root, args.audit_dir.parent)
    for url in sources:
        if url.rsplit('/', 1)[-1] in selected:
            prepare_one(url, args, config, queue)
    publish_day(args.day, sources, args.audit_dir)


if __name__ == '__main__':
    main()
