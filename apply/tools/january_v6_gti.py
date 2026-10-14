#!/usr/bin/env python3
"""Audit original January time friends and write exact interval rows for Stage F."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

GAP_SECONDS = 60.0
MJD_TOLERANCE_SECONDS = 2.e-6
TIME_FIELDS = ['mjd', 'match_status', 'unix_sec', 'unix_subsec', 'tai_sec', 'tai_subsec']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path):
    stat = Path(path).stat()
    return [stat.st_ino, stat.st_size, stat.st_mtime_ns]


def check_binding(path, binding):
    path = Path(path)
    if path.is_symlink() or ('stat' in binding and file_stat(path) != binding['stat']):
        raise ValueError(f'Source changed: {path}')
    if 'sha256' in binding and sha256(path) != binding['sha256']:
        raise ValueError(f'Source hash changed: {path}')


def validate_times(arrays):
    status = np.asarray(arrays['match_status'])
    if status.dtype.kind not in 'iu' or np.any((status < 0) | (status > 6)):
        raise ValueError('Invalid recovery status')
    selected = status == 0
    data = {key: np.asarray(arrays[key])[selected] for key in TIME_FIELDS if key != 'match_status'}
    if any(not np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError('Invalid non-finite status-0 time')
    if np.any(data['unix_sec'] <= 0) or np.any(data['tai_sec'] <= 0) or np.any(data['mjd'] <= 0):
        raise ValueError('Invalid non-positive status-0 time')
    if any(np.any((data[key] < 0) | (data[key] >= 1)) for key in ('unix_subsec', 'tai_subsec')):
        raise ValueError('Invalid time fraction')
    # All accepted January timestamps use the measured TAI-UTC offset of 37 seconds.
    if (np.any(data['tai_sec'] - data['unix_sec'] != 37)
            or np.any(data['tai_subsec'] != data['unix_subsec'])):
        raise ValueError('TAI/Unix time inconsistent')
    expected = 40587. + (data['unix_sec'] + data['unix_subsec']) / 86400.
    errors = np.abs(data['mjd'] - expected) * 86400.
    maximum = float(errors.max()) if errors.size else 0.
    # Two microseconds cover float64 MJD/Unix rounding; no clock corrections are applied.
    if maximum > MJD_TOLERANCE_SECONDS:
        raise ValueError(f'MJD/Unix time inconsistent: {maximum:.9g} seconds')
    return {'status': np.bincount(status, minlength=7).tolist(),
            'valid_entries': int(selected.sum()), 'max_mjd_unix_error_seconds': maximum}


def intervals_from_mjd(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError('Invalid GTI timestamps')
    negative_steps = int(np.count_nonzero(np.diff(values) < 0))
    ordered = np.sort(values)
    differences = np.diff(ordered)
    gaps = np.flatnonzero(differences > GAP_SECONDS / 86400.)
    boundaries = np.concatenate(([0], gaps + 1, [ordered.size]))
    intervals = [(float(ordered[start]), float(ordered[stop - 1]))
                 for start, stop in zip(boundaries[:-1], boundaries[1:]) if stop > start]
    return intervals, {'interval_count': len(intervals), 'original_order_negative_step_count': negative_steps,
                       'duplicate_time_count': int(np.count_nonzero(differences == 0)),
                       'sorted_gap_count': int(gaps.size),
                       'sorted_gap_seconds': float(differences[gaps].sum() * 86400.),
                       'duration_seconds': float(sum((stop - start) * 86400. for start, stop in intervals)),
                       'singleton_interval_count': sum(stop == start for start, stop in intervals)}


def merge_intervals(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
            raise ValueError('Invalid GTI interval')
        if stop == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def load_tasks(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest['result'] != 'PASS' or manifest['time_label'] != '202201'
            or manifest['contract']['gti_gap_seconds'] != GAP_SECONDS):
        raise ValueError('Invalid January source gate or frozen gap rule')
    accepted_paths = [p for p in manifest['hashes'] if Path(p).name == 'verify_66080.json']
    if len(accepted_paths) != 1:
        raise ValueError('Missing/ambiguous accepted time verification report')
    report_path = accepted_paths[0]
    check_binding(report_path, {'sha256': manifest['hashes'][report_path]})
    verification = json.loads(Path(report_path).read_text())
    if not verification['ok'] or verification['config']['time_root'] != manifest['original_time_root']:
        raise ValueError('Accepted original time source mismatch')
    accepted = {record['relative_path']: record for record in verification['results']}
    if len(accepted) != len(verification['results']):
        raise ValueError('Duplicate accepted inventory')
    rows = manifest['files']
    relatives = [row['relative_path'] for row in rows]
    if len(rows) != manifest['file_count'] or len(set(relatives)) != len(rows):
        raise ValueError('Invalid January inventory')
    tasks = []
    for row in rows:
        relative = row['relative_path']
        match = re.fullmatch(r'(01\d{2})/Esg202201(\d{2})_(\d{2})\.root', relative)
        if (not match or match[1] != '01' + match[2] or not 1 <= int(match[2]) <= 31
                or not 0 <= int(match[3]) <= 23 or relative not in accepted):
            raise ValueError(f'Invalid January inventory path: {relative}')
        record = accepted[relative]
        if (not record['ok'] or row['original_entries'] != record['original_entries']
                or row['prediction_entries'] != record['prediction_entries']):
            raise ValueError(f'Accepted entry count mismatch: {relative}')
        path = str(Path(manifest['original_time_root']) / Path(relative).with_suffix('.time.root'))
        if path not in record['sources']:
            raise ValueError(f'Missing accepted original time binding: {path}')
        tasks.append({'relative_path': relative, 'time_path': path, 'accepted': record,
                      'original_root': verification['config'].get('original_root')})
    if (sum(row['original_entries'] for row in rows) != manifest['original_entries']
            or sum(row['prediction_entries'] for row in rows) != manifest['prediction_entries']
            or np.sum([task['accepted']['original_status'] for task in tasks], axis=0).tolist()
            != manifest['original_status']):
        raise ValueError('Manifest total/source status mismatch')
    return manifest, sorted(tasks, key=lambda task: task['relative_path'])


def process_file(task):
    import uproot

    start_clock = time.perf_counter()
    path = Path(task['time_path'])
    record = task['accepted']
    summary_path = path.with_suffix('.summary.json')
    exceptions_path = path.with_suffix('.exceptions.jsonl')
    bindings = {**record['sources'], **record['sidecars']}
    recovery = record['recovery_receipt']
    bindings[recovery['path']] = {'sha256': recovery['sha256']}
    for name, binding in bindings.items():
        check_binding(name, binding)
    summary = json.loads(summary_path.read_text())
    exceptions = [json.loads(line) for line in exceptions_path.read_text().splitlines()]
    reasons = Counter(row['reason'] for row in exceptions)
    fallback_count = sum(bool(row['ies_fallback']) for row in exceptions)
    excluded = [row for row in exceptions if row['status'] != 0]
    expected_entries = record['original_entries']
    if (summary['entries'] != expected_entries or summary['status'] != record['original_status']
            or fallback_count != summary['ies_fallback']
            or len(excluded) != sum(summary['status'][1:])
            or any(not 0 <= row['entry_index'] < expected_entries for row in exceptions)):
        raise ValueError(f'Recovery summary/exception accounting mismatch: {path}')
    chunks = []
    status = np.zeros(7, dtype=np.int64)
    max_error = 0.
    executor = uproot.source.futures.TrivialExecutor()
    with uproot.open(path, decompression_executor=executor, interpretation_executor=executor) as root:
        if str(root.file.uuid) != record['sources'][str(path)]['uuid']:
            raise ValueError(f'ROOT UUID changed: {path}')
        tree = root['t_recovered_time']
        if (tree.num_entries != expected_entries or root['t_alignment'].num_entries != expected_entries
                or tree.typenames() != record['branch_types']):
            raise ValueError(f'Original time tree/schema/count mismatch: {path}')
        for name, expected in [('recovery_request_id', record['recovery_request_id']),
                               ('recovery_code_commit', record['recovery_code_commit'])]:
            obj = root[name]
            actual = obj.member('fTitle') if obj.classname == 'TNamed' else str(obj)
            if actual != expected:
                raise ValueError(f'Recovery identity changed: {path}')
        for arrays in tree.iterate(TIME_FIELDS, step_size=200000, library='np'):
            result = validate_times(arrays)
            status += result['status']
            max_error = max(max_error, result['max_mjd_unix_error_seconds'])
            chunks.append(np.asarray(arrays['mjd'][arrays['match_status'] == 0], dtype=np.float64))
    if status.tolist() != summary['status']:
        raise ValueError(f'Actual time status differs from acceptance: {path}')
    intervals, audit = intervals_from_mjd(np.concatenate(chunks) if chunks else np.array([]))
    headers = {'hardware_livetime_available': False}
    if task['original_root']:
        original = str(Path(task['original_root']) / task['relative_path'])
        with uproot.open(original, decompression_executor=executor, interpretation_executor=executor) as root:
            if str(root.file.uuid) != record['sources'][original]['uuid']:
                raise ValueError(f'Original observation UUID changed: {original}')
            header = root['t_eventout_h']
            names = header['filename'].array(library='np').tolist()
            headers.update(record_count=int(header.num_entries),
                           run_ids=sorted({int(name.split('.')[1]) for name in names}),
                           filename_samples=names[:2] + names[-2:],
                           note='Header filenames carry run/sequence provenance; mtime is processing time. '
                                'No hardware live-time/start-stop counter is present.')
    for name, binding in bindings.items():
        check_binding(name, binding)
    return {'relative_path': task['relative_path'], 'time_path': str(path), 'intervals': intervals,
            'original_entries': expected_entries, 'valid_entries': int(status[0]),
            'status': status.tolist(), 'exception_file': str(exceptions_path),
            'exception_sha256': record['sidecars'][str(exceptions_path)]['sha256'],
            'exception_records': len(exceptions), 'exception_reasons': dict(reasons),
            'ies_fallback': fallback_count, 'excluded_anomalies': excluded,
            'max_mjd_unix_error_seconds': max_error, 'source_binding': record['sources'][str(path)],
            'recovery_receipt': recovery, 'header_run_records': headers,
            'elapsed_seconds': time.perf_counter() - start_clock, **audit}


def run(manifest_path, output_dir, workers=4, max_files=None):
    if not 1 <= workers <= 16:
        raise ValueError('--workers must be between 1 and 16')
    if max_files is not None and max_files <= 0:
        raise ValueError('--max-files must be positive and is for smoke tests only')
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Existing output preserved: {output_dir}')
    started = time.perf_counter()
    manifest_binding = {'stat': file_stat(manifest_path), 'sha256': sha256(manifest_path)}
    manifest, tasks = load_tasks(manifest_path)
    scope = 'month' if max_files is None else 'smoke'
    if scope == 'month' and len(tasks) != 681:
        raise ValueError('Month scope requires all 681 accepted January files')
    if max_files is not None:
        tasks = tasks[:max_files]
    expected_status = np.sum([task['accepted']['original_status'] for task in tasks], axis=0).tolist()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        records = []
        if workers == 1:
            results = map(process_file, tasks)
            for record in results:
                records.append(record)
                print(f'GTI {len(records)}/{len(tasks)} {record["relative_path"]}', flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for record in pool.map(process_file, tasks):
                    records.append(record)
                    print(f'GTI {len(records)}/{len(tasks)} {record["relative_path"]}', flush=True)
        status = np.sum([record['status'] for record in records], axis=0).tolist()
        if status != expected_status:
            raise ValueError('Actual original status counters differ from selected accepted inputs')
        merged = merge_intervals([interval for record in records for interval in record['intervals']])
        if not merged:
            raise ValueError('No positive-duration January GTIs')
        check_binding(manifest_path, manifest_binding)
        for name in manifest['hashes']:
            if Path(name).name == 'verify_66080.json':
                check_binding(name, {'sha256': manifest['hashes'][name]})
        csv_path = output_dir / 'gti_source_files.csv'
        with csv_path.open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['gti_interval_index', 'time_label', 'status', 'matched_mjd_min',
                             'matched_mjd_max', 'rough_live_time_seconds'])
            for index, (start, stop) in enumerate(merged):
                writer.writerow([index, '202201', 'processed', f'{start:.15f}', f'{stop:.15f}',
                                 f'{(stop - start) * 86400.:.12f}'])
        total = sum((stop - start) * 86400. for start, stop in merged)
        per_file_total = sum(record['duration_seconds'] for record in records)
        boundary_gaps = [(start - previous_stop) * 86400.
                         for (_, previous_stop), (start, _) in zip(merged[:-1], merged[1:])]
        epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
        audit = {'result': 'PASS', 'scope': scope, 'time_label': '202201', 'run_tag': manifest.get('run_tag'),
                 'manifest': {'path': str(Path(manifest_path).resolve()), **manifest_binding},
                 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'workers': workers,
                 'file_count': len(records), 'original_entries': sum(record['original_entries'] for record in records),
                 'expected_entries': sum(task['accepted']['original_entries'] for task in tasks),
                 'expected_status': expected_status, 'month_file_count': manifest['file_count'],
                 'month_original_entries': manifest['original_entries'], 'month_original_status': manifest['original_status'],
                 'status': status, 'excluded_nonzero_status_entries': sum(status[1:]),
                 'ies_fallback': sum(record['ies_fallback'] for record in records),
                 'gap_threshold_seconds': GAP_SECONDS, 'mjd_unix_tolerance_seconds': MJD_TOLERANCE_SECONDS,
                 'max_mjd_unix_error_seconds': max(record['max_mjd_unix_error_seconds'] for record in records),
                 'method': 'Sort status-0 original full time per file, split gaps strictly greater than 60 s; '
                           'union overlaps/touching intervals globally. Positive gaps between file intervals remain excluded.',
                 'limitation': 'Gap-defined effective live time from pincness-only original observations; '
                               'no independent hardware dead-time correction or Pass5 selection.',
                 'per_file_interval_live_seconds_sum': per_file_total,
                 'overlap_removed_seconds': per_file_total - total,
                 'merged_interval_count': len(merged), 'merged_live_seconds': total, 'total_live_seconds': total,
                 'total_live_days': total / 86400., 'mjd_min': merged[0][0], 'mjd_max': merged[-1][1],
                 'utc_start': (epoch + timedelta(days=merged[0][0])).isoformat(),
                 'utc_stop': (epoch + timedelta(days=merged[-1][1])).isoformat(),
                 'remaining_positive_gaps_le60_count': sum(0 < gap <= GAP_SECONDS for gap in boundary_gaps),
                 'remaining_positive_gaps_le60_seconds': sum(gap for gap in boundary_gaps if 0 < gap <= GAP_SECONDS),
                 'gti_exposure_csv': str(csv_path.resolve()), 'gti_exposure_sha256': sha256(csv_path),
                 'elapsed_seconds': time.perf_counter() - started, 'files': records}
        with (output_dir / 'gti_audit.json').open('x') as handle:
            json.dump(audit, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps({key: value for key, value in audit.items() if key != 'files'}, indent=2), flush=True)
        return audit
    except Exception as exc:
        with (output_dir / 'failure.json').open('x') as handle:
            json.dump({'result': 'FAIL', 'type': type(exc).__name__, 'error': str(exc)}, handle, indent=2)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-files', type=int, default=None, help='Explicit smoke scope only; never a month acceptance.')
    args = parser.parse_args()
    run(args.manifest, args.output_dir, args.workers, args.max_files)
