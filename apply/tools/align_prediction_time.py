#!/usr/bin/env python3
"""Join filtered prediction events to the immutable full observation time friend."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import uproot

KEYS = ('irun', 'ies', 'iseq', 'ievent')


def file_stat(path):
    s = Path(path).stat()
    return [s.st_ino, s.st_size, s.st_mtime_ns]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key_array(arrays):
    n = len(arrays[KEYS[0]])
    result = np.empty(n, dtype=[(k, '<i8') for k in KEYS])
    for k in KEYS:
        values = arrays[k]
        if values.dtype.kind not in 'iu' or len(values) != n:
            raise ValueError('Invalid integer event identity: ' + k)
        if values.dtype.kind == 'u' and np.any(values > np.iinfo(np.int64).max):
            raise ValueError('Event identity exceeds int64: ' + k)
        result[k] = values
    return result


def unique_order(keys, label):
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    duplicate = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])
    if len(duplicate):
        key = sorted_keys[duplicate[0]].tolist()
        raise ValueError(f'Duplicate {label} event key {key}; count={len(duplicate)}')
    return order


def map_entries(original, prediction):
    source, target = key_array(original), key_array(prediction)
    order = unique_order(source, 'original')
    unique_order(target, 'prediction')
    positions = np.searchsorted(source[order], target)
    valid = positions < len(source)
    valid[valid] &= source[order[positions[valid]]] == target[valid]
    missing = np.flatnonzero(~valid)
    if len(missing):
        raise ValueError(f'Missing event key {target[missing[0]].tolist()}; prediction_row={missing[0]}; count={len(missing)}')
    return order[positions]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def equal_bits(actual, expected, label):
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise ValueError('Field type/value/precision mismatch: ' + label)


def align_file(relative, config, verify_only=False):
    rel = Path(relative)
    original = Path(config['original_root']) / rel
    prediction = Path(config['prediction_root']) / rel
    friend_rel = rel.with_name(rel.stem + '.time.root')
    time_path = Path(config['time_root']) / friend_rel
    output = Path(config['output_root']) / friend_rel
    receipt_path = output.with_suffix('.receipt.json')
    if not verify_only and (output.exists() or output.is_symlink() or receipt_path.exists()):
        raise FileExistsError('Existing target preserved: ' + str(output))
    summary_path = time_path.with_suffix('.summary.json')
    exceptions_path = time_path.with_suffix('.exceptions.jsonl')
    obs_receipt_path = Path(config['observation_receipts']) / (rel.name + '.json')
    source_paths = [original, prediction, time_path, summary_path, exceptions_path, obs_receipt_path]
    if any(p.is_symlink() for p in source_paths) or output.parent.resolve() != output.parent.absolute():
        raise ValueError('Symlink source/output rejected')
    before = {str(p): file_stat(p) for p in source_paths}
    obs_receipt, summary = read_json(obs_receipt_path), read_json(summary_path)
    with uproot.open(original) as raw_file, uproot.open(prediction) as pred_file, uproot.open(time_path) as time_file:
        raw, pred, times = raw_file['t_eventout'], pred_file['t_eventout'], time_file['t_recovered_time']
        audit = time_file['t_alignment']
        n, m = raw.num_entries, pred.num_entries
        if (obs_receipt['name'] != rel.name or obs_receipt['entries'] != n
                or obs_receipt['bytes'] != original.stat().st_size
                or times.num_entries != n or audit.num_entries != n or summary['entries'] != n):
            raise ValueError('Source correspondence/count/receipt mismatch')
        request_id = time_file['recovery_request_id']
        code_commit = time_file['recovery_code_commit']
        # Production uses TNamed; tiny uproot fixtures use TObjString.
        request_id = request_id.member('fTitle') if request_id.classname == 'TNamed' else str(request_id)
        code_commit = code_commit.member('fTitle') if code_commit.classname == 'TNamed' else str(code_commit)
        candidates = [Path(d) / (rel.name + '.json') for d in config['time_receipts']]
        candidates = [p for p in candidates if p.exists() and read_json(p)['request_id'] == request_id]
        if len(candidates) != 1:
            raise ValueError('Missing/ambiguous recovery receipt for request ' + request_id)
        recovery_path = candidates[0]
        recovery_before = file_stat(recovery_path)
        recovery = read_json(recovery_path)
        if (any(recovery[k] != obs_receipt[k] for k in ('name', 'record_id', 'entries', 'bytes'))
                or recovery['git_commit'] != code_commit or recovery['summary'] != summary
                or recovery['output_stat'] != [before[str(p)] for p in (time_path, summary_path, exceptions_path)]):
            raise ValueError('Recovery source record changed/mismatched')
        raw_keys, pred_keys = raw.arrays(list(KEYS), library='np'), pred.arrays(list(KEYS), library='np')
        audit_arrays = audit.arrays(list(KEYS) + ['entry_index'], library='np')
        for k in KEYS:
            equal_bits(audit_arrays[k], raw_keys[k], 'original audit ' + k)
        equal_bits(audit_arrays['entry_index'], np.arange(n, dtype=np.int64), 'original audit entry_index')
        indices = map_entries(raw_keys, pred_keys)
        source_times = times.arrays(library='np')
        if any(v.dtype.kind not in 'biuf' or v.ndim != 1 or len(v) != n for v in source_times.values()):
            raise ValueError('Unsupported/incomplete time branch; cannot preserve type')
        source_status = np.bincount(source_times['match_status'], minlength=7).tolist()
        if source_status != summary['status']:
            raise ValueError('Source match_status counters differ from recovery summary')
        selected = {k: v[indices] for k, v in source_times.items()}
        selected_audit = dict(pred_keys, entry_index=indices.astype(np.int64))
        provenance = dict(relative_path=str(rel), original_entries=n, prediction_entries=m,
                          mapped_original_first_rows=indices[:5].tolist(), missing_keys=0, ambiguous_keys=0,
                          original_status=source_status,
                          prediction_status=np.bincount(selected['match_status'], minlength=7).tolist(),
                          prediction_nonzero_status=int(np.count_nonzero(selected['match_status'])),
                          mapping_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                          prediction_keys_sha256=hashlib.sha256(key_array(pred_keys).tobytes()).hexdigest(),
                          recovery_request_id=request_id, recovery_code_commit=code_commit,
                          observation_record_id=obs_receipt['record_id'],
                          recovery_receipt=dict(path=str(recovery_path), sha256=sha256(recovery_path)),
                          sources={str(p): dict(stat=before[str(p)], uuid=str(f.file.uuid) if f else None)
                                   for p, f in ((original, raw_file), (prediction, pred_file), (time_path, time_file))},
                          sidecars={str(p): dict(stat=before[str(p)], sha256=sha256(p))
                                    for p in (summary_path, exceptions_path, obs_receipt_path)},
                          branch_types=times.typenames(), tool_sha256=sha256(__file__))
        tmp = None
        try:
            if verify_only:
                if output.is_symlink() or not receipt_path.is_file():
                    raise ValueError('Missing receipt or symlink output')
                prior = read_json(receipt_path)
                if prior['provenance'] != provenance or prior['output_sha256'] != sha256(output):
                    raise ValueError('Alignment source/output provenance changed')
                check_path = output
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix='.' + output.name, suffix='.tmp', dir=output.parent)
                os.close(fd)
                tmp = Path(name)
                with uproot.recreate(tmp) as f:
                    f.mktree('t_recovered_time', {k: v.dtype for k, v in selected.items()})
                    f['t_recovered_time'].extend(selected)
                    f.mktree('t_alignment', {k: v.dtype for k, v in selected_audit.items()})
                    f['t_alignment'].extend(selected_audit)
                    f['alignment_provenance'] = json.dumps(provenance, sort_keys=True)
                check_path = tmp
            with uproot.open(check_path) as result:
                t = result['t_recovered_time']
                if t.classname != 'TTree' or t.num_entries != m or t.typenames() != times.typenames():
                    raise ValueError('Output tree/schema/count mismatch')
                if str(result['alignment_provenance']) != json.dumps(provenance, sort_keys=True):
                    raise ValueError('Output embedded provenance mismatch')
                for k, expected in selected.items():
                    equal_bits(t[k].array(library='np'), expected, k)
                for k, expected in selected_audit.items():
                    equal_bits(result['t_alignment'][k].array(library='np'), expected, 'output audit ' + k)
            if any(file_stat(p) != before[str(p)] for p in source_paths) or file_stat(recovery_path) != recovery_before:
                raise ValueError('Source changed during alignment')
            if not verify_only:
                with tmp.open('rb') as f:
                    os.fsync(f.fileno())
                os.link(tmp, output)  # Atomic no-replace publication; never truncate a target.
                write_json_new(receipt_path, dict(provenance=provenance, output_sha256=sha256(output),
                                                 verification='all-fields-bitwise+unique-event-identity',
                                                 verified_at=datetime.now(timezone.utc).isoformat()))
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return provenance


def process_task(task):
    relative, config, verify_only = task
    try:
        receipt = align_file(relative, config, verify_only)
        print(f'PASS {relative} {receipt["prediction_entries"]}', flush=True)
        return dict(ok=True, **receipt)
    except Exception as exc:
        evidence = dict(ok=False, relative_path=relative, error_type=type(exc).__name__, error=str(exc))
        print(json.dumps(evidence), flush=True)
        return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', required=True)
    p.add_argument('--prediction-root', required=True)
    p.add_argument('--time-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--observation-receipts', required=True)
    p.add_argument('--time-receipts', nargs='+', required=True)
    p.add_argument('--file', help='One relative MMDD/Esg*.root path for smoke validation')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--report', required=True, help='New machine-readable acceptance report (never replaced)')
    args = p.parse_args()
    config = {k: str(Path(v).resolve()) for k, v in vars(args).items() if k.endswith('_root') or k == 'observation_receipts'}
    config['time_receipts'] = [str(Path(d).resolve()) for d in args.time_receipts]
    root = Path(config['output_root'])
    if any(root == Path(config[k]) or root in Path(config[k]).parents or Path(config[k]) in root.parents
           for k in ('original_root', 'prediction_root', 'time_root')):
        p.error('Output must be disjoint from all source roots')
    if Path(args.report).exists() or (not args.verify_only and root.exists()):
        p.error('Existing report/output directory preserved; choose a new destination')
    files = sorted(str(f.relative_to(config['prediction_root'])) for f in Path(config['prediction_root']).glob('01[0-9][0-9]/Esg*.root'))
    originals = sorted(str(f.relative_to(config['original_root'])) for f in Path(config['original_root']).glob('01[0-9][0-9]/Esg*.root'))
    if files != originals or len(files) != 681:
        p.error('January source/prediction coverage must match exactly 681 files')
    if args.file:
        if args.file not in files:
            p.error('Smoke file is not in January inventory')
        files = [args.file]
    tasks = [(rel, config, args.verify_only) for rel in files]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process_task, tasks))
    passed = [r for r in results if r['ok']]
    total = sum(r['prediction_entries'] for r in passed)
    source_total = sum(r['original_entries'] for r in passed)
    output_files = sorted(str(f.relative_to(root)) for f in root.glob('01[0-9][0-9]/Esg*.time.root'))
    expected_outputs = sorted(str(Path(f).with_name(Path(f).stem + '.time.root')) for f in files)
    ok = len(passed) == len(files) and output_files == expected_outputs
    if not args.file:
        ok &= total == 48726074 and source_total == 69279369 and sum(sum(r['original_status'][1:]) for r in passed) == 464
    report = dict(ok=bool(ok), mode='verify' if args.verify_only else 'align', slurm_job_id=os.getenv('SLURM_JOB_ID'),
                  files=len(files), passed_files=len(passed), prediction_entries=total, original_entries=source_total,
                  original_nonzero_status=sum(sum(r['original_status'][1:]) for r in passed),
                  prediction_nonzero_status=sum(r['prediction_nonzero_status'] for r in passed),
                  output_root=str(root), config=config, results=results,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write_json_new(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'config')}), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
