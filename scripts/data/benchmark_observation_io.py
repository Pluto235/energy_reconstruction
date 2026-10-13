#!/usr/bin/env python3
"""Paired complete-hour ROOT I/O benchmark; run only in IHEP HTCondor."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import signal
import time

import ROOT
from prepare_observation_day import MACRO, filter_source, inspect
from observation_state import STAGING, WORK


def compare(left_path, right_path):
    left, right = ROOT.TFile.Open(str(left_path)), ROOT.TFile.Open(str(right_path))
    try:
        if not ROOT.same_observation_tree(left.Get('t_eventout'), right.Get('t_eventout')):
            raise RuntimeError('Retained event values or order differ')
        a, b = left.Get('t_eventout_h'), right.Get('t_eventout_h')
        names = [branch.GetName() for branch in a.GetListOfBranches()]
        if a.GetEntries() != b.GetEntries() or names != [branch.GetName() for branch in b.GetListOfBranches()]:
            raise RuntimeError('Header schema/count differs')
        for i in range(a.GetEntries()):
            if a.GetEntry(i) <= 0 or b.GetEntry(i) <= 0:
                raise RuntimeError('Unreadable header')
            for name in names:
                if str(getattr(a, name)) != str(getattr(b, name)):
                    raise RuntimeError(f'Header value differs: {name}/{i}')
    finally:
        left.Close(); right.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('day')
    parser.add_argument('--baseline-repeat', action='store_true', help='Warm-order control after the cache tests')
    args = parser.parse_args()
    urls = (WORK / 'manifests' / f'{args.day}.txt').read_text().splitlines()
    url = next(u for u in urls if u.endswith('_00.root'))
    directory = STAGING / '.benchmark_opt' / args.day
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'baseline.root'
    spec = importlib.util.spec_from_file_location('baseline_prepare', WORK / 'optimization_20261013/baseline_code/prepare_observation_day.py')
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    ROOT.gROOT.ProcessLine(f'.L {MACRO}')
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1536 * 1024**2, resource.RLIM_INFINITY))
    results = []
    modes = [('baseline', -1)] if args.baseline_repeat else [('baseline', -1), ('cache_off', 0), ('cache32', 32), ('cache128', 128)]
    for mode, cache in modes:
        target = output if mode == 'baseline' else directory / f'{mode}.root'
        started, cpu = time.monotonic(), time.process_time()
        extra_scan = 0.0
        if mode == 'baseline':
            before = time.monotonic(); expected = baseline.inspect(url)
            extra_scan = time.monotonic() - before
        rc, source, metrics = filter_source(url, target, cache)
        if rc:
            raise RuntimeError(f'{mode} filter returned {rc}: {metrics}')
        if mode == 'baseline' and (source['selected_entries'], source['header_entries']) != (expected['selected_entries'], expected['header_entries']):
            raise RuntimeError('Source count disagreement')
        before = time.monotonic()
        result = baseline.inspect(target, source['selected_entries'], source['header_entries']) if mode == 'baseline' else inspect(target, source['selected_entries'], source['header_entries'])
        validation = time.monotonic() - before
        total, cpu_seconds = time.monotonic() - started, time.process_time() - cpu
        before = time.monotonic(); inspect(target, source['selected_entries'], source['header_entries'], full_read=False)
        sample_seconds = time.monotonic() - before
        before = time.monotonic(); inspect(target, source['selected_entries'], source['header_entries'])
        full_seconds = time.monotonic() - before
        equivalence_seconds = 0.0
        if mode != 'baseline':
            before = time.monotonic(); compare(output, target)
            equivalence_seconds = time.monotonic() - before
            target.unlink()
        report = dict(mode=mode, day=args.day, source=url, entries=source['entries'], selected_entries=source['selected_entries'],
                      output_bytes=result['bytes'], total_seconds=total, cpu_seconds=cpu_seconds,
                      extra_source_scan_seconds=extra_scan, validation_seconds=validation,
                      warm_sample_seconds=sample_seconds, warm_full_seconds=full_seconds,
                      equivalence_seconds=equivalence_seconds, all_retained_values_equal=True, **metrics)
        results.append(report)
        suffix = '_warm_control' if args.baseline_repeat else ''
        destination = WORK / 'optimization_20261013' / f'io_{args.day}{suffix}.json'
        destination.write_text(json.dumps(results, indent=2) + '\n')
        print('BENCHMARK ' + json.dumps(report), flush=True)
    output.unlink()


if __name__ == '__main__':
    main()
