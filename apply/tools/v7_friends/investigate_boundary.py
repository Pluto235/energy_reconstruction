"""Read-only boundary hypothesis audit of quarantined MC files, executed in Slurm."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from friends import MATCH_FIELDS, NATIVE, dump, match_rows, native_sum, sha256


def counts(q, t):
    charge = q >= .3
    return {name: ak.to_numpy(ak.sum(charge & mask, axis=1)) for name, mask in {
        'inclusive': abs(t) <= 30,
        'strict': abs(t) < 30,
        'upper_strict': (t >= -30) & (t < 30),
        'lower_strict': (t > -30) & (t <= 30),
    }.items()}


def self_check():
    q = ak.Array([[1., 1., 1., .1]])
    t = ak.Array([[-30., 30., 0., 0.]])
    assert {k: v.tolist() for k, v in counts(q, t).items()} == {
        'inclusive': [3], 'strict': [1], 'upper_strict': [2], 'lower_strict': [2]}
    # Both sides of the physical boundary can round to the same stored float32.
    assert np.float32(30 - 2**-21) == np.float32(30 + 2**-21) == 30


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    self_check()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    protected = list((args.run/'files').glob('*/receipt.json')) + [args.run/'contract.json', args.run/'summary.json']
    before = {str(p): sha256(p) for p in protected}
    results = []
    for receipt in sorted((args.run/'files').glob('*/receipt.json')):
        r = json.loads(receipt.read_text())
        if not r.get('nhit_mismatches'):
            continue
        summary = args.work/'summaries'/Path(r['summary_source']).name
        assert sha256(summary) == r['summary_sha256']
        assert sha256(r['source_path']) == r['event_sha256']
        assert sha256(args.work/'upstream_eventout'/r['file']) == r['event_sha256']
        with uproot.open(summary) as s, uproot.open(r['source_path']) as e:
            sa = s['wcdaevent'].arrays(list(MATCH_FIELDS)+list(NATIVE), library='np')
            ea = e['t_eventout'].arrays(list(MATCH_FIELDS), library='np')
            rows, stats = match_rows(ea, sa)
            assert rows is not None and not any(stats.values())
            native = native_sum(sa)[rows]
            hits = e['t_eventout'].arrays(['vqsamp','vt','vflag','vidmc','vq','vnpe'], library='ak')
            q, t = hits['vqsamp'], hits['vt']
            cs = counts(q, t)
            positive = ak.to_numpy(ak.sum((q >= .3) & (t == 30), axis=1))
            negative = ak.to_numpy(ak.sum((q >= .3) & (t == -30), axis=1))
            indices = np.flatnonzero((positive+negative > 0) | (native != cs['inclusive']))
            cases = []
            for i in indices:
                near = np.flatnonzero(ak.to_numpy((q[i] >= .3) & (abs(t[i]) == 30)))
                cases.append(dict(event_entry=int(i), summary_entry=int(rows[i]),
                    identity={k:int(sa[k][rows[i]]) for k in ('irun','ies','iseq','ievent')},
                    native=int(native[i]), counts={k:int(v[i]) for k,v in cs.items()},
                    boundary_hits=[dict(hit_index=int(j), **{k:ak.to_list(hits[k][i][j]) for k in hits.fields}) for j in near]))
            results.append(dict(file=r['file'], events=len(native), matching=stats,
                event_sha256=r['event_sha256'], summary_sha256=r['summary_sha256'],
                ihep_eventout_byte_identical=True,
                dtypes={k:e['t_eventout'][k].typename for k in hits.fields},
                mismatch_counts={k:int(np.sum(v != native)) for k,v in cs.items()},
                delta_histograms={k:dict(Counter(map(str,(native-v).tolist()))) for k,v in cs.items()},
                positive_boundary_hits=int(positive.sum()), negative_boundary_hits=int(negative.sum()), cases=cases))
    assert before == {str(p):sha256(p) for p in protected}
    out=dict(job_id=os.environ['SLURM_JOB_ID'], code_sha256=sha256(__file__), files=results,
        production_receipts_and_contract_unchanged=True,
        float32_positive30_rounding_interval=[30-2**-20,30+2**-20],
        events=sum(x['events'] for x in results),
        mismatch_counts={k:sum(x['mismatch_counts'][k] for x in results) for k in ('inclusive','strict','upper_strict','lower_strict')})
    dump(args.work/'boundary_audit.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='files'}))


if __name__ == '__main__':
    main()
