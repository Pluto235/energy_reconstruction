"""Produce a scoped machine-readable delivery summary from immutable file receipts."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from friends import dump,EDGES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);args=p.parse_args()
    inv=json.loads(args.inventory.read_text());summary=json.loads((args.run/'summary.json').read_text())
    audit=json.loads((args.run/'final_audit.json').read_text());rlist=[];missing=[]
    for item in inv['files']:
        path=args.run/'files'/item['name'].removesuffix('_eventout.root')/'receipt.json'
        if path.exists():rlist.append(json.loads(path.read_text()))
        else:missing.append(item['name'])
    verified=[r for r in rlist if r['status']=='verified'];failed=[r for r in rlist if r['status']!='verified']
    out=dict(target_files=len(inv['files']),target_events_known=sum(r['events'] or 0 for r in rlist),
             event_count_unknown_files=[r['file'] for r in rlist if r['events'] is None]+missing,
             successful_files=len(verified),successful_events=sum(r['events'] for r in verified),
             failed_files=len(failed),failed_events_known=sum(r['events'] or 0 for r in failed),pending_files=len(missing),
             pending_events=None if missing else 0,code=json.loads((args.run/'contract.json').read_text()),
             matching=summary['matching'],common_fields=summary['common_fields'],boundary_changes=summary['boundaries'],
             fields_covered_events=summary['field_coverage'],fields_coverage_denominator='all target events with known event counts',
             friend_readback=dict(verified_files=audit['verified_files'],verified_events=audit['verified_events'],
                                 quarantined_without_friend=audit['quarantine_verified_files'],failures=audit['failures']),
             failed_file_details=[dict(file=r['file'],events=r['events'],reasons=r['failures'],nhit_examples=r.get('nhit_examples',[]),
                                      retry='Review raw-hit boundary/definition differences or listed failure; use a new reviewed run, never overwrite this receipt.') for r in failed])
    out['fields_coverage_fraction']={k:v/out['target_events_known'] if out['target_events_known'] else None for k,v in summary['field_coverage'].items()}
    for scope,receipts in [('all_matched_diagnostics',rlist),('published_verified',verified)]:
        pools=Counter();poolfiles=Counter();delta=Counter();bins=[dict(lower=x,upper=EDGES[i+1] if i+1<len(EDGES) else None,events=0,compared_events=0,mismatches=0) for i,x in enumerate(EDGES)];low=Counter()
        for r in receipts:
            pools.update(r.get('poolused',{}));poolfiles.update(r.get('poolused',{}).keys());delta.update(r.get('nhit_difference_native_minus_hits',{}));low.update(r.get('nhit_below30',{}))
            for a,b in zip(bins,r.get('nhit_bins',[])):
                for k in ('events','mismatches'):a[k]+=b.get(k,0)
                a['compared_events']+=b.get('compared_events',b['events'] if 'nhit_difference_native_minus_hits' in r else 0)
        for b in bins:b['mismatch_rate']=b['mismatches']/b['compared_events'] if b['compared_events'] else None
        out[scope]=dict(poolused_events=dict(pools),poolused_files=dict(poolfiles),nhit_difference_native_minus_hits=dict(delta),
                        nhit_compared_events=sum(delta.values()),nhit_not_compared_events=sum(r.get('events') or 0 for r in receipts)-sum(delta.values()),nhit_equal_events=delta.get('0',0),
                        nhit_agreement=delta.get('0',0)/sum(delta.values()) if delta else None,nhit_bins=bins,nhit_below30=dict(low))
    out['all_target_poolused_proven_7']=(not failed and not missing and set(out['published_verified']['poolused_events'])=={'7'})
    hashes=defaultdict(list)
    for r in rlist:
        if 'event_sha256' in r:hashes[r['event_sha256']].append(r['file'])
    out['duplicate_event_content_hashes']={k:v for k,v in hashes.items() if len(v)>1}
    out['unused_summary_events']=sum(len(r.get('unused_summary_entries',[])) for r in rlist)
    out['summary_source_missing_files']=[r['file'] for r in failed if any('summary_source_missing' in x for x in r['failures'])]
    out['limitations']=['Six float32 content fields are not independent physical event ID authentication.',
                         'chi2 is diagnostic-only by explicit user instruction; its discrepancies are not resolved.',
                         'Legacy raw-hit comparisons are historical diagnostics, not the current native-only publication criterion.' if out['code'].get('contract',{}).get('version',1)>=2 else 'Raw-hit mismatch files remain quarantined; threshold adjacency does not prove an upstream cause.',
                         'For contract v2, rounding-only rmds boundary differences use the summary decision; original eventout is unchanged.',
                         'Only existing eventout events are covered; no upstream removed event is restored.']
    dump(args.run/'delivery_summary.json',out)
    print(json.dumps({k:out[k] for k in ('target_files','target_events_known','successful_files','failed_files','pending_files')}))

if __name__=='__main__':main()
