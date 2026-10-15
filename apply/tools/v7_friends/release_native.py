"""Build a new native-only acceptance version; preserve original immutable packages."""
import argparse
import json
import os
from pathlib import Path
import shutil
from friends import CONTRACT, dump, process, publish, sha256, stamp, summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('old-run','new-run','inventory','stage'):
        p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    if args.new_run.resolve()==args.old_run.resolve():raise ValueError('new immutable version required')
    inv=json.loads(args.inventory.read_text());base=Path(inv['input_root'])
    code=sha256(Path(__file__).with_name('friends.py'))
    metadata=dict(contract=CONTRACT,code_sha256=code,git_commit=os.environ.get('V7_GIT_COMMIT','uncommitted'),
                  inventory_sha256=sha256(args.inventory),previous_run=str(args.old_run),
                  reuse='Byte-identical previously verified v1 packages; their original code/contract provenance remains intact.')
    metadata=json.loads(json.dumps(metadata))
    args.new_run.mkdir(parents=True,exist_ok=True)
    cp=args.new_run/'contract.json'
    if cp.exists():
        assert json.loads(cp.read_text())==metadata,'new run identity mismatch'
    else:dump(cp,metadata)
    lineage=[];before={}
    for item in inv['files']:
        stem=item['name'].removesuffix('_eventout.root');source=args.old_run/'files'/stem
        rp=source/'receipt.json';r=json.loads(rp.read_text());before[str(rp)]=sha256(rp)
        assert stamp(base/item['name'])==r['source_stat'],'input changed'
        target=args.new_run/'files'/stem
        if r['status']=='verified':
            assert sha256(source/r['friend_name'])==r['friend_sha256'],'old friend changed'
            if target.exists():
                assert sha256(target/'receipt.json')==before[str(rp)]
                assert sha256(target/r['friend_name'])==r['friend_sha256']
            else:
                def copy_checked(tmp):
                    shutil.copytree(source,tmp,dirs_exist_ok=True)
                    assert sha256(tmp/'receipt.json')==before[str(rp)]
                    assert sha256(tmp/r['friend_name'])==r['friend_sha256']
                publish(target,copy_checked)
            action='copied_verified_unchanged'
        else:
            assert set(r['failures']) <= {'nhit_difference_requires_definition_or_alignment_review','boundary:rmds_le20'},'unapproved failure'
            assert sha256(args.stage/item['summary_name'])==r['summary_sha256'],'summary changed'
            assert sha256(base/item['name'])==r['event_sha256'],'event changed'
            assert process(item,args.new_run,base,args.stage)=='verified','new validation failed; inspect receipt'
            action='revalidated_native_only'
        lineage.append(dict(file=item['name'],action=action,previous_receipt_sha256=before[str(rp)],
                            receipt_sha256=sha256(target/'receipt.json')))
    assert all(sha256(p)==h for p,h in before.items()),'previous receipts changed'
    dump(args.new_run/'lineage.json',dict(job_id=os.environ['SLURM_JOB_ID'],old_run=str(args.old_run),
        copied_verified_files=sum(x['action']=='copied_verified_unchanged' for x in lineage),
        revalidated_files=sum(x['action']=='revalidated_native_only' for x in lineage),files=lineage))
    out=summarize(args.new_run,inv['files'])
    print(json.dumps({k:out[k] for k in ('target_files','verified_files','verified_events','quarantined_files','pending_files')}))

if __name__=='__main__':main()
