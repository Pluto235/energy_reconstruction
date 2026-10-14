#!/bin/bash
#SBATCH -J v6_202201_verify
#SBATCH -p main
#SBATCH --qos=default-safe
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH -e /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export PYTHONPATH=/mnt/newdisk/shijie/energy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
"$PY" -m unittest discover -s tests -p 'test_january_v6*.py'
"$PY" - <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('tools').resolve()))
from january_v6_inputs import TAG, load_json, sha256
from january_v6_validate import check, check_diagnostic, run_dir, require
root=Path('output')/TAG
require(not (root/'delivery_receipt.json').exists(), 'Existing delivery receipt preserved')
manifest=load_json(root/'analysis_manifest.json')
final=load_json(root/'final_acceptance.json')
c=load_json(run_dir(root,'c')/'obs_events_metadata.json')
diagnostic=load_json(root/'background_diagnostic/diagnostic.json')
with np.load(root/'background_diagnostic/background_signal_diagnostic.npz') as arrays:
    check_diagnostic(diagnostic,arrays,manifest['selected_cell_ids'],c['processing']['selected_rows'])
    require(arrays['annulus_counts'][61]==10 and arrays['annulus_pixels'][61]==5648, 'cell62 reported constants changed')
check('c',root.resolve(),Path.cwd().parent)
check('psf',root.resolve(),Path.cwd().parent)
compatibility=load_json(Path('docs/JANUARY_V6_MC_COMPATIBILITY.json'))
require(compatibility['decision']['blockers']==[] and compatibility['decision']['nominal_response']=='reuse_compatibility_passed' and compatibility['decision']['scheme_R_response']=='reuse_compatibility_passed','Compatibility decision failed')
gti=load_json(root/'gti/gti_audit.json')
require(gti['scope']=='month' and gti['file_count']==681 and gti['merged_interval_count']==756,'GTI production scope')
require(gti['status']==manifest['original_status'] and gti['original_entries']==manifest['original_entries'],'GTI status/entries')
require(sha256(root/'analysis_manifest.json')==gti['manifest']['sha256'] and sha256(root/'gti/gti_source_files.csv')==gti['gti_exposure_sha256'],'GTI source/interval hash')
for name,expected in final['hashes'].items():
    require(sha256(root/name)==expected,f'Final product hash changed: {name}')
require(final['result']=='SCIENTIFIC_BACKGROUND_CONTRACT_FAILED' and final['spectral_fit_convergence']=='NOT_RUN_D_FAILED' and final['sed']=='NOT_RUN_D_FAILED','Scientific failure must remain explicit')
parts=[]
dataset=run_dir(root,'c')/'obs_events'
for path in sorted(dataset.rglob('*.parquet')):
    parts.append({'path':str(path.relative_to(dataset)),'bytes':path.stat().st_size,'sha256':sha256(path)})
require(len(parts)==681,'C partition file count')
dataset_hash=hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
source=[*Path('tools').glob('january_v6_*.py'),Path('report/build_january_v6_report.py'),*Path('../scripts/slurm').glob('january_v6_*.sbatch')]
for path in source:
    relative=path.resolve().relative_to(Path.cwd().parent)
    committed=subprocess.check_output(['git','show',f'HEAD:{relative}'])
    require(hashlib.sha256(committed).hexdigest()==sha256(path),f'Uncommitted source: {relative}')
receipt={'result':'VERIFIED_DELIVERY_OF_SCIENTIFIC_FAILURE','data_integrity':'PASS','scientific_result':'SCIENTIFIC_BACKGROUND_CONTRACT_FAILED','formal_E_F_G':'NOT_RUN_D_FAILED','verification_job':os.environ['SLURM_JOB_ID'],'source_commit':commit,'report_generation_base_commit':final['commit'],'report_generation_sources':load_json(root/'analysis_manifest_completed.json')['runtime_source_hashes'],'committed_source_hashes':{str(p.resolve()):sha256(p) for p in source},'fixture_tests':13,'final_acceptance_sha256':sha256(root/'final_acceptance.json'),'report_sha256':sha256(root/'report_202201.html'),'parquet_dataset':{'file_count':len(parts),'bytes':sum(p['bytes'] for p in parts),'canonical_tree_sha256':dataset_hash,'parts':parts},'git_status_at_verification':subprocess.check_output(['git','status','--porcelain'],text=True),'limits':['No valid44-cell spectrum or SED exists under the frozen background contract','Gap-defined live time has no independent hardware dead-time correction']}
(root/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('parquet_dataset','committed_source_hashes','report_generation_sources')},indent=2))
PY
