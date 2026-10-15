#!/usr/bin/env python3
"""Verify actual completed jobs, reviewed source and independent diagnostic artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from html.parser import HTMLParser
from urllib.parse import unquote
from apply.tools.january_v6_nonnegative import REPO,ROOT,OLD,TAG,write,sha256,load_json,require

SOURCES=[
 'apply/stages/nonnegative_quadratic.py','apply/tools/january_v6_nonnegative.py',
 'apply/tools/january_v6_nonnegative_spectral_toys.py','apply/tools/january_v6_nonnegative_delivery.py',
 'apply/report/build_january_v6_nonnegative_report.py','apply/tests/test_nonnegative_quadratic.py',
 'scripts/slurm/january_v6_nonnegative.sbatch','apply/config/january_v6_nonnegative_pseudoexperiments.json',
 'apply/stages/04_background.py','apply/stages/05_signal.py','apply/stages/06_fit.py',
 'apply/stages/07_sed_points.py','apply/stages/poisson_roi_background.py',
 'apply/tools/january_v6_inputs.py','apply/tools/january_v6_validate.py']


class Links(HTMLParser):
    def __init__(self):super().__init__();self.ids=set();self.links=[]
    def handle_starttag(self, tag, attrs):
        values=dict(attrs)
        if 'id' in values:self.ids.add(values['id'])
        for key in ('src','href'):
            if key in values:self.links.append(values[key])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jobs',required=True);args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'Commit reviewed source before verification')
    sources={}
    for relative in SOURCES:
        blob=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=REPO)
        actual=sha256(REPO/relative)
        require(actual==hashlib.sha256(blob).hexdigest(),f'Deployed source differs from Git blob: {relative}')
        sources[relative]=actual
    acceptance=load_json(ROOT/'final_acceptance.json')
    require(acceptance['source_commit']==commit,'Report and final verification source commits differ')
    require(acceptance['diagnostic_spectrum_and_sed'] and not acceptance['legacy_reproduction'],'Incorrect scientific labeling')
    jobs=[]
    for job in args.jobs.split(','):
        fields=subprocess.check_output(['sacct','-n','-X','-j',job,'--format=JobID,JobName%40,State,ExitCode,Elapsed,AllocCPUS,ReqMem,Timelimit,NodeList','-P'],text=True).strip().splitlines()[0].split('|')
        row=dict(zip(('id','name','state','exit_code','elapsed','cpus','memory','limit','node'),fields))
        require(row['id']==job,'Actual job accounting missing')
        if job=='66091':require(row['state']=='FAILED' and row['exit_code']=='1:0','Failed smoke history changed')
        else:require(row['state']=='COMPLETED' and row['exit_code']=='0:0',f'Failed production stage: {row}')
        jobs.append(row)
    manifest=load_json(ROOT/'analysis_manifest.json')
    for name,expected in manifest['reuse_hashes'].items():require(sha256(name)==expected,f'Reused artifact changed: {name}')
    data=OLD/'c/runs'/f'{OLD.name}_stage_c/obs_events'
    for part in manifest['parquet_dataset']['parts']:require(sha256(data/part['path'])==part['sha256'],'Parquet hash mismatch')
    parser_html=Links();parser_html.feed((ROOT/'report_202201_nonnegative6.html').read_text())
    for link in parser_html.links:
        if link.startswith('#'):require(unquote(link[1:]) in parser_html.ids,f'Broken HTML anchor {link}')
        elif '://' not in link and not link.startswith(('data:','mailto:')):
            require((ROOT/unquote(link.split('#')[0])).exists(),f'Broken HTML artifact link {link}')
    require((ROOT/'report_202201_nonnegative6.html').stat().st_size>20000,'Report unexpectedly empty')
    # Copy completed logs including the report; the delivery job itself is pending completion here.
    import shutil
    for j in jobs:
        for ext in ('out','err'):
            path=REPO/'logs/slurm'/f'{j["name"]}_{j["id"]}.{ext}'
            if path.exists():shutil.copy2(path,ROOT/'logs'/path.name)
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ('delivery_receipt.json','analysis_manifest_completed.json')}
    completed={**manifest,'completion_source_commit':commit,'committed_source_hashes':sources,
        'scientific_status':acceptance['scientific_status'],'actual_completed_jobs':jobs,
        'contract_addendum':dict(continuous_nonnegative_disk_radius_deg=6.,touch_zero_allowed=True,
            fit_statistic='original weighted-ls',normalization='original sum(z)/sum(max(q_pixel,0))',
            analytic_B_on='original raw quadratic disk integral; finite positive required',
            variance='N_on+Var_boot(Bhat), independent ON/annulus/cells; legacy sqrt(N+B) preserved'),
        'artifact_hashes':hashes,'verification_job':os.environ['SLURM_JOB_ID']}
    write(ROOT/'analysis_manifest_completed.json',completed)
    write(ROOT/'delivery_receipt.json',dict(result='VERIFIED_DELIVERY_DIAGNOSTIC_ONLY',
        scientific_status=acceptance['scientific_status'],source_commit=commit,
        committed_source_hashes=sources,actual_completed_jobs=jobs,
        verification_job=os.environ['SLURM_JOB_ID'],verification_job_status='RUNNING_AT_RECEIPT_WRITE; check sacct after exit',
        analysis_manifest_completed_sha256=sha256(ROOT/'analysis_manifest_completed.json'),
        artifact_hashes=hashes,html_links_and_anchors='PASS',git_status='CLEAN',
        limits='Diagnostic-only spectra. Independent truth coverage limited to cell62 geometry; spectral ensemble conditional on observed variances. Source/report generation reruns background91 from committed code.'))
    print(f'Verified committed diagnostic delivery {commit}',flush=True)

if __name__=='__main__':main()
