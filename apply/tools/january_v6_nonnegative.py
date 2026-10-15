#!/usr/bin/env python3
"""Isolated January weighted-LS nonnegative6 execution; never promotes current/latest."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'apply/tools'))
from january_v6_inputs import TAG as OLD_TAG, BASE, SELECTOR_SHA, sha256, load_json, check_stat
from january_v6_validate import check, npz, run_dir, stage_paths, require, check_diagnostic
from apply.stages.nonnegative_quadratic import fit_nonnegative, normalized_background
D = importlib.import_module('apply.stages.04_background')
F = importlib.import_module('apply.stages.06_fit')
G = importlib.import_module('apply.stages.07_sed_points')
TAG = OLD_TAG + '_nonnegative6'
OLD = REPO / 'apply/output' / OLD_TAG
ROOT = REPO / 'apply/output' / TAG
SELECTOR = REPO / f'apply/config/cell_selector_{BASE}_fit.csv'
RESPONSE = REPO / f'apply/output/stage_e_{BASE}_fixed712979_rayleigh_annnorm/response_{BASE}_fixed712979_rayleigh.npz'


def saved_path(name):
    path = Path(name)
    return path if path.is_absolute() else REPO / 'apply' / path


def write(path, value):
    require(not path.exists(), f'Preserve existing artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(F.json_ready(value), indent=2, allow_nan=False) + '\n')


def context():
    manifest = load_json(OLD / 'analysis_manifest.json')
    cache = npz(OLD / 'background_diagnostic/background_signal_diagnostic.npz')
    edges = cache['x_edges_deg'].astype(float)
    centers = (edges[1:] + edges[:-1])/2
    x,y = np.meshgrid(centers,centers)
    full = D.surface_design_matrix(x.ravel(), y.ravel(), 2).reshape(*x.shape,6)
    return manifest, cache, full


def cell_fit(a,z,radius):
    fit = fit_nonnegative(a,z,6.)
    b,scale = normalized_background(fit,a,z,radius,.1)
    fit.update(B_on=b,scale=scale,
        normalization_residual=float(scale*np.maximum(a@fit['coefficients'],0).sum()-z.sum()),
        clipping_denominator_delta=float((np.maximum(a@fit['coefficients'],0)-a@fit['coefficients']).sum()))
    return fit


def bootstrap(a, means, radius, count, rng):
    values, active, failures = [], 0, []
    if not np.all(np.isfinite(means)) or np.any(means < 0):
        raise ValueError('Invalid Poisson mean; never clip a negative truth')
    for _ in range(count):
        try:
            fit = cell_fit(a,rng.poisson(means),radius)
            values.append(fit['B_on'])
            active += fit['active']
        except (ValueError, RuntimeError) as exc:
            values.append(np.nan)
            failures.append(str(exc))
    return np.array(values), active, failures


def interval(b, sd, probability):
    width = norm.ppf((1+probability)/2) * sd
    return [max(0.,b-width), b+width]


def smoke():
    ROOT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    old_manifest, cache, full = context()
    hashes = {}
    def track(path, expected=None):
        actual = sha256(path)
        require(expected is None or actual == expected, f'Source hash mismatch: {path}')
        hashes[str(path)] = actual
    for commit in ('f979c5b','dd8b060','7d46912','81b8a96'):
        subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=REPO,check=True)
    track(SELECTOR, SELECTOR_SHA)
    receipt = load_json(OLD / 'delivery_receipt.json')
    track(OLD/'final_acceptance.json',receipt['final_acceptance_sha256'])
    track(OLD/'report_202201.html',receipt['report_sha256'])
    completed = load_json(OLD/'analysis_manifest_completed.json')
    for name in ('analysis_manifest.json','analysis_manifest_completed.json','delivery_receipt.json','background_diagnostic/diagnostic.json','gti/gti_audit.json','theta_exposure.json','theta_exposure.npz'):
        track(OLD/name)
    for name, expected in load_json(OLD/'background_diagnostic/diagnostic.json')['hashes'].items():
        track(saved_path(name),expected)
    track(OLD/'gti/gti_audit.json',completed['gti_audit_sha256'])
    track(OLD/'analysis_manifest.json',load_json(OLD/'gti/gti_audit.json')['manifest']['sha256'])
    track(OLD/'gti/gti_source_files.csv',load_json(OLD/'gti/gti_audit.json')['gti_exposure_sha256'])
    compatibility = load_json(REPO/'apply/docs/JANUARY_V6_MC_COMPATIBILITY.json')
    require(compatibility['decision']['blockers']==[], 'MC compatibility failed')
    for artifact in compatibility['artifacts'].values():
        if 'path' in artifact and 'sha256' in artifact: track(Path(artifact['path']),artifact['sha256'])
    # Gate the already accepted inputs; no inference or time alignment is run.
    check('c',OLD,REPO)
    psf_gate = check('psf',OLD,REPO)
    hashes.update(psf_gate['hashes'])
    for row in old_manifest['files']:
        check_stat(Path(old_manifest['prediction_root'])/row['relative_path'],row['prediction_binding']['stat'])
    dataset_dir=run_dir(OLD,'c')/'obs_events'
    parts=receipt['parquet_dataset']['parts']
    require(sorted(str(p.relative_to(dataset_dir)) for p in dataset_dir.rglob('*.parquet')) == sorted(p['path'] for p in parts),'Parquet inventory changed')
    for part in parts:
        path=dataset_dir/part['path']
        require(path.stat().st_size==part['bytes'],'Parquet size changed')
        require(sha256(path)==part['sha256'],'Parquet hash changed')
    check_diagnostic(load_json(OLD/'background_diagnostic/diagnostic.json'),cache,old_manifest['selected_cell_ids'],completed['stage_c_selected_rows'])
    # Independently rebuild the ROI map and spherical ON counts from accepted C.
    cells=D.load_cells(SELECTOR)
    edges=cache['x_edges_deg']
    scan=D.scan_stage_c_roi_events(dataset_dir,cells,source_ra_deg=83.63,source_dec_deg=22.01,
        x_edges_deg=edges,y_edges_deg=edges,rho_hist_edges_deg=D.make_edges(0,12,.1),
        roi_fiducial_deg=6.,roi_edge_diagnostic_deg=8.,batch_size=500000,max_batches=None,print_every=100)
    require(np.array_equal(scan.counts_flat.reshape(cache['counts_map'].shape),cache['counts_map']),'Cached map fails independent C recount')
    E=importlib.import_module('apply.stages.05_signal')
    on=E.scan_on_region(dataset_dir,E.load_cells(SELECTOR),cache['r_opt_deg'],83.63,22.01,
        roi_fiducial_deg=6.,batch_size=500000,max_batches=None,print_every=100)
    require(np.array_equal(on.n_on,cache['N_on']),'Cached spherical ON fails recount')
    timing=[]
    for cid in (1,61,62,75,89):
        i=cid-1; mask=cache['training_mask'][i]; a=full[mask]; z=cache['counts_map'][i][mask]
        t=time.perf_counter(); fit=cell_fit(a,z,float(cache['r_opt_deg'][i]))
        draws,active,failures=bootstrap(a,(a@fit['coefficients'])*fit['scale'],float(cache['r_opt_deg'][i]),16,np.random.default_rng(6474800+cid))
        require(not failures, f'Poisson smoke failures for {cid}: {failures}')
        timing.append(dict(cell_id=cid,seconds=time.perf_counter()-t,fit=fit,draws=draws,bootstrap_active=active))
    write(ROOT/'analysis_manifest.json',dict(run_tag=TAG,branch='independent constrained weighted-LS; not legacy reproduction',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        selected_cell_ids=old_manifest['selected_cell_ids'],contract=old_manifest['contract'],
        reuse_hashes=hashes,parquet_dataset=receipt['parquet_dataset'],job_id=os.environ['SLURM_JOB_ID'],
        recount_rows=scan.input_rows,all_cell_ids=cache['cell_id'],no_promotion=True))
    write(ROOT/'smoke.json',dict(result='PASS',job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-start,cells=timing))


def de():
    require(load_json(ROOT/'smoke.json')['result']=='PASS','Smoke required')
    manifest,cache,full=context(); selected=manifest['selected_cell_ids']; rows=[]; selected_fits=[]
    for i,cid in enumerate(cache['cell_id']):
        a=full[cache['training_mask'][i]]; z=cache['counts_map'][i][cache['training_mask'][i]]
        row=dict(cell_id=int(cid),required=int(cid) in selected,N_on=int(cache['N_on'][i]),
            old_B_on=float(cache['B_on_analytic'][i]),old_objective=float(cache['fit_chi2'][i]),
            old_disk_minimum=D.minimum_quadratic_on_centered_disk(cache['surface_coefficients'][i],6.),
            annulus_counts=int(z.sum()),annulus_pixels=len(z),radius_deg=float(cache['r_opt_deg'][i]))
        try:
            fit=cell_fit(a,z,row['radius_deg']); row.update(fit,status='PASS')
            require(fit['minimum']>=-fit['tolerance'],'Global positivity')
            if row['required']: selected_fits.append(fit)
        except (ValueError,RuntimeError) as exc:
            row.update(status='FAIL',error=str(exc))
            if row['required']:
                write(ROOT/'de_failure.json',row)
                raise
        rows.append(row)
    require([r['cell_id'] for r in rows if r['required']]==selected,'Fixed44 order')
    write(ROOT/'background_all91.json',rows)
    ids=np.array(selected); idx=ids-1
    response=npz(RESPONSE); order=[response['cell_id'].tolist().index(c) for c in selected]
    signal=dict(cell_id=ids,nhit_bin=response['nhit_bin'][order],predE_bin=response['predE_bin'][order],
        N_on=cache['N_on'][idx],B_on=np.array([f['B_on'] for f in selected_fits]),
        containment_r_opt=np.ones(44),r_opt_deg=cache['r_opt_deg'][idx],
        surface_coefficients=np.array([f['coefficients'] for f in selected_fits]),
        annulus_surface_scale=np.array([f['scale'] for f in selected_fits]))
    signal['excess']=signal['N_on']-signal['B_on']
    signal['excess_err_stat']=np.sqrt(signal['N_on'])
    signal['excess_err_conservative']=np.sqrt(signal['N_on']+signal['B_on'])
    np.savez_compressed(ROOT/'signal_legacy_errors.npz',**signal)
    write(ROOT/'de_acceptance.json',dict(result='PASS',job_id=os.environ['SLURM_JOB_ID'],fixed_cells=44,
        candidate_failures=[r['cell_id'] for r in rows if r['status']=='FAIL'],
        negative_excess_cells=ids[signal['excess']<0],low_on_lt10=ids[signal['N_on']<10]))


def bootstrap_cell(task):
    i, count=task
    _,cache,full=context()
    a=full[cache['training_mask'][i]]; z=cache['counts_map'][i][cache['training_mask'][i]]
    radius=float(cache['r_opt_deg'][i]); fit=cell_fit(a,z,radius)
    means=(a@fit['coefficients'])*fit['scale']
    draws,active,failures=bootstrap(a,means,radius,count,np.random.default_rng(647480000+i))
    good=draws[np.isfinite(draws)]
    sd=float(np.std(good,ddof=1)) if len(good)>1 else float('nan')
    return dict(cell_id=i+1,draws=draws,sd=sd,variance=sd**2,
        interval68=interval(fit['B_on'],sd,.68),interval95=interval(fit['B_on'],sd,.95),
        conditional_bias=float(good.mean()-fit['B_on']),conditional_rmse=float(np.sqrt(np.mean((good-fit['B_on'])**2))),
        active_rate=active/count,failure_rate=len(failures)/count,failures=failures)


def coverage_scenario(task):
    shape, expected_count, outer, inner=task
    _,cache,full=context(); i=61
    a=full[cache['training_mask'][i]]; radius=float(cache['r_opt_deg'][i])
    shapes={'constant':[1,0,0,0,0,0], 'positive_tilt':[1,.05,-.03,.005,0,.005],
        'interior_zero':[1,-2,0,1,0,1], 'boundary_zero':[36,0,0,-1,0,-1]}
    c=np.array(shapes[shape],float); c*=expected_count/(a@c).sum()
    means=a@c; truth=D.integrate_centered_disk_polynomial(c,radius,.1)
    seed=['constant','positive_tilt','interior_zero','boundary_zero'].index(shape)*10000+expected_count
    rng=np.random.default_rng(202201+seed); estimates=[]; covered=[0,0]; failures=[]; active=0; widths=[]
    for rep in range(outer):
        try:
            fit=cell_fit(a,rng.poisson(means),radius); b=fit['B_on']; active+=fit['active']
            draws,_,inner_fail=bootstrap(a,(a@fit['coefficients'])*fit['scale'],radius,inner,rng)
            if inner_fail: raise RuntimeError(f'Nested bootstrap failures {len(inner_fail)}/{inner}')
            sd=float(np.std(draws,ddof=1)); estimates.append(b)
            bounds=[interval(b,sd,p) for p in (.68,.95)]
            for k,(lo,hi) in enumerate(bounds): covered[k]+=lo<=truth<=hi
            widths.append([hi-lo for lo,hi in bounds])
        except (ValueError,RuntimeError) as exc:
            failures.append(dict(replicate=rep,error=str(exc)))
    estimates=np.array(estimates); cov=np.array(covered)/outer
    mc=np.sqrt(cov*(1-cov)/outer)
    # Fixed margins plus three nominal binomial SE; no results-driven recalibration.
    lower=np.array([.68,.95])-.03-3*np.sqrt(np.array([.68,.95])*(1-np.array([.68,.95]))/outer)
    relative_bias=float(np.mean(estimates)/truth-1)
    passed=len(failures)/outer<=.01 and bool(np.all(cov>=lower)) and abs(relative_bias)<=.10
    return dict(shape=shape,expected_annulus_count=expected_count,true_coefficients=c,truth_B_on=truth,
        outer=outer,inner=inner,successful=len(estimates),failure_rate=len(failures)/outer,failures=failures,
        bias=float(np.mean(estimates)-truth),relative_bias=relative_bias,rmse=float(np.sqrt(np.mean((estimates-truth)**2))),
        boundary_rate=active/outer,coverage68=cov[0],coverage95=cov[1],coverage_mc_se=mc,
        acceptance_lower=lower,pass_calibration=passed,mean_width=np.mean(widths,axis=0),estimates=estimates)


def uncertainty(workers):
    design=load_json(ROOT/'pseudoexperiment_design.json')
    require(load_json(ROOT/'de_acceptance.json')['result']=='PASS','D/E gate')
    ids=load_json(ROOT/'analysis_manifest.json')['selected_cell_ids']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cells=list(pool.map(bootstrap_cell,[(i-1,design['bootstrap_replicates']) for i in ids]))
        write(ROOT/'bootstrap_cells.json',cells)
        scenarios=list(pool.map(coverage_scenario,[(shape,n,design['coverage_outer'],design['coverage_inner']) for shape in design['truth_shapes'] for n in design['annulus_counts']]))
    write(ROOT/'coverage.json',scenarios)
    if not all(r['failure_rate']==0 and np.isfinite(r['sd']) and r['sd']>0 for r in cells):
        write(ROOT/'uncertainty_acceptance.json',dict(result='FAIL_BOOTSTRAP',job_id=os.environ['SLURM_JOB_ID']))
        raise RuntimeError('Observed bootstrap failed; no accepted error estimate')
    signal=npz(ROOT/'signal_legacy_errors.npz')
    var=np.array([c['variance'] for c in cells])
    signal['background_variance']=var
    signal['excess_err_legacy']=signal['excess_err_conservative'].copy()
    signal['excess_err_conservative']=np.sqrt(signal['N_on']+var)
    np.savez_compressed(ROOT/'signal_propagated_errors.npz',**signal)
    np.savez_compressed(ROOT/'background_bootstrap.npz',cell_id=ids,B_on_draws=np.array([c['draws'] for c in cells]),background_variance=var)
    write(ROOT/'uncertainty_acceptance.json',dict(job_id=os.environ['SLURM_JOB_ID'],
        result='PASS' if all(c['pass_calibration'] for c in scenarios) else 'FAIL_CALIBRATION_DIAGNOSTIC_ONLY',
        intervals='max(0,Bhat-z*s_boot), Bhat+z*s_boot; plug-in Poisson bootstrap SD',
        variance='N_on+Var_boot(Bhat), replaces old B_on term; independent ON/annulus and independent cells',
        no_hessian=True,design_sha256=sha256(ROOT/'pseudoexperiment_design.json')))


def spectra():
    require((ROOT/'uncertainty_acceptance.json').exists(),'Uncertainty must finish')
    response=npz(RESPONSE); exposure=npz(OLD/'theta_exposure.npz')
    theta=load_json(OLD/'theta_exposure.json')['theta_exposure_sec']
    products={}
    for mode in ('legacy','propagated'):
        signal=npz(ROOT/f'signal_{mode}_errors.npz')
        subset=F.load_cell_subset(SELECTOR,response['cell_id'])
        resp,sig,_=F.apply_cell_subset(response,signal,subset)
        require(resp['cell_id'].tolist()==sig['cell_id'].tolist()==load_json(ROOT/'analysis_manifest.json')['selected_cell_ids'],'Fit cell alignment')
        require(np.all(np.isfinite(sig['excess_err_conservative'])) and np.all(sig['excess_err_conservative']>0),'All44 errors valid')
        fits={}
        for name in ('pl','logpar'):
            fit=F.fit_model(model_name=name,error_mode='conservative',observed=sig['excess'],
                errors=sig['excess_err_conservative'],a_eff_m2=resp['a_eff'],containment=sig['containment_r_opt'],
                theta_exposure_sec=np.array(theta),loge_edges=resp['logE_true_edges'],pivot_tev=3.,quadrature_points=64,
                start_gamma=2.69,start_phi0=2.114e-12)
            require(fit.model_counts.size==44 and np.all(np.isfinite(fit.pull)),'Fit retained all44')
            require(fit.valid,f'Invalid Minuit fit: {mode}/{name}; block SED')
            fits[name+'_conservative']=fit
        preferred=F.choose_preferred_fit(fits)
        metadata=dict(preferred_fit=preferred,fits={k:F.result_to_metadata(v) for k,v in fits.items()})
        frozen=G.stage_f_preferred_fit(metadata)
        points,_,_,_=G.fit_sed_points(excluded_cell_ids=[],stage_f=sig,a_eff_m2=resp['a_eff'],
            containment=sig['containment_r_opt'],theta_exposure_sec=np.array(theta),loge_edges=resp['logE_true_edges'],
            frozen_model=frozen,pivot_tev=3.,reference_phi0=2.114e-12,reference_gamma=2.69,quadrature_points=64)
        out=ROOT/mode; out.mkdir(exist_ok=False)
        F.write_npz(out/'fit.npz',sig,fits,np.array(theta))
        G.write_summary_csv(out/'sed.csv',points)
        rows=[{**asdict(p),'error_mode':mode} for p in points]
        write(out/'sed.json',rows)
        write(out/'fit.json',dict(**metadata,error_definition='sqrt(N_on+B_on)' if mode=='legacy' else 'sqrt(N_on+Var_boot(Bhat))',
            diagnostic_only=True,internal_compatibility_key='conservative is the existing API field name; error_definition overrides its legacy meaning'))
        products[mode]=dict(metadata=metadata,sed=rows)
    write(ROOT/'spectra_acceptance.json',dict(job_id=os.environ['SLURM_JOB_ID'],result='DIAGNOSTIC_PRODUCTS_COMPLETE',
        uncertainty_status=load_json(ROOT/'uncertainty_acceptance.json')['result'],outputs=products))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','de','uncertainty','spectra'])
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    start=time.perf_counter()
    {'smoke':smoke,'de':de,'uncertainty':lambda:uncertainty(args.workers),'spectra':spectra}[args.stage]()
    print(json.dumps(dict(stage=args.stage,job=os.environ.get('SLURM_JOB_ID'),elapsed=time.perf_counter()-start,result='EXECUTION_SUCCESS')),flush=True)

if __name__=='__main__':main()
