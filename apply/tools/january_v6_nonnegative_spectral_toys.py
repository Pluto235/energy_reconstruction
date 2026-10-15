#!/usr/bin/env python3
"""Conditional spectral/SED propagation ensemble; not all44 nested coverage certification."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import os
import numpy as np
from scipy.stats import norm
from apply.tools.january_v6_nonnegative import ROOT, OLD, RESPONSE, F, G, npz, load_json, write, require


def summarize(records, truth):
    good=[r for r in records if r is not None and np.isfinite(r['estimate']) and np.isfinite(r['error']) and r['error']>0]
    estimates=np.array([r['estimate'] for r in good]); errors=np.array([r['error'] for r in good])
    result=dict(attempted=len(records),successful=len(good),failure_rate=1-len(good)/len(records),truth=truth,
        bias=float(np.mean(estimates)-truth),rmse=float(np.sqrt(np.mean((estimates-truth)**2))),mean_reported_error=float(np.mean(errors)))
    for p in (.68,.95):
        coverage=float(np.sum(np.abs(estimates-truth)<=norm.ppf((1+p)/2)*errors)/len(records))
        result[f'coverage{int(100*p)}']=coverage
        result[f'mc_se{int(100*p)}']=float(np.sqrt(coverage*(1-coverage)/len(records)))
    return result


def worker(task):
    truth_name, rep=task
    design=load_json(ROOT/'pseudoexperiment_design.json')
    true_params=design['spectral_truths'][truth_name]
    signal=npz(ROOT/'signal_propagated_errors.npz'); response=npz(RESPONSE)
    order=[response['cell_id'].tolist().index(int(cid)) for cid in signal['cell_id']]
    aeff=response['a_eff'][order]; loge=response['logE_true_edges']
    theta=np.array(load_json(OLD/'theta_exposure.json')['theta_exposure_sec'])
    true_counts=F.model_counts(aeff,signal['containment_r_opt'],theta,loge,model_name=truth_name,
        params=true_params,pivot_tev=3.,quadrature_points=64)
    draws=npz(ROOT/'background_bootstrap.npz')['B_on_draws']
    rng=np.random.default_rng(np.random.SeedSequence([20220164748,rep,0 if truth_name=='pl' else 1]))
    on=rng.poisson(true_counts+signal['B_on'])
    b=draws[:,rep if truth_name=='pl' else rep+design['spectral_replicates_per_truth']]
    result=dict(truth=truth_name,replicate=rep,modes={})
    for mode in ('legacy','propagated'):
        error=np.sqrt(on+(b if mode=='legacy' else signal['background_variance']))
        sample={**signal,'N_on':on,'B_on':b,'excess':on-b,'excess_err_conservative':error}
        fits={}; errors=[]
        for model in ('pl','logpar'):
            try:
                fit=F.fit_model(model_name=model,error_mode='conservative',observed=sample['excess'],errors=error,
                    a_eff_m2=aeff,containment=signal['containment_r_opt'],theta_exposure_sec=theta,
                    loge_edges=loge,pivot_tev=3.,quadrature_points=64,start_gamma=2.69,start_phi0=2.114e-12)
                if not fit.valid: raise RuntimeError('Minuit invalid')
                fits[model+'_conservative']=fit
            except (ValueError,RuntimeError) as exc:
                errors.append(dict(model=model,error=str(exc)))
        fit_rows={k:F.result_to_metadata(v) for k,v in fits.items()}
        points=[]
        if len(fits)==2:
            frozen=G.stage_f_preferred_fit(dict(preferred_fit=F.choose_preferred_fit(fits),fits=fit_rows))
            p,_,_,_=G.fit_sed_points(excluded_cell_ids=[],stage_f=sample,a_eff_m2=aeff,
                containment=signal['containment_r_opt'],theta_exposure_sec=theta,loge_edges=loge,
                frozen_model=frozen,pivot_tev=3.,reference_phi0=2.114e-12,reference_gamma=2.69,quadrature_points=64)
            for point in p:
                row=asdict(point); energy=point.effective_energy_tev
                row['true_e2_dnde']=float(energy**2*G.flux_tev(energy,model_name=truth_name,params=true_params,pivot_tev=3.))
                points.append(row)
        result['modes'][mode]=dict(fits=fit_rows,sed=points,failures=errors)
    return result


def main():
    design=load_json(ROOT/'pseudoexperiment_design.json')
    require(load_json(ROOT/'spectra_acceptance.json')['result']=='DIAGNOSTIC_PRODUCTS_COMPLETE','Spectral fit gate')
    tasks=[(name,i) for name in design['spectral_truths'] for i in range(design['spectral_replicates_per_truth'])]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))) as pool:
        records=list(pool.map(worker,tasks,chunksize=4))
    summaries=[]
    for truth_name, params in design['spectral_truths'].items():
        selected=[r for r in records if r['truth']==truth_name]
        for mode in ('legacy','propagated'):
            for name,truth in params.items():
                measurements=[]
                for r in selected:
                    fit=r['modes'][mode]['fits'].get(truth_name+'_conservative')
                    error=fit['errors'].get(name) if fit else None
                    measurements.append(dict(estimate=fit['parameters'][name],error=error) if fit and error is not None else None)
                summaries.append(dict(truth_model=truth_name,mode=mode,parameter=name,**summarize(measurements,truth)))
    sed_summary=[]
    for truth_name in design['spectral_truths']:
        for mode in ('legacy','propagated'):
            samples=[r['modes'][mode]['sed'] for r in records if r['truth']==truth_name]
            keys=sorted(set((p['grouping'],p['group_label']) for sample in samples for p in sample))
            for grouping,label in keys:
                measurements=[]
                for sample in samples:
                    point=next((p for p in sample if p['grouping']==grouping and p['group_label']==label),None)
                    measurements.append(dict(estimate=point['e2_dnde']/point['true_e2_dnde'],error=point['e2_dnde_err']/point['true_e2_dnde']) if point else None)
                sed_summary.append(dict(truth_model=truth_name,mode=mode,grouping=grouping,label=label,**summarize(measurements,1.)))
    write(ROOT/'spectral_toys.json',dict(job_id=os.environ['SLURM_JOB_ID'],
        diagnostic_only=True,scope=design['spectral_design'],parameters=summaries,sed=sed_summary,replicates=records))
    print('Conditional spectral/SED propagation complete',flush=True)

if __name__=='__main__':main()
