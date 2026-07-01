#!/usr/bin/env python
"""v5.2 fix: Crab-calibrated, row-conserving predE energy-dispersion correction K.

MC response is NOT changed. We multiply the (fixed) MC forward-fold model by a
data/MC predE-dispersion kernel K(Delta(E), s(E)) applied WITHIN each Nhit row.
K is column-normalized => it conserves each Nhit-row total => the robust Nhit-axis
spectrum is invariant; K only redistributes counts along predE to match the data.

Cleanest test form: anchor each row total to the DATA row total (the robust SED),
leave ONLY 4 global K params (Delta and s, smooth in predE). If cell-level chi2
drops from ~5.1 toward ~1, a 4-param Crab-calibrated dispersion correction explains
the predE tilt with the MC untouched.
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/assets/v5-cellscheck/v5p2"; OUT.mkdir(parents=True, exist_ok=True)

fit = np.load(ROOT/"output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz", allow_pickle=True)
cid=fit["cell_id"].astype(int); nhit=fit["nhit_bin"].astype(str); pred=fit["predE_bin"].astype(str)
d=fit["excess"].astype(float); err=fit["excess_err_conservative"].astype(float)
mMC=fit["logpar_conservative_model_counts"].astype(float)
pull0=fit["logpar_conservative_pull"].astype(float)

def ctr(l): a,b=l.strip("[)").split(","); return 0.5*(float(a)+float(b))
cen=np.array([ctr(p) for p in pred])

# group cell indices by Nhit row
rows={}
for i,nh in enumerate(nhit): rows.setdefault(nh,[]).append(i)
CREF=3.5

def softplus(x): return np.log1p(np.exp(-np.abs(x)))+np.maximum(x,0)

def model_with_K(params, anchor="data"):
    d0,d1,s0,s1=params
    model=np.zeros(len(d))
    for nh,idx in rows.items():
        idx=np.array(idx); c=cen[idx]; m=mMC[idx]
        Delta=d0+d1*(c-CREF); s=softplus(s0+s1*(c-CREF))+1e-3
        # K_ij: source bin j -> target bin i, centered at c_j+Delta_j, width s_j; col-normalized
        diff=c[:,None]-(c[None,:]+Delta[None,:])          # (i,j)
        K=np.exp(-0.5*(diff/s[None,:])**2)
        K/=K.sum(axis=0,keepdims=True)                    # sum_i K_ij = 1 (conserve)
        mp=K@m                                            # smeared/shifted model shape
        total = d[idx].sum() if anchor=="data" else m.sum()
        model[idx]= total*mp/mp.sum()
    return model

def chi2(params):
    mdl=model_with_K(params)
    return float(np.sum(((d-mdl)/err)**2))

# baseline: anchored to data row total, NO K (shape = pure MC)
def model_noK():
    model=np.zeros(len(d))
    for nh,idx in rows.items():
        idx=np.array(idx); m=mMC[idx]; model[idx]=d[idx].sum()*m/m.sum()
    return model
m_noK=model_noK(); chi2_noK=float(np.sum(((d-m_noK)/err)**2))

res=minimize(chi2, x0=[0.10,0.05,-2.2,0.2], method="Nelder-Mead",
             options=dict(xatol=1e-4,fatol=1e-4,maxiter=20000))
p=res.x; chi2_K=res.fun
mdl_K=model_with_K(p)
pullK=(d-mdl_K)/err

nK=4
print("="*78)
print("v5.2  predE energy-dispersion correction K (MC untouched, Crab-calibrated)")
print("="*78)
print(f"original Stage F LogPar fit (3 spectrum params)         : chi2/ndof = 117.30/23 = 5.10")
print(f"anchored to data row totals, NO K (isolates predE shape): chi2/ndof = {chi2_noK:.1f}/{len(d)-7:.0f} = {chi2_noK/(len(d)-7):.2f}")
print(f"v5.2: data row totals + 4-param K (Delta,s smooth in E) : chi2/ndof = {chi2_K:.1f}/{len(d)-nK:.0f} = {chi2_K/(len(d)-nK):.2f}")
print(f"\nfitted K params: Delta(c)={p[0]:+.3f}{p[1]:+.3f}*(c-3.5)   s(c)=softplus({p[2]:.2f}{p[3]:+.2f}*(c-3.5))")
print(f"max |pull| before(orig)={np.max(np.abs(pull0)):.2f}  ->  after v5.2={np.max(np.abs(pullK)):.2f}")

# conservation self-check: row sums of model vs data
print("\nCONSERVATION self-check (model row sum must equal data row sum => Nhit spectrum invariant):")
ok=True
for nh,idx in sorted(rows.items()):
    idx=np.array(idx); ds=d[idx].sum(); ms=mdl_K[idx].sum()
    if abs(ds-ms)/ds>1e-6: ok=False
print(f"  all rows conserved to <1e-6: {ok}  => Nhit-axis SED unchanged by K (spectrum protected)")

# implied data energy resolution broadening per row vs independently measured
print("\nPer-Nhit-row: dispersion correction & chi2 (before->after K)")
print(f'{"Nhit":>12}{"ncell":>6}{"Delta(dex)":>11}{"s(dex)":>8}{"sigMC->data":>13}{"chi2_noK":>10}{"chi2_K":>9}')
import numpy as _np
for nh,idx in sorted(rows.items()):
    idx=_np.array(idx); c=cen[idx]; cm=c.mean()
    Delta=p[0]+p[1]*(cm-CREF); s=softplus(_np.array([p[2]+p[3]*(cm-CREF)]))[0]+1e-3
    cn=_np.sum(((d[idx]-m_noK[idx])/err[idx])**2); ck=_np.sum(((d[idx]-mdl_K[idx])/err[idx])**2)
    print(f'{nh:>12}{len(idx):>6}{Delta:>+11.3f}{s:>8.3f}{"+"+format(s,".2f")+"dex":>13}{cn:>10.1f}{ck:>9.1f}')

# per-cell before/after
with open(OUT/"v5p2_percell_pull.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cell","nhit","predE","excess","err","pull_orig_logpar","pull_v5p2_withK"])
    for i in range(len(d)):
        w.writerow([cid[i],nhit[i],pred[i],f"{d[i]:.1f}",f"{err[i]:.1f}",f"{pull0[i]:.3f}",f"{pullK[i]:.3f}"])
np.savez(OUT/"v5p2_K_params.npz", params=p, cref=CREF, chi2_orig=117.30, chi2_noK=chi2_noK, chi2_K=chi2_K)
print(f"\nWrote {OUT}/v5p2_percell_pull.csv  and  v5p2_K_params.npz")
print("\nSED flux points: unchanged from the Nhit-marginalized (7-bin) result by construction (K conserves rows).")
