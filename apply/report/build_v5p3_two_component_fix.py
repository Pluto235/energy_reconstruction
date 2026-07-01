#!/usr/bin/env python
"""v5.3 — unified two-component, instrument-side correction of the Stage F cell-level chi2.

Establishes the orthogonal decomposition by toggling each correction independently:
  (1) HIGH-Nhit predE energy migration -> row-conserving dispersion kernel K (Nhit-row-dependent,
      here FITTED: per-row Delta + shared width s), conserves each Nhit-row total => 7-bin SED invariant.
  (2) LOW-Nhit annulus-background failure -> de-Poissoned off-source background systematic added in
      quadrature to the error bar (row-pooled in this version; v5.4 makes it per-cell + gated).

Forward fold replicated verbatim from stages/06_fit.py and VALIDATED to reproduce 117.3/23 first.
Superseded by v5.4 (de-circularized measured-K + per-cell gated systematic), but this script is the
source of the A/B/C/D orthogonality decomposition reported in the cellscheck report.
"""
from __future__ import annotations
import csv, glob, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "report/assets/v5-cellscheck/v5p3"; OUT.mkdir(parents=True, exist_ok=True)
PIVOT_TEV = 3.0; QUAD = 64; M2_TO_CM2 = 1.0e4

def logpar_flux_tev(E, phi0, alpha, beta):
    r = np.asarray(E)/PIVOT_TEV; lr = np.log(r); return phi0*np.exp((-alpha-beta*lr)*lr)
def integrate_flux_bins(edges, phi0, alpha, beta):
    nodes, w = np.polynomial.legendre.leggauss(QUAD); out = np.zeros(edges.size-1)
    for i,(lo,hi) in enumerate(zip(edges[:-1], edges[1:])):
        xs = 0.5*(hi-lo)*nodes + 0.5*(hi+lo); E = 10.0**xs/1000.0
        out[i] = 0.5*(hi-lo)*float(np.sum(w*logpar_flux_tev(E,phi0,alpha,beta)*math.log(10.0)*E))
    return out
def fold(a_eff, texp, edges, phi0, alpha, beta):
    return M2_TO_CM2*np.einsum("bet,e,t->b", a_eff, integrate_flux_bins(edges,phi0,alpha,beta), texp)

fit = np.load(ROOT/"output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz", allow_pickle=True)
cid=fit["cell_id"].astype(int); nhit=fit["nhit_bin"].astype(str); pred=fit["predE_bin"].astype(str)
d=fit["excess"].astype(float); err_c=fit["excess_err_conservative"].astype(float)
B_on=fit["B_on"].astype(float); mMC0=fit["logpar_conservative_model_counts"].astype(float)
texp=fit["theta_exposure_sec"].astype(float); NCELL=len(cid)
resp=np.load(ROOT/"output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz", allow_pickle=True)
rpos={int(c):i for i,c in enumerate(resp["cell_id"].astype(int))}
a_eff=resp["a_eff"].astype(float)[[rpos[c] for c in cid]]; edges=resp["logE_true_edges"].astype(float)
def low(l): return float(l.strip("[)").split(",")[0])
def ctr(l): a,b=l.strip("[)").split(","); return 0.5*(float(a)+float(b))
cen=np.array([ctr(p) for p in pred])
rows={};
for i,nh in enumerate(nhit): rows.setdefault(nh,[]).append(i)
rows={k:np.array(v) for k,v in rows.items()}
MIG=[nh for nh in sorted(rows,key=low) if low(nh)>=800.0]

def softplus(x): return math.log1p(math.exp(-abs(x)))+max(x,0.0)
def apply_K(model,kpar):
    s=softplus(kpar[-1])+1e-3; m=model.copy()
    for j,nh in enumerate(MIG):
        idx=rows[nh]; c=cen[idx]; diff=c[:,None]-(c[None,:]+kpar[j])
        K=np.exp(-0.5*(diff/s)**2); K/=K.sum(axis=0,keepdims=True); m[idx]=K@model[idx]
    return m
nK=len(MIG)+1; k0=[0.08]*len(MIG)+[-1.5]

def fit_run(err, with_K):
    def model(p):
        m=fold(a_eff,texp,edges,10**p[0],p[1],p[2])
        return apply_K(m,p[3:]) if with_K else m
    def obj(p): r=(d-model(p))/err; return float(np.sum(r*r))
    best=None
    for a0 in (2.3,2.6,2.9):
        x0=[-11.6,a0,0.1]+(k0 if with_K else [])
        r=minimize(obj,x0,method="Nelder-Mead",options=dict(xatol=1e-7,fatol=1e-7,maxiter=80000,maxfev=80000))
        if best is None or r.fun<best.fun: best=r
    return best, model(best.x), NCELL-(3+(nK if with_K else 0))

# validate
bV,mV,_=fit_run(err_c,False)
assert abs(bV.fun-117.3)<1.5 and np.median(np.abs(mV-mMC0)/mMC0)<5e-3
print(f"[validate] chi2={bV.fun:.2f}/23 (stored 117.30)  OK")

# component 2 (row-pooled de-Poissoned systematic)
off=sorted(glob.glob(str(ROOT/"output/stage_d_v3_offsource_control/runs/*/background_v3_offsource.npz")))
pc={c:{"b":[],"v":[]} for c in cid}
for p in off:
    o=np.load(p,allow_pickle=True); op={int(c):i for i,c in enumerate(o["cell_id"].astype(int))}
    exmap=o["excess_map"]; onm=o["on_mask"]; cmap=o["counts_map"]; bon=o["B_on"].astype(float)
    for c in cid:
        i=op[int(c)]; on=float(exmap[i][onm[i]].sum()); N=float(cmap[i][onm[i]].sum()); B=float(bon[i])
        if B>0: pc[c]["b"].append(on/B); pc[c]["v"].append((N+B)/(B*B))
eps={}
for nh,idx in rows.items():
    bs=np.array([v for i in idx for v in pc[cid[i]]["b"]]); vs=np.array([v for i in idx for v in pc[cid[i]]["v"]])
    eps[nh]=math.sqrt(max(0.0,float(np.mean(bs**2)-np.mean(vs))))
sigma_sys=np.array([eps[nhit[i]]*B_on[i] for i in range(NCELL)]); err_tot=np.sqrt(err_c**2+sigma_sys**2)

bA,mA,ndA=fit_run(err_c,False)
bB,mB,ndB=fit_run(err_c,True)
bC,mC,ndC=fit_run(err_tot,False)
bD,mD,ndD=fit_run(err_tot,True)
def line(t,b,nd): print(f"  {t:48s} {b.fun:7.2f}/{nd:2d} = {b.fun/nd:.2f}")
print("="*70); print("v5.3 four configurations"); print("="*70)
line("A original",bA,ndA); line("B +K only",bB,ndB); line("C +bg systematic only",bC,ndC); line("D v5.3 full",bD,ndD)
hi=[i for i in range(NCELL) if low(nhit[i])>=800.0]; lo=[i for i in range(NCELL) if low(nhit[i])<500.0]
gg=lambda idx,m,e: float(np.sum(((d[idx]-m[idx])/e[idx])**2))
print(f"HIGH-Nhit(12): A={gg(hi,mA,err_c):.1f} B={gg(hi,mB,err_c):.1f} C={gg(hi,mC,err_tot):.1f} D={gg(hi,mD,err_tot):.1f}")
print(f"LOW-Nhit(11):  A={gg(lo,mA,err_c):.1f} B={gg(lo,mB,err_c):.1f} C={gg(lo,mC,err_tot):.1f} D={gg(lo,mD,err_tot):.1f}")
worst=max(abs(fold(a_eff,texp,edges,10**bD.x[0],bD.x[1],bD.x[2])[rows[nh]].sum()-mD[rows[nh]].sum())/fold(a_eff,texp,edges,10**bD.x[0],bD.x[1],bD.x[2])[rows[nh]].sum() for nh in rows)
print(f"conservation: max row-total change {worst:.1e}")
pullA=(d-mA)/err_c; pullD=(d-mD)/err_tot
with open(OUT/"v5p3_percell_pull.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cell","nhit","predE","pull_orig","pull_v5p3"])
    for i in range(NCELL): w.writerow([cid[i],nhit[i],pred[i],f"{pullA[i]:.3f}",f"{pullD[i]:.3f}"])
np.savez(OUT/"v5p3_params.npz",chi2=np.array([bA.fun,bB.fun,bC.fun,bD.fun]),ndof=np.array([ndA,ndB,ndC,ndD]),
         spectrum=bD.x[:3],K=bD.x[3:])
print(f"Wrote {OUT}/v5p3_percell_pull.csv and v5p3_params.npz")
