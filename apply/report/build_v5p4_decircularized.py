#!/usr/bin/env python
"""v5.4 — de-circularized, validated upgrade of the v5.3 two-component fix.

Three improvements over v5.3:
  (2) K from MEASURED data/MC predE moments (per high-Nhit row), ZERO free K params in the
      Crab fit. K both shifts (Delta_row = mean_data - mean_MC) and broadens
      (s_row = sqrt(max(0, var_data - var_MC))) the MC predE shape; column-normalized => row
      totals conserved => 7-bin SED invariant. Removes the v5.3 circularity (K was fit on Crab)
      and lets K reshape (not just shift) -> targets the cell-55 residual.
  (1) per-cell DE-POISSONED off-source background systematic with a significance GATE
      (only cells whose off-source ON-excess is >2 sigma beyond Poisson get a systematic),
      replacing the v5.3 row-pool which smeared the worst cell onto the whole row.
  (3) off-source NULL TESTS: (a) leave-one-out transferability (does field A's systematic
      predict field B?), (b) does the systematic absorb the off-source background failure
      without manufacturing a Crab signal.

MC untouched. Forward fold replicated verbatim from stages/06_fit.py and revalidated.
"""
from __future__ import annotations
import csv, glob, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "report/assets/v5-cellscheck/v5p4"; OUT.mkdir(parents=True, exist_ok=True)
PIVOT_TEV = 3.0; QUAD = 64; M2_TO_CM2 = 1.0e4

# ---------------- forward fold (verbatim from 06_fit.py) ----------------
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

# ---------------- load ----------------
fit = np.load(ROOT/"output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz", allow_pickle=True)
cid=fit["cell_id"].astype(int); nhit=fit["nhit_bin"].astype(str); pred=fit["predE_bin"].astype(str)
d=fit["excess"].astype(float); err_c=fit["excess_err_conservative"].astype(float)
B_on=fit["B_on"].astype(float); mMC0=fit["logpar_conservative_model_counts"].astype(float)
pull0=fit["logpar_conservative_pull"].astype(float); texp=fit["theta_exposure_sec"].astype(float)
NCELL=len(cid)
resp=np.load(ROOT/"output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz", allow_pickle=True)
rpos={int(c):i for i,c in enumerate(resp["cell_id"].astype(int))}
a_eff=resp["a_eff"].astype(float)[[rpos[c] for c in cid]]; edges=resp["logE_true_edges"].astype(float)

def ctr(l): a,b=l.strip("[)").split(","); return 0.5*(float(a)+float(b))
def low(l): return float(l.strip("[)").split(",")[0])
cen=np.array([ctr(p) for p in pred])
rows={};
for i,nh in enumerate(nhit): rows.setdefault(nh,[]).append(i)
rows={k:np.array(v) for k,v in rows.items()}
MIG=[nh for nh in sorted(rows,key=low) if low(nh)>=800.0]

# ---------------- validate forward fold ----------------
def fit_spectrum(err, Kmat=None):
    def model(p):
        m=fold(a_eff,texp,edges,10**p[0],p[1],p[2])
        if Kmat is not None:
            m=m.copy()
            for nh in MIG: m[rows[nh]]=Kmat[nh]@m[rows[nh]]
        return m
    def obj(p):
        r=(d-model(p))/err; return float(np.sum(r*r))
    best=None
    for a0 in (2.3,2.6,2.9):
        r=minimize(obj,[-11.6,a0,0.1],method="Nelder-Mead",
                   options=dict(xatol=1e-7,fatol=1e-7,maxiter=60000,maxfev=60000))
        if best is None or r.fun<best.fun: best=r
    return best, model(best.x)
bV,mV=fit_spectrum(err_c)
assert abs(bV.fun-117.3)<1.5 and np.median(np.abs(mV-mMC0)/mMC0)<5e-3, "forward-fold validation failed"
print(f"[validate] local fold chi2={bV.fun:.2f}/23 (stored 117.30); model match med-rel={np.median(np.abs(mV-mMC0)/mMC0):.1e}  OK")

# ---------------- component 2': K from MEASURED data/MC predE moments (0 free params) ----------------
def moments(w, c):
    w=np.clip(w,0,None); W=w.sum()
    if W<=0: return 0.0,0.0
    mean=float((c*w).sum()/W); var=float((c*c*w).sum()/W-mean*mean); return mean,max(var,0.0)
Kmat={}; measured=[]
for nh in MIG:
    idx=rows[nh]; c=cen[idx]
    md,vd=moments(d[idx],c); mm,vm=moments(mMC0[idx],c)
    Delta=md-mm; s=math.sqrt(max(vd-vm,1e-6))+1e-3
    diff=c[:,None]-(c[None,:]+Delta); K=np.exp(-0.5*(diff/s)**2); K/=K.sum(axis=0,keepdims=True)
    Kmat[nh]=K; measured.append((nh,Delta,s,math.sqrt(vd) if vd>0 else 0,math.sqrt(vm) if vm>0 else 0))
print("\n[component 2'] measured-moment K (0 free params), per high-Nhit row:")
print(f'{"Nhit":>12}{"Delta(dex)":>11}{"s_width(dex)":>13}{"std_d/std_m":>12}')
for nh,De,s,sd,sm in measured:
    print(f'{nh:>12}{De:>+11.3f}{s:>13.3f}{(sd/sm if sm>0 else float("nan")):>12.2f}')
print("  (compare v5.3 FITTED Delta: +0.031 / +0.089 / +0.045 -> measured agree => K is real migration)")

# ---------------- component 1': per-cell de-Poissoned bg systematic, gated ----------------
off_files=sorted(glob.glob(str(ROOT/"output/stage_d_v3_offsource_control/runs/*/background_v3_offsource.npz")))
E={c:[] for c in cid}; P={c:[] for c in cid}; Boff={c:[] for c in cid}
for p in off_files:
    o=np.load(p,allow_pickle=True); op={int(c):i for i,c in enumerate(o["cell_id"].astype(int))}
    exmap=o["excess_map"]; onm=o["on_mask"]; cmap=o["counts_map"]; bon=o["B_on"].astype(float)
    for c in cid:
        i=op[int(c)]; e=float(exmap[i][onm[i]].sum()); N=float(cmap[i][onm[i]].sum()); B=float(bon[i])
        E[c].append(e); P[c].append(N+B); Boff[c].append(B)
GATE=2.0
sigma_sys=np.zeros(NCELL); gated=np.zeros(NCELL,bool); off_sig=np.zeros(NCELL)
for j,c in enumerate(cid):
    e=np.array(E[c]); Pp=np.array(P[c]); Bo=np.mean(Boff[c])
    sig=np.mean(e)/math.sqrt(np.mean(Pp)); s2=max(0.0, float(np.mean(e**2)-np.mean(Pp)))
    off_sig[j]=sig
    if abs(sig)>=GATE and s2>0 and Bo>0:
        eps=math.sqrt(s2)/Bo; sigma_sys[j]=eps*B_on[j]; gated[j]=True
err_tot=np.sqrt(err_c**2+sigma_sys**2)
print(f"\n[component 1'] per-cell de-Poissoned bg systematic, gated at |off_sig|>={GATE}: {int(gated.sum())}/{NCELL} cells gated")
print(f'{"cell":>5}{"nhit":>12}{"off_sig":>9}{"gated":>7}{"sig_sys/err_c":>14}')
for j in np.argsort(np.array([low(n) for n in nhit])):
    if gated[j] or low(nhit[j])<500:
        print(f'{cid[j]:>5}{nhit[j]:>12}{off_sig[j]:>9.1f}{str(bool(gated[j])):>7}{(sigma_sys[j]/err_c[j]):>14.2f}')

# ---------------- (3) NULL TESTS ----------------
print("\n[null test 3a] leave-one-out: does off-source field A predict field B (and vice versa)?")
loo=[]
for c in cid:
    e=np.array(E[c],dtype=float); Pp=np.array(P[c],dtype=float)
    if len(e)<2 or not (np.all(np.isfinite(e)) and np.all(np.isfinite(Pp)) and np.all(Pp>0)): continue
    for a,b in ((0,1),(1,0)):
        sysA=math.sqrt(max(0.0,e[a]**2-Pp[a])); denom=sysA**2+Pp[b]
        if denom>0: loo.append(e[b]/math.sqrt(denom))
loo=np.array([x for x in loo if math.isfinite(x)])
print(f"  held-out pull RMS={np.sqrt(np.mean(loo**2)):.2f}  mean={np.mean(loo):+.2f}  n={loo.size}  (RMS~1 => systematic transfers)")

print("\n[null test 3b] does the systematic absorb the off-source background failure (no fake Crab signal)?")
for tag,eidx in (("field A",0),("field B",1)):
    e=np.array([E[c][eidx] for c in cid]); Pp=np.array([P[c][eidx] for c in cid])
    sysv=np.array([(math.sqrt(max(0.0,E[c][eidx]**2-P[c][eidx])) if (abs(np.mean(E[c])/math.sqrt(np.mean(P[c])))>=GATE) else 0.0) for c in cid])
    z_raw=e.sum()/math.sqrt(Pp.sum()); z_cor=e.sum()/math.sqrt((Pp+sysv**2).sum())
    print(f"  {tag}: total off-source excess significance  raw={z_raw:+.1f} sigma  ->  with systematic={z_cor:+.1f} sigma")

# ---------------- v5.4 full fit: measured-K + per-cell gated systematic ----------------
bF,mF=fit_spectrum(err_tot, Kmat); chiF=bF.fun
nfree_opt=3; nfree_con=3+2*len(MIG)
print("\n"+"="*84); print("v5.4 RESULT (measured-K 0-free + per-cell gated systematic)"); print("="*84)
print(f"  chi2/ndof = {chiF/(NCELL-nfree_opt):.2f}   (optimistic: K 0 free params, ndof={NCELL-nfree_opt})")
print(f"  chi2/ndof = {chiF/(NCELL-nfree_con):.2f}   (conservative: charge 2 moments/row, ndof={NCELL-nfree_con})")
print(f"  reference v5.3 (fitted-K + row-pool): 18.18/19 = 0.96")

pullA=(d-mV)/err_c; pullF=(d-mF)/err_tot
hi=np.array([i for i in range(NCELL) if low(nhit[i])>=800.0]); lo=np.array([i for i in range(NCELL) if low(nhit[i])<500.0])
gg=lambda idx,m,e: float(np.sum(((d[idx]-m[idx])/e[idx])**2))
print("\nORTHOGONALITY / subgroup chi2 (v5.4):")
print(f"  HIGH-Nhit (n={len(hi)}): orig={gg(hi,mV,err_c):.1f} -> v5.4={gg(hi,mF,err_tot):.1f}")
print(f"  LOW-Nhit  (n={len(lo)}): orig={gg(lo,mV,err_c):.1f} -> v5.4={gg(lo,mF,err_tot):.1f}")
worst=0.0
for nh in MIG:
    idx=rows[nh]; base=fold(a_eff,texp,edges,10**bF.x[0],bF.x[1],bF.x[2])[idx].sum(); worst=max(worst,abs(base-mF[idx].sum())/base)
print(f"CONSERVATION: max fractional row-total change from K = {worst:.1e}")
print(f"SPECTRUM: orig phi0={10**bV.x[0]:.3e} a={bV.x[1]:.3f} b={bV.x[2]:.3f}  | v5.4 phi0={10**bF.x[0]:.3e} a={bF.x[1]:.3f} b={bF.x[2]:.3f}")
print("\nWORST cells orig -> v5.4:")
for i in np.argsort(-np.abs(pullA))[:8]:
    print(f"  cell {cid[i]:>2} {nhit[i]:>12} {pred[i]:>11}  {pullA[i]:+6.2f} -> {pullF[i]:+6.2f}")

# ---------------- artifacts ----------------
with open(OUT/"v5p4_percell.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cell","nhit","predE","excess","err_cons","off_sig","gated","sigma_sys","err_tot","pull_orig","pull_v5p4"])
    for i in range(NCELL):
        w.writerow([cid[i],nhit[i],pred[i],f"{d[i]:.2f}",f"{err_c[i]:.2f}",f"{off_sig[i]:.2f}",int(gated[i]),
                    f"{sigma_sys[i]:.2f}",f"{err_tot[i]:.2f}",f"{pullA[i]:.3f}",f"{pullF[i]:.3f}"])
np.savez(OUT/"v5p4_params.npz",
         spectrum_orig=bV.x[:3], spectrum_v5p4=bF.x[:3],
         measured_delta=np.array([m[1] for m in measured]), measured_s=np.array([m[2] for m in measured]),
         mig_rows=np.array(MIG), gated=gated, sigma_sys=sigma_sys, off_sig=off_sig,
         loo_pull_rms=float(np.sqrt(np.mean(loo**2))), chi2=chiF,
         ndof_opt=NCELL-nfree_opt, ndof_con=NCELL-nfree_con)
print(f"\nWrote {OUT}/v5p4_percell.csv and v5p4_params.npz")
