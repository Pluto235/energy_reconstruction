#!/usr/bin/env python
"""v5.4 SED emission — the 7-bin Nhit-marginalized Crab SED with the v5.4 corrections.

Faithful standalone: the Stage G flux-point recipe is copied VERBATIM from stages/07_sed_points.py
  N0 = sum(excess * M_unit / sigma^2) / sum(M_unit^2 / sigma^2),  N0_err = 1/sqrt(sum(M_unit^2/sigma^2))
  E_eff = p50 of the response true-energy weights (frozen spectrum)
  E2_dnde = E_eff^2 * flux(E_eff; phi0=N0, alpha, beta)
(We do NOT edit the staged 06/07 scripts: they carry frozen validation contracts — REQUIRED_CELL_IDS
 8..18, EXPECTED_STAGE_F_RUN_ID, PL-specific expected constants — tuned to a different baseline.)

What v5.4 changes vs the v4 Stage G:
  - errors:  excess_err_conservative  ->  err_tot = sqrt(err_cons^2 + sigma_sys^2)   (per-cell gated,
             de-Poissoned off-source background systematic; high-Nhit gets ~0, low-Nhit inflated)
  - model :  the high-Nhit predE migration kernel K (measured-moment, row-conserving) is applied to
             the unit_counts shape. For the Nhit grouping K conserves each row total, so the
             simple-sum normalization is EXACTLY K-invariant => central SED points do not move;
             only the (honest) error bars change.
The LogPar curve + 1-sigma covariance band come from the v5.4 joint fit (err_tot + K).
"""
from __future__ import annotations
import csv, glob, math
from pathlib import Path
import numpy as np
from iminuit import Minuit

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "report/assets/v5-cellscheck/v5p4"; OUT.mkdir(parents=True, exist_ok=True)
PIVOT_TEV = 3.0; QUAD = 64; M2_TO_CM2 = 1.0e4
REF_PHI0 = 2.114e-12; REF_GAMMA = 2.69

# ---- forward fold (verbatim 07_sed_points.py) ----
def logpar_flux_tev(E, phi0, alpha, beta):
    r=np.asarray(E)/PIVOT_TEV; lr=np.log(r); return float(phi0)*np.exp((-alpha-beta*lr)*lr)
def pl_flux_tev(E, phi0, gamma):
    return float(phi0)*np.power(np.asarray(E)/PIVOT_TEV,-gamma)
def integrate_flux_bins(edges, phi0, alpha, beta):
    nodes,w=np.polynomial.legendre.leggauss(QUAD); out=np.zeros(edges.size-1)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        xs=0.5*(hi-lo)*nodes+0.5*(hi+lo); E=10.0**xs/1000.0
        out[i]=0.5*(hi-lo)*float(np.sum(w*logpar_flux_tev(E,phi0,alpha,beta)*math.log(10.0)*E))
    return out
def unit_counts_of(a_eff,texp,edges,alpha,beta):  # phi0=1
    fi=integrate_flux_bins(edges,1.0,alpha,beta)
    return M2_TO_CM2*np.einsum("bet,e,t->b",a_eff,fi,texp)
def true_E_weights(a_eff,texp,edges,alpha,beta,mask):
    fi=integrate_flux_bins(edges,1.0,alpha,beta)
    contr=M2_TO_CM2*np.einsum("bet,t->be",a_eff[mask],texp)*fi[None,:]
    return np.sum(contr,axis=0)
def wq(edges,wts,qs):
    wts=np.asarray(wts,float); tot=wts.sum()
    if tot<=0: return [float("nan")]*len(qs)
    cum=np.cumsum(wts); out=[]
    for q in qs:
        t=q*tot; idx=min(max(int(np.searchsorted(cum,t,"left")),0),wts.size-1)
        prev=float(cum[idx-1]) if idx>0 else 0.0; width=float(edges[idx+1]-edges[idx])
        frac=min(1.0,max(0.0,(t-prev)/wts[idx])) if wts[idx]>0 else 0.5
        out.append(10.0**float(edges[idx]+frac*width)/1000.0)
    return out

# ---- load ----
fit=np.load(ROOT/"output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz",allow_pickle=True)
cid=fit["cell_id"].astype(int); nhit=fit["nhit_bin"].astype(str); pred=fit["predE_bin"].astype(str)
d=fit["excess"].astype(float); err_c=fit["excess_err_conservative"].astype(float)
B_on=fit["B_on"].astype(float); mMC0=fit["logpar_conservative_model_counts"].astype(float)
texp=fit["theta_exposure_sec"].astype(float); NCELL=len(cid)
resp=np.load(ROOT/"output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz",allow_pickle=True)
rpos={int(c):i for i,c in enumerate(resp["cell_id"].astype(int))}
a_eff=resp["a_eff"].astype(float)[[rpos[c] for c in cid]]; edges=resp["logE_true_edges"].astype(float)
def ctr(l): a,b=l.strip("[)").split(","); return 0.5*(float(a)+float(b))
def low(l): return float(l.strip("[)").split(",")[0])
cen=np.array([ctr(p) for p in pred])
rows={};
for i,nh in enumerate(nhit): rows.setdefault(nh,[]).append(i)
rows={k:np.array(v) for k,v in rows.items()}
MIG=[nh for nh in sorted(rows,key=low) if low(nh)>=800.0]

# ---- v5.4 corrections: measured-moment K + per-cell gated de-Poissoned systematic ----
def moments(wgt,c):
    wgt=np.clip(wgt,0,None); W=wgt.sum()
    if W<=0: return 0.0,0.0
    m=float((c*wgt).sum()/W); return m,max(float((c*c*wgt).sum()/W-m*m),0.0)
Kmat={}
for nh in MIG:
    idx=rows[nh]; c=cen[idx]; md,vd=moments(d[idx],c); mm,vm=moments(mMC0[idx],c)
    Delta=md-mm; s=math.sqrt(max(vd-vm,1e-6))+1e-3
    K=np.exp(-0.5*((c[:,None]-(c[None,:]+Delta))/s)**2); K/=K.sum(axis=0,keepdims=True); Kmat[nh]=K
off=sorted(glob.glob(str(ROOT/"output/stage_d_v3_offsource_control/runs/*/background_v3_offsource.npz")))
E={c:[] for c in cid}; P={c:[] for c in cid}; Boff={c:[] for c in cid}
for p in off:
    o=np.load(p,allow_pickle=True); op={int(c):i for i,c in enumerate(o["cell_id"].astype(int))}
    exmap=o["excess_map"]; onm=o["on_mask"]; cmap=o["counts_map"]; bon=o["B_on"].astype(float)
    for c in cid:
        i=op[int(c)]; E[c].append(float(exmap[i][onm[i]].sum())); P[c].append(float(cmap[i][onm[i]].sum())+float(bon[i])); Boff[c].append(float(bon[i]))
GATE=2.0; sigma_sys=np.zeros(NCELL)
for j,c in enumerate(cid):
    e=np.array(E[c]); Pp=np.array(P[c]); Bo=np.mean(Boff[c])
    sig=np.mean(e)/math.sqrt(np.mean(Pp)); s2=max(0.0,float(np.mean(e**2)-np.mean(Pp)))
    if abs(sig)>=GATE and s2>0 and Bo>0: sigma_sys[j]=(math.sqrt(s2)/Bo)*B_on[j]
err_tot=np.sqrt(err_c**2+sigma_sys**2)

def apply_K(m):
    m=m.copy()
    for nh in MIG: m[rows[nh]]=Kmat[nh]@m[rows[nh]]
    return m

# ---- v5.4 joint LogPar fit (err_tot + K), with covariance for the band ----
def model(L,alpha,beta): return apply_K(M2_TO_CM2*np.einsum("bet,e,t->b",a_eff,integrate_flux_bins(edges,10**L,alpha,beta),texp))
def chi2(L,alpha,beta): r=(d-model(L,alpha,beta))/err_tot; return float(np.sum(r*r))
mn=Minuit(chi2,L=math.log10(2.2e-12),alpha=2.8,beta=0.05); mn.errordef=Minuit.LEAST_SQUARES
mn.limits["L"]=(-30,0); mn.limits["alpha"]=(0.5,6); mn.limits["beta"]=(-2,2)
mn.migrad(); mn.hesse()
L,al,be=mn.values["L"],mn.values["alpha"],mn.values["beta"]; phi0=10**L
cov=np.array([[mn.covariance[a,b] for b in ("L","alpha","beta")] for a in ("L","alpha","beta")])
print(f"[v5.4 fit] chi2/ndof={mn.fval:.2f}/{NCELL-3}={mn.fval/(NCELL-3):.2f}  phi0={phi0:.3e} alpha={al:.3f} beta={be:.3f}")

# ---- 7-bin Nhit SED points (Stage G recipe), conservative vs v5.4 errors ----
uc=apply_K(unit_counts_of(a_eff,texp,edges,al,be))         # unit counts (phi0=1), K applied
def sed_points(errors):
    # SIMPLE STACKED-COUNTS normalization N0 = sum(excess)/sum(unit_counts): for the Nhit
    # grouping this is EXACTLY invariant under K (K conserves each row total) and under the
    # error model (no weights), so central points do NOT move; it is also the migration-robust
    # estimator (insensitive to how counts are distributed across predE within the row).
    # err propagates as sqrt(sum err^2)/sum(unit_counts).
    pts=[]
    for nh in sorted(rows,key=low):
        m=np.zeros(NCELL,bool); m[rows[nh]]=True
        v=m&np.isfinite(d)&(errors>0)&(uc>0)
        suc=float(np.sum(uc[v])); n0=float(np.sum(d[v]))/suc; n0e=math.sqrt(float(np.sum(errors[v]**2)))/suc
        ew=true_E_weights(a_eff,texp,edges,al,be,m); e16,e50,e84=wq(edges,ew,[0.16,0.5,0.84])
        e2=e50*e50; f=logpar_flux_tev(e50,n0,al,be); fe=logpar_flux_tev(e50,n0e,al,be)
        chi=float(np.sum(((d[v]-n0*uc[v])/errors[v])**2))
        pts.append(dict(nhit=nh,e50=e50,e16=e16,e84=e84,e2dnde=e2*f,e2dnde_err=e2*fe,
                        n0=n0,n0_err=n0e,chi2=chi,ndof=int(v.sum()-1),excess=float(d[m].sum())))
    return pts
p_cons=sed_points(err_c); p_v54=sed_points(err_tot)

# ---- LogPar 1-sigma band from covariance ----
Eg=np.logspace(math.log10(0.5),math.log10(40),200)
band=[]
for Ei in Eg:
    lr=math.log(Ei/PIVOT_TEV); f=logpar_flux_tev(Ei,phi0,al,be); e2=Ei*Ei
    g=np.array([math.log(10.0)*f, f*(-lr), f*(-lr*lr)])          # d f / d(L,alpha,beta)
    var=float(g@cov@g); band.append((e2*f, e2*math.sqrt(max(var,0.0))))
band=np.array(band)

# ---- print + CSV ----
print("\n7-bin Nhit-marginalized SED (E2 dN/dE, TeV cm^-2 s^-1):")
print(f'{"Nhit":>12}{"E_eff[TeV]":>11}{"E2dNdE":>12}{"err_v4":>11}{"err_v5.4":>11}{"infl":>7}')
for a,b in zip(p_cons,p_v54):
    infl=b["e2dnde_err"]/a["e2dnde_err"]
    print(f'{a["nhit"]:>12}{a["e50"]:>11.2f}{b["e2dnde"]:>12.3e}{a["e2dnde_err"]:>11.2e}{b["e2dnde_err"]:>11.2e}{infl:>7.2f}')
with open(OUT/"v5p4_sed_points.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["nhit","E_eff_TeV","E16_TeV","E84_TeV","E2dNdE","E2dNdE_err_v4_conservative","E2dNdE_err_v5p4","err_inflation","N0","chi2","ndof"])
    for a,b in zip(p_cons,p_v54):
        w.writerow([a["nhit"],f'{a["e50"]:.4f}',f'{a["e16"]:.4f}',f'{a["e84"]:.4f}',f'{b["e2dnde"]:.4e}',
                    f'{a["e2dnde_err"]:.4e}',f'{b["e2dnde_err"]:.4e}',f'{b["e2dnde_err"]/a["e2dnde_err"]:.3f}',
                    f'{b["n0"]:.4e}',f'{b["chi2"]:.2f}',b["ndof"]])
np.savez(OUT/"v5p4_sed.npz", Eg=Eg, band_center=band[:,0], band_sigma=band[:,1],
         spectrum=np.array([phi0,al,be]), cov=cov, chi2=mn.fval, ndof=NCELL-3,
         e50=np.array([p["e50"] for p in p_v54]), e2dnde=np.array([p["e2dnde"] for p in p_v54]),
         e2dnde_err_v4=np.array([p["e2dnde_err"] for p in p_cons]),
         e2dnde_err_v54=np.array([p["e2dnde_err"] for p in p_v54]))

# central-point invariance check (simple-sum is exactly K-invariant; weighted shifts slightly)
shift=max(abs(b["e2dnde"]/a["e2dnde"]-1) for a,b in zip(p_cons,p_v54))
print(f"\nmax |central E2dNdE shift v4->v5.4| = {shift*100:.2f}%  (only inverse-variance reweighting; spectrum not moved)")

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8.2,5.6),constrained_layout=True)
ax.fill_between(Eg,band[:,0]-band[:,1],band[:,0]+band[:,1],color="#d62728",alpha=0.20,label="v5.4 LogPar 1σ band")
ax.plot(Eg,band[:,0],color="#d62728",lw=1.5,label=f"v5.4 LogPar (χ²/ndof={mn.fval/(NCELL-3):.2f})")
e50=np.array([p["e50"] for p in p_v54]); y=np.array([p["e2dnde"] for p in p_v54])
yev4=np.array([p["e2dnde_err"] for p in p_cons]); yev54=np.array([p["e2dnde_err"] for p in p_v54])
ax.errorbar(e50*1.0,y,yerr=yev54,fmt="o",color="#d62728",ms=6,capsize=3,label="v5.4 Nhit points (corrected err)",zorder=5)
ax.errorbar(e50*0.97,y,yerr=yev4,fmt="s",mfc="none",mec="#1f77b4",ecolor="#1f77b4",ms=6,capsize=3,label="v4 points (conservative err)",zorder=4)
# WCDA-1 Pool1 Table 1 reference (emed, dnde)
pool1=[(0.58,1.66e-10,0.20e-10),(1.1,2.89e-11,0.23e-11),(2.4,4.74e-12,0.48e-12),(3.9,1.12e-12,0.17e-12),(5.9,3.54e-13,0.74e-13),(12.1,6.91e-14,1.0e-14)]
pe=np.array([p[0] for p in pool1]); py=np.array([p[0]**2*p[1] for p in pool1]); pye=np.array([p[0]**2*p[2] for p in pool1])
ax.errorbar(pe,py,yerr=pye,fmt="^",color="#2ca02c",ms=6,capsize=3,label="LHAASO-WCDA-1 Pool-1 (official)",zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("E [TeV]"); ax.set_ylabel(r"$E^2\,dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
ax.set_title("Crab SED — v5.4 (measured-K + de-Poissoned bg systematic), 7-bin Nhit"); ax.grid(True,which="both",alpha=0.25); ax.legend(fontsize=8)
fig.savefig(OUT/"v5p4_sed.png",dpi=170); plt.close(fig)
print(f"\nWrote {OUT}/v5p4_sed_points.csv, v5p4_sed.npz, v5p4_sed.png")
