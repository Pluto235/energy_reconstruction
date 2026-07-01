#!/usr/bin/env python
"""v5 cellscheck Phase 0: local, zero-Slurm per-cell background-vs-PSF triage.

Reuses ONLY existing artifacts (no cluster):
  - v4 Stage E signal           : N_on, B_on, excess, err_conservative, li_ma sigma
  - official pass5 forward-fold  : per-cell closure obs/pass5 (instrument-side, no Crab fit)
  - observed-excess r715 diag    : data-MC PSF width mismatch
  - 6 x v3 background systematics: per-cell excess spread under annulus/order/link knobs
  - 2 x v3 off-source control    : fake-source residual + 2D annulus residual maps
Output: report/assets/v5-cellscheck/v5_cellscheck_phase0_percell.csv  + printed triage.
"""
from __future__ import annotations
import csv, glob, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # .../apply
OUT = ROOT / "report/assets/v5-cellscheck"
OUT.mkdir(parents=True, exist_ok=True)

FIT_CELLS = [1,2,3,14,15,16,26,27,28,29,30,40,41,42,52,53,54,55,65,66,67,68,69,81,82,83]
DROP4 = [4,17,39,43]
# Stage F drop4 LogPar conservative pulls (from fit summary)
PULL_F = {1:-1.09257,2:1.88999,3:1.57969,14:-1.42557,15:5.53795,16:2.12991,26:-2.37572,
27:1.33548,28:1.55813,29:3.00981,30:1.04622,40:-1.41263,41:-0.886141,42:-0.614422,
52:-1.08048,53:-0.903859,54:0.168863,55:3.97096,65:-3.92409,66:0.887133,67:1.91212,
68:0.775806,69:2.96277,81:0.488674,82:0.747258,83:1.35231}

def load_npz(p): return np.load(p, allow_pickle=True)

# ---- v4 signal reference ----
sig = load_npz(ROOT/"output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz")
cid = sig["cell_id"].astype(int)
idx = {int(c):i for i,c in enumerate(cid)}
N_on=sig["N_on"].astype(float); B_on=sig["B_on"]; excess=sig["excess"]
err_c=sig["excess_err_conservative"]; lima=sig["li_ma_sigma"]; ropt=sig["r_opt_deg"]
nhit=sig["nhit_bin"].astype(str); pred=sig["predE_bin"].astype(str)

# ---- closure (pass5 forward-fold, rayleigh_baseline = v4) ----
closure={}
with open(ROOT/"report/assets/v5-psf-comparison/official_pass5_forward_fold_cell_counts.csv") as f:
    for r in csv.DictReader(f):
        if r["method"]!="rayleigh_baseline": continue
        closure[int(r["cell_id"])]=(float(r["observed_over_expected"]), float(r["pull_conservative"]))

# ---- PSF mismatch r715 ----
r715={}
with open(ROOT/"report/assets/v5-psf-comparison/observed_excess_r715_diagnostic.csv") as f:
    for r in csv.DictReader(f):
        r715[int(r["cell_id"])]=float(r["observed_over_mc_quantile_r715"])

# ---- 6 background-systematics variants: per-cell excess spread ----
bg_files=sorted(glob.glob(str(ROOT/"output/stage_e_v3_background_systematics/runs/*/signal_v3_background_systematics.npz")))
bg_excess={}   # cell -> list of excess across variants
for p in bg_files:
    d=load_npz(p); c=d["cell_id"].astype(int); ex=d["excess"]
    for i,cc in enumerate(c):
        bg_excess.setdefault(int(cc),[]).append(float(ex[i]))

# ---- off-source control (2 fake positions): ON-region residual significance ----
off_files=sorted(glob.glob(str(ROOT/"output/stage_d_v3_offsource_control/runs/*/background_v3_offsource.npz")))
off_sig={}   # cell -> list of (ON_excess/sqrt(B_on)) across positions
off_annres={}
for p in off_files:
    d=load_npz(p); c=d["cell_id"].astype(int)
    exmap=d["excess_map"]; onm=d["on_mask"]; bon=d["B_on"]; annres=d["annulus_residual_mean"]
    for i,cc in enumerate(c):
        on_exc=float(exmap[i][onm[i]].sum())
        s=on_exc/math.sqrt(max(bon[i],1.0))
        off_sig.setdefault(int(cc),[]).append(s)
        off_annres.setdefault(int(cc),[]).append(float(annres[i]))

# ---- assemble per-cell rows ----
rows=[]
for c in FIT_CELLS+DROP4:
    i=idx[c]
    bgfrac=B_on[i]/N_on[i] if N_on[i]>0 else float('nan')
    sig_lima=float(lima[i])
    relerr=err_c[i]/excess[i] if excess[i]>0 else float('nan')
    pf=PULL_F.get(c, float('nan'))
    o_e, pull_pass5 = closure.get(c,(float('nan'),float('nan')))
    psf=r715.get(c,float('nan'))
    # background-knob fractional spread (version-independent) and implied sigma in v4
    bx=bg_excess.get(c,[])
    if len(bx)>=2 and np.median(bx)>0:
        frac=(max(bx)-min(bx))/np.median(bx)
        bg_shift_sigma=frac*(excess[i]/err_c[i]) if err_c[i]>0 else float('nan')
    else:
        frac=float('nan'); bg_shift_sigma=float('nan')
    os=off_sig.get(c,[]); offsig=float(np.mean(np.abs(os))) if os else float('nan')
    rows.append(dict(cell=c,nhit=nhit[i],pred=pred[i],is_fit=(c in FIT_CELLS),
        bgfrac=bgfrac, lima=sig_lima, relerr_pct=100*relerr, pull_F=pf,
        obs_pass5=o_e, pull_pass5=pull_pass5, r715=psf,
        bg_frac_spread=frac, bg_shift_sigma=bg_shift_sigma, offsource_sigma=offsig))

# ---- classify (frozen rule) ----
def classify(r):
    if not math.isfinite(r["pull_F"]) or abs(r["pull_F"])<2.0: return "clean"
    bg = (math.isfinite(r["bg_shift_sigma"]) and r["bg_shift_sigma"]>=2.0) or \
         (math.isfinite(r["offsource_sigma"]) and r["offsource_sigma"]>=3.0)
    psf = math.isfinite(r["r715"]) and r["r715"]>=2.5
    if bg and not psf: return "background"
    if psf and not bg: return "PSF"
    if bg and psf: return "mixed"
    return "unexplained"
for r in rows: r["label"]=classify(r)

# ---- write CSV ----
csvp=OUT/"v5_cellscheck_phase0_percell.csv"
with open(csvp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- print triage ----
def C(x,d=2):
    return "  nan" if (x is None or (isinstance(x,float) and not math.isfinite(x))) else f"{x:.{d}f}"
print("="*128)
print("v5 cellscheck Phase 0 — per-cell triage (instrument-side only; Crab fit NOT used to tune anything)")
print("="*128)
hdr=f'{"cell":>4}{"nhit":>11}{"pred":>11}{"bgfrac":>7}{"liMaσ":>7}{"relE%":>7}{"pullF":>7}{"o/p5":>6}{"r715":>6}{"bgΔσ":>6}{"offσ":>6}  label'
print(hdr); print("-"*len(hdr))
for r in rows:
    star="" if r["is_fit"] else " (drop4)"
    print(f'{r["cell"]:>4}{r["nhit"]:>11}{r["pred"]:>11}{C(r["bgfrac"]):>7}{C(r["lima"],1):>7}'
          f'{C(r["relerr_pct"],1):>7}{C(r["pull_F"]):>7}{C(r["obs_pass5"]):>6}{C(r["r715"]):>6}'
          f'{C(r["bg_shift_sigma"],1):>6}{C(r["offsource_sigma"],1):>6}  {r["label"]}{star}')

# ---- decisive correlations: does |pull_F| track background or PSF? ----
def pear(a,b):
    a=np.array(a);b=np.array(b);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<4: return float('nan'),int(m.sum())
    a,b=a[m],b[m];
    return float(np.corrcoef(a,b)[0,1]), int(m.sum())
fit=[r for r in rows if r["is_fit"]]
absp=[abs(r["pull_F"]) for r in fit]
print("\n"+"="*70)
print("Does cell-level |pull_F| (the χ²≈5 driver) track background or PSF?")
print("="*70)
for name,key in [("background-knob shift (σ)","bg_shift_sigma"),
                 ("off-source residual (σ)","offsource_sigma"),
                 ("background fraction B_on/N_on","bgfrac"),
                 ("PSF mismatch obs/MC r715","r715"),
                 ("Li-Ma significance (pure stats)","lima")]:
    rr,n=pear(absp,[r[key] for r in fit])
    print(f"  corr(|pull_F|, {name:32s}) = {rr:+.2f}   (n={n})")
# low-energy closure trend
print("\nLow-energy closure (obs/pass5) vs background fraction — the +24% over-recovery:")
for r in sorted(fit,key=lambda r:-r["bgfrac"])[:6]:
    print(f"   cell {r['cell']:>2} {r['nhit']:>11} bgfrac={C(r['bgfrac'])}  obs/pass5={C(r['obs_pass5'])}  pullF={C(r['pull_F'])}")
from collections import Counter
print("\nLabel counts (fit cells):", dict(Counter(r["label"] for r in fit)))
print(f"\nWrote: {csvp}")
