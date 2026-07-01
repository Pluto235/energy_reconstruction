#!/usr/bin/env python
"""Diagnostic figures for the v5.4 cellscheck report."""
from pathlib import Path
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"report/assets/v5-cellscheck/v5p4"

cells=[]; po=[]; pv=[]; nh=[]
with open(OUT/"v5p4_percell.csv") as f:
    for r in csv.DictReader(f):
        cells.append(int(r["cell"])); po.append(float(r["pull_orig"])); pv.append(float(r["pull_v5p4"])); nh.append(r["nhit"])
po=np.array(po); pv=np.array(pv); cells=np.array(cells)
order=np.argsort(-np.abs(po))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.5,5.0),constrained_layout=True)
x=np.arange(len(cells))
ax1.bar(x-0.2,np.abs(po[order]),width=0.4,color="#1f77b4",label="original Stage F |pull|")
ax1.bar(x+0.2,np.abs(pv[order]),width=0.4,color="#d62728",label="v5.4 |pull|")
ax1.axhline(2,color="#888",ls="--",lw=1); ax1.axhline(5,color="#aa0000",ls=":",lw=1)
ax1.set_xticks(x); ax1.set_xticklabels([str(c) for c in cells[order]],rotation=90,fontsize=7)
ax1.set_xlabel("cell_id (sorted by original |pull|)"); ax1.set_ylabel("|pull|")
ax1.set_title("Per-cell pull: original (χ²/ndof=5.10) → v5.4 (1.06)"); ax1.legend(fontsize=9); ax1.grid(axis="y",alpha=0.25)

s=np.load(OUT/"v5p4_sed.npz")
e50=s["e50"]; infl=s["e2dnde_err_v54"]/s["e2dnde_err_v4"]
ax2.plot(e50,infl,"o-",color="#d62728",ms=7)
for xi,yi in zip(e50,infl):
    ax2.annotate(f"{yi:.1f}×",(xi,yi),textcoords="offset points",xytext=(0,7),ha="center",fontsize=8)
ax2.axhline(1,color="#888",ls="--",lw=1)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("E_eff [TeV]")
ax2.set_ylabel("SED error inflation  err(v5.4)/err(v4)")
ax2.set_title("Honest error inflation per Nhit point\n(low-Nhit background failure → effectively upper limits)")
ax2.grid(True,which="both",alpha=0.25)
fig.savefig(OUT/"v5p4_diagnostics.png",dpi=170); plt.close(fig)
print(f"Wrote {OUT}/v5p4_diagnostics.png")
