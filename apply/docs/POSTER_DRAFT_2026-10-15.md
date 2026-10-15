# Collaboration-meeting poster draft — 15 October 2026

The English poster draft is saved locally at `prestation/poster/` and versioned in the separate poster repository:

- Template base: `e4d362ccfd5a7a34ff6de7c378df2dbdcd875a2b`.
- Draft: [poster commit d8a76b6](https://github.com/Pluto235/poster/commit/d8a76b6e724004a4ee26bf76c14c72fe9d92de04), branch `codex/energy-reconstruction-draft`.
- Deliverables: `main.pdf`, `preview.png`, editable `main.tex`/`content.tex`, `energy_poster_draft.zip`, and an English README containing the narrative, figure inventory, provenance and completion checklist.
- Original local folder: `.recovery/poster-before-template-20261015/`. The template main branch and original template files are preserved.

The 60 × 120 cm single-page draft contains six sections and five figure groups: model architecture; weighted true/predicted-energy counts; MC width and bias; the frozen 44-cell selector; and the historical common-GTI Crab spectral comparison. Red text flags unresolved authorship/contact information, MC/checkpoint provenance, baseline and systematic studies, fit quality, and the incomplete January spectrum.

Scientific boundaries are explicit: the model-64748 evaluation file contains 857,926 events; weighted log-residual bias/scatter/RMSE are 0.020/0.200/0.201 dex. The plotted bin-wise width has its original definition, standard deviation of predicted log-energy within a finite true-energy bin. The common-GTI comparison has chi-square/ndf = 20.94 and does not prove a precision improvement. January reprocessing yielded 681 prediction files but the frozen 44-cell spectral chain remains blocked by cell62 background negativity. No January spectrum is claimed.

Validation: successful Tectonic compile with no overfull/underfull warnings; one 60 × 120 cm page; all six sections present; body text clear of the footer; red annotations present; no characters outside the page; all copied-asset hashes and the exact 44/91 selector checked; final PNG visually reviewed; ZIP integrity checked. A lightweight 20-bin display table was derived from saved predictions, with overall bias/scatter matching the saved metrics within 1e-7. No training, batch reconstruction or heavy computation ran locally.

Small evidence snapshots and validation/output hashes are recorded in the poster repository under `evidence/`. Detailed source paths, public references and final-meeting requirements are in its README. The draft is ready for scientific discussion, not final printing.
