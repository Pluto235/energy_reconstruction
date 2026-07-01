# v5 cellscheck — Phase 0 findings (local, zero-Slurm)

Generated from existing artifacts only (v4 Stage E signal, official pass5 forward-fold,
observed-excess r715, 6× v3 background-systematics, 2× v3 off-source control,
v4 response + Stage F fit). No Crab fit was used to tune anything.

## Headline
The cell-level χ²≈5 in v4 Stage F is **NOT** driven by the background model or the
angular PSF. It is driven by a **predE energy-migration mismatch: the data's
reconstructed-energy distribution is wider and more up-scattered than the MC
response encodes.**

## Evidence
1. **No instrument-side proxy explains the pulls** (26 fit cells):
   - corr(|pull_F|, background-knob shift σ) = −0.05
   - corr(|pull_F|, off-source residual σ)   = +0.14
   - corr(|pull_F|, background fraction B/N)  = +0.05
   - corr(|pull_F|, PSF mismatch obs/MC r715) = −0.37 (n=12)
   - corr(|pull_F|, |obs/pass5 − 1|)          = +0.20

2. **Systematic predE tilt within each fixed-Nhit row** (obs/pass5 low→high predE):
   - 6/7 rows have a positive predE tilt (mean slope +0.90)
   - 6/7 rows: positive pull at the high-predE edge, negative pull at the low-predE edge
   - e.g. [1100,2000): 0.59→1.04→1.18→1.02→5.38 ; [2000,3000): 0.75→1.56→8.42

3. **Direct data-vs-MC predE shape (within-row, normalization removed)**:
   - 6/7 rows: data predE mean shifted higher than BOTH LogPar-model and external pass5 fold (Δ≈+0.09 dex)
   - 5/7 rows: data predE distribution wider than MC (std ratio mean 1.17, up to 1.70)
   - only clean row: [500,800) (Δ≈0, ratio≈1) — also the lowest-pull row

## Why this ties everything together
- The 3 v5 ablations (binning / PSF aperture / cell selection) could not move χ²≈5
  because none of them touches the response energy migration.
- predE-split binning (12-bin) was noisy while the Nhit axis (7-bin) was robust —
  exactly because the mismatch lives on the predE axis.
- Background IS biased at a few specific geometries (off-source σ: cell 1=13.7,
  15=6.2, 65=7.8, drop4 39=16.8) — real but secondary and localized.

## Alternatives ruled out (within-row disentangling)
The within-Nhit-row predE tilt of obs/pass5 is tested against every competing cause:
- **containment**: forced to 1.000 in all cells (containment=1 contract, spread 0) → cannot drive the tilt.
- **background**: within-row corr(obs/pass5, B_on/N_on) mean +0.15 (weak); bg-knob & off-source uncorrelated with pull → not the driver.
- **angular PSF**: a wider-than-MC PSF acts through containment with the OPPOSITE sign
  (it would push obs/pass5 *down* at large r_opt), so it cannot explain the high-predE *over*-recovery.
- **spectral model**: the tilt is present vs the EXTERNAL pass5 fold too, and is organized by
  predE at fixed Nhit (≈fixed true E), not by true energy → not a spectral-shape artifact.
- **aperture r_opt**: corr +0.45 but r_opt is monotonic in predE (collinear), and containment=1 removes any aperture-size flux effect.
- **survivor**: corr(obs/pass5, predE) mean **+0.64** (6/7 rows) → the tilt is an energy-migration effect.

## Status
v5 diagnosis is CLOSED: the cell-level χ²≈5 is an **energy-migration data-MC mismatch**
(data predE wider & up-shifted than the MC response), localized to the high-predE edges
of each Nhit row; worst at the highest-energy rows; the [500,800) row is clean.
A v5 fix is a separate, later decision (not started).

## Artifacts
- `v5_cellscheck_phase0_percell.csv` — per-cell triage table
- `build_v5_cellscheck_phase0.py` — generator (in apply/report/)
