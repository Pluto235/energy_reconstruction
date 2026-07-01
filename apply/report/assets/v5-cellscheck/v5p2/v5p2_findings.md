# v5.2 — Crab-calibrated row-conserving predE dispersion correction K

MC untouched. A data/MC predE-dispersion kernel K(Delta(E), s(E)) is applied within
each Nhit row, column-normalized so each Nhit-row total is conserved (=> Nhit-axis
spectrum invariant). Row totals anchored to DATA; the only free dof are the 4 global
K params. Calibrated on Crab.

## Result (cell-level Stage F chi2)
- original Stage F LogPar (3 spectrum params):            117.3 / 23 = 5.10
- anchored, NO K (isolates predE shape mismatch):         125.1 / 19 = 6.58
- v5.2, anchored + 4-param smooth K:                       97.9 / 22 = 4.45
- conservation self-check: all rows conserved < 1e-6  => Nhit/7-bin SED unchanged.
- fitted: Delta(c) = +0.093 + 0.030*(c-3.5) dex ; s grows with energy (~0.05->0.09 dex)
  — consistent with the independently measured dispersion (up-shift +0.09 dex, width x1.17).

## Per-row chi2 (no-K -> with-K)
| Nhit | no-K | with-K | verdict |
|---|---|---|---|
| [800,1100)  | 19.2 | 5.0  | migration, K fixes |
| [1100,2000) | 36.1 | 19.7 | migration, K helps |
| [2000,3000) | 7.7  | 3.6  | migration, K helps |
| [300,500)   | 21.0 | 19.0 | partial |
| [200,300)   | 35.7 | 36.8 | NOT migration (cell 15) — K cannot fix |
| [500,800)   | 0.1  | 8.6  | already clean — global K damages it |
| [125,200)   | 5.2  | 5.2  | flat |

## Conclusion (refined diagnosis)
The cell-level chi2 is NOT one uniform effect. It is two components:
1. a real high-energy predE migration (data wider/up-shifted than MC) that a
   dispersion correction K genuinely reduces (high-E rows improve 2-4x);
2. discrete non-migration anomalies, dominated by **cell 15 / [200,300)** (~31 of 117
   total chi2, highest off-source residual 6.2 sigma in Phase 0), that a dispersion
   model cannot and should not "fix".
A smooth global dispersion correction therefore CANNOT drive cell-level chi2 to ~1
(even protecting clean rows and excluding cell 15, the floor is ~3.4). The migration
component is real and partially correctable; the residual is cell-specific.

## What v5.2 delivers
- SED main result = Nhit-marginalized (7-bin), provably unchanged by K (conservation).
- A validated, Crab-calibrated migration correction for the high-energy rows.
- Isolation of cell 15 (and similar) as a separate, non-migration anomaly to handle individually.

## Artifacts
- build_v5p2_predE_dispersion_fix.py
- v5p2_percell_pull.csv (orig vs v5.2 pull per cell)
- v5p2_K_params.npz
