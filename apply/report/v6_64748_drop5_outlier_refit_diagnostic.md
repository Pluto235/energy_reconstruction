# v6 64748 five-cell outlier removal diagnostic

This is a diagnostic-only Stage F refit. It does not replace or promote the official 44-cell v6 result.

## Test definition

- Removed display cell IDs: `1, 2, 15, 26, 65`
- Corresponding internal IDs: `1, 2, 16, 28, 70`
- Remaining fit cells: 39
- Error model: conservative `sqrt(N_on + B_on)`
- Slurm job: `65134`
- Diagnostic output: `apply/output/stage_f_diagnostics_v6_64748_reselect44_drop5/runs/v6_64748_reselect44_drop5_stage_f`

## Result

| Metric | Official 44-cell fit | Diagnostic 39-cell refit |
| --- | ---: | ---: |
| Preferred model | LogPar | LogPar |
| chi2 | 480.5153 | 186.1195 |
| ndof | 41 | 36 |
| chi2/ndof | 11.7199 | 5.1700 |
| p-value | 2.4229e-76 | 3.8807e-22 |
| phi0 | 2.29793e-12 | 2.23436e-12 |
| alpha | 2.76065 | 2.72371 |
| beta | 0.107148 | 0.0392397 |

At the original frozen parameters, the five removed cells contribute 270.456 to chi2. Removing them without refitting leaves chi2=210.059 across 39 retained cells; refitting reduces this further to 186.120.

The reduced chi2 improves by about 56%, but the refit remains statistically unacceptable (`p=3.88e-22`, about 9.6 sigma one-sided Gaussian equivalent). Eight retained cells still have absolute pull above 3, with the largest at display cell 69 (`+4.29`). The strong beta shift also shows that the removed cells materially influence the inferred spectral curvature.
