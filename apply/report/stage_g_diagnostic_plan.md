# Stage G Diagnostic Subset Plan

Stage F after the S0 unit fix shows that the absolute normalization is now in the right range, but the low-Nhit / low-predicted-energy cells dominate the fit residuals. These cells should not enter the formal Stage G SED baseline until the Stage D sideband/background systematics are resolved.

## Low-energy diagnosis

- Diagnostic report: `apply/report/low_energy_cell_diagnostics.html`
- Diagnostic CSV: `apply/plot/low_energy_cell_diagnostics/low_energy_cell_diagnostics.csv`
- Cells 1-7 show large central quicklook excess and/or large Stage F pulls.
- Cells 1-3 are the dominant hard outliers.
- Cells 6-7 are transition cells: they are less pathological than cells 1-5, but retaining them still leaves a poor Stage F fit.

## Subset trials

| subset | included cells | reference observed/expected | preferred model | PL chi2/ndof | status |
| --- | --- | ---: | --- | ---: | --- |
| drop1to5 | 6-18 | 1.256 | LogPar | 82.65 / 11 | diagnostic only; still poor |
| cells8to18 | 8-18 | 1.305 | PL | 17.83 / 9 | Stage G diagnostic baseline |
| cell5_plus_8to18 | 5, 8-18 | 1.544 | LogPar | 39.33 / 10 | reject for baseline; cell 5 remains a strong residual |
| cell6_plus_8to18 | 6, 8-18 | 1.174 | LogPar | 50.73 / 10 | reject for baseline; requires strong curvature and leaves poor residuals |
| cells5to6_plus_8to18 | 5-6, 8-18 | 1.327 | LogPar | 82.34 / 11 | reject for baseline; comparable failure mode to drop1to5 |

The `cells8to18` subset is the current high-confidence diagnostic baseline because it passes the reference-count preflight and gives a defensible PL fit. It is still a diagnostic subset, not a formal published SED configuration.

Cells 5 and 6 can be kept only as stress-test/systematics probes for the low-energy transition region. They should not be added to the Stage G diagnostic baseline until the low-energy background and response-shape problem is fixed and revalidated.

## Files

- `apply/config/cell_subset_drop1to5_stageg_diag.csv`
- `apply/config/cell_subset_cells8to18_stageg_diag.csv`
- `apply/config/cell_subset_cell5_plus_8to18_stageg_probe.csv`
- `apply/config/cell_subset_cell6_plus_8to18_stageg_probe.csv`
- `apply/config/cell_subset_cells5to6_plus_8to18_stageg_probe.csv`
- `apply/output/stage_f/runs/codex_stage_f_drop1to5_diag/`
- `apply/output/stage_f/runs/codex_stage_f_cells8to18_diag/`
- `apply/output/stage_f/runs/codex_stage_f_cell5_plus_8to18_probe/`
- `apply/output/stage_f/runs/codex_stage_f_cell6_plus_8to18_probe/`
- `apply/output/stage_f/runs/codex_stage_f_cells5to6_plus_8to18_probe/`
- `apply/report/stage_f_drop1to5_diag_report.html`
- `apply/report/stage_f_cells8to18_diag_report.html`
- `apply/report/stage_f_cell5_plus_8to18_probe_report.html`
- `apply/report/stage_f_cell6_plus_8to18_probe_report.html`
- `apply/report/stage_f_cells5to6_plus_8to18_probe_report.html`

## Stage G rule for now

Use `cells8to18_stageg_diag` for Stage G diagnostic plots and SED-point prototyping. Do not use cells 1-7 in the Stage G baseline until low-energy sideband and response-shape systematics are fixed and revalidated.
