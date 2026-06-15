# Stage C Observation Reduction Summary

- Input files: 1258
- Processed files: 1258
- Missing time files: 0
- Entry mismatch files: 0
- Input entries: 127,692,389
- Selected v1-cell rows: 77,247,353
- Rough live-time basis: 43.4764 days
- Live-time note: Stage D owns final rate/live-time weighting.

| yyyymm | files | selected rows | rough live-time days |
| --- | ---: | ---: | ---: |
| 202201 | 658 | 39,966,647 | 22.7892 |
| 202202 | 600 | 37,280,706 | 20.6871 |

## Crab ROI coverage diagnostics

Stage C does not apply a Crab ROI cut.
These diagnostics only characterize available sky coverage for Stage D.
Current downstream baseline expects Stage D to choose a fiducial ROI, likely rho<6 deg if the coverage edge is around 8 deg.

- Coordinate: tangent_plane_small_angle around Crab (83.63, 22.01) deg
- Diagnostic status: diagnostic_only_no_cut_applied
- Fiducial radius recommendation: 6 deg
- Edge radius estimate: 7.15 deg
- Edge estimate method: smoothed_annular_density_below_half_plateau_for_3_bins

| radius | count | fraction of selected rows |
| ---: | ---: | ---: |
| rho<2 deg | 2,064,174 | 0.0267216 |
| rho<4 deg | 8,001,201 | 0.103579 |
| rho<5.5 deg | 14,553,278 | 0.188398 |
| rho<6 deg | 16,945,305 | 0.219364 |
| rho<6.5 deg | 19,287,863 | 0.24969 |
| rho<8 deg | 24,142,098 | 0.31253 |
| rho<10 deg | 27,092,422 | 0.350723 |

Warnings:
- Per-cell rho<6/rho<10 ratios span more than 0.25 across cells with at least 1000 rho<10 events.
