# Stage C Observation Reduction Summary

- Input files: 1258
- Processed files: 1258
- Missing time files: 0
- Entry mismatch files: 0
- Input entries: 127,692,389
- Selected v1-cell rows: 83,905,806
- Rough live-time basis: 43.4764 days
- Live-time note: Stage D owns final rate/live-time weighting.

| yyyymm | files | selected rows | rough live-time days |
| --- | ---: | ---: | ---: |
| 202201 | 658 | 43,409,052 | 22.7892 |
| 202202 | 600 | 40,496,754 | 20.6871 |

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
| rho<2 deg | 2,232,361 | 0.0266056 |
| rho<4 deg | 8,644,565 | 0.103027 |
| rho<5.5 deg | 15,718,830 | 0.187339 |
| rho<6 deg | 18,301,034 | 0.218114 |
| rho<6.5 deg | 20,830,268 | 0.248258 |
| rho<8 deg | 26,041,761 | 0.310369 |
| rho<10 deg | 29,178,488 | 0.347753 |

Warnings:
- Per-cell rho<6/rho<10 ratios span more than 0.25 across cells with at least 1000 rho<10 events.
