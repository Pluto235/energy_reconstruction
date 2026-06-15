# Stage C Observation Reduction Summary

- Input files: 1258
- Processed files: 1258
- Missing time files: 0
- Entry mismatch files: 0
- Input entries: 127,692,389
- Selected configured-cell rows: 28,358,345
- Rough live-time basis: 43.4764 days
- Live-time note: Stage D owns final rate/live-time weighting.

| yyyymm | files | selected rows | rough live-time days |
| --- | ---: | ---: | ---: |
| 202201 | 658 | 14,644,628 | 22.7892 |
| 202202 | 600 | 13,713,717 | 20.6871 |

## Crab ROI coverage diagnostics

Stage C does not apply a Crab ROI cut.
These diagnostics only characterize available sky coverage for Stage D.
Current downstream baseline expects Stage D to choose a fiducial ROI, likely rho<6 deg if the coverage edge is around 8 deg.

- Coordinate: tangent_plane_small_angle around Crab (83.63, 22.01) deg
- Diagnostic status: diagnostic_only_no_cut_applied
- Fiducial radius recommendation: 6 deg
- Edge radius estimate: 7.05 deg
- Edge estimate method: smoothed_annular_density_below_half_plateau_for_3_bins

| radius | count | fraction of selected rows |
| ---: | ---: | ---: |
| rho<2 deg | 875,502 | 0.0308728 |
| rho<4 deg | 3,400,444 | 0.11991 |
| rho<5.5 deg | 6,291,765 | 0.221866 |
| rho<6 deg | 7,374,765 | 0.260056 |
| rho<6.5 deg | 8,433,713 | 0.297398 |
| rho<8 deg | 10,098,031 | 0.356087 |
| rho<10 deg | 10,445,888 | 0.368353 |

Warnings:
- rho<8 and rho<10 counts differ by less than 5%; available Crab-centered coverage may be below 8 deg.
- Per-cell rho<6/rho<10 ratios span more than 0.25 across cells with at least 1000 rho<10 events.
