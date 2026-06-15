# Stage D Background Summary

- Run id: `v3_stage_d_offsource_ra73p63`
- Stage C input: `/mnt/mydisk/server/projects/energy_reconstruction/apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024`
- PSF input: `/mnt/mydisk/server/projects/energy_reconstruction/apply/output/stage_b_v3_candidate/runs/slurm_42023/psf_v3_candidate.npz`
- Background mode: `crab_roi_local`
- Background method: `annulus_quadratic`
- Background form: `direct_expectation`
- ROI fiducial radius: 6.0 deg
- ROI edge diagnostic radius: 8.0 deg

| cell | Nhit bin | predE bin | events | masked frac | r_opt deg | B_on | off pixels | warnings |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | [125,200) | [2,2.5) | 3939808 | 0.01789 | 0.78161 | 1497.6 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 2 | [125,200) | [2.5,3) | 9200327 | 0.01404 | 0.72115 | 2516.12 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 3 | [125,200) | [3,3.25) | 2149690 | 0.0198 | 0.78782 | 577.276 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 4 | [125,200) | [3.25,3.5) | 1021240 | 0.05539 | 1.0633 | 656.207 | 4400 | core_extrapolation_warning, surface_fit_not_successful |
| 5 | [125,200) | [3.5,3.75) | 496152 | 0.129 | 1.3398 | 583.631 | 5032 | core_extrapolation_warning, surface_fit_not_successful |
| 6 | [125,200) | [3.75,4.0) | 226753 | 0.2162 | 1.555 | 500.043 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 7 | [125,200) | [4.0,4.25) | 95276 | 0.322 | 1.7795 | 351.204 | 6280 | core_extrapolation_warning, surface_fit_not_successful |
| 8 | [125,200) | [4.25,4.5) | 37999 | 0.4319 | 1.9745 | 166.694 | 4928 | - |
| 9 | [125,200) | [4.5,4.75) | 15749 | 0.4467 | 2.0433 | 93.2513 | 4928 | - |
| 10 | [125,200) | [4.75,5.0) | 7028 | 0.5596 | 2.217 | 34.8526 | 4928 | - |
| 11 | [125,200) | [5,6) | 5737 | 0.5117 | 2.2655 | 38.3529 | 4928 | - |
| 12 | [125,200) | >=6 | 60 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 13 | [200,300) | [2,2.5) | 200044 | 0.00785 | 0.63856 | 15.7471 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 14 | [200,300) | [2.5,3) | 3311312 | 0.00693 | 0.55424 | 297.515 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 15 | [200,300) | [3,3.25) | 1555252 | 0.007999 | 0.54687 | 125.058 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 16 | [200,300) | [3.25,3.5) | 732511 | 0.01217 | 0.72875 | 109.992 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 17 | [200,300) | [3.5,3.75) | 349353 | 0.04519 | 1.0331 | 192.47 | 4400 | core_extrapolation_warning, surface_fit_not_successful |
| 18 | [200,300) | [3.75,4.0) | 166678 | 0.1246 | 1.301 | 237.258 | 5032 | core_extrapolation_warning, surface_fit_not_successful |
| 19 | [200,300) | [4.0,4.25) | 69627 | 0.2116 | 1.5261 | 186.031 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 20 | [200,300) | [4.25,4.5) | 26813 | 0.3379 | 1.7472 | 133.595 | 6280 | core_extrapolation_warning, surface_fit_not_successful |
| 21 | [200,300) | [4.5,4.75) | 9558 | 0.3827 | 1.904 | 85.5182 | 4928 | core_extrapolation_warning, surface_fit_not_successful |
| 22 | [200,300) | [4.75,5.0) | 3458 | 0.533 | 2.1119 | 18.0918 | 4928 | - |
| 23 | [200,300) | [5,6) | 1980 | 0.4811 | 2.2744 | 20.8982 | 4928 | - |
| 24 | [200,300) | >=6 | 2 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 25 | [300,500) | [2,2.5) | 1038 | 0.1111 | 1.58 | 6.05197 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 26 | [300,500) | [2.5,3) | 688852 | 0.003664 | 0.42141 | 17.587 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 27 | [300,500) | [3,3.25) | 1191475 | 0.004122 | 0.39609 | 30.8838 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 28 | [300,500) | [3.25,3.5) | 782322 | 0.005846 | 0.41562 | 19.3191 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 29 | [300,500) | [3.5,3.75) | 313446 | 0.007583 | 0.64775 | 20.3322 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 30 | [300,500) | [3.75,4.0) | 133652 | 0.03539 | 0.99007 | 70.967 | 4400 | core_extrapolation_warning, surface_fit_not_successful |
| 31 | [300,500) | [4.0,4.25) | 52626 | 0.1168 | 1.2538 | 84.8175 | 5032 | core_extrapolation_warning, surface_fit_not_successful |
| 32 | [300,500) | [4.25,4.5) | 20712 | 0.2143 | 1.4459 | 69.9122 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 33 | [300,500) | [4.5,4.75) | 8442 | 0.308 | 1.6468 | 29.4624 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 34 | [300,500) | [4.75,5.0) | 3006 | 0.4199 | 1.8282 | 23.2205 | 6280 | core_extrapolation_warning, surface_fit_not_successful |
| 35 | [300,500) | [5,6) | 1579 | 0.3976 | 2.0855 | 24.628 | 4928 | core_extrapolation_warning, surface_fit_not_successful |
| 36 | [300,500) | >=6 | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 37 | [500,800) | [2,2.5) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 38 | [500,800) | [2.5,3) | 4294 | 0.0717 | 1.58 | 13.0966 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 39 | [500,800) | [3,3.25) | 166432 | 0.09694 | 1.58 | 306.942 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 40 | [500,800) | [3.25,3.5) | 477133 | 0.002346 | 0.31075 | 1.84259 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 41 | [500,800) | [3.5,3.75) | 280350 | 0.002757 | 0.32525 | 1.06174 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 42 | [500,800) | [3.75,4.0) | 90323 | 0.005421 | 0.53313 | 2.90292 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 43 | [500,800) | [4.0,4.25) | 26383 | 0.01945 | 0.97408 | 8.59865 | 4400 | core_extrapolation_warning, surface_fit_not_successful |
| 44 | [500,800) | [4.25,4.5) | 6543 | 0.08215 | 1.1833 | 10.4654 | 5032 | core_extrapolation_warning, surface_fit_not_successful |
| 45 | [500,800) | [4.5,4.75) | 2484 | 0.1971 | 1.3994 | 6.41824 | 5032 | core_extrapolation_warning, surface_fit_not_successful |
| 46 | [500,800) | [4.75,5.0) | 1025 | 0.2537 | 1.5491 | 1.90781 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 47 | [500,800) | [5,6) | 568 | 0.2558 | 1.8612 | 2.5502 | 6280 | core_extrapolation_warning, surface_fit_not_successful |
| 48 | [500,800) | >=6 | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 49 | [800,1100) | [2,2.5) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 50 | [800,1100) | [2.5,3) | 4 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 51 | [800,1100) | [3,3.25) | 2366 | 0.08333 | 1.58 | 17.8728 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 52 | [800,1100) | [3.25,3.5) | 52986 | 0.08837 | 1.58 | 146.972 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 53 | [800,1100) | [3.5,3.75) | 122657 | 0.002228 | 0.26223 | 0.349986 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 54 | [800,1100) | [3.75,4.0) | 74431 | 0.002997 | 0.29693 | 0.385208 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 55 | [800,1100) | [4.0,4.25) | 23556 | 0.006103 | 0.5141 | 1.93899 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 56 | [800,1100) | [4.25,4.5) | 5522 | 0.008969 | 0.98619 | 4.18572 | 4400 | core_extrapolation_warning, surface_fit_not_successful |
| 57 | [800,1100) | [4.5,4.75) | 918 | 0.02564 | 1.2345 | 2.94476 | 5032 | core_extrapolation_warning, surface_fit_not_successful |
| 58 | [800,1100) | [4.75,5.0) | 150 | 0 | 1.4706 | 0.883882 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 59 | [800,1100) | [5,6) | 36 | 0 | 1.7528 | 0.723617 | 6280 | core_extrapolation_warning, surface_fit_not_successful |
| 60 | [800,1100) | >=6 | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 61 | [1100,2000) | [2,2.5) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 62 | [1100,2000) | [2.5,3) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 63 | [1100,2000) | [3,3.25) | 11 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 64 | [1100,2000) | [3.25,3.5) | 1544 | 0.04938 | 1.58 | 3.33234 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 65 | [1100,2000) | [3.5,3.75) | 21507 | 0.0898 | 1.58 | 59.4065 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 66 | [1100,2000) | [3.75,4.0) | 51102 | 0.002685 | 0.21723 | 0 | 3760 | non_positive_B_on, core_extrapolation_warning, surface_fit_not_successful |
| 67 | [1100,2000) | [4.0,4.25) | 46341 | 0.001602 | 0.2324 | 0.0720973 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 68 | [1100,2000) | [4.25,4.5) | 25304 | 0.001436 | 0.29243 | 0.0198872 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 69 | [1100,2000) | [4.5,4.75) | 10148 | 0.004024 | 0.55869 | 0.264388 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 70 | [1100,2000) | [4.75,5.0) | 3881 | 0.1068 | 1.58 | 23.6288 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 71 | [1100,2000) | [5,6) | 1640 | 0.1194 | 1.58 | 0.61596 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 72 | [1100,2000) | >=6 | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 73 | [2000,3000) | [2,2.5) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 74 | [2000,3000) | [2.5,3) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 75 | [2000,3000) | [3,3.25) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 76 | [2000,3000) | [3.25,3.5) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 77 | [2000,3000) | [3.5,3.75) | 0 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
| 78 | [2000,3000) | [3.75,4.0) | 9 | 0 | 1.58 | 0.790761 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 79 | [2000,3000) | [4.0,4.25) | 310 | 0.0625 | 1.58 | 2.18601 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 80 | [2000,3000) | [4.25,4.5) | 1492 | 0.08046 | 1.58 | 6.27023 | 5648 | core_extrapolation_warning, surface_fit_not_successful |
| 81 | [2000,3000) | [4.5,4.75) | 5442 | 0.003311 | 0.20239 | 0 | 3760 | non_positive_B_on, core_extrapolation_warning, surface_fit_not_successful |
| 82 | [2000,3000) | [4.75,5.0) | 11221 | 0.0064 | 0.25599 | 0 | 3760 | non_positive_B_on, core_extrapolation_warning, surface_fit_not_successful |
| 83 | [2000,3000) | [5,6) | 20659 | 0.004545 | 0.30336 | 0.048218 | 3760 | core_extrapolation_warning, surface_fit_not_successful |
| 84 | [2000,3000) | >=6 | 16 | 0 | 1.58 | 0 | 5648 | non_positive_B_on, no_fiducial_events, core_extrapolation_warning, surface_fit_not_successful |
