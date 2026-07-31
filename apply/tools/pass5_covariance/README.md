# Pass5 versus v6 Crab common-GTI covariance audit

This workspace compares the v6 64748 2D Nhit x predE Crab fit with the
official Pass5 Nhit-only fit. Both final 3x3 covariances are represented as
log10_phi0, alpha, beta for a natural-log LogPar model at a 3 TeV pivot.
The plotted propagation range is 1-100 TeV.

## Terminal production state

Production is complete. ETO jobs 65274, 65275, and 65276 completed with exit
code zero and must not be resubmitted. The IHEP recovery state is:

- 1078 candidate DI chunks;
- 928 accepted maps;
- 150 rejected chunks (104 acceptance failures and 46 with no central events);
- zero Step2, J2000, or other recovery work remaining;
- 928 unique entries in strict_recovery/accepted_maps.list;
- exact agreement with the 928 nonempty J2000 ROOT files on EOS.

The final common selection contains 3401 accepted hourly files and 4763 GTI
intervals. common_gti.tsv sums to 11287782.126103623 s, or
130.645626459533 d.

Do not use the original full-sample counts of 3969 hours and 8641 intervals as
metadata for the final covariance comparison. Those describe the
pre-intersection v6 sample.

## Effective live-time qualification

The merged Pass5 map header reports 130.592212689228 d. It is 4614.949754 s
shorter than the v6 GTI endpoint sum (0.040884% of the v6 exposure). This is
not a missing-map problem:

- 898 of 928 chunks agree within 0.2 seconds;
- 14 chunks account for 99.69% of the difference;
- 20 accepted hourly Pass5 logs have a zero hMJD_all_fine count after the GTI
  mask, totaling 3079.175830 s of v6 GTI duration.

The two pipelines therefore share the same nominal accepted-hour/common-GTI
selection, but not identical effective live time. Pass5 uses official-event/DI
time occupancy (bkg->EffLtime); v6 integrates recovered-time GTI interval
endpoints. Describe this as a common-GTI selected sample with an exposure
caveat, not as strict second-for-second identity.

## Rebuilding the terminal manifest

Run this only to refresh audit metadata from the existing final IHEP state; it
does not rerun production:

    python3 build_common_gti_manifest.py \
      --run-dir /home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance

The builder requires recovery counts to be terminal, enforces unique map URIs,
and compares accepted_maps.list with every accepted recovery record. It
writes:

    common_gti/common_gti_manifest.json
    common_gti/common_gti.tsv
    common_gti/included_source_file_ids.txt

included_source_file_ids.txt contains 3401 unique parent hourly-file IDs; the
manifest separately records 4763 expanded GTI source rows.

## Read-only live-time and provenance audits

On IHEP, ROOT can scan every accepted map header:

    root -l -b -q \
      'audit_pass5_map_livetime.C("strict_recovery/accepted_maps.list","strict_recovery/pass5_map_livetime_audit.csv")'

Completed hourly logs are parsed with:

    python3 audit_pass5_hour_logs.py \
      --log-dir /home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance \
      --output-csv strict_recovery/pass5_hour_gti_mask_audit.csv

File hashes, merge logs, EOS map membership, and the Pass5 fit chain are
captured with build_pass5_provenance.py. The audit found that data_config.yaml
and covariance_fit.yaml retain stale common_gti_fit_interactive/ paths although
the actual products are under common_gti_fit/. File hashes and monotonic
timestamps are recorded, but the embedded path provenance is not fully
self-contained; do not rewrite the original fit YAML to conceal this.

build_common_gti_audit.py consolidates the sample, live-time, provenance,
energy/binning, and objective checks into CSV and JSON inputs for the report.

## Covariance and scientific scope

The report generator fails unless both covariances are finite symmetric
positive-definite 3x3 matrices, their diagonal errors match the reported
parameter errors, their order is log10_phi0, alpha, beta, and both pivots are
3 TeV.

The comparison is a full-pipeline comparison:

- v6 uses 44 selected 2D cells, 100 <= Nhit < 3000, a predE envelope of
  0.1-316.23 TeV, and conservative chi-square on Stage-E excess counts;
- Pass5 uses seven Nhit-only bins, 30 <= Nhit < 2000, no event-level
  reconstructed-energy cut, and a Poisson spatial likelihood;
- background, PSF/IRF, binning, nuisance handling, and objective all differ.

An isolated predE gain test must hold those choices fixed within one pipeline.
The current comparison cannot attribute a covariance difference to predE.

The v6 LogPar fit has chi2/ndof = 20.943334. Its raw HESSE joint error-volume
ratio relative to Pass5 is 0.528, while applying the Birge/PDG scale factor to
v6 gives 50.617. Neither number proves that v6 has smaller total uncertainty,
and systematics and cross-method covariance are absent.
