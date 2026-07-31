#!/usr/bin/env python3
"""Consolidate common-GTI live-time, provenance, and analysis-scope checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-gti-manifest", type=Path, required=True)
    parser.add_argument("--common-gti-tsv", type=Path, required=True)
    parser.add_argument("--pass5-map-livetime-csv", type=Path, required=True)
    parser.add_argument("--pass5-hour-audit-csv", type=Path, required=True)
    parser.add_argument("--pass5-provenance-json", type=Path, required=True)
    parser.add_argument("--pass5-yaml", type=Path, required=True)
    parser.add_argument("--v6-json", type=Path, required=True)
    parser.add_argument("--v6-response-metadata", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def chunk_label(hour: str) -> str:
    date, hour_text = hour.split("_")
    return f"{date}_{int(hour_text) // 4}"


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.common_gti_manifest.read_text(encoding="utf-8"))
    provenance = json.loads(args.pass5_provenance_json.read_text(encoding="utf-8"))
    pass5 = yaml.safe_load(args.pass5_yaml.read_text(encoding="utf-8"))
    v6 = json.loads(args.v6_json.read_text(encoding="utf-8"))
    response = json.loads(args.v6_response_metadata.read_text(encoding="utf-8"))

    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"duration_seconds": 0.0, "interval_count": 0, "hours": set()}
    )
    all_hours: set[str] = set()
    interval_count = 0
    duration_sum = 0.0
    with args.common_gti_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            duration = float(row["duration_seconds"])
            label = chunk_label(row["hour"])
            grouped[label]["duration_seconds"] = (
                float(grouped[label]["duration_seconds"]) + duration
            )
            grouped[label]["interval_count"] = int(grouped[label]["interval_count"]) + 1
            grouped[label]["hours"].add(row["hour"])
            all_hours.add(row["hour"])
            interval_count += 1
            duration_sum += duration

    map_rows: dict[str, dict[str, str]] = {}
    with args.pass5_map_livetime_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] in map_rows:
                raise ValueError(f"duplicate Pass5 map live-time row: {row['label']}")
            map_rows[row["label"]] = row
    if set(grouped) != set(map_rows):
        raise ValueError("common-GTI chunk labels differ from Pass5 map labels")

    chunk_rows: list[dict[str, object]] = []
    for label in sorted(grouped):
        v6_seconds = float(grouped[label]["duration_seconds"])
        pass5_seconds = float(map_rows[label]["pass5_ltime_seconds"])
        delta = v6_seconds - pass5_seconds
        chunk_rows.append(
            {
                "label": label,
                "accepted_hour_count": len(grouped[label]["hours"]),
                "common_gti_interval_count": int(grouped[label]["interval_count"]),
                "v6_gti_seconds": v6_seconds,
                "pass5_header_seconds": pass5_seconds,
                "v6_minus_pass5_seconds": delta,
                "relative_to_v6_percent": 100.0 * delta / v6_seconds,
                "abs_delta_le_0p2_seconds": abs(delta) <= 0.2,
                "large_delta_gt_10_seconds": delta > 10.0,
            }
        )

    hour_rows: dict[str, dict[str, str]] = {}
    with args.pass5_hour_audit_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hour_rows[row["hour"]] = row
    missing_hour_logs = sorted(all_hours - set(hour_rows))
    if missing_hour_logs:
        raise ValueError(f"{len(missing_hour_logs)} accepted hours have no hourly audit log")
    zeroed_hours = [
        hour
        for hour in sorted(all_hours)
        if float(hour_rows[hour]["histogram_counts_after"]) == 0.0
    ]
    zeroed_duration = sum(float(hour_rows[hour]["gti_duration_seconds"]) for hour in zeroed_hours)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chunk_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(chunk_rows)

    deltas = [float(row["v6_minus_pass5_seconds"]) for row in chunk_rows]
    large_rows = [row for row in chunk_rows if bool(row["large_delta_gt_10_seconds"])]
    v6_seconds = sum(float(row["v6_gti_seconds"]) for row in chunk_rows)
    pass5_chunk_seconds = sum(float(row["pass5_header_seconds"]) for row in chunk_rows)
    pass5_merged_days = float(provenance["data_config"]["live_time_days"])
    pass5_merged_seconds = pass5_merged_days * 86400.0
    delta_seconds = v6_seconds - pass5_merged_seconds

    pass5_source = pass5["source_dict"]["J0534+2200"]
    pass5_representative_loge = [
        float(value) for value in pass5_source["each_bin"]["real_E"]
    ]
    v6_cells = (v6.get("cells") or [])
    v6_nhit_bins = sorted({str(row["nhit_bin"]) for row in v6_cells})
    v6_prede_bins = sorted({str(row["predE_bin"]) for row in v6_cells})
    true_edges = [float(value) for value in response["logE_true_edges"]]

    audit = {
        "sample_contract": {
            "accepted_jobs": int(manifest["accepted_job_count"]),
            "accepted_hours": len(all_hours),
            "common_gti_intervals": interval_count,
            "common_gti_tsv_duration_seconds": duration_sum,
            "manifest_duration_seconds": float(manifest["common_gti_live_time_seconds"]),
            "duration_sum_matches_manifest": math.isclose(
                duration_sum,
                float(manifest["common_gti_live_time_seconds"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "accepted_maps_line_count": int(
                manifest["accepted_maps_list_current_line_count"]
            ),
            "accepted_maps_unique_count": int(manifest["accepted_maps_list_unique_count"]),
            "accepted_maps_matches": bool(
                manifest["accepted_maps_list_matches_strict_recovery_manifest"]
            ),
            "eos_nonempty_j2000_count": int(
                provenance["eos_map_set"]["nonempty_j2000_count"]
            ),
            "eos_matches_accepted_maps": bool(
                provenance["eos_map_set"]["matches_accepted_maps_list"]
            ),
            "merged_map_input_count": int(provenance["merge"]["input_map_line_count"]),
            "merged_map_terminal_log_matches": bool(
                provenance["merge"]["terminal_line_matches"]
            ),
        },
        "live_time_comparison": {
            "v6_gti_seconds": v6_seconds,
            "v6_gti_days": v6_seconds / 86400.0,
            "pass5_merged_header_seconds": pass5_merged_seconds,
            "pass5_merged_header_days": pass5_merged_days,
            "pass5_chunk_header_sum_seconds": pass5_chunk_seconds,
            "pass5_chunk_header_sum_days": pass5_chunk_seconds / 86400.0,
            "chunk_sum_minus_merged_header_seconds": (
                pass5_chunk_seconds - pass5_merged_seconds
            ),
            "v6_minus_pass5_seconds": delta_seconds,
            "relative_difference_percent_of_v6": 100.0 * delta_seconds / v6_seconds,
            "chunk_count": len(chunk_rows),
            "chunks_abs_delta_le_0p2_seconds": sum(abs(delta) <= 0.2 for delta in deltas),
            "chunks_delta_gt_10_seconds": len(large_rows),
            "large_delta_seconds_sum": sum(
                float(row["v6_minus_pass5_seconds"]) for row in large_rows
            ),
            "large_delta_share_percent": 100.0
            * sum(float(row["v6_minus_pass5_seconds"]) for row in large_rows)
            / delta_seconds,
            "median_chunk_delta_seconds": statistics.median(deltas),
            "mean_chunk_delta_seconds": statistics.mean(deltas),
            "accepted_hours_with_zero_pass5_mask_histogram": len(zeroed_hours),
            "zeroed_hour_v6_gti_seconds": zeroed_duration,
            "zeroed_hours": zeroed_hours,
            "interpretation": (
                "The pipelines share the same nominal accepted-hour/common-GTI selection, "
                "but they do not have identical effective live time. Pass5 Ltime counts "
                "official-event/DI time occupancy after GTI masking, whereas v6 sums "
                "continuous recovered-time GTI endpoints. Use 'common-GTI selected sample' "
                f"with the {delta_seconds:.3f} s exposure caveat; do not claim strict second-for-second "
                "identity."
            ),
        },
        "pass5_provenance": provenance,
        "analysis_scope": {
            "v6": {
                "selected_cell_count": len(v6_cells),
                "nhit_range": "100 <= Nhit < 3000",
                "nhit_bins": v6_nhit_bins,
                "predE_coordinate": "log10(E_pred / GeV)",
                "selected_predE_bins": v6_prede_bins,
                "selected_predE_envelope_tev": [0.1, 10**2.5],
                "true_energy_response_log10_gev": [min(true_edges), max(true_edges)],
                "true_energy_response_tev": [10 ** (min(true_edges) - 3), 10 ** (max(true_edges) - 3)],
                "objective": "chi2 on 44 Stage-E excess cells",
                "sigma_model": "sqrt(N_on + B_on)",
                "chi2": float(v6["fits"]["logpar_conservative"]["chi2"]),
                "ndof": int(v6["fits"]["logpar_conservative"]["ndof"]),
            },
            "pass5": {
                "analysis_bin_count": 7,
                "nhit_range": "30 <= Nhit < 2000",
                "nhit_edges": [30, 60, 100, 200, 300, 500, 800, 2000],
                "event_level_energy_cut": None,
                "representative_log10_energy_tev": pass5_representative_loge,
                "representative_energy_tev": [10**value for value in pass5_representative_loge],
                "objective": "Poisson likelihood on the seven-bin spatial cube",
                "free_parameter_count": len(
                    pass5["output_option"]["gtlike"]["covariance_parameter_names"]
                ),
            },
            "comparison_energy_range_tev": [1.0, 100.0],
            "comparison_classification": "full_pipeline_comparison",
            "not_an_isolated_predE_gain_test": (
                "Cuts, Nhit coverage and edges, 2D cell selection, background/PSF/IRF "
                "construction, objective, and nuisance treatment differ. An isolated predE "
                "ablation must hold those choices fixed inside the v6 pipeline."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
