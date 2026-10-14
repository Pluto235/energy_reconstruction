#!/usr/bin/env python3
"""January-only integrity gates, with scientific fit quality recorded separately."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from january_v6_inputs import BASE, TAG, TARGET, SELECTOR_SHA, load_json, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_response(nominal, fixed):
    require(np.allclose(fixed["a_eff"], nominal["a_eff"] * TARGET, rtol=2e-7, atol=0), "Containment must be applied exactly once")
    for key in ("denominator_sumw", "denominator_count", "cell_id", "logE_true_edges", "theta_true_edges_deg", "s0_m2"):
        require(np.array_equal(nominal[key], fixed[key]), f"Response contract changed: {key}")


def check_selected_cells(selected, rows):
    frozen = [int(row["cell_id"]) for row in rows if row["include"] == "1"]
    require(len(frozen) == 44 and selected == frozen, "Manifest changed frozen 44-cell order")


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def run_dir(root, stage):
    suffix = "stage_b_psf" if stage == "psf" else f"stage_{stage}"
    return root / stage / "runs" / f"{TAG}_{suffix}"


def stage_paths(root, stage):
    stem = {"psf": "psf", "d": "background", "e": "signal"}[stage]
    base = run_dir(root, stage) / f"{stem}_{TAG}"
    return base.with_suffix(".npz"), base.with_name(base.name + "_metadata.json")


def check(stage, root, repo):
    manifest = load_json(root / "analysis_manifest.json")
    selected = manifest["selected_cell_ids"]
    selector = repo / f"apply/config/cell_selector_{BASE}_fit.csv"
    require(sha256(selector) == SELECTOR_SHA, "Selector changed")
    with selector.open() as stream:
        check_selected_cells(selected, list(csv.DictReader(stream)))
    details = {"result": "PASS", "stage": stage, "job_id": os.environ.get("SLURM_JOB_ID"), "hashes": {}, "science": {}}
    def track(path):
        details["hashes"][str(path)] = sha256(path)
    if stage == "c":
        directory = run_dir(root, "c")
        metadata_path = directory / "obs_events_metadata.json"
        metadata = load_json(metadata_path)
        processing = metadata["processing"]
        require(processing["input_file_count"] == processing["processed_file_count"] == 681, "C file coverage")
        require(processing["missing_time_file_count"] == processing["entry_mismatch_file_count"] == 0, "C time pairing")
        require(metadata["obs_root"] == manifest["prediction_root"] and metadata["time_root"] == manifest["aligned_time_root"], "C wrong sources")
        cuts = metadata["cuts"]
        require(cuts["match_status_equals"] == 0 and cuts["pincness_lt"] == 1.1 and cuts["fitstat_equals"] == 0 and cuts["theta_deg_lt"] == 50 and cuts["dcedge_gt_m"] == 20, "C frozen cuts")
        require(not metadata["quality_flags"]["allow_missing_time"] and not metadata["quality_flags"]["allow_entry_mismatch"], "C permissive pairing")
        ledger = repo / "apply/config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv"
        require(Path(metadata["cell_selection_csv"]).resolve() == ledger.resolve() and sha256(ledger) == manifest["hashes"][str(ledger)], "C frozen ledger")
        with ledger.open() as stream:
            frozen_cells = sorted(csv.DictReader(stream), key=lambda row: int(row["cell_id"]))
        actual_cells = [{key: row[key] for key in ("cell_id", "nhit_bin", "predE_bin")} for row in metadata["cells"]]
        require(actual_cells == [{"cell_id": int(row["cell_id"]), "nhit_bin": row["nhit_bin"], "predE_bin": row["predE_bin"]} for row in frozen_cells], "C bin definitions/order")
        flow = list(metadata["cutflow"].values())
        require(all(a >= b >= 0 for a, b in zip(flow, flow[1:])), "C nonmonotonic cutflow")
        require(metadata["cutflow"]["after_finite"] - metadata["assignment_audit"]["out_of_ledger_after_finite"] == processing["selected_rows"], "C cutflow reconciliation")
        require(metadata["cutflow"]["input_entries"] == 48726074, "C input row count")
        require(list(metadata["months"]) == ["202201"], "C month range")
        with (directory / "source_files.csv").open() as stream:
            files = list(csv.DictReader(stream))
        expected = {r["relative_path"]: r["prediction_entries"] for r in manifest["files"]}
        require(set(r["relative_path"] for r in files) == set(expected), "C inventory")
        require(all(int(r["event_entries"]) == int(r["time_entries"]) == expected[r["relative_path"]] and r["status"] == "processed" for r in files), "C paired entries")
        import pyarrow.dataset as ds
        require(ds.dataset(directory / "obs_events", format="parquet", partitioning="hive").count_rows() == processing["selected_rows"], "C parquet count")
        details.update(cutflow=metadata["cutflow"], assignment_audit=metadata["assignment_audit"], roi=metadata["roi_coverage"], elapsed_seconds=processing["elapsed_seconds"])
        track(metadata_path)
        track(directory / "source_files.csv")
        track(directory / "obs_events_manifest.json")
    else:
        path, metadata_path = stage_paths(root, stage)
        data, metadata = npz(path), load_json(metadata_path)
        ids = data["cell_id"].astype(int).tolist()
        require(ids == selected or (stage in ("psf", "d", "e") and ids == list(range(1, 92))), f"{stage} fixed cell order")
        require(metadata.get("promotion", {}).get("status") not in ("promoted", "promoted_physical", "promoted_diagnostic"), "Unexpected promotion")
        track(path)
        track(metadata_path)
        details["elapsed_seconds"] = metadata.get("elapsed_seconds")
        if stage == "psf":
            nominal_path = repo / "apply/output/stage_a_v6_64748_nhit100_highEplus1_split56/response_2d_v6_64748_nhit100_highEplus1_split56.npz"
            fixed_path = repo / f"apply/output/stage_e_{BASE}_fixed712979_rayleigh_annnorm/response_{BASE}_fixed712979_rayleigh.npz"
            nominal, fixed = npz(nominal_path), npz(fixed_path)
            check_response(nominal, fixed)
            require(np.array_equal(data["cell_id"], nominal["cell_id"]), "PSF response cell order")
            for key in ("nhit_bin", "predE_bin"):
                require(np.array_equal(data[key], nominal[key]), f"PSF bin mismatch: {key}")
            historical_path = repo / f".recovery/2026-10-14/astrocluster/energy_reconstruction/apply/output/stage_a_{BASE}_double_rayleigh_aperture_conditioned/response_2d_{BASE}_double_rayleigh_aperture_conditioned.npz"
            historical = npz(historical_path)
            require(np.array_equal(data["cell_id"], historical["cell_id"]), "Historical PSF cell order")
            require(np.allclose(data["r_opt_deg"], historical["aperture_r_opt_deg"], rtol=2e-7, atol=1e-7), "Rebuilt PSF changed frozen aperture")
            details["max_radius_difference_deg"] = float(np.max(np.abs(data["r_opt_deg"] - historical["aperture_r_opt_deg"])))
            details["science"]["warnings"] = metadata.get("warning_rows")
            for source in (nominal_path, fixed_path, historical_path):
                track(source)
        elif stage == "d":
            model = metadata["background_model"]
            require(model["on_aperture_integration"] == "analytic-quadratic" and model["fit_statistic"] == "weighted-ls" and model["surface_order"] == 2 and model["annulus_normalize_surface"], "D frozen background")
            require(model["analytic_required_cell_ids"] == selected and not (set(selected) & set(model["excluded_fallback_cell_ids"])), "D fallback on required cell")
            coeff, radius, scale = data["surface_coefficients"], data["r_opt_deg"], data["annulus_surface_scale"]
            step = float(data["on_aperture_grid_step_deg"].ravel()[0])
            analytic = scale / step**2 * (math.pi * radius**2 * coeff[:, 0] + math.pi * radius**4 * (coeff[:, 3] + coeff[:, 5]) / 4)
            active = np.isin(data["cell_id"], selected)
            require(np.allclose(data["B_on"][active], analytic[active], rtol=5e-12, atol=1e-8), "D analytic disk formula")
            require(np.all(np.isfinite(data["B_on"][active])) and np.all(data["B_on"][active] > 0), "D invalid B_on")
            require(np.all(data["on_aperture_surface_min"][active] > 0), "D nonpositive aperture surface")
            require(np.all(data["annulus_counts"][active] > 0), "D empty annulus")
            details["science"] = {"min_annulus_counts": float(np.min(data["annulus_counts"])), "fit_chi2": data.get("surface_fit_chi2", np.array([])).tolist()}
        elif stage == "e":
            background = npz(stage_paths(root, "d")[0])
            require(np.array_equal(data["B_on"], background["B_on"]), "E changed B_on")
            require(np.array_equal(data["containment_r_opt"], np.ones(len(ids))), "E containment must be fixed1")
            require(np.allclose(data["excess"], data["N_on"] - data["B_on"], rtol=0, atol=1e-10), "E excess arithmetic")
            require(np.allclose(data["excess_err_conservative"], np.sqrt(data["N_on"] + data["B_on"])), "E uncertainty")
            details["science"] = {"total_n_on": int(data["N_on"].sum()), "total_b_on": float(data["B_on"].sum()), "total_excess": float(data["excess"].sum()), "empty_cells": [ids[i] for i in np.flatnonzero(data["N_on"] == 0)], "negative_excess_cells": [ids[i] for i in np.flatnonzero(data["excess"] < 0)], "low_n_on_lt10_cells": [ids[i] for i in np.flatnonzero(data["N_on"] < 10)], "quality": metadata["quality_gate"]}
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["c", "psf", "d", "e"], required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "output" / TAG)
    args = parser.parse_args()
    result = check(args.stage, args.root.resolve(), Path(__file__).resolve().parents[2])
    directory = args.root / "acceptance"
    directory.mkdir(exist_ok=True)
    path = directory / f"{args.stage}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    require(not path.exists(), f"Receipt exists: {path}")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"January {args.stage} integrity PASS: {path}", flush=True)


if __name__ == "__main__":
    main()
