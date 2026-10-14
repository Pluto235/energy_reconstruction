#!/usr/bin/env python3
"""Read-only acceptance/source gate for the frozen January 64748 analysis."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TAG = "v6_64748_202201_scheme_R_double_rayleigh_analytic_bon"
BASE = "v6_64748_nhit100_reselect44_split56_miss030"
SELECTOR_SHA = "c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6"
TARGET = 0.7129790300890827


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path):
    info = Path(path).stat()
    return [info.st_ino, info.st_size, info.st_mtime_ns]


def check_stat(path, expected):
    if file_stat(path) != expected:
        raise ValueError(f"Source changed: {path}")


def check_inventory(actual, expected, count=681):
    if len(actual) != count or len(set(actual)) != count or set(actual) != set(expected):
        raise ValueError("January inventory mismatch: missing/extra/duplicate files")
    if any(not Path(p).name.startswith("Esg202201") for p in actual):
        raise ValueError("Non-January input")


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.perf_counter()
    import uproot

    subprocess.run(["git", "merge-base", "--is-ancestor", "f979c5b", "HEAD"], cwd=root, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    alignment_path = root / "apply/docs/PREDICTION_TIME_ALIGNMENT_ACCEPTANCE.json"
    alignment = load_json(alignment_path)
    if not alignment["ok"] or alignment["passed_files"] != 681 or alignment["prediction_entries"] != 48726074:
        raise ValueError("Time-alignment acceptance failed")
    hashes = {}
    for name, expected in alignment["detailed_reports"].items():
        actual = sha256(name)
        if actual != expected:
            raise ValueError(f"Acceptance report hash changed: {name}")
        hashes[name] = actual
    verification = load_json(next(p for p in hashes if p.endswith("verify_66080.json")))
    if not verification["ok"] or verification["config"]["prediction_root"] != "/mnt/newdisk/shijie/WCDA_obs_eval_64748":
        raise ValueError("Wrong accepted prediction source")
    prediction_root = Path(verification["config"]["prediction_root"])
    aligned_root = Path(alignment["output_root"])
    if aligned_root != Path("/mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned"):
        raise ValueError("Wrong accepted aligned time source")
    inference_path = root / ".recovery/2026-10-14/inference-64748/month-acceptance.json"
    inference = load_json(inference_path)
    if inference["result"] != "PASS" or inference["events_written"] != 48726074:
        raise ValueError("Inference acceptance failed")
    for name, expected in inference["provenance"].items():
        actual = sha256(name)
        if actual != expected:
            raise ValueError(f"Inference provenance changed: {name}")
        hashes[name] = actual
    expected_files = [r["relative_path"] for r in verification["results"]]
    actual_files = [str(p.relative_to(prediction_root)) for p in prediction_root.glob("[0-9][0-9][0-9][0-9]/Esg*.root") if not p.name.endswith(".time.root")]
    check_inventory(actual_files, expected_files)
    aligned_files = [str(p.relative_to(aligned_root)).replace(".time.root", ".root") for p in aligned_root.glob("[0-9][0-9][0-9][0-9]/Esg*.time.root")]
    check_inventory(aligned_files, expected_files)
    inference_counts = {r["file"]: r["events"] for r in inference["files"]}
    inventory = []
    for index, record in enumerate(verification["results"]):
        relative = record["relative_path"]
        if not record["ok"] or record["prediction_entries"] != inference_counts[relative]:
            raise ValueError(f"Acceptance event count mismatch: {relative}")
        for name, binding in record["sources"].items():
            check_stat(name, binding["stat"])
            with uproot.open(name) as source:
                if str(source.file.uuid) != binding["uuid"]:
                    raise ValueError(f"ROOT UUID changed: {name}")
        for name, binding in record["sidecars"].items():
            check_stat(name, binding["stat"])
            if sha256(name) != binding["sha256"]:
                raise ValueError(f"Sidecar changed: {name}")
        path = aligned_root / Path(relative).with_suffix(".time.root")
        receipt_path = path.with_suffix(".receipt.json")
        receipt = load_json(receipt_path)
        if receipt["verification"] != "all-fields-bitwise+unique-event-identity" or receipt["provenance"] != {k: v for k, v in record.items() if k != "ok"}:
            raise ValueError(f"Aligned source receipt mismatch: {relative}")
        digest = sha256(path)
        if digest != receipt["output_sha256"]:
            raise ValueError(f"Aligned output hash mismatch: {path}")
        with uproot.open(path) as source:
            if source["t_recovered_time"].num_entries != record["prediction_entries"]:
                raise ValueError(f"Aligned row mismatch: {path}")
        inventory.append({"relative_path": relative, "prediction_entries": record["prediction_entries"], "original_entries": record["original_entries"], "prediction_binding": record["sources"][str(prediction_root / relative)], "aligned_sha256": digest, "receipt_sha256": sha256(receipt_path)})
        if index % 50 == 0:
            print(f"source gate {index + 1}/681", flush=True)
    selector = root / f"apply/config/cell_selector_{BASE}_fit.csv"
    if sha256(selector) != SELECTOR_SHA:
        raise ValueError("Frozen selector hash mismatch")
    with selector.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = [int(r["cell_id"]) for r in rows if r["include"] == "1"]
    if len(selected) != 44 or selected != sorted(set(selected)):
        raise ValueError("Frozen 44-cell order mismatch")
    code = list((root / "apply/stages").glob("0[3-7]_*.py")) + [Path(__file__).resolve()]
    frozen = [selector, root / "apply/config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv", alignment_path, inference_path, root / "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_analytic_bon_report.html"]
    hashes.update({str(p): sha256(p) for p in code + frozen})
    result = {"result": "PASS", "run_tag": TAG, "time_label": "202201", "input_commit": commit, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "elapsed_seconds": time.perf_counter() - start, "prediction_root": str(prediction_root), "aligned_time_root": str(aligned_root), "original_time_root": verification["config"]["time_root"], "files": inventory, "file_count": 681, "prediction_entries": sum(r["prediction_entries"] for r in inventory), "original_entries": alignment["original_entries"], "original_status": alignment["original_status"], "selected_cell_ids": selected, "hashes": hashes, "contract": {"model": "64748", "nhit_branch": "nv", "match_status": 0, "cuts": {"pincness_lt": 1.1, "fitstat_eq": 0, "theta_deg_lt": 50, "dcedge_m_gt": 20}, "psf": "double_rayleigh", "target_containment": TARGET, "response": "nominal Aeff multiplied once by target_containment", "signal_containment": "fixed1", "background": "crab_roi_local annulus-quadratic order2 annulus-normalized", "aperture_integration": "analytic-quadratic", "N_on": "event spherical distance", "gti_gap_seconds": 60, "pass5_common_gti": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"January source gate PASS: {len(inventory)} files", flush=True)


if __name__ == "__main__":
    main()
