#!/usr/bin/env python
from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]

PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
STAGE_C_DIR = REPO_ROOT / "apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024"
RESPONSE_NPZ = REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz"
RESPONSE_META = REPO_ROOT / "apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned_metadata.json"
SIGNAL_DIR = REPO_ROOT / "apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow"
SIGNAL_NPZ = SIGNAL_DIR / "signal_v4_containment1_annnorm.npz"
SIGNAL_META = SIGNAL_DIR / "signal_v4_containment1_annnorm_metadata.json"
STAGE_F_OUT = REPO_ROOT / "apply/output/stage_f_v5_cell_selection"
STAGE_G_OUT = REPO_ROOT / "apply/output/stage_g_v5_cell_selection"

SELECTORS: Dict[str, Path] = {
    "strict20": REPO_ROOT / "apply/config/cell_selector_v5_cellsel_strict20.csv",
    "baseline26": REPO_ROOT / "apply/config/cell_selector_v5_cellsel_baseline26.csv",
    "loose36": REPO_ROOT / "apply/config/cell_selector_v5_cellsel_loose36.csv",
}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def included_cell_ids(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [int(row["cell_id"]) for row in csv.DictReader(handle) if truthy(row.get("include"))]


def all_cell_ids(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [int(row["cell_id"]) for row in csv.DictReader(handle)]


def check_inputs(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(str(path) for path in missing))


def run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    check_inputs([STAGE_C_DIR, RESPONSE_NPZ, RESPONSE_META, SIGNAL_NPZ, SIGNAL_META, *SELECTORS.values()])

    for selector_name, selector_csv in SELECTORS.items():
        run_id = f"v5_cellsel_{selector_name}"
        fit_name = f"fit_{run_id}.npz"
        fit_meta_name = f"fit_{run_id}_metadata.json"
        sed_name = f"sed_points_{run_id}.npz"
        sed_meta_name = f"sed_points_{run_id}_metadata.json"
        stage_f_run_dir = STAGE_F_OUT / "runs" / run_id

        included = included_cell_ids(selector_csv)
        excluded = [cell_id for cell_id in all_cell_ids(selector_csv) if cell_id not in set(included)]
        required_text = ",".join(str(cell_id) for cell_id in included)
        excluded_text = ",".join(str(cell_id) for cell_id in excluded)

        run(
            [
                PYTHON,
                "apply/stages/06_fit.py",
                "--response-npz",
                str(RESPONSE_NPZ),
                "--response-metadata",
                str(RESPONSE_META),
                "--signal-npz",
                str(SIGNAL_NPZ),
                "--signal-metadata",
                str(SIGNAL_META),
                "--stage-c-dir",
                str(STAGE_C_DIR),
                "--cell-subset-csv",
                str(selector_csv),
                "--output-dir",
                str(STAGE_F_OUT),
                "--run-id",
                run_id,
                "--npz-name",
                fit_name,
                "--metadata-name",
                fit_meta_name,
                "--summary-csv-name",
                f"fit_{run_id}_summary.csv",
                "--summary-md-name",
                f"fit_{run_id}_summary.md",
                "--report-html",
                f"apply/report/stage_f_{run_id}_report.html",
                "--overwrite-run-dir",
                "--no-promote-current",
            ]
        )

        run(
            [
                PYTHON,
                "apply/stages/07_sed_points.py",
                "--response-npz",
                str(RESPONSE_NPZ),
                "--response-metadata",
                str(RESPONSE_META),
                "--signal-npz",
                str(SIGNAL_NPZ),
                "--signal-metadata",
                str(SIGNAL_META),
                "--stage-f-npz",
                str(stage_f_run_dir / fit_name),
                "--stage-f-metadata",
                str(stage_f_run_dir / fit_meta_name),
                "--output-dir",
                str(STAGE_G_OUT),
                "--run-id",
                run_id,
                "--baseline-name",
                run_id,
                "--required-cell-ids",
                required_text,
                "--excluded-cell-ids",
                excluded_text,
                "--skip-expected-stage-f-validation",
                "--npz-name",
                sed_name,
                "--metadata-name",
                sed_meta_name,
                "--summary-csv-name",
                f"sed_points_{run_id}_summary.csv",
                "--summary-json-name",
                f"sed_points_{run_id}_summary.json",
                "--summary-md-name",
                f"sed_points_{run_id}_summary.md",
                "--report-html",
                f"apply/report/stage_g_{run_id}_report.html",
                "--overwrite-run-dir",
                "--no-promote-current",
            ]
        )

    run([PYTHON, "apply/report/build_v5_cell_selection_comparison_report.py"])


if __name__ == "__main__":
    main()
