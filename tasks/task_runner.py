import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/home/server/projects/energy_reconstruction")
TASKS_DIR = PROJECT_ROOT / "tasks"
TASK_MD = TASKS_DIR / "task.md"
LOG_PATH = TASKS_DIR / "task_runner.log"
NOTEBOOK_PATH = PROJECT_ROOT / "notebook" / "eval_relaxed_cuts_fitstat0_2727.ipynb"

TASK1_SCRIPT = PROJECT_ROOT / "scripts" / "slurm" / "allcuts_compare.sbatch"
TASK2_SCRIPT = PROJECT_ROOT / "scripts" / "slurm" / "eval_relaxed_cuts_fitstat0_2727.sbatch"

TASK1_EXPECTED = lambda job_id: [
    PROJECT_ROOT / "runs" / f"allcuts_strict_{job_id}" / "checkpoints" / "best_model.pt",
    PROJECT_ROOT / "runs" / f"allcuts_relaxed_{job_id}" / "checkpoints" / "best_model.pt",
]

TASK2_EXPECTED = [
    PROJECT_ROOT / "runs" / "fitstat0_2727" / "fig_eval_dcedge20_baseline" / "metrics.json",
    PROJECT_ROOT / "runs" / "fitstat0_2727" / "fig_eval_dcedge0_relaxed" / "metrics.json",
    PROJECT_ROOT / "runs" / "fitstat0_2727" / "fig_eval_dangle3_baseline" / "metrics.json",
    PROJECT_ROOT / "runs" / "fitstat0_2727" / "fig_eval_dangle_relaxed" / "metrics.json",
    PROJECT_ROOT / "runs" / "fitstat0_2727" / "fig_eval_pinc1p1_baseline" / "metrics.json",
    PROJECT_ROOT / "runs" / "fitstat0_2727" / "fig_eval_pinc_relaxed" / "metrics.json",
]

TASK3_EXPECTED = [
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "dcedge_resolution_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "dcedge_logRMS_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "dcedge_bias_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "dangle_resolution_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "dangle_logRMS_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "dangle_bias_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "pincness_resolution_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "pincness_logRMS_weighted_compare.png",
    PROJECT_ROOT / "notebook" / "generated" / "eval_relaxed_cuts_fitstat0_2727" / "pincness_bias_weighted_compare.png",
]

POLL_SECONDS = 180


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_cmd(cmd, cwd=None, check=True):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        log(f"STDOUT ({' '.join(cmd[:3])}...): {result.stdout.strip().splitlines()[-1]}")
    if result.stderr.strip():
        log(f"STDERR ({' '.join(cmd[:3])}...): {result.stderr.strip().splitlines()[-1]}")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def submit_sbatch(script_path: Path) -> int:
    result = run_cmd(["sbatch", str(script_path)], cwd=str(PROJECT_ROOT))
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Failed to parse sbatch job id from: {result.stdout!r}")
    job_id = int(match.group(1))
    log(f"Submitted slurm job {job_id} for {script_path.name}")
    return job_id


def get_squeue_state(job_id: int):
    result = run_cmd(["squeue", "-h", "-j", str(job_id), "-o", "%T"], check=False)
    state = result.stdout.strip()
    return state or None


def get_sacct_state(job_id: int):
    result = run_cmd(
        ["sacct", "-j", str(job_id), "--format=JobIDRaw,State", "-n", "-P"],
        check=False,
    )
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[0] == str(job_id):
            return parts[1]
    return None


def paths_exist(paths):
    return all(path.exists() for path in paths)


def read_log_tail(job_name: str, job_id: int):
    out_path = PROJECT_ROOT / "logs" / "slurm" / f"{job_name}_{job_id}.out"
    err_path = PROJECT_ROOT / "logs" / "slurm" / f"{job_name}_{job_id}.err"
    for path in [out_path, err_path]:
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                for line in text:
                    log(f"{path.name}: {line}")
            except Exception as exc:
                log(f"Failed to read {path}: {exc}")


def poll_job_once(job_id: int, job_name: str, required_paths, task_name: str) -> bool:
    squeue_state = get_squeue_state(job_id)
    sacct_state = get_sacct_state(job_id)
    log(f"Polling {task_name} job {job_id}: squeue={squeue_state or 'NONE'} sacct={sacct_state or 'NONE'}")

    if sacct_state and any(flag in sacct_state for flag in ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"]):
        log(f"{task_name} failed with state={sacct_state}")
        read_log_tail(job_name, job_id)
        raise RuntimeError(f"{task_name} failed with state={sacct_state}")

    if paths_exist(required_paths) and (not squeue_state):
        log(f"{task_name} outputs detected and job left queue")
        return True

    if sacct_state and "COMPLETED" in sacct_state and paths_exist(required_paths):
        log(f"{task_name} finished with COMPLETED and required outputs present")
        return True

    return False


def wait_for_jobs(task1_job_id: int, task2_job_id: int):
    task1_done = False
    task2_done = False

    while not (task1_done and task2_done):
        if not task1_done:
            task1_done = poll_job_once(
                task1_job_id,
                "allcuts_compare",
                TASK1_EXPECTED(task1_job_id),
                "Task1",
            )
        if not task2_done:
            task2_done = poll_job_once(
                task2_job_id,
                "eval_relaxed_fitstat0_2727",
                TASK2_EXPECTED,
                "Task2",
            )

        if task1_done and task2_done:
            return

        time.sleep(POLL_SECONDS)


def execute_notebook(notebook_path: Path):
    run_cmd(
        [
            "/home/server/anaconda3/envs/py310/bin/jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook_path),
        ],
        cwd=str(PROJECT_ROOT),
    )


def build_argparser():
    parser = argparse.ArgumentParser("Task runner for energy_reconstruction/tasks/task.md")
    parser.add_argument("--task1-job-id", type=int, default=None)
    parser.add_argument("--task2-job-id", type=int, default=None)
    return parser


def main():
    args = build_argparser().parse_args()

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    log("Task runner started")
    log(f"Read task file: {TASK_MD}")
    _ = TASK_MD.read_text(encoding="utf-8")

    if args.task1_job_id is not None:
        task1_job_id = args.task1_job_id
        log(f"Resume task1 with existing slurm job {task1_job_id}")
    else:
        log("Start task1")
        task1_job_id = submit_sbatch(TASK1_SCRIPT)

    if args.task2_job_id is not None:
        task2_job_id = args.task2_job_id
        log(f"Resume task2 with existing slurm job {task2_job_id}")
    else:
        log("Start task2")
        task2_job_id = submit_sbatch(TASK2_SCRIPT)

    wait_for_jobs(task1_job_id, task2_job_id)

    log("Start task3")
    execute_notebook(NOTEBOOK_PATH)
    if not paths_exist(TASK3_EXPECTED):
        raise RuntimeError("Task3 notebook execution finished but expected figures are missing")
    log("Task3 finished")

    summary = {
        "task1_job_id": task1_job_id,
        "task2_job_id": task2_job_id,
        "task1_outputs": [str(path.parent.parent) for path in TASK1_EXPECTED(task1_job_id)],
        "task2_outputs": [str(path.parent) for path in TASK2_EXPECTED],
        "task3_notebook": str(NOTEBOOK_PATH),
        "task3_generated_dir": str(TASK3_EXPECTED[0].parent),
        "status": "COMPLETED",
    }
    summary_path = TASKS_DIR / "task_runner_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"All tasks finished. Summary written to {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Runner failed: {type(exc).__name__}: {exc}")
        raise
