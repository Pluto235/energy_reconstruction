#!/bin/bash
#SBATCH --job-name=energy_env_check
#SBATCH --partition=main
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/mnt/newdisk/shijie/energy/apply
#SBATCH --output=/mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH --error=/mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err

set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/newdisk/shijie/conda/envs/py310
export PYTHONPATH=/mnt/newdisk/shijie/energy
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m pip check
nvidia-smi --query-gpu=name,driver_version --format=csv
python -u - <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
from apply.apply_observation_energy import build_model, load_run_config

assert sys.version_info[:2] == (3, 10), sys.version
packages = {
    "torch": "torch", "numpy": "numpy", "scikit-learn": "sklearn",
    "uproot": "uproot", "awkward": "awkward", "matplotlib": "matplotlib",
    "pandas": "pandas", "tqdm": "tqdm", "psutil": "psutil",
    "pyarrow": "pyarrow", "iminuit": "iminuit",
}
for module in packages.values():
    importlib.import_module(module)
assert torch.cuda.is_available(), "CUDA unavailable in Slurm allocation"
assert torch.cuda.device_count() == 2, "Expected both ETO GPUs in allocation"

run_dir = Path("/mnt/newdisk/shijie/energy/runs/theta_recoxy_position_embed_noemax_no_core_cut_64748")
paths = [run_dir / "config.json", run_dir / "checkpoints/best_model.pt"]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
config = load_run_config(str(run_dir))
gpus = []
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    x = torch.ones((32, 32), device=device)
    assert torch.all(x @ x == 32).item(), f"CUDA arithmetic failed on {device}"
    torch.cuda.synchronize(device)
    model = build_model(config, str(paths[1]), device)
    assert not model.training
    assert all(p.device == device and torch.isfinite(p).all().item() for p in model.parameters())
    gpus.append({"device": str(device), "name": torch.cuda.get_device_name(index),
                 "capability": torch.cuda.get_device_capability(index), "model_loaded": True})
    del model
assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
print(json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"], "python": sys.executable,
    "python_version": sys.version, "packages": {p: importlib.metadata.version(p) for p in packages},
    "cuda_runtime": torch.version.cuda, "gpus": gpus, "torch_arches": torch.cuda.get_arch_list(),
    "visible_gpu_count": torch.cuda.device_count(),
    "config_and_checkpoint_sha256": before,
    "result": "PASS: imports, CUDA arithmetic, existing config/strict weight loader; no inference",
}, indent=2))
PY
