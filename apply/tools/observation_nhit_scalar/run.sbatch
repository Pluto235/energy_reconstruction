#!/bin/bash
#SBATCH --job-name=january-nhit-scalar
#SBATCH --partition=main
#SBATCH --qos=default-safe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ssh -T -o BatchMode=yes -o ConnectTimeout=15 -o RemoteCommand=none -o ControlMaster=no -o ControlPath=none -o IdentityAgent=none ihep-work 'hostname; id'
exec /mnt/newdisk/shijie/conda/envs/py310/bin/python -u -B tools/observation_nhit_scalar/control.py "${1:?run ID required}"
