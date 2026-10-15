#!/bin/bash
#SBATCH --partition=main
#SBATCH --qos=default-safe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=v7friend_final
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
python=/mnt/newdisk/shijie/conda/envs/py310/bin/python
"$python" tools/v7_friends/audit.py --inventory output/v7_friends_20261015/input_inventory.json --run "$1"
"$python" tools/v7_friends/diagnose_hits.py --run "$1"
"$python" tools/v7_friends/deliver.py --inventory output/v7_friends_20261015/input_inventory.json --run "$1"
