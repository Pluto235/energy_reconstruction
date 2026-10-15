#!/bin/bash
set -eo pipefail
case "$(hostname -s)" in lhws183|bws0995) echo 'Excluded startup-fault node'; exit 75;; esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
exec python3 -u -B "$1/source.py" "$@"
