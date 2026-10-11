#!/bin/bash
#BSUB -J confounds_opensplice
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 16000
#BSUB -R "select[mem>=16000] rusage[mem=16000]"
#BSUB -W 1:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/confounds_opensplice_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/confounds_opensplice_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" analyze_confounds_opensplice.py
