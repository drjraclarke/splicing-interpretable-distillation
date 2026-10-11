#!/bin/bash
#BSUB -J compute_our_spearman
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>=4000] rusage[mem=4000]"
#BSUB -W 0:10
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/our_spearman_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/our_spearman_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" compute_our_spearman.py
