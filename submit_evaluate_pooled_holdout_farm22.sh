#!/bin/bash
#BSUB -J evaluate_pooled_holdout
#BSUB -q normal
#BSUB -G team354
#BSUB -n 2
#BSUB -M 16000
#BSUB -R "select[mem>=16000] rusage[mem=16000]"
#BSUB -W 3:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/evaluate_pooled_holdout_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/evaluate_pooled_holdout_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" evaluate_pooled_holdout.py
