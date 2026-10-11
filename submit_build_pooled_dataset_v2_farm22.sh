#!/bin/bash
#BSUB -J fas_build_pooled_dataset_v2
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 24000
#BSUB -R "select[mem>=24000] rusage[mem=24000]"
#BSUB -W 1:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/build_pooled_dataset_v2_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/build_pooled_dataset_v2_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" build_pooled_dataset_v2.py
