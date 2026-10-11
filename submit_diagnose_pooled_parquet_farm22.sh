#!/bin/bash
#BSUB -J diagnose_pooled_parquet
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 16000
#BSUB -R "select[mem>=16000] rusage[mem=16000]"
#BSUB -W 0:30
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/diagnose_pooled_parquet_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/diagnose_pooled_parquet_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" diagnose_pooled_parquet.py
