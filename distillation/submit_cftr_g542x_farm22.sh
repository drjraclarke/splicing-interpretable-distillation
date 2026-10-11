#!/bin/bash
#BSUB -J cftr_g542x
#BSUB -G team354
#BSUB -q normal
#BSUB -n 4
#BSUB -M 8000
#BSUB -R "select[mem>8000] rusage[mem=8000] span[hosts=1]"
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cftr_g542x.%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cftr_g542x.%J.err

set -euo pipefail

mkdir -p /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs

/software/conda/users/jc58/openspliceai/bin/python3 \
  /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/run_cftr_g542x_prediction.py
