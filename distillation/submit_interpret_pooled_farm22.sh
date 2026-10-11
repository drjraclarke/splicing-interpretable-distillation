#!/bin/bash
#BSUB -J interpret_pooled
#BSUB -G team354
#BSUB -q normal
#BSUB -n 8
#BSUB -M 32000
#BSUB -R "select[mem>32000] rusage[mem=32000] span[hosts=1]"
#BSUB -W 6:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/interpret_pooled.%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/interpret_pooled.%J.err

set -euo pipefail

mkdir -p /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs
cd /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation

PY=/software/conda/users/jc58/splicing_distill/bin/python3
echo "Using python: $PY"

"$PY" inspect_motifs_pooled.py
"$PY" plot_position_effects_pooled.py
