#!/bin/bash
#BSUB -J fas_exon6_analyze_confounds
#BSUB -q normal
#BSUB -G team354
#BSUB -n 2
#BSUB -M 24000
#BSUB -R "select[mem>=24000] rusage[mem=24000]"
#BSUB -W 0:30
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/analyze_confounds_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/analyze_confounds_%J.err

set -euo pipefail

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" analyze_confounds.py
