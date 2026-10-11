#!/bin/bash
#BSUB -J fas_exon6_data_prep
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 8000
#BSUB -R "select[mem>=8000] rusage[mem=8000]"
#BSUB -W 0:30
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/data_prep_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/data_prep_%J.err

set -euo pipefail

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

"$PYTHON" "$REPO/distillation/data_prep_fas_exon6_hek.py"
