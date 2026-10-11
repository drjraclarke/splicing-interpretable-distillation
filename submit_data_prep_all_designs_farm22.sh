#!/bin/bash
#BSUB -J fas_all_designs_data_prep
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 16000
#BSUB -R "select[mem>=16000] rusage[mem=16000]"
#BSUB -W 1:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/data_prep_all_designs_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/data_prep_all_designs_%J.err

set -euo pipefail

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

"$PYTHON" "$REPO/distillation/data_prep_all_designs_hek.py"
