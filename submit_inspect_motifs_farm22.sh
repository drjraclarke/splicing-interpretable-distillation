#!/bin/bash
#BSUB -J fas_exon6_inspect_motifs
#BSUB -q normal
#BSUB -G team354
#BSUB -n 2
#BSUB -M 48000
#BSUB -R "select[mem>=48000] rusage[mem=48000]"
#BSUB -W 2:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/inspect_motifs_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/inspect_motifs_%J.err

set -euo pipefail

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
"$PYTHON" inspect_motifs.py
