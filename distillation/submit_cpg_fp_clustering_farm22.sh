#!/bin/bash
#BSUB -J cpg_fp_clustering
#BSUB -G team354
#BSUB -q normal
#BSUB -n 8
#BSUB -M 32000
#BSUB -R "select[mem>32000] rusage[mem=32000] span[hosts=1]"
#BSUB -W 4:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cpg_fp_clustering.%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cpg_fp_clustering.%J.err

set -euo pipefail

mkdir -p /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs

/software/conda/users/jc58/openspliceai/bin/python3 \
  /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/cpg_island_fp_clustering_chr22.py
