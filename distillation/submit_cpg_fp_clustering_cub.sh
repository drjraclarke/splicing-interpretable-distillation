#!/bin/bash
# =============================================================================
# CpG-island false-positive clustering on chr22 — pretrained vs. fine-tuned
# (sl650_u16_ensemble) OpenSpliceAI.
#
# Run from a Cub-connected session:
#   bsub < submit_cpg_fp_clustering_cub.sh
# =============================================================================
#BSUB -J cpg_fp_clustering
#BSUB -q inference
#BSUB -G s10923
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -M 32000
#BSUB -R "select[mem>=32000] rusage[mem=32000] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cpg_fp_clustering_cub.%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cpg_fp_clustering_cub.%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1
module load ISG/conda

mkdir -p /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs

conda run --no-capture-output -n openspliceai-infer \
  python /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/cpg_island_fp_clustering_chr22.py
