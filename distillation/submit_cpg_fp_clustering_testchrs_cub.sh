#!/bin/bash
# =============================================================================
# CpG-island false-positive clustering on chr1/3/5/7/9 (standard SpliceAI
# test chromosomes) - pretrained vs. fine-tuned (sl650_u16_ensemble)
# OpenSpliceAI. Scaled-up version of the chr22-only check.
#
# Requires genomic_h5_testchrs_f10000/dataset_test.h5 to exist first
# (build_genomic_testchrs_h5_farm22.sh, submitted separately on Farm22,
# ~2-4h). Do not submit this until that job has completed.
#
# Run from a Cub-connected session:
#   bsub < submit_cpg_fp_clustering_testchrs_cub.sh
# =============================================================================
#BSUB -J cpg_fp_clustering_testchrs
#BSUB -q inference
#BSUB -G s10923
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -M 48000
#BSUB -R "select[mem>=48000] rusage[mem=48000] span[hosts=1]"
#BSUB -W 3:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cpg_fp_clustering_testchrs_cub.%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs/cpg_fp_clustering_testchrs_cub.%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1
module load ISG/conda

mkdir -p /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/logs

conda run --no-capture-output -n openspliceai-infer \
  python /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/cpg_island_fp_clustering_testchrs.py
