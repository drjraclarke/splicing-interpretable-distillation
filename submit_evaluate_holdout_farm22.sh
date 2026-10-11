#!/bin/bash
#BSUB -J fas_exon6_evaluate_holdout
#BSUB -q normal
#BSUB -G team354
#BSUB -n 2
#BSUB -M 24000
#BSUB -R "select[mem>=24000] rusage[mem=24000]"
#BSUB -W 0:30
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/evaluate_holdout_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/evaluate_holdout_%J.err

set -euo pipefail

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"

"$PYTHON" evaluate_on_holdout.py \
  --experiment_dir "$REPO/ray_results/fas_exon6_hek_full" \
  --trial_id 7873bd5b \
  --test_csv "$REPO/distillation/data/fas_exon6_hek_test.csv.gz" \
  --sequence_column full_seq \
  --target_column psi_shrunk \
  --out_csv "$REPO/distillation/data/fas_exon6_hek_test_predictions.csv"
