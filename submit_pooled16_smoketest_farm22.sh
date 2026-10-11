#!/bin/bash
#BSUB -J fas_pooled16_smoketest
#BSUB -q normal
#BSUB -G team354
#BSUB -n 2
#BSUB -M 32000
#BSUB -R "select[mem>=32000] rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/pooled16_smoketest_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/pooled16_smoketest_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"

echo "=== Building subsample ==="
"$PYTHON" build_pooled_smoketest_sample.py

echo "=== Running CPU smoke test (same code path as the Cub job, minus GPU/AMP) ==="
"$PYTHON" hyperparameter_search.py \
  --experiment_name fas_pooled16_smoketest_v2 \
  --num_trials 1 \
  --train_csv "$REPO/distillation/data/fas_pooled16_hek_smoketest_train.csv.gz" \
  --train_fraction 0.9 \
  --test_csv "$REPO/distillation/data/fas_pooled16_hek_smoketest_test.csv.gz" \
  --sequence_column full_seq \
  --target_column psi_shrunk \
  --data_type sequence \
  --model_architecture conv1d_nbm \
  --loss kl_divergence_logits \
  --num_cpus 4 \
  --num_cpus_per_trial 2 \
  --num_gpus_per_trial 0 \
  --asha_max_epochs 5 \
  --asha_grace_period 5 \
  --result_dir "$REPO/ray_results"

echo "Smoke test finished: $(date)"
