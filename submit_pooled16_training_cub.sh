#!/bin/bash
#BSUB -J fas_pooled16_hek_full_v2
#BSUB -q training-normal
#BSUB -G s10923
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>=64000] rusage[mem=64000]"
#BSUB -M 64000
#BSUB -W 8:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/pooled16_train_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/pooled16_train_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

# 1.5M train rows (~12.6x the earlier Exon_6-only run) - reduced trial/epoch
# budget vs. that run to keep total compute bounded (more data per epoch
# roughly proportionally increases per-epoch wall time). AMP is NOT used:
# kl_divergence_logits's baseline term calls plain binary_cross_entropy
# (not the logits variant) internally, which torch.autocast refuses to run
# under mixed precision (confirmed - crashed all 10 trials identically on
# the first training step). The one-hot encoding step was rewritten to be
# vectorized (numpy) rather than per-base Python objects, which was the
# actual memory bottleneck at this scale, not GPU compute/AMP.
"$PYTHON" hyperparameter_search.py \
  --experiment_name fas_pooled16_hek_full_v2 \
  --num_trials 10 \
  --train_csv "$REPO/distillation/data/fas_pooled16_hek_train.csv.gz" \
  --train_fraction 0.9 \
  --test_csv "$REPO/distillation/data/fas_pooled16_hek_test.csv.gz" \
  --sequence_column full_seq \
  --target_column psi_shrunk \
  --data_type sequence \
  --model_architecture conv1d_nbm \
  --loss kl_divergence_logits \
  --num_cpus 4 \
  --num_cpus_per_trial 2 \
  --num_gpus_per_trial 1 \
  --asha_max_epochs 50 \
  --asha_grace_period 10 \
  --result_dir "$REPO/ray_results"

echo "Job finished: $(date)"
