#!/bin/bash
#BSUB -J fas_exon6_hek_smoketest
#BSUB -q training-normal
#BSUB -G s10923
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>=32000] rusage[mem=32000]"
#BSUB -M 32000
#BSUB -W 1:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/smoketest_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/smoketest_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

"$PYTHON" hyperparameter_search.py \
  --experiment_name fas_exon6_hek_smoketest \
  --num_trials 2 \
  --train_csv "$REPO/distillation/data/fas_exon6_hek_smoketest_train.csv.gz" \
  --train_fraction 0.9 \
  --test_csv "$REPO/distillation/data/fas_exon6_hek_smoketest_test.csv.gz" \
  --sequence_column full_seq \
  --target_column psi_shrunk \
  --data_type sequence \
  --model_architecture conv1d_nbm \
  --loss kl_divergence_logits \
  --num_cpus 4 \
  --num_cpus_per_trial 2 \
  --num_gpus_per_trial 1 \
  --asha_max_epochs 10 \
  --asha_grace_period 10 \
  --result_dir "$REPO/ray_results"

echo "Job finished: $(date)"
