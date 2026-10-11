#!/bin/bash
#BSUB -J evaluate_pooled_holdout_gpu
#BSUB -q inference
#BSUB -G s10923
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>=16000] rusage[mem=16000]"
#BSUB -M 16000
#BSUB -W 1:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/evaluate_pooled_holdout_gpu_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/evaluate_pooled_holdout_gpu_%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/software/conda/users/jc58/splicing_distill/bin/python3
REPO=/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation

cd "$REPO/distillation"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

# Same script as the CPU version (submit_evaluate_pooled_holdout_farm22.sh) -
# evaluate_pooled_holdout.py already auto-detects CUDA, no code change needed.
"$PYTHON" evaluate_pooled_holdout.py
