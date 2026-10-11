#!/bin/bash
#BSUB -J splicing_distill_create_env
#BSUB -q normal
#BSUB -G team354
#BSUB -n 4
#BSUB -M 8000
#BSUB -R "select[mem>=8000] rusage[mem=8000]"
#BSUB -W 1:00
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/create_env_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/create_env_%J.err

set -euo pipefail
source /software/conda/miniforge/etc/profile.d/conda.sh

ENV_PATH=/software/conda/users/jc58/splicing_distill

mamba create -y -p "$ENV_PATH" python=3.11
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1

# CUDA-enabled torch build, matching the CUDA version used by other jc58 envs on Cub
# (openspliceai/alphagenome envs both use cu121).
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121

# Training-loop dependencies only (skip tensorflow/keras/spliceai/biopython/jupyterlab -
# those are only needed by the paper's original data-generation and analysis notebooks,
# not by distillation/experiment.py or hyperparameter_search.py).
pip install pandas numpy scipy "ray[tune]==2.55.1" hyperopt==0.2.7

echo "Env created at $ENV_PATH"
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
