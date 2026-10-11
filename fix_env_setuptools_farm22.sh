#!/bin/bash
#BSUB -J splicing_distill_fix_setuptools
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 2000
#BSUB -R "select[mem>=2000] rusage[mem=2000]"
#BSUB -W 0:15
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/fix_setuptools_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/fix_setuptools_%J.err

set -euo pipefail

ENV_PATH=/software/conda/users/jc58/splicing_distill
# Recent setuptools (>=81) dropped pkg_resources entirely; hyperopt's atpe submodule
# still imports it unconditionally at package-init time. Pin to a pre-removal version.
"$ENV_PATH/bin/pip" install "setuptools<81"
"$ENV_PATH/bin/python3" -c "import pkg_resources; print('pkg_resources OK')"
"$ENV_PATH/bin/python3" -c "from hyperopt import hp; print('hyperopt import OK')"
