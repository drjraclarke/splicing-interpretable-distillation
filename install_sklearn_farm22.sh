#!/bin/bash
#BSUB -J install_sklearn
#BSUB -q normal
#BSUB -G team354
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>=4000] rusage[mem=4000]"
#BSUB -W 0:15
#BSUB -o /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/install_sklearn_%J.out
#BSUB -e /nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/logs/install_sklearn_%J.err

set -euo pipefail
/software/conda/users/jc58/splicing_distill/bin/pip install scikit-learn
/software/conda/users/jc58/splicing_distill/bin/python3 -c "from sklearn.isotonic import IsotonicRegression; print('sklearn OK')"
