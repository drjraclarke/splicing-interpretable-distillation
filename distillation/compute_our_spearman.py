import pandas as pd
from scipy import stats

df = pd.read_csv(
    "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test_predictions.csv"
)
pearson = stats.pearsonr(df["predicted_psi"], df["measured_psi"])
spearman = stats.spearmanr(df["predicted_psi"], df["measured_psi"])
print(f"n = {len(df)}")
print(f"pearson_r = {pearson.statistic:.4f}, p = {pearson.pvalue:.3g}")
print(f"spearman_r = {spearman.statistic:.4f}, p = {spearman.pvalue:.3g}")
