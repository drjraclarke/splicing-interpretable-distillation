"""
Estimate the noise ceiling on real measured PSI (psi_shrunk), using the raw
per-replicate PSI values (psi1, psi2, psi3) that feed into it, restricted to
our exact FAS Exon_6/HEK test set variants for direct comparability with our
model's held-out result (pearson-logits-r = 0.7315).

Method: average pairwise correlation among the 3 raw replicates estimates
single-replicate reliability. The Spearman-Brown prophecy formula then
estimates the reliability of the combined (3-replicate) measurement that
psi_shrunk approximates - i.e., the theoretical ceiling on how well ANY
model, however good, could correlate with real measured PSI given
biological+technical replicate noise. sqrt(reliability) is the ceiling on
achievable Pearson r; our model's r as a fraction of that ceiling is a fairer
comparison to the paper's r=0.94-0.99 (which faces no such noise ceiling,
since their target is another model's own deterministic output, not a noisy
real measurement).
"""

import numpy as np
import pandas as pd

CORRECTED_PSI_HEK = "/lustre/scratch126/gengen/projects_v2/lehner_splicing/processed_pipeline_data/HEK_all_designs.canon_only.corrected_psi.tsv.gz"
OUR_TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test.csv.gz"

OUR_MODEL_R = 0.7315


def spearman_brown(r_single: float, n: int) -> float:
    """Reliability of the average of n measurements, given average pairwise
    single-measurement correlation r_single."""
    return (n * r_single) / (1 + (n - 1) * r_single)


def main() -> None:
    print("Loading our exact test var_ids...")
    our_test = pd.read_csv(OUR_TEST_CSV, usecols=["var_id"])
    our_var_ids = set(our_test["var_id"])
    print(f"n = {len(our_var_ids)}")

    print("Loading raw per-replicate PSI values...")
    df = pd.read_csv(
        CORRECTED_PSI_HEK, sep="\t",
        usecols=["design", "var_id", "psi1", "psi2", "psi3"],
    )
    df = df[(df["design"] == "Exon_6") & (df["var_id"].isin(our_var_ids))]
    print(f"Rows for Exon_6 in our test set: {len(df)}")

    df_complete = df.dropna(subset=["psi1", "psi2", "psi3"])
    print(f"Rows with all 3 replicates present: {len(df_complete)}")

    pairs = [("psi1", "psi2"), ("psi1", "psi3"), ("psi2", "psi3")]
    pairwise_r = {}
    for a, b in pairs:
        r = np.corrcoef(df_complete[a], df_complete[b])[0, 1]
        pairwise_r[f"{a}_vs_{b}"] = r
        print(f"  corr({a}, {b}) = {r:.4f}, n = {len(df_complete)}")

    mean_pairwise_r = np.mean(list(pairwise_r.values()))
    print(f"\nMean pairwise single-replicate correlation: {mean_pairwise_r:.4f}")

    reliability_3rep = spearman_brown(mean_pairwise_r, 3)
    ceiling_r = np.sqrt(reliability_3rep)
    print(f"Spearman-Brown estimated reliability of 3-replicate consensus: {reliability_3rep:.4f}")
    print(f"Estimated ceiling on achievable Pearson r (sqrt(reliability)): {ceiling_r:.4f}")

    fraction_of_ceiling_r = OUR_MODEL_R / ceiling_r
    fraction_of_explainable_variance = (OUR_MODEL_R / ceiling_r) ** 2
    print(f"\nOur model's r = {OUR_MODEL_R}")
    print(f"Our model's r as a fraction of the noise ceiling: {fraction_of_ceiling_r:.4f}")
    print(f"Our model's r^2 as a fraction of EXPLAINABLE variance: {fraction_of_explainable_variance:.4f}")
    print(
        f"\nFor context: the paper's distilled-model-vs-black-box r=0.94-0.99 faces NO such "
        f"ceiling (target is another model's deterministic output). Our r={OUR_MODEL_R} against "
        f"a target with an estimated ceiling of {ceiling_r:.3f} means we're capturing "
        f"~{fraction_of_explainable_variance*100:.1f}% of the variance that is even possible to "
        f"explain given real measurement noise."
    )


if __name__ == "__main__":
    main()
