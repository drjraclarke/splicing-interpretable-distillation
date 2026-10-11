"""
Filter existing zero-shot SpliceAI/OpenSpliceAI and AlphaGenome predictions
down to the EXACT var_ids in our own Exon_6/HEK held-out test set, and
compute Pearson and Spearman correlations against measured PSI - a clean,
apples-to-apples comparison against our from-scratch real-data model's
held-out result (pearson-logits-r = 0.7315 on the same 34,741 test variants).
"""

import pandas as pd
from scipy import stats

OUR_TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test.csv.gz"

BASE = "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning"
SPLICEAI_FILE = f"{BASE}/results/benchmarking/mpra_inference_sl650_u16_final_fulltest/canonical_benchmark_ALLSITES_FIXED/psi/psi_scores_long.tsv.gz"
ALPHAGENOME_PT_FILE = f"{BASE}/alphagenome_inference/comparison_plots/psi/psi_scores_long.tsv.gz"
ALPHAGENOME_JAX_FILE = f"{BASE}/alphagenome_jax_inference/comparison_plots_test_split/psi/psi_scores_long.tsv.gz"


def correlate(pred: pd.Series, measured: pd.Series) -> dict:
    mask = pred.notna() & measured.notna()
    pred, measured = pred[mask], measured[mask]
    pearson = stats.pearsonr(pred, measured)
    spearman = stats.spearmanr(pred, measured)
    return {
        "n": int(mask.sum()),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_r": spearman.statistic,
        "spearman_p": spearman.pvalue,
    }


def main() -> None:
    print("Loading our exact test var_ids...")
    our_test = pd.read_csv(OUR_TEST_CSV, usecols=["var_id", "psi_shrunk"])
    our_var_ids = set(our_test["var_id"])
    print(f"n = {len(our_var_ids)}")

    results = []

    print("\nLoading SpliceAI/OpenSpliceAI predictions...")
    osai = pd.read_csv(SPLICEAI_FILE, sep="\t")
    osai_exon6 = osai[(osai["design"] == "Exon_6") & (osai["var_id"].isin(our_var_ids))]
    print(f"  matched rows: {len(osai_exon6)}, models present: {osai_exon6['model'].unique()}")
    for model_name, group in osai_exon6.groupby("model"):
        stats_row = correlate(group["pred"], group["psi_shrunk"])
        results.append({"model": f"osai_{model_name}", **stats_row})

    print("\nLoading AlphaGenome (PyTorch) predictions...")
    ag_pt = pd.read_csv(ALPHAGENOME_PT_FILE, sep="\t")
    ag_pt_exon6 = ag_pt[(ag_pt["design"] == "Exon_6") & (ag_pt["var_id"].isin(our_var_ids))]
    print(f"  matched rows: {len(ag_pt_exon6)}, models present: {ag_pt_exon6['model'].unique()}")
    for model_name, group in ag_pt_exon6.groupby("model"):
        stats_row = correlate(group["pred"], group["psi_shrunk"])
        results.append({"model": f"alphagenome_pt_{model_name}", **stats_row})

    print("\nLoading AlphaGenome (JAX) predictions...")
    ag_jax = pd.read_csv(ALPHAGENOME_JAX_FILE, sep="\t")
    ag_jax_exon6 = ag_jax[(ag_jax["design"] == "Exon_6") & (ag_jax["var_id"].isin(our_var_ids))]
    print(f"  matched rows: {len(ag_jax_exon6)}, models present: {ag_jax_exon6['model'].unique()}")
    for model_name, group in ag_jax_exon6.groupby("model"):
        stats_row = correlate(group["pred"], group["psi_shrunk"])
        results.append({"model": f"alphagenome_jax_{model_name}", **stats_row})

    results_df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)
    print("\n=== Zero-shot model correlations, filtered to our exact 34,741-variant test set ===")
    print(results_df.to_string(index=False))
    print("\nFor reference: our from-scratch real-data-trained model on this exact test set: "
          "pearson-logits-r = 0.7315, n = 34,741")

    out_path = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/zero_shot_comparison_on_our_test.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
