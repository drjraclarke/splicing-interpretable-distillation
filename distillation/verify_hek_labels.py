"""
Verify the pooled dataset's psi_shrunk (originally psi_hek from the pooled
parquet) actually matches HEK measurements, by cross-checking a sample of
(design, var_id) pairs against the authoritative source table directly, and
against psi_mcf7/psi_shsy to confirm they're not accidentally identical
(which would suggest a column mix-up).
"""

import pandas as pd

CORRECTED_PSI_HEK = "/lustre/scratch126/gengen/projects_v2/lehner_splicing/processed_pipeline_data/HEK_all_designs.canon_only.corrected_psi.tsv.gz"
CORRECTED_PSI_MCF7 = "/lustre/scratch126/gengen/projects_v2/lehner_splicing/processed_pipeline_data/MCF7_all_designs.canon_only.corrected_psi.tsv.gz"
POOLED_PARQUET = (
    "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning/"
    "input_data/random_mpra_metadata/pooled_all_designs_psi_and_sequence/"
    "pooled_all_designs_psi_and_sequence.parquet"
)
POOLED_TRAIN_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_pooled16_hek_train.csv.gz"


def main() -> None:
    print("Reading pooled parquet (design, var_id, psi_hek, psi_mcf7, psi_shsy only)...")
    pooled = pd.read_parquet(POOLED_PARQUET, columns=["design", "var_id", "psi_hek", "psi_mcf7", "psi_shsy"])
    pooled_exon6 = pooled[pooled["design"] == "Exon_6"]

    sample_var_ids = pooled_exon6["var_id"].head(5).tolist()
    print(f"Sample var_ids: {sample_var_ids}")
    print("\nPooled parquet values for these var_ids:")
    print(pooled_exon6[pooled_exon6["var_id"].isin(sample_var_ids)].to_string(index=False))

    print("\nReading authoritative HEK corrected_psi table...")
    hek = pd.read_csv(CORRECTED_PSI_HEK, sep="\t", usecols=["design", "var_id", "psi_shrunk"])
    hek_exon6 = hek[(hek["design"] == "Exon_6") & (hek["var_id"].isin(sample_var_ids))]
    print("\nAuthoritative HEK_all_designs.canon_only.corrected_psi.tsv.gz values:")
    print(hek_exon6.to_string(index=False))

    print("\nReading authoritative MCF7 corrected_psi table (sanity check - should differ from HEK)...")
    mcf7 = pd.read_csv(CORRECTED_PSI_MCF7, sep="\t", usecols=["design", "var_id", "psi_shrunk"])
    mcf7_exon6 = mcf7[(mcf7["design"] == "Exon_6") & (mcf7["var_id"].isin(sample_var_ids))]
    print("\nAuthoritative MCF7_all_designs.canon_only.corrected_psi.tsv.gz values:")
    print(mcf7_exon6.to_string(index=False))

    print("\nChecking final pooled training CSV (post-clip, post-uppercase, post-pad) for same var_ids...")
    train_df = pd.read_csv(POOLED_TRAIN_CSV)
    train_sample = train_df[(train_df["design"] == "Exon_6") & (train_df["var_id"].isin(sample_var_ids))]
    print(train_sample[["design", "var_id", "psi_shrunk"]].to_string(index=False))

    # Global check: correlation between pooled psi_hek and psi_mcf7 should be
    # well below 1.0 (they measure different cell lines, not identical columns)
    both = pooled.dropna(subset=["psi_hek", "psi_mcf7"])
    corr = both["psi_hek"].corr(both["psi_mcf7"])
    print(f"\nGlobal correlation between pooled psi_hek and psi_mcf7 (all designs, n={len(both)}): {corr:.4f}")
    print("(Should be well below 1.0 - if ~1.0, columns may be duplicated/mislabeled)")


if __name__ == "__main__":
    main()
