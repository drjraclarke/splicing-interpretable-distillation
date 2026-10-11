"""
Diagnostic for building one pooled, padded/truncated training set across all
26 real FAS-family designs: (1) full length distribution across ALL retained
variants (to pick a padding target length), (2) why combi_* designs produced
zero rows in the previous per-design join.
"""

import pandas as pd

VARIANTS_MASTER = (
    "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning/"
    "input_data/opensai_intermediate_tables_0based/"
    "variants_master_HEK_qc20_0based_trimmed.tsv.gz"
)
CORRECTED_PSI = (
    "/lustre/scratch126/gengen/projects_v2/lehner_splicing/processed_pipeline_data/"
    "HEK_all_designs.canon_only.corrected_psi.tsv.gz"
)
SPLIT_ASSIGNMENTS = (
    "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning/"
    "input_data/opensai_intermediate_tables_0based/"
    "split_assignments_HEK_ALL_DESIGNS_70_10_20.tsv.gz"
)

REAL_26 = (
    ["Exon_6"]
    + [f"i5_{i}" for i in range(1, 7)]
    + [f"i6_{i}" for i in range(1, 14)]
    + [f"combi_{i}" for i in range(1, 7)]
)
COMBI = [f"combi_{i}" for i in range(1, 7)]


def main() -> None:
    print("Reading variants_master (design, var_id, full_len only)...")
    variants = pd.read_csv(VARIANTS_MASTER, sep="\t", usecols=["design", "var_id", "full_len"])
    print(f"Design value counts (all designs in variants_master):")
    print(variants["design"].value_counts().to_string())
    print()
    print(f"combi_* rows in variants_master: {variants['design'].isin(COMBI).sum()}")

    variants_26 = variants[variants["design"].isin(REAL_26)]

    print("\nReading corrected PSI (design, var_id only)...")
    psi = pd.read_csv(CORRECTED_PSI, sep="\t", usecols=["design", "var_id", "psi_shrunk"])
    print(f"combi_* rows in corrected_psi: {psi['design'].isin(COMBI).sum()}")
    print(f"combi_* rows with non-null psi_shrunk: {psi[psi['design'].isin(COMBI)]['psi_shrunk'].notna().sum()}")

    print("\nReading split assignments (design, var_id only)...")
    splits = pd.read_csv(SPLIT_ASSIGNMENTS, sep="\t")
    print(f"combi_* rows in split_assignments: {splits['design'].isin(COMBI).sum()}")

    psi_26 = psi[psi["design"].isin(REAL_26)].dropna(subset=["psi_shrunk"])
    splits_26 = splits[splits["design"].isin(REAL_26)]

    merged = variants_26.merge(psi_26[["design", "var_id", "psi_shrunk"]], on=["design", "var_id"], how="inner")
    print(f"\nAfter variants+psi merge, combi_* rows: {merged['design'].isin(COMBI).sum()}")
    merged = merged.merge(splits_26[["design", "var_id", "split"]], on=["design", "var_id"], how="inner")
    print(f"After +splits merge, combi_* rows: {merged['design'].isin(COMBI).sum()}")

    print(f"\nTotal retained variants across all designs: {len(merged)}")
    print("\nFull_len distribution (all retained variants, all designs):")
    print(merged["full_len"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.995, 0.999]).to_string())
    print(f"\nMax full_len: {merged['full_len'].max()}")
    print(f"Min full_len: {merged['full_len'].min()}")

    print("\nPer-design length range:")
    print(
        merged.groupby("design")["full_len"]
        .agg(["min", "max", "nunique", "count"])
        .sort_values("count", ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
