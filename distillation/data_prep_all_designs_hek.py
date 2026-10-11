"""
Build per-design training/test CSVs (same format as data_prep_fas_exon6_hek.py)
for ALL 26 real FAS-family random-sequence designs in HEK293, to check whether
the Exon_6 result generalizes across designs rather than being a fluke.

Reads each of the 3 source tables ONCE (not once per design) and groups by
design, since re-reading the full ~100-200MB source files per design would be
wasteful with 26 designs.

Designs with more than one distinct sequence length are skipped (the model
requires a fixed-length input per design) and reported at the end.
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

OUT_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data"

REAL_26 = (
    ["Exon_6"]
    + [f"i5_{i}" for i in range(1, 7)]
    + [f"i6_{i}" for i in range(1, 14)]
    + [f"combi_{i}" for i in range(1, 7)]
)


def main() -> None:
    print("Reading variants_master...")
    variants = pd.read_csv(
        VARIANTS_MASTER, sep="\t", usecols=["design", "var_id", "full_seq", "full_len"]
    )
    variants = variants[variants["design"].isin(REAL_26)].copy()
    variants["full_seq"] = variants["full_seq"].str.upper()

    print("Reading corrected PSI...")
    psi = pd.read_csv(CORRECTED_PSI, sep="\t", usecols=["design", "var_id", "psi_shrunk"])
    psi = psi[psi["design"].isin(REAL_26)].dropna(subset=["psi_shrunk"])

    print("Reading split assignments...")
    splits = pd.read_csv(SPLIT_ASSIGNMENTS, sep="\t")
    splits = splits[splits["design"].isin(REAL_26)]

    merged = variants.merge(psi[["design", "var_id", "psi_shrunk"]], on=["design", "var_id"], how="inner")
    merged = merged.merge(splits[["design", "var_id", "split"]], on=["design", "var_id"], how="inner")

    summary_rows = []
    skipped = []
    for design, group in merged.groupby("design"):
        n_lengths = group["full_len"].nunique()
        if n_lengths != 1:
            skipped.append((design, n_lengths))
            continue

        assert group["psi_shrunk"].between(0, 1, inclusive="neither").all()

        train = group[group["split"] == "train_pool"][["var_id", "full_seq", "psi_shrunk"]]
        test = group[group["split"] == "test"][["var_id", "full_seq", "psi_shrunk"]]

        train.to_csv(f"{OUT_DIR}/fas_{design}_hek_train.csv.gz", index=False)
        test.to_csv(f"{OUT_DIR}/fas_{design}_hek_test.csv.gz", index=False)

        summary_rows.append(
            {
                "design": design,
                "seq_len": int(group["full_len"].iloc[0]),
                "n_total": len(group),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        print(f"{design}: seq_len={group['full_len'].iloc[0]} n_train={len(train)} n_test={len(test)}")

    summary = pd.DataFrame(summary_rows).sort_values("n_total", ascending=False)
    summary.to_csv(f"{OUT_DIR}/all_designs_hek_summary.tsv", sep="\t", index=False)

    print()
    print(summary.to_string(index=False))
    if skipped:
        print()
        print(f"Skipped {len(skipped)} design(s) with inconsistent sequence length: {skipped}")


if __name__ == "__main__":
    main()
