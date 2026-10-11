"""
Build training/test CSVs for the Conv1DNBM distillation architecture from our own
real FAS Exon_6 / HEK293 random-sequence MPRA data, instead of the paper's synthetic
random-exon library.

Joins on var_id:
  - variants_master_HEK_qc20_0based_trimmed.tsv.gz  -> full_seq (fixed 535nt)
  - HEK_all_designs.canon_only.corrected_psi.tsv.gz -> psi_shrunk (measured PSI, shrunk)
  - split_assignments_HEK_ALL_DESIGNS_70_10_20.tsv.gz -> split (train_pool/val/test)

Output columns match what --sequence_column/--target_column will point at:
  var_id, full_seq, psi_shrunk

Writes both the full train/test split and a small fixed-seed subsample for a quick
smoke test.
"""

import gzip
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

DESIGN = "Exon_6"
SMOKETEST_TRAIN_N = 5000
SMOKETEST_TEST_N = 1000
RANDOM_SEED = 42


def main() -> None:
    print("Reading variants_master...")
    variants = pd.read_csv(
        VARIANTS_MASTER, sep="\t", usecols=["design", "var_id", "full_seq", "full_len"]
    )
    variants = variants[variants["design"] == DESIGN].copy()
    print(f"  {len(variants)} {DESIGN} variants")
    assert variants["full_len"].nunique() == 1, "expected a single fixed sequence length"

    # full_seq uses soft-masking (lowercase = intron, uppercase = exon); the model's
    # one-hot encoder (distillation/datasets/sequence.py) only has uppercase A/C/G/T
    # keys, so normalize case here.
    variants["full_seq"] = variants["full_seq"].str.upper()

    print("Reading corrected PSI...")
    psi = pd.read_csv(
        CORRECTED_PSI, sep="\t", usecols=["design", "var_id", "psi_shrunk"]
    )
    psi = psi[psi["design"] == DESIGN].dropna(subset=["psi_shrunk"])
    print(f"  {len(psi)} {DESIGN} variants with measured PSI")

    print("Reading split assignments...")
    splits = pd.read_csv(SPLIT_ASSIGNMENTS, sep="\t")
    splits = splits[splits["design"] == DESIGN]

    merged = variants.merge(psi[["var_id", "psi_shrunk"]], on="var_id", how="inner")
    merged = merged.merge(splits[["var_id", "split"]], on="var_id", how="inner")
    merged = merged[["var_id", "full_seq", "psi_shrunk", "split"]]
    print(f"  {len(merged)} variants with sequence + PSI + split")

    assert merged["psi_shrunk"].between(0, 1, inclusive="neither").all(), (
        "psi_shrunk must be strictly within (0, 1) for kl_divergence_logits loss"
    )

    train = merged[merged["split"] == "train_pool"].drop(columns="split")
    test = merged[merged["split"] == "test"].drop(columns="split")
    print(f"  train: {len(train)}  test: {len(test)}")

    train.to_csv(f"{OUT_DIR}/fas_exon6_hek_train.csv.gz", index=False)
    test.to_csv(f"{OUT_DIR}/fas_exon6_hek_test.csv.gz", index=False)

    train_sub = train.sample(n=min(SMOKETEST_TRAIN_N, len(train)), random_state=RANDOM_SEED)
    test_sub = test.sample(n=min(SMOKETEST_TEST_N, len(test)), random_state=RANDOM_SEED)
    train_sub.to_csv(f"{OUT_DIR}/fas_exon6_hek_smoketest_train.csv.gz", index=False)
    test_sub.to_csv(f"{OUT_DIR}/fas_exon6_hek_smoketest_test.csv.gz", index=False)

    print("Done.")
    print(f"  full train/test:      {len(train)} / {len(test)}")
    print(f"  smoketest train/test: {len(train_sub)} / {len(test_sub)}")


if __name__ == "__main__":
    main()
