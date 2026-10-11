"""
Build one pooled, padded HEK training/test set across the 16 real designs
that have a single, internally-consistent full_seq length, using the
pre-built pooled_all_designs_psi_and_sequence.parquet (which already fixes
the per-row sequence-trimming inconsistency found in the older, unversioned
variants_master_HEK_qc20_0based_trimmed.tsv.gz).

Padding strategy: insert filler ('N', all-zero one-hot) at each design's own
empirically-determined variable-region boundary, so the constant 5' flank
stays aligned at position 0 and the constant 3' flank stays aligned ending at
position L-1 for every design, regardless of internal length differences.

Split: fresh, reproducible 80/20 train/test split, hashed on (design, var_id)
- this pooled resource has no existing split column, and the project's old
split_assignments file predates combi_1-6 and would leave them unsplit.
"""

import hashlib

import numpy as np
import pandas as pd

PARQUET = (
    "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning/"
    "input_data/random_mpra_metadata/pooled_all_designs_psi_and_sequence/"
    "pooled_all_designs_psi_and_sequence.parquet"
)
OUT_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data"

# The 16 designs confirmed to have a single consistent full_seq length in
# this parquet (diagnose_pooled_parquet.py output).
CLEAN_DESIGNS = [
    "Exon_6", "combi_1", "combi_2", "combi_3", "combi_4", "combi_5", "combi_6",
    "i5_3", "i6_1", "i6_2", "i6_7", "i6_8", "i6_9", "i6_10", "i6_12", "i6_13",
]

TRAIN_FRACTION = 0.8


def find_variable_region(seqs: np.ndarray) -> tuple[int, int]:
    arr = np.array([list(s) for s in seqs])
    is_variable = np.any(arr != arr[0:1, :], axis=0)
    variable_positions = np.where(is_variable)[0]
    assert len(variable_positions) > 0, "no variable positions found"
    return int(variable_positions.min()), int(variable_positions.max()) + 1


def pad_at_boundary(seq: str, insert_pos: int, n_pad: int) -> str:
    return seq[:insert_pos] + ("N" * n_pad) + seq[insert_pos:]


def assign_split(design: str, var_id: str) -> str:
    h = hashlib.md5(f"{design}_{var_id}".encode()).hexdigest()
    return "train" if (int(h, 16) % 100) < int(TRAIN_FRACTION * 100) else "test"


def main() -> None:
    print("Reading pooled parquet...")
    df = pd.read_parquet(PARQUET, columns=["design", "var_id", "psi_hek", "full_seq"])
    df = df[df["design"].isin(CLEAN_DESIGNS)]
    df = df.dropna(subset=["psi_hek", "full_seq"])
    df["full_seq"] = df["full_seq"].str.upper()

    print(f"Rows after filtering to {len(CLEAN_DESIGNS)} clean designs + non-null psi_hek/full_seq: {len(df)}")

    # kl_divergence_logits's baseline term (binary_cross_entropy(targets, targets))
    # involves log(0)/log(1) at exact 0/1, producing inf/NaN. A handful of rows in
    # this broader pooled resource hit those exact bounds (unlike Exon_6 alone,
    # which stayed strictly interior) - clip slightly inward rather than drop.
    EPS = 1e-4
    n_clipped = ((df["psi_hek"] <= 0) | (df["psi_hek"] >= 1)).sum()
    print(f"Clipping {n_clipped} rows with psi_hek at exact 0/1 boundary to [{EPS}, {1-EPS}]")
    df["psi_hek"] = df["psi_hek"].clip(lower=EPS, upper=1 - EPS)

    assert df["psi_hek"].between(0, 1, inclusive="neither").all(), (
        "psi_hek must be strictly within (0, 1) for kl_divergence_logits loss"
    )

    design_info = {}
    print("\nDetermining each design's variable-region boundary...")
    for design, group in df.groupby("design"):
        lens = group["full_seq"].str.len()
        assert lens.nunique() == 1, f"{design} has inconsistent full_seq length: {lens.unique()}"
        seq_len = int(lens.iloc[0])

        sample = group["full_seq"].sample(n=min(10000, len(group)), random_state=0).to_numpy()
        start, end = find_variable_region(sample)
        design_info[design] = {"seq_len": seq_len, "variable_start": start, "variable_end": end, "n": len(group)}
        print(f"  {design}: n={len(group)}, seq_len={seq_len}, variable_region=[{start}, {end})")

    target_len = max(info["seq_len"] for info in design_info.values())
    print(f"\nTarget pooled length: {target_len}")

    def pad_row(row):
        info = design_info[row["design"]]
        n_pad = target_len - info["seq_len"]
        if n_pad == 0:
            return row["full_seq"]
        return pad_at_boundary(row["full_seq"], info["variable_start"], n_pad)

    print("Padding sequences...")
    df["full_seq"] = df.apply(pad_row, axis=1)
    assert (df["full_seq"].str.len() == target_len).all()

    print("Assigning train/test split...")
    df["split"] = df.apply(lambda r: assign_split(r["design"], r["var_id"]), axis=1)

    train_df = df[df["split"] == "train"][["design", "var_id", "full_seq", "psi_hek"]]
    test_df = df[df["split"] == "test"][["design", "var_id", "full_seq", "psi_hek"]]
    train_df = train_df.rename(columns={"psi_hek": "psi_shrunk"})
    test_df = test_df.rename(columns={"psi_hek": "psi_shrunk"})

    print(f"\nTrain: {len(train_df)}  Test: {len(test_df)}")
    print("\nPer-design train counts:")
    print(train_df["design"].value_counts().to_string())

    train_df.to_csv(f"{OUT_DIR}/fas_pooled16_hek_train.csv.gz", index=False)
    test_df.to_csv(f"{OUT_DIR}/fas_pooled16_hek_test.csv.gz", index=False)

    summary_df = pd.DataFrame.from_dict(design_info, orient="index")
    summary_df.index.name = "design"
    summary_df["pad_amount"] = target_len - summary_df["seq_len"]
    summary_df.to_csv(f"{OUT_DIR}/pooled16_design_info.tsv", sep="\t")
    print(f"\nWrote fas_pooled16_hek_train/test.csv.gz and pooled16_design_info.tsv to {OUT_DIR}")


if __name__ == "__main__":
    main()
