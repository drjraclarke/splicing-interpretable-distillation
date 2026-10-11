"""
Pool the 11 real FAS-family designs that each have a single, internally
consistent sequence length (no per-variant indels) into ONE combined
train/test set for a single unified model, instead of one model per design.

Padding strategy: naive left- or right-padding would misalign the constant
splice-site flanks across designs of different length, since the model uses
absolute positional encoding. Instead, each design's own randomized/variable
region is located empirically (by comparing sequences within that design,
same method as analyze_confounds.py's find_variable_region), and padding
('N', all-zero one-hot) is inserted exactly at that boundary. This keeps the
5' flank aligned starting at position 0, AND the 3' flank aligned ending at
position L-1, for every design, regardless of how much internal content
differs between designs.

Scope: only the 11 designs with nunique(full_len) == 1 (per
data_prep_all_designs_hek.py's all_designs_hek_summary.tsv). The other 9
designs vary length even within a single design (real indels / deliberate
window-width variation) and need per-variant alignment - out of scope here.
combi_* designs are entirely absent from variants_master_HEK and need
separate investigation - also out of scope here.
"""

import numpy as np
import pandas as pd

DATA_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data"

# The 11 designs with a single consistent full_len (from all_designs_hek_summary.tsv)
CLEAN_DESIGNS = [
    "Exon_6", "i5_3", "i6_1", "i6_2", "i6_7", "i6_8",
    "i6_9", "i6_10", "i6_11", "i6_12", "i6_13",
]


def find_variable_region(seqs: np.ndarray) -> tuple[int, int]:
    arr = np.array([list(s) for s in seqs])
    is_variable = np.any(arr != arr[0:1, :], axis=0)
    variable_positions = np.where(is_variable)[0]
    assert len(variable_positions) > 0, "no variable positions found"
    return int(variable_positions.min()), int(variable_positions.max()) + 1


def pad_at_boundary(seq: str, insert_pos: int, n_pad: int) -> str:
    return seq[:insert_pos] + ("N" * n_pad) + seq[insert_pos:]


def main() -> None:
    design_info = {}
    train_frames, test_frames = [], []
    skipped = []

    print("Checking each design's ACTUAL full_seq string length for uniformity "
          "(the source table's full_len column reflects pre-trimming length, "
          "not necessarily the true per-row trimmed length)...")
    for design in CLEAN_DESIGNS:
        train_df = pd.read_csv(f"{DATA_DIR}/fas_{design}_hek_train.csv.gz")
        test_df = pd.read_csv(f"{DATA_DIR}/fas_{design}_hek_test.csv.gz")

        train_lens = train_df["full_seq"].str.len()
        test_lens = test_df["full_seq"].str.len()
        if train_lens.nunique() != 1 or test_lens.nunique() != 1 or train_lens.iloc[0] != test_lens.iloc[0]:
            print(f"  {design}: SKIPPED - inconsistent actual string length "
                  f"(train nunique={train_lens.nunique()}, test nunique={test_lens.nunique()})")
            skipped.append(design)
            continue

        seq_len = train_lens.iloc[0]

        sample = train_df["full_seq"].sample(n=min(10000, len(train_df)), random_state=0).to_numpy()
        start, end = find_variable_region(sample)

        design_info[design] = {
            "seq_len": int(seq_len),
            "variable_start": start,
            "variable_end": end,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
        print(f"  {design}: seq_len={seq_len}, variable_region=[{start}, {end})")

        train_df["design"] = design
        test_df["design"] = design
        train_frames.append(train_df)
        test_frames.append(test_df)

    target_len = max(info["seq_len"] for info in design_info.values())
    print(f"\nTarget pooled length: {target_len}")

    def pad_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def pad_row(row):
            info = design_info[row["design"]]
            n_pad = target_len - info["seq_len"]
            if n_pad == 0:
                return row["full_seq"]
            return pad_at_boundary(row["full_seq"], info["variable_start"], n_pad)

        df["full_seq"] = df.apply(pad_row, axis=1)
        assert (df["full_seq"].str.len() == target_len).all()
        return df[["design", "var_id", "full_seq", "psi_shrunk"]]

    print("Padding and concatenating train set...")
    pooled_train = pd.concat([pad_frame(df) for df in train_frames], ignore_index=True)
    print("Padding and concatenating test set...")
    pooled_test = pd.concat([pad_frame(df) for df in test_frames], ignore_index=True)

    print(f"\nPooled train: {len(pooled_train)} rows, test: {len(pooled_test)} rows")
    print("\nPer-design counts in pooled train set:")
    print(pooled_train["design"].value_counts().to_string())

    pooled_train.to_csv(f"{DATA_DIR}/fas_pooled11_hek_train.csv.gz", index=False)
    pooled_test.to_csv(f"{DATA_DIR}/fas_pooled11_hek_test.csv.gz", index=False)

    summary_df = pd.DataFrame.from_dict(design_info, orient="index")
    summary_df.index.name = "design"
    summary_df["pad_amount"] = target_len - summary_df["seq_len"]
    summary_df.to_csv(f"{DATA_DIR}/pooled11_design_info.tsv", sep="\t")
    print(f"\nWrote pooled_train/test CSVs and pooled11_design_info.tsv to {DATA_DIR}")


if __name__ == "__main__":
    main()
