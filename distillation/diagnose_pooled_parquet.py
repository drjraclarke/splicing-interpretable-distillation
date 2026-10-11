"""
Inspect the pre-built pooled_all_designs_psi_and_sequence.parquet: schema,
per-design row counts, and per-design full_seq length consistency (the exact
issue that broke the from-scratch pooling attempt on the raw source tables).
"""

import pandas as pd

PARQUET = (
    "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning/"
    "input_data/random_mpra_metadata/pooled_all_designs_psi_and_sequence/"
    "pooled_all_designs_psi_and_sequence.parquet"
)


def main() -> None:
    print("Reading parquet...")
    df = pd.read_parquet(PARQUET)
    print(f"Shape: {df.shape}")
    print(f"Columns and dtypes:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isna().sum()}")

    has_seq = df["full_seq"].notna()
    print(f"\nRows with non-null full_seq: {has_seq.sum()} / {len(df)}")

    df["seq_len"] = df["full_seq"].str.len()

    print("\nPer-design sequence length stats (rows with sequence):")
    stats = (
        df[has_seq]
        .groupby("design")["seq_len"]
        .agg(["min", "max", "nunique", "count"])
        .sort_values("count", ascending=False)
    )
    print(stats.to_string())

    print("\nPer-design PSI availability (non-null counts):")
    psi_avail = df.groupby("design")[["psi_hek", "psi_mcf7", "psi_shsy"]].apply(
        lambda g: g.notna().sum()
    )
    print(psi_avail.to_string())

    # Check case (upper/lower) - does this table need the same uppercase fix?
    sample_seq = df.loc[has_seq, "full_seq"].iloc[0]
    print(f"\nSample sequence (first with seq): {sample_seq[:80]}...")
    print(f"Contains lowercase: {any(c.islower() for c in sample_seq)}")


if __name__ == "__main__":
    main()
