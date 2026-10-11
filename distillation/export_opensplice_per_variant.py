"""
Export per-variant OpenSplice data (pretrained vs stage5 OpenSpliceAI,
both SNV and deletion mutation types), with confound features and calibrated
error, for handover/plotting purposes.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

MASTER_TABLE = "/nfs/users/nfs_j/jc58/repos/OpenSplice/data/processed/open_splice_experimental_master_unique_sequence_all_predictor_cols.tsv.gz"
PRED_DIR = "/nfs/users/nfs_j/jc58/repos/alphagenome_FT_MPRA/results/opensplice_openspliceai_export"
OUT_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/opensplice_per_variant_pretrained_vs_stage5.csv"

MODELS = {
    "pretrained": "openspliceai_pretrained_predictions",
    "stage5": "openspliceai_stage5_sl650_selected_predictions",
}
MUT_TYPES = {"snv": "opensplice_snv_allexons", "del": "opensplice_del_allexons"}
EPS = 1e-6
STOP_CODONS = {"TAA", "TAG", "TGA"}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrate_isotonic(pred, measured):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(logit(pred), measured)
    return iso.predict(logit(pred))


def cpg_count(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G")


def gpc_count(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "G" and seq[i + 1] == "C")


def cpg_ratio(seq):
    n_c, n_g = seq.count("C"), seq.count("G")
    if n_c == 0 or n_g == 0:
        return np.nan
    return cpg_count(seq) * len(seq) / (n_c * n_g)


def stop_codon_count(seq):
    count = 0
    for frame in range(3):
        for i in range(frame, len(seq) - 2, 3):
            if seq[i : i + 3] in STOP_CODONS:
                count += 1
    return count


def main() -> None:
    print("Loading master table...")
    master = pd.read_csv(MASTER_TABLE, sep="\t", usecols=["variant_id", "nt_seq", "gene", "exon_id"])
    master = master.rename(columns={"variant_id": "var_id"})
    master["cpg_ratio"] = master["nt_seq"].map(cpg_ratio)
    master["cpg_count"] = master["nt_seq"].map(cpg_count)
    master["gpc_count"] = master["nt_seq"].map(gpc_count)
    master["stop_codon_count"] = master["nt_seq"].map(stop_codon_count)

    all_rows = []
    for mut_type, mut_prefix in MUT_TYPES.items():
        for model_name, model_suffix in MODELS.items():
            path = f"{PRED_DIR}/{mut_prefix}_{model_suffix}.tsv.gz"
            preds = pd.read_csv(path, sep="\t")
            test_preds = preds[preds["split"] == "test"].dropna(subset=["pred_score", "psi_shrunk"])
            test_preds = test_preds.drop(columns=[c for c in test_preds.columns if c != "var_id" and c in master.columns])
            merged = test_preds.merge(master, on="var_id", how="inner")

            calibrated = calibrate_isotonic(merged["pred_score"].to_numpy(), merged["psi_shrunk"].to_numpy())
            merged["predicted_calibrated"] = calibrated
            merged["error"] = calibrated - merged["psi_shrunk"]
            merged["mut_type"] = mut_type
            merged["model"] = model_name
            all_rows.append(merged[["mut_type", "model", "var_id", "gene", "exon_id", "psi_shrunk",
                                     "pred_score", "predicted_calibrated", "error",
                                     "cpg_ratio", "cpg_count", "gpc_count", "stop_codon_count"]])

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
