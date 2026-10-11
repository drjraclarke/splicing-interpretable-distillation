"""
CpG/GpC/stop-codon confound analysis on the OpenSplice dataset (lehner-lab/
OpenSplice, distinct from OpenSpliceAI) - a genuinely independent real
experimental MPRA dataset (608 natural exons, SNV + deletion mutagenesis,
HEK293), NOT the FAS-family random-sequence library our own models were
trained/tested on. Tests OpenSpliceAI pretrained vs. three progressive
fine-tuning stages (stage1/2/5), using already-existing predictions restricted
to the held-out test split (no fresh inference needed).

Same methodology as before: isotonic-regression calibration (logit scale),
then ANCOVA (error ~ feature + measured_psi).
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression

MASTER_TABLE = "/nfs/users/nfs_j/jc58/repos/OpenSplice/data/processed/open_splice_experimental_master_unique_sequence_all_predictor_cols.tsv.gz"
PRED_DIR = "/nfs/users/nfs_j/jc58/repos/alphagenome_FT_MPRA/results/opensplice_openspliceai_export"

MODELS = {
    "pretrained": "openspliceai_pretrained_predictions",
    "stage1": "openspliceai_stage1_sl650_highlam_selected_predictions",
    "stage2": "openspliceai_stage2_sl650_highlam_selected_predictions",
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


def ancova_ols(y, predictor, covariate):
    mask = np.isfinite(y) & np.isfinite(predictor) & np.isfinite(covariate)
    y, predictor, covariate = y[mask], predictor[mask], covariate[mask]
    if mask.sum() < 20:
        return {"n": int(mask.sum()), "raw_coefficient": np.nan, "standardized_coefficient": np.nan, "p_value": np.nan}
    predictor_z = (predictor - predictor.mean()) / predictor.std()

    def fit(x):
        X = np.column_stack([np.ones_like(x), x, covariate])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        n, p = X.shape
        dof = n - p
        sigma2 = np.sum(residuals**2) / dof
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        t_stat = beta[1] / se[1]
        p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), dof))
        return beta[1], p_value

    raw_coef, raw_p = fit(predictor)
    z_coef, z_p = fit(predictor_z)
    return {"n": int(mask.sum()), "raw_coefficient": raw_coef, "standardized_coefficient": z_coef, "p_value": raw_p}


FEATURES = ["cpg_ratio", "cpg_count", "gpc_count", "stop_codon_count"]


def main() -> None:
    print("Loading master table (variant_id, nt_seq only)...")
    master = pd.read_csv(MASTER_TABLE, sep="\t", usecols=["variant_id", "nt_seq"])
    master = master.rename(columns={"variant_id": "var_id"})
    master["cpg_ratio"] = master["nt_seq"].map(cpg_ratio)
    master["cpg_count"] = master["nt_seq"].map(cpg_count)
    master["gpc_count"] = master["nt_seq"].map(gpc_count)
    master["stop_codon_count"] = master["nt_seq"].map(stop_codon_count)
    print(f"Master table: {len(master)} unique variants")

    results = []
    for mut_type, mut_prefix in MUT_TYPES.items():
        for model_name, model_suffix in MODELS.items():
            path = f"{PRED_DIR}/{mut_prefix}_{model_suffix}.tsv.gz"
            print(f"\nLoading {mut_type}/{model_name}: {path}")
            preds = pd.read_csv(path, sep="\t")
            test_preds = preds[preds["split"] == "test"].dropna(subset=["pred_score", "psi_shrunk"])
            print(f"  test rows: {len(test_preds)} / {len(preds)} total")

            merged = test_preds.merge(master, on="var_id", how="inner")
            print(f"  merged with sequence: {len(merged)}")

            calibrated = calibrate_isotonic(merged["pred_score"].to_numpy(), merged["psi_shrunk"].to_numpy())
            error = calibrated - merged["psi_shrunk"].to_numpy()
            raw_r = np.corrcoef(merged["pred_score"], merged["psi_shrunk"])[0, 1]
            cal_r = np.corrcoef(calibrated, merged["psi_shrunk"])[0, 1]
            print(f"  raw pearson_r={raw_r:.4f}, calibrated pearson_r={cal_r:.4f}")

            for feat in FEATURES:
                r = ancova_ols(error, merged[feat].to_numpy().astype(float), merged["psi_shrunk"].to_numpy())
                results.append({"mut_type": mut_type, "model": model_name, "feature": feat,
                                 "pearson_r_calibrated": cal_r, **r})

    results_df = pd.DataFrame(results)
    for mut_type in MUT_TYPES:
        for feat in FEATURES:
            print(f"\n=== {mut_type} / {feat} across fine-tuning stages ===")
            sub = results_df[(results_df["mut_type"] == mut_type) & (results_df["feature"] == feat)]
            print(sub[["model", "n", "pearson_r_calibrated", "raw_coefficient", "standardized_coefficient", "p_value"]].to_string(index=False))

    out_path = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/confound_analysis_opensplice.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
