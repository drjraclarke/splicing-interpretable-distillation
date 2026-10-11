"""
CpG/GpC/stop-codon confound analysis for the POOLED 16-design model (trained
on all designs, not just Exon_6), using the paper's exact methodology
(isotonic calibration on the logit scale, then ANCOVA), to check whether the
same confounds found in the single-design model reproduce when the model is
trained on much more, more diverse real data.

Each design has its own variable-region location (see pooled16_design_info.tsv),
and that location shifts by `pad_amount` once padding is inserted (padding was
inserted exactly at variable_start in build_pooled_dataset_v2.py, so content
from variable_start onward - including the variable region itself - shifts
right by pad_amount in the final padded sequence used for training/testing).

Reports two scopes: all 16 designs pooled, and restricted to the 8 designs
where the model actually generalized (pearson_r > 0.5 on held-out data per
fas_pooled16_hek_test_per_design.tsv) - since confound coefficients computed
on designs where the model learned nothing meaningful (near-zero r) are not
very interpretable (residual "error" there is mostly just noise).
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression

DATA_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data"
PREDICTIONS_CSV = f"{DATA_DIR}/fas_pooled16_hek_test_predictions.csv"
TEST_CSV = f"{DATA_DIR}/fas_pooled16_hek_test.csv.gz"
DESIGN_INFO_TSV = f"{DATA_DIR}/pooled16_design_info.tsv"
PER_DESIGN_TSV = f"{DATA_DIR}/fas_pooled16_hek_test_per_design.tsv"

EPS = 1e-6
STOP_CODONS = {"TAA", "TAG", "TGA"}
GOOD_DESIGN_R_THRESHOLD = 0.5


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
    print("Loading pooled model predictions and test sequences...")
    preds = pd.read_csv(PREDICTIONS_CSV)  # design, var_id, psi_shrunk, predicted_psi
    seqs = pd.read_csv(TEST_CSV, usecols=["design", "var_id", "full_seq"])
    design_info = pd.read_csv(DESIGN_INFO_TSV, sep="\t")
    per_design_perf = pd.read_csv(PER_DESIGN_TSV, sep="\t")

    df = preds.merge(seqs, on=["design", "var_id"], how="inner")
    df = df.merge(design_info[["design", "variable_start", "variable_end", "pad_amount"]], on="design", how="left")
    print(f"Merged rows: {len(df)}")

    print("Extracting each row's variable region (adjusted for padding shift)...")
    df["adj_start"] = df["variable_start"] + df["pad_amount"]
    df["adj_end"] = df["variable_end"] + df["pad_amount"]
    df["variable_region"] = df.apply(lambda r: r["full_seq"][r["adj_start"]:r["adj_end"]], axis=1)

    print("Computing confound features...")
    df["cpg_ratio"] = df["variable_region"].map(cpg_ratio)
    df["cpg_count"] = df["variable_region"].map(cpg_count)
    df["gpc_count"] = df["variable_region"].map(gpc_count)
    df["stop_codon_count"] = df["variable_region"].map(stop_codon_count)

    print("Calibrating predictions (isotonic, logit scale, fit across the whole pooled test set)...")
    df["predicted_calibrated"] = calibrate_isotonic(df["predicted_psi"].to_numpy(), df["psi_shrunk"].to_numpy())
    df["error"] = df["predicted_calibrated"] - df["psi_shrunk"]

    raw_r = np.corrcoef(df["predicted_psi"], df["psi_shrunk"])[0, 1]
    cal_r = np.corrcoef(df["predicted_calibrated"], df["psi_shrunk"])[0, 1]
    print(f"Overall: raw pearson_r={raw_r:.4f}, calibrated pearson_r={cal_r:.4f}")

    good_designs = per_design_perf[per_design_perf["pearson_r"] > GOOD_DESIGN_R_THRESHOLD]["design"].tolist()
    print(f"\nDesigns with pearson_r > {GOOD_DESIGN_R_THRESHOLD} on held-out data: {good_designs}")

    results = []
    for scope_name, scope_df in [("all_16_designs", df), ("well_generalizing_designs_only", df[df["design"].isin(good_designs)])]:
        for feat in FEATURES:
            r = ancova_ols(scope_df["error"].to_numpy(), scope_df[feat].to_numpy().astype(float), scope_df["psi_shrunk"].to_numpy())
            results.append({"scope": scope_name, "feature": feat, **r})

    results_df = pd.DataFrame(results)
    for scope in results_df["scope"].unique():
        print(f"\n=== {scope} ===")
        print(results_df[results_df["scope"] == scope][["feature", "n", "raw_coefficient", "standardized_coefficient", "p_value"]].to_string(index=False))

    print("\n\n=== Comparison to earlier Exon_6-only model (calibrated ANCOVA results) ===")
    print("Exon_6-only model: cpg_ratio std_coef=0.0148, cpg_count std_coef=0.0381, gpc_count std_coef=0.0238, stop_codon_count std_coef=-0.0319")

    out_path = f"{DATA_DIR}/confound_analysis_pooled_model.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    df[["design", "var_id", "psi_shrunk", "predicted_psi", "predicted_calibrated", "error",
        "cpg_ratio", "cpg_count", "gpc_count", "stop_codon_count"]].to_csv(
        f"{DATA_DIR}/confound_per_variant_pooled_model.csv", index=False
    )
    print(f"\nWrote {out_path} and confound_per_variant_pooled_model.csv")


if __name__ == "__main__":
    main()
