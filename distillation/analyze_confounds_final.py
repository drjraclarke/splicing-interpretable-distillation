"""
Final, consolidated confound analysis: CpG (ratio + paper's FAS-exon6-specific
raw count), GpC (specificity control), and stop-codon count - all computed
with the paper's exact methodology (isotonic-regression calibration on the
logit scale, then ANCOVA: error ~ feature + measured_psi), for our model and
every zero-shot/fine-tuned oracle, on the exact same test variants.

Supersedes the earlier separate/partial scripts (analyze_confounds.py's
uncalibrated version, compare_confound_magnitude(_calibrated).py which only
had cpg_ratio+stop_codon, analyze_confounds_full_methodology.py which only
had the CpG family) - this is the single up-to-date source of truth.
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression

OUR_TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test.csv.gz"
# Has var_id + predicted_psi already correctly paired (computed together in
# the same run) - avoids re-attaching predictions to var_id via row order
# across separate files, which would be an alignment risk.
OUR_PREDICTIONS_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test_confound_analysis.csv"

BASE = "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning"
SPLICEAI_FILE = f"{BASE}/results/benchmarking/mpra_inference_sl650_u16_final_fulltest/canonical_benchmark_ALLSITES_FIXED/psi/psi_scores_long.tsv.gz"
ALPHAGENOME_PT_FILE = f"{BASE}/alphagenome_inference/comparison_plots/psi/psi_scores_long.tsv.gz"
ALPHAGENOME_JAX_FILE = f"{BASE}/alphagenome_jax_inference/comparison_plots_test_split/psi/psi_scores_long.tsv.gz"

ZERO_SHOT_MODELS = {
    "pretrained_ensemble": "zero_shot",
    "AlphaGenome_ss": "zero_shot",
    "AlphaGenome_usage": "zero_shot",
    "AlphaGenomeJAX_ss": "zero_shot",
    "AlphaGenomeJAX_usage": "zero_shot",
    "AlphaGenomeJAX_jxn": "zero_shot",
    "sl650_u16_ensemble": "fine_tuned_reference",
}

EPS = 1e-6
VAR_START, VAR_END = 234, 297  # empirically-determined variable/randomized region
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
        return beta[1], se[1], p_value

    raw_coef, raw_se, raw_p = fit(predictor)
    z_coef, z_se, z_p = fit(predictor_z)
    return {
        "n": int(mask.sum()),
        "raw_coefficient": raw_coef,
        "raw_se": raw_se,
        "standardized_coefficient": z_coef,
        "p_value": raw_p,
    }


FEATURES = ["cpg_ratio", "cpg_count", "gpc_count", "stop_codon_count"]


def main() -> None:
    print("Building per-variant confound features from test sequences...")
    test_df = pd.read_csv(OUR_TEST_CSV)
    test_df["variable_region"] = test_df["full_seq"].str.slice(VAR_START, VAR_END)
    test_df["cpg_ratio"] = test_df["variable_region"].map(cpg_ratio)
    test_df["cpg_count"] = test_df["variable_region"].map(cpg_count)
    test_df["gpc_count"] = test_df["variable_region"].map(gpc_count)
    test_df["stop_codon_count"] = test_df["variable_region"].map(stop_codon_count)
    features = test_df[["var_id", "psi_shrunk", "cpg_ratio", "cpg_count", "gpc_count", "stop_codon_count"]]

    per_variant_rows = test_df[
        ["var_id", "psi_shrunk", "cpg_ratio", "cpg_count", "gpc_count", "stop_codon_count"]
    ].copy()

    results = []

    def run_all_features(model_name, category, calibrated_pred, measured, feat_df):
        error = calibrated_pred - measured
        for feat_name in FEATURES:
            r = ancova_ols(error, feat_df[feat_name].to_numpy().astype(float), measured)
            results.append({"model": model_name, "category": category, "feature": feat_name, **r})
        return error

    # Our model - merge on var_id (not row order) for safety
    print("Scoring our model...")
    our_preds = pd.read_csv(OUR_PREDICTIONS_CSV)[["var_id", "predicted_psi"]]
    ours = features.merge(our_preds, on="var_id", how="inner")
    assert len(ours) == len(features), "lost rows merging predictions onto features"
    our_calibrated = calibrate_isotonic(ours["predicted_psi"].to_numpy(), ours["psi_shrunk"].to_numpy())
    our_error = run_all_features("OUR_MODEL", "real_data_trained", our_calibrated, ours["psi_shrunk"].to_numpy(), ours)
    per_variant_rows = per_variant_rows.merge(
        pd.DataFrame({"var_id": ours["var_id"], "our_model_error_calibrated": our_error}), on="var_id", how="left"
    )

    def process(file_path, label):
        print(f"\nLoading {label}...")
        df = pd.read_csv(file_path, sep="\t")
        df = df[df["design"] == "Exon_6"]
        df = df.drop(columns=[c for c in df.columns if c != "pred" and c in features.columns and c != "var_id"])
        merged = df.merge(features, on="var_id", how="inner")
        for model_name, group in merged.groupby("model"):
            if model_name not in ZERO_SHOT_MODELS:
                continue
            group = group.dropna(subset=["pred", "psi_shrunk"])
            calibrated = calibrate_isotonic(group["pred"].to_numpy(), group["psi_shrunk"].to_numpy())
            run_all_features(model_name, ZERO_SHOT_MODELS[model_name], calibrated, group["psi_shrunk"].to_numpy(), group)

    process(SPLICEAI_FILE, "SpliceAI/OpenSpliceAI")
    process(ALPHAGENOME_PT_FILE, "AlphaGenome (PyTorch)")
    process(ALPHAGENOME_JAX_FILE, "AlphaGenome (JAX)")

    results_df = pd.DataFrame(results)

    for feat in FEATURES:
        print(f"\n\n=== {feat} (calibrated predictions, paper's exact ANCOVA methodology) ===")
        sub = results_df[results_df["feature"] == feat].copy()
        sub["abs_std"] = sub["standardized_coefficient"].abs()
        print(sub.sort_values("abs_std")[["model", "category", "n", "raw_coefficient", "standardized_coefficient", "p_value"]].to_string(index=False))

    OUT_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data"
    results_df.to_csv(f"{OUT_DIR}/confound_analysis_final.tsv", sep="\t", index=False)
    per_variant_rows.to_csv(f"{OUT_DIR}/confound_per_variant_final.csv", index=False)
    print(f"\nWrote confound_analysis_final.tsv and confound_per_variant_final.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
