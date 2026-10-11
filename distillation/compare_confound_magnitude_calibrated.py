"""
Re-run the CpG/stop-codon confound magnitude comparison using the paper's own
exact methodology: calibrate each model's raw predictions to measured PSI via
isotonic regression (on the logit scale, PAVA algorithm) BEFORE computing
prediction error, rather than comparing raw predictions directly.

The raw oracle predictions (SpliceAI/OpenSpliceAI, AlphaGenome) are the mean
of acceptor/donor scores - the paper's own "exon score" definition - but we
had not applied their subsequent calibration step. This adds it, for every
model including ours, so the comparison matches their approach exactly.

Calibration is fit "on the entire dataset" (the same test set being scored),
matching the paper's own stated approach (Methods: "calibrated ... on the
entire dataset") - this is a real limitation (in-sample calibration) that the
paper itself has too, not something we're doing differently.
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression

OUR_CONFOUND_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test_confound_analysis.csv"

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


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrate_isotonic(pred: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Fit isotonic regression pred(logit-scale) -> measured, on the same data
    (matching the paper's "calibrated ... on the entire dataset" approach),
    return calibrated predictions on the original [0,1] PSI scale."""
    pred_logit = logit(pred)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred_logit, measured)
    return iso.predict(pred_logit)


def ancova_ols(y: np.ndarray, predictor: np.ndarray, covariate: np.ndarray) -> dict:
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


def main() -> None:
    print("Loading our confound features...")
    ours = pd.read_csv(OUR_CONFOUND_CSV)
    features = ours[["var_id", "psi_shrunk", "cpg_ratio", "stop_codon_count"]]

    results = []

    # Our model: calibrate too, for full methodological consistency, even
    # though we expect little change (trained directly on this PSI scale).
    our_calibrated = calibrate_isotonic(ours["predicted_psi"].to_numpy(), ours["psi_shrunk"].to_numpy())
    our_error = our_calibrated - ours["psi_shrunk"].to_numpy()
    cpg_r = ancova_ols(our_error, ours["cpg_ratio"].to_numpy(), ours["psi_shrunk"].to_numpy())
    stop_r = ancova_ols(our_error, ours["stop_codon_count"].to_numpy().astype(float), ours["psi_shrunk"].to_numpy())
    results.append({"model": "OUR_MODEL", "category": "real_data_trained", "confound": "cpg_ratio", **cpg_r})
    results.append({"model": "OUR_MODEL", "category": "real_data_trained", "confound": "stop_codon_count", **stop_r})

    # Also report calibration's effect on overall correlation, as a sanity check
    raw_r = scipy.stats.pearsonr(ours["predicted_psi"], ours["psi_shrunk"]).statistic
    calibrated_r = scipy.stats.pearsonr(our_calibrated, ours["psi_shrunk"]).statistic
    print(f"Our model: raw pearson_r={raw_r:.4f}, calibrated pearson_r={calibrated_r:.4f}")

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
            error = calibrated - group["psi_shrunk"].to_numpy()
            cpg_r = ancova_ols(error, group["cpg_ratio"].to_numpy(), group["psi_shrunk"].to_numpy())
            stop_r = ancova_ols(error, group["stop_codon_count"].to_numpy().astype(float), group["psi_shrunk"].to_numpy())
            results.append({"model": model_name, "category": ZERO_SHOT_MODELS[model_name], "confound": "cpg_ratio", **cpg_r})
            results.append({"model": model_name, "category": ZERO_SHOT_MODELS[model_name], "confound": "stop_codon_count", **stop_r})

    process(SPLICEAI_FILE, "SpliceAI/OpenSpliceAI")
    process(ALPHAGENOME_PT_FILE, "AlphaGenome (PyTorch)")
    process(ALPHAGENOME_JAX_FILE, "AlphaGenome (JAX)")

    results_df = pd.DataFrame(results)

    print("\n\n=== CpG confound magnitude, CALIBRATED predictions (paper's exact methodology) ===")
    cpg_df = results_df[results_df["confound"] == "cpg_ratio"].copy()
    cpg_df["abs_std"] = cpg_df["standardized_coefficient"].abs()
    print(cpg_df.sort_values("abs_std")[["model", "category", "n", "raw_coefficient", "standardized_coefficient", "p_value"]].to_string(index=False))

    print("\n=== Stop-codon confound magnitude, CALIBRATED predictions ===")
    stop_df = results_df[results_df["confound"] == "stop_codon_count"].copy()
    stop_df["abs_std"] = stop_df["standardized_coefficient"].abs()
    print(stop_df.sort_values("abs_std")[["model", "category", "n", "raw_coefficient", "standardized_coefficient", "p_value"]].to_string(index=False))

    out_path = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/confound_magnitude_comparison_calibrated.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
