"""
Full methodology check against the paper's exact CpG analysis:
1. CpG ratio (general formula, used for all assays except FAS exon6 in the paper)
2. CpG raw count (paper's specific choice for FAS exon6, due to "limited
   compositional diversity" in their near-WT mutagenesis assay - doesn't
   obviously apply to our fully-randomized library, but checked for rigor)
3. GpC control (their specificity check - GpC should NOT show the same bias,
   confirming it's CpG-specific rather than a generic dinucleotide/GC effect)

All three run as ANCOVA (error ~ feature + measured_psi) on CALIBRATED
predictions (isotonic regression, matching the paper's Methods), for our
model and for the zero-shot/fine-tuned oracle predictions on the exact same
test variants.
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression

OUR_CONFOUND_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test_confound_analysis.csv"
OUR_TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test.csv.gz"

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
VAR_START, VAR_END = 234, 297  # our empirically-determined variable region


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrate_isotonic(pred, measured):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(logit(pred), measured)
    return iso.predict(logit(pred))


def gpc_count(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "G" and seq[i + 1] == "C")


def cpg_count(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G")


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


def main() -> None:
    print("Loading our test sequences (for raw CpG count + GpC features)...")
    test_seqs = pd.read_csv(OUR_TEST_CSV)
    test_seqs["variable_region"] = test_seqs["full_seq"].str.slice(VAR_START, VAR_END)
    test_seqs["cpg_count"] = test_seqs["variable_region"].map(cpg_count)
    test_seqs["gpc_count"] = test_seqs["variable_region"].map(gpc_count)

    ours = pd.read_csv(OUR_CONFOUND_CSV)
    ours = ours.merge(test_seqs[["var_id", "cpg_count", "gpc_count"]], on="var_id")

    features = ours[["var_id", "psi_shrunk", "cpg_ratio", "cpg_count", "gpc_count"]]

    results = []

    def run_all_features(model_name, category, error, feat_df):
        for feat_name in ["cpg_ratio", "cpg_count", "gpc_count"]:
            r = ancova_ols(error, feat_df[feat_name].to_numpy().astype(float), feat_df["psi_shrunk"].to_numpy())
            results.append({"model": model_name, "category": category, "feature": feat_name, **r})

    # Our model (already-computed calibrated predictions from the earlier script's logic - recompute here for full feature set)
    our_calibrated = calibrate_isotonic(ours["predicted_psi"].to_numpy(), ours["psi_shrunk"].to_numpy())
    our_error = our_calibrated - ours["psi_shrunk"].to_numpy()
    run_all_features("OUR_MODEL", "real_data_trained", our_error, ours)

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
            run_all_features(model_name, ZERO_SHOT_MODELS[model_name], error, group)

    process(SPLICEAI_FILE, "SpliceAI/OpenSpliceAI")
    process(ALPHAGENOME_PT_FILE, "AlphaGenome (PyTorch)")
    process(ALPHAGENOME_JAX_FILE, "AlphaGenome (JAX)")

    results_df = pd.DataFrame(results)

    for feat in ["cpg_ratio", "cpg_count", "gpc_count"]:
        print(f"\n\n=== {feat} (calibrated predictions) ===")
        sub = results_df[results_df["feature"] == feat].copy()
        sub["abs_std"] = sub["standardized_coefficient"].abs()
        print(sub.sort_values("abs_std")[["model", "category", "n", "raw_coefficient", "standardized_coefficient", "p_value"]].to_string(index=False))

    out_path = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/confound_full_methodology_check.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")

    print("\n\n=== Paper's own reported significance for reference (text only reports p-value thresholds, no coefficients) ===")
    print("CpG: ANCOVA p < 5e-3 (all 3 models)")
    print("Stop codon: ANCOVA p < 1e-20 (all 3 models)")
    print("GpC: 'did not observe a consistent association' (their control - should be non-significant/weak)")


if __name__ == "__main__":
    main()
