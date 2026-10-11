"""
Quantify whether our real-data-trained model's CpG/stop-codon confound bias
is smaller or larger than the genome-trained black-box models' bias, on the
exact same 34,741 test variants and exact same CpG-ratio/stop-codon-count
features (computed once, from our own confound analysis).

Runs the identical ANCOVA-style regression (error ~ confound + measured_psi)
used for our own model on each zero-shot model's predictions, so the
coefficients are directly comparable. Also reports a standardized
(z-scored-predictor) coefficient, since raw coefficients can be misleading if
a model's prediction variance differs.
"""

import numpy as np
import pandas as pd
import scipy.stats

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


def ancova_ols(y: np.ndarray, predictor: np.ndarray, covariate: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(predictor) & np.isfinite(covariate)
    y, predictor, covariate = y[mask], predictor[mask], covariate[mask]

    # Standardized version: z-score the predictor so coefficients are
    # comparable in units of "per 1 SD of the confound feature", regardless
    # of each model's own prediction-error variance.
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
        return beta[1], se[1], t_stat, p_value

    raw_coef, raw_se, raw_t, raw_p = fit(predictor)
    z_coef, z_se, z_t, z_p = fit(predictor_z)

    return {
        "n": int(mask.sum()),
        "raw_coefficient": raw_coef,
        "raw_p_value": raw_p,
        "standardized_coefficient": z_coef,
        "standardized_p_value": z_p,
    }


def main() -> None:
    print("Loading our confound features (cpg_ratio, stop_codon_count) and measured PSI...")
    ours = pd.read_csv(OUR_CONFOUND_CSV)
    features = ours[["var_id", "psi_shrunk", "cpg_ratio", "stop_codon_count"]]

    results = []

    # Our own model, for reference (same numbers as before, recomputed here for a unified table)
    our_result_cpg = ancova_ols(ours["error"].to_numpy(), ours["cpg_ratio"].to_numpy(), ours["psi_shrunk"].to_numpy())
    our_result_stop = ancova_ols(ours["error"].to_numpy(), ours["stop_codon_count"].to_numpy().astype(float), ours["psi_shrunk"].to_numpy())
    results.append({"model": "OUR_MODEL", "category": "real_data_trained", "confound": "cpg_ratio", **our_result_cpg})
    results.append({"model": "OUR_MODEL", "category": "real_data_trained", "confound": "stop_codon_count", **our_result_stop})

    def process(file_path, sep, label):
        print(f"\nLoading {label}...")
        df = pd.read_csv(file_path, sep=sep)
        df = df[df["design"] == "Exon_6"]
        # Drop any measured-PSI column this file brings along - always use our
        # own psi_shrunk (from `features`) as the single canonical source, to
        # avoid a silent column-name collision on merge.
        df = df.drop(columns=[c for c in df.columns if c != "pred" and c in features.columns and c != "var_id"])
        merged = df.merge(features, on="var_id", how="inner")
        assert "psi_shrunk" in merged.columns, f"merge lost psi_shrunk, columns: {merged.columns.tolist()}"
        for model_name, group in merged.groupby("model"):
            if model_name not in ZERO_SHOT_MODELS:
                continue
            group = group.dropna(subset=["pred", "psi_shrunk"])
            error = group["pred"].to_numpy() - group["psi_shrunk"].to_numpy()
            cpg_result = ancova_ols(error, group["cpg_ratio"].to_numpy(), group["psi_shrunk"].to_numpy())
            stop_result = ancova_ols(error, group["stop_codon_count"].to_numpy().astype(float), group["psi_shrunk"].to_numpy())
            results.append({"model": model_name, "category": ZERO_SHOT_MODELS[model_name], "confound": "cpg_ratio", **cpg_result})
            results.append({"model": model_name, "category": ZERO_SHOT_MODELS[model_name], "confound": "stop_codon_count", **stop_result})

    process(SPLICEAI_FILE, "\t", "SpliceAI/OpenSpliceAI")
    process(ALPHAGENOME_PT_FILE, "\t", "AlphaGenome (PyTorch)")
    process(ALPHAGENOME_JAX_FILE, "\t", "AlphaGenome (JAX)")

    results_df = pd.DataFrame(results)

    print("\n\n=== CpG confound magnitude, all models (same test variants, same feature) ===")
    cpg_df = results_df[results_df["confound"] == "cpg_ratio"].copy()
    cpg_df["abs_standardized_coef"] = cpg_df["standardized_coefficient"].abs()
    cpg_df = cpg_df.sort_values("abs_standardized_coef")
    print(cpg_df[["model", "category", "n", "raw_coefficient", "standardized_coefficient", "raw_p_value"]].to_string(index=False))

    print("\n=== Stop-codon confound magnitude, all models (same test variants, same feature) ===")
    stop_df = results_df[results_df["confound"] == "stop_codon_count"].copy()
    stop_df["abs_standardized_coef"] = stop_df["standardized_coefficient"].abs()
    stop_df = stop_df.sort_values("abs_standardized_coef")
    print(stop_df[["model", "category", "n", "raw_coefficient", "standardized_coefficient", "raw_p_value"]].to_string(index=False))

    out_path = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/confound_magnitude_comparison.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
