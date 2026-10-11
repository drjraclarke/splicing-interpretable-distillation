"""
Test whether our real-data-trained model exhibits the same CpG-enrichment and
stop-codon-depletion "confounds" that the paper found in SpliceAI/Pangolin/
AlphaGenome (which were trained on genomic sequence, not real random-MPRA
ground truth).

Method mirrors the paper's own (Methods: "CpG composition analysis",
"Stop codon analysis"): compute a per-sequence CpG ratio / stop-codon count,
then test association with (model prediction - measured PSI), controlling for
measured PSI, via multiple linear regression (ANCOVA-style; implemented with
plain numpy/scipy rather than statsmodels, which isn't installed in this env).

Both CpG ratio and stop-codon count are computed within the RANDOMIZED region
only (not the fixed flanking exons/introns, which are identical across all
variants of a design and can't drive any real predictive relationship). The
randomized region's bounds are determined empirically: for each position,
check whether all training sequences agree (fixed) or vary (randomized).
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch

from datasets.sequence import SequenceDataset
from experiment.experiment import Experiment

EXPERIMENT_DIR = Path(
    "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/ray_results/fas_exon6_hek_full"
)
TRIAL_ID = "7873bd5b"
TRAIN_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_train.csv.gz"
TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_exon6_hek_test.csv.gz"
OUT_DIR = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data")

STOP_CODONS = {"TAA", "TAG", "TGA"}


def load_model(experiment_dir: Path, trial_id: str, device: torch.device):
    trial_dirs = list(experiment_dir.glob(f"Experiment_{trial_id}*"))
    assert len(trial_dirs) == 1
    trial_dir = trial_dirs[0]

    with open(trial_dir / "params.json") as f:
        raw_config = json.load(f)

    exp = Experiment.__new__(Experiment)
    model, config = exp.parse_config(copy.deepcopy(raw_config))

    checkpoint = torch.load(
        trial_dir / "checkpoint_000000" / "checkpoint.pt", map_location=device
    )
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def find_variable_region(train_csv: str, n_sample: int = 20000) -> tuple[int, int]:
    """
    Determine the (start, end) [start, end) of the randomized region by
    finding which positions differ across a sample of training sequences.
    """
    df = pd.read_csv(train_csv, usecols=["full_seq"]).sample(
        n=n_sample, random_state=0
    )
    seqs = df["full_seq"].to_numpy()
    arr = np.array([list(s) for s in seqs])  # (n_sample, seq_len)
    is_variable = np.any(arr != arr[0:1, :], axis=0)  # (seq_len,)

    variable_positions = np.where(is_variable)[0]
    assert len(variable_positions) > 0, "no variable positions found"
    start, end = variable_positions.min(), variable_positions.max() + 1
    # Sanity check: region should be contiguous (allow a few non-variable
    # positions inside, e.g. positions that happen to be constrained/conserved)
    span = end - start
    print(f"Variable region: [{start}, {end}) (span={span}), {len(variable_positions)} truly-variable positions within it")
    return int(start), int(end)


def cpg_ratio(seq: str) -> float:
    n = len(seq)
    n_c = seq.count("C")
    n_g = seq.count("G")
    n_cpg = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i + 1] == "G")
    if n_c == 0 or n_g == 0:
        return np.nan
    return n_cpg * n / (n_c * n_g)


def stop_codon_count(seq: str) -> int:
    count = 0
    for frame in range(3):
        for i in range(frame, len(seq) - 2, 3):
            if seq[i : i + 3] in STOP_CODONS:
                count += 1
    return count


def ancova_ols(y: np.ndarray, predictor: np.ndarray, covariate: np.ndarray) -> dict:
    """
    Fit y ~ 1 + predictor + covariate via OLS, return the predictor
    coefficient, its standard error, t-statistic and two-sided p-value.
    """
    mask = np.isfinite(y) & np.isfinite(predictor) & np.isfinite(covariate)
    y, predictor, covariate = y[mask], predictor[mask], covariate[mask]

    X = np.column_stack([np.ones_like(predictor), predictor, covariate])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    dof = n - p
    sigma2 = np.sum(residuals**2) / dof
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))

    t_stat = beta[1] / se[1]
    p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), dof))

    return {
        "n": int(n),
        "coefficient": float(beta[1]),
        "se": float(se[1]),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start, end = find_variable_region(TRAIN_CSV)

    print("Loading model...")
    model, config = load_model(EXPERIMENT_DIR, TRIAL_ID, device)

    print("Loading test set...")
    test_df = pd.read_csv(TEST_CSV)

    testset = SequenceDataset(
        data_path=None,
        data_df=test_df,
        sequence_column="full_seq",
        target_column="psi_shrunk",
        upstream_sequence=config.get("upstream_sequence", ""),
        downstream_sequence=config.get("downstream_sequence", ""),
    )
    loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    print("Running inference...")
    all_preds = []
    with torch.no_grad():
        for sequences, _ in loader:
            sequences = sequences.to(device)
            preds = model(sequences).squeeze(-1).cpu()
            all_preds.append(preds)
    predicted_logit = torch.cat(all_preds).numpy()
    predicted_psi = 1 / (1 + np.exp(-predicted_logit))

    test_df["predicted_psi"] = predicted_psi
    test_df["error"] = test_df["predicted_psi"] - test_df["psi_shrunk"]
    test_df["variable_region"] = test_df["full_seq"].str.slice(start, end)
    test_df["cpg_ratio"] = test_df["variable_region"].map(cpg_ratio)
    test_df["stop_codon_count"] = test_df["variable_region"].map(stop_codon_count)

    print(f"\nn_test = {len(test_df)}")
    print(f"Overall pearson-r (predicted vs measured) = {np.corrcoef(test_df['predicted_psi'], test_df['psi_shrunk'])[0,1]:.4f}")

    print("\n=== CpG confound test: error ~ cpg_ratio + measured_psi ===")
    cpg_result = ancova_ols(
        test_df["error"].to_numpy(),
        test_df["cpg_ratio"].to_numpy(),
        test_df["psi_shrunk"].to_numpy(),
    )
    print(cpg_result)

    print("\n=== Stop-codon confound test: error ~ stop_codon_count + measured_psi ===")
    stop_result = ancova_ols(
        test_df["error"].to_numpy(),
        test_df["stop_codon_count"].to_numpy().astype(float),
        test_df["psi_shrunk"].to_numpy(),
    )
    print(stop_result)

    out_csv = OUT_DIR / "fas_exon6_hek_test_confound_analysis.csv"
    test_df[
        ["var_id", "psi_shrunk", "predicted_psi", "error", "cpg_ratio", "stop_codon_count"]
    ].to_csv(out_csv, index=False)
    print(f"\nWrote per-variant confound analysis table to {out_csv}")

    summary = {
        "variable_region_start": start,
        "variable_region_end": end,
        "cpg_confound": cpg_result,
        "stop_codon_confound": stop_result,
    }
    with open(OUT_DIR / "confound_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
