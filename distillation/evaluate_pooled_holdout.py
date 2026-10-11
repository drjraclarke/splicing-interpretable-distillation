"""
Evaluate the pooled 16-design model's best-so-far trial on the genuinely
held-out test set (fas_pooled16_hek_test.csv.gz, 384,199 variants, never
touched during training), overall and broken down per design.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.sequence import SequenceDataset
from experiment.experiment import Experiment
from metrics import metrics

EXPERIMENT_DIR = Path(
    "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/ray_results/fas_pooled16_hek_full_v2"
)
TRIAL_ID = "5f6b62b8"
TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_pooled16_hek_test.csv.gz"
OUT_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_pooled16_hek_test_predictions.csv"


def load_model(experiment_dir: Path, trial_id: str, device: torch.device):
    trial_dirs = list(experiment_dir.glob(f"Experiment_{trial_id}*"))
    assert len(trial_dirs) == 1, f"expected exactly one trial dir, found {trial_dirs}"
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    overall_pearson = metrics.METRICS["pearson-r"](
        torch.tensor(test_df["predicted_psi"].to_numpy()), torch.tensor(test_df["psi_shrunk"].to_numpy())
    )
    overall_spearman = pd.Series(test_df["predicted_psi"]).corr(pd.Series(test_df["psi_shrunk"]), method="spearman")
    print(f"\nOVERALL: n = {len(test_df)}, pearson_r = {overall_pearson:.4f}, spearman_r = {overall_spearman:.4f}")

    print("\nPer-design breakdown:")
    rows = []
    for design, group in test_df.groupby("design"):
        if len(group) < 10:
            continue
        r = np.corrcoef(group["predicted_psi"], group["psi_shrunk"])[0, 1]
        rho = group["predicted_psi"].corr(group["psi_shrunk"], method="spearman")
        rows.append({"design": design, "n": len(group), "pearson_r": r, "spearman_r": rho})
    per_design = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(per_design.to_string(index=False))

    test_df[["design", "var_id", "psi_shrunk", "predicted_psi"]].to_csv(OUT_CSV, index=False)
    per_design.to_csv(
        "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_pooled16_hek_test_per_design.tsv",
        sep="\t", index=False,
    )
    print(f"\nWrote {OUT_CSV} and per-design breakdown")


if __name__ == "__main__":
    main()
