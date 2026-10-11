"""
Evaluate a trained trial's checkpoint on a genuinely held-out test CSV.

hyperparameter_search.py accepts --test_csv but never actually uses it: the
"valid-*" metrics logged during the search are computed on a random 10% split
carved out of --train_csv internally (via --train_fraction), not on a
disjoint test set. This script closes that gap: it reconstructs the exact
model from a trial's saved params.json + checkpoint.pt and scores it on a
separate CSV.

Usage:
  python3 evaluate_on_holdout.py \
    --experiment_dir /path/to/ray_results/fas_exon6_hek_full \
    --trial_id 7873bd5b \
    --test_csv /path/to/fas_exon6_hek_test.csv.gz \
    --sequence_column full_seq \
    --target_column psi_shrunk
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.sequence import SequenceDataset
from experiment.experiment import Experiment
from metrics import metrics


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
    # Training compiled the model with torch.compile(), which wraps it and
    # prefixes all state_dict keys with "_orig_mod.".
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--trial_id", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--sequence_column", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model(Path(args.experiment_dir), args.trial_id, device)

    testset = SequenceDataset(
        data_path=args.test_csv,
        data_df=None,
        sequence_column=args.sequence_column,
        target_column=args.target_column,
        upstream_sequence=config.get("upstream_sequence", ""),
        downstream_sequence=config.get("downstream_sequence", ""),
    )
    loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            preds = model(sequences).squeeze(-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    pearson_r = metrics.METRICS["pearson-logits-r"](preds, labels)
    pearson_p = metrics.METRICS["pearson-logits-r-pval"](preds, labels)
    mse = metrics.METRICS["mse-logits"](preds, labels)

    print(f"n_test = {len(labels)}")
    print(f"test pearson-logits-r = {pearson_r:.4f} (p = {pearson_p:.3g})")
    print(f"test mse-logits       = {mse:.4f}")

    if args.out_csv:
        pred_psi = torch.special.expit(preds).numpy()
        out = pd.DataFrame(
            {
                "measured_psi": labels.numpy(),
                "predicted_psi": pred_psi,
            }
        )
        out.to_csv(args.out_csv, index=False)
        print(f"Wrote predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
