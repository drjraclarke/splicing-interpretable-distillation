"""
Same analysis as inspect_motifs.py, but for the pooled 16-design model
(trained on all designs, not just Exon_6) - checks whether the learned
motifs/filter-importance ranking differ once the model sees the full
sequence diversity of all 16 designs rather than one repeated backbone.
"""

import copy
import itertools
import json
from pathlib import Path

if not hasattr(itertools, "batched"):
    def _batched(iterable, n):
        it = iter(iterable)
        while True:
            batch = tuple(itertools.islice(it, n))
            if not batch:
                return
            yield batch
    itertools.batched = _batched

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets.sequence import SequenceDataset
from experiment.experiment import Experiment
from metrics import metrics
import model_interpretation as mi

EXPERIMENT_DIR = Path(
    "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/ray_results/fas_pooled16_hek_full_v2"
)
TRIAL_ID = "5f6b62b8"
TEST_CSV = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/fas_pooled16_hek_test.csv.gz"
OUT_DIR = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/figures_pooled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RANDOM_KMERS = 2_000_000
TOP_ACTIVATION_FRAC = 0.05
SEED = 0
ALPHABET = ["A", "C", "G", "T"]
ALPHABET_COLORS = "classic"


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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model(EXPERIMENT_DIR, TRIAL_ID, device)

    conv_layer = model.seq_conv.convs["0"]
    kernel_size = conv_layer.kernel_size[0]
    num_filters = conv_layer.out_channels
    print(f"Conv layer: {num_filters} filters, kernel_size={kernel_size}")

    print(f"Sampling {N_RANDOM_KMERS} random {kernel_size}-mers and computing activations...")
    kmer_gen = mi._random_kmers_generator(ALPHABET, kernel_size, N_RANDOM_KMERS, SEED)
    encoded_kmers, activations = mi.compute_conv1d_activations_batched(
        conv_layer=conv_layer,
        activation_layer=model.seq_conv.activation,
        batchnorm_layer=model.seq_conv.batch_norm,
        alphabet=ALPHABET,
        kmer_generator=kmer_gen,
        device=device,
        batch_size=100_000,
    )
    activations = activations[:, :, 0]  # (num_kmers, num_filters)
    print(f"Activations shape: {activations.shape}")

    print("\nLoading test set and running ablation importance ranking...")
    # 16-design pooled test set (384,199 rows) - subsample for tractability,
    # same as the Exon_6-only analysis. Random sampling draws proportionally
    # from all designs since they're mixed row-wise in the file.
    test_df = pd.read_csv(TEST_CSV).sample(n=2000, random_state=0)
    print(f"Ablation sample design counts:\n{test_df['design'].value_counts()}")
    metric_fns = {"pearson-r": metrics.METRICS["pearson-r"]}

    ablation_results = mi.calculate_feature_nam_ablations(
        params=config,
        model=model,
        dataset=test_df,
        num_features=num_filters,
        metric_fns=metric_fns,
        device=device,
        activation_fn=lambda x: 1 / (1 + np.exp(-x)),
    )

    baseline_labels, baseline_preds = mi.get_model_predictions(
        config, model, test_df, device, activation_fn=lambda x: 1 / (1 + np.exp(-x))
    )
    baseline_r = metrics.METRICS["pearson-r"](baseline_preds, baseline_labels)
    print(f"Baseline (no ablation) pearson-r = {baseline_r:.4f}")

    importance_rows = []
    for filter_idx, result in ablation_results.items():
        drop = baseline_r - result["pearson-r"]
        importance_rows.append(
            {"filter_idx": filter_idx, "ablated_pearson_r": result["pearson-r"], "importance_drop": drop}
        )
    importance_df = pd.DataFrame(importance_rows).sort_values("importance_drop", ascending=False)
    importance_df.to_csv(OUT_DIR / "filter_importance_ranking.csv", index=False)
    print("\nTop 10 most important filters:")
    print(importance_df.head(10).to_string(index=False))

    print("\nPlotting sequence logos for top 10 filters...")
    for rank, (_, row) in enumerate(importance_df.head(10).iterrows()):
        filter_idx = int(row["filter_idx"])
        filter_activations = activations[:, filter_idx]

        threshold = np.quantile(filter_activations, 1 - TOP_ACTIVATION_FRAC)

        fig, ax = plt.subplots(figsize=(6, 2))
        mi.plot_sequence_logo(
            kernel_activations=filter_activations,
            kmers=encoded_kmers,
            alphabet=ALPHABET,
            alphabet_colors=ALPHABET_COLORS,
            ax=ax,
            min_activation=threshold,
        )
        ax.set_title(f"Rank {rank+1}, filter {filter_idx}, importance_drop={row['importance_drop']:.4f}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"motif_rank{rank+1}_filter{filter_idx}.png", dpi=150)
        plt.close(fig)

    print(f"\nWrote importance ranking and logos to {OUT_DIR}")


if __name__ == "__main__":
    main()
