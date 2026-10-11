"""
Same analysis as plot_position_effects.py, but for the pooled 16-design
model. Unlike the Exon_6-only version, we don't assume which filter (if any)
is stop-codon-like ahead of time - the periodicity check below runs across
all top-10 filters and reports which ones (if any) show 3nt-periodic
position effects, rather than assuming it's "filter 1" as in the single
design model.
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

from experiment.experiment import Experiment
import model_interpretation as mi

EXPERIMENT_DIR = Path(
    "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/ray_results/fas_pooled16_hek_full_v2"
)
TRIAL_ID = "5f6b62b8"
OUT_DIR = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/figures_pooled")
IMPORTANCE_CSV = OUT_DIR / "filter_importance_ranking.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 603  # padded pooled sequence length (see pooled16_design_info.tsv)
N_RANDOM_KMERS = 2_000_000
SEED = 0
ALPHABET = ["A", "C", "G", "T"]
TOP_N = 10


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


def periodicity_stats(diff: np.ndarray, x_vals: np.ndarray) -> dict:
    frame = x_vals % 3
    frame_means = [diff[frame == f].mean() for f in range(3)]
    grand_mean = diff.mean()
    between_var = np.mean([(m - grand_mean) ** 2 for m in frame_means])
    total_var = diff.var()
    frame_variance_ratio = between_var / total_var if total_var > 0 else float("nan")

    diff_centered = diff - diff.mean()
    autocorr = np.correlate(diff_centered, diff_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
    return {
        "frame_means": [float(m) for m in frame_means],
        "frame_variance_ratio": float(frame_variance_ratio),
        "autocorrelation_lag_1_to_6": [float(v) for v in autocorr[1:7]],
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model(EXPERIMENT_DIR, TRIAL_ID, device)
    conv_layer = model.seq_conv.convs["0"]
    kernel_size = conv_layer.kernel_size[0]
    num_filters = conv_layer.out_channels
    print(f"Conv layer: {num_filters} filters, kernel_size={kernel_size}")

    importance_df = pd.read_csv(IMPORTANCE_CSV).sort_values("importance_drop", ascending=False)
    top_filters = importance_df.head(TOP_N)["filter_idx"].astype(int).tolist()
    print(f"Top {TOP_N} filters: {top_filters}")

    print(f"\nSampling {N_RANDOM_KMERS} random {kernel_size}-mers and computing activations...")
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
    print(f"Activations shape: {activations.shape}")

    print(f"\nComputing position-dependent shape functions across all {num_filters} filters "
          f"x {SEQ_LEN} positions...")
    seq_activations_t = torch.from_numpy(activations).to(device)
    kmer_contribution_df, linspace_contribution_df = mi.calculate_first_order_shape_functions(
        seq_activations=seq_activations_t,
        model=model,
        output_idx=0,
        num_positions=SEQ_LEN,
        device=device,
    )
    linspace_contribution_df.to_csv(OUT_DIR / "linspace_contribution_df.csv", index=False)
    print(f"Wrote linspace_contribution_df.csv ({len(linspace_contribution_df)} rows)")

    print(f"\nPlotting position-effect curves (logo + contribution diff) for top {TOP_N} filters...")
    dead_filters = []
    for rank, filter_idx in enumerate(top_filters):
        try:
            fig = mi.plot_kernel(
                conv_weight=conv_layer.weight,
                kmers=encoded_kmers,
                seq_activations=activations[:, :, 0],
                linspace_contribution_df=linspace_contribution_df,
                num_seq_filters=num_filters,
                num_positions=SEQ_LEN,
                seq_filter_idx=filter_idx,
                title=f"Rank {rank+1}, filter {filter_idx}",
                label="",
            )
            fig.savefig(OUT_DIR / f"position_effect_rank{rank+1}_filter{filter_idx}.png", dpi=150)
            plt.close(fig)
        except ValueError as e:
            # A filter whose activations are constant/near-zero across all sampled
            # k-mers (a "dead" filter - flagged separately by inspect_motifs_pooled.py's
            # "All k-mer activations were zero" warning) produces duplicate bin edges
            # in the shape-function pivot. Skip it rather than crashing the whole run.
            print(f"  [skip] filter {filter_idx} (rank {rank+1}): {e}")
            dead_filters.append(int(filter_idx))

    # --- 3-nucleotide periodicity check across ALL top-10 filters ---
    # (unlike the Exon_6 model, we don't know in advance which filter, if
    # any, is stop-codon-like - so scan all of them and rank by periodicity
    # strength, rather than assuming "filter 1" as before.)
    print("\n=== 3nt periodicity check: top filters ===")
    periodicity_summary = {}
    for rank, filter_idx in enumerate(top_filters):
        if int(filter_idx) in dead_filters:
            print(f"  [skip] filter {filter_idx} (rank {rank+1}): dead filter, no periodicity to check")
            continue
        filt_df = linspace_contribution_df[linspace_contribution_df["seq_filter_idx"] == filter_idx]
        try:
            grid_df = (
                filt_df.pivot(index="feature", columns="position", values="adj_logit")
                .sort_index()
                .sort_index(axis=1)
            )
        except ValueError as e:
            print(f"  [skip] filter {filter_idx} (rank {rank+1}): {e}")
            continue
        grid = grid_df.to_numpy()
        x_vals = grid_df.columns.to_numpy()
        diff = grid[-1, :] - grid[0, :] if grid.shape[0] >= 2 else grid[0, :]
        stats = periodicity_stats(diff, x_vals)
        stats["rank"] = rank + 1
        periodicity_summary[int(filter_idx)] = stats
        print(f"  filter {filter_idx} (rank {rank+1}): frame_variance_ratio={stats['frame_variance_ratio']:.4f}, "
              f"lag-3 autocorr={stats['autocorrelation_lag_1_to_6'][2]:.4f}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x_vals, diff)
        frame = x_vals % 3
        for f, color in zip(range(3), ["red", "green", "blue"]):
            ax.scatter(x_vals[frame == f], diff[frame == f], s=5, color=color, label=f"frame {f}", alpha=0.5)
        ax.legend()
        ax.set_xlabel("Position")
        ax.set_ylabel("Position-effect (top - bottom activation)")
        ax.set_title(f"Filter {filter_idx} (rank {rank+1}) position-effect curve, colored by reading frame")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"periodicity_rank{rank+1}_filter{filter_idx}.png", dpi=150)
        plt.close(fig)

    with open(OUT_DIR / "periodicity_summary_all_top_filters.json", "w") as f:
        json.dump(periodicity_summary, f, indent=2)

    most_periodic = max(periodicity_summary.items(), key=lambda kv: kv[1]["frame_variance_ratio"])
    print(f"\nMost 3nt-periodic filter among top {TOP_N}: filter {most_periodic[0]} "
          f"(rank {most_periodic[1]['rank']}, frame_variance_ratio={most_periodic[1]['frame_variance_ratio']:.4f})")

    print(f"\nWrote position-effect plots and periodicity check to {OUT_DIR}")


if __name__ == "__main__":
    main()
