"""
CFTR G542X (chr7:117587778 G>T, GRCh38) case study: predict OpenSpliceAI
acceptor/donor scores for WT vs. mutant, using both the pretrained 5-seed
ensemble and the sl650_u16 fine-tuned 5-seed ensemble, and compare the
resulting deltas to the paper's reported SpliceAI/Pangolin/AlphaGenome/
measured deltas.

Real biology: G542X is a nonsense mutation with no meaningful effect on exon
12 splicing (paper's own measured PSI delta ~ -0.7 on a 0-100 scale). SpliceAI
and Pangolin wrongly predict large splicing disruption; AlphaGenome does not.
This checks whether MPRA fine-tuning nudges OpenSpliceAI's prediction in the
correct (near-zero) direction.
"""

import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, "/nfs/users/nfs_j/jc58/repos/OpenSpliceAI")
from openspliceai.predict import predict as P

FLANK = 10000
DATA_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/cftr_g542x"
OUT_TSV = f"{DATA_DIR}/cftr_g542x_openspliceai_results.tsv"

MANE_DIR = "/nfs/users/nfs_j/jc58/repos/OpenSpliceAI/models/openspliceai-mane/10000nt"
PRETRAINED_CKPTS = [f"{MANE_DIR}/model_10000nt_rs{seed}.pt" for seed in [10, 11, 12, 13, 14]]

LUSTRE = "/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning"
FINETUNED_CKPTS = [
    f"{LUSTRE}/results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.55_rs10_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.55_rs10_seed42_10000_0_rs42/0/models/model_2.pt",
    f"{LUSTRE}/results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.60_rs11_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.60_rs11_seed42_10000_0_rs42/0/models/model_1.pt",
    f"{LUSTRE}/results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs12_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs12_seed42_10000_0_rs42/0/models/model_1.pt",
    f"{LUSTRE}/results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs13_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs13_seed42_10000_0_rs42/0/models/model_2.pt",
    f"{LUSTRE}/results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs14_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs14_seed42_10000_0_rs42/0/models/model_1.pt",
]

# Paper's own reported PSI/scores (Model x WT/G542X), for comparison.
PAPER_VALUES = {
    "Measured_PSI": {"WT": None, "G542X": None, "delta": -0.7},
    "SpliceAI": {"WT": None, "G542X": None, "delta": -18.2},
    "Pangolin": {"WT": None, "G542X": None, "delta": -12.5},
    "AlphaGenome": {"WT": None, "G542X": None, "delta": -0.2},
}


def read_fasta_seq(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return "".join(line for line in lines if not line.startswith(">"))


def make_ensemble_dir(ckpt_paths, tag):
    d = tempfile.mkdtemp(prefix=f"cftr_ensemble_{tag}_")
    for i, p in enumerate(ckpt_paths):
        os.symlink(p, os.path.join(d, f"model_{i}.pt"))
    return d


def predict_scores(seq, ensemble_dir):
    """Returns [seq_len, 3] tensor: columns = neutral, acceptor, donor."""
    y_pred = P.predict(seq, ensemble_dir, FLANK)
    return y_pred


def main():
    wt_seq = read_fasta_seq(f"{DATA_DIR}/cftr_wt.fa")
    mut_seq = read_fasta_seq(f"{DATA_DIR}/cftr_g542x.fa")
    assert len(wt_seq) == len(mut_seq)
    print(f"[INFO] Sequence length: {len(wt_seq)}")

    variant_idx = 5000  # 0-based index of the variant within the window (see window_metadata.txt)
    diffs = [i for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
    assert diffs == [variant_idx], f"Expected single diff at {variant_idx}, got {diffs}"

    pretrained_dir = make_ensemble_dir(PRETRAINED_CKPTS, "pretrained")
    finetuned_dir = make_ensemble_dir(FINETUNED_CKPTS, "finetuned")

    try:
        print("[INFO] Running pretrained ensemble on WT...")
        pre_wt = predict_scores(wt_seq, pretrained_dir)
        print("[INFO] Running pretrained ensemble on G542X...")
        pre_mut = predict_scores(mut_seq, pretrained_dir)
        print("[INFO] Running fine-tuned ensemble on WT...")
        ft_wt = predict_scores(wt_seq, finetuned_dir)
        print("[INFO] Running fine-tuned ensemble on G542X...")
        ft_mut = predict_scores(mut_seq, finetuned_dir)
    finally:
        shutil.rmtree(pretrained_dir, ignore_errors=True)
        shutil.rmtree(finetuned_dir, ignore_errors=True)

    torch.save(
        {"pre_wt": pre_wt, "pre_mut": pre_mut, "ft_wt": ft_wt, "ft_mut": ft_mut},
        f"{DATA_DIR}/raw_predictions.pt",
    )

    # Identify the exon 12 acceptor/donor sites as the WT pretrained-model's
    # top acceptor/donor peaks (real annotated splice sites should dominate
    # a correctly-functioning model's per-position scores).
    acc_col, don_col = 1, 2
    acc_pos = int(torch.argmax(pre_wt[:, acc_col]))
    don_pos = int(torch.argmax(pre_wt[:, don_col]))
    print(f"[INFO] WT-pretrained top acceptor position: {acc_pos} (score={pre_wt[acc_pos, acc_col]:.4f})")
    print(f"[INFO] WT-pretrained top donor position: {don_pos} (score={pre_wt[don_pos, don_col]:.4f})")

    def exon_score(pred, acc_p, don_p):
        return (float(pred[acc_p, acc_col]) + float(pred[don_p, don_col])) / 2.0

    rows = []
    for model_name, wt_pred, mut_pred in [("OpenSpliceAI_pretrained", pre_wt, pre_mut), ("OpenSpliceAI_finetuned_sl650_u16", ft_wt, ft_mut)]:
        wt_acc = float(wt_pred[acc_pos, acc_col])
        wt_don = float(wt_pred[don_pos, don_col])
        mut_acc = float(mut_pred[acc_pos, acc_col])
        mut_don = float(mut_pred[don_pos, don_col])
        wt_exon = exon_score(wt_pred, acc_pos, don_pos)
        mut_exon = exon_score(mut_pred, acc_pos, don_pos)
        rows.append({
            "model": model_name,
            "acceptor_pos_0based": acc_pos,
            "donor_pos_0based": don_pos,
            "wt_acceptor": wt_acc,
            "mut_acceptor": mut_acc,
            "delta_acceptor": mut_acc - wt_acc,
            "wt_donor": wt_don,
            "mut_donor": mut_don,
            "delta_donor": mut_don - wt_don,
            "wt_exon_score_pct": wt_exon * 100,
            "mut_exon_score_pct": mut_exon * 100,
            "delta_exon_score_pct": (mut_exon - wt_exon) * 100,
        })

    import csv
    with open(OUT_TSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[RESULTS]")
    for r in rows:
        print(f"{r['model']}: WT exon score={r['wt_exon_score_pct']:.2f}%, "
              f"G542X exon score={r['mut_exon_score_pct']:.2f}%, "
              f"delta={r['delta_exon_score_pct']:.2f} percentage points")
    print(f"\n[Paper reference] Measured PSI delta={PAPER_VALUES['Measured_PSI']['delta']}, "
          f"SpliceAI delta={PAPER_VALUES['SpliceAI']['delta']}, "
          f"Pangolin delta={PAPER_VALUES['Pangolin']['delta']}, "
          f"AlphaGenome delta={PAPER_VALUES['AlphaGenome']['delta']}")
    print(f"\n[INFO] Wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
