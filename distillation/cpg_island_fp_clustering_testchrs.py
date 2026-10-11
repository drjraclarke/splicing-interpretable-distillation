"""
CpG-island false-positive clustering on the standard SpliceAI test
chromosomes (chr1, 3, 5, 7, 9) - pretrained vs. fine-tuned (sl650_u16_ensemble)
OpenSpliceAI.

Generalizes cpg_island_fp_clustering_chr22.py (single chromosome) to multiple
chromosomes at once. Confirmed the paper's own CpG-island false-positive
analysis (data/SpliceAI_CpGIsland_FalsePositive_Predictions.csv.gz, Fig. 2B)
is genome-wide (1,184 genes spanning at least 12+ chromosomes in a random
sample, via mygene.info lookup) - this scales our own replication from the
423-gene chr22 proxy to ~5x more genes on the chromosomes SpliceAI/OpenSpliceAI
itself treats as held-out test data, for much stronger statistical power on
whether fine-tuning shifts CpG-island clustering of false positives.

Same methodology as the chr22 version (see that file's docstring for the full
description and caveats vs. the paper's own control-region approach) - the
only difference is CpG-island overlap and the "expected by chance" baseline
are now computed per-chromosome and combined across chr1/3/5/7/9.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path("/nfs/users/nfs_j/jc58/repos/OpenSpliceAI")
sys.path.insert(0, str(REPO))

LUSTRE = Path("/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning")
GENOMIC_H5_DIR = LUSTRE / "input_data/genomic_h5_testchrs_f10000"
DATASET_H5 = GENOMIC_H5_DIR / "dataset_test.h5"
DATAFILE_H5 = GENOMIC_H5_DIR / "datafile_test.h5"
GFF_PATH = GENOMIC_H5_DIR / "build_scratch/MANE_testchrs.gff"
CPG_ISLANDS_PATH = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/cpg_islands/cpgIslandExt.txt.gz")
OUT_DIR = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/cpg_island_fp_testchrs")

TEST_CHROMS = {"1", "3", "5", "7", "9"}

SL = 5000
CL_MAX = 10000
CHUNK_SIZE = 100
FP_THRESHOLD = 0.2
BATCH_SIZE = 32

MANE_DIR = REPO / "models/openspliceai-mane/10000nt"
PRETRAINED_CKPTS = [MANE_DIR / f"model_10000nt_rs{s}.pt" for s in [10, 11, 12, 13, 14]]

FINETUNED_CKPTS = [
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.55_rs10_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.55_rs10_seed42_10000_0_rs42/0/models/model_2.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.60_rs11_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.60_rs11_seed42_10000_0_rs42/0/models/model_1.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs12_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs12_seed42_10000_0_rs42/0/models/model_1.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs13_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs13_seed42_10000_0_rs42/0/models/model_2.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs14_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs14_seed42_10000_0_rs42/0/models/model_1.pt",
]


# ---------------------------------------------------------------------------
# Model loading (identical to benchmark_genomic_topk.py / chr22 version)
# ---------------------------------------------------------------------------

def _get_state_dict(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {}
    for k, v in state.items():
        k2 = k
        for prefix in ("module.", "model."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        clean[k2] = v
    return clean


def _infer_W(state):
    hits = []
    for k, v in state.items():
        if hasattr(v, "shape") and "residual_units" in k and "conv1" in k and "weight" in k:
            m = re.search(r"residual_units\.(\d+).*conv1.*weight", k)
            if m:
                hits.append((int(m.group(1)), int(v.shape[-1])))
    if not hits:
        raise ValueError("Could not infer residual conv widths from checkpoint.")
    return np.asarray([w for _, w in sorted(hits)], dtype=int)


def _choose_AR(n_units):
    if n_units == 16:
        return np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25], dtype=int)
    if n_units == 32:
        return np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25,
                            64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512], dtype=int)
    raise ValueError(f"No AR schedule for n_units={n_units}")


def load_model(ckpt_path, device):
    from openspliceai.train_base.openspliceai import SpliceAI
    state = _get_state_dict(ckpt_path)
    W = _infer_W(state)
    model = SpliceAI(L=int(state["initial_conv.weight"].shape[0]), W=W, AR=_choose_AR(len(W)), apply_softmax=True)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Genomic coordinate reconstruction (multi-chromosome)
# ---------------------------------------------------------------------------

def load_transcript_metadata():
    with h5py.File(DATAFILE_H5, "r") as d:
        names = [n.decode() for n in d["NAME"][:]]
        chroms = [c.decode() for c in d["CHROM"][:]]
        strands = [s.decode() for s in d["STRAND"][:]]
        tx_starts = [int(x.decode()) for x in d["TX_START"][:]]
        tx_ends = [int(x.decode()) for x in d["TX_END"][:]]
    transcripts = []
    for name, chrom, strand, start, end in zip(names, chroms, strands, tx_starts, tx_ends):
        length = end - start + 1
        num_windows = -(-length // SL)  # ceil_div
        transcripts.append({
            "name": name, "chrom": chrom, "strand": strand,
            "tx_start": start, "tx_end": end, "length": length, "num_windows": num_windows,
        })
    return transcripts


def parse_gff_introns():
    """Returns dict: gene_id -> list of (intron_start_1based, intron_end_1based) inclusive."""
    mrna_to_gene = {}
    gene_exons = {}
    with open(GFF_PATH) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            ftype = fields[2]
            attrs = dict(kv.split("=", 1) for kv in fields[8].split(";") if "=" in kv)
            if ftype == "mRNA":
                mrna_to_gene[attrs["ID"]] = attrs["Parent"]
            elif ftype == "exon":
                gene_id = mrna_to_gene.get(attrs["Parent"])
                if gene_id is None:
                    continue
                gene_exons.setdefault(gene_id, []).append((int(fields[3]), int(fields[4])))

    gene_introns = {}
    for gene_id, exons in gene_exons.items():
        exons = sorted(exons)
        introns = [(exons[i][1] + 1, exons[i + 1][0] - 1) for i in range(len(exons) - 1) if exons[i + 1][0] - 1 >= exons[i][1] + 1]
        gene_introns[gene_id] = introns
    return gene_introns


def build_position_index(transcripts, gene_introns):
    per_transcript = []
    for t in transcripts:
        n = t["num_windows"] * SL
        rel = np.arange(n)
        if t["strand"] == "+":
            pos_arr = t["tx_start"] + rel
        else:
            pos_arr = t["tx_end"] - rel

        intronic_arr = np.zeros(n, dtype=bool)
        introns = gene_introns.get(t["name"], [])
        if introns:
            intron_starts = np.array([i[0] for i in introns])
            intron_ends = np.array([i[1] for i in introns])
            order = np.argsort(pos_arr)
            sorted_pos = pos_arr[order]
            idx = np.searchsorted(intron_starts, sorted_pos, side="right") - 1
            in_range = (idx >= 0)
            within = np.zeros(len(sorted_pos), dtype=bool)
            within[in_range] = sorted_pos[in_range] <= intron_ends[idx[in_range]]
            intronic_arr = np.empty(n, dtype=bool)
            intronic_arr[order] = within

        per_transcript.append({"genomic_pos": pos_arr, "is_intronic": intronic_arr, "valid_len": t["length"]})
    return per_transcript


def load_cpg_islands_by_chrom():
    """Returns dict: chrom (no 'chr' prefix, e.g. '1') -> (starts0 sorted, ends0 aligned)."""
    import gzip
    raw = defaultdict(list)
    with gzip.open(CPG_ISLANDS_PATH, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            chrom = fields[1]
            if not chrom.startswith("chr"):
                continue
            chrom = chrom[3:]
            if chrom not in TEST_CHROMS:
                continue
            start0, end0 = int(fields[2]), int(fields[3])
            raw[chrom].append((start0, end0))

    lookup = {}
    for chrom, intervals in raw.items():
        intervals.sort()
        starts0 = np.array([s for s, e in intervals])
        ends0 = np.array([e for s, e in intervals])
        lookup[chrom] = (starts0, ends0)
    return lookup


def positions_in_cpg(genomic_pos_1based, starts0, ends0):
    pos0 = genomic_pos_1based - 1
    idx = np.searchsorted(starts0, pos0, side="right") - 1
    in_range = idx >= 0
    result = np.zeros(len(pos0), dtype=bool)
    result[in_range] = pos0[in_range] < ends0[idx[in_range]]
    return result


def total_intronic_bp_and_cpg_bp_by_chrom(gene_introns, gene_chrom, cpg_by_chrom):
    """Returns dict: chrom -> (total_intronic_bp, cpg_intronic_bp)."""
    totals = defaultdict(lambda: [0, 0])
    for gene_id, introns in gene_introns.items():
        chrom = gene_chrom.get(gene_id)
        if chrom is None or chrom not in cpg_by_chrom:
            continue
        starts0, ends0 = cpg_by_chrom[chrom]
        for s1, e1 in introns:
            totals[chrom][0] += e1 - s1 + 1
            s0, e0 = s1 - 1, e1
            idx_lo = max(np.searchsorted(starts0, s0, side="right") - 1, 0)
            for i in range(idx_lo, len(starts0)):
                if starts0[i] >= e0:
                    break
                overlap = min(ends0[i], e0) - max(starts0[i], s0)
                if overlap > 0:
                    totals[chrom][1] += overlap
    return {c: tuple(v) for c, v in totals.items()}


# ---------------------------------------------------------------------------
# Inference + FP extraction
# ---------------------------------------------------------------------------

def run_ensemble_and_extract_fps(models, device, per_transcript, transcripts):
    """Returns dict: ss_type -> list of (chrom, genomic_pos_1based)."""
    fp_positions = {"acceptor": [], "donor": []}

    with h5py.File(DATASET_H5, "r") as h5:
        n_shards = sum(1 for k in h5 if k.startswith("X"))
        t_idx = 0
        for shard_idx in range(n_shards):
            X_np = h5[f"X{shard_idx}"][:].transpose(0, 2, 1)
            Y_np = h5[f"Y{shard_idx}"][0].transpose(0, 2, 1)
            N = X_np.shape[0]

            shard_transcripts = []
            n_accum = 0
            while n_accum < N:
                shard_transcripts.append(t_idx)
                n_accum += transcripts[t_idx]["num_windows"]
                t_idx += 1
            assert n_accum == N, f"shard {shard_idx}: window count mismatch {n_accum} != {N}"

            y_pred_acc = np.empty((N, Y_np.shape[2]), dtype=np.float32)
            y_pred_don = np.empty((N, Y_np.shape[2]), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, N, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, N)
                    x = torch.tensor(X_np[start:end], dtype=torch.float32).to(device)
                    preds = [m(x).cpu().numpy() for m in models]
                    yp = np.mean(preds, axis=0)
                    y_pred_acc[start:end] = yp[:, 1, :]
                    y_pred_don[start:end] = yp[:, 2, :]

            true_acc = Y_np[:, 1, :]
            true_don = Y_np[:, 2, :]

            row = 0
            for ti in shard_transcripts:
                nw = transcripts[ti]["num_windows"]
                chrom = transcripts[ti]["chrom"]
                pos_full = per_transcript[ti]["genomic_pos"].reshape(nw, SL)
                intronic_full = per_transcript[ti]["is_intronic"].reshape(nw, SL)
                valid_len = per_transcript[ti]["valid_len"]

                for w in range(nw):
                    r = row + w
                    n_valid = min(SL, valid_len - w * SL)
                    if n_valid <= 0:
                        continue
                    pred_acc_w = y_pred_acc[r, :n_valid]
                    pred_don_w = y_pred_don[r, :n_valid]
                    true_acc_w = true_acc[r, :n_valid]
                    true_don_w = true_don[r, :n_valid]
                    pos_w = pos_full[w, :n_valid]
                    intronic_w = intronic_full[w, :n_valid]

                    acc_fp_mask = (pred_acc_w > FP_THRESHOLD) & (true_acc_w < 0.5) & intronic_w
                    don_fp_mask = (pred_don_w > FP_THRESHOLD) & (true_don_w < 0.5) & intronic_w

                    if acc_fp_mask.any():
                        fp_positions["acceptor"].extend((chrom, p) for p in pos_w[acc_fp_mask].tolist())
                    if don_fp_mask.any():
                        fp_positions["donor"].extend((chrom, p) for p in pos_w[don_fp_mask].tolist())
                row += nw

            del X_np, Y_np, y_pred_acc, y_pred_don
            print(f"  shard {shard_idx+1}/{n_shards} done ({N} windows, {len(shard_transcripts)} transcripts)", flush=True)

    return fp_positions


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading transcript metadata...", flush=True)
    transcripts = load_transcript_metadata()
    chrom_counts = defaultdict(int)
    for t in transcripts:
        chrom_counts[t["chrom"]] += 1
    print(f"  {len(transcripts)} transcripts, total windows = {sum(t['num_windows'] for t in transcripts)}")
    print(f"  Per-chromosome gene counts: {dict(chrom_counts)}")

    gene_chrom = {t["name"]: t["chrom"] for t in transcripts}

    print("Parsing GFF for intron structure...", flush=True)
    gene_introns = parse_gff_introns()
    n_introns = sum(len(v) for v in gene_introns.values())
    print(f"  {n_introns} introns across {len(gene_introns)} genes")

    print("Building per-transcript genomic coordinate index...", flush=True)
    per_transcript = build_position_index(transcripts, gene_introns)

    print("Loading CpG islands (genome-wide file, filtered to test chromosomes)...", flush=True)
    cpg_by_chrom = load_cpg_islands_by_chrom()
    for c in sorted(cpg_by_chrom, key=lambda x: (len(x), x)):
        print(f"  chr{c}: {len(cpg_by_chrom[c][0])} CpG islands")

    intronic_totals = total_intronic_bp_and_cpg_bp_by_chrom(gene_introns, gene_chrom, cpg_by_chrom)
    total_bp = sum(v[0] for v in intronic_totals.values())
    total_cpg_bp = sum(v[1] for v in intronic_totals.values())
    expected_frac = total_cpg_bp / total_bp if total_bp > 0 else float("nan")
    print(f"  Total intronic bp (all test chroms) = {total_bp:,}")
    print(f"  Intronic bp inside CpG islands = {total_cpg_bp:,} ({100*expected_frac:.3f}% expected by chance)")

    results = {}
    for label, ckpts in [("pretrained", PRETRAINED_CKPTS), ("finetuned_sl650_u16", FINETUNED_CKPTS)]:
        print(f"\n{'='*60}\nModel: {label}\n{'='*60}", flush=True)
        models = [load_model(p, device) for p in ckpts]
        fp_positions = run_ensemble_and_extract_fps(models, device, per_transcript, transcripts)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model_result = {}
        for ss_type in ("acceptor", "donor"):
            entries = fp_positions[ss_type]
            n_fp = len(entries)
            if n_fp == 0:
                model_result[ss_type] = {"n_fp": 0, "n_fp_in_cpg": 0, "observed_frac": float("nan")}
                continue

            by_chrom = defaultdict(list)
            for chrom, pos in entries:
                by_chrom[chrom].append(pos)

            n_in_cpg = 0
            for chrom, positions in by_chrom.items():
                if chrom not in cpg_by_chrom:
                    continue
                starts0, ends0 = cpg_by_chrom[chrom]
                in_cpg = positions_in_cpg(np.array(positions, dtype=np.int64), starts0, ends0)
                n_in_cpg += int(in_cpg.sum())

            model_result[ss_type] = {
                "n_fp": n_fp,
                "n_fp_in_cpg": n_in_cpg,
                "observed_frac": n_in_cpg / n_fp,
                "per_chrom_fp_counts": {c: len(v) for c, v in by_chrom.items()},
            }
            print(f"  {ss_type}: n_intronic_FP={n_fp}, in_CpG_island={n_in_cpg} ({100*n_in_cpg/n_fp:.2f}%), expected={100*expected_frac:.2f}%")
        results[label] = model_result

    summary = {
        "test_chromosomes": sorted(TEST_CHROMS, key=lambda x: (len(x), x)),
        "fp_threshold": FP_THRESHOLD,
        "total_intronic_bp": int(total_bp),
        "cpg_intronic_bp": int(total_cpg_bp),
        "expected_fraction_pct": 100 * expected_frac,
        "per_chrom_intronic_totals": {c: {"total_bp": int(v[0]), "cpg_bp": int(v[1])} for c, v in intronic_totals.items()},
        "models": results,
    }
    with open(OUT_DIR / "cpg_fp_clustering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'cpg_fp_clustering_summary.json'}")


if __name__ == "__main__":
    main()
