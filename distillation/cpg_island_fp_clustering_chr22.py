"""
CpG-island false-positive clustering on chr22, pretrained vs. fine-tuned
(sl650_u16_ensemble) OpenSpliceAI.

Replicates the spirit of the paper's SpliceAI-CpG-island false-positive
analysis (data/SpliceAI_CpGIsland_FalsePositive_Predictions.csv.gz,
notebooks/SpliceAI_CpG_Islands_Predictions.ipynb) on our own two model
ensembles, scoped to chr22 only.

Methodology (a simplified, explicitly-caveated version of the paper's own
method - see README written alongside this script's output):
  1. For each of the 423 chr22 protein-coding (MANE) genes already used to
     build genomic_h5_chr22_f10000, reconstruct per-position genomic
     coordinates for every base in the gene body (using TX_START/TX_END/
     STRAND from datafile_test.h5, and the exact windowing scheme used by
     openspliceai create-data).
  2. Reconstruct intron intervals per gene from the source GFF
     (chr22_ensembl.gff - single MANE-Select transcript per gene, so exon
     structure is unambiguous - no canonical-transcript-selection logic
     needed).
  3. Run both model ensembles (pretrained 5-seed, sl650_u16 fine-tuned
     5-seed) over all 5 shards of dataset_test.h5, exactly as
     benchmark_genomic_topk.py does.
  4. A "false positive" = a genomic position with predicted acceptor OR
     donor score > FP_THRESHOLD that is (a) inside an intron of its gene,
     and (b) not itself an annotated true splice-site position (using the
     H5's own labels, which mark only canonical GT-AG/GC-AG/AT-AC sites).
  5. For each FP, check whether it falls inside a chr22 CpG island (UCSC
     cpgIslandExt track). Report observed fraction of intronic FPs inside
     CpG islands vs. the expected fraction under random placement (intronic
     bp inside islands / total intronic bp), for each model and each
     splice-site type (acceptor/donor).

Caveat vs. the paper's own method: the paper builds an explicit matched
control region and re-runs the model on it (columns CTRL_FP/CTRL_total_bp
in their CSV); this script instead uses an analytical expected-fraction
control (intronic bp inside islands / total intronic bp), which is simpler
but does not require additional model inference. This is flagged clearly in
the output rather than presented as an exact replication.
"""

import gzip
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path("/nfs/users/nfs_j/jc58/repos/OpenSpliceAI")
sys.path.insert(0, str(REPO))

GENOMIC_H5_DIR = Path("/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning/input_data/genomic_h5_chr22_f10000")
DATASET_H5 = GENOMIC_H5_DIR / "dataset_test.h5"
DATAFILE_H5 = GENOMIC_H5_DIR / "datafile_test.h5"
GFF_PATH = REPO / "examples/data/chr22_ensembl.gff"
CPG_ISLANDS_PATH = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/cpg_islands/chr22_cpg_islands.txt")
OUT_DIR = Path("/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/cpg_island_fp_chr22")

SL = 5000
CL_MAX = 10000
CHUNK_SIZE = 100
FP_THRESHOLD = 0.2
BATCH_SIZE = 32

MANE_DIR = REPO / "models/openspliceai-mane/10000nt"
PRETRAINED_CKPTS = [MANE_DIR / f"model_10000nt_rs{s}.pt" for s in [10, 11, 12, 13, 14]]

LUSTRE = Path("/lustre/scratch124/gengen/teams_v2/lehner/jc58/open_splice_ai_fine_tuning")
FINETUNED_CKPTS = [
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.55_rs10_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.55_rs10_seed42_10000_0_rs42/0/models/model_2.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.60_rs11_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.60_rs11_seed42_10000_0_rs42/0/models/model_1.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs12_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs12_seed42_10000_0_rs42/0/models/model_1.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs13_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs13_seed42_10000_0_rs42/0/models/model_2.pt",
    LUSTRE / "results/mpra_finetune/ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs14_seed42/finetune_checkpoints/SpliceAI_ft_progunfreeze_sl650_20260805_stage5_u16_lam0.45_rs14_seed42_10000_0_rs42/0/models/model_1.pt",
]


# ---------------------------------------------------------------------------
# Model loading (identical logic to benchmark_genomic_topk.py)
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
# Genomic coordinate reconstruction
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


def parse_gff_introns(transcripts_by_name):
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
    """
    For each transcript, build a per-window array marking, for every position
    0..SL-1: (is_valid, genomic_pos_1based, is_intronic).
    Returns a list (aligned with `transcripts`) of dicts with 'genomic_pos'
    (num_windows*SL,) and 'is_intronic' (same shape) and 'valid_len'.
    """
    per_transcript = []
    for t in transcripts:
        n = t["num_windows"] * SL
        pos_arr = np.zeros(n, dtype=np.int64)
        intronic_arr = np.zeros(n, dtype=bool)
        valid_len = t["length"]
        rel = np.arange(n)
        if t["strand"] == "+":
            pos_arr = t["tx_start"] + rel
        else:
            pos_arr = t["tx_end"] - rel

        introns = gene_introns.get(t["name"], [])
        if introns:
            intron_starts = np.array([i[0] for i in introns])
            intron_ends = np.array([i[1] for i in introns])
            # positions beyond valid_len are padding; harmless to compute, will be masked out
            order = np.argsort(pos_arr)
            sorted_pos = pos_arr[order]
            idx = np.searchsorted(intron_starts, sorted_pos, side="right") - 1
            in_range = (idx >= 0)
            within = np.zeros(len(sorted_pos), dtype=bool)
            within[in_range] = sorted_pos[in_range] <= intron_ends[idx[in_range]]
            intronic_sorted = within
            intronic_arr = np.empty(n, dtype=bool)
            intronic_arr[order] = intronic_sorted

        per_transcript.append({"genomic_pos": pos_arr, "is_intronic": intronic_arr, "valid_len": valid_len, "n": n})
    return per_transcript


def load_cpg_islands():
    islands = []
    with open(CPG_ISLANDS_PATH) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            chrom = fields[1].replace("chr", "")
            start0, end0 = int(fields[2]), int(fields[3])  # UCSC: 0-based start, exclusive end
            islands.append((chrom, start0, end0))
    return islands


def build_cpg_lookup(islands, chrom="22"):
    starts = np.array(sorted(s for c, s, e in islands if c == chrom))
    ends_by_start = {s: e for c, s, e in islands if c == chrom}
    ends = np.array([ends_by_start[s] for s in starts])
    return starts, ends


def positions_in_cpg(genomic_pos_1based, starts0, ends0):
    """genomic_pos_1based: array of 1-based positions. starts0/ends0: UCSC 0-based half-open."""
    pos0 = genomic_pos_1based - 1
    idx = np.searchsorted(starts0, pos0, side="right") - 1
    in_range = idx >= 0
    result = np.zeros(len(pos0), dtype=bool)
    result[in_range] = pos0[in_range] < ends0[idx[in_range]]
    return result


def total_intronic_bp_and_cpg_bp(gene_introns, starts0, ends0):
    total_bp = 0
    cpg_bp = 0
    for introns in gene_introns.values():
        for s1, e1 in introns:  # 1-based inclusive
            total_bp += e1 - s1 + 1
            s0, e0 = s1 - 1, e1  # 0-based half-open
            idx_lo = np.searchsorted(starts0, s0, side="right") - 1
            for i in range(max(idx_lo, 0), len(starts0)):
                if starts0[i] >= e0:
                    break
                overlap = min(ends0[i], e0) - max(starts0[i], s0)
                if overlap > 0:
                    cpg_bp += overlap
    return total_bp, cpg_bp


# ---------------------------------------------------------------------------
# Inference + FP extraction
# ---------------------------------------------------------------------------

def run_ensemble_and_extract_fps(models, device, per_transcript, transcripts):
    """Runs ensemble inference over all shards, returns FP genomic positions per type."""
    fp_positions = {"acceptor": [], "donor": []}

    with h5py.File(DATASET_H5, "r") as h5:
        n_shards = sum(1 for k in h5 if k.startswith("X"))
        t_idx = 0
        for shard_idx in range(n_shards):
            X_np = h5[f"X{shard_idx}"][:].transpose(0, 2, 1)  # (N,4,L_in)
            Y_np = h5[f"Y{shard_idx}"][0].transpose(0, 2, 1)  # (N,3,L_out)
            N = X_np.shape[0]

            # figure out which transcripts (in order) make up this shard's N windows
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

            # stitch together this shard's windows -> per-transcript genomic coords
            row = 0
            for ti in shard_transcripts:
                nw = transcripts[ti]["num_windows"]
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
                        fp_positions["acceptor"].extend(pos_w[acc_fp_mask].tolist())
                    if don_fp_mask.any():
                        fp_positions["donor"].extend(pos_w[don_fp_mask].tolist())
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
    print(f"  {len(transcripts)} transcripts, total windows = {sum(t['num_windows'] for t in transcripts)}")

    print("Parsing GFF for intron structure...", flush=True)
    gene_introns = parse_gff_introns(transcripts)
    n_introns = sum(len(v) for v in gene_introns.values())
    print(f"  {n_introns} introns across {len(gene_introns)} genes")

    print("Building per-transcript genomic coordinate index...", flush=True)
    per_transcript = build_position_index(transcripts, gene_introns)

    print("Loading CpG islands...", flush=True)
    islands = load_cpg_islands()
    starts0, ends0 = build_cpg_lookup(islands, chrom="22")
    print(f"  {len(starts0)} CpG islands on chr22")

    total_intronic_bp, cpg_intronic_bp = total_intronic_bp_and_cpg_bp(gene_introns, starts0, ends0)
    expected_frac = cpg_intronic_bp / total_intronic_bp if total_intronic_bp > 0 else float("nan")
    print(f"  Total intronic bp (chr22, these genes) = {total_intronic_bp:,}")
    print(f"  Intronic bp inside CpG islands = {cpg_intronic_bp:,} ({100*expected_frac:.3f}% expected by chance)")

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
            pos = np.array(fp_positions[ss_type], dtype=np.int64)
            n_fp = len(pos)
            if n_fp == 0:
                model_result[ss_type] = {"n_fp": 0, "n_fp_in_cpg": 0, "observed_frac": float("nan")}
                continue
            in_cpg = positions_in_cpg(pos, starts0, ends0)
            n_in_cpg = int(in_cpg.sum())
            model_result[ss_type] = {
                "n_fp": n_fp,
                "n_fp_in_cpg": n_in_cpg,
                "observed_frac": n_in_cpg / n_fp,
            }
            print(f"  {ss_type}: n_intronic_FP={n_fp}, in_CpG_island={n_in_cpg} ({100*n_in_cpg/n_fp:.2f}%), expected={100*expected_frac:.2f}%")
        results[label] = model_result

    import json
    summary = {
        "fp_threshold": FP_THRESHOLD,
        "total_intronic_bp": int(total_intronic_bp),
        "cpg_intronic_bp": int(cpg_intronic_bp),
        "expected_fraction_pct": 100 * expected_frac,
        "models": results,
    }
    with open(OUT_DIR / "cpg_fp_clustering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'cpg_fp_clustering_summary.json'}")


if __name__ == "__main__":
    main()
