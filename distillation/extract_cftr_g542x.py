"""
Extract genomic context around CFTR c.1624G>T (G542X, chr7:117587778 G>T,
GRCh38, verified against ClinVar) for both wildtype and mutant, as FASTA
files ready for `openspliceai predict`.
"""

from pyfaidx import Fasta

REF_FASTA = "/lustre/scratch124/gengen/teams_v2/lehner/jc58/genomes/GRCh38.primary_assembly.genome.fa"
CHROM = "7"
VARIANT_POS_1BASED = 117587778  # chr7:117587778 G>T, GRCh38, verified vs ClinVar RCV000007535
REF_ALLELE = "G"
ALT_ALLELE = "T"
FLANK = 5000  # 5000nt each side -> 10,000nt total context, matching paper's convention

OUT_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data/cftr_g542x"


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = Fasta(REF_FASTA)
    chrom_seq = ref[CHROM]

    start_0based = VARIANT_POS_1BASED - 1 - FLANK
    end_0based = VARIANT_POS_1BASED - 1 + FLANK + 1  # +1 to include variant position, half-open slice

    window = str(chrom_seq[start_0based:end_0based]).upper()
    variant_idx_in_window = FLANK  # 0-based index of the variant within the extracted window

    assert window[variant_idx_in_window] == REF_ALLELE, (
        f"Reference mismatch at variant position: expected {REF_ALLELE}, got {window[variant_idx_in_window]}"
    )

    mutant_window = window[:variant_idx_in_window] + ALT_ALLELE + window[variant_idx_in_window + 1:]

    print(f"Window: chr{CHROM}:{start_0based+1}-{end_0based} (GRCh38), length={len(window)}")
    print(f"Variant at 0-based window index {variant_idx_in_window} (1-based genomic pos {VARIANT_POS_1BASED})")
    print(f"WT base: {window[variant_idx_in_window]}, MUT base: {mutant_window[variant_idx_in_window]}")

    with open(f"{OUT_DIR}/cftr_wt.fa", "w") as f:
        f.write(">cftr_e12_wildtype\n")
        f.write(window + "\n")

    with open(f"{OUT_DIR}/cftr_g542x.fa", "w") as f:
        f.write(">cftr_e12_g542x\n")
        f.write(mutant_window + "\n")

    with open(f"{OUT_DIR}/window_metadata.txt", "w") as f:
        f.write(f"chrom={CHROM}\n")
        f.write(f"window_start_1based={start_0based+1}\n")
        f.write(f"window_end_1based={end_0based}\n")
        f.write(f"window_length={len(window)}\n")
        f.write(f"variant_pos_1based={VARIANT_POS_1BASED}\n")
        f.write(f"variant_0based_index_in_window={variant_idx_in_window}\n")
        f.write(f"ref_allele={REF_ALLELE}\n")
        f.write(f"alt_allele={ALT_ALLELE}\n")

    print(f"\nWrote cftr_wt.fa, cftr_g542x.fa, window_metadata.txt to {OUT_DIR}")


if __name__ == "__main__":
    main()
