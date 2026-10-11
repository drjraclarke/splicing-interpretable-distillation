# CpG/stop-codon confound analysis on an independent dataset (OpenSplice), pretrained vs. progressively fine-tuned OpenSpliceAI

## What this tests, and why

All our earlier confound analyses were computed on **our own held-out test set** (FAS Exon_6/pooled
designs) - the same broad distribution the fine-tuned checkpoints were tuned toward, even though
the specific test variants were never seen during training. This package tests something stronger:
does the "fine-tuning reduces CpG/stop-codon bias" finding hold up on a **completely independent,
real experimental dataset** that has nothing to do with the FAS random-sequence library?

**OpenSplice** (`lehner-lab/OpenSplice`, distinct from OpenSpliceAI the model - similar name,
different thing) is exactly that: 590,104 real HEK293 variants across 608 natural exons, SNV
(306,343) and short-deletion (283,761) mutagenesis, real measured PSI. Not the FAS-family
randomized-sequence library - genuinely different exons, genuinely different biology.

We compare **pretrained OpenSpliceAI** (zero-shot, never trained on any real MPRA data) against
three **progressive fine-tuning stages** (stage1, stage2, stage5 - matching the project's gradual
layer-unfreezing methodology), using predictions that already exist and are already restricted to
this dataset's own held-out test split (40 exons, leak-free, exon-level 80/20 split) - no fresh
inference was needed. See file 04 for the full leakage-verification writeup (both for this dataset
and for the earlier FAS-test-set comparison) before trusting any of this.

## Headline finding: dose-response confirmed for 2 of 4 features, but the *direction* is not universal

File 03 has the side-by-side FAS-vs-OpenSplice comparison. Key points:

**Stop-codon count** shows the cleanest result: a monotonic reduction in confound magnitude across
every fine-tuning stage, for **both** SNVs and deletions:
- SNV: pretrained 0.0124 → stage1 0.0088 → stage2 0.0068 → stage5 0.0041
- Deletion: pretrained 0.0177 → stage1 0.0117 → stage2 0.0100 → stage5 0.0069

**CpG ratio** shows the same monotonic-reduction pattern for SNVs (pretrained -0.0063 → stage5
-0.0011), but the effect is small and not statistically significant for deletions past the
pretrained stage (p > 0.6 for stage1/2/5).

**CpG count and GpC count do NOT show this benefit** - if anything they mildly *increase* with more
fine-tuning (e.g. CpG count SNV: 0.0029 → 0.0095). Full numbers in file 02.

**The important surprise, worth leading with in any write-up**: on our own FAS test set, CpG ratio
was **positive** (higher CpG → inflated prediction, matching the paper) and stop-codon count was
**negative** (more stop codons → deflated prediction, also matching the paper). On OpenSplice,
**both signs flip** - CpG ratio is negative, stop-codon count is positive. This means these are not
fixed-direction genomic-composition artifacts that hold everywhere; the relationship between these
sequence features and prediction bias appears genuinely dataset/context-dependent. Don't describe
this as "the same confound, just smaller on the new dataset" - the mechanism looks different, even
though the *fine-tuning-helps* direction (shrinking magnitude toward zero) is consistent for two of
the four features tested.

**Overall accuracy context**: calibrated Pearson r on OpenSplice is much lower (0.38-0.55 across
all model stages) than on our own FAS test set (~0.73-0.85 depending on model) - expected, since
this is genuinely out-of-distribution data. Interestingly accuracy dips at stage1/stage2 before
recovering by stage5 for both mutation types - early fine-tuning may trade some generalization to
unrelated loci before later stages recover it.

## File guide

- **02_confound_ancova_opensplice_all_stages.tsv**: full results, every combination of `mut_type`
  (snv/del) x `model` (pretrained/stage1/stage2/stage5) x `feature` (cpg_ratio, cpg_count,
  gpc_count, stop_codon_count), with `n, pearson_r_calibrated, raw_coefficient,
  standardized_coefficient, p_value`.
- **03_confound_comparison_fas_vs_opensplice_headline.tsv**: the compact side-by-side table used
  above, for quick reference/plotting.
- **04_leakage_verification_notes.md**: full writeup of the leakage checks for both the FAS test
  set (confirmed clean, despite a split-file version mismatch) and OpenSplice (its own dedicated
  split, with a caveat about shared vector backbone with FAS - not variant-level leakage, but not
  fully independent either).
- **05_opensplice_per_variant_pretrained_vs_stage5.csv**: per-variant data (both mutation types,
  pretrained and stage5 only - the two most informative endpoints) with
  `gene, exon_id, psi_shrunk, pred_score, predicted_calibrated, error, cpg_ratio, cpg_count,
  gpc_count, stop_codon_count` - for your own plots.

## Caveats

- Confound features are computed over the **whole 145nt sequence window** (not a restricted
  "variable region" the way our own randomized-library analyses did) - appropriate here since these
  are natural exons with point mutations/short deletions anywhere in the window, not a fully
  randomized synthetic region.
- p-values are trivially significant at this sample size (n≈50-55K per group) even for small
  effects - the magnitude/sign comparisons above are what matter, not significance.
- This does not include our own from-scratch (non-OpenSpliceAI) models - those have never been run
  on OpenSplice, since their fixed input-window architecture would need the same
  flank-reconstruction work flagged as infeasible-without-more-care for Baeza-Centurion earlier.
