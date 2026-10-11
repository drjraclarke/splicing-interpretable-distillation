# Interpretable-distillation-on-real-data: handover bundle

## Background

The paper (Liu, Zhang & Regev, Genome Biology 2026) trains a small, fully interpretable
architecture ("distilled model": conv1d motif detector -> neural-basis position-effect layer ->
additive summation) to mimic SpliceAI/Pangolin/AlphaGenome on ~1M **synthetic** random-exon
sequences, labeled with each black-box model's own in-silico score. They then show the black-box
models rely on genomic confounders (CpG enrichment, stop-codon depletion) unrelated to real
splicing, and are blind to RNA structure effects.

We instead trained the **same unmodified architecture from scratch directly on our own real
random-sequence MPRA data** (FAS Exon_6 design, HEK293 cells) with real measured PSI as the
target - skipping the "distill a black box" step entirely. This tests whether a model that never
saw genomic reference sequence, and never mimicked a black box, still shows the same confounds.

## Paper's own reported numbers (for context - different dataset, read the caveat)

Quoted directly from the paper (Liu, Zhang & Regev, Genome Biology 2026):

| Quantity | Value | What it actually measures |
|---|---|---|
| SpliceAI/Pangolin/AlphaGenome vs. measured PSI, **FAS exon 6 assay** | Pearson r = **0.81-0.88** | Calibrated (isotonic regression) black-box predictions vs. real measured PSI, on **Baeza-Centurion et al. 2025's small point-mutagenesis library** (~5,976 unique sequences, small SNV/indel perturbations near wild-type) |
| Same, Liao et al. 2023 synthetic exon assay | Pearson r = 0.83-0.86 | Same idea, different (also real) assay |
| Same, MFASS assay | Pearson r = 0.37-0.44 | Same idea, fluorescence-based assay (they note this likely explains the lower number) |
| Distilled model vs. **the original DL model's own score** (not real PSI) | Pearson r = **0.94-0.99** | How well the interpretable distilled model replicates SpliceAI/Pangolin/AlphaGenome's *own* synthetic-sequence predictions - not a real-biology accuracy number at all |
| Sequence-motif-only variance explained (in the distilled model, vs. the full DL model) | 88-97% | Their argument that structure/interactions contribute little to what the DL models actually do |
| CpG confound ANCOVA | p < 5x10^-3 (all 3 models) | Same test design as our `05_confound_analysis_summary.json`, different dataset |
| Stop-codon confound ANCOVA | p < 1x10^-20 (all 3 models) | Same test design as ours |

**Important caveat - do not present the 0.81-0.88 FAS-exon-6 number as directly comparable to our
0.7315 or to file 02's numbers.** Their "FAS exon 6" is a *different experimental library*
(Baeza-Centurion's small mutagenesis-near-wild-type set) from ours (Lehner lab's fully-randomized
MPRA design) - same gene/exon, different assay, different sequence diversity, and their number is
also evaluated in-sample (calibrated and evaluated on the same data, no held-out split mentioned
for that specific figure) rather than on a genuinely disjoint test set the way ours is. It's useful
context for "how well can any model predict FAS exon 6 splicing at all," not a fair head-to-head.
File 02's zero-shot/fine-tuned numbers - which score the *same* models on *our exact* held-out test
variants - are the fair comparison.

## Headline result

Held-out test set: 34,741 real FAS Exon_6/HEK variants, never touched during training or
hyperparameter selection.

**Our model: Pearson r = 0.7315, Spearman r = 0.7244** (predicted vs. measured PSI).

## File 02: zero-shot model comparison (`02_zero_shot_model_comparison.tsv`)

Filters existing SpliceAI/OpenSpliceAI and AlphaGenome prediction tables down to the *exact same*
34,741 test variants, for a clean apples-to-apples comparison. **Important distinction (don't
conflate these two categories in a plot legend):**

| model | category |
|---|---|
| `osai_sl650_u16_ensemble` | **fine-tuned** on real MPRA data (bigger pretrained foundation model, additionally fine-tuned - not zero-shot) |
| `osai_pretrained_ensemble` | zero-shot (never trained on any real MPRA data) |
| `alphagenome_pt_*`, `alphagenome_jax_*` | all zero-shot |

Key numbers:
- True zero-shot models: Pearson 0.42-0.67, Spearman 0.42-0.53 - **all below our model**.
- Fine-tuned OpenSpliceAI: Pearson 0.806 (**above** ours) but Spearman 0.588 (**below** ours,
  0.724). Every other model in the table shows a big Pearson-vs-Spearman gap (~0.2-0.25); ours
  barely has one. Interpretation: our model's predictions preserve the correct *rank order* of
  variants almost as well as their linear magnitude - the other models' linear correlation looks
  better than their rank correlation actually supports.

Good plot ideas: grouped bar chart of Pearson and Spearman side by side per model, with the two
categories (zero-shot vs. fine-tuned) visually distinguished (color or a divider); or a
Pearson-vs-Spearman scatter with one point per model, highlighting how far each sits from the
diagonal.

## File 03: our model's raw predictions (`03_our_model_test_predictions.csv`)

Columns: `measured_psi`, `predicted_psi`. One row per held-out test variant (n=34,741). Good for a
predicted-vs-measured scatter/hexbin plot (matching the paper's own Fig. 1B style).

## Files 04-05: confound analysis (CpG / stop-codon "cheat codes")

The paper found SpliceAI/Pangolin/AlphaGenome rely on CpG enrichment and stop-codon depletion as
spurious "exon identity" proxies (genomic-training artifacts, not real splicing biology). We
tested whether *our* real-data-trained model shows the same biases, using their own
ANCOVA-style method: regress (predicted - measured) against a confound feature, controlling for
measured PSI.

- `04_confound_analysis_per_variant.csv`: columns `var_id, psi_shrunk (measured), predicted_psi,
  error, cpg_ratio, stop_codon_count` - all computed within each variant's own randomized/variable
  region (positions 234-297 of the 476nt input), not the constant flanks.
- `05_confound_analysis_summary.json`: the two regression results.
  - CpG: coefficient = **+0.039**, p ≈ 0 (n=34,741) - higher CpG -> inflated prediction. **Same
    direction as the paper's finding**, despite our model never seeing genomic sequence.
  - Stop codons: coefficient = **-0.022**, p ≈ 0 - more stop codons -> deflated prediction. Same
    direction as the paper.

Good plot idea: reproduce the paper's own Fig. 2A/3A style - box plots of `error`, binned by
measured PSI (x-axis, e.g. 5 bins of width 20%) and grouped/colored by cpg_ratio or
stop_codon_count bins.

## Files 06-07 + figures/: what did the model actually learn (motif interpretation)

We reused the paper's own interpretation code (`model_interpretation.py`, unmodified) to rank our
model's 32 learned convolutional filters by ablation importance (drop in held-out Pearson r when
that filter is zeroed out) and visualize each one as a sequence logo + position-effect curve,
matching their Fig. 1C format.

- `06_filter_importance_ranking.csv`: all 32 filters ranked by `importance_drop`.
- `figures/motif_rank{1-10}_filter{N}.png`: sequence logo for each of the top 10 filters (built
  from the top 5% of activating k-mers out of 2M random 20-mers, weighted PWM - same method as
  the paper's Methods).
- `figures/position_effect_rank{1-10}_filter{N}.png`: logo (top) + position-dependent contribution
  curve (bottom) for the same top 10 filters, via the paper's `calculate_first_order_shape_functions`
  + `plot_kernel`.

**Filter 22 (rank 1, importance_drop=0.055, by far the top feature)**: a strongly G-rich motif,
with a **flat, position-independent negative effect** across the entire 476nt input - consistent
with a genomic composition-based confound rather than a positionally-localized regulatory element.
This is the feature most directly responsible for the CpG/GC-richness confound found above.

**Filter 1 (rank 2, importance_drop=0.025)**: a T/A-rich motif, statistically linked to the
stop-codon confound above, but see the periodicity check:

- `07_filter1_periodicity_summary.json` / `figures/filter1_periodicity_check.png`: the paper found
  their models' stop-codon effect showed genuine 3-nucleotide (reading-frame) periodicity. We
  tested the same thing on filter 1: **no periodicity found** (between-frame variance = 0.96% of
  total; autocorrelation decays smoothly with lag, no spike at multiples of 3). Instead there's a
  smooth ~120-150nt oscillation, unrelated to reading frame.
- **Interpretation**: our model reproduces the *statistical symptom* of the stop-codon confound
  (the ANCOVA association) but not the *mechanism* (frame-awareness) the paper found in models
  trained on real genomic/coding sequence. Plausible explanation: more T's/A's mechanically raises
  the odds of TAA/TAG/TGA-*like* triplets occurring in any frame by chance, producing the
  statistical correlation without genuine codon-frame recognition.

Good plot idea: the periodicity PNG already shows this (position-effect curve colored by reading
frame, frames visually indistinguishable) - could be cleaned up for a slide.

## Caveats / things to remember when plotting

- All numbers here are for **FAS Exon_6 / HEK293 only** - a single design, one cell line. A
  16-design pooled model (all HEK) is training separately; these findings haven't yet been checked
  for generalization across designs.
- "Zero-shot" vs "fine-tuned" is a real, important distinction in file 02 - don't average them
  together or present the fine-tuned number as a "black box baseline."
- `psi_shrunk` = empirical-Bayes shrinkage-corrected PSI (the project's standard real-measured-PSI
  column), same across all files here.
