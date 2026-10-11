# GC/CpG and stop-codon confound analysis — final, calibrated version

## What this is

The paper (Liu, Zhang & Regev, Genome Biology 2026) shows SpliceAI/Pangolin/AlphaGenome rely on
two genomic "cheat codes" unrelated to real splicing: CpG enrichment and stop-codon depletion, used
as spurious proxies for "exon identity." We tested whether a model trained **directly on our own
real random-sequence MPRA data** (FAS Exon_6, HEK293), instead of distilling a black box trained on
genomic reference sequence, shows the same biases. This package is the final, methodology-checked
version of that analysis, superseding earlier uncalibrated drafts.

**Paper's exact method, which this replicates:** calibrate each model's raw predictions to measured
PSI via isotonic regression (logit scale, fit on the same data - matching their own stated
approach), then test error = calibrated_prediction − measured_PSI against each confound feature,
controlling for measured PSI, via ANCOVA (ours: OLS regression with a standardized-coefficient
version for cross-model comparability, since the paper's own text reports only significance
thresholds, not effect sizes).

## Headline results

**CpG (both metrics) and GpC control:** our model sits mid-pack among 8 models tested (itself, 2
SpliceAI/OpenSpliceAI variants, 4 AlphaGenome variants) — never the best, never the worst. The
single worst offender on every metric, consistently, is the plain **zero-shot** OpenSpliceAI
ensemble. GpC (their "specificity control," meant to show *no* association) is in fact
significantly associated with error for every model in our replication, including SpliceAI/
AlphaGenome — likely because our test set (34,741-69,482 variants) is far larger than their FAS
exon6 assay (5,976 sequences), giving more power to detect a real-but-smaller effect. Directionally
their claim still holds (GpC's effect is consistently ~40-65% the size of CpG's for every model),
but their stronger "no consistent association" claim does not replicate at our sample size.

**Stop codon:** same mid-pack pattern statistically, but the mechanism looks different from what
the paper found. Their models showed genuine 3-nucleotide (reading-frame) periodicity in the
stop-codon effect - i.e. real codon-frame awareness. Our model's analogous top filter shows **no
periodicity at all** (see file 04) - it looks like a frame-blind T/A-richness detector that
happens to statistically correlate with stop-codon count (more T's/A's mechanically raises the odds
of TAA/TAG/TGA-*like* triplets in any frame by chance), not a genuine codon-aware mechanism.

## File guide

- **02_confound_ancova_results_calibrated.tsv**: the main results table. Columns: `model,
  category, feature, n, raw_coefficient, raw_se, standardized_coefficient, p_value`. `feature` is
  one of `cpg_ratio` (general formula, all assays), `cpg_count` (paper's FAS-exon6-*specific*
  choice, due to "limited compositional diversity" in their near-WT mutagenesis assay - doesn't
  obviously apply to our fully-randomized library, but checked for rigor), `gpc_count` (specificity
  control), `stop_codon_count`. `category` distinguishes `real_data_trained` (us), `zero_shot`
  (never saw any real MPRA data), and `fine_tuned_reference` (also trained on real MPRA data - a
  much bigger pretrained model, not a fair "black box" comparison point, kept for context only).
  `standardized_coefficient` (effect per 1 SD of the feature) is the one to use for cross-model bar
  charts - `raw_coefficient` isn't directly comparable across models with different prediction
  variance.

- **03_confound_per_variant_data.csv**: per-variant data for building your own plots (e.g. the
  paper's Fig. 2A/3A binned-boxplot style). Columns: `var_id, psi_shrunk (measured), cpg_ratio,
  cpg_count, gpc_count, stop_codon_count, our_model_error_calibrated`. Only has our model's error
  (re-deriving each oracle's per-variant error would require re-merging the large source prediction
  files - ask if you need that too).

- **04_stopcodon_periodicity_check.json**: the reading-frame periodicity test on our top
  stop-codon-linked filter. `frame_variance_ratio` = fraction of the position-effect curve's
  variance explained by reading frame (0.0096 = 0.96% - negligible, no periodicity).
  `autocorrelation_lag_1_to_6`: decays smoothly with lag, no spike at multiples of 3 (a genuine
  frame effect would spike at lag 3 and 6).

- **05_filter_importance_ranking.csv**: all 32 learned convolutional filters in our model, ranked
  by ablation importance (drop in held-out Pearson r when that filter is zeroed out). Filter 22
  (rank 1) drives the CpG/GC story; filter 1 (rank 2) drives the stop-codon story.

- **figures/**:
  - `motif_rank1_filter22_GCrich.png` / `position_effect_rank1_filter22.png`: the top-ranked
    filter's sequence logo and position-dependent effect curve. Strongly G-rich, and the effect is
    **flat/position-independent** across the whole 476nt input - the signature of a composition-based
    confound, not a localized regulatory element.
  - `motif_rank2_filter1_TArich.png` / `position_effect_rank2_filter1.png`: the second-ranked
    filter (T/A-rich), linked to the stop-codon confound.
  - `filter1_periodicity_check.png`: filter 1's position-effect curve colored by reading frame -
    the three frame colors are visually indistinguishable (no periodicity), instead showing a
    smooth ~120-150nt oscillation unrelated to codon frame.

## Caveats to keep in mind

- All numbers are for **FAS Exon_6 / HEK293 only** (one design, one cell line). A 16-design pooled
  model is training separately; not yet checked for these same confounds.
- Calibration is fit **on the same data being evaluated** (in-sample) - matching the paper's own
  stated approach ("calibrated ... on the entire dataset"), not something we did differently, but
  still a real limitation of both analyses.
- The paper's own text gives no extractable coefficient to compare against numerically - only
  p-value thresholds (CpG: p<5e-3, stop codon: p<1e-20), which every model here clears easily given
  our much larger sample size. The cross-model table in file 02 (us vs. the same real
  SpliceAI/AlphaGenome models, same test variants, same methodology) is the closest thing to a
  head-to-head "how do we compare to the paper's models" comparison that's actually possible to
  compute.
