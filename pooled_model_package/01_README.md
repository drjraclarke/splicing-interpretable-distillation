# Pooled 16-design model — methodology, held-out performance, confound analysis

## What this model is

Same unmodified architecture as the paper's distilled model (`Conv1DNBM`: conv1d motif detector →
neural-basis position-effect layer → additive summation), but trained **from scratch directly on
real experimental MPRA data** (real measured PSI, not a black box's synthetic-sequence predictions)
- and, unlike the earlier single-design run, trained on **16 real FAS-family designs pooled
together** (Exon_6, all 6 `combi_1-6`, `i5_3`, and `i6_1/2/7/8/9/10/12/13`), not just Exon_6 alone.

**Why pool multiple designs**: to test whether the "train directly on real data" approach
generalizes across genuinely different randomization strategies (intron-only vs exon-wholesale vs
mixed) and sequence compositions, rather than being an Exon_6-specific fluke.

**Data scale**: 1,537,526 training / 384,199 held-out test variants, HEK293 cells only. Each
design's sequence was padded to a common 603nt length by inserting filler ('N', all-zero one-hot)
exactly at that design's own empirically-determined variable-region boundary - this keeps each
design's constant 5' flank aligned at position 0 and 3' flank aligned ending at position 602,
regardless of how much internal content differs between designs (see file 07 for each design's
exact boundary/padding amount).

**Training**: 10-trial Bayesian hyperparameter search (HyperOpt via Ray Tune, ASHA early-stopping
scheduler, max 50 epochs, grace period 10), on a single H100 GPU. File 02 has the winning trial's
exact raw (pre-transform) hyperparameters - note `batch_size` and `sequence_kernels` are stored
pre-transform (`hyperparameter_search.py` applies `2**batch_size` and maps `sequence_kernels` dict
→ list-of-tuples internally); the actual values used were **batch_size=128**,
**sequence_kernels=[(8 filters, kernel_size=7)]** - a notably smaller/simpler architecture than the
Exon_6-only model's winning config (which used a single (32, 20) kernel).

## Held-out test performance (file 03, 04)

**Overall: Pearson r = 0.8526, Spearman r = 0.8418** (n=384,199, genuinely held out - these
variants were never seen during training or hyperparameter selection). This closely matches the
internal validation number reached during training (0.8525), so no overfitting to that split.

**But this aggregate number is misleading on its own** - per-design breakdown (file 04) shows a
sharp split:

| Design group | Training set size (per design) | Held-out Pearson r |
|---|---|---|
| Exon_6, combi_1-6, i5_3 (8 "large" designs) | 100K-305K each | **0.70 - 0.79** |
| i6_1/2/7/8/9/10/12/13 (8 "small" designs) | 1K-26K each | **-0.04 to 0.42**, mostly ~0 |

The 8 well-performing designs make up ~93% of the test set (i5_3 alone is 20%), so the strong
overall number is dominated by them - the model essentially failed to learn anything useful for
several of the low-data designs (worst: i6_1 and i6_10, ~1,000-1,300 training examples each).
**Conclusion: pooling multiple designs into one model works, but is data-hungry per design** - it
does not give you a splicing "grammar" that transfers for free to designs with only a few thousand
examples.

## Confound analysis (files 05, 06, 08) - the CpG/stop-codon "cheat codes" from the paper

Same methodology as before (isotonic-regression calibration on the logit scale, then ANCOVA:
error ~ feature + measured_psi), computed per-variant using each variant's own design-specific
variable-region location (shifted to account for the padding inserted at that boundary - see file
07's `pad_amount` column).

**The headline finding - comparison to the single-design (Exon_6-only) model (file 08):**

| Feature | Exon_6-only model | Pooled 16-design model | Pooled, well-generalizing designs only |
|---|---|---|---|
| CpG ratio | 0.0148 | -0.0011 | -0.0011 |
| CpG count | 0.0381 | 0.0020 | 0.0014 |
| GpC count | 0.0238 | 0.0044 | 0.0043 |
| Stop-codon count | -0.0319 | **+0.0099** | +0.0078 |

Training on 16 pooled, diverse designs **reduces the CpG/GC confound coefficient by 10-30x**
compared to training on Exon_6 alone - essentially negligible now. The stop-codon coefficient
doesn't just shrink, it **flips sign** (paper's models, and our Exon_6-only model, show a
NEGATIVE association - more stop codons → deflated prediction; the pooled model shows a small
POSITIVE one instead). Results are nearly identical whether restricted to the 8 well-generalizing
designs or all 16, so this isn't an artifact of the failing small designs adding noise.

**Interpretation**: the CpG/GC reliance found training on Exon_6 alone looks like it was a
design-specific shortcut - with only one design's worth of compositional variation available, CpG
content was a convenient (if spurious) predictive proxy. With 16 designs' worth of diverse
composition, that shortcut stops being reliably useful, and the model appears to rely on more
genuine, generalizable signal instead. (p-values are still ~0 for every feature given n=384,199 -
statistical significance is trivially easy to reach at this scale even for a negligible effect;
the magnitude comparison above is what actually matters, not significance.)

## File guide

- **02_winning_trial_raw_config.json**: winning trial's hyperparameters, pre-transform (see note
  above on batch_size/sequence_kernels).
- **03_held_out_test_predictions.csv**: `design, var_id, psi_shrunk (measured), predicted_psi` for
  all 384,199 held-out test variants - for your own scatter/hexbin plots.
- **04_held_out_per_design_performance.tsv**: `design, n, pearson_r, spearman_r` per design - the
  large-vs-small-design split described above.
- **05_confound_ancova_results.tsv**: full ANCOVA table, `scope` (all_16_designs /
  well_generalizing_designs_only) × `feature` (cpg_ratio, cpg_count, gpc_count,
  stop_codon_count), with `n, raw_coefficient, standardized_coefficient, p_value`.
- **06_confound_per_variant_full.csv**: all 384,199 held-out test variants, one row each, with
  `design, var_id, psi_shrunk (measured), predicted_psi, predicted_calibrated, error, cpg_ratio,
  cpg_count, gpc_count, stop_codon_count` - for your own plots (e.g. the paper's Fig. 2A/3A
  binned-boxplot style), or to pull `cpg_ratio` (or any other column) for every variant directly.
  38MB.
- **07_design_metadata.tsv**: per-design `seq_len, variable_start, variable_end, n, pad_amount`
  (pre-padding coordinates - see the note in the analysis script about the `+ pad_amount`
  adjustment needed to locate the variable region in the final padded sequence).
- **08_confound_comparison_single_vs_pooled.tsv**: the small headline comparison table above, as a
  standalone file for easy reference/plotting.

## Related, not included here

The equivalent motif-interpretation package (sequence logos, position-effect curves, stop-codon
periodicity check) for the **single-design (Exon_6-only) model** was assembled separately as
`gc_stopcodon_confound_package/` - not yet redone for this pooled model.
