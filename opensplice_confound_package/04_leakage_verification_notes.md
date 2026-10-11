# Leakage verification for the fine-tuned-vs-pretrained OpenSpliceAI comparison

Before trusting any "fine-tuning reduces confound bias" claim, we checked directly whether the
fine-tuned OpenSpliceAI checkpoint used in these comparisons (`sl650_u16_ensemble` for the FAS
Exon_6 test; `stage1/stage2/stage5` for OpenSplice) could have been trained on the exact variants
we evaluated it on.

## FAS Exon_6 test set

- Training for `sl650_u16_ensemble` actually used `split_assignments_HEK_ALL_DESIGNS_70_10_20_v2_20260804.tsv.gz`
  - a **different, versioned** split file than the one used to build our own test CSV
  (`split_assignments_HEK_ALL_DESIGNS_70_10_20.tsv.gz`, no suffix). This was not previously
  documented as a discrepancy anywhere in the project.
- Direct empirical check (not just trusting docs): loaded both split files and our actual test CSV,
  intersected var_ids directly against the `train_pool` bucket that was actually fed into the
  training H5 file.
- **Result: zero overlap.** All 34,741 of our Exon_6 test var_ids are labeled `test` in the v2
  split file too (100% agreement with the unversioned file's test assignment for this design,
  despite the version mismatch) and none appear in `train_pool`.
- The replay/rehearsal mechanism used during fine-tuning (`REHEARSAL_H5`) mixes in genomic
  (MANE/chromosome) splice-site examples to prevent catastrophic forgetting - not MPRA data - so it
  cannot reintroduce test variants either.
- **Conclusion: clean, no leakage**, though this happened to be confirmed rather than guaranteed by
  the project's existing documentation - worth re-checking for any future split-file version
  changes, since multiple versions (`v2_20260723`, `v2_20260804`, `v3_20260727`, `v4_20260727`)
  coexist on disk.

## OpenSplice dataset

- OpenSpliceAI's `stage1/stage2/stage5` predictions used here are already restricted to `split ==
  "test"` per an exon-level 80/20 split (`opensplice_exon_train_test_split.csv`) built specifically
  for this dataset, independent of the FAS split files above.
- The manifest-building script for this design explicitly **excludes `gene=="FAS"` rows** (510
  rows) specifically to avoid overlap with the project's own FAS training data.
- **Caveat, not fully resolved**: the OpenSplice construct shares the same reporter vector backbone
  as the FAS MPRA data (though testing different genomic loci/exons) - a documented note in the
  manifest-building script's own docstring. This means OpenSplice is independent in terms of *which
  exons/genomic loci* are tested, but not necessarily fully independent in terms of *vector/flank
  context* - worth keeping in mind, though it does not constitute variant-level leakage.
