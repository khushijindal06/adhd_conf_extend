# ADHD-200 Experimental Results Audit

Audit of the initial real-data run on 896 ADHD-200 subjects, performed before
starting Phase 3. Code-level findings below are verified directly against
the scripts in `code/`; items marked **PENDING** require the real-run
artifacts (configs, result CSVs, prediction files, `processed_index.csv`)
from the actual Kaggle run to complete.

Reported real-data results under audit:

| Model | Validation AUROC |
|---|---|
| Original ADHDNet (baseline) | 0.9516 |
| A4 — conventional MHA TRM | 0.7019 |
| A4-new — Pairwise TRM | 0.6973 |
| A5-new — Pairwise TRM + anatomical attention | 0.6799 |
| M0 — without consistency | 0.7394 |
| M1 — with consistency | 0.7552 (higher cosine representation stability, lower ADHD sensitivity/F1) |

## 1. Baseline comparability audit

**Critical finding: the 0.9516 baseline is not MRI-only.** `code/model.py`'s
`ADHDNet` (used by `code/train.py`, which produced 0.9516) has signature
`forward(self, vol, pheno)` — it fuses the MRI embedding with a 5-feature
phenotype vector (`age, sex, med_status, site_id, adhd_index`) via learned
gated fusion (`model.py:169-178`). Every Phase-2 script
(`phase2_trm_ablation.py`, `phase2_consistency.py`) loads
`(vol, pheno, label)` from the same `ADHDDataset` but **discards `pheno`**
— `model(vol)` is called with no phenotype argument, and
`ADHDProposedModelV2` defaults `use_phenotype=False` (confirmed in both
files' `ARCH_KWARGS`).

| | Input |
|---|---|
| Original ADHDNet (0.9516) | MRI + 5 phenotype features |
| A3 / A4 / A4-new / A5-new / M0 / M1 (0.68–0.76) | MRI only |

**0.9516 vs 0.68–0.76 compares two different input modalities, not two
architectures.** This alone could account for most or all of the apparent
gap.

**Leakage risk in the phenotype features.** Two of the five fused features
are plausible near-label-proxies in the standard ADHD-200 phenotypic
coding:
- `adhd_index` — a continuous clinical symptom-severity score (Conners'
  ADHD Index), typically strongly separating diagnosis groups almost by
  construction.
- `med_status` — medication status is frequently coded with a fixed
  sentinel for typically-developing controls and a non-sentinel value only
  for diagnosed subjects, which can make it near-deterministic of the
  label.

`analysis/check_phenotype_leakage.py` computes univariate AUROC (and a
`med_status` breakdown by label, and an `adhd_index` distribution by
label) for all five phenotype features directly against your real
`processed_index.csv`. **PENDING: result of running this script on the
real data** — an AUROC near 1.0 or near 0.0 for `med_status` or
`adhd_index` alone would confirm leakage.

**The correct fair baseline already exists in the codebase and appears
unused so far.** `code/ablation_study.py`'s "A2 w/o phenotype" variant
(`ADHDNetMRIOnly`) is the true MRI-only lightweight baseline. **Running it
on the real 896-subject data is the single highest-priority missing data
point** before any Phase-3 discussion — it's the only apples-to-apples
reference for A3/A4/A4-new/A5-new/M0/M1.

**Split/preprocessing mechanics — identical by construction, identity not
yet independently verified.** `train.py`, `ablation_study.py`,
`phase2_trm_ablation.py`, and `phase2_consistency.py` all import the same
`dataset.py`/`ADHDDataset` (same `.npy` loading, same flip-only
augmentation, same normalization) and all build the split via
`StratifiedShuffleSplit(test_size=0.2, random_state=42)` after the
identical `site != 'Pittsburgh'` filter. This should produce bit-identical
subject-ID splits across scripts — but only if `processed_index.csv`'s row
order was unchanged between runs. **PENDING: diff the actual
`train_subject_ids.csv`/`val_subject_ids.csv` files** from each script's
output directory to confirm, rather than relying on "same seed" alone.

**Evaluation procedure — consistent, no leakage found.** `evaluate.py`
correctly reuses `train.py`'s train-only `fit_stats.json` and exact saved
`val_ids` (no re-fit contamination). Metric definitions (`roc_auc_score`,
0.5 threshold) match the Phase-2 scripts' conventions.

**Minor, low-impact inconsistency:** `train.py`/`ablation_study.py`/
`cross_validate.py` still use a `best_auc = 0.0` sentinel — the same bug
class fixed in the Phase-2 scripts was never backported here. Given the
baseline's final AUROC (0.9516) is far above 0, this almost certainly
didn't affect the result, but these scripts also lack the config-fingerprint
resume-safety guard Phase 2 now has.

## 2. Four TRM-variant audit

From code (param counts verified via direct instantiation + parameter
count, not yet checked against real-run artifacts — see pending asks
below):

| Tag | `trm_type` | `use_attention` | Params |
|---|---|---|---|
| A3 tokens+FastKAN | `none` | False | 105,424 |
| A4 +MHA TRM (Phase 1) | `mha` | False | 113,968 |
| A4-new +Pairwise TRM | `pairwise` | False | 113,937 |
| A5-new +Pairwise+AnatAttn | `pairwise` | True | 114,482 |

All four share the identical `ADHDProposedModelV2` encoder/tokenizer/
FastKAN/classifier head — only the relationship-module stage differs. A4's
`mha` path literally imports and reuses Phase-1's original
`MultiheadAttention`-based TRM class from `proposed_model.py`
(`phase2_model.py:24,118` — not a reimplementation), so A4 vs A4-new is a
clean "mechanism-only" ablation (near-identical param counts, 113,968 vs
113,937).

**The "A3/A4 labeling discrepancy" — most likely explanation.** The
codebase defines "A3"/"A4" **three separate times**, for different
things:

1. `ablation_study.py` (Phase-1 baseline ablations): A3 =
   `ADHDNetConcatFusion`, A4 = `ADHDNetStandardConv` — about the original
   ADHDNet's fusion/conv design, unrelated to TRMs.
2. `proposed_ablation_study.py` (Phase-1 token architecture,
   `ADHDProposedModel`): A3 = Tokens+FastKAN, A4 = +TRM (the original MHA
   TRM).
3. `phase2_trm_ablation.py` (Phase-2 Step 1, `ADHDProposedModelV2`): A3 =
   tokens+FastKAN, A4 = +MHA TRM (a deliberate re-verification control
   reusing #2's module), A4-new/A5-new = the actual new contributions.

Scripts #2 and #3 define nearly-identical A3/A4 *by design* (so #3 can
confirm #2's numbers transfer to the new wrapper) — but they're separate
model classes, separate checkpoints, separate output directories. **If the
real execution log shows two different AUROCs or param counts both
labeled "A3" or "A4," it is almost certainly because the log merges output
from two of these three scripts under the same short label**, not a bug in
either script.

**PENDING to finish verification against the real run:** `A3_config.json`,
`A4_config.json`, `A4-new_config.json`, `A5-new_config.json`,
`trm_ablation_results.csv`, and the four `*_val_predictions.csv` files.

## 3. Consistency-learning diagnostics (M0 vs M1)

Split into two parts:

**(a) Computable directly from existing prediction files.**
`val_predictions_m0.csv` / `val_predictions_m1.csv` (per-subject
`y_true`/`y_prob`/`y_pred`, already saved by `phase2_consistency.py`) give
ROC curve, PR curve, confusion matrix + per-class sensitivity/specificity,
probability histograms by class, ECE, and Brier score directly — read-only
analysis of existing predictions, no new experiment. No threshold is
tuned on this data (fixed 0.5 decision rule, matching every other script
in the package). **PENDING: the two CSV files from the real run.**

**(b) `analysis/extract_representations_m0_m1.py`** — a new, inference-only
script (no gradient, no training, no architecture change) that loads
`best_m0_model.pt`/`best_m1_model.pt` and extracts each validation
subject's 128-d representation via the model's own existing
`encode_representation()` method. Produces:
- per-subject embeddings (`embeddings_val_{m0,m1}.csv`)
- within-subject stability per subject across repeated
  `consistency_augment()` draws (`within_subject_stability_{m0,m1}.csv`)
- a diagnostic summary (`representation_diagnostics.csv`): effective rank
  (SVD-entropy collapse indicator), mean per-dimension variance, mean
  pairwise cosine similarity within-class vs between-class, the
  within/between separation gap, and mean within-subject stability.

Verified end-to-end against synthetic checkpoints before being handed off
(see `analysis/` script docstring); **PENDING: run against the real
`best_m0_model.pt`/`best_m1_model.pt`.**

A pattern worth watching once real numbers are in: M1's lower
sensitivity/F1 alongside higher stability is the textbook signature of
representation collapse — stability rising while between-class separation
(part (b)'s between-class similarity, and part (a)'s AUPRC/sensitivity)
falls together. These diagnostics are designed to distinguish "collapsed
toward one point" from "legitimately smoother but still discriminative."

## 4. Proposed controlled MRI-only comparison (design only — not executed)

**Augmentation decision.** Original-baseline/A3/A4/A4-new/A5-new were all
trained with `augment=True` (random flips); M0/M1 use `augment=False` by
necessary design (View 1 must stay deterministic so View 2's
`consistency_augment()` perturbation is isolated — a necessary
objective-specific difference, not a bug, so it is preserved rather than
changed). The one real mismatch is **A5-new vs M0**, which should be
architecture-identical minus the consistency loss but currently differs in
augmentation too. Recommended minimal fix: **one new run** — A5-new
retrained under `augment=False` — giving a clean 3-way
A5-new(augment=False)/M0/M1 comparison, rather than retraining all four
Step-1 variants.

| Tier | Models | Augmentation | Purpose |
|---|---|---|---|
| 1 — mechanism ablation | Original MRI-only (A2), A3, A4, A4-new, A5-new | `True` (existing results, no rerun needed) | Which relationship module is best, holding everything else fixed |
| 2 — objective ablation | A5-new (new run), M0, M1 | `False` | Does the consistency loss help, on top of the best mechanism from Tier 1 |

**Evaluation protocol per model:**
- Repeated stratified CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
  random_state=42)` → 10 folds/model, reusing `cross_validate.py`'s
  existing loop structure (no architecture change, just swap the model
  class in).
- Leave-one-site-out: reuse `cross_site_eval.py`'s existing pattern,
  restricted to sites where `site_label_counts.csv` (already produced by
  the Phase-2 scripts) shows both classes present with a usable minimum
  count (suggested floor: ≥5 per class). **PENDING: the real
  `site_label_counts.csv`** to confirm which sites qualify.
- Reduced per-fold budget (matching `cross_validate.py`'s own convention):
  `EPOCHS=60, PATIENCE=10`, vs. the `150/15` used for the single-split
  headline numbers.

**Compute requirements (estimate — needs calibration):**
- Training runs: 7 models × 10 CV folds = 70, + 7 models × ~4 eligible
  sites = 28 → **~98 total training runs**.
- Parametric estimate only (no access to real per-epoch timing): this
  model is tiny (~114K params, 0.2 GFLOPs) on 64³ volumes — on a Kaggle
  T4/P100, roughly 2–5 s/epoch, ~30–40 epochs/run typical early-stopping
  point → **~3–6 GPU-hours total**; CPU would likely be 10–20× that and
  impractical for the full grid. **PENDING: actual per-epoch time from the
  existing single-split runs** (visible in `training_history_*.csv`
  timestamps) to replace this estimate with a real number. Recommend
  piloting with `n_repeats=1` (35 runs instead of 98) first to confirm
  timing before committing to the full repeated design.

## 5. Preservation

Nothing in this audit read, modified, or deleted any checkpoint, CSV, or
figure from the real run — this is static code review of the package in
`code/`, plus two proposed (inference-only / read-only) scripts in
`analysis/`. No architecture or model-module changes were made or
proposed. Phase 3 has not been started.

## Outstanding items to close this audit

1. Run `analysis/check_phenotype_leakage.py` against the real
   `processed_index.csv` and record the result here.
2. Run `code/ablation_study.py`'s A2 (`ADHDNetMRIOnly`) on the real data —
   the missing fair MRI-only baseline.
3. Share: the four Step-1 `*_config.json` + `trm_ablation_results.csv` +
   `*_val_predictions.csv`; `val_predictions_m0.csv`/`val_predictions_m1.csv`;
   `site_label_counts.csv`; the raw execution-log excerpt showing the
   apparent A3/A4 discrepancy.
4. Run `analysis/extract_representations_m0_m1.py` against the real
   `best_m0_model.pt`/`best_m1_model.pt` checkpoints.
5. Decide Tier-2 scope (minimal: 1 new A5-new run vs. full: retrain all
   four Step-1 variants under `augment=False`) and approve the CV/LOSO
   compute budget before execution.
