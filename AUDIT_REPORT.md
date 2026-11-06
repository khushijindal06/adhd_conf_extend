# ADHD-200 Experimental Results Audit

Audit of the initial real-data run on 896 ADHD-200 subjects, performed before
starting Phase 3. Code-level findings below are verified directly against
the scripts in `code/`; items marked **PENDING** require the real-run
artifacts (configs, result CSVs, prediction files, `processed_index.csv`)
from the actual Kaggle run to complete.

**Update:** the actual Kaggle notebook (and its real execution log for all
896 subjects) has since been reviewed directly. Findings below are updated
accordingly; items still marked **PENDING** need a fresh run of the newly
added notebook cells (phenotype leakage check, M0/M1 embedding extraction,
A2 MRI-only baseline) rather than code-level speculation.

**Update 2:** a separate audit module, `code/experimental_audit.py`, now
implements every audit utility described below as executable code
(baseline-input introspection, split-identity verification, TRM-variant
verification, log-consistency checking, calibration/ROC-PR/embedding
diagnostics, and a repeated-CV + leave-one-site-out controlled-comparison
harness covering all 7 models). It has been tested with synthetic data
only (13 tests, A-M, all passing — see
`experimental_audit_sanity_output.txt`) and has **not** been run against
real data or real compute. See "Which analyses need existing checkpoints
vs. new training" at the end of this report for exactly what each piece
needs to produce real numbers.

**Update 3:** the user ran this module's Test G against real data on
Kaggle (GPU) and hit a device-mismatch runtime error — fixed below (§6).
Fixing it required two new tests (Test L, Test M) that a prior,
unverified version of the embedding-collapse check would have passed
incorrectly; writing Test L's negative control caught a real
methodological bug (naive in-sample centroid-projection AUC is
overoptimistic in a ~128-dimension/~180-sample regime) and the fix is
now in the diagnostic function itself, not just the test. See §6 for the
full account and §7 for the resulting read of the user's real M0/M1
representation numbers.

Reported real-data results under audit:

| Model | Validation AUROC |
|---|---|
| Original ADHDNet (baseline, MRI+phenotype) | 0.9516 |
| A3 — tokens+FastKAN, no TRM | 0.6930 |
| A4 — conventional MHA TRM | 0.7019 |
| A4-new — Pairwise TRM | 0.6973 |
| A5-new — Pairwise TRM + anatomical attention | 0.6799 |
| M0 — without consistency | 0.7394 (stability 0.4747 ± 0.5873) |
| M1 — with consistency | 0.7552 (stability 0.9371 ± 0.1046, lower ADHD sensitivity/F1) |

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

**Split identity across stages — CONFIRMED from the real log.** The
notebook's three stages reported matching val-set counts: Baseline
(716 train/180 val, ADHD=70/Control=110), TRM (716/180, label dist.
`{0: 110, 1: 70}`), Consistency (716/180, `{0: 110, 1: 70}`). TRM and
Consistency additionally printed the **identical** dataset fingerprint —
`index_hash=93bb5480ba9c... split_hash=e0343d42c716...` — cryptographic
proof those two stages trained/evaluated on exactly the same 716 train /
180 val subjects. (Note: the notebook version actually run does **not**
filter out the Pittsburgh site — all three stages used the full,
unfiltered 896 subjects — unlike the standalone `code/` package audited
below, which does filter it. This is internally consistent across all
three stages of the real run, just a difference from `code/`'s default.)
Baseline's script has no fingerprinting of its own, but recomputes the
identical `StratifiedShuffleSplit(test_size=0.2, random_state=42)` over
the identical, unfiltered file — deterministic by construction given
unchanged row order. **Recommended final check:** diff
`baseline/val_index.csv`'s subject_id column against
`trm/phase2/val_subject_ids.csv` directly, to remove all doubt.

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

Param counts verified via direct instantiation AND against the real
notebook's own final aggregate table (identical in both):

| Tag | `trm_type` | `use_attention` | Params | Real AUROC |
|---|---|---|---|---|
| A3 tokens+FastKAN | `none` | False | 105,424 | 0.6930 |
| A4 +MHA TRM (Phase 1) | `mha` | False | 113,968 | **0.7019** |
| A4-new +Pairwise TRM | `pairwise` | False | 113,937 | 0.6973 |
| A5-new +Pairwise+AnatAttn | `pairwise` | True | 114,482 | 0.6799 |

All four share the identical `ADHDProposedModelV2` encoder/tokenizer/
FastKAN/classifier head — only the relationship-module stage differs. A4's
`mha` path literally imports and reuses Phase-1's original
`MultiheadAttention`-based TRM class from `proposed_model.py`
(`phase2_model.py:24,118` — not a reimplementation), so A4 vs A4-new is a
clean "mechanism-only" ablation (near-identical param counts, 113,968 vs
113,937).

**The "A3/A4 labeling discrepancy" — RESOLVED. It is a console-log
artifact, not a data or training bug.** The real notebook's live
per-variant progress block shows only **3** `===== header =====` blocks
for 4 variants — the "A4 +MHA TRM (Phase 1)" header line is missing from
the captured stream, so A4's own result line (`AUROC=0.7019,
params=113,968`) prints directly under the **A3** header, creating the
appearance that A3 produced that result. It did not: `main()`'s final
aggregate table is built by reading each variant's own persisted
`{tag}_results.csv` back from disk (not from in-memory loop state) and
shows all four rows present and distinct, with parameter counts matching
exactly what `VARIANTS` defines for each tag (table above) — internally
consistent and matching the task's own summary numbers exactly (0.7019 =
"Conventional MHA TRM" = A4; 0.6973 = "Pairwise TRM" = A4-new; 0.6799 =
"+anatomical attention" = A5-new). The task's own top-level attribution
was already correct; only the mid-run console text (one dropped header
line, 3 lines) looked inconsistent. Separately: this explains why the
codebase's three different "A3"/"A4"-numbered ablation studies
(`ablation_study.py`, `proposed_ablation_study.py`,
`phase2_trm_ablation.py` — detailed in the previous version of this
report) were a plausible but, in the end, NOT the actual cause here;
worth keeping in mind if a similar-looking discrepancy appears from a
different log in the future.

`run_sanity_check()` was skipped on this real run (`--code-sanity` not
passed), so its Test A — which asserts each tag's parameter count against
`expected_params = {'A3': 105424, 'A4': 113968, 'A4-new': 113937,
'A5-new': 114482}` before any real training starts — did not get a
chance to catch/confirm this ahead of time. Recommend enabling
`--code-sanity` on the next run for the extra guard (synthetic data only,
near-zero added cost).

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
(see `analysis/` script docstring), and a notebook-adapted copy has been
added directly to the real Kaggle notebook (see below) — re-verified
end-to-end against that notebook's own module code before being added.
**PENDING: run the added cell against the real
`best_m0_model.pt`/`best_m1_model.pt`.**

**Real numbers already visible from the log, worth flagging now:** M0's
own within-subject stability is 0.4747 ± 0.5873 — a standard deviation
roughly as large as the mean, meaning M0's representations are themselves
quite unstable under `consistency_augment()` perturbation (expected,
since M0 was never trained to resist it). M1's is 0.9371 ± 0.1046 — both
a much higher mean AND a much tighter spread. That gain is real and
substantial, not noise. The open question this report flagged is whether
M1's accompanying sensitivity/F1 drop (0.4429/0.5345 vs M0's 0.7286/
0.6415) reflects genuine representation collapse (discriminative
information traded away for stability) or is confined to the decision
threshold — M1's lower sensitivity/F1 alongside much higher stability is
the textbook signature to check for collapse, and the newly-run
embedding diagnostics (effective rank, within- vs between-class
separation gap) are what will answer it directly.

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

**Now implemented, not just designed.** `code/experimental_audit.py`'s
`MODEL_SPECS` dict and `run_controlled_comparison()` function implement
exactly this design as runnable code: all 7 models (`baseline_mri_only`
→ `ADHDNetMRIOnly`, `A3`/`A4`/`A4-new`/`A5-new`/`M0` → `ADHDProposedModelV2`
with the matching `trm_type`/`use_attention`, `M1` → the same plus the
two-view consistency loss, imported unmodified from
`phase2_consistency.py`), each trained with its own correct
`augment` setting, through one shared, fingerprint-guarded, resumable
`train_one_fold()` trainer. `build_loso_folds()` excludes any site
below a configurable per-class minimum (default 5; verified in
synthetic Test I/J to correctly exclude a single-class site). Verified
end-to-end on synthetic data (Test K trains all 7 specs for one fold
each; Test J runs the full CV+LOSO orchestration for one model) — **not
yet run on real data.**

**Evaluation protocol per model:**
- Repeated stratified CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
  random_state=42)` → 10 folds/model, reusing `cross_validate.py`'s
  existing loop structure (no architecture change, just swap the model
  class in).
- Leave-one-site-out: reuse `cross_site_eval.py`'s existing pattern,
  restricted to sites where both classes are present in a usable count.
  **CONFIRMED from the real `site_label_counts.csv`**: KKI, NYU,
  NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3 all have both classes in
  double digits on both the train and val side — **7 eligible sites**.
  Pittsburgh (train: 63 control/2 ADHD; val: 19 control/2 ADHD) and WashU
  (47 control/0 ADHD everywhere) do not have a usable ADHD count and
  should be excluded.
- Reduced per-fold budget (matching `cross_validate.py`'s own convention):
  `EPOCHS=60, PATIENCE=10`, vs. the `150/15` used for the single-split
  headline numbers.

**Compute requirements:**
- Training runs: 7 models × 10 CV folds = 70, + 7 models × 7 eligible
  sites = 49 → **119 total training runs** (revised up from the earlier
  ~98 estimate now that the real site breakdown gives 7 eligible sites,
  not ~4).
- From the real run's own per-variant timings (Step-1 variants: 177–329s
  at their observed early-stopping points of 23–43 epochs out of a 150
  max; baseline took 101 epochs end to end) — roughly 3–8 s/epoch
  observed — at a conservative ~3s/epoch and a typical ~30-epoch early
  stop, that's **~3 GPU-hours** for the full 119-run matrix on the same
  GPU this notebook already ran on. Recommend piloting with `n_repeats=1`
  (49 CV runs instead of 70) first to confirm timing before committing to
  the full repeated design.

## 5. Preservation

Nothing in this audit read, modified, or deleted any checkpoint, CSV, or
figure from the real run — this is static code review of the package in
`code/`, plus two proposed (inference-only / read-only) scripts in
`analysis/`, plus `code/experimental_audit.py` (see below), tested only
against synthetic data in its own temp directories. No architecture or
model-module changes were made or proposed. Phase 3 has not been started.

## 6. Audit module bug fix (device handling)

**The bug.** `code/experimental_audit.py`'s Test G built a fresh
`ADHDProposedModelV2` for its synthetic embedding-extraction check
without moving it to `DEVICE`:
```python
model = ADHDProposedModelV2(trm_type='pairwise', use_attention=True)   # stayed on CPU
```
while `extract_val_embeddings()`/`within_subject_stability()` move the
MRI batch to `DEVICE` (`vol = vol.to(DEVICE)`) before calling
`encode_representation()`. On this sandbox (CPU-only, `DEVICE == 'cpu'`)
both land on the same device and the bug was invisible; on the user's
real Kaggle GPU run (`DEVICE == 'cuda'`), `model` stayed on CPU while
`vol` moved to CUDA, exactly as reported.

**The fix, applied in two places (belt and suspenders), not one:**
1. Test G itself now does `.to(DEVICE)` when building the model.
2. `extract_val_embeddings()` and `within_subject_stability()` — the two
   functions meant to be called directly against a real, freshly-loaded
   checkpoint, not only through Test G — now call `model = model.to(
   DEVICE)` defensively at their own entry point, so a future caller
   that forgets to move the model can't reintroduce this bug. `.to()` on
   a tensor/module already on the right device is a no-op, so this adds
   no behavior change for a model that's already correctly placed.

**Rerun of the complete synthetic suite, as required** ("do not claim
completion if a test fails"): all tests now pass, **A through M** (two
new tests were added alongside the fix — see §7 below for why). Full
output: `experimental_audit_sanity_output.txt`. Also re-verified from a
fresh, isolated extraction of the repo's `code/` directory (self-
contained) and separately against the real notebook's own
`_adhd200_combined_*` modules (not just the standalone package) — both
pass identically. This sandbox has no GPU, so the exact CUDA/CPU
mismatch Test G hit cannot be reproduced here; the fix is verified
logically (device placement is now explicit and idempotent, not
inferred from the environment) and should be re-confirmed on the user's
own GPU Kaggle run.

## 7. Representation diagnostics: reading the real M0/M1 numbers

The user reported, from a real run of this module's embedding
diagnostics against the actual `best_m0_model.pt`/`best_m1_model.pt`:

| | M0 | M1 |
|---|---|---|
| Within-subject stability | 0.4263 | 0.9324 |
| Between-class cosine similarity | 0.0017 | 0.7499 |
| Class-separation gap | 0.2069 | 0.0706 |
| (for reference) Classification AUROC | 0.7394 | 0.7552 |

**One fact worth stating up front: M1's classification AUROC is not
worse than M0's — it's slightly better** (0.7552 vs 0.7394). AUROC is
threshold-independent; if the representation had genuinely lost
class-discriminative information, AUROC would be expected to degrade
toward 0.5 alongside the raw similarity numbers, not hold up or improve.
That it didn't is itself evidence against "genuine class-discrimination
loss" as the explanation for M1's lower sensitivity/F1 — those are both
threshold-dependent (fixed 0.5 cutoff), so a drop there alongside a
*held* or *improved* AUROC points toward a shifted decision boundary
/ calibration effect, not a destroyed signal.

**Reading the raw numbers**: M0's between-class similarity (0.0017)
is close to zero — different-class embeddings are nearly orthogonal on
average — and its within-class similarity is modest too (≈0.2086,
i.e. gap + between ≈ 0.2069 + 0.0017), consistent with M0 producing
relatively diffuse, loosely-clustered embeddings overall (which lines up
with its own *low* within-subject stability of 0.4263 — M0's
representations aren't very reproducible under small perturbations
either). M1's between-class similarity (0.7499) is high — even
different-class embeddings are now quite similar to each other — and
its within-class similarity is higher still (≈0.8205). That pattern (both
within- and between-class similarity rising together, while within-
subject stability also rises) is exactly the signature of the two
hypotheses the user asked to be distinguished:
- **Genuine class-discrimination loss**: the consistency objective
  destroyed the class-relevant signal in the representation.
- **Global embedding alignment**: the consistency objective pulled every
  representation (regardless of class) toward a more similar shared
  region/direction of the embedding space — which mechanically raises
  *both* within- and between-class raw cosine similarity together
  (since raw cosine similarity is sensitive to any shared component
  common to all vectors), without necessarily destroying the relative,
  class-specific structure.

**The five-part check now implemented to tell them apart**
(`representation_collapse_diagnostics()`, wired automatically into
`representation_quality_report()`'s output as `collapse_diagnostics`):
centered pairwise similarity (removes any shared global direction before
comparing), class-centroid Euclidean distance + cosine similarity +
a scale/shift-invariant Fisher-like ratio, embedding norms per class
(rules out "collapsed toward the origin" as a trivial alternative), and
a **cross-validated** linear centroid-projection AUC (fits the
discriminant axis on a train fold, scores a held-out fold — see §6 note
below on why this has to be cross-validated). Verified in Test L against
two engineered scenarios with known ground truth: a genuinely
well-separated pair of classes with a large shared offset added to every
embedding (simulating "global alignment") correctly shows a small raw
gap but a large centered gap, a high Fisher ratio, and a cross-validated
projection AUC of 1.0000; two classes drawn from the *identical*
distribution (simulating "genuine collapse", no class signal at all)
correctly shows both the raw and centered gap near zero, a low Fisher
ratio, and a projection AUC near chance (0.5375 in the actual test run).

**A methodological fix this uncovered, independent of the device bug**:
the first version of the centroid-projection AUC fit the discriminant
axis and scored it on the *same* samples (in-sample). Test L's negative
control — two classes from the identical distribution, which must score
near 0.5 — instead scored near 1.0, because with ~128 embedding
dimensions and a finite sample, fitting and evaluating a centroid
discriminant on the same data is optimistically biased (the axis is
chosen, using the data, to maximize exactly the split being measured).
This is precisely the regime the real checkpoints run in (~180 real
validation subjects, 128-d embeddings), so an in-sample version of this
check could have overstated how much discriminative signal survived in
M1. The fix — stratified k-fold, fit on train, score on held-out, pooled
— is in `_cv_centroid_projection_auc()` and is what `representation_
collapse_diagnostics()` now actually calls.

**What this means for M0 vs M1, honestly**: the hypothesis that M1's
pattern (rising within- *and* between-class similarity together, rising
stability, held/improved AUROC) reflects global alignment rather than
genuine collapse is well-supported by the circumstantial evidence above
(held AUROC; within- and between-class similarity moving together rather
than between-class alone cannibalizing within-class). **This report does
not yet have the definitive answer** — that requires running the now-
fixed `representation_quality_report()` (or `representation_collapse_
diagnostics()` directly on the real M0/M1 embeddings) against the actual
checkpoints, which needs the embedding-extraction cell re-run with the
corrected module. The centered gap, Fisher ratio, and especially the
cross-validated projection AUC on the *real* embeddings will say
definitively whether M1's class information survived (projection AUC
well above 0.5, comparable to its own 0.7552 classification AUROC) or
was genuinely lost (projection AUC near 0.5 despite the held
classification AUROC — which would itself be a surprising, separately
interesting finding worth its own follow-up). **Pending**: rerun against
real data with the corrected module.

## Which analyses need existing checkpoints vs. new training

Per-function breakdown of `code/experimental_audit.py` (all 11 synthetic
tests, A-K, pass — see `experimental_audit_sanity_output.txt`):

**Read-only / inference-only — run on EXISTING checkpoints and files,
no training:**

| Function | Needs |
|---|---|
| `audit_baseline_inputs()` | nothing — pure code introspection, already run (§1 above) |
| `verify_split_identity()` | existing `val_index.csv` / `val_subject_ids.csv` files |
| `verify_trm_variant()` / `verify_all_trm_variants()` | existing `{tag}_config.json` + `{tag}_best_model.pt` + `{tag}_results.csv` |
| `check_log_header_result_alignment()` | the raw console log text (already resolved the real A3/A4 case by hand in §2; this makes it a reusable, automatic check for next time) |
| `roc_pr_comparison()`, `probability_distributions()`, `per_class_metrics()` | existing `val_predictions_{m0,m1}.csv` |
| `calibration_metrics()` + `plot_reliability_diagram()` | existing `val_predictions_{m0,m1}.csv` |
| `extract_val_embeddings()`, `within_subject_stability()`, `between_subject_similarity()`, `representation_collapse_diagnostics()`, `representation_quality_report()` | existing `best_m0_model.pt`/`best_m1_model.pt` + the saved train/val subject-ID split — one frozen forward pass per subject, no gradient, no architecture change (this is the "implement that functionality so it can run on existing checkpoints without retraining" ask, directly) |
| `verify_phenotype_blindness()` | a freshly-constructed (untrained) instance of any `MODEL_SPECS` entry — proves MRI-only status behaviorally, needs no checkpoint at all |

**Requires NEW training — not run by this module on its own:**

| Function | What it would train |
|---|---|
| `prepare_mri_only_baseline_run()` → `run_controlled_comparison(model_names=['baseline_mri_only'])` | the missing fair MRI-only baseline (A2) — 1 run on the real 896 subjects |
| `run_controlled_comparison()` (full) | the 119-fold repeated-CV + LOSO matrix across all 7 models — ~3 GPU-hours estimated (§4 above) |

Everything in the first table can produce real numbers for this audit
**today**, from files that already exist on the Kaggle output, with zero
additional compute. Everything in the second table is implemented and
synthetic-tested but is new, real, GPU training and was deliberately
**not** run here, per the instruction not to run expensive experiments
yet.

## Outstanding items to close this audit

The actual Kaggle notebook has been reviewed (its 8508fc5/967af49-era
version here now has 8 new cells appended covering items 1, 3, and 5
below as runnable-but-not-yet-run cells — see "Notebook" below). Status:

1. **DONE** — A3/A4 labeling discrepancy: resolved (§2 above), console-log
   artifact, not a data bug.
2. **DONE** — Split identity across stages: confirmed via matching
   fingerprint hashes (TRM vs Consistency) and matching counts (vs
   Baseline) (§1 above).
3. **DONE** — Leave-one-site-out eligible sites: confirmed from the real
   `site_label_counts.csv` — 7 sites qualify, not ~4 (§4 above).
4. **PENDING** — Run the phenotype-leakage-check cell (now added to the
   notebook) against the real `processed_index.csv` and record the
   result here.
5. **PENDING** — Run `_adhd200_combined_ablation_study.py`'s A2
   (`ADHDNetMRIOnly`) on the real data — the missing fair MRI-only
   baseline (cell added to the notebook with run instructions, not
   auto-run).
6. **PENDING** — Run the M0/M1 embedding-extraction cell (now added to
   the notebook) against the real `best_m0_model.pt`/`best_m1_model.pt`.
7. **PENDING** — Decide Tier-2 scope (minimal: 1 new A5-new run vs. full:
   retrain all four Step-1 variants under `augment=False`) and approve
   the revised 119-run CV/LOSO compute budget before execution.

## Notebook

The actual Kaggle notebook this run came from has 8 new cells appended
(all 26 original cells and their outputs left untouched): an "Audit
findings from this run" markdown cell with the resolved findings above,
the phenotype-leakage-check script + its runner, the M0/M1
embedding-extraction script + its runner, a note on how to run the A2
MRI-only baseline, and the Item 4 design/compute-budget writeup. Delivered
directly to the user as a file (not committed here, since it contains
this run's actual Kaggle paths and is a working notebook rather than
package source) — ask the user for it if you need to see the exact cells.
