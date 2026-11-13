# ADHDNet MRI Pipeline & Architecture Audit

Response to the 8-phase developer task (preprocessing audit, fair
MRI-only baseline, architecture investigation, M0/M1 consistency
investigation, augmentation audit, repeated-CV/LOSO validation,
implementation reliability, deliverables). Per the task's own Phase 5/8
instructions, **no expensive real-data experiment has been run**: this
delivers Deliverables 1, 3, and 4 as code-level audit + synthetic-tested
tooling; **Deliverable 2** (the minimal code changes the audit supports)
is implemented alongside it as three modules; **Deliverable 5** (final
recommendation) is explicitly **not** given — it requires the real
repeated-evaluation results this report's tooling is built to produce,
not a single favourable synthetic run.

**Constraint check**: no architecture was redesigned, no large
Transformer/ConvNeXt-hybrid was introduced, no learned token was
hardcoded to an atlas ROI, no ADHD Index/phenotype feature enters any
MRI-only model (behaviorally proven, not just asserted — see §7), and
every existing script/checkpoint/result is untouched — all new logic
lives in three new files.

## What was actually run vs. what needs real data

This sandbox has **no GPU and no access to the real ADHD-200 NIfTI
files or `processed_index.csv`** (896 real subjects). Everything below
marked "Confirmed" is confirmed either by reading the actual pipeline
code (`_adhd200_combined_mri_preprocessing.py` / `_adhd200_combined_
prepare_adhd200.py` / `proposed_model.py` / `phase2_model.py`, as
extracted from the real notebook) or by a synthetic test with a known
ground truth. Everything marked "Potential — requires real-data
verification" needs the new modules below run against the real dataset
on Kaggle; this report does not claim those results.

## Deliverable 1 — Audit findings, classified

### Phase 1.1 — preprocessing pipeline (code-level, confirmed by reading `_adhd200_combined_mri_preprocessing.py` / `_adhd200_combined_prepare_adhd200.py`)

| # | Finding | Classification |
|---|---|---|
| 1 | `nib.as_closest_canonical()` only reorders/flips axes to RAS+ orientation. It is **not** spatial registration: the pipeline never reads an affine for resampling and never aligns to a named template (no "MNI"/"ICBM" string anywhere in the code). | **Confirmed** (code never performs registration) |
| 2 | `resize_volume()` computes its `scipy.ndimage.zoom` factors from voxel-**count** ratios only (`target_size / size`), never from physical voxel spacing (mm). If two scans differ in native shape/spacing, resizing both to 64³ this way does not give them the same physical scale. | **Confirmed** (code reads shape only, never `get_zooms()`/affine) |
| 3 | Whether the *source* NIfTI files are already on a common voxel grid (which would make finding #2 harmless) — the preferred filename pattern `normalized_resampled_128_*` implies upstream resampling by the dataset's curator, but nothing in this pipeline verifies it, and the fallback tiers (`SCAN_PATTERNS`, items 2-5) could select native-resolution files for some subjects if the preferred tier is missing. | **Potential — requires real-data verification.** `preprocessing_audit.py`'s `audit_affine_consistency()` + `preprocessing_variant_distribution()` (new, below) answer this directly from the real `source_nifti` paths already stored in `processed_index.csv`; not yet run against real data. Also could not reach kaggle.com from this sandbox to check the dataset's own documentation (network policy) — worth checking directly. |
| 4 | `qc_anatomical` (from `qcanatomical1`/`qcs1anat`) **is** captured into every subject's row, but is never read again anywhere in `build_manifest()`/`prepare_volumes()` to exclude anything. | **Confirmed** (recorded, not enforced — by direct code reading) |
| 5 | `dataset_preparation.json`'s `selected_variant` field is a single hardcoded string (`'normalized_resampled_128 (preferred)'`), not a per-subject fact — it does not reflect which fallback tier each subject's file actually matched. | **Confirmed** (the summary-writing code never inspects which pattern matched) |
| 6 | Per-volume min-max normalization is sensitive to outlier voxels (a single scanner artifact) and maps each scan's own raw intensity range to [0,1] independently — if raw intensity ranges differ systematically by site/scanner, this could introduce subtle inter-site normalization differences. | **Potential — requires real-data verification** (would need per-site raw-intensity-range statistics from real scans) |
| 7 | Duplicate-subject-ID-across-sites handling drops **both** copies (`duplicate_subject_also_in_X`); diagnosis-conflict handling drops the subject entirely rather than guessing. | **No issue found** — correct leakage/ambiguity handling |

### Phase 1.2 — audit reports

Implemented in `preprocessing_audit.py` (new): `site_class_report()`,
`qc_statistics()`, `missing_mri_report()`, `preprocessing_variant_
distribution()`, `audit_affine_consistency()`, `save_representative_
slices()`. All synthetic-tested (Tests A–D); **not yet run against the
real dataset** — doing so is a single script run on Kaggle (read-only,
no training), not a new experiment.

### Phase 1.3 — resolution (64³ vs 96³)

| Finding | Classification |
|---|---|
| The architecture itself is resolution-agnostic by construction: `LightweightEncoder`'s conv/pool stack and `SpatialTokenizer`'s attention-pooling both operate on whatever spatial size reaches them (`SpatialTokenizer` flattens `D*H*W` generically via `.view(B, n_tokens, -1)`/einsum — no hardcoded 8×8×8 assumption anywhere). | **No issue found** — confirmed by reading `proposed_model.py` directly; a 96³ experiment is a data/config change (new `.npy` volumes + smaller batch + gradient accumulation), not an architecture change. |
| 96³ costs `(96/64)³ ≈ 3.375×` the voxels/activation-memory/FLOPs of 64³ per sample. | **Confirmed** (arithmetic, via `resolution_cost_estimate()`) |

`train_one_epoch_with_grad_accumulation()` (new, in `preprocessing_
audit.py`) implements the accumulation needed to run 96³ at a smaller
batch size with a comparable effective batch size; synthetic-tested
(Test E) — mathematically equivalent gradient, no model changes. The
96³ `.npy` volumes themselves must be regenerated from the real NIfTI
files (needs real data — not run here).

### Phase 3.1 — encoder/token diagnostics

Implemented in `architecture_audit.py` (new): `feature_map_statistics()`,
`token_diagnostics()` (activation stats, 16×16 token-similarity matrix,
per-slot variance across the batch, effective rank, spatial-attention
entropy), `cross_class_token_diversity()`. Synthetic-tested (Test A) —
runs correctly and returns sane shapes/ranges on an untrained model and
random input; **whether the trained real model's tokens have actually
collapsed is a Potential finding requiring a real checkpoint + real
validation volumes**, not yet run.

### Phase 3.2 — TRM relationship-matrix uniformity

`relationship_matrix_uniformity()` (new) — entropy of each row of the
MHA or pairwise relationship matrix, normalized against a perfectly
uniform distribution (1.0 = learned nothing beyond "every token equally
related to every other token"). **Verified against known ground truth**
in Test B: a constructed perfectly-uniform matrix scores 1.0000, a
constructed near-identity (maximally peaked) matrix scores 0.0519 — the
function correctly distinguishes the two extremes it's meant to
measure. The real A4/A4-new/A5-new checkpoints' actual relationship
matrices have not been measured (needs real checkpoints).

### Phase 3.3 — Anatomical Attention scale

| Finding | Classification |
|---|---|
| `AnatomicalAttention.forward()` computes `weights = softmax(logits, dim=-1)` over the 16 tokens (mean weight = 1/16 under near-uniform attention) then does `weighted = tokens * weights.unsqueeze(-1)` — an elementwise multiply, not a weighted sum/pool. Under near-uniform attention, every one of the 16 output tokens individually shrinks to ≈1/16 of its input magnitude. | **Confirmed**, and **measured**: Test C constructs tokens and forces the score network's weights to zero (producing exactly uniform softmax regardless of token content), then runs both the real `AnatomicalAttention` and the new `ScalePreservingAnatomicalAttention` — existing attention's output-to-input norm ratio = **0.0625** (= 1/16, exactly as predicted); the scale-preserving alternative's ratio = **1.0000**. |

`ScalePreservingAnatomicalAttention` (new, separate class — the existing
`AnatomicalAttention` is untouched and remains the default) rescales the
softmax weights by `n_tokens` before the multiply, so near-uniform
attention leaves magnitude unchanged while a token that genuinely
stands out still gets proportionally more weight. `attention_magnitude_
report()` reports weight stats and norm-before/after for either module,
for the "no attention / existing / scale-preserving" 3-way comparison
the task asks for. Whether this measurably changes real validation
AUROC is **not yet tested on real data**.

### Phase 4 — M0/M1 consistency investigation

| Finding | Classification |
|---|---|
| M0/M1 use identical subject splits, identical architecture (`ADHDProposedModelV2(trm_type='pairwise', use_attention=True)`), differing only in the consistency loss term — confirmed by direct code reading (`phase2_consistency.py`'s `main()` builds one `train_ds`/`val_ds` pair, shared by both `train_model()` calls) and by the identical dataset fingerprint hash both stages printed in the real run's own log. | **No issue found** |
| The two views are generated from the same subject: View 1 is the deterministic, unaugmented volume; View 2 is `consistency_augment()` applied to that SAME tensor (`vol.clone()`), never a different subject. | **No issue found** (confirmed by code reading) |
| `consistency_augment()`'s transformations (intensity gain/bias, Gaussian noise, ≤2-voxel integer roll) are explicitly documented as excluding rotation/flip/warp — "anatomically defensible" by the task's own standard. | **No issue found** |
| Whether the consistency loss (λ=0.05, no warmup) dominates the classification loss, and whether a smaller λ or a warmup schedule changes the sensitivity/F1 trade-off, requires the real λ-sweep. | **Potential — requires real-data verification** |

`architecture_audit.py`'s `run_lambda_sweep_one_epoch()` (new) runs M0
(λ=0) through M1 (λ=0.005/0.01/0.025/0.05) from the same seed, one
epoch at a time, so a caller can combine it with a warmup schedule
(`linear_warmup_lambda()`, new, verified against known ground truth in
Test D) across epochs. Synthetic-tested end-to-end (Test E) on 3 λ
values; **the real 5-point sweep with real training budgets has not
been run.**

### Phase 5 — MRI augmentation audit

| Finding | Classification |
|---|---|
| `dataset.py`'s existing augmentation (`augment=True`) is 3-axis random flips — this is *not* automatically anatomically safe for structural brain MRI (a left-right flip in particular can be defensible or not depending on whether hemispheric asymmetry matters to the task; a top-bottom or front-back flip of a brain volume has no anatomical counterpart at all). | **Potential — worth treating as a real concern**, not dismissed: left-right flips are common practice in brain-MRI augmentation specifically because approximate left-right symmetry is a real anatomical property; axis-0 (superior-inferior) and axis-2 flips, if `dataset.py`'s flip is applied to *all three* axes independently as its code suggests, do not have the same justification. This audit does not change `dataset.py`'s default (preserving existing behavior per the task's constraint) but flags it for the user's judgment. |
| `consistency_augment()` (intensity/noise/small-shift, no rotation/flip) is already exactly the "conservative MRI-safe augmentation" the task asks to compare against — not a new transform. | **No issue found** — reused, not reimplemented |

`ConservativeAugmentDataset` (new, in `architecture_audit.py`) wraps an
existing `ADHDDataset(augment=False)` instance and applies `consistency_
augment()` per sample, without touching `dataset.py` or its `augment=
True` behavior at all. `augmentation_ablation_specs()` returns the
three ready-to-train arms (`none`/`existing_flip`/`conservative`).
**Verified with a position-based test (Test F)**, not just "it runs": a
deliberately placed, off-center bright blob stays on the same side of
the volume's center after conservative augmentation (ruling out an
accidental flip) and moves at most a few voxels (consistent with the
documented ≤2-voxel shift). Real training comparison across the three
arms has not been run.

### Phase 6 — repeated CV / LOSO protocol

`experimental_audit.py`'s existing harness (`train_one_fold()`,
`build_repeated_cv_folds()`, `build_loso_folds()`, `run_controlled_
comparison()`) already covered subject-level fold construction,
fingerprint-guarded resumability, and single-class-site exclusion from
LOSO. Extended for this task with:

- **Balanced accuracy, sensitivity, specificity, F1, AUPRC, confusion
  matrix** added to every fold's result row (previously only AUROC +
  epoch + params).
- **`aggregate_fold_metrics()`** (new): mean/std/95% CI per model,
  computed **separately** for `fold_type='cv'` vs `'loso'` rows — never
  pooled (verified in Test O, including that it refuses to run at all
  without a `fold_type` column).
- **`fold_distribution_report()`** (new): class *and* site counts for
  train and val on every single fold, plus a direct re-confirmation
  that no subject appears in both (verified in Test N; `run_controlled_
  comparison()` now asserts this for every fold before training
  anything).
- **`build_repeated_cv_folds(..., stratify_by_site=True)`** (new
  option): stratifies jointly by (label, site) where every combined bin
  has enough subjects, falling back to label-only stratification for
  any site too small for that (and reporting exactly which sites fell
  back) rather than silently using one scheme for some subjects and
  another for others. Verified in Test P against a deliberately
  single-class site.
- **Threshold-selection policy made explicit in code**, not just in
  prose: a comment at the exact line computing `y_pred = (probs >=
  0.5)` states that 0.5 is a fixed constant applied identically to
  every fold, never fit/searched/adjusted from a held-out fold's own
  labels — the only thing chosen per-fold is which *epoch* is "best" by
  AUROC (threshold-independent), not the cutoff itself.

Real CV/LOSO numbers have not been generated — only the harness itself,
synthetic-tested.

### Phase 7 — implementation & checkpoint reliability

Checked against the task's explicit 9-point list:

| # | Item | Status |
|---|---|---|
| 1 | Model/input tensors on the same device | **Bug found and fixed** (see `AUDIT_REPORT.md` §6 — a prior module's Test G built a model without `.to(DEVICE)`; fixed there and defensively in two embedding-extraction functions) |
| 2 | CPU and CUDA inference paths | Code is device-generic (`DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` everywhere, no hardcoded device string found by inspection); verified on CPU here, **CUDA path needs the user's own GPU run to confirm** (this sandbox has none) |
| 3 | Latest and best-model checkpoints saved separately | **Bug found and fixed this round**: `experimental_audit.py`'s CV/LOSO `train_one_fold()` only ever wrote the per-epoch `latest` checkpoint — no standalone best-weights file, unlike `phase2_trm_ablation.py`/`phase2_consistency.py`'s existing convention. Now also writes `{fold_tag}_best_model.pt` whenever a new best is found. |
| 4 | Best predictions recoverable after resuming | **Bug found and fixed this round** (the significant one): `best_probs`/`best_labels` were only ever set in-memory inside the `if auroc > best['auroc']` branch; on a resumed run whose remaining epochs never re-beat the already-loaded best epoch, they stayed `None` all the way to the final OOF write, which would crash. Fixed by saving a `{fold_tag}_best_preds.csv` sidecar whenever a new best is found and reloading it on resume. **Proven with a real reproduction, not just reasoning**: Test Q forces epoch 1 to AUROC=0.99 then interrupts, forces every subsequent epoch to AUROC=0.10, resumes, and confirms the final OOF predictions exactly match epoch 1's (not a crash, not a silently-wrong later epoch). |
| 5 | Python/NumPy/PyTorch CPU+CUDA RNG + DataLoader generator state saved/restored | Already correct in `phase2_trm_ablation.py`/`phase2_consistency.py` (unchanged, verified in earlier rounds); `experimental_audit.py`'s CV/LOSO harness saves/restores model/optimizer/scheduler state across resume but **does not currently save the DataLoader generator's own RNG state** — flagged as a **Potential issue**: a resumed fold's training-batch order will not bit-for-bit match an uninterrupted run's (unlike the Phase-2 scripts, which do save `train_generator_state`). Not fixed this round (scope); recommend addressing before relying on CV/LOSO for bit-exact reproducibility claims. |
| 6 | Interrupted vs. uninterrupted synthetic runs reproducible | Phase-2 scripts: yes (verified in earlier rounds). `experimental_audit.py`'s CV/LOSO harness: not bit-exact per item 5 above, but now at least *correct* (produces the right best-epoch predictions, per the Test Q fix) rather than exactly reproducible epoch-by-epoch. |
| 7 | OOF predictions from best checkpoint, not final epoch | **Was the Test Q bug** — now fixed and proven (see item 4). |
| 8 | (duplicate of 7 in the task's numbering) | see above |
| 9 | Variant labeling consistent across logs/CSVs/plots | `verify_trm_variant()`/`check_log_header_result_alignment()` (existing, from the previous round) already provide this check generally, and were what resolved the real A3/A4 log discrepancy. `fold_tag` naming is now used consistently across every new checkpoint/config/results/OOF/best-model/best-preds file in the CV/LOSO harness. |

## Deliverable 3 — controlled experiment plan

| Exp. ID | Model config | Modification tested | Baseline | Expected cost | Metrics |
|---|---|---|---|---|---|
| P1.3-a | `ADHDProposedModelV2` (best Step-1 config), 96³ input | Input resolution | Same config, 64³ | ~3.4× memory/FLOPs per sample; grad-accum to match effective batch | AUROC/AUPRC + wall-clock vs. baseline |
| P2-a | `ADHDNetMRIOnly` | Fair MRI-only baseline | — (this IS the reference) | 1 run, ~100 epochs (matches existing `ADHDNet` budget) | Full metric set below |
| P3.3-a | `ADHDProposedModelV2` + `ScalePreservingAnatomicalAttention` | Scale-preserving attention | A5-new (existing `AnatomicalAttention`) | Same as A5-new | AUROC/AUPRC + token-magnitude report |
| P3.3-b | `ADHDProposedModelV2`, `use_attention=False` | No attention at all | A5-new | Same as A4-new (already run) | AUROC/AUPRC |
| P4-a..e | M0/M1 at λ ∈ {0, 0.005, 0.01, 0.025, 0.05} | Consistency weight | M0 (λ=0) | 5× one M0/M1-scale run | Full metric set + embedding collapse diagnostics (existing `representation_collapse_diagnostics()`) |
| P4-f | M1, λ=0.05, linear warmup over first N epochs | Warmup vs. no warmup | P4-e (no warmup) | Same as P4-e | Full metric set |
| P5-a..c | Best Step-1 config | `none` / `existing_flip` / `conservative` augmentation | `existing_flip` (current default) | 3× one Step-1-variant-scale run | AUROC/AUPRC |
| P6-full | All 7 models (`baseline_mri_only`, A3, A4, A4-new, A5-new, M0, M1) | Repeated CV (5×2) + LOSO (7 eligible sites) | — (this IS the headline comparison) | 119 training runs, **~3 GPU-hours estimated** (see `AUDIT_REPORT.md` §4/§7) | Full metric set, mean/std/95% CI, per fold_type, confusion matrix, params, inference time |

Full metric set (Phase 6): AUROC, AUPRC, accuracy, balanced accuracy,
sensitivity, specificity, F1, confusion matrix, mean±std across folds,
95% CI, params, and inference time where feasible — all now computed
automatically by `train_one_fold()`/`aggregate_fold_metrics()`.

## Deliverable 4 — test report

31 synthetic tests total, all passing, across three files (re-verified
from a fresh, isolated extraction of the repo — self-contained):

- `preprocessing_audit.py`: Tests A–E (NIfTI header inspection against
  known ground truth, affine-consistency flagging of deliberately mixed
  spacing, scan-pattern classification against known filenames,
  end-to-end against a synthetic index, resolution-cost + grad-
  accumulation training step).
- `architecture_audit.py`: Tests A–F (feature/token diagnostics,
  relationship-matrix uniformity against known uniform/peaked
  matrices, scale-preserving attention proven to preserve magnitude
  under forced-uniform weights where the existing module shrinks it by
  exactly 1/16, warmup schedule against known ground truth, λ-sweep
  end-to-end, augmentation ablation with a position-based no-flip
  check).
- `experimental_audit.py`: Tests A–Q (17 tests spanning baseline-input
  audit, split-identity verification, TRM-variant verification against
  a real synthetic training run, the A3/A4 log-artifact detector,
  calibration metrics, ROC/PR/per-class metrics, embedding/collapse
  diagnostics proven against two engineered ground-truth scenarios,
  fold-trainer resumability/fingerprinting, every one of the 7 model
  specs training end-to-end, phenotype-blindness proven behaviorally
  against a negative control, fold distribution reporting, CV/LOSO
  metric aggregation kept separate, site-stratified CV fallback, and
  the resume/best-prediction bug fix proven by forced reproduction).

Confirmed: every model completes forward+backward; checkpoint
save/resume works (including the newly-fixed best-prediction-survives-
resume case); evaluation produces valid predictions; no train/val
subject overlap (asserted automatically on every fold); all output
files save correctly. **No real-data smoke test was run** (no GPU/data
in this sandbox) — recommend running one short real-data pass through
`run_controlled_comparison()` with a tiny `epochs`/`n_splits` before the
full matrix, the same way every previous round's "smoke test" step was
done.

## Deliverable 5 — final recommendation

**Not given.** The task's own instruction is explicit: "The final
recommendation must be based on repeated evaluation, not a single
favourable result." Nothing in this round ran a single real-data
epoch. What exists now is the audited, bug-fixed tooling to produce
that repeated evaluation — Deliverable 3's table is the proposed path
to it, pending review.
