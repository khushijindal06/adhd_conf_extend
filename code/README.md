# ADHD MRI Classification — Phase 1 + Phase 2 (Steps 1–2)

Lightweight 3D-MRI ADHD classifier: DS-Conv3D encoder → learned spatial
tokens → FastKAN refinement → Token Relationship Module → Anatomical
Attention → compact brain representation → classifier. This package
contains the verified, run-ready code for **Phase 1** (baseline ADHDNet +
ablations) and **Phase 2, Steps 1–2** (pairwise Token Relationship Module,
representation-consistency learning). Steps 3–5 (anatomical reproducibility,
statistical comparison, parameter-efficiency audit) are a separate,
not-yet-approved stage and are intentionally not included here.

## Setup

```bash
pip install -r requirements.txt
```

All dependencies are pre-installed on Kaggle's standard notebook image —
`pip install` is only needed outside Kaggle.

## Configuration

Every script in this package reads your dataset from the same two places.
Edit these at the top of the script you're running (defaults shown):

```python
DATA_ROOT  = '/kaggle/working'
INDEX_FILE = f'{DATA_ROOT}/processed_index.csv'
```

`processed_index.csv` must have (at minimum) the columns: `subject_id`,
`label`, `site`, `npy_path` (path to each subject's preprocessed MRI
volume as a `.npy` file), plus the phenotype columns `age`, `sex`,
`med_status`, `site_id`, `adhd_index` used by the (secondary) phenotype
branch.

## Running Phase 2, Step 2 (representation-consistency learning)

```bash
python phase2_consistency.py
```

This single command:
1. Runs code-level sanity checks on synthetic data (Tests A–H + Sections
   14/14b) — catches implementation errors before touching your real data.
2. Validates your real dataset (duplicate subject IDs, missing `.npy`
   files, train/val leakage, validation-set class balance, per-site label
   counts).
3. Builds the leakage-free, subject-level 80/20 train/val split (loaded
   from disk on a resumed run, to guarantee the identical split).
4. Trains **M0** (classification loss only) and **M1** (classification +
   0.05 × representation-consistency loss) — same split, same seed, same
   initial weights, same optimizer/scheduler/evaluation for both; the
   *only* difference is the consistency term.
5. Saves every result, checkpoint, prediction, and figure listed below.

**Interruption recovery**: `RESUME = True` (default) at the top of the
file — M0 and M1 each resume INDEPENDENTLY. If the process is killed, just
re-run the same command: a variant whose `{tag}_results.csv` already
exists is skipped; a variant with a `{tag}_latest_checkpoint.pt` but no
results file resumes from that exact epoch (model, optimizer, scheduler,
RNG state, and training history all restored). Verified with a real
mid-training kill in testing — see the chat report.

Toggle `RUN_M0` / `RUN_M1` / `CONSISTENCY_LAMBDA` at the top of the file to
run only one variant or change the consistency weight — no other edits
needed.

**Important**: Step 1 and Step 2 use *different* augmentation policies by
design (Step 1: `augment=True` flips; Step 2: `augment=False`, isolated
two-view consistency augmentation only). Never cross-compare a Step-1
variant's number against a Step-2 M0/M1 number — see the
AUGMENTATION-COMPARISON NOTE in `phase2_consistency.py`'s module docstring.
All reported metrics are VALIDATION-split results, not an independent test
set (none exists in this pipeline).

## Outputs (under `{DATA_ROOT}/phase2/consistency/`)

| File | Contents |
|---|---|
| `train_subject_ids.csv`, `val_subject_ids.csv` | exact subject IDs used in each split |
| `site_label_counts.csv` | per-site, per-split label counts |
| `m0_results.csv`, `m1_results.csv` | per-variant aggregate metrics |
| `consistency_ablation_results.csv` | M0 vs M1 side-by-side table |
| `representation_stability.csv` | cosine-stability summary, M0 vs M1 |
| `val_predictions_m0.csv`, `val_predictions_m1.csv` | **per-subject** validation `y_true`/`y_prob`/`y_pred` |
| `training_history_m0.csv`, `training_history_m1.csv` | per-epoch loss/metric history |
| `best_m0_model.pt`, `best_m1_model.pt` | best-checkpoint state dicts |
| `m0_latest_checkpoint.pt`, `m1_latest_checkpoint.pt` | full resumable training state (model/optimizer/scheduler/RNG) |
| `config_m0.json`, `config_m1.json` | full run config (seeds, hyperparams, arch) |
| `confusion_matrix_m0.png`, `confusion_matrix_m1.png` | validation confusion matrices |
| `training_curves.png`, `comparison_figure.png` | loss/AUROC curves; stability-vs-performance comparison |

## Running Phase 2, Step 1 (pairwise TRM ablation)

```bash
python phase2_trm_ablation.py
```

Compares A3 (no TRM) / A4 (original MHA-based TRM) / A4-new (pairwise TRM)
/ A5-new (pairwise TRM + Anatomical Attention) on AUROC/AUPRC/accuracy/F1/
sensitivity/specificity/params/FLOPs/inference latency. Writes to its own
separate Phase-2 output directory — never overwrites Phase-1 results. Same
`RESUME=True` interruption-recovery design as Step 2, with each of the four
variants resuming independently.

## Phase 1

`train.py`, `evaluate.py`, `predict.py`, `cross_validate.py`,
`ablation_study.py`, `cross_site_eval.py`, `benchmark.py` — the original
baseline pipeline (`model.py` / `dataset.py`), its ablations, and
cross-validation/cross-site evaluation. `generate_architecture_figure.py`
/ `generate_paper_figures.py` produce the paper figures from saved results.

## File map

| File | Role |
|---|---|
| `model.py` | Phase-1 `ADHDNet` baseline + ablation variants |
| `dataset.py` | `ADHDDataset` — leakage-free loading/normalization/imputation |
| `train.py`, `evaluate.py`, `predict.py` | Phase-1 train/eval/inference |
| `cross_validate.py`, `cross_site_eval.py` | Phase-1 CV and leave-one-site-out |
| `ablation_study.py` | Phase-1 ablation table |
| `benchmark.py` | Phase-1 parameter/FLOPs/latency audit |
| `generate_architecture_figure.py`, `generate_paper_figures.py` | Paper figures |
| `proposed_model.py` | Token-based architecture (encoder/tokenizer/FastKAN/MHA-TRM/AnatomicalAttention) |
| `proposed_ablation_study.py`, `proposed_explainability.py` | Token-model ablations + saliency/attention extraction |
| `phase2_model.py` | **Step 1**: pairwise Token Relationship Module + `ADHDProposedModelV2` |
| `phase2_trm_ablation.py` | **Step 1**: A3/A4/A4-new/A5-new TRM comparison |
| `phase2_consistency.py` | **Step 2**: representation-consistency learning (M0 vs M1) |

## Status

Ready for the real-data Step-2 (M0/M1) run. See the chat report accompanying
this package for the full verification results.
