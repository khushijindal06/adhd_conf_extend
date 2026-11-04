"""
PHASE 2, Step 2 ONLY: representation-consistency learning -- WITH experiment
tracking, per-epoch checkpointing, and interruption recovery (same design
as phase2_trm_ablation.py).

Does NOT modify phase2_model.py (Step 1) or anything in Phase 1 (cells
0-34). The architecture under test is exactly Step 1's A5-new config
(pairwise TRM + Anatomical Attention), imported unmodified from
phase2_model.py. The representation (`encode_representation` below) is
recomputed by calling the model's existing public submodules -- the same
technique proposed_explainability.py already uses for its saliency map --
so there is no new method on ADHDProposedModelV2 and no risk to Step 1's
already-verified numbers.

Hypothesis under test: mild acquisition/preprocessing variation should not
substantially change the learned subject-level representation. This is
tested directly by comparing cos(z_original, z_augmented) on held-out
validation subjects, M0 vs M1 -- NOT by only checking that the training
loss went down.

    M0 : A5-new, classification loss only (identical to Step 1's A5-new)
    M1 : A5-new, classification loss + lambda * consistency loss (lambda=0.05)

Everything else (data, split, seed, optimizer, schedule, epochs,
architecture) is identical between M0 and M1 -- the only difference is
whether the consistency term is added to the loss.

View-1/View-2 isolation (Step 2 ONLY, does not touch dataset.py): `main()`
builds its training dataset with `augment=False` -- i.e. View 1 is always
the SAME deterministic, unaugmented preprocessed MRI that validation uses,
never dataset.py's own `augment=True` random-flip path (that path is
untouched and still used exactly as before by Phase-1's own scripts, and
by Step 1's phase2_trm_ablation.py). View 2 is produced EXCLUSIVELY by
`consistency_augment()` inside `train_model()`'s training loop (intensity
scaling/bias, Gaussian noise, small spatial shift -- never a flip).

*** AUGMENTATION-COMPARISON NOTE (do not skip this) ***
Step 1 (phase2_trm_ablation.py) trains A3/A4/A4-new/A5-new with
`augment=True` (dataset.py's own random-flip augmentation on a SINGLE
view) -- the same augmentation policy for all four Step-1 variants, so
Step 1's own internal A3-vs-A4-vs-A4-new-vs-A5-new comparison is valid.
Step 2 (this file) trains M0/M1 with `augment=False` (deterministic View 1,
per the isolation above) -- the same augmentation policy for both M0 and
M1, so Step 2's own internal M0-vs-M1 comparison is ALSO valid.
What is NOT valid: comparing Step 1's A5-new number against Step 2's M0 or
M1 number. Even though M0 is "A5-new, classification loss only" in
architecture, M0 is trained under augment=False while Step 1's A5-new is
trained under augment=True -- a real confound. M0 already IS the correct,
augmentation-matched "classification-loss-only" control for the M0-vs-M1
comparison (that is its entire purpose); no additional variant is needed.
Step 1's A5-new is simply a different experiment (different augmentation
policy) and must never be substituted for M0 in any comparison.

This file connects to the REAL ADHD-200 pipeline already used everywhere
else in this project (dataset.ADHDDataset, the same `processed_index.csv`,
the same subject-level StratifiedShuffleSplit, the same leakage-free
fit_stats convention) -- it does not invent a new dataset format and does
not use synthetic data as its main execution path. The ONLY things you
should need to change for your own Kaggle/local layout are DATA_ROOT and
INDEX_FILE below.

*** VALIDATION, NOT AN INDEPENDENT TEST SET ***
Every metric this script reports (AUROC/AUPRC/accuracy/F1/sensitivity/
specificity/confusion matrix/representation stability) is computed on the
single held-out VALIDATION split from this run's train/val
StratifiedShuffleSplit -- there is no independent, never-touched test set
anywhere in this pipeline. Report these numbers as validation results.
Multi-fold cross-validation and leave-one-site-out cross-site evaluation
(which would give a more robust estimate) are a separate, not-yet-approved
future stage -- not implemented here.

Outputs go to a SEPARATE directory, '{DATA_ROOT}/phase2/consistency/' --
Phase-1 results, Step-1's trm_ablation_results.csv, and all existing
notebooks/CSVs/figures are untouched.

Interruption recovery (same design as phase2_trm_ablation.py): set
RESUME = True (default) and re-run this script. M0 and M1 each resume
INDEPENDENTLY -- a variant whose `{tag}_results.csv` already exists is
treated as COMPLETE and skipped; a variant with a `{tag}_latest_
checkpoint.pt` but no `{tag}_results.csv` is resumed from that checkpoint
-- model, optimizer, scheduler, early-stopping state, training history,
AND the exact RNG/DataLoader-shuffle-generator state are all restored, so
the per-epoch training-loss sequence after resuming continues the sequence
an uninterrupted run would have produced (verified in run_sanity_check()'s
Test H below). Each variant's results are saved immediately when that
variant finishes, not held until both M0 and M1 complete. Checkpoints and
CSVs are written atomically (temp file + os.replace()).

Strict configuration validation: every checkpoint/result carries a
`config_fingerprint` (sha256 of the dataset file + the exact subject
split, plus arch_kwargs/use_consistency/lam/epochs/patience/batch_size/
lr/val_split/seed). Before skipping a completed variant OR resuming an
interrupted one, the CURRENT run's fingerprint is compared against the
saved one; any mismatch RAISES (never silently reuses a result produced
under different settings) -- see `check_fingerprint_or_raise()` and
Test I. A checkpoint/result with NO `config_fingerprint` at all (it
predates this safeguard, or was produced by a different script version)
is refused the same way, never silently trusted -- see
`require_fingerprint_or_raise()`. Either case requires pointing `OUT_DIR`
at a fresh, empty directory. The first valid validation checkpoint is
always saved, even if its AUROC is exactly 0.0 (the "best so far" sentinel
is -1.0, not 0.0 -- see Test J).

`run_sanity_check()` runs CODE-LEVEL tests only (synthetic data, no real
dataset, no training run, no AUROC/AUPRC/F1 reported) and is called once at
the start of `main()` purely to catch implementation errors before any
real training happens; it is NOT the main execution path:
    Test A -- forward pass (MRI -> representation -> classifier)
    Test B -- two-view representation (same subject, same weights)
    Test C -- consistency loss returns a finite scalar
    Test D -- gradient flow (backward pass, finite, no NaN/Inf)
    Test E -- identical representations give ~zero consistency loss
    Test F -- augmentation isolation (no randomness outside the two-view step)
    Test G -- encode_representation()+classify_from_repr() == model.forward()
              exactly (proves no parallel/divergent path was introduced)
    Test H -- genuine interrupt-then-resume reproduces the uninterrupted
              per-epoch training-loss sequence (same test design as
              phase2_trm_ablation.py's Test C)
    Test I -- strict configuration validation raises on a changed dataset
              (index_hash), changed split (split_hash), or changed
              consistency weight (lambda), both when skipping a completed
              variant and when resuming an interrupted one; an UNCHANGED
              configuration still skips cleanly; a completed result or
              checkpoint with NO config_fingerprint at all is also
              refused, both when skipping and when resuming
    Test J -- the first-epoch checkpoint is saved even when that epoch's
              AUROC is exactly 0.0 (forced via a temporary monkeypatch of
              evaluate(), restored immediately afterward)
    Section 14  -- parameter count with vs without consistency (expected equal)
    Section 14b -- M0/M1 use equivalent initial weights (same seed -> same
                   state_dict, verified tensor-by-tensor, not just a param count)

Before touching the real dataset, `main()` also runs
`validate_dataset_integrity()`: duplicate subject_id check, missing .npy
file check, subject-level train/val leakage check, a check that the
validation split contains both classes, AND now a per-site label-count
breakdown (saved to `site_label_counts.csv`) so site distribution is
visible, not just assumed.

After the sanity + dataset checks, `main()` builds the leakage-free
subject-level train/val split (loaded from disk on a RESUME if already
saved, to guarantee the identical split across restarts), then trains
whichever of M0/M1 are enabled (RUN_M0 / RUN_M1 below) end to end:
dataloaders -> model -> optimizer -> scheduler -> training loop ->
validation -> early stopping -> best-checkpoint selection -> metrics ->
per-subject validation predictions -> representation-stability measurement
(same fixed STABILITY_SEED for M0 and M1, so both see IDENTICAL evaluation
perturbations -- see measure_representation_stability()) -> FLOPs/parameter/
latency audit -> every artifact saved to disk. Nothing here is a placeholder.
"""

import os
import hashlib
import json
import random
import tempfile
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ADHDDataset
from phase2_model import ADHDProposedModelV2

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIG -- the only things you should need to edit for your own setup  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Change DATA_ROOT (and/or INDEX_FILE directly) if your Kaggle/local copy
# of the ADHD-200 pipeline's `processed_index.csv` + preprocessed `.npy`
# volumes live somewhere other than '/kaggle/working'. Nothing else in this
# file needs to change to point at your real dataset.
DATA_ROOT   = '/kaggle/working'
INDEX_FILE  = f'{DATA_ROOT}/processed_index.csv'
INDEX_CSV   = INDEX_FILE    # alias kept for consistency with the rest of this project's scripts
OUT_DIR     = f'{DATA_ROOT}/phase2/consistency'    # SEPARATE from Phase-1 and Step-1 outputs
os.makedirs(OUT_DIR, exist_ok=True)

RESUME = True    # True: skip completed variants (M0/M1 independently), resume an interrupted
                 # one from its latest checkpoint. False: ignore any existing checkpoints/
                 # results and retrain from scratch (files overwritten as training proceeds).

# Which experiments to actually run. Both True by default (full M0 vs M1
# comparison); set either to False to run only the other one without
# touching the training loop.
RUN_M0 = True
RUN_M1 = True

EPOCHS      = 150
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
STABILITY_SEED = 12345   # fixed seed for the POST-HOC stability measurement only (see
                          # measure_representation_stability() below) -- independent of
                          # SEED/training so M0 and M1 draw the exact same sequence of
                          # consistency_augment() perturbations at evaluation time, even
                          # though M1's training loop consumes a different number of
                          # random draws than M0's (M1 calls consistency_augment() every
                          # training step; M0 never does) and would otherwise leave the
                          # global RNG in a different state by the time stability is measured.
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# USE_CONSISTENCY=False reproduces M0 (classification loss only, no
# consistency term, no two-view augmentation) from the exact same code path
# as M1 -- this is what makes the M0 vs M1 ablation clean (RUN_M0/RUN_M1
# above control which of the two actually get trained in main(); this flag
# documents/exercises the same switch at the single-run level).
USE_CONSISTENCY          = True
CONSISTENCY_LAMBDA       = 0.05   # Section 7 default; configurable
CONSISTENCY_AUGMENTATION = True   # if False while USE_CONSISTENCY=True, view 2 = view 1 (debug only;
                                   # not a normal operating mode -- see Test E)
RUN_LAMBDA_SWEEP = False      # set True to additionally run M1a(0.01)/M1b(0.05)/M1c(0.10)
LAMBDA_SWEEP = [0.01, 0.05, 0.10]

# Architecture under test: exactly Step 1's A5-new (frozen, not redesigned here)
ARCH_KWARGS = dict(trm_type='pairwise', use_attention=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Atomic writes -- never leave a half-written checkpoint/CSV on disk, even if
# the process is killed mid-write. Same helpers as phase2_trm_ablation.py.
# ============================================================================
def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_to_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def checkpoint_paths(tag, out_dir):
    return dict(
        latest=f'{out_dir}/{tag}_latest_checkpoint.pt',
        best=f'{out_dir}/best_{tag}_model.pt',
        history=f'{out_dir}/training_history_{tag}.csv',
        valpred=f'{out_dir}/val_predictions_{tag}.csv',
        results=f'{out_dir}/{tag}_results.csv',
        config=f'{out_dir}/config_{tag}.json',
        confusion=f'{out_dir}/confusion_matrix_{tag}.png',
    )


def hash_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def hash_split(train_ids, val_ids):
    return hashlib.sha256((','.join(sorted(train_ids)) + '|' + ','.join(sorted(val_ids))).encode()).hexdigest()


def compute_config_fingerprint(use_consistency, lam, index_hash, split_hash, epochs, patience):
    """
    Everything that must be IDENTICAL between the run that produced a saved
    checkpoint/result and the current run before it is safe to resume or
    skip. If any of these differ, the checkpoint/result was produced under
    different conditions and must never be silently reused.
    """
    return {
        'arch_kwargs': ARCH_KWARGS, 'use_consistency': use_consistency, 'lam': lam,
        'consistency_augmentation': CONSISTENCY_AUGMENTATION, 'stability_seed': STABILITY_SEED,
        'index_hash': index_hash, 'split_hash': split_hash,
        'epochs_max': epochs, 'patience': patience, 'batch_size': BATCH_SIZE,
        'lr': LR, 'val_split': VAL_SPLIT, 'seed': SEED,
    }


def check_fingerprint_or_raise(current_fp, saved_fp, tag, action):
    """
    Strict configuration validation (item 1): raises -- never warns and
    continues -- if any essential training setting differs between the
    current run and the one that produced the saved checkpoint/result this
    `action` ('skip' or 'resume') is about to reuse. Dataset identity
    (index_hash: a sha256 of processed_index.csv's bytes), the exact
    subject-level split (split_hash), model/loss configuration
    (arch_kwargs/use_consistency/lam), seed, learning rate, epochs/patience
    budget, batch size, and val_split are all covered.
    """
    mismatches = {k: (saved_fp.get(k), current_fp[k]) for k in current_fp if saved_fp.get(k) != current_fp[k]}
    if mismatches:
        lines = "\n".join(f"    {k}: saved={old!r}  current={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- configuration has changed since the saved "
            f"checkpoint/result was produced:\n{lines}\n"
            f"  Never silently reusing incompatible results. Either revert the changed "
            f"setting(s) above, or point OUT_DIR at a fresh, empty directory for this new "
            f"configuration (e.g. OUT_DIR = f'{{DATA_ROOT}}/phase2/consistency_v2')."
        )


def require_fingerprint_or_raise(saved_fp, tag, action, source_path):
    """
    Strict configuration validation, extended: a checkpoint/result that
    predates the config_fingerprint safeguard (or was produced by a script
    version that never saved one) carries no proof of what configuration
    produced it. Reusing it anyway would be exactly the silent-reuse-of-
    incompatible-results failure mode item 1 exists to prevent -- so a
    MISSING fingerprint refuses the same way a MISMATCHED one does, never
    with a "proceeding as before" fallback.
    """
    if saved_fp is None:
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- {source_path} has no config_fingerprint "
            f"(it predates this safeguard, or was produced by a different script version), "
            f"so there is no way to verify it matches the current dataset/split/model/"
            f"training configuration.\n"
            f"  Never silently reusing a result whose configuration cannot be verified. "
            f"Point OUT_DIR at a fresh, empty directory to regenerate this result under "
            f"the current safeguard (e.g. OUT_DIR = f'{{DATA_ROOT}}/phase2/consistency_v2')."
        )


# ============================================================================
# Section 2A -- realistic, MRI-safe augmentation for the second ("View 2") pass
# ============================================================================
def consistency_augment(vol, intensity_std=0.05, noise_std=0.02, max_shift=2):
    """
    Mild, realistic perturbation used ONLY to build the second view for the
    consistency loss (and for the post-hoc stability measurement). Applied
    on top of whatever the data pipeline already produced for `vol`.

      - intensity gain + bias (small multiplicative + additive shift)
      - Gaussian noise
      - small integer-voxel spatial translation (torch.roll, +/- max_shift
        voxels per axis on a 64^3 volume -- bounded, does not crop/warp)

    No rotation, flip, or elastic warp here: this explicitly rules out
    transformations that could alter or duplicate structural interpretation,
    or change clinically meaningful morphology.
    Differentiable end-to-end except the integer roll amount itself (which
    is sampled, not learned, exactly like dataset.py's existing flip
    augmentation) -- gradients flow normally through the returned tensor.
    Only ever applied to the training view -- validation volumes never
    pass through this function (see Test F and `evaluate()` below).
    """
    B = vol.shape[0]
    device = vol.device

    gain = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * intensity_std
    bias = (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * intensity_std
    vol = vol * gain + bias
    vol = vol + torch.randn_like(vol) * noise_std

    shifts = torch.randint(-max_shift, max_shift + 1, (B, 3))
    shifted = torch.empty_like(vol)
    for i in range(B):
        dz, dy, dx = shifts[i].tolist()
        shifted[i] = torch.roll(vol[i], shifts=(dz, dy, dx), dims=(1, 2, 3))
    return shifted


# ============================================================================
# Section 3 -- representation extraction (same weights, no new encoder)
# ============================================================================
def encode_representation(model, vol, pheno=None):
    """
    Recomputes the compact brain representation using the SAME model
    instance's existing submodules (encoder/tokenizer/refine/trm/anat_attn/
    token_agg[/pheno fusion]) -- i.e. literally the same weights used by
    model.forward(), called here only so the pre-classifier representation
    can be extracted without adding a method to phase2_model.py.
    """
    feat_map = model.encoder(vol)
    tokens = model.tokenizer(feat_map)
    tokens = model.refine(tokens)
    tokens = model.trm(tokens)
    if model.anat_attn is not None:
        weighted, _ = model.anat_attn(tokens)
    else:
        weighted = tokens
    brain_repr = model.token_agg(weighted.flatten(1))

    if model.use_phenotype:
        zphen = model.pheno_mlp(pheno)
        combined = torch.cat([brain_repr, zphen], dim=1)
        g = model.gate(combined)
        brain_repr = g * brain_repr + (1 - g) * model.pheno_proj(zphen)
    return brain_repr


def classify_from_repr(model, brain_repr):
    return model.evidential_head(brain_repr) if model.use_evidential else model.head(brain_repr)


def cosine_consistency_loss(z1, z2):
    """L_consistency = 1 - cosine_similarity(z1, z2), averaged over the batch."""
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    cos_sim = (z1n * z2n).sum(dim=-1)
    return (1.0 - cos_sim).mean(), cos_sim


# ============================================================================
# Efficiency measurement (reused unchanged -- no second large model is built)
# ============================================================================
def measure_flops_and_latency(model, n_timed=20):
    model.eval()
    vol = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
    with torch.no_grad():
        fc = FlopCounterMode(display=False, depth=None)
        with fc:
            _ = model(vol)
        flops = fc.get_total_flops()
        for _ in range(5):
            _ = model(vol)
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            _ = model(vol)
            times.append((time.perf_counter() - t0) * 1000)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return dict(flops=flops, inference_ms_mean=float(np.mean(times)), inference_ms_std=float(np.std(times)),
                trainable_params=trainable, total_params=total)


def evaluate(model, loader):
    """
    Validation evaluation -- deterministic, NEVER calls consistency_augment().
    Documented AUROC-undefined fallback (same convention as phase2_trm_
    ablation.py's evaluate_full()): if the loader's labels contain only one
    class, AUROC/AUPRC fall back to (0.5, mean label) and `auroc_undefined`
    is set so callers can flag it rather than silently treating 0.5 as real.
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            out = model(vol)
            p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(label.numpy())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(int)
    auroc_undefined = False
    try:
        auc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except Exception:
        auc, auprc = 0.5, float(labels.mean())
        auroc_undefined = True
    acc = (preds == labels).mean()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return dict(auroc=auc, auprc=auprc, accuracy=acc, f1=f1, sensitivity=sens, specificity=spec,
                confusion_matrix=cm, probs=probs, labels=labels, preds=preds, auroc_undefined=auroc_undefined)


def measure_representation_stability(model, loader, n_repeats=3, seed=STABILITY_SEED):
    """
    HOW REPRESENTATION STABILITY IS MEASURED: for every held-out validation
    subject, encode the deterministic, unaugmented volume once (z1 =
    encode_representation(model, vol)), then draw `n_repeats` independent
    consistency_augment() perturbations of that SAME volume and encode each
    (z2), recording cos(z1, z2) every time. NOT used for model selection or
    early stopping -- purely a post-hoc diagnostic, computed once after the
    best checkpoint is already loaded.

    `seed` is reset to a FIXED constant (STABILITY_SEED, independent of the
    training SEED) immediately before the perturbation loop starts, so M0
    and M1 draw the EXACT SAME sequence of perturbations here, regardless
    of how much randomness each consumed during its own training loop.
    """
    set_seed(seed)
    model.eval()
    all_cos = []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            z1 = encode_representation(model, vol)
            for _ in range(n_repeats):
                vol2 = consistency_augment(vol.clone())
                z2 = encode_representation(model, vol2)
                _, cos_sim = cosine_consistency_loss(z1, z2)
                all_cos.extend(cos_sim.cpu().numpy().tolist())
    return float(np.mean(all_cos)), float(np.std(all_cos))


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'ADHD'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Control', 'ADHD'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_val_predictions_df(val_ds, ev):
    subject_ids = val_ds.df['subject_id'].tolist()
    sites = val_ds.df['site'].tolist()
    assert len(subject_ids) == len(ev['probs']), \
        "val subject-id count does not match prediction count -- loader order assumption broke!"
    return pd.DataFrame({
        'subject_id': subject_ids, 'site': sites,
        'y_true': ev['labels'], 'y_prob': ev['probs'], 'y_pred': ev['preds'],
    })


# ============================================================================
# Per-variant (M0/M1) training with per-epoch checkpointing and interruption
# recovery -- same design as phase2_trm_ablation.py's train_variant().
# ============================================================================
def train_model(variant_name, tag, use_consistency, lam, train_ds, val_ds,
                 out_dir=None, resume=None, epochs=None, patience=None, _stop_after_epoch=None,
                 index_hash=None, split_hash=None):
    """
    `_stop_after_epoch` is a TEST-ONLY hook (used by run_sanity_check()'s
    Test H): if set, training returns immediately after that epoch's
    checkpoint is saved, WITHOUT finalizing results -- simulating a process
    kill right after a checkpoint write. Leave it None for every real run.

    `index_hash`/`split_hash` (item 1, strict configuration validation): sha256
    fingerprints of the dataset file and the exact train/val subject split,
    computed once in main() and passed to every variant. Combined with
    `ARCH_KWARGS`/`use_consistency`/`lam`/epochs/patience/batch_size/lr/
    val_split/seed into a single config fingerprint that is compared against
    the fingerprint stored in any existing checkpoint/result before it is
    reused -- see check_fingerprint_or_raise().
    """
    out_dir = OUT_DIR if out_dir is None else out_dir
    resume = RESUME if resume is None else resume
    epochs = EPOCHS if epochs is None else epochs
    patience = PATIENCE if patience is None else patience
    paths = checkpoint_paths(tag, out_dir)
    current_fp = compute_config_fingerprint(use_consistency, lam, index_hash, split_hash, epochs, patience)

    if resume and os.path.isfile(paths['results']):
        saved_fp = None
        if os.path.isfile(paths['config']):
            saved_config = json.load(open(paths['config']))
            saved_fp = saved_config.get('config_fingerprint')
        require_fingerprint_or_raise(saved_fp, tag, 'skip (reuse completed results)', paths['config'])
        check_fingerprint_or_raise(current_fp, saved_fp, tag, 'skip (reuse completed results)')
        print(f"  [{tag}] already COMPLETE (found {paths['results']}) -- skipping.")
        return pd.read_csv(paths['results']).iloc[0].to_dict(), 'skipped'

    train_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=train_gen, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    model = ADHDProposedModelV2(**ARCH_KWARGS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    # Sentinel is -1.0, not 0.0: a real AUROC can legitimately equal exactly 0.0 on an
    # early/degenerate epoch, and the selection check below is `val_metrics['auroc'] >
    # best['auroc']` -- with a 0.0 sentinel, 0.0 > 0.0 is False, so NO checkpoint would
    # ever be saved and `best_state` would stay None for the whole run, silently falling
    # back to whatever the LAST epoch's weights happen to be, with placeholder (zero)
    # metrics in the saved results. -1.0 is lower than any real AUROC (always in [0, 1],
    # including the documented AUROC-undefined fallback of 0.5 -- see evaluate()), so the
    # FIRST valid validation epoch is always accepted as the initial checkpoint (item 2).
    best = dict(auroc=-1.0, auprc=0.0, accuracy=0.0, f1=0.0, sensitivity=0.0, specificity=0.0,
                confusion_matrix=np.zeros((2, 2), dtype=int), epoch=0)
    best_state = None
    patience_ctr = 0
    history = []
    resumed = False

    if resume and os.path.isfile(paths['latest']):
        # NOTE: config-fingerprint mismatches must ABORT the run (RuntimeError), never be
        # treated as "checkpoint failed to load, start fresh" -- so the fingerprint check
        # happens in its OWN step, deliberately not caught by the broad except below that
        # is reserved for genuine load/corruption failures (item 1: never silently reuse
        # incompatible results).
        try:
            ckpt = torch.load(paths['latest'], map_location=DEVICE, weights_only=False)
        except Exception as e:
            ckpt = None
            print(f"  [{tag}] WARNING: latest checkpoint exists but failed to load ({e!r}) -- "
                  f"starting this variant fresh instead.")

        if ckpt is not None:
            saved_fp = ckpt.get('config_fingerprint')
            require_fingerprint_or_raise(saved_fp, tag, 'resume from latest checkpoint', paths['latest'])
            check_fingerprint_or_raise(current_fp, saved_fp, tag, 'resume from latest checkpoint')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            train_gen.set_state(ckpt['train_generator_state'])
            torch.set_rng_state(ckpt['torch_rng_state'])
            if torch.cuda.is_available() and ckpt.get('torch_cuda_rng_state') is not None:
                torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng_state'])
            np.random.set_state(ckpt['numpy_rng_state'])
            random.setstate(ckpt['python_rng_state'])
            best = ckpt['best']
            best_state = ckpt['best_model_state_dict']
            patience_ctr = ckpt['patience_ctr']
            history = ckpt['history']
            start_epoch = ckpt['epoch'] + 1
            resumed = True
            print(f"  [{tag}] RESUMING from epoch {ckpt['epoch']} "
                  f"(latest checkpoint found, best so far: AUROC={best['auroc']:.4f} @ epoch {best['epoch']})")

    if not resumed:
        set_seed(SEED)
        model = ADHDProposedModelV2(**ARCH_KWARGS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        train_gen.manual_seed(SEED)

    n_params = sum(p.numel() for p in model.parameters())
    auroc_was_undefined_any_epoch = any(h.get('auroc_undefined') for h in history)
    t0 = time.time()
    epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss_cls, epoch_loss_con, n_batches = 0.0, 0.0, 0
        for vol, pheno, label in train_loader:
            vol, label = vol.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()

            z1 = encode_representation(model, vol)
            logits = classify_from_repr(model, z1)
            loss_cls = criterion(logits, label)
            loss = loss_cls
            loss_c_val = 0.0

            if use_consistency:
                # CONSISTENCY_AUGMENTATION=False is a debug-only path (view 2 = view 1, so
                # L_consistency should be ~0 by Test E) -- normal operation keeps this True.
                vol2 = consistency_augment(vol.clone()) if CONSISTENCY_AUGMENTATION else vol.clone()
                z2 = encode_representation(model, vol2)
                loss_c, _ = cosine_consistency_loss(z1, z2)
                loss = loss_cls + lam * loss_c
                loss_c_val = loss_c.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_con += loss_c_val
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader)
        if val_metrics['auroc_undefined']:
            auroc_was_undefined_any_epoch = True
        history.append({
            'epoch': epoch,
            'train_loss_classification': epoch_loss_cls / max(n_batches, 1),
            'train_loss_consistency': epoch_loss_con / max(n_batches, 1),
            'train_loss_total': (epoch_loss_cls + lam * epoch_loss_con) / max(n_batches, 1)
                                 if use_consistency else epoch_loss_cls / max(n_batches, 1),
            'val_auroc': val_metrics['auroc'], 'val_auprc': val_metrics['auprc'],
            'val_accuracy': val_metrics['accuracy'], 'val_f1': val_metrics['f1'],
            'val_sensitivity': val_metrics['sensitivity'], 'val_specificity': val_metrics['specificity'],
            'auroc_undefined': val_metrics['auroc_undefined'],
        })

        if val_metrics['auroc'] > best['auroc']:
            best = dict(auroc=val_metrics['auroc'], auprc=val_metrics['auprc'], accuracy=val_metrics['accuracy'],
                         f1=val_metrics['f1'], sensitivity=val_metrics['sensitivity'],
                         specificity=val_metrics['specificity'], confusion_matrix=val_metrics['confusion_matrix'],
                         probs=val_metrics['probs'], labels=val_metrics['labels'], preds=val_metrics['preds'],
                         epoch=epoch)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            atomic_save(best_state, paths['best'])
        else:
            patience_ctr += 1

        # "Latest" checkpoint saved after EVERY epoch -- this is what makes
        # interruption recovery possible at all.
        checkpoint = dict(
            epoch=epoch, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            train_generator_state=train_gen.get_state(),
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(),
            best=best, best_model_state_dict=best_state, patience_ctr=patience_ctr,
            history=history, seed=SEED, use_consistency=use_consistency, lam=lam,
            config_fingerprint=current_fp,
        )
        atomic_save(checkpoint, paths['latest'])

        if _stop_after_epoch is not None and epoch >= _stop_after_epoch:
            print(f"  [{tag}] ** SIMULATED INTERRUPTION ** after epoch {epoch} "
                  f"(test-only _stop_after_epoch hook) -- returning without finalizing.")
            return None, 'interrupted'

        if patience_ctr >= patience:
            break

    train_time_s = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    eff = measure_flops_and_latency(model)
    stability_mean, stability_std = measure_representation_stability(model, val_loader)

    result = {
        'variant': variant_name,
        'auroc': best['auroc'], 'auprc': best['auprc'], 'accuracy': best['accuracy'],
        'f1': best['f1'], 'sensitivity': best['sensitivity'], 'specificity': best['specificity'],
        'params': n_params, 'trainable_params': eff['trainable_params'], 'flops': eff['flops'],
        'inference_ms_mean': eff['inference_ms_mean'], 'inference_ms_std': eff['inference_ms_std'],
        'train_time_s': round(train_time_s, 1), 'epochs_run': epoch, 'best_epoch': best['epoch'],
        'cosine_stability_mean': stability_mean, 'cosine_stability_std': stability_std,
        'lambda_consistency': lam if use_consistency else 0.0,
        'use_consistency': use_consistency, 'seed': SEED, 'stability_seed': STABILITY_SEED,
        'auroc_was_undefined_any_epoch': auroc_was_undefined_any_epoch,
    }

    # best['probs']/['labels']/['preds'] are the validation-set predictions from the
    # SAME epoch where best_state was captured -- exactly the predictions of the
    # checkpoint being saved, not a re-computed/approximate version.
    predictions = dict(probs=best['probs'], labels=best['labels'], preds=best['preds'])
    atomic_to_csv(build_val_predictions_df(val_ds, predictions), paths['valpred'])
    atomic_to_csv(pd.DataFrame(history), paths['history'])
    plot_confusion_matrix(best['confusion_matrix'], f'{variant_name} -- Confusion Matrix (validation)',
                           paths['confusion'])

    config = {
        'variant': variant_name, 'variant_tag': tag,
        'use_consistency': use_consistency, 'lambda_consistency': result['lambda_consistency'],
        'consistency_augmentation': CONSISTENCY_AUGMENTATION,
        'arch_kwargs': ARCH_KWARGS,
        'epochs_max': epochs, 'epochs_run': epoch, 'best_epoch': best['epoch'], 'batch_size': BATCH_SIZE,
        'lr': LR, 'val_split': VAL_SPLIT, 'patience': patience, 'seed': SEED,
        'stability_seed': STABILITY_SEED,
        'index_file': INDEX_FILE, 'out_dir': out_dir,
        'python_random_seed': SEED, 'numpy_seed': SEED, 'torch_seed': SEED,
        'cuda_available': torch.cuda.is_available(), 'cuda_seed': SEED if torch.cuda.is_available() else None,
        'cudnn_deterministic': True, 'cudnn_benchmark': False,
        'config_fingerprint': current_fp,
    }
    with open(paths['config'], 'w') as f:
        json.dump(config, f, indent=2, default=str)

    atomic_to_csv(pd.DataFrame([result]).drop(columns=['use_consistency'], errors='ignore'), paths['results'])

    if auroc_was_undefined_any_epoch:
        print(f"  [{tag}] WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch -- "
              f"see {paths['history']}'s auroc_undefined column.")

    return result, ('resumed' if resumed else 'fresh')


def plot_training_curves(histories, out_dir):
    """training_curves.png -- loss and val AUROC per epoch, M0 vs M1 overlaid where both ran."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {'m0': '#888888', 'm1': '#2166AC'}
    for tag, hist in histories.items():
        if not hist:
            continue
        df = pd.DataFrame(hist)
        axes[0].plot(df['epoch'], df['train_loss_total'], label=f'{tag.upper()} train loss', color=colors.get(tag))
        axes[1].plot(df['epoch'], df['val_auroc'], label=f'{tag.upper()} val AUROC', color=colors.get(tag))
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Training loss'); axes[0].set_title('Training Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation AUROC'); axes[1].set_title('Validation AUROC')
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/training_curves.png', dpi=300)
    plt.close(fig)


# ============================================================================
# Code-level sanity tests ONLY (synthetic data, no real dataset, no training
# run). Tests A-H below are code-correctness checks, NOT model-performance
# experiments -- none of them produce or report AUROC/AUPRC/F1/etc. Called
# once at the top of main(), before the real dataset is touched.
# ============================================================================
class _SyntheticDS(Dataset):
    """Minimal in-memory stand-in for ADHDDataset, used ONLY by run_sanity_check()'s Test H."""
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.vols = torch.randn(n, 1, 64, 64, 64, generator=g)
        self.labels = torch.randint(0, 2, (n,), generator=g)
        self.df = pd.DataFrame({'subject_id': [f'SYN{i:03d}' for i in range(n)], 'site': ['SIM'] * n})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vols[idx], torch.zeros(5), self.labels[idx]

    def class_weights(self):
        return torch.tensor([1.0, 1.0])


def run_sanity_check():
    print(f"\n{'='*70}\n  PHASE 2 STEP 2 -- CODE-LEVEL SANITY TESTS (synthetic data only)\n{'='*70}")
    torch.manual_seed(0)
    model = ADHDProposedModelV2(**ARCH_KWARGS)
    vol = torch.randn(4, 1, 64, 64, 64, requires_grad=False)

    # ---- Test A -- forward pass: MRI -> model -> representation -> classifier
    print("\n  Test A -- forward pass (MRI -> representation -> classifier):")
    z_a = encode_representation(model, vol)
    logits_a = classify_from_repr(model, z_a)
    print(f"    vol {tuple(vol.shape)} -> representation {tuple(z_a.shape)} -> logits {tuple(logits_a.shape)} "
          f"(expect (4,128) then (4,2))")
    assert z_a.shape == (4, 128) and logits_a.shape == (4, 2)
    print("  TEST A PASSED")

    # ---- Test B -- two-view representation (original MRI -> z1, augmented MRI -> z2)
    print("\n  Test B -- two-view representation (same subject, same weights):")
    vol_v1 = vol
    vol_v2 = consistency_augment(vol.clone())
    print(f"    View 1 == View 2 (should be False): {torch.equal(vol_v1, vol_v2)}")
    assert not torch.equal(vol_v1, vol_v2)
    z1 = encode_representation(model, vol_v1)
    z2 = encode_representation(model, vol_v2)
    print(f"    z1 shape: {tuple(z1.shape)}  z2 shape: {tuple(z2.shape)}  (expect (4, 128) both)")
    assert z1.shape == z2.shape == (4, 128)
    print(f"    Same model instance (same weights) used for both views: True "
          f"(one `model` object, called twice via encode_representation(model, ...) -- "
          f"no separate encoder was created for view 2)")
    print("  TEST B PASSED")

    # ---- Test C -- consistency loss returns a finite scalar
    print("\n  Test C -- consistency loss (finite scalar):")
    loss_c, cos_sim = cosine_consistency_loss(z1, z2)
    print(f"    L_consistency = {loss_c.item():.6f}  (scalar: {loss_c.dim() == 0})  "
          f"finite: {torch.isfinite(loss_c).item()}")
    print(f"    cosine similarities per sample: {cos_sim.detach().numpy().round(4)}")
    assert loss_c.dim() == 0 and torch.isfinite(loss_c)
    print("  TEST C PASSED")

    # ---- Test D -- gradient flow (backward pass, finite, no NaN/Inf)
    print("\n  Test D -- gradient flow through the combined objective:")
    logits = classify_from_repr(model, z1)
    target = torch.tensor([0, 1, 0, 1])
    criterion = nn.CrossEntropyLoss()
    loss_cls = criterion(logits, target)
    loss_total = loss_cls + CONSISTENCY_LAMBDA * loss_c
    print(f"    L_classification={loss_cls.item():.4f}  L_consistency={loss_c.item():.4f}  "
          f"L_total={loss_total.item():.4f}  (L_total = L_cls + {CONSISTENCY_LAMBDA} * L_consistency)")
    assert torch.isfinite(loss_total), "L_total is NaN/Inf!"

    model.zero_grad()
    loss_total.backward()
    grad_ok = model.encoder.block1.conv.pointwise.weight.grad is not None
    print(f"    Gradients exist (encoder received a gradient): {grad_ok}")
    assert grad_ok
    n_checked = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}!"
            n_checked += 1
    print(f"    No NaN/Inf in any of the {n_checked} gradient tensors: True")
    print("  TEST D PASSED")

    # ---- Test E -- identical representations -> consistency loss ~ 0
    print("\n  Test E -- identical representations give ~zero consistency loss:")
    loss_identical, cos_identical = cosine_consistency_loss(z1, z1)
    print(f"    cosine_consistency_loss(z1, z1) = {loss_identical.item():.8f}  "
          f"(expect ~0, i.e. cos_sim ~1): cos_sim={cos_identical.detach().numpy().round(6)}")
    assert torch.allclose(loss_identical, torch.zeros_like(loss_identical), atol=1e-5), \
        f"expected ~0 for identical representations, got {loss_identical.item()}"
    print("  TEST E PASSED")

    # ---- Test F -- validation mode does not apply random augmentation
    print("\n  Test F -- augmentation isolation (no random augmentation outside the two-view step):")
    model.eval()
    with torch.no_grad():
        z_val_1 = encode_representation(model, vol)   # validation-style call: NO consistency_augment()
        z_val_2 = encode_representation(model, vol)   # same input, same eval-mode model, called again
    val_deterministic = torch.allclose(z_val_1, z_val_2, atol=1e-7)
    print(f"    encode_representation(model, vol) called twice on IDENTICAL input in eval mode "
          f"(validation code path never calls consistency_augment()): "
          f"outputs identical = {val_deterministic}")
    assert val_deterministic, "validation-path representation is non-deterministic -- augmentation leaked in!"
    print("    (Enforced at the dataset level too: train_ds and val_ds are both built with "
          "augment=False in main() -- View 1 is always the deterministic, unaugmented "
          "preprocessed MRI; dataset.py's own augment=True random-flip path is never used by "
          "this file, and is untouched for Phase-1's own scripts; consistency_augment() is "
          "called explicitly ONLY inside train_model()'s training loop and the post-hoc "
          "stability measurement, never inside evaluate(), to produce View 2. There is no "
          "separate held-out test_ds anywhere in this pipeline -- see the module docstring's "
          "VALIDATION, NOT AN INDEPENDENT TEST SET note.)")
    model.train()
    print("  TEST F PASSED")

    # ---- Test G -- encode_representation()+classify_from_repr() == model.forward() exactly
    print("\n  Test G -- representation path matches the model's own forward() exactly:")
    model.eval()
    with torch.no_grad():
        logits_direct = model(vol)                                    # model's own forward()
        logits_via_repr = classify_from_repr(model, encode_representation(model, vol))
    same_logits = torch.allclose(logits_direct, logits_via_repr, atol=1e-6)
    print(f"    model(vol) vs classify_from_repr(model, encode_representation(model, vol)): "
          f"max abs diff = {(logits_direct - logits_via_repr).abs().max().item():.2e}  "
          f"identical = {same_logits}")
    assert same_logits, ("encode_representation()/classify_from_repr() diverge from model.forward() -- "
                          "this would mean the consistency loss is computed on a DIFFERENT "
                          "representation than the one actually used for classification!")
    model.train()
    print("  TEST G PASSED (no parallel/divergent computation path -- same architecture, same weights)")

    # ---- Test H -- genuine interrupt-then-resume, compared against an uninterrupted reference ----
    print("\n  Test H -- interrupt-then-resume reproduces the uninterrupted training-loss sequence:")
    with tempfile.TemporaryDirectory(prefix='phase2_consistency_resume_test_') as tmp:
        train_ds = _SyntheticDS(16, seed=1)
        val_ds = _SyntheticDS(8, seed=2)
        TEST_EPOCHS = 4

        set_seed(SEED)
        ref_result, ref_status = train_model('Synthetic (ref)', 'synref', True, CONSISTENCY_LAMBDA,
                                              train_ds, val_ds, out_dir=tmp, resume=False,
                                              epochs=TEST_EPOCHS, patience=100)
        ref_history = pd.read_csv(f'{tmp}/training_history_synref.csv')
        print(f"    Uninterrupted reference run: {len(ref_history)} epochs, status={ref_status}, "
              f"final train_loss_total={ref_history['train_loss_total'].iloc[-1]:.6f}")
        assert len(ref_history) == TEST_EPOCHS

        set_seed(SEED)
        _, int_status = train_model('Synthetic (interrupted)', 'synint', True, CONSISTENCY_LAMBDA,
                                     train_ds, val_ds, out_dir=tmp, resume=False,
                                     epochs=TEST_EPOCHS, patience=100, _stop_after_epoch=2)
        assert int_status == 'interrupted'
        assert os.path.isfile(f'{tmp}/synint_latest_checkpoint.pt')
        assert not os.path.isfile(f'{tmp}/synint_results.csv')
        print(f"    Simulated interruption after epoch 2: latest checkpoint exists, results.csv does not (correct)")

        resumed_result, resumed_status = train_model('Synthetic (resumed)', 'synint', True, CONSISTENCY_LAMBDA,
                                                        train_ds, val_ds, out_dir=tmp, resume=True,
                                                        epochs=TEST_EPOCHS, patience=100)
        assert resumed_status == 'resumed'
        resumed_history = pd.read_csv(f'{tmp}/training_history_synint.csv')
        print(f"    Resumed run: status={resumed_status}, {len(resumed_history)} epochs total "
              f"(2 before interruption + 2 after resuming), "
              f"final train_loss_total={resumed_history['train_loss_total'].iloc[-1]:.6f}")
        assert len(resumed_history) == TEST_EPOCHS

        loss_diff = (ref_history['train_loss_total'].values - resumed_history['train_loss_total'].values)
        max_abs_diff = np.abs(loss_diff).max()
        print(f"    Per-epoch train_loss_total, reference vs resumed: max abs diff = {max_abs_diff:.3e}")
        print(f"      reference epochs 1-4 : {ref_history['train_loss_total'].values.round(6).tolist()}")
        print(f"      resumed   epochs 1-4 : {resumed_history['train_loss_total'].values.round(6).tolist()}")
        assert max_abs_diff < 1e-5, (
            f"resumed training-loss sequence diverges from the uninterrupted reference "
            f"(max abs diff {max_abs_diff:.3e}) -- RNG/DataLoader-generator state was not "
            f"restored correctly on resume!")
        print("    Resumed sequence matches the uninterrupted reference to within 1e-5 "
              "(CPU, NUM_WORKERS=0 -- exact reproducibility is expected here).")

        print(f"    Resumed variant's architecture unchanged: params={resumed_result['params']:,} "
              f"(expect 114,482: {'MATCH' if resumed_result['params'] == 114482 else 'MISMATCH!'})")
        assert resumed_result['params'] == 114482
    print("  TEST H PASSED")

    # ---- Test I -- strict configuration validation (item 1) ----
    print("\n  Test I -- strict configuration validation refuses to reuse a changed configuration:")
    with tempfile.TemporaryDirectory(prefix='phase2_consistency_fingerprint_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=40)
        val_ds = _SyntheticDS(4, seed=41)

        result, status = train_model('FP test', 'synfp', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                       out_dir=tmp, resume=False, epochs=2, patience=100,
                                       index_hash='hashA', split_hash='splitA')
        assert status == 'fresh'
        assert os.path.isfile(f'{tmp}/synfp_results.csv')

        raised = False
        try:
            train_model('FP test', 'synfp', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=2, patience=100,
                        index_hash='hashB_DIFFERENT', split_hash='splitA')
        except RuntimeError as e:
            raised = True
            print(f"    Changed dataset (index_hash) correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised, "a changed index_hash was NOT detected/refused when skipping completed results!"

        result2, status2 = train_model('FP test', 'synfp', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                         out_dir=tmp, resume=True, epochs=2, patience=100,
                                         index_hash='hashA', split_hash='splitA')
        assert status2 == 'skipped'
        print("    Identical configuration correctly reuses the completed result (status=skipped)")

        raised_lam = False
        try:
            train_model('FP test', 'synfp', True, 0.10, train_ds, val_ds,   # DIFFERENT lambda
                        out_dir=tmp, resume=True, epochs=2, patience=100,
                        index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised_lam = True
            print(f"    Changed consistency weight (lambda) correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised_lam, "a changed lambda was NOT detected/refused when skipping completed results!"

        _, int_status = train_model('FP test 2', 'synfp2', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                     out_dir=tmp, resume=False, epochs=4, patience=100,
                                     index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status == 'interrupted'
        raised2 = False
        try:
            train_model('FP test 2', 'synfp2', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=4, patience=100,
                        index_hash='hashA', split_hash='splitB_DIFFERENT')
        except RuntimeError as e:
            raised2 = True
            print(f"    Changed split (split_hash) correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised2, "a changed split_hash was NOT detected/refused when resuming from a checkpoint!"

        # A results.csv/config.json with NO config_fingerprint at all (predates the
        # safeguard, or a different script version) must be refused for skip too --
        # never silently treated as "nothing to compare, go ahead".
        _, status3 = train_model('FP test 3', 'synfp3', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                  out_dir=tmp, resume=False, epochs=2, patience=100,
                                  index_hash='hashA', split_hash='splitA')
        assert status3 == 'fresh'
        cfg_path = f'{tmp}/config_synfp3.json'
        cfg = json.load(open(cfg_path))
        del cfg['config_fingerprint']
        json.dump(cfg, open(cfg_path, 'w'))
        raised3 = False
        try:
            train_model('FP test 3', 'synfp3', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=2, patience=100,
                        index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised3 = True
            print(f"    Missing config_fingerprint correctly REFUSED on skip: {str(e).splitlines()[0]}")
        assert raised3, "a missing config_fingerprint was NOT refused when skipping completed results!"

        # Same check on the RESUME path: a checkpoint saved with no config_fingerprint
        # (e.g. produced by a pre-safeguard script version) must also be refused.
        _, int_status2 = train_model('FP test 4', 'synfp4', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                      out_dir=tmp, resume=False, epochs=4, patience=100,
                                      index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status2 == 'interrupted'
        ckpt_path = f'{tmp}/synfp4_latest_checkpoint.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        del ckpt['config_fingerprint']
        torch.save(ckpt, ckpt_path)
        raised4 = False
        try:
            train_model('FP test 4', 'synfp4', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=4, patience=100,
                        index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised4 = True
            print(f"    Missing config_fingerprint correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised4, "a missing config_fingerprint was NOT refused when resuming from a checkpoint!"
    print("  TEST I PASSED")

    # ---- Test J -- first valid checkpoint saved even when AUROC equals exactly 0.0 (item 2) ----
    print("\n  Test J -- first-epoch checkpoint is saved even when AUROC is exactly 0.0:")
    with tempfile.TemporaryDirectory(prefix='phase2_consistency_auroc0_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=50)
        val_ds = _SyntheticDS(4, seed=51)

        global evaluate
        original_evaluate = evaluate
        call_count = {'n': 0}

        def _forced_zero_evaluate(model, loader):
            ev = original_evaluate(model, loader)
            call_count['n'] += 1
            if call_count['n'] == 1:
                ev = dict(ev)
                ev['auroc'] = 0.0   # force exactly zero on the very first validation epoch
            return ev

        evaluate = _forced_zero_evaluate
        try:
            result, status = train_model('AUROC0 test', 'synauroc0', False, 0.0, train_ds, val_ds,
                                          out_dir=tmp, resume=False, epochs=1, patience=100)
        finally:
            evaluate = original_evaluate

        best_ckpt_exists = os.path.isfile(f'{tmp}/best_synauroc0_model.pt')
        print(f"    Forced epoch-1 AUROC=0.0 -> best checkpoint saved: {best_ckpt_exists}")
        assert best_ckpt_exists, \
            "no best checkpoint was saved even though epoch 1 completed -- sentinel bug not fixed!"
        print(f"    Saved best_epoch = {result['best_epoch']}  (expect 1, not left at the 0 placeholder)")
        assert result['best_epoch'] == 1
    print("  TEST J PASSED")

    # ---- Section 14 -- parameter count must not unexpectedly increase
    print("\n  Section 14 -- parameter count with vs without consistency:")
    model_m0 = ADHDProposedModelV2(**ARCH_KWARGS)   # same class/kwargs as M1 -- consistency only changes the LOSS
    params_m0 = sum(p.numel() for p in model_m0.parameters())
    params_m1 = sum(p.numel() for p in model.parameters())
    print(f"    params (M0, no consistency) = {params_m0:,}")
    print(f"    params (M1, with consistency) = {params_m1:,}")
    print(f"    difference = {params_m1 - params_m0:,} "
          f"(expected 0: consistency learning only adds a loss term at training time -- "
          f"encode_representation()/cosine_consistency_loss() introduce zero new nn.Parameters, "
          f"and both M0 and M1 instantiate the identical ADHDProposedModelV2(**ARCH_KWARGS))")
    assert params_m0 == params_m1, "parameter count changed between M0 and M1 -- investigate!"
    print("  SECTION 14 CHECK PASSED")

    # ---- Section 14b -- M0 and M1 start from EQUIVALENT initial weights (same seed)
    print("\n  Section 14b -- M0/M1 initial weights are equivalent (same seed -> identical init):")
    set_seed(SEED)
    model_a = ADHDProposedModelV2(**ARCH_KWARGS)
    set_seed(SEED)
    model_b = ADHDProposedModelV2(**ARCH_KWARGS)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    all_equal = all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    print(f"    Two fresh ADHDProposedModelV2(**ARCH_KWARGS) instances, both built right after "
          f"set_seed({SEED}): all {len(sd_a)} parameter/buffer tensors identical = {all_equal}")
    assert all_equal, "same seed did not produce identical initial weights -- investigate!"
    print("    (train_model() does exactly this -- set_seed(SEED) immediately before "
          f"constructing the model -- for BOTH the M0 and M1 calls in main(), with the same "
          f"optimizer (AdamW, same lr/weight_decay) and the same evaluate()/scheduler, so the "
          f"ONLY difference between the two runs is the consistency loss term.)")
    print("  SECTION 14b CHECK PASSED")

    print(f"\n{'='*70}\n  ALL CODE-LEVEL SANITY TESTS (A-J) + SECTIONS 14/14b PASSED ✓\n{'='*70}")
    print("  (Synthetic data only -- no real dataset was loaded, no AUROC/AUPRC/F1/etc. "
          "was computed. The real M0/M1 training run on your actual dataset starts next.)")


# ============================================================================
# Dataset/experiment-setup integrity checks (real-CSV-level, NOT synthetic).
# Run once in main(), before any ADHDDataset is built.
# ============================================================================
def validate_dataset_integrity(raw_df, train_ids, val_ids):
    print(f"\n{'='*70}\n  DATASET INTEGRITY CHECKS\n{'='*70}")

    dup_mask = raw_df['subject_id'].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    print(f"  Duplicate subject_id rows: {n_dup}")
    if n_dup > 0:
        print(f"    Example duplicates: {raw_df.loc[dup_mask, 'subject_id'].unique()[:10].tolist()}")
    assert n_dup == 0, f"{n_dup} duplicate subject_id rows found in {INDEX_FILE} -- fix the index before training."

    missing = [p for p in raw_df['npy_path'] if not os.path.isfile(p)]
    print(f"  Missing .npy files: {len(missing)} / {len(raw_df)}")
    if missing:
        print(f"    Example missing paths: {missing[:10]}")
    assert not missing, f"{len(missing)} .npy files referenced in {INDEX_FILE} do not exist on disk."

    overlap = set(train_ids) & set(val_ids)
    print(f"  Train/val subject overlap: {len(overlap)}")
    assert not overlap, f"{len(overlap)} subjects appear in BOTH train and val -- subject-level leakage!"
    print("  Subject-level independence confirmed: every subject is in exactly one of train/val.")

    val_labels = raw_df[raw_df['subject_id'].isin(val_ids)]['label']
    val_counts = val_labels.value_counts().to_dict()
    print(f"  Validation label distribution: {val_counts}")
    assert val_labels.nunique() == 2, (
        f"validation split contains only one class ({val_counts}) -- AUROC/AUPRC are undefined "
        f"in this case; re-check the split (StratifiedShuffleSplit should prevent this on a "
        f"reasonably sized, reasonably balanced dataset).")

    print(f"  ARCH_KWARGS = {ARCH_KWARGS}  (no 'use_phenotype' key -> defaults to False in "
          f"ADHDProposedModelV2 -> primary experiment is MRI-only, phenotype branch not used)")
    assert 'use_phenotype' not in ARCH_KWARGS or not ARCH_KWARGS['use_phenotype'], \
        "ARCH_KWARGS sets use_phenotype=True -- this would make the primary experiment MRI+phenotype, not MRI-only!"

    print(f"  Train subjects: {len(train_ids)}   Val subjects: {len(val_ids)}   Total: {len(raw_df)}")
    print("  ALL DATASET INTEGRITY CHECKS PASSED ✓")


def save_site_label_counts(raw_df, train_ids, val_ids, out_dir):
    """
    Per-site label-count breakdown for both splits (item 3: 'save ... per-site
    label counts') -- makes site distribution and class balance per split
    visible and reviewable, not just asserted by a pass/fail check.
    """
    rows = []
    for split_name, ids in [('train', train_ids), ('val', val_ids)]:
        sub = raw_df[raw_df['subject_id'].isin(ids)]
        counts = sub.groupby(['site', 'label']).size().reset_index(name='count')
        counts.insert(0, 'split', split_name)
        rows.append(counts)
    site_label_df = pd.concat(rows, ignore_index=True)
    atomic_to_csv(site_label_df, f'{out_dir}/site_label_counts.csv')
    print(f"\n  Per-site label counts (train + val) -> {out_dir}/site_label_counts.csv")
    print(site_label_df.to_string(index=False))
    return site_label_df


def main():
    # ---- code-level sanity checks FIRST, on synthetic data only ----
    run_sanity_check()

    # ---- Real dataset: connect to the SAME ADHD-200 pipeline used everywhere
    #      else in this project (subject-level split, leakage-free fit_stats) ----
    print(f"\n{'='*70}\n  LOADING REAL DATASET\n{'='*70}")
    print(f"  INDEX_FILE = {INDEX_FILE}")
    raw_df = pd.read_csv(INDEX_FILE)
    raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
    labels_all = raw_df['label'].values
    print(f"  Total subjects : {len(raw_df)}")

    # ---- Subject-level split: loaded from disk on a RESUME (if already saved,
    #      guaranteeing the identical split across any restart), else computed fresh ----
    train_ids_path = f'{OUT_DIR}/train_subject_ids.csv'
    val_ids_path = f'{OUT_DIR}/val_subject_ids.csv'
    if RESUME and os.path.isfile(train_ids_path) and os.path.isfile(val_ids_path):
        train_ids = pd.read_csv(train_ids_path)['subject_id'].tolist()
        val_ids = pd.read_csv(val_ids_path)['subject_id'].tolist()
        print(f"  Loaded EXISTING train/val subject-ID split from disk ({train_ids_path}, "
              f"{val_ids_path}) -- guarantees the identical split across any restart.")
    else:
        set_seed(SEED)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
        train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
        val_ids = raw_df.iloc[val_idx]['subject_id'].tolist()
        atomic_to_csv(pd.DataFrame({'subject_id': train_ids}), train_ids_path)
        atomic_to_csv(pd.DataFrame({'subject_id': val_ids}), val_ids_path)
        print(f"  Computed a new split and saved it to {train_ids_path} / {val_ids_path}")

    validate_dataset_integrity(raw_df, train_ids, val_ids)
    save_site_label_counts(raw_df, train_ids, val_ids, OUT_DIR)

    # Dataset-identity and split fingerprints (item 1) -- computed once here, passed to
    # every variant, so a changed dataset file or a changed split is detected BEFORE any
    # checkpoint/result produced under the old one is silently reused.
    index_hash = hash_file(INDEX_FILE)
    split_hash = hash_split(train_ids, val_ids)
    print(f"  Dataset fingerprint: index_hash={index_hash[:12]}...  split_hash={split_hash[:12]}...")

    # Imputation/normalization statistics are fit on the TRAINING subjects only,
    # then reused (via fit_stats) for validation -- no leakage.
    train_stats_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()

    # View-1 isolation (Step 2 ONLY, see module docstring): train_ds is built with
    # augment=False -- View 1 is the deterministic, unaugmented preprocessed MRI;
    # View 2 comes ONLY from consistency_augment() inside train_model(). M0 and M1
    # both consume this identical, deterministic train_ds.
    train_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False, fit_stats=fit_stats)
    val_ds = ADHDDataset(INDEX_FILE, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
    print(f"  Train : {len(train_ds)}   Val : {len(val_ds)}  "
          f"(train augment=False -> View 1 deterministic; View 2 via consistency_augment() only; "
          f"val augment=False, deterministic)")

    # ---- Build the run list from RUN_M0 / RUN_M1 -- no manual training-loop editing ----
    runs = []
    if RUN_M0:
        runs.append(('m0', 'M0 (no consistency)', False, 0.0))
    if RUN_M1:
        runs.append(('m1', f'M1 (consistency, lambda={CONSISTENCY_LAMBDA})', True, CONSISTENCY_LAMBDA))
        if RUN_LAMBDA_SWEEP:
            for lam in LAMBDA_SWEEP:
                runs.append((f'm1_lam{lam}', f'M1 (consistency, lambda={lam})', True, lam))
    assert runs, "Both RUN_M0 and RUN_M1 are False -- nothing to train."

    histories = {}
    for tag, name, use_consistency, lam in runs:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        result, status = train_model(name, tag, use_consistency, lam, train_ds, val_ds,
                                      index_hash=index_hash, split_hash=split_hash)
        if status != 'skipped':
            hist_df = pd.read_csv(f'{OUT_DIR}/training_history_{tag}.csv')
            histories[tag] = hist_df.to_dict('records')
        else:
            try:
                histories[tag] = pd.read_csv(f'{OUT_DIR}/training_history_{tag}.csv').to_dict('records')
            except FileNotFoundError:
                histories[tag] = []

        print(f"  [{status}]  AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}  "
              f"Acc={result['accuracy']:.4f}  F1={result['f1']:.4f}  "
              f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}")
        print(f"  params={int(result['params']):,}  GFLOPs={result['flops']/1e9:.4f}  "
              f"inference={result['inference_ms_mean']:.2f}+/-{result['inference_ms_std']:.2f}ms")
        print(f"  representation cosine stability (val, held-out): "
              f"{result['cosine_stability_mean']:.4f} +/- {result['cosine_stability_std']:.4f}")
        print(f"  Saved: {tag}_results.csv, training_history_{tag}.csv, confusion_matrix_{tag}.png, "
              f"best_{tag}_model.pt, {tag}_latest_checkpoint.pt, config_{tag}.json, val_predictions_{tag}.csv")

    # ---- Combined ablation table + representation-stability table, built from each
    #      variant's OWN saved results.csv (resume-safe, same pattern as Step 1) ----
    all_results = []
    for tag, name, use_consistency, lam in runs:
        rpath = f'{OUT_DIR}/{tag}_results.csv'
        assert os.path.isfile(rpath), f"{tag}_results.csv missing after train_model() -- should not happen"
        all_results.append(pd.read_csv(rpath).iloc[0].to_dict())

    results_df = pd.DataFrame(all_results)[
        ['variant', 'lambda_consistency', 'params', 'trainable_params', 'flops', 'auroc', 'auprc',
         'accuracy', 'f1', 'sensitivity', 'specificity', 'cosine_stability_mean', 'cosine_stability_std',
         'inference_ms_mean', 'inference_ms_std', 'train_time_s', 'epochs_run', 'best_epoch', 'seed',
         'auroc_was_undefined_any_epoch']
    ]
    atomic_to_csv(results_df, f'{OUT_DIR}/consistency_ablation_results.csv')
    atomic_to_csv(results_df[['variant', 'lambda_consistency', 'cosine_stability_mean', 'cosine_stability_std']],
                  f'{OUT_DIR}/representation_stability.csv')

    print(f"\n{'='*90}\n  PHASE 2 STEP 2 -- M0 vs M1 SUMMARY (VALIDATION-SET METRICS)\n{'='*90}")
    print(results_df[['variant', 'auroc', 'auprc', 'accuracy', 'f1',
                       'cosine_stability_mean', 'cosine_stability_std']].to_string(index=False))
    print("\n  NOTE: all metrics above are from the single held-out VALIDATION split -- not an "
          "independent test set (none exists in this pipeline). Do not cross-compare these "
          "numbers against Step 1's A5-new (different augmentation policy -- see the "
          "AUGMENTATION-COMPARISON NOTE in this file's module docstring). Multi-fold CV / "
          "cross-site evaluation are a separate, not-yet-approved future stage.")

    if RUN_M0 and RUN_M1:
        m0 = results_df.iloc[0]
        m1 = results_df.iloc[1]
        stability_improved = m1['cosine_stability_mean'] > m0['cosine_stability_mean']
        perf_maintained = m1['auroc'] >= m0['auroc'] - 0.02   # small tolerance, not a hard pass/fail
        print(f"\nStability higher with consistency loss: {stability_improved} "
              f"({m0['cosine_stability_mean']:.4f} -> {m1['cosine_stability_mean']:.4f})")
        print(f"AUROC maintained or improved (within 0.02 tolerance): {perf_maintained} "
              f"({m0['auroc']:.4f} -> {m1['auroc']:.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(results_df['variant'], results_df['cosine_stability_mean'],
                    yerr=results_df['cosine_stability_std'], color=['#888888', '#2166AC'][:len(results_df)],
                    capsize=4)
        axes[0].set_ylabel('Mean cosine similarity (z_orig, z_aug)')
        axes[0].set_title('Representation Stability')
        axes[0].tick_params(axis='x', rotation=20)

        axes[1].bar(results_df['variant'], results_df['auroc'], color=['#888888', '#2166AC'][:len(results_df)])
        axes[1].set_ylabel('Validation AUROC')
        axes[1].set_title('Classification Performance')
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis='x', rotation=20)

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{OUT_DIR}/comparison_figure.png', dpi=300)
        plt.close(fig)
        print(f"Figure  -> {OUT_DIR}/comparison_figure.png")

    plot_training_curves(histories, OUT_DIR)

    print(f"\nResults -> {OUT_DIR}/consistency_ablation_results.csv")
    print(f"Stability -> {OUT_DIR}/representation_stability.csv")
    print(f"Training curves -> {OUT_DIR}/training_curves.png")
    print(f"Subject IDs -> {OUT_DIR}/train_subject_ids.csv, {OUT_DIR}/val_subject_ids.csv")
    print(f"Per-site label counts -> {OUT_DIR}/site_label_counts.csv")

    if results_df['auroc_was_undefined_any_epoch'].any():
        flagged = results_df[results_df['auroc_was_undefined_any_epoch']]['variant'].tolist()
        print(f"\n*** WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch for: "
              f"{flagged} -- check training_history_{{tag}}.csv's auroc_undefined column. ***")

    print(f"\nAll Step-2 outputs saved under {OUT_DIR}/ "
          f"(Phase-1 results and Step-1's trm_ablation_results.csv are untouched).")


if __name__ == '__main__':
    main()
