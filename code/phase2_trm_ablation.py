"""
PHASE 2, Section 1D: the required TRM ablation -- WITH experiment tracking,
per-epoch checkpointing, and interruption recovery.

    A3      : tokens + FastKAN                              (no TRM)
    A4      : tokens + FastKAN + original MHA-based TRM      (Phase 1, unmodified)
    A4-new  : tokens + FastKAN + proposed pairwise TRM        (Phase 2)
    A5-new  : tokens + FastKAN + proposed pairwise TRM + Anatomical Attention

Same stratified 80/20 split, same seeding, same leakage-free fit_stats
methodology, same model architecture/loss/augmentation policy/parameter
counts/ablation definitions as the original version of this script --
NOTHING about the experiment itself changed. What changed is purely
non-architectural instrumentation, in the same pattern already used by
phase2_consistency.py:

  - Per-variant checkpointing: a "best" checkpoint (by validation AUROC,
    with a documented fallback -- see `evaluate_full()` -- when AUROC is
    undefined) and a "latest" checkpoint saved after EVERY epoch, atomically
    (write to a .tmp file, then os.replace() -- never leaves a half-written
    checkpoint on disk even if the process dies mid-write).
  - Interruption recovery: set RESUME = True (default) and re-run this
    script. Any variant whose `{tag}_results.csv` already exists is treated
    as COMPLETE and skipped; any variant with a `{tag}_latest_checkpoint.pt`
    but no `{tag}_results.csv` is resumed from that checkpoint -- model,
    optimizer, scheduler, early-stopping state, training history, AND the
    exact RNG/DataLoader-shuffle state are all restored, so the per-epoch
    training-loss sequence after resuming continues the same sequence an
    uninterrupted run would have produced (verified in run_sanity_check()'s
    resume test below -- NUM_WORKERS=0 and CPU execution make this exact;
    on a CUDA GPU, cuDNN non-determinism can introduce tiny numerical drift
    even with the RNG states restored, which is a CUDA/cuDNN property, not
    a bug in the checkpointing logic).
  - Per-variant outputs (ALL under OUT_DIR, prefixed by the variant's short
    tag -- A3/A4/A4-new/A5-new): best-checkpoint .pt, latest-checkpoint .pt,
    per-epoch training history .csv, per-subject validation predictions
    .csv (subject_id, site, y_true, y_prob, y_pred), final per-variant
    results .csv, and a config .json (seed, hyperparams, arch kwargs, best
    epoch). Each variant's results.csv is written immediately when that
    variant finishes -- NOT held in memory until all four complete -- so
    completed variants' results survive even if a later variant is
    interrupted.
  - The final aggregate `trm_ablation_results.csv` + comparison figure are
    built by READING each variant's saved `{tag}_results.csv` (not from
    in-memory state), so the aggregate step itself is resume-safe too.
  - Exact train/validation subject IDs are saved once (`train_subject_ids.
    csv` / `val_subject_ids.csv`) and, on any restart, loaded from disk
    rather than re-derived -- an extra guarantee (on top of the fixed seed)
    that every variant, across any number of restarts, trains and
    validates on the IDENTICAL subject-level split.
  - Strict configuration validation: every checkpoint/result carries a
    `config_fingerprint` (a sha256 of the dataset file + the exact subject
    split, plus kwargs/epochs/patience/batch_size/lr/val_split/seed). Before
    skipping a completed variant OR resuming an interrupted one, the CURRENT
    run's fingerprint is compared against the saved one; any mismatch
    RAISES (never silently reuses a result produced under different
    settings) -- see `check_fingerprint_or_raise()`. A checkpoint/result with
    NO `config_fingerprint` at all (it predates this safeguard, or was
    produced by a different script version) is refused the same way, never
    silently trusted -- see `require_fingerprint_or_raise()`. Either case
    requires pointing `OUT_DIR` at a fresh, empty directory.
  - The first valid validation checkpoint is always saved, even if its
    AUROC is exactly 0.0 (the "best so far" sentinel is -1.0, not 0.0 --
    see the comment at `best = dict(auroc=-1.0, ...)` in `train_variant()`).

`run_sanity_check()` (called first in `main()`, synthetic data only, no
real dataset touched) verifies:
    Test A -- forward pass + parameter counts for all four variants
    Test B -- checkpoint save/load round-trip
    Test C -- genuine interrupt-then-resume reproduces the uninterrupted
              per-epoch training-loss sequence
    Test D -- strict configuration validation raises on a changed dataset
              (index_hash) or changed split (split_hash), both when
              skipping a completed variant and when resuming an
              interrupted one; an UNCHANGED configuration still skips
              cleanly; a completed result or checkpoint with NO
              config_fingerprint at all is also refused, both when
              skipping and when resuming
    Test E -- the first-epoch checkpoint is saved even when that epoch's
              AUROC is exactly 0.0 (forced via a temporary monkeypatch of
              evaluate_full(), restored immediately afterward)
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
DATA_ROOT   = '/kaggle/working'
INDEX_FILE  = f'{DATA_ROOT}/processed_index.csv'
INDEX_CSV   = INDEX_FILE    # alias kept for consistency with the rest of this project's scripts
OUT_DIR     = f'{DATA_ROOT}/phase2'          # SEPARATE from Phase-1 outputs
os.makedirs(OUT_DIR, exist_ok=True)

RESUME      = True    # True: skip completed variants, resume interrupted ones from their
                       # latest checkpoint. False: ignore any existing checkpoints/results
                       # and retrain every variant from scratch (existing files are overwritten
                       # as training proceeds, same as this script's original behaviour).

EPOCHS      = 150     # same max budget as every other training script here
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

VARIANTS = [
    ('A3 tokens+FastKAN',             'A3',      dict(trm_type='none',     use_attention=False)),
    ('A4 +MHA TRM (Phase 1)',         'A4',      dict(trm_type='mha',      use_attention=False)),
    ('A4-new +Pairwise TRM',          'A4-new',  dict(trm_type='pairwise', use_attention=False)),
    ('A5-new +Pairwise TRM +AnatAttn','A5-new',  dict(trm_type='pairwise', use_attention=True)),
]


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
# the process is killed mid-write.
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
        best=f'{out_dir}/{tag}_best_model.pt',
        history=f'{out_dir}/{tag}_training_history.csv',
        valpred=f'{out_dir}/{tag}_val_predictions.csv',
        results=f'{out_dir}/{tag}_results.csv',
        config=f'{out_dir}/{tag}_config.json',
    )


def hash_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def hash_split(train_ids, val_ids):
    return hashlib.sha256((','.join(sorted(train_ids)) + '|' + ','.join(sorted(val_ids))).encode()).hexdigest()


def compute_config_fingerprint(kwargs, index_hash, split_hash, epochs, patience):
    """
    Everything that must be IDENTICAL between the run that produced a saved
    checkpoint/result and the current run before it is safe to resume or
    skip. If any of these differ, the checkpoint/result was produced under
    different conditions and must never be silently reused.
    """
    return {
        'kwargs': kwargs, 'index_hash': index_hash, 'split_hash': split_hash,
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
    subject-level split (split_hash), model configuration (kwargs), seed,
    learning rate, epochs/patience budget, batch size, and val_split are
    all covered.
    """
    mismatches = {k: (saved_fp.get(k), current_fp[k]) for k in current_fp if saved_fp.get(k) != current_fp[k]}
    if mismatches:
        lines = "\n".join(f"    {k}: saved={old!r}  current={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- configuration has changed since the saved "
            f"checkpoint/result was produced:\n{lines}\n"
            f"  Never silently reusing incompatible results. Either revert the changed "
            f"setting(s) above, or point OUT_DIR at a fresh, empty directory for this new "
            f"configuration (e.g. OUT_DIR = '/kaggle/working/phase2_v2')."
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
            f"the current safeguard (e.g. OUT_DIR = '/kaggle/working/phase2_v2')."
        )


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
    return flops, float(np.mean(times)), float(np.std(times))


def evaluate_full(model, loader):
    """
    Documented AUROC-undefined fallback: if the validation batch passed to
    roc_auc_score/average_precision_score contains only one class (can only
    happen if the validation split itself is degenerate -- the integrity
    checks in main() guard against this at the subject level), both metrics
    fall back to (0.5, mean label) and `auroc_undefined=True` is returned so
    callers can flag it rather than silently treating 0.5 as a real score.
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            p = torch.softmax(model(vol), dim=1)[:, 1].cpu().numpy()
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
    return dict(probs=probs, labels=labels, preds=preds, auroc=auc, auprc=auprc,
                accuracy=acc, f1=f1, sensitivity=sens, specificity=spec,
                confusion_matrix=cm, auroc_undefined=auroc_undefined)


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
# Per-variant training, with per-epoch checkpointing and interruption recovery
# ============================================================================
def train_variant(name, tag, kwargs, train_ds, val_ds, out_dir=None, resume=None,
                   epochs=None, patience=None, _stop_after_epoch=None,
                   index_hash=None, split_hash=None):
    """
    `_stop_after_epoch` is a TEST-ONLY hook (used by run_sanity_check()'s resume
    test): if set, training returns immediately after that epoch's checkpoint
    is saved, WITHOUT finalizing results -- simulating a process kill at the
    worst realistic moment (right after a checkpoint write). Leave it None
    (default) for every real run.

    `index_hash`/`split_hash` (item 1, strict configuration validation): sha256
    fingerprints of the dataset file and the exact train/val subject split,
    computed once in main() and passed to every variant. Combined with
    `kwargs`/epochs/patience/batch_size/lr/val_split/seed into a single
    config fingerprint that is compared against the fingerprint stored in
    any existing checkpoint/result before it is reused -- see
    check_fingerprint_or_raise().
    """
    out_dir = OUT_DIR if out_dir is None else out_dir
    resume = RESUME if resume is None else resume
    epochs = EPOCHS if epochs is None else epochs
    patience = PATIENCE if patience is None else patience
    paths = checkpoint_paths(tag, out_dir)
    current_fp = compute_config_fingerprint(kwargs, index_hash, split_hash, epochs, patience)

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

    model = ADHDProposedModelV2(**kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    # Sentinel is -1.0, not 0.0: a real AUROC can legitimately equal exactly 0.0 on an
    # early/degenerate epoch, and the selection check below is `ev['auroc'] > best['auroc']`
    # -- with a 0.0 sentinel, 0.0 > 0.0 is False, so NO checkpoint would ever be saved and
    # `best_state` would stay None for the whole run, silently falling back to whatever the
    # LAST epoch's weights happen to be, with placeholder (zero) metrics in the saved
    # results. -1.0 is lower than any real AUROC (always in [0, 1], including the
    # documented AUROC-undefined fallback of 0.5 -- see evaluate_full()), so the FIRST
    # valid validation epoch is always accepted as the initial checkpoint (item 2).
    best = dict(auroc=-1.0, auprc=0.0, accuracy=0.0, f1=0.0, sensitivity=0.0, specificity=0.0, epoch=0)
    best_state = None
    patience_ctr = 0
    history = []
    resumed = False

    if resume and os.path.isfile(paths['latest']):
        # NOTE: config-fingerprint mismatches must ABORT the run (RuntimeError), never be
        # treated as "checkpoint failed to load, start fresh" -- so the fingerprint check
        # happens in its OWN try block, deliberately not caught by the broad except below
        # that is reserved for genuine load/corruption failures (item 1: never silently
        # reuse incompatible results).
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
        model = ADHDProposedModelV2(**kwargs).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        train_gen.manual_seed(SEED)

    n_params = sum(p.numel() for p in model.parameters())
    t_train_start = time.time()
    auroc_was_undefined_any_epoch = any(h.get('auroc_undefined') for h in history)
    epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for vol, pheno, label in train_loader:
            vol, label = vol.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(vol), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        ev = evaluate_full(model, val_loader)
        if ev['auroc_undefined']:
            auroc_was_undefined_any_epoch = True
        history.append(dict(
            epoch=epoch, train_loss=epoch_loss / max(n_batches, 1),
            val_auroc=ev['auroc'], val_auprc=ev['auprc'], val_accuracy=ev['accuracy'],
            val_f1=ev['f1'], val_sensitivity=ev['sensitivity'], val_specificity=ev['specificity'],
            auroc_undefined=ev['auroc_undefined'],
        ))

        if ev['auroc'] > best['auroc']:
            best = dict(auroc=ev['auroc'], auprc=ev['auprc'], accuracy=ev['accuracy'], f1=ev['f1'],
                         sensitivity=ev['sensitivity'], specificity=ev['specificity'], epoch=epoch)
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
            history=history, seed=SEED, kwargs=kwargs, config_fingerprint=current_fp,
        )
        atomic_save(checkpoint, paths['latest'])

        if _stop_after_epoch is not None and epoch >= _stop_after_epoch:
            print(f"  [{tag}] ** SIMULATED INTERRUPTION ** after epoch {epoch} "
                  f"(test-only _stop_after_epoch hook) -- returning without finalizing.")
            return None, 'interrupted'

        if patience_ctr >= patience:
            break

    train_time_s = time.time() - t_train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    flops, lat_mean, lat_std = measure_flops_and_latency(model)
    final_ev = evaluate_full(model, val_loader)   # predictions from the BEST checkpoint, for saving
    atomic_to_csv(build_val_predictions_df(val_ds, final_ev), paths['valpred'])
    atomic_to_csv(pd.DataFrame(history), paths['history'])

    result = {
        'variant': name, 'label': tag, 'params': n_params,
        'auroc': best['auroc'], 'auprc': best['auprc'], 'accuracy': best['accuracy'],
        'f1': best['f1'], 'sensitivity': best['sensitivity'], 'specificity': best['specificity'],
        'flops': flops, 'inference_ms_mean': lat_mean, 'inference_ms_std': lat_std,
        'train_time_s': round(train_time_s, 1), 'epochs_run': epoch, 'best_epoch': best['epoch'],
        'seed': SEED, 'auroc_was_undefined_any_epoch': auroc_was_undefined_any_epoch,
    }
    atomic_to_csv(pd.DataFrame([result]), paths['results'])

    config = {
        'variant': name, 'label': tag, 'kwargs': kwargs,
        'epochs_max': epochs, 'epochs_run': epoch, 'best_epoch': best['epoch'],
        'batch_size': BATCH_SIZE, 'lr': LR, 'val_split': VAL_SPLIT, 'patience': patience,
        'seed': SEED, 'index_file': INDEX_FILE, 'out_dir': out_dir,
        'config_fingerprint': current_fp,
    }
    with open(paths['config'], 'w') as f:
        json.dump(config, f, indent=2, default=str)

    if auroc_was_undefined_any_epoch:
        print(f"  [{tag}] WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch -- "
              f"see {paths['history']}'s auroc_undefined column.")

    return result, ('resumed' if resumed else 'fresh')


# ============================================================================
# Synthetic sanity tests -- NOT the main execution path. No real dataset is
# touched; `main()` calls this once, first, before loading real data.
# ============================================================================
class _SyntheticDS(Dataset):
    """Minimal in-memory stand-in for ADHDDataset, used ONLY by run_sanity_check()."""
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
    print(f"\n{'='*70}\n  PHASE 2 STEP 1 -- CODE-LEVEL SANITY TESTS (synthetic data only)\n{'='*70}")
    torch.manual_seed(0)
    vol = torch.randn(4, 1, 64, 64, 64)

    # ---- Test A: forward pass + parameter counts for all 4 variants (unchanged architecture) ----
    print("\n  Test A -- forward pass + parameter counts, all 4 variants:")
    expected_params = {'A3': 105424, 'A4': 113968, 'A4-new': 113937, 'A5-new': 114482}
    for name, tag, kwargs in VARIANTS:
        m = ADHDProposedModelV2(**kwargs)
        out = m(vol)
        assert out.shape == (4, 2), f"{tag}: unexpected output shape {tuple(out.shape)}"
        n_params = sum(p.numel() for p in m.parameters())
        print(f"    {name:<34} params={n_params:>8,}  output={tuple(out.shape)}  "
              f"(expected {expected_params[tag]:,}: {'MATCH' if n_params == expected_params[tag] else 'MISMATCH!'})")
        assert n_params == expected_params[tag], \
            f"{tag} parameter count changed ({n_params:,} vs expected {expected_params[tag]:,}) -- " \
            f"architecture must NOT change between runs of this script!"
    print("  TEST A PASSED")

    # ---- Test B: checkpoint save/load round-trip ----
    print("\n  Test B -- checkpoint save/load round-trip:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_ckpt_test_') as tmp:
        m1 = ADHDProposedModelV2(trm_type='pairwise', use_attention=True)
        path = os.path.join(tmp, 'test_model.pt')
        atomic_save(m1.state_dict(), path)
        assert os.path.isfile(path) and not os.path.isfile(path + '.tmp'), \
            "atomic_save left a .tmp file behind or didn't write the final file!"
        m2 = ADHDProposedModelV2(trm_type='pairwise', use_attention=True)
        m2.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        all_equal = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
        print(f"    Saved {len(sd1)} tensors, reloaded into a fresh instance, all equal: {all_equal}")
        assert all_equal, "checkpoint round-trip did not reproduce the exact weights!"
    print("  TEST B PASSED")

    # ---- Test C: genuine interrupt-then-resume, compared against an uninterrupted reference ----
    print("\n  Test C -- interrupt-then-resume reproduces the uninterrupted training-loss sequence:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_resume_test_') as tmp:
        train_ds = _SyntheticDS(16, seed=1)
        val_ds = _SyntheticDS(8, seed=2)
        kwargs = dict(trm_type='pairwise', use_attention=True)
        TEST_EPOCHS = 4

        set_seed(SEED)
        ref_result, ref_status = train_variant('Synthetic (ref)', 'synref', kwargs, train_ds, val_ds,
                                                 out_dir=tmp, resume=False, epochs=TEST_EPOCHS, patience=100)
        ref_history = pd.read_csv(f'{tmp}/synref_training_history.csv')
        print(f"    Uninterrupted reference run: {len(ref_history)} epochs, "
              f"status={ref_status}, final train_loss={ref_history['train_loss'].iloc[-1]:.6f}")
        assert len(ref_history) == TEST_EPOCHS

        set_seed(SEED)
        _, int_status = train_variant('Synthetic (interrupted)', 'synint', kwargs, train_ds, val_ds,
                                       out_dir=tmp, resume=False, epochs=TEST_EPOCHS, patience=100,
                                       _stop_after_epoch=2)
        assert int_status == 'interrupted'
        assert os.path.isfile(f'{tmp}/synint_latest_checkpoint.pt'), "no latest checkpoint after simulated interruption!"
        assert not os.path.isfile(f'{tmp}/synint_results.csv'), "results.csv should NOT exist for an interrupted run!"
        print(f"    Simulated interruption after epoch 2: latest checkpoint exists, results.csv does not (correct)")

        resumed_result, resumed_status = train_variant('Synthetic (resumed)', 'synint', kwargs, train_ds, val_ds,
                                                          out_dir=tmp, resume=True, epochs=TEST_EPOCHS, patience=100)
        assert resumed_status == 'resumed'
        resumed_history = pd.read_csv(f'{tmp}/synint_training_history.csv')
        print(f"    Resumed run: status={resumed_status}, {len(resumed_history)} epochs total "
              f"(2 before interruption + 2 after resuming), final train_loss={resumed_history['train_loss'].iloc[-1]:.6f}")
        assert len(resumed_history) == TEST_EPOCHS, \
            f"resumed run has {len(resumed_history)} epochs of history, expected {TEST_EPOCHS} -- " \
            f"resume did not continue from the interruption point correctly!"

        loss_diff = (ref_history['train_loss'].values - resumed_history['train_loss'].values)
        max_abs_diff = np.abs(loss_diff).max()
        print(f"    Per-epoch train_loss, reference vs resumed: max abs diff = {max_abs_diff:.3e}")
        print(f"      reference epochs 1-4 : {ref_history['train_loss'].values.round(6).tolist()}")
        print(f"      resumed   epochs 1-4 : {resumed_history['train_loss'].values.round(6).tolist()}")
        assert max_abs_diff < 1e-5, (
            f"resumed training-loss sequence diverges from the uninterrupted reference "
            f"(max abs diff {max_abs_diff:.3e}) -- RNG/DataLoader-generator state was not "
            f"restored correctly on resume!")
        print("    Resumed sequence matches the uninterrupted reference to within 1e-5 "
              "(CPU, NUM_WORKERS=0 -- exact reproducibility is expected here; a CUDA run "
              "may show tiny cuDNN-nondeterminism drift, which is not a checkpointing bug).")

        final_params = sum(p.numel() for p in ADHDProposedModelV2(**kwargs).parameters())
        print(f"    Resumed variant's architecture unchanged: params={resumed_result['params']:,} "
              f"(expect 114,482: {'MATCH' if resumed_result['params'] == 114482 else 'MISMATCH!'})")
        assert resumed_result['params'] == 114482
    print("  TEST C PASSED")

    # ---- Test D -- strict configuration validation (item 1) ----
    print("\n  Test D -- strict configuration validation refuses to reuse a changed configuration:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_fingerprint_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=20)
        val_ds = _SyntheticDS(4, seed=21)
        kwargs = dict(trm_type='pairwise', use_attention=True)

        result, status = train_variant('FP test', 'synfp', kwargs, train_ds, val_ds,
                                         out_dir=tmp, resume=False, epochs=2, patience=100,
                                         index_hash='hashA', split_hash='splitA')
        assert status == 'fresh'
        assert os.path.isfile(f'{tmp}/synfp_results.csv')

        raised = False
        try:
            train_variant('FP test', 'synfp', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=2, patience=100,
                           index_hash='hashB_DIFFERENT', split_hash='splitA')
        except RuntimeError as e:
            raised = True
            print(f"    Changed dataset (index_hash) correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised, "a changed index_hash was NOT detected/refused when skipping completed results!"

        result2, status2 = train_variant('FP test', 'synfp', kwargs, train_ds, val_ds,
                                           out_dir=tmp, resume=True, epochs=2, patience=100,
                                           index_hash='hashA', split_hash='splitA')
        assert status2 == 'skipped'
        print("    Identical configuration correctly reuses the completed result (status=skipped)")

        _, int_status = train_variant('FP test 2', 'synfp2', kwargs, train_ds, val_ds,
                                       out_dir=tmp, resume=False, epochs=4, patience=100,
                                       index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status == 'interrupted'
        raised2 = False
        try:
            train_variant('FP test 2', 'synfp2', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=4, patience=100,
                           index_hash='hashA', split_hash='splitB_DIFFERENT')
        except RuntimeError as e:
            raised2 = True
            print(f"    Changed split (split_hash) correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised2, "a changed split_hash was NOT detected/refused when resuming from a checkpoint!"

        # A results.csv/config.json with NO config_fingerprint at all (predates the
        # safeguard, or a different script version) must be refused for skip too --
        # never silently treated as "nothing to compare, go ahead".
        _, status3 = train_variant('FP test 3', 'synfp3', kwargs, train_ds, val_ds,
                                    out_dir=tmp, resume=False, epochs=2, patience=100,
                                    index_hash='hashA', split_hash='splitA')
        assert status3 == 'fresh'
        cfg_path = f'{tmp}/synfp3_config.json'
        cfg = json.load(open(cfg_path))
        del cfg['config_fingerprint']
        json.dump(cfg, open(cfg_path, 'w'))
        raised3 = False
        try:
            train_variant('FP test 3', 'synfp3', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=2, patience=100,
                           index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised3 = True
            print(f"    Missing config_fingerprint correctly REFUSED on skip: {str(e).splitlines()[0]}")
        assert raised3, "a missing config_fingerprint was NOT refused when skipping completed results!"

        # Same check on the RESUME path: a checkpoint saved with no config_fingerprint
        # (e.g. produced by a pre-safeguard script version) must also be refused.
        _, int_status2 = train_variant('FP test 4', 'synfp4', kwargs, train_ds, val_ds,
                                        out_dir=tmp, resume=False, epochs=4, patience=100,
                                        index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status2 == 'interrupted'
        ckpt_path = f'{tmp}/synfp4_latest_checkpoint.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        del ckpt['config_fingerprint']
        torch.save(ckpt, ckpt_path)
        raised4 = False
        try:
            train_variant('FP test 4', 'synfp4', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=4, patience=100,
                           index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised4 = True
            print(f"    Missing config_fingerprint correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised4, "a missing config_fingerprint was NOT refused when resuming from a checkpoint!"
    print("  TEST D PASSED")

    # ---- Test E -- first valid checkpoint saved even when AUROC equals exactly 0.0 (item 2) ----
    print("\n  Test E -- first-epoch checkpoint is saved even when AUROC is exactly 0.0:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_auroc0_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=30)
        val_ds = _SyntheticDS(4, seed=31)
        kwargs = dict(trm_type='pairwise', use_attention=True)

        global evaluate_full
        original_evaluate_full = evaluate_full
        call_count = {'n': 0}

        def _forced_zero_evaluate_full(model, loader):
            ev = original_evaluate_full(model, loader)
            call_count['n'] += 1
            if call_count['n'] == 1:
                ev = dict(ev)
                ev['auroc'] = 0.0   # force exactly zero on the very first validation epoch
            return ev

        evaluate_full = _forced_zero_evaluate_full
        try:
            result, status = train_variant('AUROC0 test', 'synauroc0', kwargs, train_ds, val_ds,
                                            out_dir=tmp, resume=False, epochs=1, patience=100)
        finally:
            evaluate_full = original_evaluate_full

        best_ckpt_exists = os.path.isfile(f'{tmp}/synauroc0_best_model.pt')
        print(f"    Forced epoch-1 AUROC=0.0 -> best checkpoint saved: {best_ckpt_exists}")
        assert best_ckpt_exists, \
            "no best checkpoint was saved even though epoch 1 completed -- sentinel bug not fixed!"
        print(f"    Saved best_epoch = {result['best_epoch']}  (expect 1, not left at the 0 placeholder)")
        assert result['best_epoch'] == 1
    print("  TEST E PASSED")

    print(f"\n{'='*70}\n  ALL CODE-LEVEL SANITY TESTS (A-E) PASSED ✓\n{'='*70}")
    print("  (Synthetic data only -- no real dataset was loaded, no real checkpoints were "
          "left behind (temp dirs auto-cleaned). The real 4-variant training run starts next.)")


# ============================================================================
# Dataset integrity (same checks as phase2_consistency.py's validate_dataset_integrity)
# ============================================================================
def validate_dataset_integrity(raw_df, train_ids, val_ids):
    print(f"\n{'='*70}\n  DATASET INTEGRITY CHECKS\n{'='*70}")
    dup_mask = raw_df['subject_id'].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    print(f"  Duplicate subject_id rows: {n_dup}")
    assert n_dup == 0, f"{n_dup} duplicate subject_id rows found in {INDEX_FILE}."

    missing = [p for p in raw_df['npy_path'] if not os.path.isfile(p)]
    print(f"  Missing .npy files: {len(missing)} / {len(raw_df)}")
    assert not missing, f"{len(missing)} .npy files referenced in {INDEX_FILE} do not exist on disk."

    overlap = set(train_ids) & set(val_ids)
    print(f"  Train/val subject overlap: {len(overlap)}")
    assert not overlap, f"{len(overlap)} subjects appear in BOTH train and val -- subject-level leakage!"
    print("  Subject-level independence confirmed: every subject is in exactly one of train/val.")

    val_labels = raw_df[raw_df['subject_id'].isin(val_ids)]['label']
    val_counts = val_labels.value_counts().to_dict()
    print(f"  Validation label distribution: {val_counts}")
    assert val_labels.nunique() == 2, f"validation split contains only one class ({val_counts})."

    print(f"  Train subjects: {len(train_ids)}   Val subjects: {len(val_ids)}   Total: {len(raw_df)}")
    print("  ALL DATASET INTEGRITY CHECKS PASSED ✓")


def save_site_label_counts(raw_df, train_ids, val_ids, out_dir):
    """Per-site label-count breakdown for both splits -- same helper as phase2_consistency.py."""
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
    run_sanity_check()

    print(f"\n{'='*70}\n  LOADING REAL DATASET\n{'='*70}")
    print(f"  INDEX_FILE = {INDEX_FILE}")
    raw_df = pd.read_csv(INDEX_FILE)
    raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
    labels_all = raw_df['label'].values
    print(f"  Total subjects : {len(raw_df)}")

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

    train_stats_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()
    train_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=True, fit_stats=fit_stats)
    val_ds = ADHDDataset(INDEX_FILE, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
    print(f"  Train : {len(train_ds)}   Val : {len(val_ds)}")

    for name, tag, kwargs in VARIANTS:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        result, status = train_variant(name, tag, kwargs, train_ds, val_ds,
                                        index_hash=index_hash, split_hash=split_hash)
        if status == 'skipped':
            print(f"  [skipped -- already complete]  AUROC={result['auroc']:.4f}  "
                  f"params={int(result['params']):,}")
        else:
            print(f"  [{status}]  AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}  "
                  f"Acc={result['accuracy']:.4f}  F1={result['f1']:.4f}  "
                  f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}")
            print(f"  params={result['params']:,}  GFLOPs={result['flops']/1e9:.4f}  "
                  f"inference={result['inference_ms_mean']:.2f}+/-{result['inference_ms_std']:.2f}ms  "
                  f"(best epoch {result['best_epoch']}/{result['epochs_run']}, {result['train_time_s']}s)")

    # ---- Aggregate table built from each variant's OWN saved results.csv (resume-safe) ----
    all_results = []
    for name, tag, kwargs in VARIANTS:
        rpath = f"{OUT_DIR}/{tag}_results.csv"
        assert os.path.isfile(rpath), f"{tag}_results.csv missing after train_variant() -- should not happen"
        all_results.append(pd.read_csv(rpath).iloc[0].to_dict())

    results_df = pd.DataFrame(all_results)[
        ['variant', 'label', 'params', 'flops', 'auroc', 'auprc', 'accuracy', 'f1',
         'sensitivity', 'specificity', 'inference_ms_mean', 'inference_ms_std',
         'train_time_s', 'epochs_run', 'best_epoch', 'seed', 'auroc_was_undefined_any_epoch']
    ]
    atomic_to_csv(results_df, f'{OUT_DIR}/trm_ablation_results.csv')

    print(f"\n{'='*90}")
    print("  PHASE 2 -- TRM ABLATION (A3 / A4 / A4-new / A5-new)")
    print(f"{'='*90}")
    print(f"  {'Variant':<22}{'Params':>9}{'GFLOPs':>9}{'AUROC':>8}{'AUPRC':>8}{'Acc':>8}{'F1':>8}")
    for _, r in results_df.iterrows():
        print(f"  {r['variant']:<22}{int(r['params']):>9,}{r['flops']/1e9:>9.4f}"
              f"{r['auroc']:>8.4f}{r['auprc']:>8.4f}{r['accuracy']*100:>7.2f}%{r['f1']:>8.4f}")
    print("\n  NOTE: all metrics above are from the single held-out VALIDATION split -- not an "
          "independent test set (none exists in this pipeline). These four variants share the "
          "SAME augmentation policy (dataset.py's augment=True), so the A3-vs-A4-vs-A4-new-vs-"
          "A5-new comparison above is valid. Do NOT cross-compare these numbers against Step "
          "2's M0/M1 (phase2_consistency.py) -- those are trained with augment=False (a "
          "deliberate, documented difference; see that file's AUGMENTATION-COMPARISON NOTE). "
          "Multi-fold CV / cross-site evaluation are a separate, not-yet-approved future stage.")

    # ── Figure: AUROC vs params, A3->A4->A4-new->A5-new ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#888888', '#B2182B', '#2166AC', '#2166AC']
    bars = ax.bar(results_df['label'], results_df['auroc'], color=colors, width=0.6)
    for bar, auc in zip(bars, results_df['auroc']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Validation AUROC')
    ax.set_title('Phase 2 -- TRM Ablation: MHA vs. Pairwise Relationship Module')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_trm_ablation_auroc.png', dpi=300)
    plt.close(fig)

    print(f"\nResults -> {OUT_DIR}/trm_ablation_results.csv")
    print(f"Figure  -> {OUT_DIR}/fig_trm_ablation_auroc.png")
    a4 = results_df[results_df['label'] == 'A4']['params'].values[0]
    a4new = results_df[results_df['label'] == 'A4-new']['params'].values[0]
    print(f"\nNote: A4 vs A4-new have nearly identical parameter counts "
          f"({int(a4):,} vs {int(a4new):,}), so any AUROC difference reflects the "
          f"relationship MECHANISM, not added capacity.")

    if results_df['auroc_was_undefined_any_epoch'].any():
        flagged = results_df[results_df['auroc_was_undefined_any_epoch']]['label'].tolist()
        print(f"\n*** WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch for: "
              f"{flagged} -- check {{tag}}_training_history.csv's auroc_undefined column. ***")


if __name__ == '__main__':
    main()
