"""
Trains and compares the Baseline ADHDNet against every proposed-architecture
ablation from Section 14 of the spec, on the SAME stratified 80/20 split,
SAME seeding, and SAME leakage-free fit_stats methodology already used by
ablation_study.py and train.py in this notebook.

    Baseline            : existing ADHDNet (unmodified, frozen)
    A1 Encoder+Tokens   : encoder -> 16 tokens -> mean-pool -> classifier (no refinement)
    A2 Tokens+MLP       : + conventional MLP token refinement
    A3 Tokens+FastKAN   : + FastKAN token refinement
    A4 +TRM             : + Token Relationship Module
    A5 Final (MRI-only) : + Anatomical (token-importance) Attention
    A5 + Phenotype      : A5 with late/adaptive phenotype fusion
    A5 + Evidential     : A5 with the Dirichlet evidential head instead of softmax

This answers Section 14's question directly: where does the performance
improvement actually come from (tokens alone? FastKAN? the relationship
module? attention? phenotype? none of the above?).
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ADHDDataset
from model import ADHDNet
from proposed_model import ADHDProposedModel, evidential_loss

# ── Config ─────────────────────────────────────────────────────────────────
INDEX_CSV   = '/kaggle/working/processed_index.csv'
OUT_DIR     = '/kaggle/working'
EPOCHS      = 150     # same max budget as train.py / ablation_study.py
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# (name, short_label, model_factory, head_type, needs_pheno)
VARIANTS = [
    ('Baseline ADHDNet',   'Baseline', lambda: ADHDNet(n_pheno=5, dropout=0.3), 'standard', True),
    ('A1 Encoder+Tokens',  'A1', lambda: ADHDProposedModel(refine='none',    use_trm=False, use_attention=False), 'standard', False),
    ('A2 Tokens+MLP',      'A2', lambda: ADHDProposedModel(refine='mlp',     use_trm=False, use_attention=False), 'standard', False),
    ('A3 Tokens+FastKAN',  'A3', lambda: ADHDProposedModel(refine='fastkan', use_trm=False, use_attention=False), 'standard', False),
    ('A4 +TRM',            'A4', lambda: ADHDProposedModel(refine='fastkan', use_trm=True,  use_attention=False), 'standard', False),
    ('A5 Final (MRI-only)','A5', lambda: ADHDProposedModel(refine='fastkan', use_trm=True,  use_attention=True),  'standard', False),
    ('A5 + Phenotype',     'A5+Phen', lambda: ADHDProposedModel(refine='fastkan', use_trm=True, use_attention=True, use_phenotype=True), 'standard', True),
    ('A5 + Evidential',    'A5+Evid', lambda: ADHDProposedModel(refine='fastkan', use_trm=True, use_attention=True, use_evidential=True), 'evidential', False),
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


set_seed(SEED)

# ── Stratified split (shared across all variants, same as ablation_study.py) ─
raw_df = pd.read_csv(INDEX_CSV)
raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
labels_all = raw_df['label'].values
print(f"Total subjects : {len(raw_df)}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
val_ids   = raw_df.iloc[val_idx]['subject_id'].tolist()

train_stats_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=False)
fit_stats = train_stats_ds.get_fit_stats()

train_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=True,  fit_stats=fit_stats)
val_ds   = ADHDDataset(INDEX_CSV, subject_ids=val_ids,   augment=False, fit_stats=fit_stats)
print(f"Train : {len(train_ds)}   Val : {len(val_ds)}")


def expected_calibration_error(probs, labels, n_bins=10):
    """Simple ECE: |confidence - accuracy| per probability bin, weighted by bin size."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidences = np.where(probs >= 0.5, probs, 1 - probs)
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return ece


def train_variant(model_factory, head_type, needs_pheno):
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=g, worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    set_seed(SEED)
    model = model_factory().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_acc, best_sens, best_spec, best_f1, best_ece = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    best_uncertainty_mean = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for vol, pheno, label in train_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            if needs_pheno:
                out = model(vol, pheno)
            else:
                out = model(vol)
            if head_type == 'evidential':
                loss = evidential_loss(out['alpha'], label, n_classes=2, epoch=epoch)
            else:
                loss = criterion(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        va_probs, va_labels, va_uncert = [], [], []
        with torch.no_grad():
            for vol, pheno, label in val_loader:
                vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
                out = model(vol, pheno) if needs_pheno else model(vol)
                if head_type == 'evidential':
                    probs = out['prob'][:, 1].cpu().numpy()
                    va_uncert.extend(out['uncertainty'].cpu().numpy())
                else:
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                va_probs.extend(probs)
                va_labels.extend(label.numpy())

        va_probs, va_labels = np.array(va_probs), np.array(va_labels)
        va_preds = (va_probs >= 0.5).astype(int)
        try:
            va_auc = roc_auc_score(va_labels, va_probs)
        except Exception:
            va_auc = 0.5
        va_acc = (va_preds == va_labels).mean()

        if va_auc > best_auc:
            best_auc = va_auc
            best_acc = va_acc
            tp = int(((va_preds == 1) & (va_labels == 1)).sum())
            fn = int(((va_preds == 0) & (va_labels == 1)).sum())
            tn = int(((va_preds == 0) & (va_labels == 0)).sum())
            fp = int(((va_preds == 1) & (va_labels == 0)).sum())
            best_sens = tp / (tp + fn) if (tp + fn) else 0.0
            best_spec = tn / (tn + fp) if (tn + fp) else 0.0
            best_f1   = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            best_ece  = expected_calibration_error(va_probs, va_labels)
            best_uncertainty_mean = float(np.mean(va_uncert)) if va_uncert else None
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            break

    return {
        'params': n_params, 'accuracy': best_acc, 'auc': best_auc, 'sensitivity': best_sens,
        'specificity': best_spec, 'f1': best_f1, 'ece': best_ece,
        'mean_uncertainty': best_uncertainty_mean if head_type == 'evidential' else None,
        'epochs_run': epoch,
    }


# ── Run every variant on the same split ───────────────────────────────────────
records = []
for name, short, factory, head_type, needs_pheno in VARIANTS:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    result = train_variant(factory, head_type, needs_pheno)
    result['variant'] = name
    result['label']   = short
    records.append(result)
    ece_str = f"  ECE={result['ece']:.4f}" if head_type != 'evidential' or result['ece'] is not None else ""
    print(f"  Acc={result['accuracy']:.4f}  AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
          f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}{ece_str}  "
          f"({result['params']:,} params, {result['epochs_run']} epochs)")

results_df = pd.DataFrame(records)[
    ['variant', 'label', 'params', 'accuracy', 'auc', 'f1', 'sensitivity',
     'specificity', 'ece', 'mean_uncertainty', 'epochs_run']
]
results_df.to_csv(f'{OUT_DIR}/proposed_ablation_results.csv', index=False)

print(f"\n{'='*80}")
print("  PROPOSED ARCHITECTURE ABLATION TABLE")
print(f"{'='*80}")
print(f"  {'Variant':<22}{'Params':>10}{'Accuracy':>10}{'AUC':>8}{'F1':>8}{'ECE':>8}")
for r in records:
    ece_disp = f"{r['ece']:.4f}" if r['ece'] is not None else "  n/a"
    print(f"  {r['variant']:<22}{r['params']:>10,}{r['accuracy']*100:>9.2f}%{r['auc']:>8.4f}"
          f"{r['f1']:>8.4f}{ece_disp:>8}")

# ── Figure — accuracy across the proposed-architecture ablation chain ────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#888888'] + ['#2166AC'] * 5 + ['#7B9EC7', '#B2182B']
bars = ax.bar(results_df['label'], results_df['accuracy'] * 100, color=colors, width=0.6)
for bar, acc in zip(bars, results_df['accuracy']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Proposed Architecture Ablation Chain')
ax.set_ylim(0, max(results_df['accuracy']) * 100 + 10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig6_proposed_ablation.png', dpi=300)

print(f"\nResults table → {OUT_DIR}/proposed_ablation_results.csv")
print(f"Figure        → {OUT_DIR}/fig6_proposed_ablation.png")
