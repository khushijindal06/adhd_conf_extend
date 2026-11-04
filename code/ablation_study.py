
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
from model import (ADHDNet, ADHDNetNoECA, ADHDNetMRIOnly,
                    ADHDNetConcatFusion, ADHDNetStandardConv)

# ── Config ─────────────────────────────────────────────────────────────────
INDEX_CSV   = '/kaggle/working/processed_index.csv'
OUT_DIR     = '/kaggle/working'
EPOCHS      = 150   # same max training budget as train.py, for a fair ablation
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
NUM_WORKERS = 0     # 0 while debugging reproducibility; raise for speed once runs match
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# Single stratified 80/20 split, same methodology as train.py, used for all
# 5 variants so the comparison is apples-to-apples. (Swap in the 5-fold CV
# harness from cross_validate.py instead if you have the compute budget —
# the per-fold pattern there, including leakage-free fit_stats, is the same.)
#
# A4 is ADHDNetStandardConv: same 3 blocks, same 24-48-96 channels, same
# single post-backbone ECA, same phenotype branch, same gated fusion, same
# classifier — only the conv type changes (standard Conv3d vs DS-Conv3D).
# This is a genuine controlled ablation isolating depthwise-separable
# factorisation specifically. ADHDNetEnhanced (1.376M params, 4 blocks,
# GeM pooling, etc.) is a separate heavy baseline — kept and used in
# cross_validate.py's efficiency comparison, not here.
ABLATIONS = [
    ('A0 Full model',  'Full',   ADHDNet),
    ('A1 w/o ECA',     '-ECA',   ADHDNetNoECA),
    ('A2 w/o phenotype', '-Phen', ADHDNetMRIOnly),
    ('A3 Concat fusion', 'Concat', ADHDNetConcatFusion),
    ('A4 Standard Conv3D', 'StdConv', ADHDNetStandardConv),
]

# Static component flags for the Table 2 ablation table (independent of the
# measured metrics below). Every row other than the one ablated component is
# identical to A0 — including A4, which only swaps the conv type.
ABLATION_META = {
    'A0 Full model':      {'eca': '✓', 'phen': '✓', 'fusion': 'Adaptive gated', 'conv': 'DS-Conv3D'},
    'A1 w/o ECA':         {'eca': '✗', 'phen': '✓', 'fusion': 'Adaptive gated', 'conv': 'DS-Conv3D'},
    'A2 w/o phenotype':   {'eca': '✓', 'phen': '✗', 'fusion': '—',             'conv': 'DS-Conv3D'},
    'A3 Concat fusion':   {'eca': '✓', 'phen': '✓', 'fusion': 'Concatenation', 'conv': 'DS-Conv3D'},
    'A4 Standard Conv3D': {'eca': '✓', 'phen': '✓', 'fusion': 'Adaptive gated', 'conv': 'Standard Conv3D'},
}
DISPLAY_NAME = {
    'A0 Full model':      'Full LiteFusion-3D',
    'A1 w/o ECA':         'w/o ECA',
    'A2 w/o phenotype':   'w/o phenotype',
    'A3 Concat fusion':   'Concat. fusion',
    'A4 Standard Conv3D': 'Standard Conv3D',
}


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

# ── Stratified split (shared across all variants) ────────────────────────────
raw_df = pd.read_csv(INDEX_CSV)
raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
labels_all = raw_df['label'].values
print(f"Total subjects : {len(raw_df)}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
val_ids   = raw_df.iloc[val_idx]['subject_id'].tolist()

# ── Normalisation stats fit on TRAINING subjects only ─────────────────────────
train_stats_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=False)
fit_stats = train_stats_ds.get_fit_stats()

train_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=True,  fit_stats=fit_stats)
val_ds   = ADHDDataset(INDEX_CSV, subject_ids=val_ids,   augment=False, fit_stats=fit_stats)
print(f"Train : {len(train_ds)}   Val : {len(val_ds)}")


def train_variant(model_cls):
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=g, worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               worker_init_fn=worker_init_fn)

    cw        = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    set_seed(SEED)      # deterministic weight init, same seed for every variant
    model     = model_cls(n_pheno=5, dropout=0.3).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_acc, best_sens, best_spec, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for vol, pheno, label in train_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(vol, pheno), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        va_probs, va_labels = [], []
        with torch.no_grad():
            for vol, pheno, label in val_loader:
                vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
                probs = torch.softmax(model(vol, pheno), dim=1)[:, 1].cpu().numpy()
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
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            break

    return {
        'params'      : n_params,
        'accuracy'    : best_acc,
        'auc'         : best_auc,
        'sensitivity' : best_sens,
        'specificity' : best_spec,
        'f1'          : best_f1,
        'epochs_run'  : epoch,
    }


# ── Run every ablation variant on the same split ──────────────────────────────
records = []
for name, short, cls in ABLATIONS:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    result = train_variant(cls)
    result['variant'] = name
    result['label']   = short
    records.append(result)
    print(f"  Acc={result['accuracy']:.4f}  AUC={result['auc']:.4f}  "
          f"F1={result['f1']:.4f}  Sens={result['sensitivity']:.4f}  "
          f"Spec={result['specificity']:.4f}  ({result['params']:,} params, "
          f"{result['epochs_run']} epochs)")

results_df = pd.DataFrame(records)[
    ['variant', 'label', 'params', 'accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 'epochs_run']
]
results_df.to_csv(f'{OUT_DIR}/ablation_results.csv', index=False)

print(f"\n{'='*70}")
print(f"  ABLATION TABLE")
print(f"{'='*70}")
print(f"  {'Variant':<20}{'Params':>10}{'Accuracy':>10}{'AUC':>8}{'F1':>8}")
for r in records:
    print(f"  {r['variant']:<20}{r['params']:>10,}{r['accuracy']*100:>9.2f}%"
          f"{r['auc']:>8.4f}{r['f1']:>8.4f}")

# ── Figure 5 — Accuracy across ablation variants ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#2166AC'] + ['#7B9EC7'] * (len(records) - 1)
bars = ax.bar(results_df['label'], results_df['accuracy'] * 100, color=colors, width=0.6)
for bar, acc in zip(bars, results_df['accuracy']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Ablation Study — ADHDNet')
ax.set_ylim(0, max(results_df['accuracy']) * 100 + 8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig5_ablation_accuracy.png', dpi=300)

# ── Table 2 — component-wise ablation table (markdown, for the paper) ────────
header = ("| Variant | ECA | Phenotypic branch | Fusion strategy | MRI convolution "
          "| Params (M) | Acc. (%) | AUC | F1 |")
sep    = "|---|:-:|:-:|---|---|---:|---:|---:|---:|"
table_lines = [header, sep]
for name, short, cls in ABLATIONS:
    meta   = ABLATION_META[name]
    row    = results_df[results_df['variant'] == name].iloc[0]
    is_full = (name == 'A0 Full model')
    display = f"**{DISPLAY_NAME[name]}**" if is_full else DISPLAY_NAME[name]
    params_str = f"{row['params'] / 1e6:.4f}"
    acc_str    = f"{row['accuracy'] * 100:.2f}"
    auc_str    = f"{row['auc']:.4f}"
    f1_str     = f"{row['f1']:.4f}"
    if is_full:
        params_str, acc_str, auc_str, f1_str = (f"**{params_str}**", f"**{acc_str}**",
                                                   f"**{auc_str}**", f"**{f1_str}**")
    table_lines.append(
        f"| {display} | {meta['eca']} | {meta['phen']} | {meta['fusion']} | {meta['conv']} "
        f"| {params_str} | {acc_str} | {auc_str} | {f1_str} |"
    )
table_md = "\n".join(table_lines)

print(f"\n{'='*70}")
print("  TABLE 2 — COMPONENT-WISE ABLATION (markdown)")
print(f"{'='*70}")
print(table_md)

with open(f'{OUT_DIR}/table2_ablation.md', 'w') as f:
    f.write(table_md + "\n")

print(f"\nResults table → {OUT_DIR}/ablation_results.csv")
print(f"Table 2 (md)  → {OUT_DIR}/table2_ablation.md")
print(f"Figure 5      → {OUT_DIR}/fig5_ablation_accuracy.png")
