
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ADHDDataset
from model   import ADHDNet, ADHDNetEnhanced

# ── Config ─────────────────────────────────────────────────────────────────
INDEX_CSV   = '/kaggle/working/processed_index.csv'
OUT_DIR     = '/kaggle/working'
N_FOLDS     = 5
EPOCHS      = 60
PATIENCE    = 10
BATCH_SIZE  = 8
LR          = 1e-4
SEED        = 42
NUM_WORKERS = 0     # 0 while debugging reproducibility; raise for speed once runs match
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

MODEL_REGISTRY = {
    'lightweight' : ADHDNet,
    'enhanced'    : ADHDNetEnhanced,
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

# ── Raw index, used only to build the stratified folds ───────────────────────
# (no normalisation stats are computed from this — each fold fits its own,
# on that fold's training subjects only — see run_fold)
raw_df = pd.read_csv(INDEX_CSV)
raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
subject_ids = raw_df['subject_id'].values
labels      = raw_df['label'].values
print(f"Total subjects : {len(raw_df)}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def run_fold(model_cls, train_ids, val_ids):
    # ── normalisation stats fit on this fold's TRAINING subjects only ────────
    train_stats_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()

    train_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=True,  fit_stats=fit_stats)
    val_ds   = ADHDDataset(INDEX_CSV, subject_ids=val_ids,   augment=False, fit_stats=fit_stats)

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

    set_seed(SEED)      # re-seed immediately before model init for deterministic weights
    model     = model_cls(n_pheno=5, dropout=0.3).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_acc, best_sens, best_spec, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    patience_ctr = 0
    epoch = 0
    t0 = time.time()

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

    train_time = time.time() - t0
    return {
        'params'       : n_params,
        'auc'          : best_auc,
        'accuracy'     : best_acc,
        'sensitivity'  : best_sens,
        'specificity'  : best_spec,
        'f1'           : best_f1,
        'epochs_run'   : epoch,
        'train_time_s' : round(train_time, 1),
    }


# ── Run 5-fold CV for each model variant ──────────────────────────────────────
records = []
for model_name, model_cls in MODEL_REGISTRY.items():
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels), start=1):
        train_ids = subject_ids[train_idx].tolist()
        val_ids   = subject_ids[val_idx].tolist()

        result = run_fold(model_cls, train_ids, val_ids)
        result['model'] = model_name
        result['fold']  = fold
        records.append(result)
        print(f"  Fold {fold}: AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}  "
              f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}  "
              f"F1={result['f1']:.4f}  ({result['epochs_run']} epochs, "
              f"{result['train_time_s']}s, {result['params']:,} params)")

results_df = pd.DataFrame(records)
results_df.to_csv(f'{OUT_DIR}/cv_results.csv', index=False)

# ── Summary: mean ± std per model ─────────────────────────────────────────────
summary = results_df.groupby('model').agg(
    params    = ('params',      'first'),
    auc_mean  = ('auc',         'mean'),
    auc_std   = ('auc',         'std'),
    acc_mean  = ('accuracy',    'mean'),
    acc_std   = ('accuracy',    'std'),
    sens_mean = ('sensitivity', 'mean'),
    sens_std  = ('sensitivity', 'std'),
    spec_mean = ('specificity', 'mean'),
    spec_std  = ('specificity', 'std'),
    f1_mean   = ('f1',          'mean'),
    f1_std    = ('f1',          'std'),
).reset_index()
summary.to_csv(f'{OUT_DIR}/cv_summary.csv', index=False)

print(f"\n{'='*60}")
print(f"  5-FOLD STRATIFIED CV — Lightweight vs Enhanced")
print(f"{'='*60}")
for model_name in MODEL_REGISTRY:
    sub = results_df[results_df['model'] == model_name]
    print(f"\n{model_name.upper()}  ({int(sub['params'].iloc[0]):,} params)")
    print(f"  {'Metric':<12} {'Mean ± SD':>16}")
    for metric, label in [('accuracy', 'Accuracy'), ('auc', 'AUC'),
                           ('sensitivity', 'Sensitivity'),
                           ('specificity', 'Specificity'), ('f1', 'F1')]:
        m, s = sub[metric].mean(), sub[metric].std()
        print(f"  {label:<12} {m*100:6.2f} ± {s*100:4.2f}%")

# ── Comparison plot: per-fold AUC, Lightweight vs Enhanced ────────────────────
plt.figure(figsize=(7, 5))
for model_name, color in [('lightweight', 'steelblue'), ('enhanced', 'darkorange')]:
    sub = results_df[results_df['model'] == model_name].sort_values('fold')
    plt.plot(sub['fold'], sub['auc'], marker='o', color=color,
             label=f"{model_name} (mean={sub['auc'].mean():.3f})")
plt.xlabel('Fold')
plt.ylabel('Validation AUC')
plt.title('5-Fold CV — Lightweight vs Enhanced')
plt.xticks(range(1, N_FOLDS + 1))
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cv_comparison_auc.png', dpi=150)

print(f"\nComparison plot saved → {OUT_DIR}/cv_comparison_auc.png")
print(f"Per-fold results     → {OUT_DIR}/cv_results.csv")
print(f"Summary              → {OUT_DIR}/cv_summary.csv")
