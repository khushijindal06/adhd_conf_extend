
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from dataset import ADHDDataset
from model   import ADHDNet

# ── Config ─────────────────────────────────────────────────────────────────
INDEX_CSV      = '/kaggle/working/processed_index.csv'
MODEL_PTH      = '/kaggle/working/best_model.pth'
VAL_CSV        = '/kaggle/working/val_index.csv'
FIT_STATS_JSON = '/kaggle/working/fit_stats.json'

EPOCHS      = 150
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
NUM_WORKERS = 0     # 0 while debugging reproducibility; raise for speed once runs match
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")


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

# ── Raw index, used only to build the stratified split ───────────────────────
# (no normalisation stats are computed from this — see below)
raw_df = pd.read_csv(INDEX_CSV)
raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
labels_all = raw_df['label'].values
print(f"Total subjects : {len(raw_df)}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))

train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
val_ids   = raw_df.iloc[val_idx]['subject_id'].tolist()

# ── Normalisation stats fit on TRAINING subjects only ─────────────────────────
# Fitting on the full dataset (train+val) leaks validation-set statistics into
# the age/ADHD-index z-scores. Fit only on train_ids, then apply the same
# stats to the validation subjects.
train_stats_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=False)
fit_stats = train_stats_ds.get_fit_stats()
with open(FIT_STATS_JSON, 'w') as f:
    json.dump(fit_stats, f, indent=2)

train_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=True,  fit_stats=fit_stats)
val_ds   = ADHDDataset(INDEX_CSV, subject_ids=val_ids,   augment=False, fit_stats=fit_stats)

print(f"Train : {len(train_ds)}")
print(f"Val   : {len(val_ds)}")
print(f"Class weights  : {train_ds.class_weights()}")

# ── Dataloaders ────────────────────────────────────────────────────────────────
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           generator=g, worker_init_fn=worker_init_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           worker_init_fn=worker_init_fn)

# ── Loss with label smoothing + class weights (fit on train split only) ───────
cw        = train_ds.class_weights().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

# ── Model / optimiser / scheduler ────────────────────────────────────────────
set_seed(SEED)      # re-seed immediately before model init for deterministic weights
model     = ADHDNet(n_pheno=5, dropout=0.3).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ───────────────────────────────────────────────────────────
best_auc     = 0.0
patience_ctr = 0

print(f"\n{'Ep':>4} | {'TrLoss':>7} | {'TrAUC':>6} | {'VaLoss':>7} | {'VaAUC':>6} | {'VaAcc':>6} |")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):

    # ── train ───────────────────────────────────────────────────────────
    model.train()
    tr_losses, tr_probs, tr_labels = [], [], []

    for vol, pheno, label in train_loader:
        vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        logits = model(vol, pheno)
        loss   = criterion(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        tr_probs.extend(probs)
        tr_labels.extend(label.cpu().numpy())
        tr_losses.append(loss.item())

    scheduler.step()
    tr_loss = np.mean(tr_losses)
    try:
        tr_auc = roc_auc_score(tr_labels, tr_probs)
    except Exception:
        tr_auc = 0.5

    # ── validate ────────────────────────────────────────────────────────
    model.eval()
    va_losses, va_probs, va_labels = [], [], []

    with torch.no_grad():
        for vol, pheno, label in val_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            logits = model(vol, pheno)
            loss   = criterion(logits, label)

            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            va_probs.extend(probs)
            va_labels.extend(label.cpu().numpy())
            va_losses.append(loss.item())

    va_loss = np.mean(va_losses)
    va_preds = (np.array(va_probs) >= 0.5).astype(int)
    va_acc   = (va_preds == np.array(va_labels)).mean()
    try:
        va_auc = roc_auc_score(va_labels, va_probs)
    except Exception:
        va_auc = 0.5

    # ── save best & early stop ──────────────────────────────────────────
    saved = ""
    if va_auc > best_auc:
        best_auc     = va_auc
        patience_ctr = 0
        torch.save(model.state_dict(), MODEL_PTH)
        saved = "✓"
    else:
        patience_ctr += 1

    print(f"{epoch:>4} | {tr_loss:>7.4f} | {tr_auc:>6.4f} | "
          f"{va_loss:>7.4f} | {va_auc:>6.4f} | {va_acc:>6.4f} | {saved}")

    if patience_ctr >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# ── Save val index for evaluate.py ───────────────────────────────────────────
val_index = raw_df[raw_df['subject_id'].isin(val_ids)].reset_index(drop=True)
val_index.to_csv(VAL_CSV, index=False)

print(f"\nDone. Best Val AUC : {best_auc:.4f}")
print(f"Model saved      → {MODEL_PTH}")
print(f"Val index        → {VAL_CSV}")
print(f"Fit stats        → {FIT_STATS_JSON}")
