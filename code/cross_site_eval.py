"""
Cross-site (leave-one-site-out) generalization evaluation -- a different,
harder generalization test than the random stratified splits used
everywhere else in this notebook (train.py, cross_validate.py,
ablation_study.py, proposed_ablation_study.py all split subjects randomly,
mixing sites between train and val).

Here, for each site S in {KKI, NYU, OHSU, Peking_1, NeuroIMAGE} (Pittsburgh
is already excluded everywhere in this notebook -- it has 0 ADHD subjects,
see dataset.py), the model trains on every OTHER site and is evaluated on
S entirely held out. This tests whether the model generalizes across
scanner/protocol differences, not just across subjects from sites it has
already seen -- directly answering the "cross-site evaluation" requirement.

Uses the exact same leakage-free fit_stats pattern (fit on the training
sites only, applied to the held-out site) and the same seeding as every
other script in this notebook. Runs for both the Baseline ADHDNet and the
final proposed model (A5), so the two can be compared on robustness, not
just in-distribution accuracy.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ADHDDataset
from model import ADHDNet
from proposed_model import ADHDProposedModel

# ── Config ─────────────────────────────────────────────────────────────────
INDEX_CSV   = '/kaggle/working/processed_index.csv'
OUT_DIR     = '/kaggle/working'
EPOCHS      = 100     # per-site training budget; cross-site has more folds than
                       # cross_validate.py's 5, so a slightly smaller budget keeps
                       # total runtime comparable -- document this if you raise it
BATCH_SIZE  = 8
LR          = 1e-4
PATIENCE    = 12
SEED        = 42
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

MODEL_REGISTRY = {
    'Baseline ADHDNet': (lambda: ADHDNet(n_pheno=5, dropout=0.3), True),
    'A5 Proposed (MRI-only)': (lambda: ADHDProposedModel(refine='fastkan', use_trm=True, use_attention=True), False),
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

raw_df = pd.read_csv(INDEX_CSV)
raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
sites = sorted(raw_df['site'].unique())
print(f"Sites (Pittsburgh excluded, 0 ADHD subjects) : {sites}")
for s in sites:
    sub = raw_df[raw_df['site'] == s]
    print(f"  {s:<12} n={len(sub):>3}  ADHD={int((sub['label']==1).sum()):>3}  Control={int((sub['label']==0).sum()):>3}")


def run_fold(model_cls, needs_pheno, train_ids, test_ids):
    train_stats_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()

    train_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=True,  fit_stats=fit_stats)
    test_ds  = ADHDDataset(INDEX_CSV, subject_ids=test_ids,  augment=False, fit_stats=fit_stats)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=g, worker_init_fn=worker_init_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    set_seed(SEED)
    model = model_cls().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # held out site has no internal val split to early-stop on without more
    # leakage, so we carve a small stratified slice of the TRAINING sites
    # for early stopping, and report final metrics on the fully held-out site.
    from sklearn.model_selection import StratifiedShuffleSplit
    tr_labels = train_ds.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    inner_train_idx, inner_val_idx = next(sss.split(np.zeros(len(tr_labels)), tr_labels))

    from torch.utils.data import Subset
    inner_val_loader = DataLoader(Subset(train_ds, inner_val_idx), batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)
    train_loader = DataLoader(Subset(train_ds, inner_train_idx), batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=g, worker_init_fn=worker_init_fn)

    best_inner_auc = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for vol, pheno, label in train_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(vol, pheno) if needs_pheno else model(vol)
            loss = criterion(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        iv_probs, iv_labels = [], []
        with torch.no_grad():
            for vol, pheno, label in inner_val_loader:
                vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
                out = model(vol, pheno) if needs_pheno else model(vol)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                iv_probs.extend(probs)
                iv_labels.extend(label.numpy())
        try:
            inner_auc = roc_auc_score(iv_labels, iv_probs)
        except Exception:
            inner_auc = 0.5

        if inner_auc > best_inner_auc:
            best_inner_auc = inner_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── final evaluation on the fully held-out site ───────────────────────
    model.eval()
    te_probs, te_labels = [], []
    with torch.no_grad():
        for vol, pheno, label in test_loader:
            vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
            out = model(vol, pheno) if needs_pheno else model(vol)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            te_probs.extend(probs)
            te_labels.extend(label.numpy())
    te_probs, te_labels = np.array(te_probs), np.array(te_labels)
    te_preds = (te_probs >= 0.5).astype(int)
    try:
        te_auc = roc_auc_score(te_labels, te_probs)
    except Exception:
        te_auc = 0.5
    te_acc = (te_preds == te_labels).mean()

    return {'held_out_auc': te_auc, 'held_out_acc': te_acc, 'n_test': len(te_labels), 'epochs_run': epoch}


records = []
for model_name, (model_cls, needs_pheno) in MODEL_REGISTRY.items():
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    for held_out_site in sites:
        test_ids  = raw_df[raw_df['site'] == held_out_site]['subject_id'].tolist()
        train_ids = raw_df[raw_df['site'] != held_out_site]['subject_id'].tolist()
        result = run_fold(model_cls, needs_pheno, train_ids, test_ids)
        result['model'] = model_name
        result['held_out_site'] = held_out_site
        records.append(result)
        print(f"  Held out {held_out_site:<12} (n={result['n_test']:>3}): "
              f"AUC={result['held_out_auc']:.4f}  Acc={result['held_out_acc']:.4f}  "
              f"({result['epochs_run']} epochs)")

results_df = pd.DataFrame(records)[
    ['model', 'held_out_site', 'n_test', 'held_out_auc', 'held_out_acc', 'epochs_run']
]
results_df.to_csv(f'{OUT_DIR}/cross_site_results.csv', index=False)

print(f"\n{'='*60}\n  CROSS-SITE GENERALIZATION SUMMARY\n{'='*60}")
summary = results_df.groupby('model').agg(
    auc_mean=('held_out_auc', 'mean'), auc_std=('held_out_auc', 'std'),
    acc_mean=('held_out_acc', 'mean'), acc_std=('held_out_acc', 'std'),
)
print(summary.to_string())
summary.to_csv(f'{OUT_DIR}/cross_site_summary.csv')

# ── Figure: per-site AUC, both models ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(sites))
width = 0.35
for i, (model_name, _) in enumerate(MODEL_REGISTRY.items()):
    sub = results_df[results_df['model'] == model_name].set_index('held_out_site').loc[sites]
    ax.bar(x + (i - 0.5) * width, sub['held_out_auc'], width, label=model_name)
ax.set_xticks(x)
ax.set_xticklabels(sites, rotation=30, ha='right')
ax.set_ylabel('Held-out-site AUC')
ax.set_title('Leave-One-Site-Out Cross-Site Generalization')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig7_cross_site.png', dpi=300)

print(f"\nPer-fold results → {OUT_DIR}/cross_site_results.csv")
print(f"Summary          → {OUT_DIR}/cross_site_summary.csv")
print(f"Figure           → {OUT_DIR}/fig7_cross_site.png")
