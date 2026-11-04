
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              confusion_matrix, classification_report,
                              roc_curve)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dataset import ADHDDataset
from model   import ADHDNet

# ── Config ──────────────────────────────────────────────────────────────────
INDEX_CSV      = '/kaggle/working/processed_index.csv'
VAL_CSV        = '/kaggle/working/val_index.csv'
MODEL_PTH      = '/kaggle/working/best_model.pth'
FIT_STATS_JSON = '/kaggle/working/fit_stats.json'
OUT_DIR        = '/kaggle/working'
SEED           = 42
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load val subjects ────────────────────────────────────────────────────────
val_df      = pd.read_csv(VAL_CSV)
val_ids     = val_df['subject_id'].tolist()
print(f"Val subjects : {len(val_ids)}")

# ── Reuse the normalisation stats fit on TRAINING subjects only ─────────────
# (saved by train.py — recomputing from the full index here would leak
# validation-set age/ADHD-index statistics into the z-scores)
with open(FIT_STATS_JSON) as f:
    fit_stats = json.load(f)

val_ds  = ADHDDataset(INDEX_CSV, subject_ids=val_ids,
                      augment=False, fit_stats=fit_stats)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
print(f"ADHD={int((val_ds.labels==1).sum())}  Control={int((val_ds.labels==0).sum())}")

# ── Load model ───────────────────────────────────────────────────────────────
model = ADHDNet(n_pheno=5, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model.eval()
print(f"Model loaded ✓")

# ── Inference ────────────────────────────────────────────────────────────────
all_probs  = []
all_preds  = []
all_labels = []
all_ids    = val_ds.df['subject_id'].tolist()

with torch.no_grad():
    for vol, pheno, label in val_loader:
        vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
        logits = model(vol, pheno)
        probs  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(label.numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ──────────────────────────────────────────────────────────────────
acc  = accuracy_score(all_labels, all_preds)
auc  = roc_auc_score(all_labels, all_probs)
cm   = confusion_matrix(all_labels, all_preds)

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)          # recall for ADHD class
specificity = tn / (tn + fp)          # recall for Control class
ppv         = tp / (tp + fp)          # precision for ADHD
npv         = tn / (tn + fn)          # precision for Control

print(f"\n{'='*50}")
print(f"  EVALUATION RESULTS — ADHDNET")
print(f"{'='*50}")
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  AUC         : {auc:.4f}")
print(f"  Sensitivity : {sensitivity:.4f}  (ADHD recall)")
print(f"  Specificity : {specificity:.4f}  (Control recall)")
print(f"  PPV         : {ppv:.4f}  (ADHD precision)")
print(f"  NPV         : {npv:.4f}  (Control precision)")
print(f"\n  Confusion Matrix:")
print(f"                Pred Control  Pred ADHD")
print(f"  True Control      {tn:>3}           {fp:>3}")
print(f"  True ADHD         {fn:>3}           {tp:>3}")
print(f"\n  Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=['Control','ADHD']))
print(f"{'='*50}")

# ── Save per-subject scores (for multimodal fusion later) ────────────────────
results_df = pd.DataFrame({
    'subject_id' : all_ids,
    'true_label' : all_labels,
    'mri_score'  : all_probs,
    'mri_pred'   : all_preds,
    'mri_label'  : ['ADHD' if p==1 else 'Control' for p in all_preds],
    'true_label_str': ['ADHD' if l==1 else 'Control' for l in all_labels],
})
results_df.to_csv(f'{OUT_DIR}/mri_scores.csv', index=False)
print(f"\nPer-subject scores saved → {OUT_DIR}/mri_scores.csv")

# ── ROC Curve plot ───────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='steelblue', lw=2,
         label=f'ADHDNet (AUC = {auc:.4f})')
plt.plot([0,1],[0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — ADHDNet')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/roc_curve_mri.png', dpi=150)
print(f"ROC curve saved  → {OUT_DIR}/roc_curve_mri.png")

# ── Confusion matrix plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Control','ADHD'])
ax.set_yticklabels(['Control','ADHD'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — ADHDNet')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]),
                ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confusion_matrix_mri.png', dpi=150)
print(f"Confusion matrix → {OUT_DIR}/confusion_matrix_mri.png")