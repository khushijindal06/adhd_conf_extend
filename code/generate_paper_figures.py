
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

# ── Config ─────────────────────────────────────────────────────────────────
OUT_DIR       = '.'
MRI_SCORES    = f'{OUT_DIR}/mri_scores_NONEXISTENT.csv'     # written by evaluate.py, if present

# Fallback summary statistics — used only when mri_scores.csv (the real
# per-subject predictions) isn't available. These are the exact numbers
# reported for the paper; the ROC curve drawn from them is a smooth
# binormal approximation through that AUC and that operating point, NOT a
# curve traced from real per-subject scores. Re-run this script after
# evaluate.py on the actual held-out set to replace it with the exact curve.
#
# !! STALE AFTER THE ECA-PLACEMENT FIX !!
# These AUC/confusion-matrix numbers were measured on the OLD ADHDNet (ECA
# after each of the 3 DS-Conv3D blocks). The architecture now has a single
# ECA after all 3 blocks, so these numbers no longer describe this model.
# Keep them only as a placeholder until you retrain + rerun evaluate.py on
# the corrected architecture, then replace them (or just use mri_scores.csv).
FALLBACK_AUC = 0.9535
FALLBACK_CM  = {'tp': 42, 'fn': 3, 'fp': 8, 'tn': 47}   # Actual ADHD / Actual Control rows

LIGHTWEIGHT_PARAMS = 71_552       # ADHDNet: DS-Conv3D x3 -> single 3D ECA -> phenotype -> gated fusion
STANDARD_PARAMS    = 1_376_080    # ADHDNetEnhanced (Standard Conv3D + ECA + GeM), matches the paper
REDUCTION_PCT      = 100 * (1 - LIGHTWEIGHT_PARAMS / STANDARD_PARAMS)

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
})

using_real_data = os.path.exists(MRI_SCORES)
print(f"Real per-subject scores found at {MRI_SCORES} : {using_real_data}")
if not using_real_data:
    print("WARNING: falling back to hardcoded AUC/confusion-matrix numbers that were "
          "measured on the OLD per-block-ECA ADHDNet. The architecture now uses a single "
          "ECA after all 3 DS-Conv3D blocks, so Figs 2-3 below are PLACEHOLDERS until you "
          "retrain and rerun evaluate.py to produce a real mri_scores.csv.")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — ROC curve
# ══════════════════════════════════════════════════════════════════════════
if using_real_data:
    from sklearn.metrics import roc_curve, roc_auc_score
    df = pd.read_csv(MRI_SCORES)
    fpr, tpr, _ = roc_curve(df['true_label'], df['mri_score'])
    auc = roc_auc_score(df['true_label'], df['mri_score'])
    op_fpr, op_tpr = None, None   # the real curve already passes through the op. point
else:
    auc = FALLBACK_AUC
    cm = FALLBACK_CM
    op_fpr = cm['fp'] / (cm['fp'] + cm['tn'])
    op_tpr = cm['tp'] / (cm['tp'] + cm['fn'])

    # Equal-variance binormal ROC curve calibrated to hit the reported AUC:
    #   TPR(FPR) = Phi(a + Phi^-1(FPR)),  a = sqrt(2) * Phi^-1(AUC)
    a = np.sqrt(2) * norm.ppf(auc)
    fpr = np.linspace(0, 1, 400)
    fpr[0], fpr[-1] = 1e-6, 1 - 1e-6
    tpr = norm.cdf(a + norm.ppf(fpr))
    tpr[0], tpr[-1] = 0.0, 1.0
    fpr[0], fpr[-1] = 0.0, 1.0

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, color='#2166AC', lw=2.5, label=f'ADHDNet (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance (AUC = 0.50)')
if not using_real_data:
    ax.scatter([op_fpr], [op_tpr], color='#B2182B', s=45, zorder=5,
               label=f'Operating point (threshold = 0.5)')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — ADHDNet')
ax.legend(loc='lower right', frameon=False, fontsize=9)
ax.set_aspect('equal')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_roc_curve.png')
print(f"Fig. 2 saved -> {OUT_DIR}/fig2_roc_curve.png  (AUC={auc:.4f}, "
      f"{'real data' if using_real_data else 'binormal approximation from reported summary stats'})")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Confusion matrix (single-color sequential heatmap)
# ══════════════════════════════════════════════════════════════════════════
if using_real_data:
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(df['true_label'], (df['mri_score'] >= 0.5).astype(int)).ravel()
else:
    tp, fn, fp, tn = FALLBACK_CM['tp'], FALLBACK_CM['fn'], FALLBACK_CM['fp'], FALLBACK_CM['tn']

cm = np.array([[tp, fn],
               [fp, tn]])          # rows = Actual [ADHD, Control], cols = Predicted [ADHD, Control]
total = cm.sum()

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted\nADHD', 'Predicted\nControl'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual\nADHD', 'Actual\nControl'])
ax.tick_params(length=0)

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = 100 * val / total
        color = 'white' if val > cm.max() * 0.55 else 'black'
        ax.text(j, i - 0.12, f'{val}', ha='center', va='center',
                 fontsize=20, fontweight='bold', color=color)
        ax.text(j, i + 0.18, f'{pct:.1f}%', ha='center', va='center',
                 fontsize=10, color=color)

ax.set_title(f'Confusion Matrix — ADHDNet (n={total})')
for spine in ax.spines.values():
    spine.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Count')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_confusion_matrix.png')
print(f"Fig. 3 saved -> {OUT_DIR}/fig3_confusion_matrix.png  "
      f"({'real data' if using_real_data else 'reported counts'})")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Parameter efficiency (Option B: compact flow panel)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

box_kw = dict(boxstyle='round,pad=0.5', facecolor='#F2F2F2', edgecolor='#555555', linewidth=1.2)
ax.text(0.5, 0.92, 'Standard Conv3D', ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=box_kw)
ax.text(0.5, 0.80, f'{STANDARD_PARAMS/1e6:.3f} M parameters', ha='center', va='center', fontsize=10)

arrow1 = FancyArrowPatch((0.5, 0.73), (0.5, 0.45), arrowstyle='-|>', mutation_scale=18,
                          color='#B2182B', linewidth=2)
ax.add_patch(arrow1)
ax.text(0.68, 0.59, f'↓ {REDUCTION_PCT:.1f}%\nreduction', ha='left', va='center',
        fontsize=11, color='#B2182B', fontweight='bold')

box_kw2 = dict(boxstyle='round,pad=0.5', facecolor='#DCEBF7', edgecolor='#2166AC', linewidth=1.2)
ax.text(0.5, 0.33, 'ADHDNet', ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=box_kw2)
ax.text(0.5, 0.21, f'{LIGHTWEIGHT_PARAMS/1e6:.4f} M parameters', ha='center', va='center', fontsize=10)

ax.set_title('Parameter Efficiency', fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig4_parameter_efficiency.png')
print(f"Fig. 4 saved -> {OUT_DIR}/fig4_parameter_efficiency.png  "
      f"({STANDARD_PARAMS:,} -> {LIGHTWEIGHT_PARAMS:,} params, {REDUCTION_PCT:.1f}% reduction)")

print("\nAll figures saved to", OUT_DIR)
