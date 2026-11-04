"""
Phenotype leakage check -- audit support script (read-only, no training)
===========================================================================

Checks whether any of the baseline ADHDNet's 5 phenotype input features
(age, sex, med_status, site_id, adhd_index) trivially separates the
diagnostic label on their own. If one does, the original baseline's
validation AUROC (reported at 0.9516) may be substantially inflated by
quasi-label-leakage through the phenotype branch rather than genuine
MRI-based signal -- see the audit report for why this matters before
comparing that number to the MRI-only Phase-2 models.

This is READ-ONLY: it only reads processed_index.csv and prints results.
It does not train anything, does not touch any checkpoint/CSV/figure, and
does not modify processed_index.csv.

Usage (on Kaggle, or anywhere processed_index.csv is accessible):
    python check_phenotype_leakage.py
or edit INDEX_CSV below if your file isn't at the default path.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

INDEX_CSV = '/kaggle/working/processed_index.csv'

PHENO_COLS = ['age', 'sex', 'med_status', 'site_id', 'adhd_index']
MISSING_SENTINEL = -999   # dataset.py's own convention: df.replace(-999, np.nan)


def univariate_auroc(df, col):
    """
    AUROC of the raw feature alone as a predictor of `label`, with the
    dataset's own -999 missing-value sentinel excluded first (left in,
    it's an arbitrary large negative number that would distort the AUROC
    in either direction depending on which class has more missingness --
    not what we want to measure here).
    """
    s = pd.to_numeric(df[col], errors='coerce')
    s = s.replace(MISSING_SENTINEL, np.nan)
    mask = s.notna()
    n_missing = int((~mask).sum())
    if mask.sum() < 2 or df.loc[mask, 'label'].nunique() < 2:
        return None, n_missing, int(mask.sum())
    auc = roc_auc_score(df.loc[mask, 'label'], s[mask])
    return float(auc), n_missing, int(mask.sum())


def main():
    print(f"Loading {INDEX_CSV} ...")
    df = pd.read_csv(INDEX_CSV)
    print(f"Total subjects (before any site filtering): {len(df)}")
    if 'site' in df.columns:
        n_pitt = (df['site'] == 'Pittsburgh').sum()
        if n_pitt:
            print(f"  (note: {n_pitt} Pittsburgh subjects present here -- every training "
                  f"script in this package excludes them via `site != 'Pittsburgh'` before "
                  f"splitting; left IN below so this check sees the full raw file. Re-run "
                  f"with that filter applied first if you want the exact subset actually "
                  f"trained on.)")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}\n")

    print(f"{'='*72}\n  UNIVARIATE AUROC -- each phenotype feature alone vs label\n{'='*72}")
    print(f"{'feature':<12}{'AUROC':>8}{'  n_used':>10}{'  n_missing(-999)':>18}   interpretation")
    print('-' * 72)
    for col in PHENO_COLS:
        if col not in df.columns:
            print(f"{col:<12}  column not found in {INDEX_CSV} -- skipped")
            continue
        auc, n_missing, n_used = univariate_auroc(df, col)
        if auc is None:
            print(f"{col:<12}  skipped (fewer than 2 classes present after dropping missing)")
            continue
        if auc > 0.90 or auc < 0.10:
            note = "<<< NEAR-PERFECT SEPARATION -- likely label leakage"
        elif auc > 0.75 or auc < 0.25:
            note = "<<< strong association -- investigate"
        else:
            note = "weak/no association on its own"
        print(f"{col:<12}{auc:>8.4f}{n_used:>10}{n_missing:>18}   {note}")

    print(f"\n{'='*72}\n  med_status BREAKDOWN BY LABEL (checks for a fixed control sentinel)\n{'='*72}")
    if 'med_status' in df.columns:
        breakdown = df.groupby('label')['med_status'].value_counts(dropna=False).sort_index()
        print(breakdown)
        print(
            "\n  If every label==0 (control) row shows a single med_status value and every "
            "label==1 (ADHD) row shows a different value (or a spread of values), med_status "
            "is -- by itself -- enough to reconstruct the label, independent of the AUROC "
            "number above (which can look only 'strong', not 'perfect', if missingness "
            "coding differs slightly by site)."
        )
    else:
        print("  'med_status' column not found -- skipped.")

    print(f"\n{'='*72}\n  adhd_index DISTRIBUTION BY LABEL\n{'='*72}")
    if 'adhd_index' in df.columns:
        s = pd.to_numeric(df['adhd_index'], errors='coerce').replace(MISSING_SENTINEL, np.nan)
        print(df.assign(adhd_index_clean=s).groupby('label')['adhd_index_clean'].describe())
        print(
            "\n  adhd_index is a continuous clinical symptom-severity score (Conners' ADHD "
            "Index in the standard ADHD-200 phenotype file). Non-overlapping or barely-"
            "overlapping ranges between the two groups here (e.g. control max << ADHD min) "
            "would mean the baseline model can separate classes almost perfectly from this "
            "ONE feature alone, before the MRI branch contributes anything."
        )
    else:
        print("  'adhd_index' column not found -- skipped.")

    print(f"\n{'='*72}\n  DONE -- nothing was modified. This script only read {INDEX_CSV}.\n{'='*72}")


if __name__ == '__main__':
    main()
