"""
NOTEBOOK ANALYSIS CELLS -- single-file flattened reference copy
=================================================================

Concatenates the two new audit-support cells added to the real Kaggle
notebook (adhd200_notebook_updated.ipynb, cells 28 and 30), verbatim,
under "===== filename =====" section headers, so both can be read or
submitted as supplementary material from one file.

Each section's content starts with a literal `%%writefile ...` Jupyter
magic line -- that is how each one is meant to be pasted into its own
Kaggle/Jupyter cell (the magic writes it to adhd200_src/ as its own
module, matching that notebook's `_adhd200_combined_*` naming and import
convention, then a following cell runs it with
`!python adhd200_src/_adhd200_combined_<name>.py`). Because of that
leading magic line, this flattened file is NOT valid standalone Python on
its own -- paste each section into its own notebook cell, don't execute
this file directly. The repo's `analysis/` directory has the same two
scripts in a plain, directly-importable form (no notebook magic, generic
`dataset`/`phase2_model`/`phase2_consistency` imports) for use outside
this specific notebook.

Both scripts are READ-ONLY / INFERENCE-ONLY: no training, no gradient, no
architecture change, no existing checkpoint/CSV/figure overwritten --
each only reads what an earlier, already-completed stage produced and
writes new, distinctly-named files.

Generated: 2026-11-04
Source cells, in order: check_phenotype_leakage_cell.py, extract_representations_m0_m1_cell.py
"""



#==============================================================================
# NOTEBOOK CELL: check_phenotype_leakage_cell.py
#==============================================================================

%%writefile adhd200_src/_adhd200_combined_check_phenotype_leakage.py
"""
Phenotype leakage check -- audit support script (read-only, no training)
===========================================================================

Checks whether any of the baseline ADHDNet's 5 phenotype input features
(age, sex, med_status, site_id, adhd_index) trivially separates the
diagnostic label on their own. If one does, the baseline's validation
AUROC (0.9516 in this run) may be substantially inflated by quasi-label-
leakage through the phenotype branch rather than genuine MRI-based
signal -- see the "Audit findings" cell above for why this matters.

READ-ONLY: only reads processed_index.csv and prints. Does not train
anything, does not touch any checkpoint/CSV/figure, does not modify
processed_index.csv.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# EDIT if your actual run used a different path -- this is the one printed
# by the TRM/Consistency stages above in this notebook's own run.
INDEX_CSV = '/kaggle/working/adhd200_combined_full_all/processed_index.csv'

PHENO_COLS = ['age', 'sex', 'med_status', 'site_id', 'adhd_index']
MISSING_SENTINEL = -999   # dataset.py's own convention: df.replace(-999, np.nan)


def univariate_auroc(df, col):
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
    print(f"Total subjects: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}\n")

    print(f"{'='*72}\n  UNIVARIATE AUROC -- each phenotype feature alone vs label\n{'='*72}")
    print(f"{'feature':<12}{'AUROC':>8}{'  n_used':>10}{'  n_missing(-999)':>18}   interpretation")
    print('-' * 72)
    for col in PHENO_COLS:
        if col not in df.columns:
            print(f"{col:<12}  column not found -- skipped")
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
        print(df.groupby('label')['med_status'].value_counts(dropna=False).sort_index())
    else:
        print("  'med_status' column not found -- skipped.")

    print(f"\n{'='*72}\n  adhd_index DISTRIBUTION BY LABEL\n{'='*72}")
    if 'adhd_index' in df.columns:
        s = pd.to_numeric(df['adhd_index'], errors='coerce').replace(MISSING_SENTINEL, np.nan)
        print(df.assign(adhd_index_clean=s).groupby('label')['adhd_index_clean'].describe())
    else:
        print("  'adhd_index' column not found -- skipped.")

    print(f"\n{'='*72}\n  DONE -- nothing was modified. This script only read {INDEX_CSV}.\n{'='*72}")


if __name__ == '__main__':
    main()


#==============================================================================
# NOTEBOOK CELL: extract_representations_m0_m1_cell.py
#==============================================================================

%%writefile adhd200_src/_adhd200_combined_extract_representations_m0_m1.py
"""
Phase 2 Step 2 -- post-hoc representation extraction for M0 and M1
(audit support script, NOT part of the core pipeline)
======================================================================

Loads the ALREADY-TRAINED best_m0_model.pt / best_m1_model.pt checkpoints
produced by the Consistency stage above and extracts, for every held-out
VALIDATION subject, the 128-d "brain representation" vector -- the same
pre-classifier embedding the consistency loss itself operates on. Then
computes the diagnostics needed to check whether the consistency
objective improved stability WITHOUT causing representation collapse or
losing discriminative information:

  - per-subject embeddings (saved to CSV)
  - embedding variance across subjects + effective rank (collapse
    indicator: a sharp drop relative to the embedding dimension means
    most dimensions carry ~zero independent information)
  - pairwise cosine similarity between ALL validation-subject pairs,
    split into within-class (control-control, ADHD-ADHD) vs between-class
    (control-ADHD) means
  - within-SUBJECT stability per subject: cos(z(original),
    z(consistency_augment(original))), averaged over repeated draws with
    the SAME fixed STABILITY_SEED this notebook's consistency stage uses

READ-ONLY / INFERENCE-ONLY: no gradient, no optimizer, no training loop,
loads frozen weights from checkpoints already produced by the cell above;
no architecture change (encode_representation/consistency_augment/
cosine_consistency_loss below are copied verbatim from
_adhd200_combined_phase2_consistency.py, not reimplemented); writes only
NEW files, never opens an existing output file in write mode.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, 'adhd200_src')
from _adhd200_combined_dataset import ADHDDataset
from _adhd200_combined_phase2_model import ADHDProposedModelV2

# EDIT if your actual run used different paths -- these are the ones
# printed by the Consistency stage above in this notebook's own run.
INDEX_FILE = '/kaggle/working/adhd200_combined_full_all/processed_index.csv'
OUT_DIR = '/kaggle/working/adhd200_combined_full_all/consistency/phase2/consistency'

ARCH_KWARGS = dict(trm_type='pairwise', use_attention=True)
SEED = 42
STABILITY_SEED = 12345
BATCH_SIZE = 8
N_STABILITY_REPEATS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_paths(tag, out_dir):
    return dict(best=f'{out_dir}/best_{tag}_model.pt')


def consistency_augment(vol, intensity_std=0.05, noise_std=0.02, max_shift=2):
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


def encode_representation(model, vol, pheno=None):
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


def cosine_consistency_loss(z1, z2):
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    cos_sim = (z1n * z2n).sum(dim=-1)
    return (1.0 - cos_sim).mean(), cos_sim


def load_val_dataset():
    train_ids_path = f'{OUT_DIR}/train_subject_ids.csv'
    val_ids_path = f'{OUT_DIR}/val_subject_ids.csv'
    if not (os.path.isfile(train_ids_path) and os.path.isfile(val_ids_path)):
        raise FileNotFoundError(
            f"{train_ids_path} / {val_ids_path} not found. Run the Consistency "
            f"stage above to completion first.")
    train_ids = pd.read_csv(train_ids_path)['subject_id'].tolist()
    val_ids = pd.read_csv(val_ids_path)['subject_id'].tolist()
    print(f"  Loaded existing split: {len(train_ids)} train / {len(val_ids)} val subjects")

    train_stats_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()
    val_ds = ADHDDataset(INDEX_FILE, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
    print(f"  Val subjects loaded: {len(val_ds)}  "
          f"(ADHD={int((val_ds.labels == 1).sum())}  Control={int((val_ds.labels == 0).sum())})")
    return val_ds


def load_best_model(tag):
    paths = checkpoint_paths(tag, OUT_DIR)
    if not os.path.isfile(paths['best']):
        raise FileNotFoundError(f"{paths['best']} not found. Run the Consistency stage "
                                 f"above to completion for variant '{tag}' first.")
    model = ADHDProposedModelV2(**ARCH_KWARGS).to(DEVICE)
    state_dict = torch.load(paths['best'], map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded {paths['best']}  (use_phenotype={model.use_phenotype})")
    return model


def extract_embeddings(model, val_ds):
    subject_ids = val_ds.df['subject_id'].tolist()
    labels = val_ds.labels.tolist()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_z = []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            z = encode_representation(model, vol)
            all_z.append(z.cpu().numpy())
    embeddings = np.concatenate(all_z, axis=0)
    assert embeddings.shape[0] == len(subject_ids)
    return embeddings, subject_ids, labels


def extract_within_subject_stability(model, val_ds, n_repeats=N_STABILITY_REPEATS):
    set_seed(STABILITY_SEED)
    model.eval()
    subject_ids = val_ds.df['subject_id'].tolist()
    labels = val_ds.labels.tolist()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    per_subject_cos = []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            z1 = encode_representation(model, vol)
            batch_cos = [[] for _ in range(vol.shape[0])]
            for _ in range(n_repeats):
                vol2 = consistency_augment(vol.clone())
                z2 = encode_representation(model, vol2)
                _, cos_sim = cosine_consistency_loss(z1, z2)
                cos_sim = cos_sim.cpu().numpy()
                for i in range(vol.shape[0]):
                    batch_cos[i].append(float(cos_sim[i]))
            per_subject_cos.extend(batch_cos)
    df = pd.DataFrame({
        'subject_id': subject_ids, 'label': labels,
        **{f'cos_rep{i+1}': [row[i] for row in per_subject_cos] for i in range(n_repeats)},
    })
    df['cos_mean'] = df[[f'cos_rep{i+1}' for i in range(n_repeats)]].mean(axis=1)
    df['cos_std'] = df[[f'cos_rep{i+1}' for i in range(n_repeats)]].std(axis=1)
    return df


def pairwise_cosine_matrix(embeddings):
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


def effective_rank(embeddings):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def diagnose(tag, embeddings, labels, stability_df):
    labels = np.array(labels)
    cos_mat = pairwise_cosine_matrix(embeddings)
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    pair_labels = labels[iu[0]] == labels[iu[1]]
    within_class_sims = cos_mat[iu][pair_labels]
    between_class_sims = cos_mat[iu][~pair_labels]
    per_dim_var = embeddings.var(axis=0)
    eff_rank = effective_rank(embeddings)
    row = dict(
        variant=tag, n_subjects=n, embed_dim=embeddings.shape[1],
        mean_per_dim_variance=float(per_dim_var.mean()), effective_rank=eff_rank,
        mean_pairwise_cosine_within_class=float(within_class_sims.mean()),
        mean_pairwise_cosine_between_class=float(between_class_sims.mean()),
        class_separation_gap=float(within_class_sims.mean() - between_class_sims.mean()),
        mean_within_subject_stability=float(stability_df['cos_mean'].mean()),
        std_within_subject_stability=float(stability_df['cos_mean'].std()),
    )
    print(f"\n  [{tag}] n={n}  effective_rank={eff_rank:.2f} / {embeddings.shape[1]}  "
          f"mean_per_dim_var={per_dim_var.mean():.6f}")
    print(f"  [{tag}] pairwise cosine: within-class={within_class_sims.mean():.4f}  "
          f"between-class={between_class_sims.mean():.4f}  gap={row['class_separation_gap']:.4f}")
    print(f"  [{tag}] within-subject stability: {row['mean_within_subject_stability']:.4f} "
          f"+/- {row['std_within_subject_stability']:.4f}")
    return row


def main():
    print(f"{'='*70}\n  REPRESENTATION EXTRACTION -- M0 vs M1 (inference-only, no training)\n{'='*70}")
    print(f"  INDEX_FILE = {INDEX_FILE}\n  OUT_DIR    = {OUT_DIR}")
    val_ds = load_val_dataset()

    summary_rows = []
    for tag in ['m0', 'm1']:
        print(f"\n{'-'*60}\n  Variant: {tag}\n{'-'*60}")
        model = load_best_model(tag)
        embeddings, subject_ids, labels = extract_embeddings(model, val_ds)
        emb_df = pd.DataFrame(embeddings, columns=[f'z_{i}' for i in range(embeddings.shape[1])])
        emb_df.insert(0, 'label', labels)
        emb_df.insert(0, 'subject_id', subject_ids)
        emb_path = f'{OUT_DIR}/embeddings_val_{tag}.csv'
        emb_df.to_csv(emb_path, index=False)
        print(f"  Saved {emb_df.shape[0]} x {embeddings.shape[1]}-d embeddings -> {emb_path}")

        stability_df = extract_within_subject_stability(model, val_ds)
        stab_path = f'{OUT_DIR}/within_subject_stability_{tag}.csv'
        stability_df.to_csv(stab_path, index=False)
        print(f"  Saved per-subject stability -> {stab_path}")

        summary_rows.append(diagnose(tag, embeddings, labels, stability_df))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = f'{OUT_DIR}/representation_diagnostics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*70}\n  SUMMARY (M0 vs M1)\n{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {summary_path}")
    print(f"\n  No thresholds were tuned on this data -- descriptive statistics of the "
          f"existing best_m0_model.pt / best_m1_model.pt checkpoints only; nothing was "
          f"trained, and no existing file was overwritten.")


if __name__ == '__main__':
    main()
