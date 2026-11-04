"""
Phase 2 Step 2 -- post-hoc representation extraction for M0 and M1
(audit support script, NOT part of the core pipeline)
======================================================================

Purpose
-------
Loads the ALREADY-TRAINED best_m0_model.pt / best_m1_model.pt checkpoints
produced by phase2_consistency.py and extracts, for every held-out
VALIDATION subject, the 128-d "brain representation" vector
(encode_representation()'s output -- the same pre-classifier embedding
the consistency loss itself operates on). It then computes the
representation-level diagnostics needed to check whether the consistency
objective improved stability WITHOUT causing representation collapse or
losing discriminative information:

  - per-subject embeddings (saved to CSV -- for your own further analysis)
  - embedding variance across subjects, per-dimension and aggregate
  - effective rank of the embedding matrix (collapse indicator: a sharp
    drop relative to the embedding dimension means most dimensions carry
    ~zero independent information)
  - pairwise cosine similarity between ALL validation-subject pairs,
    split into WITHIN-class (control-control, ADHD-ADHD) vs BETWEEN-class
    (control-ADHD) means
  - within-SUBJECT stability: for each subject, cos(z(original),
    z(consistency_augment(original))) averaged over repeated draws --
    reusing phase2_consistency.py's own measure_representation_stability()
    design (same STABILITY_SEED, same consistency_augment()), just also
    keeping the PER-SUBJECT values instead of only the overall mean

This is READ-ONLY / INFERENCE-ONLY:
  - no gradient, no optimizer, no training loop
  - loads frozen weights from checkpoints you already produced
  - no architecture change: `encode_representation()` is imported
    unmodified from phase2_consistency.py -- the exact same function
    called during training/evaluation, not a reimplementation
  - does not touch, overwrite, or move any existing checkpoint, CSV, or
    figure -- only ever WRITES NEW files (listed below), never opens an
    existing output file in write mode

Usage
-----
Place this file in the SAME directory as phase2_consistency.py (so the
`from phase2_consistency import ...` below resolves), after you have
already run `python phase2_consistency.py` to completion for both M0 and
M1 (i.e. `best_m0_model.pt`, `best_m1_model.pt`, `train_subject_ids.csv`,
`val_subject_ids.csv` already exist under OUT_DIR). Then:

    python extract_representations_m0_m1.py

New files written (all under phase2_consistency.py's OUT_DIR, all NEW
filenames -- nothing here collides with an existing output):
    embeddings_val_m0.csv              per-subject 128-d embedding, M0
    embeddings_val_m1.csv              per-subject 128-d embedding, M1
    within_subject_stability_m0.csv    per-subject cos(orig, augmented), M0
    within_subject_stability_m1.csv    per-subject cos(orig, augmented), M1
    representation_diagnostics.csv     one row per variant: the summary
                                        numbers below, M0 vs M1 side by side
"""

import os
import numpy as np
import pandas as pd
import torch

from dataset import ADHDDataset
from phase2_model import ADHDProposedModelV2
from phase2_consistency import (
    DATA_ROOT, INDEX_FILE, OUT_DIR, SEED, VAL_SPLIT, ARCH_KWARGS, DEVICE,
    STABILITY_SEED, BATCH_SIZE,
    set_seed, checkpoint_paths, consistency_augment, encode_representation,
    cosine_consistency_loss,
)

N_STABILITY_REPEATS = 5   # independent consistency_augment() draws per subject


# ============================================================================
# Section 1 -- rebuild the EXACT validation set used during training
# ============================================================================
def load_val_dataset():
    """
    Reproduces main()'s dataset construction exactly (same files, same
    function calls, same order), so the embeddings extracted here are
    guaranteed to come from the SAME subjects, in the SAME preprocessed
    form, that best_m0_model.pt / best_m1_model.pt were trained/evaluated
    against. Requires train_subject_ids.csv / val_subject_ids.csv to
    already exist (written by phase2_consistency.py's main() on its first
    run) -- this script never invents a new split.
    """
    train_ids_path = f'{OUT_DIR}/train_subject_ids.csv'
    val_ids_path = f'{OUT_DIR}/val_subject_ids.csv'
    if not (os.path.isfile(train_ids_path) and os.path.isfile(val_ids_path)):
        raise FileNotFoundError(
            f"{train_ids_path} / {val_ids_path} not found. Run "
            f"`python phase2_consistency.py` to completion first -- this "
            f"script only extracts embeddings from an ALREADY-TRAINED run, "
            f"it does not create a split or train anything itself."
        )
    train_ids = pd.read_csv(train_ids_path)['subject_id'].tolist()
    val_ids = pd.read_csv(val_ids_path)['subject_id'].tolist()
    print(f"  Loaded existing split: {len(train_ids)} train / {len(val_ids)} val subjects "
          f"(from {train_ids_path}, {val_ids_path})")

    # Imputation/normalization stats, fit on train subjects only -- identical
    # two-line computation main() uses, so this is exact, not an approximation.
    train_stats_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()

    val_ds = ADHDDataset(INDEX_FILE, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
    print(f"  Val subjects loaded: {len(val_ds)}  "
          f"(ADHD={int((val_ds.labels == 1).sum())}  Control={int((val_ds.labels == 0).sum())})")
    return val_ds


def load_best_model(tag):
    paths = checkpoint_paths(tag, OUT_DIR)
    if not os.path.isfile(paths['best']):
        raise FileNotFoundError(
            f"{paths['best']} not found. Run `python phase2_consistency.py` to "
            f"completion for variant '{tag}' first."
        )
    model = ADHDProposedModelV2(**ARCH_KWARGS).to(DEVICE)
    state_dict = torch.load(paths['best'], map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded {paths['best']}  (use_phenotype={model.use_phenotype})")
    return model


# ============================================================================
# Section 2 -- per-subject embedding extraction (no gradient, eval mode)
# ============================================================================
def extract_embeddings(model, val_ds):
    """
    One deterministic forward pass per validation subject through
    encode_representation() -- the SAME function train_model() calls to
    build z1 for the consistency loss, imported unmodified. Returns an
    (N, 128) float array (subject order == val_ds.df row order) plus the
    aligned subject_id / label lists.
    """
    subject_ids = val_ds.df['subject_id'].tolist()
    labels = val_ds.labels.tolist()

    loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=0)
    all_z = []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            z = encode_representation(model, vol)
            all_z.append(z.cpu().numpy())
    embeddings = np.concatenate(all_z, axis=0)
    assert embeddings.shape[0] == len(subject_ids), \
        f"embedding count {embeddings.shape[0]} != subject count {len(subject_ids)}"
    return embeddings, subject_ids, labels


def extract_within_subject_stability(model, val_ds, n_repeats=N_STABILITY_REPEATS):
    """
    Per-subject version of phase2_consistency.py's measure_representation_
    stability(): same STABILITY_SEED, same consistency_augment() calls (so
    M0 and M1 see the identical perturbation sequence here too), but keeps
    each subject's own cosine similarities instead of collapsing straight
    to a single overall mean/std.
    """
    set_seed(STABILITY_SEED)
    model.eval()
    subject_ids = val_ds.df['subject_id'].tolist()
    labels = val_ds.labels.tolist()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=0)
    per_subject_cos = []   # list of lists, one per subject, len == n_repeats
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
    assert len(per_subject_cos) == len(subject_ids)
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'label': labels,
        **{f'cos_rep{i+1}': [row[i] for row in per_subject_cos] for i in range(n_repeats)},
    })
    df['cos_mean'] = df[[f'cos_rep{i+1}' for i in range(n_repeats)]].mean(axis=1)
    df['cos_std'] = df[[f'cos_rep{i+1}' for i in range(n_repeats)]].std(axis=1)
    return df


# ============================================================================
# Section 3 -- collapse / discriminability diagnostics on the embedding matrix
# ============================================================================
def pairwise_cosine_matrix(embeddings):
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


def effective_rank(embeddings):
    """
    Effective rank via normalized singular-value entropy (Roy & Vetterli,
    2007): exp(entropy of the normalized singular-value distribution).
    Ranges from 1 (all variance in one direction -- severe collapse) up to
    min(N, 128) (variance spread evenly across every dimension). A sharp
    drop from M0 to M1 is evidence of representation collapse; a value
    that stays close to M0's is evidence collapse did NOT happen.
    """
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
    iu = np.triu_indices(n, k=1)   # upper triangle, excluding diagonal -- each pair once
    pair_labels = labels[iu[0]] == labels[iu[1]]
    within_class_sims = cos_mat[iu][pair_labels]
    between_class_sims = cos_mat[iu][~pair_labels]

    per_dim_var = embeddings.var(axis=0)
    eff_rank = effective_rank(embeddings)

    row = dict(
        variant=tag,
        n_subjects=n,
        embed_dim=embeddings.shape[1],
        mean_per_dim_variance=float(per_dim_var.mean()),
        effective_rank=eff_rank,
        mean_pairwise_cosine_within_class=float(within_class_sims.mean()),
        mean_pairwise_cosine_between_class=float(between_class_sims.mean()),
        class_separation_gap=float(within_class_sims.mean() - between_class_sims.mean()),
        mean_within_subject_stability=float(stability_df['cos_mean'].mean()),
        std_within_subject_stability=float(stability_df['cos_mean'].std()),
    )
    print(f"\n  [{tag}] n={n}  effective_rank={eff_rank:.2f} / {embeddings.shape[1]}  "
          f"mean_per_dim_var={per_dim_var.mean():.6f}")
    print(f"  [{tag}] pairwise cosine: within-class={within_class_sims.mean():.4f}  "
          f"between-class={between_class_sims.mean():.4f}  "
          f"gap(within-between)={row['class_separation_gap']:.4f}")
    print(f"  [{tag}] within-subject stability (orig vs consistency_augment()): "
          f"{row['mean_within_subject_stability']:.4f} +/- {row['std_within_subject_stability']:.4f}")
    return row


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"{'='*70}\n  REPRESENTATION EXTRACTION -- M0 vs M1 (inference-only, no training)\n{'='*70}")
    print(f"  DATA_ROOT = {DATA_ROOT}\n  OUT_DIR   = {OUT_DIR}")

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

    m0, m1 = summary_df.iloc[0], summary_df.iloc[1]
    print(f"\n  Effective rank:        M0={m0['effective_rank']:.2f}  M1={m1['effective_rank']:.2f}  "
          f"(M1 much lower than M0 => possible collapse)")
    print(f"  Class separation gap:  M0={m0['class_separation_gap']:.4f}  M1={m1['class_separation_gap']:.4f}  "
          f"(M1 much lower than M0 => consistency may be trading away discriminative information)")
    print(f"  Within-subject stab.:  M0={m0['mean_within_subject_stability']:.4f}  "
          f"M1={m1['mean_within_subject_stability']:.4f}  (M1 higher is the intended effect)")
    print(f"\n  No thresholds were tuned on this data -- all numbers above are descriptive "
          f"statistics of the existing best_m0_model.pt / best_m1_model.pt checkpoints and "
          f"the existing validation split; nothing was trained, and no existing file was "
          f"overwritten (only the 5 new files listed in this script's docstring were written).")


if __name__ == '__main__':
    main()
