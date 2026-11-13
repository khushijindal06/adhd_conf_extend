"""
architecture_audit.py -- Phase 3/4/5 architecture-investigation module
=========================================================================

Separate module implementing:
  Section 1 -- Phase 3.1: encoder feature-map / token diagnostics
  Section 2 -- Phase 3.2: TRM relationship-matrix uniformity diagnostics
  Section 3 -- Phase 3.3: Anatomical Attention scale investigation +
               a scale-preserving alternative, kept as a SEPARATE class
               (existing AnatomicalAttention is untouched)
  Section 4 -- Phase 4: consistency-loss lambda sweep + warmup schedule
  Section 5 -- Phase 5: MRI augmentation ablation (none / existing flips /
               conservative MRI-safe)
  Section 6 -- run_sanity_check(): synthetic-only tests

Preserves all existing code: imports LightweightEncoder/SpatialTokenizer/
FastKANRefinement/TokenRelationshipModule/AnatomicalAttention from
proposed_model.py, PairwiseTokenRelationshipModule/ADHDProposedModelV2
from phase2_model.py, and encode_representation/consistency_augment/
cosine_consistency_loss/classify_from_repr/set_seed/DEVICE from
phase2_consistency.py -- all unmodified. Defines exactly one new model
class (ScalePreservingAnatomicalAttention, Section 3), as an explicit,
separate ablation alongside the existing one, never replacing it.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import ADHDNet
from proposed_model import (LightweightEncoder, SpatialTokenizer, FastKANRefinement,
                             TokenRelationshipModule, AnatomicalAttention)
from phase2_model import PairwiseTokenRelationshipModule, ADHDProposedModelV2
from phase2_consistency import (
    DEVICE, SEED, set_seed, consistency_augment, encode_representation,
    classify_from_repr, cosine_consistency_loss,
)


# ============================================================================
# SECTION 1 -- Phase 3.1: encoder feature-map / token diagnostics
# ============================================================================

def feature_map_statistics(model, vol):
    """
    Phase 3.1: "is the encoder losing important spatial information /
    does the final feature map contain sufficient anatomical detail."
    Runs ONLY the encoder (model.encoder), inference-only, and reports
    basic activation statistics on the (B, 96, d, h, w) feature map --
    mean/std/fraction-dead (exactly-zero after the ReLUs inside
    ConvBlock) per channel, plus the spatial size itself (so a
    resolution change's effect on feature-map granularity is visible
    directly, not just assumed).
    """
    model.eval()
    with torch.no_grad():
        feat = model.encoder(vol.to(DEVICE))
    B, C, D, H, W = feat.shape
    flat = feat.reshape(B, C, -1)
    per_channel_mean = flat.mean(dim=(0, 2))
    per_channel_std = flat.std(dim=(0, 2))
    frac_dead = (flat == 0).float().mean(dim=(0, 2))
    return dict(
        feature_map_shape=(C, D, H, W), batch_size=B,
        mean_activation=float(flat.mean()), std_activation=float(flat.std()),
        mean_frac_dead_channels=float(frac_dead.mean()),
        n_fully_dead_channels=int((frac_dead > 0.99).sum()),
        per_channel_mean=per_channel_mean.cpu().numpy(),
        per_channel_std=per_channel_std.cpu().numpy(),
    )


def token_diagnostics(model, vol):
    """
    Phase 3.1: "do the 16 tokens capture diverse spatial information, or
    are they becoming too similar" + "is the 32-d token representation
    sufficient." Runs the encoder + tokenizer only, inference-only.
    Returns, per sample and aggregated over the batch:
      - token activation stats (mean/std/norm per token slot)
      - a (16, 16) token-similarity matrix (cosine, averaged over the
        batch) -- near-1.0 off-diagonal values mean the tokens have
        collapsed toward each other
      - per-token-slot variance ACROSS THE BATCH (a token slot with ~0
        variance across different subjects carries no discriminative
        information regardless of its within-sample activation)
      - effective rank of the (B, 16*32) flattened token representation
        (same SVD-entropy measure used elsewhere in this audit -- a
        sharp drop relative to 16*32=512 is a collapse signal)
      - the tokenizer's own spatial attention maps (`last_attn`), so
        "are different tokens attending to different places" can be
        checked directly rather than inferred from token similarity alone
    """
    model.eval()
    with torch.no_grad():
        feat = model.encoder(vol.to(DEVICE))
        tokens = model.tokenizer(feat)   # (B, 16, token_dim)
    B, n_tokens, token_dim = tokens.shape
    tok_np = tokens.cpu().numpy()

    norm = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-12)
    sim = torch.einsum('bid,bjd->bij', norm, norm).mean(dim=0)   # (16, 16), averaged over batch
    off_diag_mask = ~torch.eye(n_tokens, dtype=torch.bool)
    mean_off_diag_similarity = float(sim[off_diag_mask].mean())

    per_slot_variance_across_batch = tok_np.var(axis=0).mean(axis=-1)   # (16,)

    flat = tok_np.reshape(B, -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0

    attn_entropy = None
    if model.tokenizer.last_attn is not None:
        attn = model.tokenizer.last_attn.reshape(B, n_tokens, -1)   # (B, 16, L)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)
        ent = -(attn * (attn + 1e-12).log()).sum(dim=-1)   # (B, 16)
        attn_entropy = float(ent.mean())

    return dict(
        n_tokens=n_tokens, token_dim=token_dim,
        mean_token_norm=float(tokens.norm(dim=-1).mean()),
        mean_off_diag_cosine_similarity=mean_off_diag_similarity,
        per_slot_variance_across_batch=per_slot_variance_across_batch,
        min_slot_variance=float(per_slot_variance_across_batch.min()),
        max_slot_variance=float(per_slot_variance_across_batch.max()),
        effective_rank=eff_rank, max_possible_rank=n_tokens * token_dim,
        spatial_attention_entropy=attn_entropy,
        interpretation=(
            "mean_off_diag_cosine_similarity close to 1.0 means the 16 tokens have "
            "collapsed toward each other (same information repeated 16 times, not 16 "
            "diverse views). min_slot_variance close to 0 means that particular token "
            "SLOT carries almost no information across different subjects, regardless of "
            "token_dim. effective_rank well below max_possible_rank (16*token_dim) is the "
            "same collapse signal used for the M0/M1 embedding diagnostics, here applied "
            "to the raw tokens before any downstream refinement/TRM/attention. "
            "spatial_attention_entropy near log(L) (L = feature-map voxel count) means "
            "each token's own spatial pooling is itself close to uniform over the whole "
            "feature map -- i.e. not actually localizing to any particular region."
        ),
    )


def cross_class_token_diversity(model, vol, labels):
    """Phase 3.1: "feature diversity across subjects and diagnostic
    classes" -- token-level version of the embedding within/between-
    class check used elsewhere, applied to the raw (pre-refinement)
    tokens, flattened per subject."""
    model.eval()
    labels = np.asarray(labels)
    with torch.no_grad():
        feat = model.encoder(vol.to(DEVICE))
        tokens = model.tokenizer(feat).reshape(vol.shape[0], -1).cpu().numpy()
    norm = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-12)
    cos_mat = norm @ norm.T
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same_class = labels[iu[0]] == labels[iu[1]]
    return dict(
        within_class_similarity=float(cos_mat[iu][same_class].mean()) if same_class.any() else None,
        between_class_similarity=float(cos_mat[iu][~same_class].mean()) if (~same_class).any() else None,
    )


# ============================================================================
# SECTION 2 -- Phase 3.2: TRM relationship-matrix uniformity diagnostics
# ============================================================================

def relationship_matrix_uniformity(relationship_matrix):
    """
    Phase 3.2: "is the pairwise relationship matrix too uniform."
    Accepts either TokenRelationshipModule's (B, n_heads, 16, 16)
    attention weights or PairwiseTokenRelationshipModule's (B, 16, 16)
    matrix (averaged over heads first if present). Each row is a
    softmax-like distribution over the 16 tokens; its entropy, relative
    to log(16) (a perfectly uniform row), measures how far from uniform
    the learned relationships actually are. 1.0 = perfectly uniform
    (every token treated identically -- the module is not discriminating
    between tokens at all); 0.0 = maximally peaked (each token attends
    to exactly one other token).
    """
    R = relationship_matrix
    if R.dim() == 4:   # (B, n_heads, 16, 16) -> average heads
        R = R.mean(dim=1)
    B, n, _ = R.shape
    row_sums = R.sum(dim=-1, keepdim=True)
    R_norm = R / (row_sums + 1e-12)
    entropy = -(R_norm * (R_norm + 1e-12).log()).sum(dim=-1)   # (B, n)
    max_entropy = float(np.log(n))
    normalized_uniformity = (entropy / max_entropy).mean()
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=R.device)
    mean_off_diag_weight = float(R_norm[off_diag_mask.unsqueeze(0).expand(B, -1, -1)].mean())
    expected_uniform_weight = 1.0 / n
    return dict(
        n_tokens=n, mean_row_entropy=float(entropy.mean()), max_possible_entropy=max_entropy,
        normalized_uniformity=float(normalized_uniformity),
        mean_off_diag_weight=mean_off_diag_weight, expected_weight_if_uniform=expected_uniform_weight,
        interpretation=(
            "normalized_uniformity close to 1.0 means the relationship module has learned "
            "almost nothing beyond 'every token is equally related to every other token' "
            "-- it is adding parameters/compute without adding discriminative structure. "
            "mean_off_diag_weight close to expected_weight_if_uniform (1/n_tokens) is the "
            "same finding stated as a raw attention weight rather than an entropy."
        ),
    )


# ============================================================================
# SECTION 3 -- Phase 3.3: Anatomical Attention scale investigation
# ============================================================================

class ScalePreservingAnatomicalAttention(nn.Module):
    """
    Separate ablation alternative to the existing AnatomicalAttention
    (proposed_model.py) -- NOT a replacement; that class is untouched
    and remains the default. Same learned per-token importance score and
    softmax over the 16 tokens, but the softmax weights (which sum to 1,
    i.e. average 1/n_tokens) are rescaled by n_tokens before the
    elementwise multiply, so a near-uniform attention distribution
    leaves token magnitudes roughly UNCHANGED (average weight ~= 1)
    instead of shrinking every token by ~1/n_tokens. A token that
    genuinely stands out still gets proportionally more weight than one
    that doesn't -- only the OVERALL scale under near-uniform attention
    changes, which is exactly what Phase 3.3 asks to isolate.
    """
    def __init__(self, token_dim, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
        )
        self.last_attn_weights = None

    def forward(self, tokens):
        n_tokens = tokens.shape[1]
        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        self.last_attn_weights = weights.detach()
        scaled_weights = weights * n_tokens   # mean 1.0 instead of mean 1/n_tokens
        weighted = tokens * scaled_weights.unsqueeze(-1)
        return weighted, weights


def attention_magnitude_report(tokens, attention_module):
    """
    Phase 3.3: "report attention distributions, token magnitudes before
    and after attention." Runs ONE given attention module (either the
    existing AnatomicalAttention or ScalePreservingAnatomicalAttention)
    on the given tokens and compares per-token norms before vs after.
    """
    attention_module.eval()
    with torch.no_grad():
        before_norm = tokens.norm(dim=-1)   # (B, 16)
        weighted, weights = attention_module(tokens)
        after_norm = weighted.norm(dim=-1)
    ratio = (after_norm / (before_norm + 1e-12))
    return dict(
        module=type(attention_module).__name__,
        mean_weight=float(weights.mean()), std_weight=float(weights.std()),
        expected_weight_if_uniform=1.0 / tokens.shape[1],
        mean_norm_before=float(before_norm.mean()), mean_norm_after=float(after_norm.mean()),
        mean_norm_ratio_after_over_before=float(ratio.mean()),
    )


# ============================================================================
# SECTION 4 -- Phase 4: consistency-loss lambda sweep + warmup
# ============================================================================

def linear_warmup_lambda(epoch, target_lambda, warmup_epochs):
    """Phase 4: 'gradual consistency-loss warm-up rather than applying
    the full lambda value from the first epoch.' Linear ramp from 0 at
    epoch 1 to target_lambda at epoch warmup_epochs (held at
    target_lambda afterward). warmup_epochs=0 reproduces the existing
    (no-warmup) behavior exactly."""
    if warmup_epochs <= 0:
        return target_lambda
    return target_lambda * min(1.0, epoch / warmup_epochs)


def consistency_training_step(model, vol, pheno, label, criterion, lam):
    """One training step of M0/M1's own loss construction (classification
    + lam * two-view consistency), calling the SAME encode_representation/
    consistency_augment/cosine_consistency_loss/classify_from_repr
    functions phase2_consistency.py's own train_model() uses -- not a
    reimplementation. lam=0.0 is exactly M0 (classification only, but
    still paying for the second forward pass so M0/M1 timing is
    comparable -- pass use_consistency=False upstream instead if that
    overhead should be avoided for a plain M0 run)."""
    z1 = encode_representation(model, vol)
    logits = classify_from_repr(model, z1)
    loss_cls = criterion(logits, label)
    if lam == 0.0:
        return loss_cls, logits, 0.0
    vol2 = consistency_augment(vol.clone())
    z2 = encode_representation(model, vol2)
    loss_con, _ = cosine_consistency_loss(z1, z2)
    return loss_cls + lam * loss_con, logits, float(loss_con.detach())


def run_lambda_sweep_one_epoch(model_kwargs, train_ds, val_ds, lambdas=(0.0, 0.005, 0.01, 0.025, 0.05),
                                warmup_epochs=0, epoch=1, batch_size=8, lr=1e-4, seed=SEED):
    """
    Phase 4's controlled lambda sweep, one epoch at a time (the caller
    loops epochs and calls this per epoch, so warmup scheduling and
    early stopping stay in the caller's hands -- kept this way rather
    than a monolithic multi-epoch function so Test for this stays fast
    and the real multi-epoch driver can reuse experimental_audit.py's
    existing checkpoint/resume machinery instead of a second copy of it).
    Returns one results row per lambda: train loss components +
    validation AUROC/AUPRC/sensitivity/specificity/F1/balanced accuracy
    + cosine representation stability, all from a FRESH model per
    lambda (same seed, so only lambda differs).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for lam in lambdas:
        set_seed(seed)
        model = ADHDProposedModelV2(**model_kwargs).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        cw = train_ds.class_weights().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        effective_lam = linear_warmup_lambda(epoch, lam, warmup_epochs)

        model.train()
        total_cls, total_con = 0.0, 0.0
        for vol, pheno, label in train_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss, _, con_term = consistency_training_step(model, vol, pheno, label, criterion, effective_lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_cls += float(criterion(classify_from_repr(model, encode_representation(model, vol)).detach(), label))
            total_con += con_term

        model.eval()
        probs, labels_list = [], []
        with torch.no_grad():
            for vol, pheno, label in val_loader:
                vol = vol.to(DEVICE)
                z = encode_representation(model, vol)
                p = torch.softmax(classify_from_repr(model, z), dim=1)[:, 1].cpu().numpy()
                probs.extend(p); labels_list.extend(label.numpy())
        preds = (np.array(probs) >= 0.5).astype(int)
        try:
            auroc = roc_auc_score(labels_list, probs)
            auprc = average_precision_score(labels_list, probs)
        except Exception:
            auroc, auprc = 0.5, float(np.mean(labels_list))
        cm = confusion_matrix(labels_list, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = balanced_accuracy_score(labels_list, preds)
        rows.append(dict(lam=lam, effective_lam=effective_lam, auroc=auroc, auprc=auprc,
                          sensitivity=sens, specificity=spec, balanced_accuracy=bal_acc,
                          train_loss_cls=total_cls / len(train_loader),
                          train_loss_consistency=total_con / len(train_loader)))
    return pd.DataFrame(rows)


# ============================================================================
# SECTION 5 -- Phase 5: MRI augmentation ablation
# ============================================================================

class ConservativeAugmentDataset(Dataset):
    """
    Phase 5: wraps an EXISTING ADHDDataset instance (built with
    augment=False, so dataset.py's own flip augmentation never runs)
    and applies ONLY phase2_consistency.py's existing consistency_
    augment() -- intensity gain/bias + Gaussian noise + a small
    integer-voxel translation, explicitly no rotation/flip/warp -- as
    the "conservative MRI-safe augmentation" training-time transform.
    Does not modify the wrapped dataset or dataset.py's own augment=True
    behavior at all; Phase-1 scripts that still use augment=True flips
    are completely unaffected by this class existing.
    """
    def __init__(self, base_dataset, seed=SEED):
        self.base = base_dataset
        self._seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        vol, pheno, label = self.base[idx]
        vol = consistency_augment(vol.unsqueeze(0)).squeeze(0)
        return vol, pheno, label

    def class_weights(self):
        return self.base.class_weights()


def augmentation_ablation_specs(base_train_ds_no_aug, base_train_ds_with_flip):
    """
    Phase 5's 3-way comparison, expressed as ready-to-use training
    datasets rather than a bespoke training loop (so each arm plugs
    directly into train_one_fold()/run_controlled_comparison() from
    experimental_audit.py, or phase2_trm_ablation.py's train_variant(),
    unchanged):
      'none'          -- base_train_ds_no_aug as given (augment=False)
      'existing_flip' -- base_train_ds_with_flip as given (augment=True,
                          dataset.py's own random-flip augmentation)
      'conservative'  -- base_train_ds_no_aug wrapped in
                          ConservativeAugmentDataset (intensity/noise/
                          shift only, no flips)
    Both base datasets must be built from the SAME subject split and
    SAME fit_stats for this to be a controlled comparison -- that is the
    caller's responsibility (same as every other experiment in this
    package), not re-derived here.
    """
    return {
        'none': base_train_ds_no_aug,
        'existing_flip': base_train_ds_with_flip,
        'conservative': ConservativeAugmentDataset(base_train_ds_no_aug),
    }


# ============================================================================
# SECTION 6 -- synthetic-only sanity checks (no real dataset touched)
# ============================================================================

def run_sanity_check():
    print(f"{'=' * 70}\n  ARCHITECTURE AUDIT MODULE -- SYNTHETIC SANITY CHECKS\n{'=' * 70}")
    from dataset import ADHDDataset
    from experimental_audit import _make_synthetic_dataset
    import tempfile

    # ---- Test A: feature-map + token diagnostics run and return sane shapes ----
    print("\n  Test A -- feature-map + token diagnostics (inference-only):")
    model = ADHDProposedModelV2(trm_type='pairwise', use_attention=True).to(DEVICE)
    vol = torch.randn(6, 1, 64, 64, 64, device=DEVICE)
    labels = np.array([0, 1, 0, 1, 0, 1])

    feat_report = feature_map_statistics(model, vol)
    assert feat_report['feature_map_shape'][0] == 96
    assert feat_report['batch_size'] == 6
    print(f"    feature_map_shape={feat_report['feature_map_shape']}  "
          f"mean_frac_dead_channels={feat_report['mean_frac_dead_channels']:.4f}")

    tok_report = token_diagnostics(model, vol)
    assert tok_report['n_tokens'] == 16 and tok_report['token_dim'] == 32
    assert 1 <= tok_report['effective_rank'] <= tok_report['max_possible_rank']
    assert -1.0 <= tok_report['mean_off_diag_cosine_similarity'] <= 1.0
    print(f"    n_tokens=16  effective_rank={tok_report['effective_rank']:.2f}/"
          f"{tok_report['max_possible_rank']}  "
          f"mean_off_diag_cosine={tok_report['mean_off_diag_cosine_similarity']:.4f}")

    diversity = cross_class_token_diversity(model, vol, labels)
    assert diversity['within_class_similarity'] is not None
    print(f"    cross_class_token_diversity: within={diversity['within_class_similarity']:.4f}  "
          f"between={diversity['between_class_similarity']:.4f}")
    print("  TEST A PASSED")

    # ---- Test B: relationship-matrix uniformity against KNOWN ground truth ----
    print("\n  Test B -- relationship-matrix uniformity, known uniform vs known peaked:")
    n = 16
    uniform_R = torch.full((4, n, n), 1.0 / n)
    uniform_report = relationship_matrix_uniformity(uniform_R)
    assert uniform_report['normalized_uniformity'] > 0.99, \
        f"a perfectly uniform matrix must score ~1.0, got {uniform_report['normalized_uniformity']}"

    peaked_R = torch.eye(n).unsqueeze(0).repeat(4, 1, 1) * 0.98 + 0.02 / n
    peaked_report = relationship_matrix_uniformity(peaked_R)
    assert peaked_report['normalized_uniformity'] < 0.3, \
        f"a near-identity (maximally peaked) matrix must score close to 0.0, got {peaked_report['normalized_uniformity']}"
    print(f"    uniform matrix: normalized_uniformity={uniform_report['normalized_uniformity']:.4f} "
          f"(expect ~1.0)  peaked matrix: normalized_uniformity={peaked_report['normalized_uniformity']:.4f} "
          f"(expect close to 0.0)")

    real_model = ADHDProposedModelV2(trm_type='pairwise', use_attention=True).to(DEVICE)
    with torch.no_grad():
        real_model(vol)
    real_report = relationship_matrix_uniformity(real_model.trm.last_relationship_matrix)
    print(f"    untrained pairwise TRM's own relationship matrix: "
          f"normalized_uniformity={real_report['normalized_uniformity']:.4f}")
    print("  TEST B PASSED")

    # ---- Test C: scale-preserving attention actually preserves scale under near-uniform weights ----
    print("\n  Test C -- scale-preserving attention vs existing AnatomicalAttention, forced near-uniform:")
    token_dim = 32
    tokens = torch.randn(8, 16, token_dim, device=DEVICE)
    existing = AnatomicalAttention(token_dim).to(DEVICE)
    scale_preserving = ScalePreservingAnatomicalAttention(token_dim).to(DEVICE)
    # zero the score network's weights so softmax is forced near-uniform regardless of
    # token content, isolating the SCALING behavior under the "near-uniform attention"
    # case Phase 3.3 explicitly describes
    with torch.no_grad():
        for attn_module in (existing, scale_preserving):
            for layer in attn_module.score:
                if isinstance(layer, nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
    existing_report = attention_magnitude_report(tokens, existing)
    scaled_report = attention_magnitude_report(tokens, scale_preserving)
    assert abs(existing_report['mean_weight'] - 1.0 / 16) < 1e-4
    assert existing_report['mean_norm_ratio_after_over_before'] < 0.1, \
        "existing AnatomicalAttention under forced near-uniform weights should shrink magnitude by ~1/16"
    assert abs(scaled_report['mean_norm_ratio_after_over_before'] - 1.0) < 0.05, \
        "ScalePreservingAnatomicalAttention under forced near-uniform weights should preserve magnitude"
    print(f"    existing AnatomicalAttention:         mean_weight={existing_report['mean_weight']:.4f}  "
          f"norm_ratio_after/before={existing_report['mean_norm_ratio_after_over_before']:.4f} (collapses toward ~1/16)")
    print(f"    ScalePreservingAnatomicalAttention:    mean_weight={scaled_report['mean_weight']:.4f}  "
          f"norm_ratio_after/before={scaled_report['mean_norm_ratio_after_over_before']:.4f} (preserved, ~1.0)")
    print("  TEST C PASSED")

    # ---- Test D: warmup schedule against known ground truth ----
    print("\n  Test D -- lambda warmup schedule, known ground truth:")
    assert abs(linear_warmup_lambda(1, 0.05, 5) - 0.01) < 1e-9
    assert abs(linear_warmup_lambda(5, 0.05, 5) - 0.05) < 1e-9
    assert abs(linear_warmup_lambda(10, 0.05, 5) - 0.05) < 1e-9
    assert linear_warmup_lambda(1, 0.05, 0) == 0.05   # warmup_epochs=0 -> no warmup, full lambda immediately
    print("    epoch 1/5 -> 0.01, epoch 5/5 -> 0.05, epoch 10/5 -> 0.05 (held), warmup=0 -> 0.05 (unchanged) -- all correct")
    print("  TEST D PASSED")

    # ---- Test E: lambda sweep runs end-to-end on tiny synthetic data ----
    print("\n  Test E -- lambda sweep, tiny synthetic data, 1 epoch, 3 lambda values:")
    with tempfile.TemporaryDirectory(prefix='arch_audit_lambda_') as tmp:
        index_path, df = _make_synthetic_dataset(tmp, n_subjects=16, sites=('SiteA', 'SiteB'))
        train_ids = df['subject_id'].tolist()[:12]
        val_ids = df['subject_id'].tolist()[12:]
        fit_stats = ADHDDataset(index_path, subject_ids=train_ids, augment=False).get_fit_stats()
        train_ds = ADHDDataset(index_path, subject_ids=train_ids, augment=False, fit_stats=fit_stats)
        val_ds = ADHDDataset(index_path, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
        sweep_df = run_lambda_sweep_one_epoch(
            dict(trm_type='pairwise', use_attention=True), train_ds, val_ds,
            lambdas=(0.0, 0.025, 0.05), warmup_epochs=0, epoch=1)
        assert len(sweep_df) == 3
        assert set(sweep_df['lam']) == {0.0, 0.025, 0.05}
        assert sweep_df['auroc'].between(0, 1).all()
        print(f"    {sweep_df[['lam', 'auroc', 'sensitivity', 'specificity', 'balanced_accuracy']].to_string(index=False)}")
    print("  TEST E PASSED")

    # ---- Test F: augmentation ablation -- 'none' deterministic, 'conservative' varies without flipping ----
    print("\n  Test F -- augmentation ablation ('none' deterministic, 'conservative' varies, no flips):")
    with tempfile.TemporaryDirectory(prefix='arch_audit_aug_') as tmp:
        # A deliberately STRUCTURED volume (a single bright blob well off-center along
        # axis 0), not random noise: pure white noise has no spatial correlation across
        # even a 1-voxel shift, which would make "no flip" unverifiable by position. A
        # localized blob's position after augmentation IS a clean, direct check: a small
        # integer-voxel shift moves it by at most max_shift voxels and keeps it on the
        # SAME side of center; a flip would move it to the OPPOSITE side.
        blob_vol = np.zeros((64, 64, 64), dtype=np.float32)
        blob_center = (16, 32, 32)   # off-center along axis 0 (64/2=32, so 16 is clearly on one side)
        blob_vol[blob_center[0] - 2:blob_center[0] + 3,
                 blob_center[1] - 2:blob_center[1] + 3,
                 blob_center[2] - 2:blob_center[2] + 3] = 1.0
        npy_path = f'{tmp}/blob.npy'
        np.save(npy_path, blob_vol)
        index_path = f'{tmp}/processed_index.csv'
        pd.DataFrame([dict(subject_id='BLOB0', label=0, site='SiteA', npy_path=npy_path,
                            age=10, sex=0, med_status=1, site_id=0, adhd_index=40)]).to_csv(index_path, index=False)

        no_aug = ADHDDataset(index_path, augment=False)
        with_flip = ADHDDataset(index_path, augment=True)
        specs = augmentation_ablation_specs(no_aug, with_flip)
        assert set(specs.keys()) == {'none', 'existing_flip', 'conservative'}

        vol_none_1, _, _ = specs['none'][0]
        vol_none_2, _, _ = specs['none'][0]
        assert torch.equal(vol_none_1, vol_none_2), "'none' (augment=False) must be deterministic across reads"

        vol_cons_1, _, _ = specs['conservative'][0]
        vol_cons_2, _, _ = specs['conservative'][0]
        assert not torch.equal(vol_cons_1, vol_cons_2), \
            "'conservative' must vary across reads (intensity/noise/shift are randomized)"
        assert vol_cons_1.shape == vol_none_1.shape

        # locate the blob (peak of summed intensity along axis 0) before and after
        # conservative augmentation -- the direct, position-based "did this flip?" check:
        # a small translation keeps the blob on the SAME side of the volume's center;
        # a flip would move it to the OPPOSITE side. (The exact displacement magnitude
        # isn't asserted tightly here -- max_shift=2 bounds the roll itself, but the
        # 5-voxel-wide blob's own extent adds a little slack to where "the peak" reads
        # out -- so this checks side-of-center, a flip-or-not question, not the precise
        # voxel count, and separately rules out a gross runaway shift.)
        augmented_peak_axis0 = int(vol_cons_1[0].sum(dim=(1, 2)).argmax())
        shift_distance = abs(augmented_peak_axis0 - blob_center[0])
        same_side_of_center = (augmented_peak_axis0 < 32) == (blob_center[0] < 32)
        assert same_side_of_center, \
            f"conservative augmentation moved the blob to the OPPOSITE side of center -- looks like a flip! " \
            f"original={blob_center[0]}, after={augmented_peak_axis0}"
        assert shift_distance <= 10, \
            f"conservative augmentation moved the blob implausibly far ({shift_distance} voxels) for a " \
            f"documented max_shift=2 roll -- investigate before trusting this as 'conservative'"
        print(f"    'none': deterministic={torch.equal(vol_none_1, vol_none_2)}  "
              f"'conservative': varies={not torch.equal(vol_cons_1, vol_cons_2)}  "
              f"blob position: original axis0={blob_center[0]}, after conservative aug={augmented_peak_axis0} "
              f"(same side of center: {same_side_of_center}, shift={shift_distance} voxels -- no flip)")
    print("  TEST F PASSED")

    print(f"\n{'=' * 70}\n  ALL SYNTHETIC SANITY CHECKS (A-F) PASSED ✓\n{'=' * 70}")
    print("  (No real dataset was loaded, nothing under baseline/, phase2/, or "
          "phase2/consistency/ was read or written. AnatomicalAttention itself was not "
          "modified -- ScalePreservingAnatomicalAttention is a separate, additional class.)")


if __name__ == '__main__':
    run_sanity_check()
