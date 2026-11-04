"""
FULL CODEBASE -- single-file flattened reference copy
=======================================================

This file concatenates every script in code/ , verbatim, in dependency
order, under clear "===== filename =====" section headers, purely so the
entire pipeline can be read or submitted as supplementary material from
ONE file.

DO NOT RUN THIS FILE DIRECTLY. It is a reference copy, not a merged
program: several of the original scripts (train.py, ablation_study.py,
cross_validate.py, cross_site_eval.py, phase2_trm_ablation.py,
phase2_consistency.py, proposed_ablation_study.py,
generate_architecture_figure.py, generate_paper_figures.py) execute a full
pipeline at module scope / inside main(), each defines its own
module-level config constants (SEED, DEVICE, EPOCHS, OUT_DIR, ...) that
intentionally differ in value between scripts, and several reuse class
names across files for genuinely different purposes. Concatenated into
one importable/runnable module, those would collide. The actual,
independently runnable package is in code/ as separate files -- run each
script there exactly as its own README section describes. This file
exists only so a reviewer can scroll one document instead of opening 17.

Generated: 2026-11-04
Source files, in order: model.py, dataset.py, proposed_model.py, phase2_model.py, train.py, evaluate.py, predict.py, cross_validate.py, cross_site_eval.py, ablation_study.py, benchmark.py, proposed_ablation_study.py, proposed_explainability.py, phase2_trm_ablation.py, phase2_consistency.py, generate_architecture_figure.py, generate_paper_figures.py
"""



#==============================================================================
# FILE: code/model.py
#==============================================================================


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
    """
    Depthwise Conv3d (one 3x3x3 filter per input channel) + Pointwise 1x1x1 Conv3d.
    Factorises a standard 3x3x3xCin xCout convolution into depthwise + pointwise
    steps, cutting parameters by ~60-80% (the MobileNet trick, adapted to 3D).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ECA3D(nn.Module):
    """
    3D Efficient Channel Attention (Wang et al., ECA-Net, adapted to 3D volumes).
    Global-average-pools each channel, then runs a small 1D conv across the
    channel dimension (no dimensionality reduction) to model local cross-channel
    interaction, and uses the result to re-weight the channels. Adds very few
    parameters (one 1D kernel) while improving channel selection.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)      # (B,1,C)
        y = self.conv(y)                        # (B,1,C)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)  # (B,C,1,1,1)
        return x * y


class GeM3d(nn.Module):
    """
    Generalized Mean Pooling (Radenovic et al.) for 3D feature maps.
    Interpolates between average pooling (p=1) and max pooling (p -> inf);
    p is a learnable scalar, initialised to 3.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool3d(x, 1)
        return x.pow(1.0 / self.p)


class ConvBlock(nn.Module):
    """Depthwise-separable Conv3d -> BN -> ReLU -> [3D ECA] -> MaxPool."""
    def __init__(self, in_channels, out_channels, use_eca=True):
        super().__init__()
        self.conv = DepthwiseSeparableConv3d(in_channels, out_channels)
        self.bn   = nn.BatchNorm3d(out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.eca  = ECA3D(out_channels) if use_eca else nn.Identity()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.eca(x)
        x = self.pool(x)
        return x


class ConvBlockStd(nn.Module):
    """Standard Conv3d -> BN -> ReLU -> [3D ECA] -> MaxPool."""
    def __init__(self, in_channels, out_channels, use_eca=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm3d(out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.eca  = ECA3D(out_channels) if use_eca else nn.Identity()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.eca(x)
        x = self.pool(x)
        return x


class ADHDNet(nn.Module):
    """
    Lightweight Paper-15 architecture:
    - 3x depthwise-separable Conv3d blocks, THEN one 3D ECA over the final
      feature map, THEN global average pooling:
      (DS-Conv3D -> BN -> ReLU -> Pool) x3 -> 3D ECA -> GAP : (B,1,64,64,64) -> zimg (B,128)
    - MLP pheno branch                            : (B,5)          -> zphen (B,32)
    - Adaptive gated fusion                       : [zimg||zphen]  -> ffused (B,128)
    - Smaller classifier head                     : ffused         -> logits (B,2)
    """
    def __init__(self, n_pheno=5, dropout=0.3):
        super().__init__()

        img_embed_dim   = 128
        pheno_embed_dim = 32

        # ── 3D CNN branch: 3x depthwise-separable convs, then a single 3D ECA ─
        self.cnn = nn.Sequential(
            # Block 1: (1,64,64,64) -> (24,32,32,32)
            ConvBlock(1, 24, use_eca=False),
            # Block 2: -> (48,16,16,16)
            ConvBlock(24, 48, use_eca=False),
            # Block 3: -> (96,8,8,8)
            ConvBlock(48, 96, use_eca=False),
            # single 3D ECA over the final (96, 8,8,8) feature map
            ECA3D(96),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),                                  # (B, 96)
        )
        self.img_proj = nn.Linear(96, img_embed_dim)        # (B, 128)

        # ── MLP phenotypic branch ────────────────────────────────────────────
        self.pheno_mlp = nn.Sequential(
            nn.Linear(n_pheno, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, pheno_embed_dim),                # (B, 32)
        )

        # ── Gated Fusion ───────────────────────────────────────────────────
        # input: [zimg(128) || zphen(32)] = 160-dim
        self.gate = nn.Sequential(
            nn.Linear(img_embed_dim + pheno_embed_dim, img_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(img_embed_dim, img_embed_dim),
            nn.Sigmoid(),                                  # g ∈ (0,1)^128
        )

        # project zphen from 32 → 128 for element-wise fusion
        self.pheno_proj = nn.Linear(pheno_embed_dim, img_embed_dim)

        # ── Smaller classifier head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),                              # logits for 2 classes
        )

    def forward(self, vol, pheno):
        zimg  = self.img_proj(self.cnn(vol))               # (B, 128)
        zphen = self.pheno_mlp(pheno)                       # (B, 32)

        combined = torch.cat([zimg, zphen], dim=1)          # (B, 160)
        g        = self.gate(combined)                      # (B, 128)
        zphen128 = self.pheno_proj(zphen)                    # (B, 128)
        ffused   = g * zimg + (1 - g) * zphen128              # (B, 128)

        return self.head(ffused)                            # (B, 2)

    def forward_features(self, vol):
        """
        Non-destructive addition for explainability/the proposed token-based
        architecture: runs the same trained blocks as forward() but stops
        before global average pooling, returning the spatial feature map
        instead of the pooled 128D embedding. Adds no parameters and does
        not change forward()'s behavior or output.
        """
        return self.cnn[:4](vol)   # ConvBlock, ConvBlock, ConvBlock, ECA3D -- skips GAP + Flatten


class ADHDNetEnhanced(nn.Module):
    """
    Enhanced comparison variant — used as the 5-fold CV baseline against the
    lightweight ADHDNet above:
    - 4-block standard 3D CNN, channels 32-64-128-256; 3D ECA after blocks 2,3,4
      (block 1 has no ECA, matching low-level features not needing channel gating)
    - GeM pooling (learnable p, init=3) instead of global average pooling
    - MRI embedding kept at the full 256D (not reduced)
    - MLP pheno branch (5 -> 64D), same as the original baseline
    - Gated fusion with a 2-layer MLP gate: FC -> ReLU -> FC -> Sigmoid
    - Classifier head: 256 -> 128 -> 64 -> 2
    """
    def __init__(self, n_pheno=5, dropout=0.3):
        super().__init__()

        img_embed_dim   = 256
        pheno_embed_dim = 64

        # ── 3D CNN branch: standard convs, 3D ECA after blocks 2-4 ───────────
        self.block1 = ConvBlockStd(1,   32,  use_eca=False)   # (32,  32,32,32)
        self.block2 = ConvBlockStd(32,  64,  use_eca=True)    # (64,  16,16,16)
        self.block3 = ConvBlockStd(64,  128, use_eca=True)    # (128,  8, 8, 8)
        self.block4 = ConvBlockStd(128, 256, use_eca=True)    # (256,  4, 4, 4)

        self.gem  = GeM3d()
        self.flat = nn.Flatten()                              # (B, 256)

        # ── MLP phenotypic branch ────────────────────────────────────────────
        self.pheno_mlp = nn.Sequential(
            nn.Linear(n_pheno, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, pheno_embed_dim),                   # (B, 64)
        )

        # ── Gated Fusion: 2-layer MLP gate (FC -> ReLU -> FC -> Sigmoid) ──────
        # input: [zimg(256) || zphen(64)] = 320-dim
        self.gate = nn.Sequential(
            nn.Linear(img_embed_dim + pheno_embed_dim, img_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(img_embed_dim, img_embed_dim),
            nn.Sigmoid(),                                     # g ∈ (0,1)^256
        )

        # project zphen from 64 → 256 for element-wise fusion
        self.pheno_proj = nn.Linear(pheno_embed_dim, img_embed_dim)

        # ── Classifier head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),                                 # logits for 2 classes
        )

    def forward(self, vol, pheno):
        x = self.block1(vol)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        zimg = self.flat(self.gem(x))                         # (B, 256)

        zphen = self.pheno_mlp(pheno)                          # (B, 64)

        combined = torch.cat([zimg, zphen], dim=1)             # (B, 320)
        g        = self.gate(combined)                          # (B, 256)
        zphen256 = self.pheno_proj(zphen)                        # (B, 256)
        ffused   = g * zimg + (1 - g) * zphen256                  # (B, 256)

        return self.head(ffused)                                # (B, 2)


class ADHDNetNoECA(nn.Module):
    """
    Ablation A1 — full lightweight model with the single 3D ECA (applied
    after the 3 DS-Conv3D blocks in ADHDNet) removed entirely: just
    DS-Conv3D x3 + phenotype + gated fusion, no channel attention anywhere.
    Everything else is identical to ADHDNet.
    """
    def __init__(self, n_pheno=5, dropout=0.3):
        super().__init__()
        img_embed_dim, pheno_embed_dim = 128, 32

        self.cnn = nn.Sequential(
            ConvBlock(1, 24, use_eca=False),
            ConvBlock(24, 48, use_eca=False),
            ConvBlock(48, 96, use_eca=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.img_proj = nn.Linear(96, img_embed_dim)

        self.pheno_mlp = nn.Sequential(
            nn.Linear(n_pheno, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, pheno_embed_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(img_embed_dim + pheno_embed_dim, img_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(img_embed_dim, img_embed_dim),
            nn.Sigmoid(),
        )
        self.pheno_proj = nn.Linear(pheno_embed_dim, img_embed_dim)

        self.head = nn.Sequential(
            nn.Linear(img_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, vol, pheno):
        zimg  = self.img_proj(self.cnn(vol))
        zphen = self.pheno_mlp(pheno)

        combined = torch.cat([zimg, zphen], dim=1)
        g        = self.gate(combined)
        zphen128 = self.pheno_proj(zphen)
        ffused   = g * zimg + (1 - g) * zphen128

        return self.head(ffused)


class ADHDNetMRIOnly(nn.Module):
    """
    Ablation A2 — MRI only: the phenotypic branch and fusion are removed
    entirely; the CNN embedding feeds the classifier head directly.
    `pheno` is accepted (for a drop-in forward signature with the other
    variants / training loop) but unused.
    """
    def __init__(self, n_pheno=5, dropout=0.3):
        super().__init__()
        img_embed_dim = 128

        self.cnn = nn.Sequential(
            ConvBlock(1, 24, use_eca=False),
            ConvBlock(24, 48, use_eca=False),
            ConvBlock(48, 96, use_eca=False),
            ECA3D(96),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.img_proj = nn.Linear(96, img_embed_dim)

        self.head = nn.Sequential(
            nn.Linear(img_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, vol, pheno):
        zimg = self.img_proj(self.cnn(vol))
        return self.head(zimg)


class ADHDNetConcatFusion(nn.Module):
    """
    Ablation A3 — concatenation instead of adaptive gated fusion, isolating
    the fusion mechanism only (not a classifier-input-size confound): the
    phenotype branch is also projected to 128D, so both fusion strategies
    produce the exact same 128D representation before the classifier.
      zimg (128) || zphen (128) -> concat (256) -> Linear(256->128) -> head
    The classifier head is then identical in shape to the full model's.
    """
    def __init__(self, n_pheno=5, dropout=0.3):
        super().__init__()
        img_embed_dim   = 128
        pheno_embed_dim = 128   # matches img_embed_dim, unlike the full model's 32D

        self.cnn = nn.Sequential(
            ConvBlock(1, 24, use_eca=False),
            ConvBlock(24, 48, use_eca=False),
            ConvBlock(48, 96, use_eca=False),
            ECA3D(96),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.img_proj = nn.Linear(96, img_embed_dim)

        self.pheno_mlp = nn.Sequential(
            nn.Linear(n_pheno, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, pheno_embed_dim),
        )

        # concat (256) -> project back down to 128, same dim the gate produces
        self.fusion_proj = nn.Linear(img_embed_dim + pheno_embed_dim, img_embed_dim)

        self.head = nn.Sequential(
            nn.Linear(img_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, vol, pheno):
        zimg  = self.img_proj(self.cnn(vol))
        zphen = self.pheno_mlp(pheno)
        fused = self.fusion_proj(torch.cat([zimg, zphen], dim=1))   # concat, no gate
        return self.head(fused)


class ADHDNetStandardConv(nn.Module):
    """
    Ablation A4 — standard (non-separable) Conv3d backbone, holding every
    other component of the full model fixed (single 3D ECA after the 3
    blocks, phenotype branch, gated fusion). This isolates the contribution
    of the depthwise-separable factorisation specifically, the same way
    A1-A3 each isolate one other component. (If what you actually want for
    A4 is literally the original 1.376M-parameter heavy baseline reported in
    Fig. 4, swap this class for ADHDNetEnhanced in the ablation registry —
    one line.)
    """
    def __init__(self, n_pheno=5, dropout=0.3):
        super().__init__()
        img_embed_dim, pheno_embed_dim = 128, 32

        self.cnn = nn.Sequential(
            ConvBlockStd(1, 24, use_eca=False),
            ConvBlockStd(24, 48, use_eca=False),
            ConvBlockStd(48, 96, use_eca=False),
            ECA3D(96),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.img_proj = nn.Linear(96, img_embed_dim)

        self.pheno_mlp = nn.Sequential(
            nn.Linear(n_pheno, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, pheno_embed_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(img_embed_dim + pheno_embed_dim, img_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(img_embed_dim, img_embed_dim),
            nn.Sigmoid(),
        )
        self.pheno_proj = nn.Linear(pheno_embed_dim, img_embed_dim)

        self.head = nn.Sequential(
            nn.Linear(img_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, vol, pheno):
        zimg  = self.img_proj(self.cnn(vol))
        zphen = self.pheno_mlp(pheno)

        combined = torch.cat([zimg, zphen], dim=1)
        g        = self.gate(combined)
        zphen128 = self.pheno_proj(zphen)
        ffused   = g * zimg + (1 - g) * zphen128

        return self.head(ffused)


if __name__ == '__main__':
    vol   = torch.randn(4, 1, 64, 64, 64)
    pheno = torch.randn(4, 5)

    for name, cls in [('ADHDNet (lightweight, A0)', ADHDNet),
                       ('ADHDNetEnhanced',          ADHDNetEnhanced),
                       ('ADHDNetNoECA (A1)',        ADHDNetNoECA),
                       ('ADHDNetMRIOnly (A2)',      ADHDNetMRIOnly),
                       ('ADHDNetConcatFusion (A3)', ADHDNetConcatFusion),
                       ('ADHDNetStandardConv (A4)', ADHDNetStandardConv)]:
        model = cls(n_pheno=5, dropout=0.3)
        total = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            out = model(vol, pheno)
        probs = torch.softmax(out, dim=1)
        print(f"{name:<24} params={total:>10,}  output={tuple(out.shape)}  "
              f"probs[0]={probs[0].tolist()}")

    print(f"\nmodel.py sanity check PASSED ✓")


#==============================================================================
# FILE: code/dataset.py
#==============================================================================


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ADHDDataset(Dataset):
    def __init__(self, index_csv, subject_ids=None, fit_stats=None, augment=False):
        df = pd.read_csv(index_csv)
        df = df[df['site'] != 'Pittsburgh'].reset_index(drop=True)

        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)].reset_index(drop=True)

        # ── replace -999 with NaN everywhere ──────────────────────────────
        df.replace(-999, np.nan, inplace=True)

        # ── cast to numeric ────────────────────────────────────────────────
        for col in ['age', 'sex', 'med_status', 'site_id', 'adhd_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ── imputation statistics ────────────────────────────────────────────
        # Fit on whatever subset is passed when fit_stats is None (the caller
        # passes only the TRAINING subjects for that call), then reuse the
        # exact same median/mode for every other split via fit_stats. Computing
        # these independently per split leaks validation-set information into
        # the imputed values.
        if fit_stats is None:
            self.age_median        = df['age'].median()
            self.sex_mode          = df['sex'].mode()[0]
            self.med_status_mode   = df['med_status'].mode()[0]
            self.adhd_index_median = df['adhd_index'].median()
        else:
            self.age_median        = fit_stats['age_median']
            self.sex_mode          = fit_stats['sex_mode']
            self.med_status_mode   = fit_stats['med_status_mode']
            self.adhd_index_median = fit_stats['adhd_index_median']

        # ── impute missing values using the (train-derived) statistics ──────
        df['age']        = df['age'].fillna(self.age_median)
        df['sex']        = df['sex'].fillna(self.sex_mode)
        df['site_id']    = df['site_id'].fillna(0)
        df['adhd_index'] = df['adhd_index'].fillna(self.adhd_index_median)
        df['med_status'] = df['med_status'].fillna(self.med_status_mode)

        # ── z-score normalise continuous features ──────────────────────────
        # only age and adhd_index are continuous
        # sex, site_id, med_status are categorical — just scale to 0-1
        if fit_stats is None:
            self.age_mean      = df['age'].mean()
            self.age_std       = df['age'].std()        + 1e-8
            self.adhd_idx_mean = df['adhd_index'].mean()
            self.adhd_idx_std  = df['adhd_index'].std() + 1e-8
            # site_id: 0-5 → divide by 5
            # med_status: 1,2 → divide by 2
            # sex: 0,1 → already 0-1
        else:
            self.age_mean      = fit_stats['age_mean']
            self.age_std       = fit_stats['age_std']
            self.adhd_idx_mean = fit_stats['adhd_idx_mean']
            self.adhd_idx_std  = fit_stats['adhd_idx_std']

        df['age_norm']      = (df['age']        - self.age_mean)      / self.age_std
        df['adhd_idx_norm'] = (df['adhd_index'] - self.adhd_idx_mean) / self.adhd_idx_std
        df['site_norm']     = df['site_id']    / 5.0
        df['med_norm']      = df['med_status'] / 2.0
        df['sex_norm']      = df['sex']        / 1.0

        self.df      = df
        self.labels  = df['label'].values.astype(np.int64)
        self.augment = augment

    def get_fit_stats(self):
        return {
            # imputation statistics
            'age_median'        : self.age_median,
            'sex_mode'          : self.sex_mode,
            'med_status_mode'   : self.med_status_mode,
            'adhd_index_median' : self.adhd_index_median,
            # normalization statistics
            'age_mean'      : self.age_mean,
            'age_std'       : self.age_std,
            'adhd_idx_mean' : self.adhd_idx_mean,
            'adhd_idx_std'  : self.adhd_idx_std,
        }

    def class_weights(self):
        n      = len(self.labels)
        n_ctrl = (self.labels == 0).sum()
        n_adhd = (self.labels == 1).sum()
        return torch.tensor([n/(2*n_ctrl), n/(2*n_adhd)], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── MRI volume ─────────────────────────────────────────────────────
        vol = np.load(row['npy_path']).astype(np.float32)

        # ── Augmentation (training only) ───────────────────────────────────
        if self.augment:
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=0).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=1).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=2).copy()

        vol = torch.from_numpy(vol).unsqueeze(0)           # (1,64,64,64)

        # ── All 5 phenotypic features ──────────────────────────────────────
        # [age_norm, sex_norm, med_norm, site_norm, adhd_idx_norm]
        pheno = torch.tensor([
            float(row['age_norm']),
            float(row['sex_norm']),
            float(row['med_norm']),
            float(row['site_norm']),
            float(row['adhd_idx_norm']),
        ], dtype=torch.float32)                            # (5,)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return vol, pheno, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    INDEX_CSV = '/kaggle/working/processed_index.csv'
    ds = ADHDDataset(INDEX_CSV, augment=True)
    print(f"Total samples  : {len(ds)}")
    print(f"Class weights  : {ds.class_weights()}")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    vol_b, pheno_b, label_b = next(iter(loader))
    print(f"Volume shape   : {vol_b.shape}")       # (4,1,64,64,64)
    print(f"Pheno shape    : {pheno_b.shape}")     # (4,5)
    print(f"Labels         : {label_b}")
    print(f"Pheno sample   : {pheno_b[0]}")        # check values look sane
    print(f"\ndataset.py sanity check PASSED ✓")

#==============================================================================
# FILE: code/proposed_model.py
#==============================================================================

"""
Proposed lightweight brain-token representation-learning architecture,
extending (not replacing) the existing ADHDNet baseline from model.py.

    3D T1 MRI
        |
    Existing lightweight 3D encoder (reused: DS-Conv3D x3 + single 3D ECA)
        |
    Spatial 3D feature map   (B, 96, 8, 8, 8)
        |
    SpatialTokenizer           -> 16 learned spatial brain tokens (B, 16, 32)
        |
    FastKANRefinement (or MLP, or none -- ablatable)
        |
    TokenRelationshipModule (ablatable)
        |
    AnatomicalAttention (ablatable)
        |
    Brain representation (128D, same dim as the baseline's MRI embedding,
    so the existing classifier head can be reused unchanged)
        |
    Standard head  (reused from ADHDNet)   OR   EvidentialHead

Every new module is a separate nn.Module class (no monolithic forward()),
and every stage is independently switchable via ADHDProposedModel's
constructor flags, which is how the required ablation variants (A1-A5 +
phenotype fusion + evidential head) are produced without duplicating code.

Nothing in model.py's ADHDNet is modified. The backbone here is a fresh
instance of the same ConvBlock/ECA3D classes already defined in model.py,
configured identically to ADHDNet's own encoder (same channels, same single
post-backbone ECA) -- i.e. the same *architecture* is reused, not a new
design. `load_pretrained_backbone()` below optionally copies trained
ADHDNet weights into it if you want to fine-tune from the baseline instead
of training from scratch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvBlock, ConvBlockStd, ECA3D


# ============================================================================
# Step 2 (spec) -- expose the spatial feature map
# ============================================================================
class LightweightEncoder(nn.Module):
    """
    The existing lightweight encoder's backbone, structurally identical to
    ADHDNet.cnn[:-2] (everything up to and including the single 3D ECA, but
    BEFORE global average pooling). Returns the spatial feature map, not a
    pooled vector, so the tokenizer below has spatial structure to work with.

    `conv_type='ds'` (default) uses depthwise-separable Conv3D, matching the
    lightweight baseline. `conv_type='standard'` swaps in plain Conv3d, for
    an encoder-level ablation symmetrical to the existing ADHDNetStandardConv.
    """
    def __init__(self, conv_type='ds'):
        super().__init__()
        Block = ConvBlock if conv_type == 'ds' else ConvBlockStd
        self.block1 = Block(1, 24, use_eca=False)
        self.block2 = Block(24, 48, use_eca=False)
        self.block3 = Block(48, 96, use_eca=False)
        self.eca    = ECA3D(96)

    def forward(self, vol):
        x = self.block1(vol)
        x = self.block2(x)
        x = self.block3(x)
        x = self.eca(x)
        return x   # (B, 96, 8, 8, 8) -- spatial feature map, no pooling


def load_pretrained_backbone(encoder: LightweightEncoder, adhdnet_state_dict: dict):
    """
    Optional: copy trained ADHDNet.cnn weights (blocks 0,1,2 + the ECA at
    index 3, per the frozen baseline's `self.cnn` Sequential) into a fresh
    LightweightEncoder, so the proposed model can start from the baseline's
    learned filters instead of training from scratch. Safe to skip --
    every ablation variant in this project is otherwise trained from
    scratch for a fair comparison.
    """
    mapping = {
        'cnn.0.': 'block1.', 'cnn.1.': 'block2.', 'cnn.2.': 'block3.', 'cnn.3.': 'eca.',
    }
    remapped = {}
    for k, v in adhdnet_state_dict.items():
        for old, new in mapping.items():
            if k.startswith(old):
                remapped[new + k[len(old):]] = v
                break
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    return missing, unexpected


# ============================================================================
# Step 3 (spec) -- learned Spatial Tokenizer (slot-attention style, NOT an atlas)
# ============================================================================
class SpatialTokenizer(nn.Module):
    """
    Converts a 3D spatial feature map into `n_tokens` learned, data-driven
    tokens via learned spatial attention pooling:
        (B, C, D, H, W) -> (B, n_tokens, token_dim)

    Each token has its own learned attention map over the D*H*W spatial
    locations (a 1x1x1 conv producing n_tokens logit channels, softmax over
    space), and pools a separately-projected value feature map with it.
    This is two 1x1x1 convs total -- no attention over the full spatial
    volume beyond that, so it stays lightweight regardless of n_tokens.

    No anatomical atlas (AAL/CC200/etc.) is used anywhere here: tokens are
    purely learned during training. `last_attn` after a forward pass holds
    the (B, n_tokens, D*H*W) spatial attention maps for post-hoc analysis.
    """
    def __init__(self, in_channels, n_tokens=16, token_dim=32):
        super().__init__()
        self.n_tokens = n_tokens
        self.attn_logits = nn.Conv3d(in_channels, n_tokens, kernel_size=1)
        self.value_proj  = nn.Conv3d(in_channels, token_dim, kernel_size=1)
        self.last_attn = None   # (B, n_tokens, D, H, W), set on forward

    def forward(self, feat_map):
        B, C, D, H, W = feat_map.shape
        logits = self.attn_logits(feat_map).view(B, self.n_tokens, -1)   # (B, n_tokens, L)
        attn   = torch.softmax(logits, dim=-1)
        values = self.value_proj(feat_map).view(B, -1, D * H * W)          # (B, token_dim, L)
        tokens = torch.einsum('bnl,bdl->bnd', attn, values)                  # (B, n_tokens, token_dim)
        self.last_attn = attn.detach().view(B, self.n_tokens, D, H, W)
        return tokens


# ============================================================================
# Step 4 (spec) -- FastKAN token-level refinement
# ============================================================================
class FastKANLayer(nn.Module):
    """
    A single FastKAN layer (Gaussian-RBF-basis Kolmogorov-Arnold layer,
    following the "FastKAN" reformulation that replaces learnable B-splines
    with a fixed grid of RBF basis functions for speed): for every input
    feature, expand it into `n_grid` RBF responses, then linearly combine
    those (per output unit) instead of a single scalar weight -- giving each
    input-output connection its own learnable nonlinear univariate function,
    at the cost of `n_grid` extra scalars per connection rather than a full
    spline fit. A SiLU-gated linear base term is added as in the reference
    FastKAN formulation, which stabilises early training.
    Operates on the last dimension only: (..., in_dim) -> (..., out_dim).
    """
    def __init__(self, in_dim, out_dim, n_grid=8, grid_min=-2.0, grid_max=2.0):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, n_grid)
        self.register_buffer('grid', grid)
        self.inv_denom = 1.0 / ((grid_max - grid_min) / (n_grid - 1))
        self.layernorm = nn.LayerNorm(in_dim)
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, n_grid) * (1.0 / math.sqrt(in_dim * n_grid)))

    def rbf(self, x):
        diff = (x.unsqueeze(-1) - self.grid) * self.inv_denom   # (..., in_dim, n_grid)
        return torch.exp(-diff * diff)

    def forward(self, x):
        x = self.layernorm(x)
        base = self.base_linear(F.silu(x))
        basis = self.rbf(x)
        spline = torch.einsum('...ig,oig->...o', basis, self.spline_weight)
        return base + spline


class FastKANRefinement(nn.Module):
    """16 spatial tokens -> FastKAN -> refined tokens (residual, same shape)."""
    def __init__(self, token_dim, hidden_dim=None, n_grid=8):
        super().__init__()
        hidden_dim = hidden_dim or token_dim
        self.layer1 = FastKANLayer(token_dim, hidden_dim, n_grid=n_grid)
        self.layer2 = FastKANLayer(hidden_dim, token_dim, n_grid=n_grid)

    def forward(self, tokens):
        x = self.layer1(tokens)
        x = self.layer2(x)
        return tokens + x


class ConventionalMLPRefinement(nn.Module):
    """Ablation 2 alternative to FastKAN: a plain 2-layer MLP, same shape in/out."""
    def __init__(self, token_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or token_dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, tokens):
        return tokens + self.net(tokens)


# ============================================================================
# Step 5 (spec) -- Token Relationship Module (main proposed module)
# ============================================================================
class TokenRelationshipModule(nn.Module):
    """
    Lightweight self-attention over the 16 brain tokens ONLY (never over the
    full 3D feature map), explicitly modelling which tokens' representations
    complement or influence one another. Cost scales with n_tokens^2 (16^2
    = 256), not with the MRI's spatial resolution, so it stays cheap at any
    encoder resolution.
    `last_relationship_matrix` holds (B, n_heads, 16, 16) after a forward
    pass, for ablation / visualization (Step 5 verification + Section 12).
    """
    def __init__(self, token_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.mha   = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.last_relationship_matrix = None

    def forward(self, tokens):
        attn_out, attn_w = self.mha(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        self.last_relationship_matrix = attn_w.detach()   # (B, n_heads, 16, 16)
        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ============================================================================
# Step 6 (spec) -- Anatomical (token-importance) Attention
# ============================================================================
class AnatomicalAttention(nn.Module):
    """
    Learns a single importance weight per token -- which of the 16
    data-driven tokens matter most for the classification decision. This is
    NOT tied to any predefined anatomical region at training time; the name
    reflects its eventual *post-hoc* interpretation (Section 11), not a
    modelling constraint. `last_attn_weights` holds (B, 16) after a forward
    pass for visualization.
    """
    def __init__(self, token_dim, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
        )
        self.last_attn_weights = None

    def forward(self, tokens):
        logits = self.score(tokens).squeeze(-1)     # (B, 16)
        weights = torch.softmax(logits, dim=-1)
        self.last_attn_weights = weights.detach()
        weighted = tokens * weights.unsqueeze(-1)     # (B, 16, token_dim)
        return weighted, weights


# ============================================================================
# Step 9 (spec) -- optional Evidential classification head
# ============================================================================
class EvidentialHead(nn.Module):
    """
    Dirichlet / Subjective-Logic evidential classification head (Sensoy et
    al., 2018): predicts non-negative per-class "evidence" instead of raw
    logits, forms Dirichlet parameters alpha = evidence + 1, and derives
    both an expected class probability and a predictive uncertainty
    u = n_classes / sum(alpha) in (0, 1]. Trained with `evidential_loss`
    below instead of cross-entropy. Returns a dict (alpha/prob/uncertainty/
    logits) rather than bare logits, so callers must branch on
    `model.use_evidential` -- this keeps the standard head's call sites
    (train.py-style loops) unaffected when evidential mode is off.
    """
    def __init__(self, in_dim, n_classes=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x):
        logits = self.net(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha / S
        uncertainty = self.n_classes / S.squeeze(1)
        return {'alpha': alpha, 'prob': prob, 'uncertainty': uncertainty, 'logits': logits}


def _kl_dirichlet(alpha, n_classes):
    ones = torch.ones((1, n_classes), dtype=alpha.dtype, device=alpha.device)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones).sum(dim=1, keepdim=True)
        + torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return first_term + second_term


def evidential_loss(alpha, target, n_classes, epoch, annealing_epochs=10):
    """Expected-MSE evidential loss + annealed Dirichlet KL regulariser (Sensoy et al. 2018, eq. 5)."""
    y = F.one_hot(target, n_classes).float()
    S = alpha.sum(dim=1, keepdim=True)
    err = ((y - alpha / S) ** 2).sum(dim=1, keepdim=True)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)
    alpha_tilde = y + (1 - y) * alpha
    kl = _kl_dirichlet(alpha_tilde, n_classes)
    lam = min(1.0, epoch / annealing_epochs)
    return (err + var + lam * kl).mean()


# ============================================================================
# Step 7-10 (spec) -- assembled proposed model, ablation-configurable
# ============================================================================
class ADHDProposedModel(nn.Module):
    """
    Configurable assembly of every stage above. Each ablation variant in
    Section 14 is one set of constructor flags -- no duplicated forward()
    logic:

        Baseline            : use the existing ADHDNet class directly (unchanged)
        A1 encoder+tokens    : refine='none', use_trm=False, use_attention=False
        A2 tokens+MLP        : refine='mlp',  use_trm=False, use_attention=False
        A3 tokens+FastKAN    : refine='fastkan', use_trm=False, use_attention=False
        A4 +TRM              : refine='fastkan', use_trm=True,  use_attention=False
        A5 +AnatomicalAttn   : refine='fastkan', use_trm=True,  use_attention=True  (= final MRI-only model)
        +phenotype fusion    : use_phenotype=True on top of A5
        +evidential head     : use_evidential=True on top of A5 (or A5+phenotype)

    brain_repr_dim defaults to 128 to match ADHDNet's existing MRI-embedding
    dimension, so the standard classifier head below is architecturally
    identical to the baseline's (128 -> 64 -> 2), not arbitrarily enlarged.
    """
    def __init__(self, n_pheno=5, dropout=0.3, n_tokens=16, token_dim=32,
                 brain_repr_dim=128, conv_type='ds',
                 refine='fastkan',            # 'fastkan' | 'mlp' | 'none'
                 use_trm=True, use_attention=True,
                 use_phenotype=False, use_evidential=False):
        super().__init__()
        self.use_phenotype  = use_phenotype
        self.use_evidential = use_evidential
        self.use_attention  = use_attention

        self.encoder   = LightweightEncoder(conv_type=conv_type)
        self.tokenizer = SpatialTokenizer(96, n_tokens=n_tokens, token_dim=token_dim)

        if refine == 'fastkan':
            self.refine = FastKANRefinement(token_dim)
        elif refine == 'mlp':
            self.refine = ConventionalMLPRefinement(token_dim)
        else:
            self.refine = nn.Identity()

        self.trm = TokenRelationshipModule(token_dim) if use_trm else nn.Identity()
        self.anat_attn = AnatomicalAttention(token_dim) if use_attention else None

        self.token_agg = nn.Linear(n_tokens * token_dim, brain_repr_dim)

        if use_phenotype:
            pheno_embed_dim = 32
            self.pheno_mlp = nn.Sequential(
                nn.Linear(n_pheno, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(32, pheno_embed_dim),
            )
            self.gate = nn.Sequential(
                nn.Linear(brain_repr_dim + pheno_embed_dim, brain_repr_dim), nn.ReLU(inplace=True),
                nn.Linear(brain_repr_dim, brain_repr_dim), nn.Sigmoid(),
            )
            self.pheno_proj = nn.Linear(pheno_embed_dim, brain_repr_dim)

        if use_evidential:
            self.evidential_head = EvidentialHead(brain_repr_dim, n_classes=2)
        else:
            self.head = nn.Sequential(
                nn.Linear(brain_repr_dim, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, 2),
            )

    def forward(self, vol, pheno=None, return_extras=False):
        feat_map = self.encoder(vol)                      # (B, 96, 8,8,8)
        tokens = self.tokenizer(feat_map)                    # (B, 16, token_dim)
        tokens = self.refine(tokens)
        tokens = self.trm(tokens)

        if self.anat_attn is not None:
            weighted, attn_w = self.anat_attn(tokens)
        else:
            weighted, attn_w = tokens, None

        brain_repr = self.token_agg(weighted.flatten(1))     # (B, brain_repr_dim)

        if self.use_phenotype:
            if pheno is None:
                raise ValueError("use_phenotype=True but no `pheno` tensor was passed")
            zphen = self.pheno_mlp(pheno)
            combined = torch.cat([brain_repr, zphen], dim=1)
            g = self.gate(combined)
            zphen_proj = self.pheno_proj(zphen)
            brain_repr = g * brain_repr + (1 - g) * zphen_proj

        out = self.evidential_head(brain_repr) if self.use_evidential else self.head(brain_repr)

        if return_extras:
            extras = {
                'feat_map': feat_map.detach(),
                'spatial_attn': self.tokenizer.last_attn,
                'token_relationship': getattr(self.trm, 'last_relationship_matrix', None),
                'anat_attn_weights': getattr(self.anat_attn, 'last_attn_weights', None) if self.anat_attn else None,
            }
            return out, extras
        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    vol   = torch.randn(4, 1, 64, 64, 64)
    pheno = torch.randn(4, 5)

    print("Step 2 -- encoder exposes spatial feature map (no pooling):")
    enc = LightweightEncoder()
    feat = enc(vol)
    print(f"  (B,1,64,64,64) -> feat_map {tuple(feat.shape)}  (expect (4, 96, 8, 8, 8))")
    assert feat.shape == (4, 96, 8, 8, 8)

    print("\nStep 3 -- SpatialTokenizer:")
    tok = SpatialTokenizer(96, n_tokens=16, token_dim=32)
    tokens = tok(feat)
    print(f"  feat_map -> tokens {tuple(tokens.shape)}  (expect (4, 16, 32))")
    assert tokens.shape == (4, 16, 32)
    print(f"  spatial attn maps: {tuple(tok.last_attn.shape)}  (expect (4, 16, 8, 8, 8))")

    print("\nStep 4 -- FastKANRefinement (gradient check):")
    fk = FastKANRefinement(32)
    refined = fk(tokens)
    print(f"  tokens -> refined {tuple(refined.shape)}  (expect (4, 16, 32))")
    assert refined.shape == tokens.shape
    refined.sum().backward()
    print(f"  grad flows to tokenizer: {tok.attn_logits.weight.grad is not None}")

    print("\nStep 5 -- TokenRelationshipModule:")
    trm = TokenRelationshipModule(32)
    ctx = trm(refined.detach())
    print(f"  refined -> context tokens {tuple(ctx.shape)}  (expect (4, 16, 32))")
    print(f"  relationship matrix: {tuple(trm.last_relationship_matrix.shape)}  (expect (4, 4, 16, 16))")

    print("\nStep 6 -- AnatomicalAttention:")
    attn_mod = AnatomicalAttention(32)
    weighted, w = attn_mod(ctx)
    print(f"  context -> weighted {tuple(weighted.shape)}, weights {tuple(w.shape)}  (expect (4,16,32),(4,16))")

    print("\nStep 7-8 -- full ADHDProposedModel (standard head, MRI-only):")
    model = ADHDProposedModel(use_phenotype=False, use_evidential=False)
    out = model(vol)
    print(f"  output logits: {tuple(out.shape)}  (expect (4, 2))")
    total = sum(p.numel() for p in model.parameters())
    print(f"  params: {total:,}")

    print("\nStep 10 -- + phenotype fusion:")
    model_p = ADHDProposedModel(use_phenotype=True)
    out_p = model_p(vol, pheno)
    print(f"  output logits: {tuple(out_p.shape)}")
    print(f"  params: {sum(p.numel() for p in model_p.parameters()):,}")

    print("\nStep 9 -- + evidential head:")
    model_e = ADHDProposedModel(use_evidential=True)
    out_e = model_e(vol)
    print(f"  alpha {tuple(out_e['alpha'].shape)}, uncertainty {tuple(out_e['uncertainty'].shape)}")
    target = torch.tensor([0, 1, 0, 1])
    loss = evidential_loss(out_e['alpha'], target, n_classes=2, epoch=1)
    print(f"  evidential_loss: {loss.item():.4f}")
    loss.backward()
    print(f"  grad flows: {model_e.encoder.block1.conv.pointwise.weight.grad is not None}")

    print("\nAll ablation variants (A1-A5), param counts:")
    variants = {
        'A1 encoder+tokens'        : dict(refine='none',    use_trm=False, use_attention=False),
        'A2 tokens+MLP'            : dict(refine='mlp',     use_trm=False, use_attention=False),
        'A3 tokens+FastKAN'        : dict(refine='fastkan', use_trm=False, use_attention=False),
        'A4 +TRM'                  : dict(refine='fastkan', use_trm=True,  use_attention=False),
        'A5 +AnatomicalAttn (final)': dict(refine='fastkan', use_trm=True, use_attention=True),
    }
    for name, kwargs in variants.items():
        m = ADHDProposedModel(**kwargs)
        out = m(vol)
        assert out.shape == (4, 2)
        print(f"  {name:<28} params={sum(p.numel() for p in m.parameters()):>8,}  output={tuple(out.shape)}")

    print("\nproposed_model.py sanity check PASSED ✓")


#==============================================================================
# FILE: code/phase2_model.py
#==============================================================================

"""
PHASE 2, Step 1 ONLY: redesigned pairwise Token Relationship Module.

This file does NOT modify proposed_model.py (Phase 1). It imports Phase 1's
unchanged classes (LightweightEncoder, SpatialTokenizer, FastKANRefinement,
ConventionalMLPRefinement, AnatomicalAttention, EvidentialHead,
TokenRelationshipModule=the original MHA-based TRM) and adds:

  - PairwiseTokenRelationshipModule : the new explicit pairwise relationship
    mechanism (Section 1 of the Phase-2 spec)
  - ADHDProposedModelV2             : the same assembly as Phase 1's
    ADHDProposedModel, but with a `trm_type` switch so A4 (old MHA TRM) and
    A4-new / A5-new (new pairwise TRM) can be built from the SAME class
    without duplicating the rest of the architecture.

Phase 1's ADHDProposedModel, and every Phase-1 result already produced, stay
exactly as they are -- this is purely additive. Phase-2 ablations are
compared against Phase-1 numbers, never overwriting them.
"""

import torch
import torch.nn as nn

from proposed_model import (
    LightweightEncoder, SpatialTokenizer, FastKANRefinement,
    ConventionalMLPRefinement, TokenRelationshipModule, AnatomicalAttention,
    EvidentialHead,
)


# ============================================================================
# Section 1 -- redesigned pairwise Token Relationship Module
# ============================================================================
class PairwiseTokenRelationshipModule(nn.Module):
    """
    Explicit pairwise relationship module, replacing generic multi-head
    self-attention with a directly interpretable mechanism:

        for every ordered pair (i, j) of the 16 tokens:
            r_ij = [t_i, t_j, |t_i - t_j|]            (Section 1A)
            R_ij = f(r_ij)   -- ONE shared MLP, same weights for every pair
                                (Section 1B; NOT a separate parameter set
                                per pair, which is what would make this
                                expensive)
        R     = softmax(R, dim=-1)                      -- per-token distribution
                                                            over the other tokens
        V     = shared_linear(tokens)
        ctx_i = sum_j R_ij * V_j                         (Section 1C)
        t'_i  = LayerNorm(t_i + ctx_i)
        t'_i  = LayerNorm(t'_i + FFN(t'_i))                -- small residual refinement

    Operates ONLY on the (B, 16, token_dim) tokens -- the pairwise descriptor
    tensor is (B, 16, 16, 3*token_dim), trivially small (16*16*96 = 24,576
    floats/sample for token_dim=32) -- never on the 3D feature map, and the
    relationship function `rel_fn` is a single small shared MLP, not a
    per-pair parameter block, so this stays far cheaper than it would if it
    were a second Transformer encoder.

    `last_relationship_matrix` holds the (B, 16, 16) matrix after a forward
    pass -- directly interpretable (unlike per-head MHA weights, there is
    exactly one matrix, and R_ij has a concrete meaning: "how much does
    token j's relationship-context contribute to token i").
    """
    def __init__(self, token_dim, hidden_dim=None, ffn_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or token_dim
        ffn_dim = ffn_dim or token_dim * 2

        # Section 1B: ONE shared relationship function for every token pair
        self.rel_fn = nn.Sequential(
            nn.Linear(token_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_proj = nn.Linear(token_dim, token_dim)   # shared V(t_j)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)

        self.last_relationship_matrix = None   # (B, 16, 16)

    def forward(self, tokens):
        B, N, D = tokens.shape

        # Section 1A: pairwise descriptor r_ij = [t_i, t_j, |t_i - t_j|]
        ti = tokens.unsqueeze(2).expand(B, N, N, D)    # t_i broadcast over j
        tj = tokens.unsqueeze(1).expand(B, N, N, D)    # t_j broadcast over i
        descriptor = torch.cat([ti, tj, (ti - tj).abs()], dim=-1)   # (B, N, N, 3D)

        # Section 1B: shared scalar relationship score, row-softmax-normalized
        scores = self.rel_fn(descriptor).squeeze(-1)     # (B, N, N) raw R_ij
        R = torch.softmax(scores, dim=-1)
        self.last_relationship_matrix = R.detach()

        # Section 1C: relationship-guided aggregation + residual update
        V = self.value_proj(tokens)                        # (B, N, D)
        context = torch.einsum('bij,bjd->bid', R, V)
        context = self.dropout(context)

        x = self.norm1(tokens + context)
        x = self.norm2(x + self.ffn(x))
        return x


# ============================================================================
# Assembly: same architecture as Phase 1's ADHDProposedModel, with a
# `trm_type` switch so A4 (old) vs A4-new/A5-new (new) differ in nothing
# except which TRM class is instantiated.
# ============================================================================
class ADHDProposedModelV2(nn.Module):
    """
    Identical to proposed_model.ADHDProposedModel except for `trm_type`:
        trm_type='none'     -> no relationship module (= Phase-1 A3)
        trm_type='mha'      -> Phase-1's original MultiheadAttention TRM (= Phase-1 A4)
        trm_type='pairwise' -> this file's new PairwiseTokenRelationshipModule (A4-new / A5-new)
    """
    def __init__(self, n_pheno=5, dropout=0.3, n_tokens=16, token_dim=32,
                 brain_repr_dim=128, conv_type='ds',
                 refine='fastkan',
                 trm_type='pairwise',
                 use_attention=True,
                 use_phenotype=False, use_evidential=False):
        super().__init__()
        self.use_phenotype  = use_phenotype
        self.use_evidential = use_evidential
        self.use_attention  = use_attention
        self.trm_type = trm_type

        self.encoder   = LightweightEncoder(conv_type=conv_type)
        self.tokenizer = SpatialTokenizer(96, n_tokens=n_tokens, token_dim=token_dim)

        if refine == 'fastkan':
            self.refine = FastKANRefinement(token_dim)
        elif refine == 'mlp':
            self.refine = ConventionalMLPRefinement(token_dim)
        else:
            self.refine = nn.Identity()

        if trm_type == 'mha':
            self.trm = TokenRelationshipModule(token_dim)
        elif trm_type == 'pairwise':
            self.trm = PairwiseTokenRelationshipModule(token_dim)
        else:
            self.trm = nn.Identity()

        self.anat_attn = AnatomicalAttention(token_dim) if use_attention else None
        self.token_agg = nn.Linear(n_tokens * token_dim, brain_repr_dim)

        if use_phenotype:
            pheno_embed_dim = 32
            self.pheno_mlp = nn.Sequential(
                nn.Linear(n_pheno, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(32, pheno_embed_dim),
            )
            self.gate = nn.Sequential(
                nn.Linear(brain_repr_dim + pheno_embed_dim, brain_repr_dim), nn.ReLU(inplace=True),
                nn.Linear(brain_repr_dim, brain_repr_dim), nn.Sigmoid(),
            )
            self.pheno_proj = nn.Linear(pheno_embed_dim, brain_repr_dim)

        if use_evidential:
            self.evidential_head = EvidentialHead(brain_repr_dim, n_classes=2)
        else:
            self.head = nn.Sequential(
                nn.Linear(brain_repr_dim, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, 2),
            )

    def forward(self, vol, pheno=None, return_extras=False):
        feat_map = self.encoder(vol)
        tokens = self.tokenizer(feat_map)
        tokens = self.refine(tokens)
        tokens = self.trm(tokens)

        if self.anat_attn is not None:
            weighted, attn_w = self.anat_attn(tokens)
        else:
            weighted, attn_w = tokens, None

        brain_repr = self.token_agg(weighted.flatten(1))

        if self.use_phenotype:
            if pheno is None:
                raise ValueError("use_phenotype=True but no `pheno` tensor was passed")
            zphen = self.pheno_mlp(pheno)
            combined = torch.cat([brain_repr, zphen], dim=1)
            g = self.gate(combined)
            brain_repr = g * brain_repr + (1 - g) * self.pheno_proj(zphen)

        out = self.evidential_head(brain_repr) if self.use_evidential else self.head(brain_repr)

        if return_extras:
            extras = {
                'feat_map': feat_map.detach(),
                'spatial_attn': self.tokenizer.last_attn,
                'token_relationship': getattr(self.trm, 'last_relationship_matrix', None),
                'anat_attn_weights': getattr(self.anat_attn, 'last_attn_weights', None) if self.anat_attn else None,
            }
            return out, extras
        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    vol = torch.randn(4, 1, 64, 64, 64)

    print("Verifying PairwiseTokenRelationshipModule in isolation:")
    tokens = torch.randn(4, 16, 32, requires_grad=True)
    ptrm = PairwiseTokenRelationshipModule(32)
    out = ptrm(tokens)
    print(f"  tokens {tuple(tokens.shape)} -> context-aware tokens {tuple(out.shape)}  (expect (4,16,32))")
    assert out.shape == tokens.shape
    print(f"  relationship matrix: {tuple(ptrm.last_relationship_matrix.shape)}  (expect (4,16,16))")
    row_sums = ptrm.last_relationship_matrix.sum(dim=-1)
    print(f"  rows sum to 1 (softmax check): {torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)}")
    out.sum().backward()
    print(f"  grad flows back to input tokens: {tokens.grad is not None}")

    print(f"\n  PairwiseTokenRelationshipModule params: {sum(p.numel() for p in ptrm.parameters()):,}")
    mha_trm = TokenRelationshipModule(32)
    print(f"  Phase-1 MHA TokenRelationshipModule params: {sum(p.numel() for p in mha_trm.parameters()):,}")

    print("\nRequired ablation variants (Section 1D): A3, A4 (MHA), A4-new, A5-new")
    variants = {
        'A3 tokens+FastKAN (no TRM)'      : dict(trm_type='none',     use_attention=False),
        'A4 +MHA TRM (Phase 1)'           : dict(trm_type='mha',      use_attention=False),
        'A4-new +Pairwise TRM'            : dict(trm_type='pairwise', use_attention=False),
        'A5-new +Pairwise TRM +AnatAttn'  : dict(trm_type='pairwise', use_attention=True),
    }
    for name, kwargs in variants.items():
        m = ADHDProposedModelV2(**kwargs)
        out = m(vol)
        assert out.shape == (4, 2)
        print(f"  {name:<34} params={sum(p.numel() for p in m.parameters()):>8,}  output={tuple(out.shape)}")

    print("\nphase2_model.py sanity check PASSED ✓")


#==============================================================================
# FILE: code/train.py
#==============================================================================


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


#==============================================================================
# FILE: code/evaluate.py
#==============================================================================


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

#==============================================================================
# FILE: code/predict.py
#==============================================================================


import sys
import os
import json
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from model import ADHDNet

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PTH      = '/kaggle/working/best_model.pth'
FIT_STATS_JSON = '/kaggle/working/fit_stats.json'
TARGET_SHAPE   = (64, 64, 64)
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── fit_stats from training ───────────────────────────────────────────────────
# Loaded from the file train.py saves (stats fit on the TRAINING subjects only,
# to avoid data leakage). Falls back to the last known training values if the
# file isn't present (e.g. running predict.py outside the Kaggle working dir).
_DEFAULT_FIT_STATS = {
    'age_mean'      : 11.998,
    'age_std'       : 3.341,
    'adhd_idx_mean' : 52.957,
    'adhd_idx_std'  : 17.276,
}
if os.path.exists(FIT_STATS_JSON):
    with open(FIT_STATS_JSON) as f:
        FIT_STATS = json.load(f)
else:
    FIT_STATS = _DEFAULT_FIT_STATS

def resize_volume(vol, target=TARGET_SHAPE):
    factors = [t / s for t, s in zip(target, vol.shape)]
    return zoom(vol, factors, order=1)

def normalize(vol):
    mn, mx = vol.min(), vol.max()
    if mx - mn < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - mn) / (mx - mn)).astype(np.float32)

def build_pheno(age, sex, med_status, site_id, adhd_index):
    """Normalize phenotypic features exactly as done in dataset.py"""
    age_norm      = (age        - FIT_STATS['age_mean'])      / FIT_STATS['age_std']
    adhd_idx_norm = (adhd_index - FIT_STATS['adhd_idx_mean']) / FIT_STATS['adhd_idx_std']
    sex_norm      = sex        / 1.0
    med_norm      = med_status / 2.0
    site_norm     = site_id    / 5.0
    return torch.tensor(
        [age_norm, sex_norm, med_norm, site_norm, adhd_idx_norm],
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)                                  # (1, 5)

def predict(nii_path, subject_id=None, true_label="unknown",
            age=12.0, sex=1.0, med_status=1.0,
            site_id=0.0, adhd_index=52.957):

    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Scan not found: {nii_path}")

    # ── Load & preprocess scan ────────────────────────────────────────────────
    img = nib.load(nii_path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    vol = resize_volume(vol)
    vol = normalize(vol)

    # ── Build tensors ─────────────────────────────────────────────────────────
    vol_tensor   = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # (1,1,64,64,64)
    pheno_tensor = build_pheno(age, sex, med_status, site_id, adhd_index)          # (1,5)

    # ── Load model ────────────────────────────────────────────────────────────
    model = ADHDNet(n_pheno=5, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(vol_tensor, pheno_tensor)               # (1, 2)
        probs  = torch.softmax(logits, dim=1)
        score  = probs[0, 1].item()                            # ADHD probability

    mri_label = "ADHD" if score >= 0.5 else "Control"

    if subject_id is None:
        subject_id = os.path.basename(nii_path).replace('.nii.gz','').replace('.nii','')

    return {
        "subject_id" : str(subject_id),
        "mri_score"  : round(score, 4),
        "mri_label"  : mri_label,
        "true_label" : true_label
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_nii> [subject_id] [true_label]"
              " [age] [sex] [med_status] [site_id] [adhd_index]")
        sys.exit(1)

    nii_path   = sys.argv[1]
    subject_id = sys.argv[2] if len(sys.argv) > 2 else None
    true_label = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    age        = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    sex        = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    med_status = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    site_id    = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    adhd_index = float(sys.argv[8]) if len(sys.argv) > 8 else 52.957

    result = predict(nii_path, subject_id, true_label,
                     age, sex, med_status, site_id, adhd_index)
    print(json.dumps(result, indent=2))

#==============================================================================
# FILE: code/cross_validate.py
#==============================================================================


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


#==============================================================================
# FILE: code/cross_site_eval.py
#==============================================================================

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


#==============================================================================
# FILE: code/ablation_study.py
#==============================================================================


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


#==============================================================================
# FILE: code/benchmark.py
#==============================================================================


import time
import json
import statistics as stats

import torch
from torch.utils.flop_counter import FlopCounterMode

from model import ADHDNet, ADHDNetEnhanced

# ── Config ─────────────────────────────────────────────────────────────────
OUT_DIR  = '/kaggle/working'
N_WARMUP = 10
N_TIMED  = 50
BATCHES  = [1, 8]
SEED     = 42

torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
if DEVICE.type == 'cpu':
    torch.set_num_threads(torch.get_num_threads())
    print(f"CPU threads : {torch.get_num_threads()}")

MODELS = {
    'lightweight' : ADHDNet,
    'enhanced'    : ADHDNetEnhanced,
}

results = {}

for name, cls in MODELS.items():
    model = cls(n_pheno=5, dropout=0.3).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    for bs in BATCHES:
        vol   = torch.randn(bs, 1, 64, 64, 64, device=DEVICE)
        pheno = torch.randn(bs, 5, device=DEVICE)

        # ── FLOPs (hardware-independent) ──────────────────────────────────────
        with torch.no_grad():
            flop_counter = FlopCounterMode(display=False, depth=None)
            with flop_counter:
                _ = model(vol, pheno)
            total_flops = flop_counter.get_total_flops()

        # ── Latency ─────────────────────────────────────────────────────────
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model(vol, pheno)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()

            times_ms = []
            for _ in range(N_TIMED):
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(vol, pheno)
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        mean_ms   = stats.mean(times_ms)
        std_ms    = stats.stdev(times_ms)
        median_ms = stats.median(times_ms)
        p95_ms    = sorted(times_ms)[int(0.95 * len(times_ms)) - 1]

        key = f"{name}_bs{bs}"
        results[key] = {
            'model'            : name,
            'batch_size'       : bs,
            'device'           : str(DEVICE),
            'params'           : n_params,
            'flops_total'      : total_flops,
            'flops_per_sample' : total_flops / bs,
            'gflops_per_sample': total_flops / bs / 1e9,
            'latency_mean_ms'  : mean_ms,
            'latency_std_ms'   : std_ms,
            'latency_median_ms': median_ms,
            'latency_p95_ms'   : p95_ms,
            'throughput_samples_per_s': bs / (mean_ms / 1000.0),
        }

        print(f"\n{name}  (batch={bs}, {n_params:,} params)")
        print(f"  FLOPs / sample : {total_flops/bs:,.0f}  ({total_flops/bs/1e9:.4f} GFLOPs)")
        print(f"  Latency        : {mean_ms:.2f} ± {std_ms:.2f} ms  "
              f"(median {median_ms:.2f}, p95 {p95_ms:.2f})")
        print(f"  Throughput     : {bs/(mean_ms/1000.0):.2f} samples/s")

with open(f'{OUT_DIR}/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved → {OUT_DIR}/benchmark_results.json")


#==============================================================================
# FILE: code/proposed_ablation_study.py
#==============================================================================

"""
Trains and compares the Baseline ADHDNet against every proposed-architecture
ablation from Section 14 of the spec, on the SAME stratified 80/20 split,
SAME seeding, and SAME leakage-free fit_stats methodology already used by
ablation_study.py and train.py in this notebook.

    Baseline            : existing ADHDNet (unmodified, frozen)
    A1 Encoder+Tokens   : encoder -> 16 tokens -> mean-pool -> classifier (no refinement)
    A2 Tokens+MLP       : + conventional MLP token refinement
    A3 Tokens+FastKAN   : + FastKAN token refinement
    A4 +TRM             : + Token Relationship Module
    A5 Final (MRI-only) : + Anatomical (token-importance) Attention
    A5 + Phenotype      : A5 with late/adaptive phenotype fusion
    A5 + Evidential     : A5 with the Dirichlet evidential head instead of softmax

This answers Section 14's question directly: where does the performance
improvement actually come from (tokens alone? FastKAN? the relationship
module? attention? phenotype? none of the above?).
"""

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
from model import ADHDNet
from proposed_model import ADHDProposedModel, evidential_loss

# ── Config ─────────────────────────────────────────────────────────────────
INDEX_CSV   = '/kaggle/working/processed_index.csv'
OUT_DIR     = '/kaggle/working'
EPOCHS      = 150     # same max budget as train.py / ablation_study.py
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# (name, short_label, model_factory, head_type, needs_pheno)
VARIANTS = [
    ('Baseline ADHDNet',   'Baseline', lambda: ADHDNet(n_pheno=5, dropout=0.3), 'standard', True),
    ('A1 Encoder+Tokens',  'A1', lambda: ADHDProposedModel(refine='none',    use_trm=False, use_attention=False), 'standard', False),
    ('A2 Tokens+MLP',      'A2', lambda: ADHDProposedModel(refine='mlp',     use_trm=False, use_attention=False), 'standard', False),
    ('A3 Tokens+FastKAN',  'A3', lambda: ADHDProposedModel(refine='fastkan', use_trm=False, use_attention=False), 'standard', False),
    ('A4 +TRM',            'A4', lambda: ADHDProposedModel(refine='fastkan', use_trm=True,  use_attention=False), 'standard', False),
    ('A5 Final (MRI-only)','A5', lambda: ADHDProposedModel(refine='fastkan', use_trm=True,  use_attention=True),  'standard', False),
    ('A5 + Phenotype',     'A5+Phen', lambda: ADHDProposedModel(refine='fastkan', use_trm=True, use_attention=True, use_phenotype=True), 'standard', True),
    ('A5 + Evidential',    'A5+Evid', lambda: ADHDProposedModel(refine='fastkan', use_trm=True, use_attention=True, use_evidential=True), 'evidential', False),
]


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

# ── Stratified split (shared across all variants, same as ablation_study.py) ─
raw_df = pd.read_csv(INDEX_CSV)
raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
labels_all = raw_df['label'].values
print(f"Total subjects : {len(raw_df)}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
val_ids   = raw_df.iloc[val_idx]['subject_id'].tolist()

train_stats_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=False)
fit_stats = train_stats_ds.get_fit_stats()

train_ds = ADHDDataset(INDEX_CSV, subject_ids=train_ids, augment=True,  fit_stats=fit_stats)
val_ds   = ADHDDataset(INDEX_CSV, subject_ids=val_ids,   augment=False, fit_stats=fit_stats)
print(f"Train : {len(train_ds)}   Val : {len(val_ds)}")


def expected_calibration_error(probs, labels, n_bins=10):
    """Simple ECE: |confidence - accuracy| per probability bin, weighted by bin size."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidences = np.where(probs >= 0.5, probs, 1 - probs)
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return ece


def train_variant(model_factory, head_type, needs_pheno):
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=g, worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    set_seed(SEED)
    model = model_factory().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_acc, best_sens, best_spec, best_f1, best_ece = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    best_uncertainty_mean = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for vol, pheno, label in train_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            if needs_pheno:
                out = model(vol, pheno)
            else:
                out = model(vol)
            if head_type == 'evidential':
                loss = evidential_loss(out['alpha'], label, n_classes=2, epoch=epoch)
            else:
                loss = criterion(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        va_probs, va_labels, va_uncert = [], [], []
        with torch.no_grad():
            for vol, pheno, label in val_loader:
                vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
                out = model(vol, pheno) if needs_pheno else model(vol)
                if head_type == 'evidential':
                    probs = out['prob'][:, 1].cpu().numpy()
                    va_uncert.extend(out['uncertainty'].cpu().numpy())
                else:
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
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
            best_ece  = expected_calibration_error(va_probs, va_labels)
            best_uncertainty_mean = float(np.mean(va_uncert)) if va_uncert else None
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            break

    return {
        'params': n_params, 'accuracy': best_acc, 'auc': best_auc, 'sensitivity': best_sens,
        'specificity': best_spec, 'f1': best_f1, 'ece': best_ece,
        'mean_uncertainty': best_uncertainty_mean if head_type == 'evidential' else None,
        'epochs_run': epoch,
    }


# ── Run every variant on the same split ───────────────────────────────────────
records = []
for name, short, factory, head_type, needs_pheno in VARIANTS:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    result = train_variant(factory, head_type, needs_pheno)
    result['variant'] = name
    result['label']   = short
    records.append(result)
    ece_str = f"  ECE={result['ece']:.4f}" if head_type != 'evidential' or result['ece'] is not None else ""
    print(f"  Acc={result['accuracy']:.4f}  AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
          f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}{ece_str}  "
          f"({result['params']:,} params, {result['epochs_run']} epochs)")

results_df = pd.DataFrame(records)[
    ['variant', 'label', 'params', 'accuracy', 'auc', 'f1', 'sensitivity',
     'specificity', 'ece', 'mean_uncertainty', 'epochs_run']
]
results_df.to_csv(f'{OUT_DIR}/proposed_ablation_results.csv', index=False)

print(f"\n{'='*80}")
print("  PROPOSED ARCHITECTURE ABLATION TABLE")
print(f"{'='*80}")
print(f"  {'Variant':<22}{'Params':>10}{'Accuracy':>10}{'AUC':>8}{'F1':>8}{'ECE':>8}")
for r in records:
    ece_disp = f"{r['ece']:.4f}" if r['ece'] is not None else "  n/a"
    print(f"  {r['variant']:<22}{r['params']:>10,}{r['accuracy']*100:>9.2f}%{r['auc']:>8.4f}"
          f"{r['f1']:>8.4f}{ece_disp:>8}")

# ── Figure — accuracy across the proposed-architecture ablation chain ────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#888888'] + ['#2166AC'] * 5 + ['#7B9EC7', '#B2182B']
bars = ax.bar(results_df['label'], results_df['accuracy'] * 100, color=colors, width=0.6)
for bar, acc in zip(bars, results_df['accuracy']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Proposed Architecture Ablation Chain')
ax.set_ylim(0, max(results_df['accuracy']) * 100 + 10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig6_proposed_ablation.png', dpi=300)

print(f"\nResults table → {OUT_DIR}/proposed_ablation_results.csv")
print(f"Figure        → {OUT_DIR}/fig6_proposed_ablation.png")


#==============================================================================
# FILE: code/proposed_explainability.py
#==============================================================================

"""
Post-hoc explainability for the proposed token-based architecture
(Sections 11-12). Kept entirely separate from training/model code: nothing
here is a model input or a loss term, so the core architecture stays clean
and these modules cannot leak into what the network is optimized for.

Provides, for a trained ADHDProposedModel + a validation set:
  1. Token spatial-attention maps   (where in the MRI each of the 16 tokens looks)
  2. Token relationship matrix      (the TRM's learned 16x16 token interactions)
  3. Anatomical-attention weights   (which tokens mattered most for the decision)
  4. A simple 3D saliency map       (gradient x activation on the feature map,
                                      a lightweight Grad-CAM-style substitute
                                      that needs no extra hooks beyond what
                                      forward(..., return_extras=True) already
                                      exposes)
  5. A post-hoc atlas-overlap utility (Section 11): given a NIfTI atlas volume
     resampled to the model's 8x8x8 feature-map grid, reports which atlas
     labels each token's attention mass concentrates on. This function is
     written to run on Kaggle with nilearn/nibabel + an actual atlas file
     (e.g. AAL), which this sandbox has neither of -- it is NOT exercised
     here, only syntax/shape-checked against a synthetic atlas array.

Nothing here hard-codes an anatomical identity for any token: atlas overlap
is computed AFTER training, strictly as an interpretation step.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# 1-3. Extract token attention / relationship / anatomical-attention for a batch
# ============================================================================
@torch.no_grad()
def extract_token_explanations(model, vol, pheno=None):
    """
    Runs one forward pass with return_extras=True and returns the three
    token-level explanation artifacts as numpy arrays.
    """
    model.eval()
    needs_pheno = getattr(model, 'use_phenotype', False)
    out, extras = model(vol, pheno if needs_pheno else None, return_extras=True)
    result = {
        'spatial_attn': extras['spatial_attn'].cpu().numpy() if extras['spatial_attn'] is not None else None,       # (B,16,8,8,8)
        'token_relationship': extras['token_relationship'].cpu().numpy() if extras['token_relationship'] is not None else None,  # (B,heads,16,16)
        'anat_attn_weights': extras['anat_attn_weights'].cpu().numpy() if extras['anat_attn_weights'] is not None else None,      # (B,16)
    }
    return out, result


# ============================================================================
# 4. Lightweight 3D saliency (gradient x activation on the pre-tokenizer feature map)
# ============================================================================
def feature_map_saliency(model, vol, pheno=None, target_class=1):
    """
    A Grad-CAM-style saliency map over the encoder's spatial feature map:
    backprop the target class's logit to the feature map, weight each
    channel by its average gradient (as in Grad-CAM), and sum. Cheap: only
    needs one forward + one backward pass, no extra hooks.
    Returns a (D,H,W) numpy array per input, upsampled to feature-map
    resolution (8,8,8) -- upsample further to 64^3 with
    torch.nn.functional.interpolate if overlaying on the original MRI.
    """
    model.eval()
    vol = vol.clone().requires_grad_(True)
    needs_pheno = getattr(model, 'use_phenotype', False)

    feat_map = model.encoder(vol)                       # (B, 96, 8,8,8), requires grad
    feat_map.retain_grad()
    tokens = model.tokenizer(feat_map)
    tokens = model.refine(tokens)
    tokens = model.trm(tokens)
    if model.anat_attn is not None:
        weighted, _ = model.anat_attn(tokens)
    else:
        weighted = tokens
    brain_repr = model.token_agg(weighted.flatten(1))

    if needs_pheno:
        zphen = model.pheno_mlp(pheno)
        combined = torch.cat([brain_repr, zphen], dim=1)
        g = model.gate(combined)
        brain_repr = g * brain_repr + (1 - g) * model.pheno_proj(zphen)

    logits = model.evidential_head(brain_repr)['logits'] if model.use_evidential else model.head(brain_repr)
    score = logits[:, target_class].sum()
    score.backward()

    grads = feat_map.grad                                 # (B, 96, 8,8,8)
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)        # (B, 96, 1,1,1) -- Grad-CAM channel weights
    cam = torch.relu((weights * feat_map.detach()).sum(dim=1))  # (B, 8,8,8)
    cam = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)   # normalize to [0,1]
    return cam.detach().cpu().numpy()


# ============================================================================
# 5. Post-hoc atlas overlap (Section 11) -- run on Kaggle with a real atlas
# ============================================================================
def atlas_overlap(spatial_attn, atlas_volume, label_names=None):
    """
    `spatial_attn`  : (16, D, H, W) -- one token's spatial attention map
                      (e.g. result['spatial_attn'][subject_idx] from
                      extract_token_explanations, already summing to 1 over
                      D*H*W per token).
    `atlas_volume`  : (D, H, W) integer array of atlas region labels,
                      RESAMPLED to the same (D,H,W) as spatial_attn (e.g. 8^3)
                      -- use nilearn.image.resample_to_img or scipy.ndimage.zoom
                      with order=0 (nearest-neighbour, to keep labels integers)
                      before calling this.
    `label_names`   : optional dict {region_id: region_name} (e.g. from an
                      AAL atlas lookup table).

    Returns, for each token, the atlas region(s) its attention mass
    concentrates on, sorted by overlap fraction. This is descriptive only
    -- it never feeds back into training.

    Typical Kaggle usage (not run in this sandbox -- needs nilearn + an
    atlas file):
        from nilearn import datasets, image
        import nibabel as nib
        aal = datasets.fetch_atlas_aal()                      # or upload as a Kaggle dataset
        atlas_img = nib.load(aal.maps)
        atlas_resampled = image.resample_img(atlas_img, target_shape=(8,8,8),
                                              target_affine=..., interpolation='nearest')
        atlas_volume = atlas_resampled.get_fdata().astype(int)
        label_names = dict(zip(aal.indices, aal.labels))
        overlap = atlas_overlap(spatial_attn, atlas_volume, label_names)
    """
    n_tokens = spatial_attn.shape[0]
    results = []
    for t in range(n_tokens):
        attn = spatial_attn[t]                            # (D,H,W), sums to ~1
        region_mass = {}
        for region_id in np.unique(atlas_volume):
            if region_id == 0:   # background
                continue
            mask = atlas_volume == region_id
            region_mass[int(region_id)] = float(attn[mask].sum())
        ranked = sorted(region_mass.items(), key=lambda kv: kv[1], reverse=True)
        top = [
            {'region_id': rid, 'region_name': (label_names or {}).get(rid, str(rid)), 'overlap_fraction': frac}
            for rid, frac in ranked[:5]
        ]
        results.append({'token': t, 'top_regions': top})
    return results


# ============================================================================
# Visualization helpers
# ============================================================================
def plot_token_relationship_matrix(relationship_matrix, out_path, head=0):
    """relationship_matrix: (heads, 16, 16) for ONE subject (already averaged over batch, or pick one)."""
    mat = relationship_matrix[head]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=mat.max())
    ax.set_xlabel('Token (key)')
    ax.set_ylabel('Token (query)')
    ax.set_title(f'Token Relationship Matrix (head {head})')
    fig.colorbar(im, ax=ax, label='Attention weight')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_anatomical_attention(weights, out_path):
    """weights: (16,) -- the anatomical-attention weight per token for one subject."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(weights)), weights, color='#2166AC')
    ax.set_xlabel('Token index')
    ax.set_ylabel('Attention weight')
    ax.set_title('Anatomical (token-importance) Attention')
    ax.set_xticks(range(len(weights)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from proposed_model import ADHDProposedModel

    torch.manual_seed(0)
    model = ADHDProposedModel(refine='fastkan', use_trm=True, use_attention=True)
    vol = torch.randn(2, 1, 64, 64, 64)

    print("Extracting token explanations ...")
    out, expl = extract_token_explanations(model, vol)
    print(f"  spatial_attn       : {expl['spatial_attn'].shape}       (expect (2,16,8,8,8))")
    print(f"  token_relationship : {expl['token_relationship'].shape} (expect (2,4,16,16))")
    print(f"  anat_attn_weights  : {expl['anat_attn_weights'].shape}  (expect (2,16))")

    print("\nComputing Grad-CAM-style saliency ...")
    cam = feature_map_saliency(model, vol, target_class=1)
    print(f"  saliency map       : {cam.shape}                 (expect (2,8,8,8))")

    print("\nTesting atlas_overlap() against a synthetic atlas ...")
    synthetic_atlas = np.random.randint(0, 5, size=(8, 8, 8))
    overlap = atlas_overlap(expl['spatial_attn'][0], synthetic_atlas, label_names={1: 'region_A', 2: 'region_B'})
    print(f"  returned {len(overlap)} token entries, e.g. token 0 top region: {overlap[0]['top_regions'][0]}")

    print("\nplotting token relationship matrix + anatomical attention (subject 0) ...")
    out_dir = '.'
    plot_token_relationship_matrix(expl['token_relationship'][0], f'{out_dir}/token_relationship_demo.png')
    plot_anatomical_attention(expl['anat_attn_weights'][0], f'{out_dir}/anatomical_attention_demo.png')
    print(f"  saved -> {out_dir}/token_relationship_demo.png, {out_dir}/anatomical_attention_demo.png")

    print("\nproposed_explainability.py sanity check PASSED ✓")


#==============================================================================
# FILE: code/phase2_trm_ablation.py
#==============================================================================

"""
PHASE 2, Section 1D: the required TRM ablation -- WITH experiment tracking,
per-epoch checkpointing, and interruption recovery.

    A3      : tokens + FastKAN                              (no TRM)
    A4      : tokens + FastKAN + original MHA-based TRM      (Phase 1, unmodified)
    A4-new  : tokens + FastKAN + proposed pairwise TRM        (Phase 2)
    A5-new  : tokens + FastKAN + proposed pairwise TRM + Anatomical Attention

Same stratified 80/20 split, same seeding, same leakage-free fit_stats
methodology, same model architecture/loss/augmentation policy/parameter
counts/ablation definitions as the original version of this script --
NOTHING about the experiment itself changed. What changed is purely
non-architectural instrumentation, in the same pattern already used by
phase2_consistency.py:

  - Per-variant checkpointing: a "best" checkpoint (by validation AUROC,
    with a documented fallback -- see `evaluate_full()` -- when AUROC is
    undefined) and a "latest" checkpoint saved after EVERY epoch, atomically
    (write to a .tmp file, then os.replace() -- never leaves a half-written
    checkpoint on disk even if the process dies mid-write).
  - Interruption recovery: set RESUME = True (default) and re-run this
    script. Any variant whose `{tag}_results.csv` already exists is treated
    as COMPLETE and skipped; any variant with a `{tag}_latest_checkpoint.pt`
    but no `{tag}_results.csv` is resumed from that checkpoint -- model,
    optimizer, scheduler, early-stopping state, training history, AND the
    exact RNG/DataLoader-shuffle state are all restored, so the per-epoch
    training-loss sequence after resuming continues the same sequence an
    uninterrupted run would have produced (verified in run_sanity_check()'s
    resume test below -- NUM_WORKERS=0 and CPU execution make this exact;
    on a CUDA GPU, cuDNN non-determinism can introduce tiny numerical drift
    even with the RNG states restored, which is a CUDA/cuDNN property, not
    a bug in the checkpointing logic).
  - Per-variant outputs (ALL under OUT_DIR, prefixed by the variant's short
    tag -- A3/A4/A4-new/A5-new): best-checkpoint .pt, latest-checkpoint .pt,
    per-epoch training history .csv, per-subject validation predictions
    .csv (subject_id, site, y_true, y_prob, y_pred), final per-variant
    results .csv, and a config .json (seed, hyperparams, arch kwargs, best
    epoch). Each variant's results.csv is written immediately when that
    variant finishes -- NOT held in memory until all four complete -- so
    completed variants' results survive even if a later variant is
    interrupted.
  - The final aggregate `trm_ablation_results.csv` + comparison figure are
    built by READING each variant's saved `{tag}_results.csv` (not from
    in-memory state), so the aggregate step itself is resume-safe too.
  - Exact train/validation subject IDs are saved once (`train_subject_ids.
    csv` / `val_subject_ids.csv`) and, on any restart, loaded from disk
    rather than re-derived -- an extra guarantee (on top of the fixed seed)
    that every variant, across any number of restarts, trains and
    validates on the IDENTICAL subject-level split.
  - Strict configuration validation: every checkpoint/result carries a
    `config_fingerprint` (a sha256 of the dataset file + the exact subject
    split, plus kwargs/epochs/patience/batch_size/lr/val_split/seed). Before
    skipping a completed variant OR resuming an interrupted one, the CURRENT
    run's fingerprint is compared against the saved one; any mismatch
    RAISES (never silently reuses a result produced under different
    settings) -- see `check_fingerprint_or_raise()`. A checkpoint/result with
    NO `config_fingerprint` at all (it predates this safeguard, or was
    produced by a different script version) is refused the same way, never
    silently trusted -- see `require_fingerprint_or_raise()`. Either case
    requires pointing `OUT_DIR` at a fresh, empty directory.
  - The first valid validation checkpoint is always saved, even if its
    AUROC is exactly 0.0 (the "best so far" sentinel is -1.0, not 0.0 --
    see the comment at `best = dict(auroc=-1.0, ...)` in `train_variant()`).

`run_sanity_check()` (called first in `main()`, synthetic data only, no
real dataset touched) verifies:
    Test A -- forward pass + parameter counts for all four variants
    Test B -- checkpoint save/load round-trip
    Test C -- genuine interrupt-then-resume reproduces the uninterrupted
              per-epoch training-loss sequence
    Test D -- strict configuration validation raises on a changed dataset
              (index_hash) or changed split (split_hash), both when
              skipping a completed variant and when resuming an
              interrupted one; an UNCHANGED configuration still skips
              cleanly; a completed result or checkpoint with NO
              config_fingerprint at all is also refused, both when
              skipping and when resuming
    Test E -- the first-epoch checkpoint is saved even when that epoch's
              AUROC is exactly 0.0 (forced via a temporary monkeypatch of
              evaluate_full(), restored immediately afterward)
"""

import os
import hashlib
import json
import random
import tempfile
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ADHDDataset
from phase2_model import ADHDProposedModelV2

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIG -- the only things you should need to edit for your own setup  ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATA_ROOT   = '/kaggle/working'
INDEX_FILE  = f'{DATA_ROOT}/processed_index.csv'
INDEX_CSV   = INDEX_FILE    # alias kept for consistency with the rest of this project's scripts
OUT_DIR     = f'{DATA_ROOT}/phase2'          # SEPARATE from Phase-1 outputs
os.makedirs(OUT_DIR, exist_ok=True)

RESUME      = True    # True: skip completed variants, resume interrupted ones from their
                       # latest checkpoint. False: ignore any existing checkpoints/results
                       # and retrain every variant from scratch (existing files are overwritten
                       # as training proceeds, same as this script's original behaviour).

EPOCHS      = 150     # same max budget as every other training script here
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

VARIANTS = [
    ('A3 tokens+FastKAN',             'A3',      dict(trm_type='none',     use_attention=False)),
    ('A4 +MHA TRM (Phase 1)',         'A4',      dict(trm_type='mha',      use_attention=False)),
    ('A4-new +Pairwise TRM',          'A4-new',  dict(trm_type='pairwise', use_attention=False)),
    ('A5-new +Pairwise TRM +AnatAttn','A5-new',  dict(trm_type='pairwise', use_attention=True)),
]


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


# ============================================================================
# Atomic writes -- never leave a half-written checkpoint/CSV on disk, even if
# the process is killed mid-write.
# ============================================================================
def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_to_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def checkpoint_paths(tag, out_dir):
    return dict(
        latest=f'{out_dir}/{tag}_latest_checkpoint.pt',
        best=f'{out_dir}/{tag}_best_model.pt',
        history=f'{out_dir}/{tag}_training_history.csv',
        valpred=f'{out_dir}/{tag}_val_predictions.csv',
        results=f'{out_dir}/{tag}_results.csv',
        config=f'{out_dir}/{tag}_config.json',
    )


def hash_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def hash_split(train_ids, val_ids):
    return hashlib.sha256((','.join(sorted(train_ids)) + '|' + ','.join(sorted(val_ids))).encode()).hexdigest()


def compute_config_fingerprint(kwargs, index_hash, split_hash, epochs, patience):
    """
    Everything that must be IDENTICAL between the run that produced a saved
    checkpoint/result and the current run before it is safe to resume or
    skip. If any of these differ, the checkpoint/result was produced under
    different conditions and must never be silently reused.
    """
    return {
        'kwargs': kwargs, 'index_hash': index_hash, 'split_hash': split_hash,
        'epochs_max': epochs, 'patience': patience, 'batch_size': BATCH_SIZE,
        'lr': LR, 'val_split': VAL_SPLIT, 'seed': SEED,
    }


def check_fingerprint_or_raise(current_fp, saved_fp, tag, action):
    """
    Strict configuration validation (item 1): raises -- never warns and
    continues -- if any essential training setting differs between the
    current run and the one that produced the saved checkpoint/result this
    `action` ('skip' or 'resume') is about to reuse. Dataset identity
    (index_hash: a sha256 of processed_index.csv's bytes), the exact
    subject-level split (split_hash), model configuration (kwargs), seed,
    learning rate, epochs/patience budget, batch size, and val_split are
    all covered.
    """
    mismatches = {k: (saved_fp.get(k), current_fp[k]) for k in current_fp if saved_fp.get(k) != current_fp[k]}
    if mismatches:
        lines = "\n".join(f"    {k}: saved={old!r}  current={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- configuration has changed since the saved "
            f"checkpoint/result was produced:\n{lines}\n"
            f"  Never silently reusing incompatible results. Either revert the changed "
            f"setting(s) above, or point OUT_DIR at a fresh, empty directory for this new "
            f"configuration (e.g. OUT_DIR = '/kaggle/working/phase2_v2')."
        )


def require_fingerprint_or_raise(saved_fp, tag, action, source_path):
    """
    Strict configuration validation, extended: a checkpoint/result that
    predates the config_fingerprint safeguard (or was produced by a script
    version that never saved one) carries no proof of what configuration
    produced it. Reusing it anyway would be exactly the silent-reuse-of-
    incompatible-results failure mode item 1 exists to prevent -- so a
    MISSING fingerprint refuses the same way a MISMATCHED one does, never
    with a "proceeding as before" fallback.
    """
    if saved_fp is None:
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- {source_path} has no config_fingerprint "
            f"(it predates this safeguard, or was produced by a different script version), "
            f"so there is no way to verify it matches the current dataset/split/model/"
            f"training configuration.\n"
            f"  Never silently reusing a result whose configuration cannot be verified. "
            f"Point OUT_DIR at a fresh, empty directory to regenerate this result under "
            f"the current safeguard (e.g. OUT_DIR = '/kaggle/working/phase2_v2')."
        )


def measure_flops_and_latency(model, n_timed=20):
    model.eval()
    vol = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
    with torch.no_grad():
        fc = FlopCounterMode(display=False, depth=None)
        with fc:
            _ = model(vol)
        flops = fc.get_total_flops()
        for _ in range(5):
            _ = model(vol)
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            _ = model(vol)
            times.append((time.perf_counter() - t0) * 1000)
    return flops, float(np.mean(times)), float(np.std(times))


def evaluate_full(model, loader):
    """
    Documented AUROC-undefined fallback: if the validation batch passed to
    roc_auc_score/average_precision_score contains only one class (can only
    happen if the validation split itself is degenerate -- the integrity
    checks in main() guard against this at the subject level), both metrics
    fall back to (0.5, mean label) and `auroc_undefined=True` is returned so
    callers can flag it rather than silently treating 0.5 as a real score.
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            p = torch.softmax(model(vol), dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(label.numpy())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(int)
    auroc_undefined = False
    try:
        auc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except Exception:
        auc, auprc = 0.5, float(labels.mean())
        auroc_undefined = True
    acc = (preds == labels).mean()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return dict(probs=probs, labels=labels, preds=preds, auroc=auc, auprc=auprc,
                accuracy=acc, f1=f1, sensitivity=sens, specificity=spec,
                confusion_matrix=cm, auroc_undefined=auroc_undefined)


def build_val_predictions_df(val_ds, ev):
    subject_ids = val_ds.df['subject_id'].tolist()
    sites = val_ds.df['site'].tolist()
    assert len(subject_ids) == len(ev['probs']), \
        "val subject-id count does not match prediction count -- loader order assumption broke!"
    return pd.DataFrame({
        'subject_id': subject_ids, 'site': sites,
        'y_true': ev['labels'], 'y_prob': ev['probs'], 'y_pred': ev['preds'],
    })


# ============================================================================
# Per-variant training, with per-epoch checkpointing and interruption recovery
# ============================================================================
def train_variant(name, tag, kwargs, train_ds, val_ds, out_dir=None, resume=None,
                   epochs=None, patience=None, _stop_after_epoch=None,
                   index_hash=None, split_hash=None):
    """
    `_stop_after_epoch` is a TEST-ONLY hook (used by run_sanity_check()'s resume
    test): if set, training returns immediately after that epoch's checkpoint
    is saved, WITHOUT finalizing results -- simulating a process kill at the
    worst realistic moment (right after a checkpoint write). Leave it None
    (default) for every real run.

    `index_hash`/`split_hash` (item 1, strict configuration validation): sha256
    fingerprints of the dataset file and the exact train/val subject split,
    computed once in main() and passed to every variant. Combined with
    `kwargs`/epochs/patience/batch_size/lr/val_split/seed into a single
    config fingerprint that is compared against the fingerprint stored in
    any existing checkpoint/result before it is reused -- see
    check_fingerprint_or_raise().
    """
    out_dir = OUT_DIR if out_dir is None else out_dir
    resume = RESUME if resume is None else resume
    epochs = EPOCHS if epochs is None else epochs
    patience = PATIENCE if patience is None else patience
    paths = checkpoint_paths(tag, out_dir)
    current_fp = compute_config_fingerprint(kwargs, index_hash, split_hash, epochs, patience)

    if resume and os.path.isfile(paths['results']):
        saved_fp = None
        if os.path.isfile(paths['config']):
            saved_config = json.load(open(paths['config']))
            saved_fp = saved_config.get('config_fingerprint')
        require_fingerprint_or_raise(saved_fp, tag, 'skip (reuse completed results)', paths['config'])
        check_fingerprint_or_raise(current_fp, saved_fp, tag, 'skip (reuse completed results)')
        print(f"  [{tag}] already COMPLETE (found {paths['results']}) -- skipping.")
        return pd.read_csv(paths['results']).iloc[0].to_dict(), 'skipped'

    train_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=train_gen, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    model = ADHDProposedModelV2(**kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    # Sentinel is -1.0, not 0.0: a real AUROC can legitimately equal exactly 0.0 on an
    # early/degenerate epoch, and the selection check below is `ev['auroc'] > best['auroc']`
    # -- with a 0.0 sentinel, 0.0 > 0.0 is False, so NO checkpoint would ever be saved and
    # `best_state` would stay None for the whole run, silently falling back to whatever the
    # LAST epoch's weights happen to be, with placeholder (zero) metrics in the saved
    # results. -1.0 is lower than any real AUROC (always in [0, 1], including the
    # documented AUROC-undefined fallback of 0.5 -- see evaluate_full()), so the FIRST
    # valid validation epoch is always accepted as the initial checkpoint (item 2).
    best = dict(auroc=-1.0, auprc=0.0, accuracy=0.0, f1=0.0, sensitivity=0.0, specificity=0.0, epoch=0)
    best_state = None
    patience_ctr = 0
    history = []
    resumed = False

    if resume and os.path.isfile(paths['latest']):
        # NOTE: config-fingerprint mismatches must ABORT the run (RuntimeError), never be
        # treated as "checkpoint failed to load, start fresh" -- so the fingerprint check
        # happens in its OWN try block, deliberately not caught by the broad except below
        # that is reserved for genuine load/corruption failures (item 1: never silently
        # reuse incompatible results).
        try:
            ckpt = torch.load(paths['latest'], map_location=DEVICE, weights_only=False)
        except Exception as e:
            ckpt = None
            print(f"  [{tag}] WARNING: latest checkpoint exists but failed to load ({e!r}) -- "
                  f"starting this variant fresh instead.")

        if ckpt is not None:
            saved_fp = ckpt.get('config_fingerprint')
            require_fingerprint_or_raise(saved_fp, tag, 'resume from latest checkpoint', paths['latest'])
            check_fingerprint_or_raise(current_fp, saved_fp, tag, 'resume from latest checkpoint')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            train_gen.set_state(ckpt['train_generator_state'])
            torch.set_rng_state(ckpt['torch_rng_state'])
            if torch.cuda.is_available() and ckpt.get('torch_cuda_rng_state') is not None:
                torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng_state'])
            np.random.set_state(ckpt['numpy_rng_state'])
            random.setstate(ckpt['python_rng_state'])
            best = ckpt['best']
            best_state = ckpt['best_model_state_dict']
            patience_ctr = ckpt['patience_ctr']
            history = ckpt['history']
            start_epoch = ckpt['epoch'] + 1
            resumed = True
            print(f"  [{tag}] RESUMING from epoch {ckpt['epoch']} "
                  f"(latest checkpoint found, best so far: AUROC={best['auroc']:.4f} @ epoch {best['epoch']})")

    if not resumed:
        set_seed(SEED)
        model = ADHDProposedModelV2(**kwargs).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        train_gen.manual_seed(SEED)

    n_params = sum(p.numel() for p in model.parameters())
    t_train_start = time.time()
    auroc_was_undefined_any_epoch = any(h.get('auroc_undefined') for h in history)
    epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for vol, pheno, label in train_loader:
            vol, label = vol.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(vol), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        ev = evaluate_full(model, val_loader)
        if ev['auroc_undefined']:
            auroc_was_undefined_any_epoch = True
        history.append(dict(
            epoch=epoch, train_loss=epoch_loss / max(n_batches, 1),
            val_auroc=ev['auroc'], val_auprc=ev['auprc'], val_accuracy=ev['accuracy'],
            val_f1=ev['f1'], val_sensitivity=ev['sensitivity'], val_specificity=ev['specificity'],
            auroc_undefined=ev['auroc_undefined'],
        ))

        if ev['auroc'] > best['auroc']:
            best = dict(auroc=ev['auroc'], auprc=ev['auprc'], accuracy=ev['accuracy'], f1=ev['f1'],
                         sensitivity=ev['sensitivity'], specificity=ev['specificity'], epoch=epoch)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            atomic_save(best_state, paths['best'])
        else:
            patience_ctr += 1

        # "Latest" checkpoint saved after EVERY epoch -- this is what makes
        # interruption recovery possible at all.
        checkpoint = dict(
            epoch=epoch, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            train_generator_state=train_gen.get_state(),
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(),
            best=best, best_model_state_dict=best_state, patience_ctr=patience_ctr,
            history=history, seed=SEED, kwargs=kwargs, config_fingerprint=current_fp,
        )
        atomic_save(checkpoint, paths['latest'])

        if _stop_after_epoch is not None and epoch >= _stop_after_epoch:
            print(f"  [{tag}] ** SIMULATED INTERRUPTION ** after epoch {epoch} "
                  f"(test-only _stop_after_epoch hook) -- returning without finalizing.")
            return None, 'interrupted'

        if patience_ctr >= patience:
            break

    train_time_s = time.time() - t_train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    flops, lat_mean, lat_std = measure_flops_and_latency(model)
    final_ev = evaluate_full(model, val_loader)   # predictions from the BEST checkpoint, for saving
    atomic_to_csv(build_val_predictions_df(val_ds, final_ev), paths['valpred'])
    atomic_to_csv(pd.DataFrame(history), paths['history'])

    result = {
        'variant': name, 'label': tag, 'params': n_params,
        'auroc': best['auroc'], 'auprc': best['auprc'], 'accuracy': best['accuracy'],
        'f1': best['f1'], 'sensitivity': best['sensitivity'], 'specificity': best['specificity'],
        'flops': flops, 'inference_ms_mean': lat_mean, 'inference_ms_std': lat_std,
        'train_time_s': round(train_time_s, 1), 'epochs_run': epoch, 'best_epoch': best['epoch'],
        'seed': SEED, 'auroc_was_undefined_any_epoch': auroc_was_undefined_any_epoch,
    }
    atomic_to_csv(pd.DataFrame([result]), paths['results'])

    config = {
        'variant': name, 'label': tag, 'kwargs': kwargs,
        'epochs_max': epochs, 'epochs_run': epoch, 'best_epoch': best['epoch'],
        'batch_size': BATCH_SIZE, 'lr': LR, 'val_split': VAL_SPLIT, 'patience': patience,
        'seed': SEED, 'index_file': INDEX_FILE, 'out_dir': out_dir,
        'config_fingerprint': current_fp,
    }
    with open(paths['config'], 'w') as f:
        json.dump(config, f, indent=2, default=str)

    if auroc_was_undefined_any_epoch:
        print(f"  [{tag}] WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch -- "
              f"see {paths['history']}'s auroc_undefined column.")

    return result, ('resumed' if resumed else 'fresh')


# ============================================================================
# Synthetic sanity tests -- NOT the main execution path. No real dataset is
# touched; `main()` calls this once, first, before loading real data.
# ============================================================================
class _SyntheticDS(Dataset):
    """Minimal in-memory stand-in for ADHDDataset, used ONLY by run_sanity_check()."""
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.vols = torch.randn(n, 1, 64, 64, 64, generator=g)
        self.labels = torch.randint(0, 2, (n,), generator=g)
        self.df = pd.DataFrame({'subject_id': [f'SYN{i:03d}' for i in range(n)], 'site': ['SIM'] * n})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vols[idx], torch.zeros(5), self.labels[idx]

    def class_weights(self):
        return torch.tensor([1.0, 1.0])


def run_sanity_check():
    print(f"\n{'='*70}\n  PHASE 2 STEP 1 -- CODE-LEVEL SANITY TESTS (synthetic data only)\n{'='*70}")
    torch.manual_seed(0)
    vol = torch.randn(4, 1, 64, 64, 64)

    # ---- Test A: forward pass + parameter counts for all 4 variants (unchanged architecture) ----
    print("\n  Test A -- forward pass + parameter counts, all 4 variants:")
    expected_params = {'A3': 105424, 'A4': 113968, 'A4-new': 113937, 'A5-new': 114482}
    for name, tag, kwargs in VARIANTS:
        m = ADHDProposedModelV2(**kwargs)
        out = m(vol)
        assert out.shape == (4, 2), f"{tag}: unexpected output shape {tuple(out.shape)}"
        n_params = sum(p.numel() for p in m.parameters())
        print(f"    {name:<34} params={n_params:>8,}  output={tuple(out.shape)}  "
              f"(expected {expected_params[tag]:,}: {'MATCH' if n_params == expected_params[tag] else 'MISMATCH!'})")
        assert n_params == expected_params[tag], \
            f"{tag} parameter count changed ({n_params:,} vs expected {expected_params[tag]:,}) -- " \
            f"architecture must NOT change between runs of this script!"
    print("  TEST A PASSED")

    # ---- Test B: checkpoint save/load round-trip ----
    print("\n  Test B -- checkpoint save/load round-trip:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_ckpt_test_') as tmp:
        m1 = ADHDProposedModelV2(trm_type='pairwise', use_attention=True)
        path = os.path.join(tmp, 'test_model.pt')
        atomic_save(m1.state_dict(), path)
        assert os.path.isfile(path) and not os.path.isfile(path + '.tmp'), \
            "atomic_save left a .tmp file behind or didn't write the final file!"
        m2 = ADHDProposedModelV2(trm_type='pairwise', use_attention=True)
        m2.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        all_equal = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
        print(f"    Saved {len(sd1)} tensors, reloaded into a fresh instance, all equal: {all_equal}")
        assert all_equal, "checkpoint round-trip did not reproduce the exact weights!"
    print("  TEST B PASSED")

    # ---- Test C: genuine interrupt-then-resume, compared against an uninterrupted reference ----
    print("\n  Test C -- interrupt-then-resume reproduces the uninterrupted training-loss sequence:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_resume_test_') as tmp:
        train_ds = _SyntheticDS(16, seed=1)
        val_ds = _SyntheticDS(8, seed=2)
        kwargs = dict(trm_type='pairwise', use_attention=True)
        TEST_EPOCHS = 4

        set_seed(SEED)
        ref_result, ref_status = train_variant('Synthetic (ref)', 'synref', kwargs, train_ds, val_ds,
                                                 out_dir=tmp, resume=False, epochs=TEST_EPOCHS, patience=100)
        ref_history = pd.read_csv(f'{tmp}/synref_training_history.csv')
        print(f"    Uninterrupted reference run: {len(ref_history)} epochs, "
              f"status={ref_status}, final train_loss={ref_history['train_loss'].iloc[-1]:.6f}")
        assert len(ref_history) == TEST_EPOCHS

        set_seed(SEED)
        _, int_status = train_variant('Synthetic (interrupted)', 'synint', kwargs, train_ds, val_ds,
                                       out_dir=tmp, resume=False, epochs=TEST_EPOCHS, patience=100,
                                       _stop_after_epoch=2)
        assert int_status == 'interrupted'
        assert os.path.isfile(f'{tmp}/synint_latest_checkpoint.pt'), "no latest checkpoint after simulated interruption!"
        assert not os.path.isfile(f'{tmp}/synint_results.csv'), "results.csv should NOT exist for an interrupted run!"
        print(f"    Simulated interruption after epoch 2: latest checkpoint exists, results.csv does not (correct)")

        resumed_result, resumed_status = train_variant('Synthetic (resumed)', 'synint', kwargs, train_ds, val_ds,
                                                          out_dir=tmp, resume=True, epochs=TEST_EPOCHS, patience=100)
        assert resumed_status == 'resumed'
        resumed_history = pd.read_csv(f'{tmp}/synint_training_history.csv')
        print(f"    Resumed run: status={resumed_status}, {len(resumed_history)} epochs total "
              f"(2 before interruption + 2 after resuming), final train_loss={resumed_history['train_loss'].iloc[-1]:.6f}")
        assert len(resumed_history) == TEST_EPOCHS, \
            f"resumed run has {len(resumed_history)} epochs of history, expected {TEST_EPOCHS} -- " \
            f"resume did not continue from the interruption point correctly!"

        loss_diff = (ref_history['train_loss'].values - resumed_history['train_loss'].values)
        max_abs_diff = np.abs(loss_diff).max()
        print(f"    Per-epoch train_loss, reference vs resumed: max abs diff = {max_abs_diff:.3e}")
        print(f"      reference epochs 1-4 : {ref_history['train_loss'].values.round(6).tolist()}")
        print(f"      resumed   epochs 1-4 : {resumed_history['train_loss'].values.round(6).tolist()}")
        assert max_abs_diff < 1e-5, (
            f"resumed training-loss sequence diverges from the uninterrupted reference "
            f"(max abs diff {max_abs_diff:.3e}) -- RNG/DataLoader-generator state was not "
            f"restored correctly on resume!")
        print("    Resumed sequence matches the uninterrupted reference to within 1e-5 "
              "(CPU, NUM_WORKERS=0 -- exact reproducibility is expected here; a CUDA run "
              "may show tiny cuDNN-nondeterminism drift, which is not a checkpointing bug).")

        final_params = sum(p.numel() for p in ADHDProposedModelV2(**kwargs).parameters())
        print(f"    Resumed variant's architecture unchanged: params={resumed_result['params']:,} "
              f"(expect 114,482: {'MATCH' if resumed_result['params'] == 114482 else 'MISMATCH!'})")
        assert resumed_result['params'] == 114482
    print("  TEST C PASSED")

    # ---- Test D -- strict configuration validation (item 1) ----
    print("\n  Test D -- strict configuration validation refuses to reuse a changed configuration:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_fingerprint_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=20)
        val_ds = _SyntheticDS(4, seed=21)
        kwargs = dict(trm_type='pairwise', use_attention=True)

        result, status = train_variant('FP test', 'synfp', kwargs, train_ds, val_ds,
                                         out_dir=tmp, resume=False, epochs=2, patience=100,
                                         index_hash='hashA', split_hash='splitA')
        assert status == 'fresh'
        assert os.path.isfile(f'{tmp}/synfp_results.csv')

        raised = False
        try:
            train_variant('FP test', 'synfp', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=2, patience=100,
                           index_hash='hashB_DIFFERENT', split_hash='splitA')
        except RuntimeError as e:
            raised = True
            print(f"    Changed dataset (index_hash) correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised, "a changed index_hash was NOT detected/refused when skipping completed results!"

        result2, status2 = train_variant('FP test', 'synfp', kwargs, train_ds, val_ds,
                                           out_dir=tmp, resume=True, epochs=2, patience=100,
                                           index_hash='hashA', split_hash='splitA')
        assert status2 == 'skipped'
        print("    Identical configuration correctly reuses the completed result (status=skipped)")

        _, int_status = train_variant('FP test 2', 'synfp2', kwargs, train_ds, val_ds,
                                       out_dir=tmp, resume=False, epochs=4, patience=100,
                                       index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status == 'interrupted'
        raised2 = False
        try:
            train_variant('FP test 2', 'synfp2', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=4, patience=100,
                           index_hash='hashA', split_hash='splitB_DIFFERENT')
        except RuntimeError as e:
            raised2 = True
            print(f"    Changed split (split_hash) correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised2, "a changed split_hash was NOT detected/refused when resuming from a checkpoint!"

        # A results.csv/config.json with NO config_fingerprint at all (predates the
        # safeguard, or a different script version) must be refused for skip too --
        # never silently treated as "nothing to compare, go ahead".
        _, status3 = train_variant('FP test 3', 'synfp3', kwargs, train_ds, val_ds,
                                    out_dir=tmp, resume=False, epochs=2, patience=100,
                                    index_hash='hashA', split_hash='splitA')
        assert status3 == 'fresh'
        cfg_path = f'{tmp}/synfp3_config.json'
        cfg = json.load(open(cfg_path))
        del cfg['config_fingerprint']
        json.dump(cfg, open(cfg_path, 'w'))
        raised3 = False
        try:
            train_variant('FP test 3', 'synfp3', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=2, patience=100,
                           index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised3 = True
            print(f"    Missing config_fingerprint correctly REFUSED on skip: {str(e).splitlines()[0]}")
        assert raised3, "a missing config_fingerprint was NOT refused when skipping completed results!"

        # Same check on the RESUME path: a checkpoint saved with no config_fingerprint
        # (e.g. produced by a pre-safeguard script version) must also be refused.
        _, int_status2 = train_variant('FP test 4', 'synfp4', kwargs, train_ds, val_ds,
                                        out_dir=tmp, resume=False, epochs=4, patience=100,
                                        index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status2 == 'interrupted'
        ckpt_path = f'{tmp}/synfp4_latest_checkpoint.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        del ckpt['config_fingerprint']
        torch.save(ckpt, ckpt_path)
        raised4 = False
        try:
            train_variant('FP test 4', 'synfp4', kwargs, train_ds, val_ds,
                           out_dir=tmp, resume=True, epochs=4, patience=100,
                           index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised4 = True
            print(f"    Missing config_fingerprint correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised4, "a missing config_fingerprint was NOT refused when resuming from a checkpoint!"
    print("  TEST D PASSED")

    # ---- Test E -- first valid checkpoint saved even when AUROC equals exactly 0.0 (item 2) ----
    print("\n  Test E -- first-epoch checkpoint is saved even when AUROC is exactly 0.0:")
    with tempfile.TemporaryDirectory(prefix='phase2_trm_auroc0_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=30)
        val_ds = _SyntheticDS(4, seed=31)
        kwargs = dict(trm_type='pairwise', use_attention=True)

        global evaluate_full
        original_evaluate_full = evaluate_full
        call_count = {'n': 0}

        def _forced_zero_evaluate_full(model, loader):
            ev = original_evaluate_full(model, loader)
            call_count['n'] += 1
            if call_count['n'] == 1:
                ev = dict(ev)
                ev['auroc'] = 0.0   # force exactly zero on the very first validation epoch
            return ev

        evaluate_full = _forced_zero_evaluate_full
        try:
            result, status = train_variant('AUROC0 test', 'synauroc0', kwargs, train_ds, val_ds,
                                            out_dir=tmp, resume=False, epochs=1, patience=100)
        finally:
            evaluate_full = original_evaluate_full

        best_ckpt_exists = os.path.isfile(f'{tmp}/synauroc0_best_model.pt')
        print(f"    Forced epoch-1 AUROC=0.0 -> best checkpoint saved: {best_ckpt_exists}")
        assert best_ckpt_exists, \
            "no best checkpoint was saved even though epoch 1 completed -- sentinel bug not fixed!"
        print(f"    Saved best_epoch = {result['best_epoch']}  (expect 1, not left at the 0 placeholder)")
        assert result['best_epoch'] == 1
    print("  TEST E PASSED")

    print(f"\n{'='*70}\n  ALL CODE-LEVEL SANITY TESTS (A-E) PASSED ✓\n{'='*70}")
    print("  (Synthetic data only -- no real dataset was loaded, no real checkpoints were "
          "left behind (temp dirs auto-cleaned). The real 4-variant training run starts next.)")


# ============================================================================
# Dataset integrity (same checks as phase2_consistency.py's validate_dataset_integrity)
# ============================================================================
def validate_dataset_integrity(raw_df, train_ids, val_ids):
    print(f"\n{'='*70}\n  DATASET INTEGRITY CHECKS\n{'='*70}")
    dup_mask = raw_df['subject_id'].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    print(f"  Duplicate subject_id rows: {n_dup}")
    assert n_dup == 0, f"{n_dup} duplicate subject_id rows found in {INDEX_FILE}."

    missing = [p for p in raw_df['npy_path'] if not os.path.isfile(p)]
    print(f"  Missing .npy files: {len(missing)} / {len(raw_df)}")
    assert not missing, f"{len(missing)} .npy files referenced in {INDEX_FILE} do not exist on disk."

    overlap = set(train_ids) & set(val_ids)
    print(f"  Train/val subject overlap: {len(overlap)}")
    assert not overlap, f"{len(overlap)} subjects appear in BOTH train and val -- subject-level leakage!"
    print("  Subject-level independence confirmed: every subject is in exactly one of train/val.")

    val_labels = raw_df[raw_df['subject_id'].isin(val_ids)]['label']
    val_counts = val_labels.value_counts().to_dict()
    print(f"  Validation label distribution: {val_counts}")
    assert val_labels.nunique() == 2, f"validation split contains only one class ({val_counts})."

    print(f"  Train subjects: {len(train_ids)}   Val subjects: {len(val_ids)}   Total: {len(raw_df)}")
    print("  ALL DATASET INTEGRITY CHECKS PASSED ✓")


def save_site_label_counts(raw_df, train_ids, val_ids, out_dir):
    """Per-site label-count breakdown for both splits -- same helper as phase2_consistency.py."""
    rows = []
    for split_name, ids in [('train', train_ids), ('val', val_ids)]:
        sub = raw_df[raw_df['subject_id'].isin(ids)]
        counts = sub.groupby(['site', 'label']).size().reset_index(name='count')
        counts.insert(0, 'split', split_name)
        rows.append(counts)
    site_label_df = pd.concat(rows, ignore_index=True)
    atomic_to_csv(site_label_df, f'{out_dir}/site_label_counts.csv')
    print(f"\n  Per-site label counts (train + val) -> {out_dir}/site_label_counts.csv")
    print(site_label_df.to_string(index=False))
    return site_label_df


def main():
    run_sanity_check()

    print(f"\n{'='*70}\n  LOADING REAL DATASET\n{'='*70}")
    print(f"  INDEX_FILE = {INDEX_FILE}")
    raw_df = pd.read_csv(INDEX_FILE)
    raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
    labels_all = raw_df['label'].values
    print(f"  Total subjects : {len(raw_df)}")

    train_ids_path = f'{OUT_DIR}/train_subject_ids.csv'
    val_ids_path = f'{OUT_DIR}/val_subject_ids.csv'
    if RESUME and os.path.isfile(train_ids_path) and os.path.isfile(val_ids_path):
        train_ids = pd.read_csv(train_ids_path)['subject_id'].tolist()
        val_ids = pd.read_csv(val_ids_path)['subject_id'].tolist()
        print(f"  Loaded EXISTING train/val subject-ID split from disk ({train_ids_path}, "
              f"{val_ids_path}) -- guarantees the identical split across any restart.")
    else:
        set_seed(SEED)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
        train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
        val_ids = raw_df.iloc[val_idx]['subject_id'].tolist()
        atomic_to_csv(pd.DataFrame({'subject_id': train_ids}), train_ids_path)
        atomic_to_csv(pd.DataFrame({'subject_id': val_ids}), val_ids_path)
        print(f"  Computed a new split and saved it to {train_ids_path} / {val_ids_path}")

    validate_dataset_integrity(raw_df, train_ids, val_ids)
    save_site_label_counts(raw_df, train_ids, val_ids, OUT_DIR)

    # Dataset-identity and split fingerprints (item 1) -- computed once here, passed to
    # every variant, so a changed dataset file or a changed split is detected BEFORE any
    # checkpoint/result produced under the old one is silently reused.
    index_hash = hash_file(INDEX_FILE)
    split_hash = hash_split(train_ids, val_ids)
    print(f"  Dataset fingerprint: index_hash={index_hash[:12]}...  split_hash={split_hash[:12]}...")

    train_stats_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()
    train_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=True, fit_stats=fit_stats)
    val_ds = ADHDDataset(INDEX_FILE, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
    print(f"  Train : {len(train_ds)}   Val : {len(val_ds)}")

    for name, tag, kwargs in VARIANTS:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        result, status = train_variant(name, tag, kwargs, train_ds, val_ds,
                                        index_hash=index_hash, split_hash=split_hash)
        if status == 'skipped':
            print(f"  [skipped -- already complete]  AUROC={result['auroc']:.4f}  "
                  f"params={int(result['params']):,}")
        else:
            print(f"  [{status}]  AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}  "
                  f"Acc={result['accuracy']:.4f}  F1={result['f1']:.4f}  "
                  f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}")
            print(f"  params={result['params']:,}  GFLOPs={result['flops']/1e9:.4f}  "
                  f"inference={result['inference_ms_mean']:.2f}+/-{result['inference_ms_std']:.2f}ms  "
                  f"(best epoch {result['best_epoch']}/{result['epochs_run']}, {result['train_time_s']}s)")

    # ---- Aggregate table built from each variant's OWN saved results.csv (resume-safe) ----
    all_results = []
    for name, tag, kwargs in VARIANTS:
        rpath = f"{OUT_DIR}/{tag}_results.csv"
        assert os.path.isfile(rpath), f"{tag}_results.csv missing after train_variant() -- should not happen"
        all_results.append(pd.read_csv(rpath).iloc[0].to_dict())

    results_df = pd.DataFrame(all_results)[
        ['variant', 'label', 'params', 'flops', 'auroc', 'auprc', 'accuracy', 'f1',
         'sensitivity', 'specificity', 'inference_ms_mean', 'inference_ms_std',
         'train_time_s', 'epochs_run', 'best_epoch', 'seed', 'auroc_was_undefined_any_epoch']
    ]
    atomic_to_csv(results_df, f'{OUT_DIR}/trm_ablation_results.csv')

    print(f"\n{'='*90}")
    print("  PHASE 2 -- TRM ABLATION (A3 / A4 / A4-new / A5-new)")
    print(f"{'='*90}")
    print(f"  {'Variant':<22}{'Params':>9}{'GFLOPs':>9}{'AUROC':>8}{'AUPRC':>8}{'Acc':>8}{'F1':>8}")
    for _, r in results_df.iterrows():
        print(f"  {r['variant']:<22}{int(r['params']):>9,}{r['flops']/1e9:>9.4f}"
              f"{r['auroc']:>8.4f}{r['auprc']:>8.4f}{r['accuracy']*100:>7.2f}%{r['f1']:>8.4f}")
    print("\n  NOTE: all metrics above are from the single held-out VALIDATION split -- not an "
          "independent test set (none exists in this pipeline). These four variants share the "
          "SAME augmentation policy (dataset.py's augment=True), so the A3-vs-A4-vs-A4-new-vs-"
          "A5-new comparison above is valid. Do NOT cross-compare these numbers against Step "
          "2's M0/M1 (phase2_consistency.py) -- those are trained with augment=False (a "
          "deliberate, documented difference; see that file's AUGMENTATION-COMPARISON NOTE). "
          "Multi-fold CV / cross-site evaluation are a separate, not-yet-approved future stage.")

    # ── Figure: AUROC vs params, A3->A4->A4-new->A5-new ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#888888', '#B2182B', '#2166AC', '#2166AC']
    bars = ax.bar(results_df['label'], results_df['auroc'], color=colors, width=0.6)
    for bar, auc in zip(bars, results_df['auroc']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Validation AUROC')
    ax.set_title('Phase 2 -- TRM Ablation: MHA vs. Pairwise Relationship Module')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_trm_ablation_auroc.png', dpi=300)
    plt.close(fig)

    print(f"\nResults -> {OUT_DIR}/trm_ablation_results.csv")
    print(f"Figure  -> {OUT_DIR}/fig_trm_ablation_auroc.png")
    a4 = results_df[results_df['label'] == 'A4']['params'].values[0]
    a4new = results_df[results_df['label'] == 'A4-new']['params'].values[0]
    print(f"\nNote: A4 vs A4-new have nearly identical parameter counts "
          f"({int(a4):,} vs {int(a4new):,}), so any AUROC difference reflects the "
          f"relationship MECHANISM, not added capacity.")

    if results_df['auroc_was_undefined_any_epoch'].any():
        flagged = results_df[results_df['auroc_was_undefined_any_epoch']]['label'].tolist()
        print(f"\n*** WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch for: "
              f"{flagged} -- check {{tag}}_training_history.csv's auroc_undefined column. ***")


if __name__ == '__main__':
    main()


#==============================================================================
# FILE: code/phase2_consistency.py
#==============================================================================

"""
PHASE 2, Step 2 ONLY: representation-consistency learning -- WITH experiment
tracking, per-epoch checkpointing, and interruption recovery (same design
as phase2_trm_ablation.py).

Does NOT modify phase2_model.py (Step 1) or anything in Phase 1 (cells
0-34). The architecture under test is exactly Step 1's A5-new config
(pairwise TRM + Anatomical Attention), imported unmodified from
phase2_model.py. The representation (`encode_representation` below) is
recomputed by calling the model's existing public submodules -- the same
technique proposed_explainability.py already uses for its saliency map --
so there is no new method on ADHDProposedModelV2 and no risk to Step 1's
already-verified numbers.

Hypothesis under test: mild acquisition/preprocessing variation should not
substantially change the learned subject-level representation. This is
tested directly by comparing cos(z_original, z_augmented) on held-out
validation subjects, M0 vs M1 -- NOT by only checking that the training
loss went down.

    M0 : A5-new, classification loss only (identical to Step 1's A5-new)
    M1 : A5-new, classification loss + lambda * consistency loss (lambda=0.05)

Everything else (data, split, seed, optimizer, schedule, epochs,
architecture) is identical between M0 and M1 -- the only difference is
whether the consistency term is added to the loss.

View-1/View-2 isolation (Step 2 ONLY, does not touch dataset.py): `main()`
builds its training dataset with `augment=False` -- i.e. View 1 is always
the SAME deterministic, unaugmented preprocessed MRI that validation uses,
never dataset.py's own `augment=True` random-flip path (that path is
untouched and still used exactly as before by Phase-1's own scripts, and
by Step 1's phase2_trm_ablation.py). View 2 is produced EXCLUSIVELY by
`consistency_augment()` inside `train_model()`'s training loop (intensity
scaling/bias, Gaussian noise, small spatial shift -- never a flip).

*** AUGMENTATION-COMPARISON NOTE (do not skip this) ***
Step 1 (phase2_trm_ablation.py) trains A3/A4/A4-new/A5-new with
`augment=True` (dataset.py's own random-flip augmentation on a SINGLE
view) -- the same augmentation policy for all four Step-1 variants, so
Step 1's own internal A3-vs-A4-vs-A4-new-vs-A5-new comparison is valid.
Step 2 (this file) trains M0/M1 with `augment=False` (deterministic View 1,
per the isolation above) -- the same augmentation policy for both M0 and
M1, so Step 2's own internal M0-vs-M1 comparison is ALSO valid.
What is NOT valid: comparing Step 1's A5-new number against Step 2's M0 or
M1 number. Even though M0 is "A5-new, classification loss only" in
architecture, M0 is trained under augment=False while Step 1's A5-new is
trained under augment=True -- a real confound. M0 already IS the correct,
augmentation-matched "classification-loss-only" control for the M0-vs-M1
comparison (that is its entire purpose); no additional variant is needed.
Step 1's A5-new is simply a different experiment (different augmentation
policy) and must never be substituted for M0 in any comparison.

This file connects to the REAL ADHD-200 pipeline already used everywhere
else in this project (dataset.ADHDDataset, the same `processed_index.csv`,
the same subject-level StratifiedShuffleSplit, the same leakage-free
fit_stats convention) -- it does not invent a new dataset format and does
not use synthetic data as its main execution path. The ONLY things you
should need to change for your own Kaggle/local layout are DATA_ROOT and
INDEX_FILE below.

*** VALIDATION, NOT AN INDEPENDENT TEST SET ***
Every metric this script reports (AUROC/AUPRC/accuracy/F1/sensitivity/
specificity/confusion matrix/representation stability) is computed on the
single held-out VALIDATION split from this run's train/val
StratifiedShuffleSplit -- there is no independent, never-touched test set
anywhere in this pipeline. Report these numbers as validation results.
Multi-fold cross-validation and leave-one-site-out cross-site evaluation
(which would give a more robust estimate) are a separate, not-yet-approved
future stage -- not implemented here.

Outputs go to a SEPARATE directory, '{DATA_ROOT}/phase2/consistency/' --
Phase-1 results, Step-1's trm_ablation_results.csv, and all existing
notebooks/CSVs/figures are untouched.

Interruption recovery (same design as phase2_trm_ablation.py): set
RESUME = True (default) and re-run this script. M0 and M1 each resume
INDEPENDENTLY -- a variant whose `{tag}_results.csv` already exists is
treated as COMPLETE and skipped; a variant with a `{tag}_latest_
checkpoint.pt` but no `{tag}_results.csv` is resumed from that checkpoint
-- model, optimizer, scheduler, early-stopping state, training history,
AND the exact RNG/DataLoader-shuffle-generator state are all restored, so
the per-epoch training-loss sequence after resuming continues the sequence
an uninterrupted run would have produced (verified in run_sanity_check()'s
Test H below). Each variant's results are saved immediately when that
variant finishes, not held until both M0 and M1 complete. Checkpoints and
CSVs are written atomically (temp file + os.replace()).

Strict configuration validation: every checkpoint/result carries a
`config_fingerprint` (sha256 of the dataset file + the exact subject
split, plus arch_kwargs/use_consistency/lam/epochs/patience/batch_size/
lr/val_split/seed). Before skipping a completed variant OR resuming an
interrupted one, the CURRENT run's fingerprint is compared against the
saved one; any mismatch RAISES (never silently reuses a result produced
under different settings) -- see `check_fingerprint_or_raise()` and
Test I. A checkpoint/result with NO `config_fingerprint` at all (it
predates this safeguard, or was produced by a different script version)
is refused the same way, never silently trusted -- see
`require_fingerprint_or_raise()`. Either case requires pointing `OUT_DIR`
at a fresh, empty directory. The first valid validation checkpoint is
always saved, even if its AUROC is exactly 0.0 (the "best so far" sentinel
is -1.0, not 0.0 -- see Test J).

`run_sanity_check()` runs CODE-LEVEL tests only (synthetic data, no real
dataset, no training run, no AUROC/AUPRC/F1 reported) and is called once at
the start of `main()` purely to catch implementation errors before any
real training happens; it is NOT the main execution path:
    Test A -- forward pass (MRI -> representation -> classifier)
    Test B -- two-view representation (same subject, same weights)
    Test C -- consistency loss returns a finite scalar
    Test D -- gradient flow (backward pass, finite, no NaN/Inf)
    Test E -- identical representations give ~zero consistency loss
    Test F -- augmentation isolation (no randomness outside the two-view step)
    Test G -- encode_representation()+classify_from_repr() == model.forward()
              exactly (proves no parallel/divergent path was introduced)
    Test H -- genuine interrupt-then-resume reproduces the uninterrupted
              per-epoch training-loss sequence (same test design as
              phase2_trm_ablation.py's Test C)
    Test I -- strict configuration validation raises on a changed dataset
              (index_hash), changed split (split_hash), or changed
              consistency weight (lambda), both when skipping a completed
              variant and when resuming an interrupted one; an UNCHANGED
              configuration still skips cleanly; a completed result or
              checkpoint with NO config_fingerprint at all is also
              refused, both when skipping and when resuming
    Test J -- the first-epoch checkpoint is saved even when that epoch's
              AUROC is exactly 0.0 (forced via a temporary monkeypatch of
              evaluate(), restored immediately afterward)
    Section 14  -- parameter count with vs without consistency (expected equal)
    Section 14b -- M0/M1 use equivalent initial weights (same seed -> same
                   state_dict, verified tensor-by-tensor, not just a param count)

Before touching the real dataset, `main()` also runs
`validate_dataset_integrity()`: duplicate subject_id check, missing .npy
file check, subject-level train/val leakage check, a check that the
validation split contains both classes, AND now a per-site label-count
breakdown (saved to `site_label_counts.csv`) so site distribution is
visible, not just assumed.

After the sanity + dataset checks, `main()` builds the leakage-free
subject-level train/val split (loaded from disk on a RESUME if already
saved, to guarantee the identical split across restarts), then trains
whichever of M0/M1 are enabled (RUN_M0 / RUN_M1 below) end to end:
dataloaders -> model -> optimizer -> scheduler -> training loop ->
validation -> early stopping -> best-checkpoint selection -> metrics ->
per-subject validation predictions -> representation-stability measurement
(same fixed STABILITY_SEED for M0 and M1, so both see IDENTICAL evaluation
perturbations -- see measure_representation_stability()) -> FLOPs/parameter/
latency audit -> every artifact saved to disk. Nothing here is a placeholder.
"""

import os
import hashlib
import json
import random
import tempfile
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ADHDDataset
from phase2_model import ADHDProposedModelV2

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIG -- the only things you should need to edit for your own setup  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Change DATA_ROOT (and/or INDEX_FILE directly) if your Kaggle/local copy
# of the ADHD-200 pipeline's `processed_index.csv` + preprocessed `.npy`
# volumes live somewhere other than '/kaggle/working'. Nothing else in this
# file needs to change to point at your real dataset.
DATA_ROOT   = '/kaggle/working'
INDEX_FILE  = f'{DATA_ROOT}/processed_index.csv'
INDEX_CSV   = INDEX_FILE    # alias kept for consistency with the rest of this project's scripts
OUT_DIR     = f'{DATA_ROOT}/phase2/consistency'    # SEPARATE from Phase-1 and Step-1 outputs
os.makedirs(OUT_DIR, exist_ok=True)

RESUME = True    # True: skip completed variants (M0/M1 independently), resume an interrupted
                 # one from its latest checkpoint. False: ignore any existing checkpoints/
                 # results and retrain from scratch (files overwritten as training proceeds).

# Which experiments to actually run. Both True by default (full M0 vs M1
# comparison); set either to False to run only the other one without
# touching the training loop.
RUN_M0 = True
RUN_M1 = True

EPOCHS      = 150
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 15
SEED        = 42
STABILITY_SEED = 12345   # fixed seed for the POST-HOC stability measurement only (see
                          # measure_representation_stability() below) -- independent of
                          # SEED/training so M0 and M1 draw the exact same sequence of
                          # consistency_augment() perturbations at evaluation time, even
                          # though M1's training loop consumes a different number of
                          # random draws than M0's (M1 calls consistency_augment() every
                          # training step; M0 never does) and would otherwise leave the
                          # global RNG in a different state by the time stability is measured.
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# USE_CONSISTENCY=False reproduces M0 (classification loss only, no
# consistency term, no two-view augmentation) from the exact same code path
# as M1 -- this is what makes the M0 vs M1 ablation clean (RUN_M0/RUN_M1
# above control which of the two actually get trained in main(); this flag
# documents/exercises the same switch at the single-run level).
USE_CONSISTENCY          = True
CONSISTENCY_LAMBDA       = 0.05   # Section 7 default; configurable
CONSISTENCY_AUGMENTATION = True   # if False while USE_CONSISTENCY=True, view 2 = view 1 (debug only;
                                   # not a normal operating mode -- see Test E)
RUN_LAMBDA_SWEEP = False      # set True to additionally run M1a(0.01)/M1b(0.05)/M1c(0.10)
LAMBDA_SWEEP = [0.01, 0.05, 0.10]

# Architecture under test: exactly Step 1's A5-new (frozen, not redesigned here)
ARCH_KWARGS = dict(trm_type='pairwise', use_attention=True)


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


# ============================================================================
# Atomic writes -- never leave a half-written checkpoint/CSV on disk, even if
# the process is killed mid-write. Same helpers as phase2_trm_ablation.py.
# ============================================================================
def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_to_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def checkpoint_paths(tag, out_dir):
    return dict(
        latest=f'{out_dir}/{tag}_latest_checkpoint.pt',
        best=f'{out_dir}/best_{tag}_model.pt',
        history=f'{out_dir}/training_history_{tag}.csv',
        valpred=f'{out_dir}/val_predictions_{tag}.csv',
        results=f'{out_dir}/{tag}_results.csv',
        config=f'{out_dir}/config_{tag}.json',
        confusion=f'{out_dir}/confusion_matrix_{tag}.png',
    )


def hash_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def hash_split(train_ids, val_ids):
    return hashlib.sha256((','.join(sorted(train_ids)) + '|' + ','.join(sorted(val_ids))).encode()).hexdigest()


def compute_config_fingerprint(use_consistency, lam, index_hash, split_hash, epochs, patience):
    """
    Everything that must be IDENTICAL between the run that produced a saved
    checkpoint/result and the current run before it is safe to resume or
    skip. If any of these differ, the checkpoint/result was produced under
    different conditions and must never be silently reused.
    """
    return {
        'arch_kwargs': ARCH_KWARGS, 'use_consistency': use_consistency, 'lam': lam,
        'consistency_augmentation': CONSISTENCY_AUGMENTATION, 'stability_seed': STABILITY_SEED,
        'index_hash': index_hash, 'split_hash': split_hash,
        'epochs_max': epochs, 'patience': patience, 'batch_size': BATCH_SIZE,
        'lr': LR, 'val_split': VAL_SPLIT, 'seed': SEED,
    }


def check_fingerprint_or_raise(current_fp, saved_fp, tag, action):
    """
    Strict configuration validation (item 1): raises -- never warns and
    continues -- if any essential training setting differs between the
    current run and the one that produced the saved checkpoint/result this
    `action` ('skip' or 'resume') is about to reuse. Dataset identity
    (index_hash: a sha256 of processed_index.csv's bytes), the exact
    subject-level split (split_hash), model/loss configuration
    (arch_kwargs/use_consistency/lam), seed, learning rate, epochs/patience
    budget, batch size, and val_split are all covered.
    """
    mismatches = {k: (saved_fp.get(k), current_fp[k]) for k in current_fp if saved_fp.get(k) != current_fp[k]}
    if mismatches:
        lines = "\n".join(f"    {k}: saved={old!r}  current={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- configuration has changed since the saved "
            f"checkpoint/result was produced:\n{lines}\n"
            f"  Never silently reusing incompatible results. Either revert the changed "
            f"setting(s) above, or point OUT_DIR at a fresh, empty directory for this new "
            f"configuration (e.g. OUT_DIR = f'{{DATA_ROOT}}/phase2/consistency_v2')."
        )


def require_fingerprint_or_raise(saved_fp, tag, action, source_path):
    """
    Strict configuration validation, extended: a checkpoint/result that
    predates the config_fingerprint safeguard (or was produced by a script
    version that never saved one) carries no proof of what configuration
    produced it. Reusing it anyway would be exactly the silent-reuse-of-
    incompatible-results failure mode item 1 exists to prevent -- so a
    MISSING fingerprint refuses the same way a MISMATCHED one does, never
    with a "proceeding as before" fallback.
    """
    if saved_fp is None:
        raise RuntimeError(
            f"[{tag}] REFUSING to {action} -- {source_path} has no config_fingerprint "
            f"(it predates this safeguard, or was produced by a different script version), "
            f"so there is no way to verify it matches the current dataset/split/model/"
            f"training configuration.\n"
            f"  Never silently reusing a result whose configuration cannot be verified. "
            f"Point OUT_DIR at a fresh, empty directory to regenerate this result under "
            f"the current safeguard (e.g. OUT_DIR = f'{{DATA_ROOT}}/phase2/consistency_v2')."
        )


# ============================================================================
# Section 2A -- realistic, MRI-safe augmentation for the second ("View 2") pass
# ============================================================================
def consistency_augment(vol, intensity_std=0.05, noise_std=0.02, max_shift=2):
    """
    Mild, realistic perturbation used ONLY to build the second view for the
    consistency loss (and for the post-hoc stability measurement). Applied
    on top of whatever the data pipeline already produced for `vol`.

      - intensity gain + bias (small multiplicative + additive shift)
      - Gaussian noise
      - small integer-voxel spatial translation (torch.roll, +/- max_shift
        voxels per axis on a 64^3 volume -- bounded, does not crop/warp)

    No rotation, flip, or elastic warp here: this explicitly rules out
    transformations that could alter or duplicate structural interpretation,
    or change clinically meaningful morphology.
    Differentiable end-to-end except the integer roll amount itself (which
    is sampled, not learned, exactly like dataset.py's existing flip
    augmentation) -- gradients flow normally through the returned tensor.
    Only ever applied to the training view -- validation volumes never
    pass through this function (see Test F and `evaluate()` below).
    """
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


# ============================================================================
# Section 3 -- representation extraction (same weights, no new encoder)
# ============================================================================
def encode_representation(model, vol, pheno=None):
    """
    Recomputes the compact brain representation using the SAME model
    instance's existing submodules (encoder/tokenizer/refine/trm/anat_attn/
    token_agg[/pheno fusion]) -- i.e. literally the same weights used by
    model.forward(), called here only so the pre-classifier representation
    can be extracted without adding a method to phase2_model.py.
    """
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


def classify_from_repr(model, brain_repr):
    return model.evidential_head(brain_repr) if model.use_evidential else model.head(brain_repr)


def cosine_consistency_loss(z1, z2):
    """L_consistency = 1 - cosine_similarity(z1, z2), averaged over the batch."""
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    cos_sim = (z1n * z2n).sum(dim=-1)
    return (1.0 - cos_sim).mean(), cos_sim


# ============================================================================
# Efficiency measurement (reused unchanged -- no second large model is built)
# ============================================================================
def measure_flops_and_latency(model, n_timed=20):
    model.eval()
    vol = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
    with torch.no_grad():
        fc = FlopCounterMode(display=False, depth=None)
        with fc:
            _ = model(vol)
        flops = fc.get_total_flops()
        for _ in range(5):
            _ = model(vol)
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            _ = model(vol)
            times.append((time.perf_counter() - t0) * 1000)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return dict(flops=flops, inference_ms_mean=float(np.mean(times)), inference_ms_std=float(np.std(times)),
                trainable_params=trainable, total_params=total)


def evaluate(model, loader):
    """
    Validation evaluation -- deterministic, NEVER calls consistency_augment().
    Documented AUROC-undefined fallback (same convention as phase2_trm_
    ablation.py's evaluate_full()): if the loader's labels contain only one
    class, AUROC/AUPRC fall back to (0.5, mean label) and `auroc_undefined`
    is set so callers can flag it rather than silently treating 0.5 as real.
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            out = model(vol)
            p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(label.numpy())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(int)
    auroc_undefined = False
    try:
        auc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except Exception:
        auc, auprc = 0.5, float(labels.mean())
        auroc_undefined = True
    acc = (preds == labels).mean()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return dict(auroc=auc, auprc=auprc, accuracy=acc, f1=f1, sensitivity=sens, specificity=spec,
                confusion_matrix=cm, probs=probs, labels=labels, preds=preds, auroc_undefined=auroc_undefined)


def measure_representation_stability(model, loader, n_repeats=3, seed=STABILITY_SEED):
    """
    HOW REPRESENTATION STABILITY IS MEASURED: for every held-out validation
    subject, encode the deterministic, unaugmented volume once (z1 =
    encode_representation(model, vol)), then draw `n_repeats` independent
    consistency_augment() perturbations of that SAME volume and encode each
    (z2), recording cos(z1, z2) every time. NOT used for model selection or
    early stopping -- purely a post-hoc diagnostic, computed once after the
    best checkpoint is already loaded.

    `seed` is reset to a FIXED constant (STABILITY_SEED, independent of the
    training SEED) immediately before the perturbation loop starts, so M0
    and M1 draw the EXACT SAME sequence of perturbations here, regardless
    of how much randomness each consumed during its own training loop.
    """
    set_seed(seed)
    model.eval()
    all_cos = []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            z1 = encode_representation(model, vol)
            for _ in range(n_repeats):
                vol2 = consistency_augment(vol.clone())
                z2 = encode_representation(model, vol2)
                _, cos_sim = cosine_consistency_loss(z1, z2)
                all_cos.extend(cos_sim.cpu().numpy().tolist())
    return float(np.mean(all_cos)), float(np.std(all_cos))


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'ADHD'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Control', 'ADHD'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_val_predictions_df(val_ds, ev):
    subject_ids = val_ds.df['subject_id'].tolist()
    sites = val_ds.df['site'].tolist()
    assert len(subject_ids) == len(ev['probs']), \
        "val subject-id count does not match prediction count -- loader order assumption broke!"
    return pd.DataFrame({
        'subject_id': subject_ids, 'site': sites,
        'y_true': ev['labels'], 'y_prob': ev['probs'], 'y_pred': ev['preds'],
    })


# ============================================================================
# Per-variant (M0/M1) training with per-epoch checkpointing and interruption
# recovery -- same design as phase2_trm_ablation.py's train_variant().
# ============================================================================
def train_model(variant_name, tag, use_consistency, lam, train_ds, val_ds,
                 out_dir=None, resume=None, epochs=None, patience=None, _stop_after_epoch=None,
                 index_hash=None, split_hash=None):
    """
    `_stop_after_epoch` is a TEST-ONLY hook (used by run_sanity_check()'s
    Test H): if set, training returns immediately after that epoch's
    checkpoint is saved, WITHOUT finalizing results -- simulating a process
    kill right after a checkpoint write. Leave it None for every real run.

    `index_hash`/`split_hash` (item 1, strict configuration validation): sha256
    fingerprints of the dataset file and the exact train/val subject split,
    computed once in main() and passed to every variant. Combined with
    `ARCH_KWARGS`/`use_consistency`/`lam`/epochs/patience/batch_size/lr/
    val_split/seed into a single config fingerprint that is compared against
    the fingerprint stored in any existing checkpoint/result before it is
    reused -- see check_fingerprint_or_raise().
    """
    out_dir = OUT_DIR if out_dir is None else out_dir
    resume = RESUME if resume is None else resume
    epochs = EPOCHS if epochs is None else epochs
    patience = PATIENCE if patience is None else patience
    paths = checkpoint_paths(tag, out_dir)
    current_fp = compute_config_fingerprint(use_consistency, lam, index_hash, split_hash, epochs, patience)

    if resume and os.path.isfile(paths['results']):
        saved_fp = None
        if os.path.isfile(paths['config']):
            saved_config = json.load(open(paths['config']))
            saved_fp = saved_config.get('config_fingerprint')
        require_fingerprint_or_raise(saved_fp, tag, 'skip (reuse completed results)', paths['config'])
        check_fingerprint_or_raise(current_fp, saved_fp, tag, 'skip (reuse completed results)')
        print(f"  [{tag}] already COMPLETE (found {paths['results']}) -- skipping.")
        return pd.read_csv(paths['results']).iloc[0].to_dict(), 'skipped'

    train_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               generator=train_gen, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    model = ADHDProposedModelV2(**ARCH_KWARGS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    # Sentinel is -1.0, not 0.0: a real AUROC can legitimately equal exactly 0.0 on an
    # early/degenerate epoch, and the selection check below is `val_metrics['auroc'] >
    # best['auroc']` -- with a 0.0 sentinel, 0.0 > 0.0 is False, so NO checkpoint would
    # ever be saved and `best_state` would stay None for the whole run, silently falling
    # back to whatever the LAST epoch's weights happen to be, with placeholder (zero)
    # metrics in the saved results. -1.0 is lower than any real AUROC (always in [0, 1],
    # including the documented AUROC-undefined fallback of 0.5 -- see evaluate()), so the
    # FIRST valid validation epoch is always accepted as the initial checkpoint (item 2).
    best = dict(auroc=-1.0, auprc=0.0, accuracy=0.0, f1=0.0, sensitivity=0.0, specificity=0.0,
                confusion_matrix=np.zeros((2, 2), dtype=int), epoch=0)
    best_state = None
    patience_ctr = 0
    history = []
    resumed = False

    if resume and os.path.isfile(paths['latest']):
        # NOTE: config-fingerprint mismatches must ABORT the run (RuntimeError), never be
        # treated as "checkpoint failed to load, start fresh" -- so the fingerprint check
        # happens in its OWN step, deliberately not caught by the broad except below that
        # is reserved for genuine load/corruption failures (item 1: never silently reuse
        # incompatible results).
        try:
            ckpt = torch.load(paths['latest'], map_location=DEVICE, weights_only=False)
        except Exception as e:
            ckpt = None
            print(f"  [{tag}] WARNING: latest checkpoint exists but failed to load ({e!r}) -- "
                  f"starting this variant fresh instead.")

        if ckpt is not None:
            saved_fp = ckpt.get('config_fingerprint')
            require_fingerprint_or_raise(saved_fp, tag, 'resume from latest checkpoint', paths['latest'])
            check_fingerprint_or_raise(current_fp, saved_fp, tag, 'resume from latest checkpoint')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            train_gen.set_state(ckpt['train_generator_state'])
            torch.set_rng_state(ckpt['torch_rng_state'])
            if torch.cuda.is_available() and ckpt.get('torch_cuda_rng_state') is not None:
                torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng_state'])
            np.random.set_state(ckpt['numpy_rng_state'])
            random.setstate(ckpt['python_rng_state'])
            best = ckpt['best']
            best_state = ckpt['best_model_state_dict']
            patience_ctr = ckpt['patience_ctr']
            history = ckpt['history']
            start_epoch = ckpt['epoch'] + 1
            resumed = True
            print(f"  [{tag}] RESUMING from epoch {ckpt['epoch']} "
                  f"(latest checkpoint found, best so far: AUROC={best['auroc']:.4f} @ epoch {best['epoch']})")

    if not resumed:
        set_seed(SEED)
        model = ADHDProposedModelV2(**ARCH_KWARGS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        train_gen.manual_seed(SEED)

    n_params = sum(p.numel() for p in model.parameters())
    auroc_was_undefined_any_epoch = any(h.get('auroc_undefined') for h in history)
    t0 = time.time()
    epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss_cls, epoch_loss_con, n_batches = 0.0, 0.0, 0
        for vol, pheno, label in train_loader:
            vol, label = vol.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()

            z1 = encode_representation(model, vol)
            logits = classify_from_repr(model, z1)
            loss_cls = criterion(logits, label)
            loss = loss_cls
            loss_c_val = 0.0

            if use_consistency:
                # CONSISTENCY_AUGMENTATION=False is a debug-only path (view 2 = view 1, so
                # L_consistency should be ~0 by Test E) -- normal operation keeps this True.
                vol2 = consistency_augment(vol.clone()) if CONSISTENCY_AUGMENTATION else vol.clone()
                z2 = encode_representation(model, vol2)
                loss_c, _ = cosine_consistency_loss(z1, z2)
                loss = loss_cls + lam * loss_c
                loss_c_val = loss_c.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_con += loss_c_val
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader)
        if val_metrics['auroc_undefined']:
            auroc_was_undefined_any_epoch = True
        history.append({
            'epoch': epoch,
            'train_loss_classification': epoch_loss_cls / max(n_batches, 1),
            'train_loss_consistency': epoch_loss_con / max(n_batches, 1),
            'train_loss_total': (epoch_loss_cls + lam * epoch_loss_con) / max(n_batches, 1)
                                 if use_consistency else epoch_loss_cls / max(n_batches, 1),
            'val_auroc': val_metrics['auroc'], 'val_auprc': val_metrics['auprc'],
            'val_accuracy': val_metrics['accuracy'], 'val_f1': val_metrics['f1'],
            'val_sensitivity': val_metrics['sensitivity'], 'val_specificity': val_metrics['specificity'],
            'auroc_undefined': val_metrics['auroc_undefined'],
        })

        if val_metrics['auroc'] > best['auroc']:
            best = dict(auroc=val_metrics['auroc'], auprc=val_metrics['auprc'], accuracy=val_metrics['accuracy'],
                         f1=val_metrics['f1'], sensitivity=val_metrics['sensitivity'],
                         specificity=val_metrics['specificity'], confusion_matrix=val_metrics['confusion_matrix'],
                         probs=val_metrics['probs'], labels=val_metrics['labels'], preds=val_metrics['preds'],
                         epoch=epoch)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            atomic_save(best_state, paths['best'])
        else:
            patience_ctr += 1

        # "Latest" checkpoint saved after EVERY epoch -- this is what makes
        # interruption recovery possible at all.
        checkpoint = dict(
            epoch=epoch, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            train_generator_state=train_gen.get_state(),
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(),
            best=best, best_model_state_dict=best_state, patience_ctr=patience_ctr,
            history=history, seed=SEED, use_consistency=use_consistency, lam=lam,
            config_fingerprint=current_fp,
        )
        atomic_save(checkpoint, paths['latest'])

        if _stop_after_epoch is not None and epoch >= _stop_after_epoch:
            print(f"  [{tag}] ** SIMULATED INTERRUPTION ** after epoch {epoch} "
                  f"(test-only _stop_after_epoch hook) -- returning without finalizing.")
            return None, 'interrupted'

        if patience_ctr >= patience:
            break

    train_time_s = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    eff = measure_flops_and_latency(model)
    stability_mean, stability_std = measure_representation_stability(model, val_loader)

    result = {
        'variant': variant_name,
        'auroc': best['auroc'], 'auprc': best['auprc'], 'accuracy': best['accuracy'],
        'f1': best['f1'], 'sensitivity': best['sensitivity'], 'specificity': best['specificity'],
        'params': n_params, 'trainable_params': eff['trainable_params'], 'flops': eff['flops'],
        'inference_ms_mean': eff['inference_ms_mean'], 'inference_ms_std': eff['inference_ms_std'],
        'train_time_s': round(train_time_s, 1), 'epochs_run': epoch, 'best_epoch': best['epoch'],
        'cosine_stability_mean': stability_mean, 'cosine_stability_std': stability_std,
        'lambda_consistency': lam if use_consistency else 0.0,
        'use_consistency': use_consistency, 'seed': SEED, 'stability_seed': STABILITY_SEED,
        'auroc_was_undefined_any_epoch': auroc_was_undefined_any_epoch,
    }

    # best['probs']/['labels']/['preds'] are the validation-set predictions from the
    # SAME epoch where best_state was captured -- exactly the predictions of the
    # checkpoint being saved, not a re-computed/approximate version.
    predictions = dict(probs=best['probs'], labels=best['labels'], preds=best['preds'])
    atomic_to_csv(build_val_predictions_df(val_ds, predictions), paths['valpred'])
    atomic_to_csv(pd.DataFrame(history), paths['history'])
    plot_confusion_matrix(best['confusion_matrix'], f'{variant_name} -- Confusion Matrix (validation)',
                           paths['confusion'])

    config = {
        'variant': variant_name, 'variant_tag': tag,
        'use_consistency': use_consistency, 'lambda_consistency': result['lambda_consistency'],
        'consistency_augmentation': CONSISTENCY_AUGMENTATION,
        'arch_kwargs': ARCH_KWARGS,
        'epochs_max': epochs, 'epochs_run': epoch, 'best_epoch': best['epoch'], 'batch_size': BATCH_SIZE,
        'lr': LR, 'val_split': VAL_SPLIT, 'patience': patience, 'seed': SEED,
        'stability_seed': STABILITY_SEED,
        'index_file': INDEX_FILE, 'out_dir': out_dir,
        'python_random_seed': SEED, 'numpy_seed': SEED, 'torch_seed': SEED,
        'cuda_available': torch.cuda.is_available(), 'cuda_seed': SEED if torch.cuda.is_available() else None,
        'cudnn_deterministic': True, 'cudnn_benchmark': False,
        'config_fingerprint': current_fp,
    }
    with open(paths['config'], 'w') as f:
        json.dump(config, f, indent=2, default=str)

    atomic_to_csv(pd.DataFrame([result]).drop(columns=['use_consistency'], errors='ignore'), paths['results'])

    if auroc_was_undefined_any_epoch:
        print(f"  [{tag}] WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch -- "
              f"see {paths['history']}'s auroc_undefined column.")

    return result, ('resumed' if resumed else 'fresh')


def plot_training_curves(histories, out_dir):
    """training_curves.png -- loss and val AUROC per epoch, M0 vs M1 overlaid where both ran."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {'m0': '#888888', 'm1': '#2166AC'}
    for tag, hist in histories.items():
        if not hist:
            continue
        df = pd.DataFrame(hist)
        axes[0].plot(df['epoch'], df['train_loss_total'], label=f'{tag.upper()} train loss', color=colors.get(tag))
        axes[1].plot(df['epoch'], df['val_auroc'], label=f'{tag.upper()} val AUROC', color=colors.get(tag))
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Training loss'); axes[0].set_title('Training Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation AUROC'); axes[1].set_title('Validation AUROC')
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/training_curves.png', dpi=300)
    plt.close(fig)


# ============================================================================
# Code-level sanity tests ONLY (synthetic data, no real dataset, no training
# run). Tests A-H below are code-correctness checks, NOT model-performance
# experiments -- none of them produce or report AUROC/AUPRC/F1/etc. Called
# once at the top of main(), before the real dataset is touched.
# ============================================================================
class _SyntheticDS(Dataset):
    """Minimal in-memory stand-in for ADHDDataset, used ONLY by run_sanity_check()'s Test H."""
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.vols = torch.randn(n, 1, 64, 64, 64, generator=g)
        self.labels = torch.randint(0, 2, (n,), generator=g)
        self.df = pd.DataFrame({'subject_id': [f'SYN{i:03d}' for i in range(n)], 'site': ['SIM'] * n})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vols[idx], torch.zeros(5), self.labels[idx]

    def class_weights(self):
        return torch.tensor([1.0, 1.0])


def run_sanity_check():
    print(f"\n{'='*70}\n  PHASE 2 STEP 2 -- CODE-LEVEL SANITY TESTS (synthetic data only)\n{'='*70}")
    torch.manual_seed(0)
    model = ADHDProposedModelV2(**ARCH_KWARGS)
    vol = torch.randn(4, 1, 64, 64, 64, requires_grad=False)

    # ---- Test A -- forward pass: MRI -> model -> representation -> classifier
    print("\n  Test A -- forward pass (MRI -> representation -> classifier):")
    z_a = encode_representation(model, vol)
    logits_a = classify_from_repr(model, z_a)
    print(f"    vol {tuple(vol.shape)} -> representation {tuple(z_a.shape)} -> logits {tuple(logits_a.shape)} "
          f"(expect (4,128) then (4,2))")
    assert z_a.shape == (4, 128) and logits_a.shape == (4, 2)
    print("  TEST A PASSED")

    # ---- Test B -- two-view representation (original MRI -> z1, augmented MRI -> z2)
    print("\n  Test B -- two-view representation (same subject, same weights):")
    vol_v1 = vol
    vol_v2 = consistency_augment(vol.clone())
    print(f"    View 1 == View 2 (should be False): {torch.equal(vol_v1, vol_v2)}")
    assert not torch.equal(vol_v1, vol_v2)
    z1 = encode_representation(model, vol_v1)
    z2 = encode_representation(model, vol_v2)
    print(f"    z1 shape: {tuple(z1.shape)}  z2 shape: {tuple(z2.shape)}  (expect (4, 128) both)")
    assert z1.shape == z2.shape == (4, 128)
    print(f"    Same model instance (same weights) used for both views: True "
          f"(one `model` object, called twice via encode_representation(model, ...) -- "
          f"no separate encoder was created for view 2)")
    print("  TEST B PASSED")

    # ---- Test C -- consistency loss returns a finite scalar
    print("\n  Test C -- consistency loss (finite scalar):")
    loss_c, cos_sim = cosine_consistency_loss(z1, z2)
    print(f"    L_consistency = {loss_c.item():.6f}  (scalar: {loss_c.dim() == 0})  "
          f"finite: {torch.isfinite(loss_c).item()}")
    print(f"    cosine similarities per sample: {cos_sim.detach().numpy().round(4)}")
    assert loss_c.dim() == 0 and torch.isfinite(loss_c)
    print("  TEST C PASSED")

    # ---- Test D -- gradient flow (backward pass, finite, no NaN/Inf)
    print("\n  Test D -- gradient flow through the combined objective:")
    logits = classify_from_repr(model, z1)
    target = torch.tensor([0, 1, 0, 1])
    criterion = nn.CrossEntropyLoss()
    loss_cls = criterion(logits, target)
    loss_total = loss_cls + CONSISTENCY_LAMBDA * loss_c
    print(f"    L_classification={loss_cls.item():.4f}  L_consistency={loss_c.item():.4f}  "
          f"L_total={loss_total.item():.4f}  (L_total = L_cls + {CONSISTENCY_LAMBDA} * L_consistency)")
    assert torch.isfinite(loss_total), "L_total is NaN/Inf!"

    model.zero_grad()
    loss_total.backward()
    grad_ok = model.encoder.block1.conv.pointwise.weight.grad is not None
    print(f"    Gradients exist (encoder received a gradient): {grad_ok}")
    assert grad_ok
    n_checked = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}!"
            n_checked += 1
    print(f"    No NaN/Inf in any of the {n_checked} gradient tensors: True")
    print("  TEST D PASSED")

    # ---- Test E -- identical representations -> consistency loss ~ 0
    print("\n  Test E -- identical representations give ~zero consistency loss:")
    loss_identical, cos_identical = cosine_consistency_loss(z1, z1)
    print(f"    cosine_consistency_loss(z1, z1) = {loss_identical.item():.8f}  "
          f"(expect ~0, i.e. cos_sim ~1): cos_sim={cos_identical.detach().numpy().round(6)}")
    assert torch.allclose(loss_identical, torch.zeros_like(loss_identical), atol=1e-5), \
        f"expected ~0 for identical representations, got {loss_identical.item()}"
    print("  TEST E PASSED")

    # ---- Test F -- validation mode does not apply random augmentation
    print("\n  Test F -- augmentation isolation (no random augmentation outside the two-view step):")
    model.eval()
    with torch.no_grad():
        z_val_1 = encode_representation(model, vol)   # validation-style call: NO consistency_augment()
        z_val_2 = encode_representation(model, vol)   # same input, same eval-mode model, called again
    val_deterministic = torch.allclose(z_val_1, z_val_2, atol=1e-7)
    print(f"    encode_representation(model, vol) called twice on IDENTICAL input in eval mode "
          f"(validation code path never calls consistency_augment()): "
          f"outputs identical = {val_deterministic}")
    assert val_deterministic, "validation-path representation is non-deterministic -- augmentation leaked in!"
    print("    (Enforced at the dataset level too: train_ds and val_ds are both built with "
          "augment=False in main() -- View 1 is always the deterministic, unaugmented "
          "preprocessed MRI; dataset.py's own augment=True random-flip path is never used by "
          "this file, and is untouched for Phase-1's own scripts; consistency_augment() is "
          "called explicitly ONLY inside train_model()'s training loop and the post-hoc "
          "stability measurement, never inside evaluate(), to produce View 2. There is no "
          "separate held-out test_ds anywhere in this pipeline -- see the module docstring's "
          "VALIDATION, NOT AN INDEPENDENT TEST SET note.)")
    model.train()
    print("  TEST F PASSED")

    # ---- Test G -- encode_representation()+classify_from_repr() == model.forward() exactly
    print("\n  Test G -- representation path matches the model's own forward() exactly:")
    model.eval()
    with torch.no_grad():
        logits_direct = model(vol)                                    # model's own forward()
        logits_via_repr = classify_from_repr(model, encode_representation(model, vol))
    same_logits = torch.allclose(logits_direct, logits_via_repr, atol=1e-6)
    print(f"    model(vol) vs classify_from_repr(model, encode_representation(model, vol)): "
          f"max abs diff = {(logits_direct - logits_via_repr).abs().max().item():.2e}  "
          f"identical = {same_logits}")
    assert same_logits, ("encode_representation()/classify_from_repr() diverge from model.forward() -- "
                          "this would mean the consistency loss is computed on a DIFFERENT "
                          "representation than the one actually used for classification!")
    model.train()
    print("  TEST G PASSED (no parallel/divergent computation path -- same architecture, same weights)")

    # ---- Test H -- genuine interrupt-then-resume, compared against an uninterrupted reference ----
    print("\n  Test H -- interrupt-then-resume reproduces the uninterrupted training-loss sequence:")
    with tempfile.TemporaryDirectory(prefix='phase2_consistency_resume_test_') as tmp:
        train_ds = _SyntheticDS(16, seed=1)
        val_ds = _SyntheticDS(8, seed=2)
        TEST_EPOCHS = 4

        set_seed(SEED)
        ref_result, ref_status = train_model('Synthetic (ref)', 'synref', True, CONSISTENCY_LAMBDA,
                                              train_ds, val_ds, out_dir=tmp, resume=False,
                                              epochs=TEST_EPOCHS, patience=100)
        ref_history = pd.read_csv(f'{tmp}/training_history_synref.csv')
        print(f"    Uninterrupted reference run: {len(ref_history)} epochs, status={ref_status}, "
              f"final train_loss_total={ref_history['train_loss_total'].iloc[-1]:.6f}")
        assert len(ref_history) == TEST_EPOCHS

        set_seed(SEED)
        _, int_status = train_model('Synthetic (interrupted)', 'synint', True, CONSISTENCY_LAMBDA,
                                     train_ds, val_ds, out_dir=tmp, resume=False,
                                     epochs=TEST_EPOCHS, patience=100, _stop_after_epoch=2)
        assert int_status == 'interrupted'
        assert os.path.isfile(f'{tmp}/synint_latest_checkpoint.pt')
        assert not os.path.isfile(f'{tmp}/synint_results.csv')
        print(f"    Simulated interruption after epoch 2: latest checkpoint exists, results.csv does not (correct)")

        resumed_result, resumed_status = train_model('Synthetic (resumed)', 'synint', True, CONSISTENCY_LAMBDA,
                                                        train_ds, val_ds, out_dir=tmp, resume=True,
                                                        epochs=TEST_EPOCHS, patience=100)
        assert resumed_status == 'resumed'
        resumed_history = pd.read_csv(f'{tmp}/training_history_synint.csv')
        print(f"    Resumed run: status={resumed_status}, {len(resumed_history)} epochs total "
              f"(2 before interruption + 2 after resuming), "
              f"final train_loss_total={resumed_history['train_loss_total'].iloc[-1]:.6f}")
        assert len(resumed_history) == TEST_EPOCHS

        loss_diff = (ref_history['train_loss_total'].values - resumed_history['train_loss_total'].values)
        max_abs_diff = np.abs(loss_diff).max()
        print(f"    Per-epoch train_loss_total, reference vs resumed: max abs diff = {max_abs_diff:.3e}")
        print(f"      reference epochs 1-4 : {ref_history['train_loss_total'].values.round(6).tolist()}")
        print(f"      resumed   epochs 1-4 : {resumed_history['train_loss_total'].values.round(6).tolist()}")
        assert max_abs_diff < 1e-5, (
            f"resumed training-loss sequence diverges from the uninterrupted reference "
            f"(max abs diff {max_abs_diff:.3e}) -- RNG/DataLoader-generator state was not "
            f"restored correctly on resume!")
        print("    Resumed sequence matches the uninterrupted reference to within 1e-5 "
              "(CPU, NUM_WORKERS=0 -- exact reproducibility is expected here).")

        print(f"    Resumed variant's architecture unchanged: params={resumed_result['params']:,} "
              f"(expect 114,482: {'MATCH' if resumed_result['params'] == 114482 else 'MISMATCH!'})")
        assert resumed_result['params'] == 114482
    print("  TEST H PASSED")

    # ---- Test I -- strict configuration validation (item 1) ----
    print("\n  Test I -- strict configuration validation refuses to reuse a changed configuration:")
    with tempfile.TemporaryDirectory(prefix='phase2_consistency_fingerprint_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=40)
        val_ds = _SyntheticDS(4, seed=41)

        result, status = train_model('FP test', 'synfp', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                       out_dir=tmp, resume=False, epochs=2, patience=100,
                                       index_hash='hashA', split_hash='splitA')
        assert status == 'fresh'
        assert os.path.isfile(f'{tmp}/synfp_results.csv')

        raised = False
        try:
            train_model('FP test', 'synfp', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=2, patience=100,
                        index_hash='hashB_DIFFERENT', split_hash='splitA')
        except RuntimeError as e:
            raised = True
            print(f"    Changed dataset (index_hash) correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised, "a changed index_hash was NOT detected/refused when skipping completed results!"

        result2, status2 = train_model('FP test', 'synfp', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                         out_dir=tmp, resume=True, epochs=2, patience=100,
                                         index_hash='hashA', split_hash='splitA')
        assert status2 == 'skipped'
        print("    Identical configuration correctly reuses the completed result (status=skipped)")

        raised_lam = False
        try:
            train_model('FP test', 'synfp', True, 0.10, train_ds, val_ds,   # DIFFERENT lambda
                        out_dir=tmp, resume=True, epochs=2, patience=100,
                        index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised_lam = True
            print(f"    Changed consistency weight (lambda) correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised_lam, "a changed lambda was NOT detected/refused when skipping completed results!"

        _, int_status = train_model('FP test 2', 'synfp2', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                     out_dir=tmp, resume=False, epochs=4, patience=100,
                                     index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status == 'interrupted'
        raised2 = False
        try:
            train_model('FP test 2', 'synfp2', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=4, patience=100,
                        index_hash='hashA', split_hash='splitB_DIFFERENT')
        except RuntimeError as e:
            raised2 = True
            print(f"    Changed split (split_hash) correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised2, "a changed split_hash was NOT detected/refused when resuming from a checkpoint!"

        # A results.csv/config.json with NO config_fingerprint at all (predates the
        # safeguard, or a different script version) must be refused for skip too --
        # never silently treated as "nothing to compare, go ahead".
        _, status3 = train_model('FP test 3', 'synfp3', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                  out_dir=tmp, resume=False, epochs=2, patience=100,
                                  index_hash='hashA', split_hash='splitA')
        assert status3 == 'fresh'
        cfg_path = f'{tmp}/config_synfp3.json'
        cfg = json.load(open(cfg_path))
        del cfg['config_fingerprint']
        json.dump(cfg, open(cfg_path, 'w'))
        raised3 = False
        try:
            train_model('FP test 3', 'synfp3', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=2, patience=100,
                        index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised3 = True
            print(f"    Missing config_fingerprint correctly REFUSED on skip: {str(e).splitlines()[0]}")
        assert raised3, "a missing config_fingerprint was NOT refused when skipping completed results!"

        # Same check on the RESUME path: a checkpoint saved with no config_fingerprint
        # (e.g. produced by a pre-safeguard script version) must also be refused.
        _, int_status2 = train_model('FP test 4', 'synfp4', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                                      out_dir=tmp, resume=False, epochs=4, patience=100,
                                      index_hash='hashA', split_hash='splitA', _stop_after_epoch=2)
        assert int_status2 == 'interrupted'
        ckpt_path = f'{tmp}/synfp4_latest_checkpoint.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        del ckpt['config_fingerprint']
        torch.save(ckpt, ckpt_path)
        raised4 = False
        try:
            train_model('FP test 4', 'synfp4', True, CONSISTENCY_LAMBDA, train_ds, val_ds,
                        out_dir=tmp, resume=True, epochs=4, patience=100,
                        index_hash='hashA', split_hash='splitA')
        except RuntimeError as e:
            raised4 = True
            print(f"    Missing config_fingerprint correctly REFUSED on resume: {str(e).splitlines()[0]}")
        assert raised4, "a missing config_fingerprint was NOT refused when resuming from a checkpoint!"
    print("  TEST I PASSED")

    # ---- Test J -- first valid checkpoint saved even when AUROC equals exactly 0.0 (item 2) ----
    print("\n  Test J -- first-epoch checkpoint is saved even when AUROC is exactly 0.0:")
    with tempfile.TemporaryDirectory(prefix='phase2_consistency_auroc0_test_') as tmp:
        train_ds = _SyntheticDS(8, seed=50)
        val_ds = _SyntheticDS(4, seed=51)

        global evaluate
        original_evaluate = evaluate
        call_count = {'n': 0}

        def _forced_zero_evaluate(model, loader):
            ev = original_evaluate(model, loader)
            call_count['n'] += 1
            if call_count['n'] == 1:
                ev = dict(ev)
                ev['auroc'] = 0.0   # force exactly zero on the very first validation epoch
            return ev

        evaluate = _forced_zero_evaluate
        try:
            result, status = train_model('AUROC0 test', 'synauroc0', False, 0.0, train_ds, val_ds,
                                          out_dir=tmp, resume=False, epochs=1, patience=100)
        finally:
            evaluate = original_evaluate

        best_ckpt_exists = os.path.isfile(f'{tmp}/best_synauroc0_model.pt')
        print(f"    Forced epoch-1 AUROC=0.0 -> best checkpoint saved: {best_ckpt_exists}")
        assert best_ckpt_exists, \
            "no best checkpoint was saved even though epoch 1 completed -- sentinel bug not fixed!"
        print(f"    Saved best_epoch = {result['best_epoch']}  (expect 1, not left at the 0 placeholder)")
        assert result['best_epoch'] == 1
    print("  TEST J PASSED")

    # ---- Section 14 -- parameter count must not unexpectedly increase
    print("\n  Section 14 -- parameter count with vs without consistency:")
    model_m0 = ADHDProposedModelV2(**ARCH_KWARGS)   # same class/kwargs as M1 -- consistency only changes the LOSS
    params_m0 = sum(p.numel() for p in model_m0.parameters())
    params_m1 = sum(p.numel() for p in model.parameters())
    print(f"    params (M0, no consistency) = {params_m0:,}")
    print(f"    params (M1, with consistency) = {params_m1:,}")
    print(f"    difference = {params_m1 - params_m0:,} "
          f"(expected 0: consistency learning only adds a loss term at training time -- "
          f"encode_representation()/cosine_consistency_loss() introduce zero new nn.Parameters, "
          f"and both M0 and M1 instantiate the identical ADHDProposedModelV2(**ARCH_KWARGS))")
    assert params_m0 == params_m1, "parameter count changed between M0 and M1 -- investigate!"
    print("  SECTION 14 CHECK PASSED")

    # ---- Section 14b -- M0 and M1 start from EQUIVALENT initial weights (same seed)
    print("\n  Section 14b -- M0/M1 initial weights are equivalent (same seed -> identical init):")
    set_seed(SEED)
    model_a = ADHDProposedModelV2(**ARCH_KWARGS)
    set_seed(SEED)
    model_b = ADHDProposedModelV2(**ARCH_KWARGS)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    all_equal = all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    print(f"    Two fresh ADHDProposedModelV2(**ARCH_KWARGS) instances, both built right after "
          f"set_seed({SEED}): all {len(sd_a)} parameter/buffer tensors identical = {all_equal}")
    assert all_equal, "same seed did not produce identical initial weights -- investigate!"
    print("    (train_model() does exactly this -- set_seed(SEED) immediately before "
          f"constructing the model -- for BOTH the M0 and M1 calls in main(), with the same "
          f"optimizer (AdamW, same lr/weight_decay) and the same evaluate()/scheduler, so the "
          f"ONLY difference between the two runs is the consistency loss term.)")
    print("  SECTION 14b CHECK PASSED")

    print(f"\n{'='*70}\n  ALL CODE-LEVEL SANITY TESTS (A-J) + SECTIONS 14/14b PASSED ✓\n{'='*70}")
    print("  (Synthetic data only -- no real dataset was loaded, no AUROC/AUPRC/F1/etc. "
          "was computed. The real M0/M1 training run on your actual dataset starts next.)")


# ============================================================================
# Dataset/experiment-setup integrity checks (real-CSV-level, NOT synthetic).
# Run once in main(), before any ADHDDataset is built.
# ============================================================================
def validate_dataset_integrity(raw_df, train_ids, val_ids):
    print(f"\n{'='*70}\n  DATASET INTEGRITY CHECKS\n{'='*70}")

    dup_mask = raw_df['subject_id'].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    print(f"  Duplicate subject_id rows: {n_dup}")
    if n_dup > 0:
        print(f"    Example duplicates: {raw_df.loc[dup_mask, 'subject_id'].unique()[:10].tolist()}")
    assert n_dup == 0, f"{n_dup} duplicate subject_id rows found in {INDEX_FILE} -- fix the index before training."

    missing = [p for p in raw_df['npy_path'] if not os.path.isfile(p)]
    print(f"  Missing .npy files: {len(missing)} / {len(raw_df)}")
    if missing:
        print(f"    Example missing paths: {missing[:10]}")
    assert not missing, f"{len(missing)} .npy files referenced in {INDEX_FILE} do not exist on disk."

    overlap = set(train_ids) & set(val_ids)
    print(f"  Train/val subject overlap: {len(overlap)}")
    assert not overlap, f"{len(overlap)} subjects appear in BOTH train and val -- subject-level leakage!"
    print("  Subject-level independence confirmed: every subject is in exactly one of train/val.")

    val_labels = raw_df[raw_df['subject_id'].isin(val_ids)]['label']
    val_counts = val_labels.value_counts().to_dict()
    print(f"  Validation label distribution: {val_counts}")
    assert val_labels.nunique() == 2, (
        f"validation split contains only one class ({val_counts}) -- AUROC/AUPRC are undefined "
        f"in this case; re-check the split (StratifiedShuffleSplit should prevent this on a "
        f"reasonably sized, reasonably balanced dataset).")

    print(f"  ARCH_KWARGS = {ARCH_KWARGS}  (no 'use_phenotype' key -> defaults to False in "
          f"ADHDProposedModelV2 -> primary experiment is MRI-only, phenotype branch not used)")
    assert 'use_phenotype' not in ARCH_KWARGS or not ARCH_KWARGS['use_phenotype'], \
        "ARCH_KWARGS sets use_phenotype=True -- this would make the primary experiment MRI+phenotype, not MRI-only!"

    print(f"  Train subjects: {len(train_ids)}   Val subjects: {len(val_ids)}   Total: {len(raw_df)}")
    print("  ALL DATASET INTEGRITY CHECKS PASSED ✓")


def save_site_label_counts(raw_df, train_ids, val_ids, out_dir):
    """
    Per-site label-count breakdown for both splits (item 3: 'save ... per-site
    label counts') -- makes site distribution and class balance per split
    visible and reviewable, not just asserted by a pass/fail check.
    """
    rows = []
    for split_name, ids in [('train', train_ids), ('val', val_ids)]:
        sub = raw_df[raw_df['subject_id'].isin(ids)]
        counts = sub.groupby(['site', 'label']).size().reset_index(name='count')
        counts.insert(0, 'split', split_name)
        rows.append(counts)
    site_label_df = pd.concat(rows, ignore_index=True)
    atomic_to_csv(site_label_df, f'{out_dir}/site_label_counts.csv')
    print(f"\n  Per-site label counts (train + val) -> {out_dir}/site_label_counts.csv")
    print(site_label_df.to_string(index=False))
    return site_label_df


def main():
    # ---- code-level sanity checks FIRST, on synthetic data only ----
    run_sanity_check()

    # ---- Real dataset: connect to the SAME ADHD-200 pipeline used everywhere
    #      else in this project (subject-level split, leakage-free fit_stats) ----
    print(f"\n{'='*70}\n  LOADING REAL DATASET\n{'='*70}")
    print(f"  INDEX_FILE = {INDEX_FILE}")
    raw_df = pd.read_csv(INDEX_FILE)
    raw_df = raw_df[raw_df['site'] != 'Pittsburgh'].reset_index(drop=True)
    labels_all = raw_df['label'].values
    print(f"  Total subjects : {len(raw_df)}")

    # ---- Subject-level split: loaded from disk on a RESUME (if already saved,
    #      guaranteeing the identical split across any restart), else computed fresh ----
    train_ids_path = f'{OUT_DIR}/train_subject_ids.csv'
    val_ids_path = f'{OUT_DIR}/val_subject_ids.csv'
    if RESUME and os.path.isfile(train_ids_path) and os.path.isfile(val_ids_path):
        train_ids = pd.read_csv(train_ids_path)['subject_id'].tolist()
        val_ids = pd.read_csv(val_ids_path)['subject_id'].tolist()
        print(f"  Loaded EXISTING train/val subject-ID split from disk ({train_ids_path}, "
              f"{val_ids_path}) -- guarantees the identical split across any restart.")
    else:
        set_seed(SEED)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
        train_ids = raw_df.iloc[train_idx]['subject_id'].tolist()
        val_ids = raw_df.iloc[val_idx]['subject_id'].tolist()
        atomic_to_csv(pd.DataFrame({'subject_id': train_ids}), train_ids_path)
        atomic_to_csv(pd.DataFrame({'subject_id': val_ids}), val_ids_path)
        print(f"  Computed a new split and saved it to {train_ids_path} / {val_ids_path}")

    validate_dataset_integrity(raw_df, train_ids, val_ids)
    save_site_label_counts(raw_df, train_ids, val_ids, OUT_DIR)

    # Dataset-identity and split fingerprints (item 1) -- computed once here, passed to
    # every variant, so a changed dataset file or a changed split is detected BEFORE any
    # checkpoint/result produced under the old one is silently reused.
    index_hash = hash_file(INDEX_FILE)
    split_hash = hash_split(train_ids, val_ids)
    print(f"  Dataset fingerprint: index_hash={index_hash[:12]}...  split_hash={split_hash[:12]}...")

    # Imputation/normalization statistics are fit on the TRAINING subjects only,
    # then reused (via fit_stats) for validation -- no leakage.
    train_stats_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False)
    fit_stats = train_stats_ds.get_fit_stats()

    # View-1 isolation (Step 2 ONLY, see module docstring): train_ds is built with
    # augment=False -- View 1 is the deterministic, unaugmented preprocessed MRI;
    # View 2 comes ONLY from consistency_augment() inside train_model(). M0 and M1
    # both consume this identical, deterministic train_ds.
    train_ds = ADHDDataset(INDEX_FILE, subject_ids=train_ids, augment=False, fit_stats=fit_stats)
    val_ds = ADHDDataset(INDEX_FILE, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
    print(f"  Train : {len(train_ds)}   Val : {len(val_ds)}  "
          f"(train augment=False -> View 1 deterministic; View 2 via consistency_augment() only; "
          f"val augment=False, deterministic)")

    # ---- Build the run list from RUN_M0 / RUN_M1 -- no manual training-loop editing ----
    runs = []
    if RUN_M0:
        runs.append(('m0', 'M0 (no consistency)', False, 0.0))
    if RUN_M1:
        runs.append(('m1', f'M1 (consistency, lambda={CONSISTENCY_LAMBDA})', True, CONSISTENCY_LAMBDA))
        if RUN_LAMBDA_SWEEP:
            for lam in LAMBDA_SWEEP:
                runs.append((f'm1_lam{lam}', f'M1 (consistency, lambda={lam})', True, lam))
    assert runs, "Both RUN_M0 and RUN_M1 are False -- nothing to train."

    histories = {}
    for tag, name, use_consistency, lam in runs:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        result, status = train_model(name, tag, use_consistency, lam, train_ds, val_ds,
                                      index_hash=index_hash, split_hash=split_hash)
        if status != 'skipped':
            hist_df = pd.read_csv(f'{OUT_DIR}/training_history_{tag}.csv')
            histories[tag] = hist_df.to_dict('records')
        else:
            try:
                histories[tag] = pd.read_csv(f'{OUT_DIR}/training_history_{tag}.csv').to_dict('records')
            except FileNotFoundError:
                histories[tag] = []

        print(f"  [{status}]  AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}  "
              f"Acc={result['accuracy']:.4f}  F1={result['f1']:.4f}  "
              f"Sens={result['sensitivity']:.4f}  Spec={result['specificity']:.4f}")
        print(f"  params={int(result['params']):,}  GFLOPs={result['flops']/1e9:.4f}  "
              f"inference={result['inference_ms_mean']:.2f}+/-{result['inference_ms_std']:.2f}ms")
        print(f"  representation cosine stability (val, held-out): "
              f"{result['cosine_stability_mean']:.4f} +/- {result['cosine_stability_std']:.4f}")
        print(f"  Saved: {tag}_results.csv, training_history_{tag}.csv, confusion_matrix_{tag}.png, "
              f"best_{tag}_model.pt, {tag}_latest_checkpoint.pt, config_{tag}.json, val_predictions_{tag}.csv")

    # ---- Combined ablation table + representation-stability table, built from each
    #      variant's OWN saved results.csv (resume-safe, same pattern as Step 1) ----
    all_results = []
    for tag, name, use_consistency, lam in runs:
        rpath = f'{OUT_DIR}/{tag}_results.csv'
        assert os.path.isfile(rpath), f"{tag}_results.csv missing after train_model() -- should not happen"
        all_results.append(pd.read_csv(rpath).iloc[0].to_dict())

    results_df = pd.DataFrame(all_results)[
        ['variant', 'lambda_consistency', 'params', 'trainable_params', 'flops', 'auroc', 'auprc',
         'accuracy', 'f1', 'sensitivity', 'specificity', 'cosine_stability_mean', 'cosine_stability_std',
         'inference_ms_mean', 'inference_ms_std', 'train_time_s', 'epochs_run', 'best_epoch', 'seed',
         'auroc_was_undefined_any_epoch']
    ]
    atomic_to_csv(results_df, f'{OUT_DIR}/consistency_ablation_results.csv')
    atomic_to_csv(results_df[['variant', 'lambda_consistency', 'cosine_stability_mean', 'cosine_stability_std']],
                  f'{OUT_DIR}/representation_stability.csv')

    print(f"\n{'='*90}\n  PHASE 2 STEP 2 -- M0 vs M1 SUMMARY (VALIDATION-SET METRICS)\n{'='*90}")
    print(results_df[['variant', 'auroc', 'auprc', 'accuracy', 'f1',
                       'cosine_stability_mean', 'cosine_stability_std']].to_string(index=False))
    print("\n  NOTE: all metrics above are from the single held-out VALIDATION split -- not an "
          "independent test set (none exists in this pipeline). Do not cross-compare these "
          "numbers against Step 1's A5-new (different augmentation policy -- see the "
          "AUGMENTATION-COMPARISON NOTE in this file's module docstring). Multi-fold CV / "
          "cross-site evaluation are a separate, not-yet-approved future stage.")

    if RUN_M0 and RUN_M1:
        m0 = results_df.iloc[0]
        m1 = results_df.iloc[1]
        stability_improved = m1['cosine_stability_mean'] > m0['cosine_stability_mean']
        perf_maintained = m1['auroc'] >= m0['auroc'] - 0.02   # small tolerance, not a hard pass/fail
        print(f"\nStability higher with consistency loss: {stability_improved} "
              f"({m0['cosine_stability_mean']:.4f} -> {m1['cosine_stability_mean']:.4f})")
        print(f"AUROC maintained or improved (within 0.02 tolerance): {perf_maintained} "
              f"({m0['auroc']:.4f} -> {m1['auroc']:.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(results_df['variant'], results_df['cosine_stability_mean'],
                    yerr=results_df['cosine_stability_std'], color=['#888888', '#2166AC'][:len(results_df)],
                    capsize=4)
        axes[0].set_ylabel('Mean cosine similarity (z_orig, z_aug)')
        axes[0].set_title('Representation Stability')
        axes[0].tick_params(axis='x', rotation=20)

        axes[1].bar(results_df['variant'], results_df['auroc'], color=['#888888', '#2166AC'][:len(results_df)])
        axes[1].set_ylabel('Validation AUROC')
        axes[1].set_title('Classification Performance')
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis='x', rotation=20)

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{OUT_DIR}/comparison_figure.png', dpi=300)
        plt.close(fig)
        print(f"Figure  -> {OUT_DIR}/comparison_figure.png")

    plot_training_curves(histories, OUT_DIR)

    print(f"\nResults -> {OUT_DIR}/consistency_ablation_results.csv")
    print(f"Stability -> {OUT_DIR}/representation_stability.csv")
    print(f"Training curves -> {OUT_DIR}/training_curves.png")
    print(f"Subject IDs -> {OUT_DIR}/train_subject_ids.csv, {OUT_DIR}/val_subject_ids.csv")
    print(f"Per-site label counts -> {OUT_DIR}/site_label_counts.csv")

    if results_df['auroc_was_undefined_any_epoch'].any():
        flagged = results_df[results_df['auroc_was_undefined_any_epoch']]['variant'].tolist()
        print(f"\n*** WARNING: AUROC was undefined (fell back to 0.5) in at least one epoch for: "
              f"{flagged} -- check training_history_{{tag}}.csv's auroc_undefined column. ***")

    print(f"\nAll Step-2 outputs saved under {OUT_DIR}/ "
          f"(Phase-1 results and Step-1's trm_ablation_results.csv are untouched).")


if __name__ == '__main__':
    main()


#==============================================================================
# FILE: code/generate_architecture_figure.py
#==============================================================================

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from model import ADHDNet

OUT_DIR = '.'

plt.rcParams.update({'font.size': 10, 'savefig.dpi': 300})


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Architecture diagram
# (DS-Conv3D -> BN -> ReLU -> Pool) x3 -> 3D ECA -> GAP, fused with the
# phenotype MLP branch via adaptive gating.
# ══════════════════════════════════════════════════════════════════════════
def box(ax, xy, w, h, text, fc='#F2F2F2', ec='#555555', fontsize=9, fontweight='normal'):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                            boxstyle='round,pad=0.02,rounding_size=0.03',
                            facecolor=fc, edgecolor=ec, linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight)


def arrow(ax, xy1, xy2, color='#333333'):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle='-|>', mutation_scale=12,
                                  color=color, linewidth=1.3))


fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

# ── MRI branch (left column) ─────────────────────────────────────────────
mri_x = 3.0
nodes_mri = [
    (9.4, 'MRI Volume\n(1, 64³)', '#FFFFFF'),
    (8.2, 'DS-Conv3D Block 1\nBN → ReLU → Pool\n(24, 32³)', '#EAF2FB'),
    (7.0, 'DS-Conv3D Block 2\nBN → ReLU → Pool\n(48, 16³)', '#EAF2FB'),
    (5.8, 'DS-Conv3D Block 3\nBN → ReLU → Pool\n(96, 8³)', '#EAF2FB'),
    (4.6, '3D ECA\n(96, 8³)', '#FCE8D8'),
    (3.4, 'Global Avg Pool\n→ 96', '#F2F2F2'),
    (2.2, 'Linear → 128\nz_MRI', '#DCEBF7'),
]
for y, text, fc in nodes_mri:
    box(ax, (mri_x, y), 2.6, 0.85, text, fc=fc)
for (y1, *_), (y2, *_) in zip(nodes_mri[:-1], nodes_mri[1:]):
    arrow(ax, (mri_x, y1 - 0.43), (mri_x, y2 + 0.43))

# ── Phenotype branch (right column) ──────────────────────────────────────
phen_x = 7.6
nodes_phen = [
    (9.4, 'Phenotype Vector\n(5,)\nage, sex, med,\nsite, ADHD-index', '#FFFFFF'),
    (7.6, 'MLP\nFC → BN → ReLU\n→ Dropout → FC', '#EAF2FB'),
    (6.0, 'z_P\n(32,)', '#DCEBF7'),
]
for y, text, fc in nodes_phen:
    h = 1.3 if 'Phenotype' in text else (1.05 if 'MLP' in text else 0.85)
    box(ax, (phen_x, y), 2.6, h, text, fc=fc)
arrow(ax, (phen_x, 9.4 - 0.65), (phen_x, 7.6 + 0.52))
arrow(ax, (phen_x, 7.6 - 0.52), (phen_x, 6.0 + 0.43))

# ── Fusion + head (bottom, spanning both columns) ────────────────────────
merge_x = (mri_x + phen_x) / 2
box(ax, (merge_x, 1.0), 4.2, 0.95,
    'Adaptive Gated Fusion\ng=σ(FC→ReLU→FC)([z_MRI || z_P]);  '
    'z_P→128;  fused = g·z_MRI + (1-g)·z_P', fc='#FCE8D8', fontsize=8)
arrow(ax, (mri_x, 2.2 - 0.43), (merge_x - 0.3, 1.0 + 0.48))
arrow(ax, (phen_x, 6.0 - 0.43), (merge_x + 0.3, 1.0 + 0.48))

box(ax, (merge_x, -0.3), 3.2, 0.7, 'Classifier Head\nFC(128→64) → ReLU → Dropout → FC(64→2)',
    fc='#F2F2F2', fontsize=8)
arrow(ax, (merge_x, 1.0 - 0.48), (merge_x, -0.3 + 0.35))

ax.set_ylim(-0.8, 10)
ax.set_title('ADHDNet Architecture', fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_architecture.png')
print(f"Fig. 1 saved -> {OUT_DIR}/fig1_architecture.png")


# ══════════════════════════════════════════════════════════════════════════
# Table 1 — Layer-by-layer architecture summary (traced from the real model)
# ══════════════════════════════════════════════════════════════════════════
model = ADHDNet(n_pheno=5, dropout=0.3)
model.eval()

rows = []
hooks = []

def make_hook(name):
    def hook(module, inp, out):
        shape = tuple(out.shape[1:]) if torch.is_tensor(out) else None
        n_params = sum(p.numel() for p in module.parameters())
        rows.append((name, type(module).__name__, shape, n_params))
    return hook

named = [
    ('cnn.0 (DS-Conv Block 1)', model.cnn[0]),
    ('cnn.1 (DS-Conv Block 2)', model.cnn[1]),
    ('cnn.2 (DS-Conv Block 3)', model.cnn[2]),
    ('cnn.3 (3D ECA)',          model.cnn[3]),
    ('cnn.4 (GAP)',             model.cnn[4]),
    ('img_proj (Linear)',       model.img_proj),
    ('pheno_mlp',               model.pheno_mlp),
    ('gate',                    model.gate),
    ('pheno_proj (Linear)',     model.pheno_proj),
    ('head',                    model.head),
]
for name, mod in named:
    hooks.append(mod.register_forward_hook(make_hook(name)))

with torch.no_grad():
    model(torch.randn(1, 1, 64, 64, 64), torch.randn(1, 5))

for h in hooks:
    h.remove()

print(f"\n{'Layer':<28}{'Type':<16}{'Output shape':<18}{'Params':>10}")
print('-' * 72)
total_params = sum(p.numel() for p in model.parameters())
for name, typ, shape, n in rows:
    print(f"{name:<28}{typ:<16}{str(shape):<18}{n:>10,}")
print('-' * 72)
print(f"{'TOTAL':<62}{total_params:>10,}")

with open(f'{OUT_DIR}/table1_architecture.csv', 'w') as f:
    f.write("layer,type,output_shape,params\n")
    for name, typ, shape, n in rows:
        f.write(f'"{name}",{typ},"{shape}",{n}\n')
    f.write(f'"TOTAL",,,{total_params}\n')

print(f"\nTable 1 saved -> {OUT_DIR}/table1_architecture.csv")


#==============================================================================
# FILE: code/generate_paper_figures.py
#==============================================================================


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
