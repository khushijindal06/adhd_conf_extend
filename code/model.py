
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
