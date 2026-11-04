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
