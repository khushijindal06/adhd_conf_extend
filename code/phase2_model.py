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
