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
