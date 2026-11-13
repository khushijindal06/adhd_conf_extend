"""
preprocessing_audit.py -- Phase 1 dataset & preprocessing audit module
=========================================================================

Implements Phase 1.1-1.3 of the architecture-improvement audit as
executable, read-only diagnostics (plus one training-side harness for
1.3), without changing prepare_adhd200.py / mri_preprocessing.py /
dataset.py / model.py. Preserves all existing code.

Section 1 -- affine/voxel-spacing consistency, and per-subject
            preprocessing-variant classification (1.1, 1.2)
Section 2 -- site/class/QC/missing-scan reporting + representative
            slice visualization (1.2)
Section 3 -- 64^3 vs 96^3 resolution-comparison harness, including
            gradient accumulation (1.3)
Section 4 -- run_sanity_check(): synthetic-only tests (no real dataset)

CODE-LEVEL FINDING, confirmed by reading _adhd200_combined_
prepare_adhd200.py / mri_preprocessing.py directly (not assumed):
  - `nib.as_closest_canonical()` only reorders/flips axes to RAS+; it is
    ORIENTATION canonicalization, not spatial REGISTRATION. The pipeline
    never reads an affine/zoom for resampling, never aligns to a named
    template (no "MNI"/"ICBM" anywhere in the code).
  - `resize_volume()` computes its zoom factors from voxel-COUNT ratios
    only (`target_size / size`), never from physical voxel spacing (mm).
    If two scans differ in native shape/spacing, resizing both to 64^3
    this way does NOT give them the same physical scale.
  - `SCAN_PATTERNS` has 5 fallback tiers, most-preferred first
    ("normalized_resampled_128_..." implies the dataset author already
    resampled those files to a common grid -- but nothing in this
    pipeline VERIFIES that, or records which tier was actually used per
    subject in the final processed_index.csv/dataset_preparation.json;
    the summary's `selected_variant` field is one static, hardcoded
    string, not a per-subject fact).
  - `qc_anatomical` IS captured into the index (from the phenotype
    files' `qcanatomical1`/`qcs1anat` columns) but is never read again
    anywhere in build_manifest()/prepare_volumes() to exclude anything
    -- confirmed: recorded, not enforced.
This module's Section 1/2 functions turn each of those into a concrete,
runnable check against the real dataset, rather than leaving them as
code-reading conclusions.
"""

import os
import re
import fnmatch
import numpy as np
import pandas as pd
import torch

try:
    import nibabel as nib
except ImportError:
    nib = None

# Same fallback patterns as _adhd200_combined_prepare_adhd200.py's
# SCAN_PATTERNS, duplicated here (this module is read-only audit code,
# not a dependency of the preparation script) so a subject's recorded
# `source_nifti` filename can be classified by which tier it matched.
SCAN_PATTERNS = (
    'normalized_resampled_128_*_T1_biascorr_brain.nii*',
    'normalized_resampled_128_*brain*.nii*',
    'normalized_*T1*brain*.nii*',
    '*T1*biascorr_brain.nii*',
    '*T1*brain*.nii*',
)
_PATTERN_REGEXES = [re.compile(fnmatch.translate(p)) for p in SCAN_PATTERNS]


# ============================================================================
# SECTION 1 -- affine/spacing consistency + preprocessing-variant classification
# ============================================================================

def classify_scan_pattern(path):
    """Which SCAN_PATTERNS tier (0 = most preferred) a source NIfTI
    filename matches, or None if none match."""
    name = os.path.basename(str(path))
    for i, rx in enumerate(_PATTERN_REGEXES):
        if rx.match(name):
            return i
    return None


def preprocessing_variant_distribution(index_df, source_nifti_col='source_nifti'):
    """
    Phase 1.2 ("selected MRI filename pattern counts" / "preprocessing
    variant distribution"): classifies every subject's recorded source
    NIfTI filename by SCAN_PATTERNS tier. `dataset_preparation.json`'s
    single hardcoded `selected_variant` string does not do this -- it
    states one preferred label regardless of which tier actually
    matched for each subject.
    """
    if source_nifti_col not in index_df.columns:
        raise ValueError(f"'{source_nifti_col}' column not found -- cannot classify preprocessing "
                          f"variant without each subject's source NIfTI path.")
    tiers = index_df[source_nifti_col].map(classify_scan_pattern)
    out = index_df.assign(scan_pattern_tier=tiers)
    overall = tiers.value_counts(dropna=False).sort_index()
    by_site = None
    if 'site' in index_df.columns:
        by_site = out.groupby(['site', 'scan_pattern_tier']).size().unstack(fill_value=0)
    return dict(
        overall_counts={(f'tier_{int(k)}' if pd.notna(k) else 'unmatched'): int(v)
                        for k, v in overall.items()},
        by_site=by_site,
        n_unmatched=int(tiers.isna().sum()),
        tier_labels={i: p for i, p in enumerate(SCAN_PATTERNS)},
        flag_nonuniform=bool(tiers.nunique(dropna=True) > 1),
        interpretation=(
            "flag_nonuniform=True means at least two different fallback filename "
            "patterns were actually selected across subjects, contradicting a single "
            "blanket 'selected_variant' label. Combine with audit_affine_consistency() "
            "to check whether the different tiers also differ in voxel spacing/shape "
            "(which would mean resizing to 64^3 by voxel count, not physical size, "
            "scales different subjects' anatomy by different amounts)."
        ),
    )


def inspect_nifti_header(path):
    if nib is None:
        raise ImportError("nibabel is required for inspect_nifti_header() -- pip install nibabel")
    img = nib.load(str(path))
    header = img.header
    zooms = tuple(float(z) for z in header.get_zooms()[:3])
    affine = np.asarray(img.affine)
    orientation = ''.join(nib.aff2axcodes(affine))
    return dict(
        path=str(path), shape=tuple(int(s) for s in img.shape[:3]),
        voxel_spacing_mm=zooms, orientation=orientation,
        affine=affine.tolist(), sform_code=int(header['sform_code']),
        qform_code=int(header['qform_code']),
    )


def audit_affine_consistency(paths, reference=None, spacing_tolerance_mm=0.05):
    """
    Phase 1.1, bullets 3/4/6: inspects each file's affine, voxel
    spacing, orientation, and shape, and checks whether they are
    CONSISTENT across files -- not whether any one of them is
    individually "correct". A dataset that is internally consistent,
    even if not registered to a named template, can still be resized
    uniformly; one that mixes spacings/shapes cannot be, since
    resize_volume() scales by voxel COUNT, not physical size.
    `reference` defaults to the first file's own stats.
    """
    records = [inspect_nifti_header(p) for p in paths]
    ref = reference or dict(voxel_spacing_mm=records[0]['voxel_spacing_mm'],
                             shape=records[0]['shape'], orientation=records[0]['orientation'])
    for r in records:
        r['spacing_matches_reference'] = all(
            abs(a - b) <= spacing_tolerance_mm for a, b in zip(r['voxel_spacing_mm'], ref['voxel_spacing_mm']))
        r['shape_matches_reference'] = (r['shape'] == tuple(ref['shape']))
        r['orientation_matches_reference'] = (r['orientation'] == ref['orientation'])
    n_spacing_mismatch = sum(not r['spacing_matches_reference'] for r in records)
    n_shape_mismatch = sum(not r['shape_matches_reference'] for r in records)
    n_orientation_mismatch = sum(not r['orientation_matches_reference'] for r in records)
    return dict(
        n_files=len(records), reference=ref,
        n_spacing_mismatch=n_spacing_mismatch, n_shape_mismatch=n_shape_mismatch,
        n_orientation_mismatch=n_orientation_mismatch,
        already_on_common_grid=(n_spacing_mismatch == 0 and n_shape_mismatch == 0),
        records=records,
        interpretation=(
            "already_on_common_grid=True means every file shares the reference's voxel "
            "spacing AND array shape -- consistent with (but not proof of) registration "
            "to a common template; same grid does not by itself guarantee the anatomy "
            "inside it is spatially aligned, only that it CAN be resized uniformly. "
            "already_on_common_grid=False means naive voxel-count resizing to 64^3 "
            "scales different subjects' physical anatomy by different amounts."
        ),
    )


# ============================================================================
# SECTION 2 -- site/class/QC/missing-scan reporting + visualization
# ============================================================================

def site_class_report(index_df, site_col='site', label_col='label'):
    """Phase 1.2: subjects per site, ADHD/control per site, sites with a
    missing or extremely underrepresented class flagged -- never
    silently dropped, only flagged."""
    counts = index_df.groupby([site_col, label_col]).size().unstack(fill_value=0)
    if set(counts.columns) >= {0, 1}:
        counts = counts.rename(columns={0: 'control', 1: 'adhd'})
    flagged = counts[(counts.get('control', 0) == 0) | (counts.get('adhd', 0) == 0)]
    return dict(counts=counts, sites_with_missing_class=flagged.index.tolist())


def qc_statistics(index_df, qc_col='qc_anatomical'):
    """Phase 1.2 QC stats + Phase 1.1 bullet 7's direct answer: is
    qc_anatomical used to exclude anything downstream, or only recorded?
    (Confirmed by code reading: only recorded -- see module docstring.)"""
    if qc_col not in index_df.columns:
        return dict(found=False)
    vc = index_df[qc_col].value_counts(dropna=False)
    return dict(
        found=True, value_counts={str(k): int(v) for k, v in vc.items()},
        n_missing=int(index_df[qc_col].isna().sum()),
        note=("Recorded in the index but not read anywhere in prepare_adhd200.py's "
              "build_manifest()/prepare_volumes() to exclude a subject -- confirmed "
              "from code: QC is captured, not enforced."),
    )


def missing_mri_report(exclusions):
    """Tabulates the exclusion-reason records prepare_adhd200.py's
    build_manifest() already produces (e.g. reason='no_structural_scan')."""
    df = pd.DataFrame(exclusions)
    if df.empty:
        return dict(total=0, by_reason={}, by_site=None)
    by_site = df.groupby(['site', 'reason']).size().unstack(fill_value=0) if 'site' in df.columns else None
    return dict(total=len(df), by_reason=df['reason'].value_counts().to_dict(), by_site=by_site)


def save_representative_slices(index_df, out_dir, n_per_site=2, site_col='site',
                                npy_col='npy_path', seed=0):
    """
    Phase 1.2: representative MRI visualizations from different sites --
    axial/coronal/sagittal mid-slices from the ALREADY-PREPROCESSED .npy
    volumes (what the model actually sees), a few subjects per site.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for site, group in index_df.groupby(site_col):
        chosen = group.sample(n=min(n_per_site, len(group)), random_state=seed)
        for _, row in chosen.iterrows():
            vol = np.load(row[npy_col])
            d, h, w = vol.shape
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(vol[d // 2, :, :], cmap='gray'); axes[0].set_title('axial')
            axes[1].imshow(vol[:, h // 2, :], cmap='gray'); axes[1].set_title('coronal')
            axes[2].imshow(vol[:, :, w // 2], cmap='gray'); axes[2].set_title('sagittal')
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f"{row.get('subject_id', '?')} ({site})")
            fig.tight_layout()
            path = f'{out_dir}/{site}_{row.get("subject_id", "unknown")}.png'
            fig.savefig(path, dpi=120)
            plt.close(fig)
            saved.append(path)
    return saved


# ============================================================================
# SECTION 3 -- 64^3 vs 96^3 resolution-comparison harness
# ============================================================================

def resolution_cost_estimate(baseline_shape=(64, 64, 64), candidate_shape=(96, 96, 96)):
    """
    Phase 1.3: the architecture itself is resolution-agnostic by
    construction -- confirmed by reading proposed_model.py:
    LightweightEncoder's conv/pool stack and SpatialTokenizer's
    attention-pooling both operate on whatever spatial size reaches
    them (SpatialTokenizer flattens D*H*W generically via `.view(B,
    n_tokens, -1)` / einsum, with no hardcoded 8x8x8 assumption). So a
    96^3 experiment is a DATA/config change (a new, separately-prepared
    set of .npy volumes + a smaller batch size / gradient accumulation),
    not an architecture change. This function estimates the resulting
    compute cost ratio so the comparison's cost is known before running it.
    """
    voxel_ratio = np.prod(candidate_shape) / np.prod(baseline_shape)
    linear_ratio = (np.prod(candidate_shape) / np.prod(baseline_shape)) ** (1 / 3)
    return dict(
        baseline_shape=baseline_shape, candidate_shape=candidate_shape,
        voxel_count_ratio=float(voxel_ratio),
        approx_memory_ratio=float(voxel_ratio),
        approx_compute_ratio=float(voxel_ratio),
        linear_dimension_ratio=float(linear_ratio),
        note=(f"{candidate_shape} has {voxel_ratio:.3f}x the voxels of {baseline_shape} -- "
              f"expect roughly that much more activation memory and FLOPs per sample for "
              f"the same batch size. Recommended: divide batch size by ~{voxel_ratio:.1f} and "
              f"use gradient accumulation of the same factor to keep the EFFECTIVE batch "
              f"size (and therefore the optimization dynamics) comparable to the 64^3 run."),
    )


def train_one_epoch_with_grad_accumulation(model, train_loader, optimizer, criterion,
                                            accumulation_steps=1, device='cpu',
                                            clip_grad_norm=1.0):
    """
    Phase 1.3: a generic one-epoch training step with gradient
    accumulation, for use when a higher input resolution no longer fits
    the 64^3 batch size in GPU memory. Mathematically equivalent to
    training with batch_size * accumulation_steps in one step (same
    total gradient, averaged the same way), at the cost of
    accumulation_steps forward/backward passes per optimizer step
    instead of one. Model-agnostic: works for ADHDNet, ADHDNetMRIOnly,
    or ADHDProposedModelV2 (anything whose forward takes (vol, pheno)).
    """
    model.train()
    optimizer.zero_grad()
    total_loss, n_batches = 0.0, 0
    for i, (vol, pheno, label) in enumerate(train_loader):
        vol, pheno, label = vol.to(device), pheno.to(device), label.to(device)
        logits = model(vol, pheno)
        loss = criterion(logits, label) / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
    if n_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / max(n_batches, 1)


# ============================================================================
# SECTION 4 -- synthetic-only sanity checks (no real dataset touched)
# ============================================================================

def _make_fake_nifti(path, shape=(64, 64, 64), zooms=(3.0, 3.0, 3.0), seed=0):
    """Writes a minimal, valid synthetic NIfTI file -- random data, a
    simple diagonal affine with the given voxel spacing, RAS-canonical
    orientation already. Used only to exercise inspect_nifti_header()/
    audit_affine_consistency() without any real patient data."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32)
    affine = np.diag(list(zooms) + [1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _make_synthetic_index(root, n_subjects=18, sites=('SiteA', 'SiteB'), seed=0):
    """Real on-disk synthetic processed_index.csv + .npy volumes +
    source NIfTI files (with deliberately MIXED spacing/pattern tiers
    across subjects), exercising every Section 1/2 function end to end."""
    rng = np.random.default_rng(seed)
    nifti_dir = f'{root}/raw_nifti'
    npy_dir = f'{root}/npy'
    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    rows = []
    for i in range(n_subjects):
        sid = f'SUB{i:04d}'
        site = sites[i % len(sites)]
        label = int(i % 2 == 0)
        # every 3rd subject gets a DIFFERENT (coarser) voxel spacing and a
        # LOWER-preference filename pattern, simulating a fallback-tier scan
        if i % 3 == 0:
            zooms, fname = (1.0, 1.0, 1.0), f'{sid}_T1_brain.nii.gz'
        else:
            zooms, fname = (1.5, 1.5, 1.5), f'normalized_resampled_128_{sid}_T1_biascorr_brain.nii.gz'
        nifti_path = f'{nifti_dir}/{fname}'
        _make_fake_nifti(nifti_path, zooms=zooms, seed=i)
        npy_path = f'{npy_dir}/{sid}.npy'
        np.save(npy_path, rng.standard_normal((64, 64, 64)).astype(np.float32))
        rows.append(dict(subject_id=f'{site}_{sid}', label=label, site=site, npy_path=npy_path,
                          source_nifti=nifti_path, qc_anatomical=('pass' if i % 5 else np.nan)))
    df = pd.DataFrame(rows)
    index_path = f'{root}/processed_index.csv'
    df.to_csv(index_path, index=False)
    return index_path, df


def run_sanity_check():
    print(f"{'=' * 70}\n  PREPROCESSING AUDIT MODULE -- SYNTHETIC SANITY CHECKS\n{'=' * 70}")
    import tempfile

    # ---- Test A: NIfTI header inspection on a known-constructed file ----
    print("\n  Test A -- NIfTI header inspection against known ground truth:")
    with tempfile.TemporaryDirectory(prefix='preproc_audit_header_') as tmp:
        path = f'{tmp}/known.nii.gz'
        _make_fake_nifti(path, shape=(80, 90, 100), zooms=(2.0, 2.0, 2.0), seed=1)
        info = inspect_nifti_header(path)
        assert info['shape'] == (80, 90, 100)
        assert all(abs(z - 2.0) < 1e-6 for z in info['voxel_spacing_mm'])
        print(f"    shape={info['shape']}  voxel_spacing_mm={info['voxel_spacing_mm']}  "
              f"orientation={info['orientation']}  (matches known ground truth)")
    print("  TEST A PASSED")

    # ---- Test B: affine consistency -- correctly flags mixed spacing ----
    print("\n  Test B -- affine consistency check correctly flags mixed voxel spacing:")
    with tempfile.TemporaryDirectory(prefix='preproc_audit_affine_') as tmp:
        p_consistent = [f'{tmp}/c{i}.nii.gz' for i in range(4)]
        for p in p_consistent:
            _make_fake_nifti(p, zooms=(1.5, 1.5, 1.5))
        report_consistent = audit_affine_consistency(p_consistent)
        assert report_consistent['already_on_common_grid'] is True

        p_mixed = p_consistent + [f'{tmp}/mixed.nii.gz']
        _make_fake_nifti(p_mixed[-1], zooms=(3.0, 3.0, 3.0))
        report_mixed = audit_affine_consistency(p_mixed)
        assert report_mixed['already_on_common_grid'] is False
        assert report_mixed['n_spacing_mismatch'] == 1
        print(f"    4 identical-spacing files: already_on_common_grid={report_consistent['already_on_common_grid']}  "
              f"5 files (1 different spacing): already_on_common_grid={report_mixed['already_on_common_grid']}, "
              f"n_spacing_mismatch={report_mixed['n_spacing_mismatch']}")
    print("  TEST B PASSED")

    # ---- Test C: scan-pattern classification against known filenames ----
    print("\n  Test C -- scan-pattern tier classification against known filenames:")
    assert classify_scan_pattern('normalized_resampled_128_sub01_T1_biascorr_brain.nii.gz') == 0
    assert classify_scan_pattern('sub02_T1_brain.nii.gz') == 4
    assert classify_scan_pattern('completely_unrelated_file.nii.gz') is None
    print("    tier-0 (preferred), tier-4 (least-preferred fallback), and unmatched filenames "
          "all classified correctly")
    print("  TEST C PASSED")

    # ---- Test D: end-to-end against a synthetic index with mixed variants/sites ----
    print("\n  Test D -- end-to-end against a synthetic processed_index.csv:")
    with tempfile.TemporaryDirectory(prefix='preproc_audit_e2e_') as tmp:
        index_path, df = _make_synthetic_index(tmp, n_subjects=18, sites=('SiteA', 'SiteB'))

        variant_report = preprocessing_variant_distribution(df)
        assert variant_report['flag_nonuniform'] is True, \
            "the synthetic index deliberately mixes two filename-pattern tiers"
        print(f"    preprocessing_variant_distribution: {variant_report['overall_counts']}  "
              f"flag_nonuniform={variant_report['flag_nonuniform']}")

        affine_report = audit_affine_consistency(df['source_nifti'].tolist())
        assert affine_report['already_on_common_grid'] is False, \
            "the synthetic index deliberately mixes two voxel spacings"
        print(f"    audit_affine_consistency: n_spacing_mismatch={affine_report['n_spacing_mismatch']}/"
              f"{affine_report['n_files']}  already_on_common_grid={affine_report['already_on_common_grid']}")

        site_report = site_class_report(df)
        assert set(site_report['counts'].index) == {'SiteA', 'SiteB'}
        print(f"    site_class_report:\n{site_report['counts']}")

        qc_report = qc_statistics(df)
        assert qc_report['found'] is True
        assert qc_report['n_missing'] > 0
        print(f"    qc_statistics: {qc_report['value_counts']}  n_missing={qc_report['n_missing']}")

        exclusions = [dict(site='SiteA', scan_subject_id='9999', reason='no_structural_scan')]
        missing_report = missing_mri_report(exclusions)
        assert missing_report['total'] == 1
        print(f"    missing_mri_report: total={missing_report['total']}  by_reason={missing_report['by_reason']}")

        slice_paths = save_representative_slices(df, f'{tmp}/slices', n_per_site=1)
        assert len(slice_paths) == 2 and all(os.path.isfile(p) for p in slice_paths)
        print(f"    save_representative_slices: wrote {len(slice_paths)} figures")
    print("  TEST D PASSED")

    # ---- Test E: resolution cost estimate + gradient-accumulation training step ----
    print("\n  Test E -- resolution cost estimate + gradient-accumulation training step:")
    cost = resolution_cost_estimate()
    assert abs(cost['voxel_count_ratio'] - (96 ** 3) / (64 ** 3)) < 1e-6
    print(f"    64^3 -> 96^3: voxel_count_ratio={cost['voxel_count_ratio']:.3f}  "
          f"linear_dimension_ratio={cost['linear_dimension_ratio']:.3f}")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase2_model import ADHDProposedModelV2
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = ADHDProposedModelV2(trm_type='pairwise', use_attention=True)
    vol = torch.randn(8, 1, 64, 64, 64)
    pheno = torch.randn(8, 5)
    label = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(vol, pheno, label), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss = train_one_epoch_with_grad_accumulation(model, loader, optimizer, criterion,
                                                   accumulation_steps=4)
    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before
                  if after[k].dtype.is_floating_point)
    assert changed, "gradient-accumulation training step did not update any parameters"
    assert np.isfinite(loss)
    print(f"    one epoch (batch_size=2, accumulation_steps=4 -> effective batch 8): "
          f"loss={loss:.4f}  parameters updated={changed}")
    print("  TEST E PASSED")

    print(f"\n{'=' * 70}\n  ALL SYNTHETIC SANITY CHECKS (A-E) PASSED ✓\n{'=' * 70}")
    print("  (No real MRI data was used. nibabel-written synthetic NIfTI files and random "
          ".npy volumes only. Nothing under baseline/, phase2/, or phase2/consistency/ was "
          "read or written.)")


if __name__ == '__main__':
    run_sanity_check()
