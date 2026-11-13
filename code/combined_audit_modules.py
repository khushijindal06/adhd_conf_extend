"""
COMBINED AUDIT MODULES -- single-file flattened reference copy
=================================================================

Concatenates the three audit modules from this round --
preprocessing_audit.py (Phase 1), architecture_audit.py (Phases 3-5),
experimental_audit.py (Phases 2/6/7) -- verbatim, in dependency order,
under "===== FILE: ... =====" section headers, so all three can be read
or submitted as supplementary material from ONE file.

KNOWN COLLISION if this file is imported or run as a single script:
all three modules define their own `run_sanity_check()` function and
their own `if __name__ == '__main__': run_sanity_check()` guard. In one
combined namespace, each later definition OVERWRITES the earlier one of
the same name -- so only experimental_audit.py's run_sanity_check()
(the last one in the file) would actually run; preprocessing_audit.py's
and architecture_audit.py's own synthetic test suites would be silently
shadowed, not executed. This mirrors the same reference-only caveat as
this repo's other flattened files (full_codebase.py, full_codebase_
notebook.py, notebook_cells_full.py).

DO NOT RUN THIS FILE DIRECTLY if you want all three test suites to
execute. Use the three files separately (as delivered, and as they sit
in code/ in this repo) -- each one's own
    python preprocessing_audit.py
    python architecture_audit.py
    python experimental_audit.py
runs that module's own synthetic sanity suite correctly. This combined
file exists only so a reviewer can scroll one document instead of
opening three.

Generated: 2026-11-13
Source files, in order: preprocessing_audit.py, architecture_audit.py, experimental_audit.py
"""



#==============================================================================
# FILE: preprocessing_audit.py
#==============================================================================

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


#==============================================================================
# FILE: architecture_audit.py
#==============================================================================

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


#==============================================================================
# FILE: experimental_audit.py
#==============================================================================

"""
experimental_audit.py -- Phase 2 Experimental Audit Module
=============================================================

A SEPARATE module implementing the audit recommendations on top of the
existing, unmodified pipeline:

  1. Baseline comparability audit        (Section 1)
  2. TRM variant verification             (Section 2)
  3. Representation quality / calibration analysis, M0 vs M1   (Section 3)
  4. Controlled comparison: repeated CV + leave-one-site-out,
     across all 7 MRI-only-vs-multimodal model specs           (Section 4)
  5. run_sanity_check() -- synthetic-only tests for everything above
     (Section 5)

PRESERVES all existing code, architectures, checkpoints and results:
  - imports ADHDNet / ADHDNetMRIOnly / ADHDProposedModelV2 UNMODIFIED
    from model.py / phase2_model.py -- no new model class is defined here
  - imports encode_representation / consistency_augment /
    cosine_consistency_loss / atomic_save / atomic_to_csv / hash_file /
    hash_split / set_seed / worker_init_fn UNMODIFIED from
    phase2_consistency.py -- the SAME functions that script's own
    training loop uses, not reimplementations
  - Section 4's harness writes ONLY to its own new output directory
    (OUT_DIR below, default '<DATA_ROOT>/exp_audit'), never to
    baseline/ , phase2/ , or phase2/consistency/
  - Sections 1-3 are READ-ONLY / INFERENCE-ONLY against existing outputs
    (configs, results CSVs, checkpoints, predictions) -- no training, no
    gradient, no architecture change

This module does NOT run real-data training on import. `main()` below is
guarded (`if __name__ == '__main__':`) and, even then, only calls
run_sanity_check() by default -- see main()'s own docstring for exactly
how (and why not yet) to point Section 4 at real data.
"""

import os
import json
import inspect
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                              precision_recall_curve, confusion_matrix, brier_score_loss)

from dataset import ADHDDataset
from model import ADHDNet, ADHDNetMRIOnly
from phase2_model import ADHDProposedModelV2
from phase2_consistency import (
    set_seed, worker_init_fn, atomic_save, atomic_to_csv, hash_file, hash_split,
    consistency_augment, encode_representation, classify_from_repr,
    cosine_consistency_loss, DEVICE, SEED, NUM_WORKERS,
)

OUT_DIR = os.environ.get('EXP_AUDIT_OUT_DIR', '/kaggle/working/exp_audit')
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 8
LR = 1e-4

PHENOTYPE_FEATURES = ['age', 'sex', 'med_status', 'site_id', 'adhd_index']


# ============================================================================
# SECTION 1 -- Baseline comparability audit
# ============================================================================

def document_model_inputs(model_cls, **kwargs):
    """
    Introspects an ACTUAL model class's forward() signature and a real
    instance's submodules -- a runtime fact about what the model
    consumes, not prose. Returns a dict.
    """
    sig = inspect.signature(model_cls.forward)
    params = [p for p in sig.parameters if p != 'self']
    model = model_cls(**kwargs)
    pheno_mlp = getattr(model, 'pheno_mlp', None)
    return dict(
        model_class=model_cls.__name__,
        forward_params=params,
        pheno_argument_accepted='pheno' in params,
        pheno_actually_wired_into_computation=pheno_mlp is not None,
        pheno_input_dim=(pheno_mlp[0].in_features if pheno_mlp is not None else None),
        n_params=sum(p.numel() for p in model.parameters()),
    )


def audit_baseline_inputs():
    """
    Item 1, bullets 1-2: document exactly what the original baseline
    consumes, and confirm the 5 phenotype features against the dataset
    contract, plus a genuinely MRI-only comparison point.
    """
    adhdnet_report = document_model_inputs(ADHDNet, n_pheno=5, dropout=0.3)
    mri_only_report = document_model_inputs(ADHDNetMRIOnly, n_pheno=5, dropout=0.3)
    proposed_report = document_model_inputs(ADHDProposedModelV2, trm_type='pairwise', use_attention=True)

    return dict(
        phenotype_features_in_dataset=PHENOTYPE_FEATURES,
        ADHDNet=adhdnet_report,
        ADHDNetMRIOnly_existing_class=mri_only_report,
        ADHDProposedModelV2_default=proposed_report,
        conclusion=(
            "ADHDNet wires phenotype into its forward computation (pheno_mlp exists; "
            "gated fusion combines it with the MRI embedding) -- it is multimodal. "
            "ADHDNetMRIOnly accepts a pheno argument for a drop-in-compatible forward "
            "signature but has no pheno_mlp/gate at all -- it is genuinely MRI-only. "
            "ADHDProposedModelV2 with use_phenotype=False (the default, and what every "
            "A3/A4/A4-new/A5-new/M0/M1 run uses) likewise has no pheno_mlp -- also "
            "genuinely MRI-only. Comparing ADHDNet's AUROC against any of these is "
            "comparing MRI+phenotype against MRI-only input information, not two "
            "architectures on identical inputs."
        ),
    )


def prepare_mri_only_baseline_run(index_file, out_dir, **kwargs):
    """
    Item 1, bullet 3: "prepare a comparison using the existing MRI-only
    ADHDNet class" -- this builds (but does not execute, per item 5) a
    ready-to-run fold spec for ADHDNetMRIOnly through the SAME
    train_one_fold() harness every other model in this module uses (see
    Section 4), so it is directly comparable rather than a bespoke
    one-off script. Returns the spec dict; call
    `run_controlled_comparison(..., model_names=['baseline_mri_only'])`
    to actually run it.
    """
    return dict(model_name='baseline_mri_only', index_file=index_file, out_dir=out_dir,
                spec=MODEL_SPECS['baseline_mri_only'], **kwargs)


def verify_split_identity(baseline_val_index_csv=None, other_val_subject_ids_csvs=None):
    """
    Item 1, bullet 4: direct, file-level confirmation that two or more
    pipeline stages evaluated on the EXACT SAME validation subjects --
    comparing subject_id SETS, not just matching counts. Read-only;
    raises nothing (this is a diagnostic report, not a gate).

    `other_val_subject_ids_csvs`: {stage_name: path to a CSV with a
    subject_id column -- that stage's val_subject_ids.csv or val_index.csv}.
    """
    other_val_subject_ids_csvs = other_val_subject_ids_csvs or {}
    report = {}
    baseline_ids = None
    if baseline_val_index_csv and os.path.isfile(baseline_val_index_csv):
        baseline_ids = set(pd.read_csv(baseline_val_index_csv)['subject_id'])
        report['baseline'] = dict(path=baseline_val_index_csv, n=len(baseline_ids))

    id_sets = {}
    for name, path in other_val_subject_ids_csvs.items():
        if not os.path.isfile(path):
            report[name] = dict(path=path, found=False)
            continue
        ids = set(pd.read_csv(path)['subject_id'])
        id_sets[name] = ids
        entry = dict(path=path, found=True, n=len(ids))
        if baseline_ids is not None:
            entry['identical_to_baseline'] = (ids == baseline_ids)
            entry['missing_from_this_stage'] = sorted(baseline_ids - ids)[:5]
            entry['extra_in_this_stage'] = sorted(ids - baseline_ids)[:5]
        report[name] = entry

    names = list(id_sets.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f'{a}_vs_{b}'] = (id_sets[a] == id_sets[b])
    report['pairwise_identical'] = pairwise
    return report


# ============================================================================
# SECTION 2 -- TRM variant verification
# ============================================================================

EXPECTED_TRM_KWARGS = {
    'A3':     dict(trm_type='none',     use_attention=False),
    'A4':     dict(trm_type='mha',      use_attention=False),
    'A4-new': dict(trm_type='pairwise', use_attention=False),
    'A5-new': dict(trm_type='pairwise', use_attention=True),
}
EXPECTED_TRM_PARAMS = {'A3': 105424, 'A4': 113968, 'A4-new': 113937, 'A5-new': 114482}


def verify_trm_variant(tag, out_dir, checkpoint_filename=None, config_filename=None,
                        results_filename=None):
    """
    Cross-checks ONE Step-1 variant's on-disk artifacts against the
    code-level expected configuration: "verify each variant's actual
    model class, configuration, parameter count, checkpoint ... and
    aggregate result row." Read-only; does not raise -- returns a report
    dict with an explicit `all_consistent` flag.
    """
    checkpoint_filename = checkpoint_filename or f'{tag}_best_model.pt'
    config_filename = config_filename or f'{tag}_config.json'
    results_filename = results_filename or f'{tag}_results.csv'

    report = dict(tag=tag, out_dir=out_dir)
    expected_kwargs = EXPECTED_TRM_KWARGS.get(tag)
    expected_params = EXPECTED_TRM_PARAMS.get(tag)
    report['expected_kwargs'] = expected_kwargs
    report['expected_params'] = expected_params
    if expected_kwargs is None:
        report['error'] = f"Unknown tag '{tag}' -- not one of {list(EXPECTED_TRM_KWARGS)}"
        report['all_consistent'] = False
        return report

    fresh_model = ADHDProposedModelV2(**expected_kwargs)
    fresh_params = sum(p.numel() for p in fresh_model.parameters())
    report['fresh_build_params'] = fresh_params
    report['fresh_build_matches_expected'] = (fresh_params == expected_params)

    config_path = os.path.join(out_dir, config_filename)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            saved_config = json.load(f)
        config_kwargs = saved_config.get('kwargs')
        report['config_path'] = config_path
        report['config_kwargs'] = config_kwargs
        report['config_kwargs_match_expected'] = (config_kwargs == expected_kwargs)
    else:
        report['config_path'] = None
        report['config_kwargs_match_expected'] = None

    ckpt_path = os.path.join(out_dir, checkpoint_filename)
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Only count entries that are actual trainable Parameters in a fresh
        # instance of the EXPECTED architecture, not buffers (BatchNorm
        # running_mean/running_var/num_batches_tracked etc.) -- buffers
        # inflate the raw state_dict sum well above `expected_params`
        # (which is sum(p.numel() for p in model.parameters())), which
        # would make an architecturally-correct checkpoint look wrong.
        param_keys = {k for k, _ in fresh_model.named_parameters()}
        ckpt_params = sum(v.numel() for k, v in state_dict.items() if k in param_keys)
        report['checkpoint_path'] = ckpt_path
        report['checkpoint_params'] = ckpt_params
        report['checkpoint_matches_expected'] = (ckpt_params == expected_params)
        try:
            fresh_model.load_state_dict(state_dict)
            report['checkpoint_loads_into_expected_architecture'] = True
        except Exception as e:
            report['checkpoint_loads_into_expected_architecture'] = False
            report['checkpoint_load_error'] = str(e)
    else:
        report['checkpoint_path'] = None
        report['checkpoint_matches_expected'] = None

    results_path = os.path.join(out_dir, results_filename)
    if os.path.isfile(results_path):
        row = pd.read_csv(results_path).iloc[0]
        results_params = int(row['params']) if 'params' in row else None
        report['results_path'] = results_path
        report['results_row'] = row.to_dict()
        report['results_params_match_expected'] = (results_params == expected_params)
    else:
        report['results_path'] = None
        report['results_params_match_expected'] = None

    checks = [report.get('fresh_build_matches_expected'),
              report.get('config_kwargs_match_expected'),
              report.get('checkpoint_matches_expected'),
              report.get('results_params_match_expected')]
    report['all_consistent'] = all(c is not False for c in checks) and any(c is True for c in checks)
    return report


def verify_all_trm_variants(out_dir, **kwargs):
    return {tag: verify_trm_variant(tag, out_dir, **kwargs) for tag in EXPECTED_TRM_KWARGS}


def check_log_header_result_alignment(log_text, variant_names):
    """
    Item 2, bullet 3, generalized: scans a raw console log for the
    failure MODE found in the real run (a variant header line dropped
    from the captured stream, so a later variant's result line appears
    to print under an earlier variant's header) -- so this is caught
    automatically rather than by eye. Returns a report dict; raises
    nothing.
    """
    lines = log_text.splitlines()
    header_positions = {}
    for name in variant_names:
        for i, line in enumerate(lines):
            if line.strip() == name:
                header_positions[name] = i
                break

    result_positions = [i for i, line in enumerate(lines)
                         if re.match(r'\s*\[(fresh|resumed|skipped[^\]]*)\]\s+AUROC=', line)]

    ordered_headers = sorted(header_positions.items(), key=lambda kv: kv[1])
    header_result_map = {}
    for idx, (name, pos) in enumerate(ordered_headers):
        next_pos = ordered_headers[idx + 1][1] if idx + 1 < len(ordered_headers) else len(lines)
        header_result_map[name] = sum(1 for p in result_positions if pos < p < next_pos)

    total_headers = len(header_positions)
    total_results = len(result_positions)
    missing_headers = [n for n in variant_names if n not in header_positions]

    return dict(
        variant_names=variant_names,
        headers_found=list(header_positions.keys()),
        headers_missing=missing_headers,
        total_header_lines_found=total_headers,
        total_result_lines_found=total_results,
        results_per_header=header_result_map,
        likely_console_log_artifact=(total_headers < len(variant_names) and total_results == len(variant_names)),
        note=(
            "likely_console_log_artifact=True means every variant's result line IS "
            "present (so training/checkpointing ran correctly for all variants) but "
            "fewer header lines were captured than variants -- exactly the failure "
            "mode found in the real run's log (A4's header line dropped, its result "
            "line appearing to print under A3's header instead)."
        ),
    )


# ============================================================================
# SECTION 3 -- Representation quality & calibration analysis (M0 vs M1)
# ============================================================================

def roc_pr_comparison(val_predictions_csvs):
    """Item 3, bullet 1. `val_predictions_csvs`: {name: path to a CSV
    with y_true/y_prob columns}. Read-only."""
    out = {}
    for name, path in val_predictions_csvs.items():
        df = pd.read_csv(path)
        y_true, y_prob = df['y_true'].values, df['y_prob'].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        out[name] = dict(
            fpr=fpr.tolist(), tpr=tpr.tolist(), auroc=float(roc_auc_score(y_true, y_prob)),
            precision=prec.tolist(), recall=rec.tolist(),
            auprc=float(average_precision_score(y_true, y_prob)), n=len(df),
        )
    return out


def calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Item 3, bullet 2: ECE, Brier score, and the raw reliability-diagram
    bin data. No threshold is tuned here -- this measures calibration of
    the probabilities as already produced.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    brier = float(brier_score_loss(y_true, y_prob))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bins.append(dict(bin=b, lo=float(bin_edges[b]), hi=float(bin_edges[b + 1]),
                              count=0, mean_confidence=None, empirical_accuracy=None))
            continue
        mean_conf = float(y_prob[mask].mean())
        emp_acc = float(y_true[mask].mean())
        ece += (count / len(y_true)) * abs(mean_conf - emp_acc)
        bins.append(dict(bin=b, lo=float(bin_edges[b]), hi=float(bin_edges[b + 1]),
                          count=count, mean_confidence=mean_conf, empirical_accuracy=emp_acc))
    return dict(ece=float(ece), brier_score=brier, n_bins=n_bins, bins=bins, n=len(y_true))


def plot_reliability_diagram(calibration_report, title, path):
    """Renders calibration_metrics()'s bin data as a reliability diagram
    PNG. matplotlib import is local so the rest of the module has no
    plotting dependency at import time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = [b for b in calibration_report['bins'] if b['count'] > 0]
    conf = [b['mean_confidence'] for b in bins]
    acc = [b['empirical_accuracy'] for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(conf, acc, 'o-', label='Observed')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Empirical accuracy (fraction positive)')
    ax.set_title(f"{title}\nECE={calibration_report['ece']:.4f}  Brier={calibration_report['brier_score']:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def probability_distributions(y_true, y_prob):
    """Item 3, bullet 3."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out = {}
    for label, name in [(0, 'control'), (1, 'adhd')]:
        vals = y_prob[y_true == label]
        out[name] = dict(
            n=int(len(vals)),
            mean=float(vals.mean()) if len(vals) else None,
            std=float(vals.std()) if len(vals) else None,
            min=float(vals.min()) if len(vals) else None,
            max=float(vals.max()) if len(vals) else None,
            median=float(np.median(vals)) if len(vals) else None,
        )
    return out


def per_class_metrics(y_true, y_pred):
    """Item 3, bullet 6: sensitivity/specificity/F1 per class, from the
    fixed 0.5-threshold predictions already saved -- no threshold tuning."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    precision_adhd = tp / (tp + fp) if (tp + fp) else None
    f1_adhd = (2 * precision_adhd * sensitivity / (precision_adhd + sensitivity)
               if precision_adhd and sensitivity and (precision_adhd + sensitivity) else None)
    precision_ctrl = tn / (tn + fn) if (tn + fn) else None
    f1_ctrl = (2 * precision_ctrl * specificity / (precision_ctrl + specificity)
               if precision_ctrl and specificity and (precision_ctrl + specificity) else None)
    return dict(confusion_matrix=cm.tolist(), sensitivity_adhd_recall=sensitivity,
                specificity_control_recall=specificity, precision_adhd=precision_adhd,
                f1_adhd=f1_adhd, precision_control=precision_ctrl, f1_control=f1_ctrl)


def pairwise_cosine_matrix(embeddings):
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


def effective_rank(embeddings):
    """Normalized singular-value entropy (Roy & Vetterli, 2007): a sharp
    drop relative to the embedding dimension is a collapse signal."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def extract_val_embeddings(model, val_ds, batch_size=BATCH_SIZE):
    """
    Item 3's "if embedding analysis requires actual MRI inference,
    implement that functionality so it can run on the existing
    checkpoints without retraining" -- this IS that functionality: a
    frozen forward pass (model.eval(), torch.no_grad()) through the
    model's own, unmodified encode_representation(). No gradient, no
    optimizer, no architecture change.

    `model` is moved to DEVICE here defensively (idempotent if it's
    already there) -- this function is meant to be callable directly
    against a freshly-loaded checkpoint, not only through
    representation_quality_report(), so it must not assume the caller
    remembered to do so.
    """
    model = model.to(DEVICE)
    model.eval()
    subject_ids = val_ds.df['subject_id'].tolist()
    labels = val_ds.labels.tolist()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_z = []
    with torch.no_grad():
        for vol, pheno, label in loader:
            vol = vol.to(DEVICE)
            z = encode_representation(model, vol)
            all_z.append(z.cpu().numpy())
    embeddings = np.concatenate(all_z, axis=0)
    assert embeddings.shape[0] == len(subject_ids)
    return embeddings, subject_ids, labels


def within_subject_stability(model, val_ds, n_repeats=5, stability_seed=12345, batch_size=BATCH_SIZE):
    """Item 3, bullet 4 (within-subject half): per-subject
    cos(z(original), z(consistency_augment(original))) -- same design as
    phase2_consistency.py's measure_representation_stability(), kept
    per-subject instead of collapsed to one mean. `model` is moved to
    DEVICE defensively, same reasoning as extract_val_embeddings()."""
    set_seed(stability_seed)
    model = model.to(DEVICE)
    model.eval()
    subject_ids = val_ds.df['subject_id'].tolist()
    labels = val_ds.labels.tolist()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
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
    df = pd.DataFrame({'subject_id': subject_ids, 'label': labels})
    for i in range(n_repeats):
        df[f'cos_rep{i + 1}'] = [row[i] for row in per_subject_cos]
    rep_cols = [f'cos_rep{i + 1}' for i in range(n_repeats)]
    df['cos_mean'] = df[rep_cols].mean(axis=1)
    df['cos_std'] = df[rep_cols].std(axis=1)
    return df


def between_subject_similarity(embeddings, labels):
    """Item 3, bullets 4 (between-subject half) and 5 (embedding
    variance / effective rank)."""
    labels = np.asarray(labels)
    cos_mat = pairwise_cosine_matrix(embeddings)
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same_class = labels[iu[0]] == labels[iu[1]]
    within_class_sims = cos_mat[iu][same_class]
    between_class_sims = cos_mat[iu][~same_class]
    return dict(
        n_subjects=n, embed_dim=embeddings.shape[1],
        mean_per_dim_variance=float(embeddings.var(axis=0).mean()),
        effective_rank=effective_rank(embeddings),
        mean_pairwise_cosine_within_class=float(within_class_sims.mean()),
        mean_pairwise_cosine_between_class=float(between_class_sims.mean()),
        class_separation_gap=float(within_class_sims.mean() - between_class_sims.mean()),
    )


def _cv_centroid_projection_auc(embeddings, labels, n_folds=5, seed=0):
    """
    Out-of-sample linear-centroid-projection AUC: fits the (centroid_1 -
    centroid_0) axis on a TRAIN fold only, scores a held-out fold with
    it, repeated across stratified folds and pooled. This avoids the
    in-sample optimism of fitting AND scoring the discriminant axis on
    the same data -- which is severely inflated whenever the embedding
    dimension is comparable to or larger than the sample size (exactly
    the regime these 128-d checkpoints run in: e.g. ~180 real validation
    subjects vs. 128 dimensions). An in-sample version of this check was
    caught producing a near-1.0 AUC on two classes drawn from the
    IDENTICAL distribution during this module's own testing (Test L) --
    this cross-validated version is the fix.
    """
    labels = np.asarray(labels)
    n_min_class = min((labels == 0).sum(), (labels == 1).sum())
    n_folds = max(2, min(n_folds, int(n_min_class)))
    if n_min_class < 2:
        return None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_true, all_score = [], []
    for train_idx, test_idx in skf.split(embeddings, labels):
        train_emb, train_lab = embeddings[train_idx], labels[train_idx]
        if (train_lab == 0).sum() == 0 or (train_lab == 1).sum() == 0:
            continue
        mu0 = train_emb[train_lab == 0].mean(axis=0)
        mu1 = train_emb[train_lab == 1].mean(axis=0)
        axis_unit = (mu1 - mu0) / (np.linalg.norm(mu1 - mu0) + 1e-12)
        all_true.extend(labels[test_idx].tolist())
        all_score.extend((embeddings[test_idx] @ axis_unit).tolist())
    if len(set(all_true)) < 2:
        return None
    return float(roc_auc_score(all_true, all_score))


def representation_collapse_diagnostics(embeddings, labels):
    """
    Distinguishes GLOBAL EMBEDDING ALIGNMENT (every representation pulled
    toward a similar shared direction/region, which mechanically inflates
    BOTH within- and between-class raw pairwise cosine similarity
    together) from GENUINE CLASS-DISCRIMINATION LOSS (the class-relevant
    signal itself destroyed). Five complementary views, as requested:

      1. raw pairwise cosine similarity (within vs between class) --
         same numbers between_subject_similarity() reports.
      2. CENTERED pairwise cosine similarity -- subtract the GLOBAL mean
         embedding from every sample first, removing any shared "DC"
         direction common to all embeddings, before computing cosine
         similarity. A shared direction inflates raw cosine similarity
         for every pair equally (within- and between-class alike); it
         does not depend on class and disappears once centered.
      3. class-CENTROID separation -- Euclidean distance and cosine
         similarity between the two classes' mean embeddings, plus a
         Fisher-like ratio (centroid distance / average within-class
         spread) that is scale- and shift-invariant -- adding a constant
         shift to every embedding does not move the centroids apart or
         together.
      4. embedding NORMS per class -- rules out "collapsed toward the
         origin" (near-zero vectors, where cosine similarity is
         numerically dominated by noise) as a trivial alternative
         explanation for a high raw between-class similarity.
      5. a CROSS-VALIDATED linear centroid-projection AUC -- fit the
         (centroid_1 - centroid_0) axis on a train fold only, score a
         held-out fold with it (via _cv_centroid_projection_auc(),
         stratified k-fold, pooled), same metric as the model's own
         classification AUROC. Deliberately NOT fit-and-scored on the
         same samples: an in-sample version of this was caught by this
         module's own Test L producing a near-1.0 AUC on two classes
         drawn from the IDENTICAL distribution (pure sampling noise in a
         ~128-dimensional space with a finite sample looks separable if
         you evaluate the discriminant on the data that defined it) --
         exactly the regime these checkpoints' 128-d embeddings run in
         (e.g. ~180 real validation subjects). Shift-invariant, like the
         centroid stats above. If this stays high (not near 0.5) even
         while raw between-class cosine similarity looks "collapsed",
         the class-relevant signal has NOT been destroyed -- it has
         simply stopped dominating the embedding's overall shared
         direction (global alignment), and a simple linear readout can
         still recover it out-of-sample.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)

    raw = between_subject_similarity(embeddings, labels)

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    centered_cos_mat = pairwise_cosine_matrix(centered)
    iu = np.triu_indices(n, k=1)
    same_class = labels[iu[0]] == labels[iu[1]]
    centered_within = centered_cos_mat[iu][same_class]
    centered_between = centered_cos_mat[iu][~same_class]

    has_both_classes = (labels == 0).any() and (labels == 1).any()
    centroid_report = dict(euclidean_distance=None, cosine_similarity=None,
                            within_class_spread=None, fisher_like_ratio=None)
    projection_auc = None
    if has_both_classes:
        mu0 = embeddings[labels == 0].mean(axis=0)
        mu1 = embeddings[labels == 1].mean(axis=0)
        centroid_euclidean_distance = float(np.linalg.norm(mu0 - mu1))
        centroid_cosine_similarity = float(
            (mu0 @ mu1) / (np.linalg.norm(mu0) * np.linalg.norm(mu1) + 1e-12))
        spreads = [np.linalg.norm(embeddings[labels == c] - embeddings[labels == c].mean(axis=0),
                                   axis=1).mean() for c in (0, 1)]
        within_class_spread = float(np.mean(spreads))
        centroid_report = dict(
            euclidean_distance=centroid_euclidean_distance,
            cosine_similarity=centroid_cosine_similarity,
            within_class_spread=within_class_spread,
            fisher_like_ratio=centroid_euclidean_distance / (within_class_spread + 1e-12),
        )
        try:
            projection_auc = _cv_centroid_projection_auc(embeddings, labels)
        except Exception:
            projection_auc = None

    norms = np.linalg.norm(embeddings, axis=1)
    norm_stats = dict(
        overall_mean_norm=float(norms.mean()), overall_std_norm=float(norms.std()),
        control_mean_norm=(float(norms[labels == 0].mean()) if (labels == 0).any() else None),
        adhd_mean_norm=(float(norms[labels == 1].mean()) if (labels == 1).any() else None),
    )

    return dict(
        raw_pairwise=dict(
            within_class=float(raw['mean_pairwise_cosine_within_class']),
            between_class=float(raw['mean_pairwise_cosine_between_class']),
            gap=float(raw['class_separation_gap']),
        ),
        centered_pairwise=dict(
            within_class=float(centered_within.mean()),
            between_class=float(centered_between.mean()),
            gap=float(centered_within.mean() - centered_between.mean()),
        ),
        centroid_separation=centroid_report,
        embedding_norms=norm_stats,
        linear_centroid_projection_auc=projection_auc,
        interpretation=(
            "If raw_pairwise shows high between-class similarity and a small gap, but "
            "centered_pairwise's gap is much closer to (or larger than) the raw gap, most "
            "of the apparent collapse is a shared GLOBAL direction common to every "
            "embedding (global alignment), not loss of class information -- centering "
            "removes the shared direction and recovers the real separation. "
            "fisher_like_ratio is scale- and shift-invariant: a value comparable to "
            "another model's despite very different raw cosine numbers means the classes "
            "are equally separable relative to their own spread in both cases. "
            "linear_centroid_projection_auc (cross-validated -- the discriminant axis is "
            "fit on a train fold and scored on a held-out fold, never the same samples) "
            "close to the model's own classification AUROC (not close to 0.5) means a "
            "simple linear readout can still recover the class label from the embedding "
            "OUT OF SAMPLE -- the discriminative signal survived, even if it no longer "
            "dominates the embedding's overall shape."
        ),
    )


def representation_quality_report(model, val_ds, val_predictions_csv=None, tag='model'):
    """
    Combines item 3 for ONE checkpoint: embedding-level diagnostics
    (always computed, inference-only) plus prediction-level diagnostics
    (ROC/PR/calibration/probability distributions/per-class metrics, if
    `val_predictions_csv` is given -- read directly, not recomputed).
    """
    embeddings, subject_ids, labels = extract_val_embeddings(model, val_ds)
    stability_df = within_subject_stability(model, val_ds)
    between_report = between_subject_similarity(embeddings, labels)
    collapse_report = representation_collapse_diagnostics(embeddings, labels)
    report = dict(
        tag=tag,
        embedding=dict(
            **between_report,
            mean_within_subject_stability=float(stability_df['cos_mean'].mean()),
            std_within_subject_stability=float(stability_df['cos_mean'].std()),
        ),
        collapse_diagnostics=collapse_report,
    )
    if val_predictions_csv and os.path.isfile(val_predictions_csv):
        df = pd.read_csv(val_predictions_csv)
        y_true, y_prob, y_pred = df['y_true'].values, df['y_prob'].values, df['y_pred'].values
        report['predictions'] = dict(
            roc_auc=float(roc_auc_score(y_true, y_prob)),
            auprc=float(average_precision_score(y_true, y_prob)),
            calibration=calibration_metrics(y_true, y_prob),
            probability_distributions=probability_distributions(y_true, y_prob),
            per_class_metrics=per_class_metrics(y_true, y_pred),
        )
    return report, embeddings, subject_ids, labels, stability_df


# ============================================================================
# SECTION 4 -- Controlled comparison harness (repeated CV + LOSO)
# ============================================================================

MODEL_SPECS = {
    'baseline_mri_only': dict(model_cls=ADHDNetMRIOnly, kwargs=dict(n_pheno=5, dropout=0.3),
                               use_consistency=False, lam=0.0, augment=True),
    'A3':     dict(model_cls=ADHDProposedModelV2, kwargs=dict(trm_type='none', use_attention=False),
                    use_consistency=False, lam=0.0, augment=True),
    'A4':     dict(model_cls=ADHDProposedModelV2, kwargs=dict(trm_type='mha', use_attention=False),
                    use_consistency=False, lam=0.0, augment=True),
    'A4-new': dict(model_cls=ADHDProposedModelV2, kwargs=dict(trm_type='pairwise', use_attention=False),
                    use_consistency=False, lam=0.0, augment=True),
    'A5-new': dict(model_cls=ADHDProposedModelV2, kwargs=dict(trm_type='pairwise', use_attention=True),
                    use_consistency=False, lam=0.0, augment=True),
    'M0':     dict(model_cls=ADHDProposedModelV2, kwargs=dict(trm_type='pairwise', use_attention=True),
                    use_consistency=False, lam=0.0, augment=False),
    'M1':     dict(model_cls=ADHDProposedModelV2, kwargs=dict(trm_type='pairwise', use_attention=True),
                    use_consistency=True, lam=0.05, augment=False),
}


def _build_model(spec):
    return spec['model_cls'](**spec['kwargs']).to(DEVICE)


def verify_phenotype_blindness(model_name_or_spec, batch_size=4, vol_shape=(1, 64, 64, 64)):
    """
    Item 3: a BEHAVIORAL (not just structural) proof that a given model
    spec is genuinely MRI-only. document_model_inputs() already confirms
    structurally that ADHDNetMRIOnly/ADHDProposedModelV2(use_phenotype=
    False) have no pheno_mlp submodule at all -- this goes one step
    further and proves the model's OUTPUT is actually invariant to the
    phenotype tensor's contents: builds the model fresh, runs the SAME
    random MRI volume through it twice with two wildly different,
    randomly-drawn phenotype tensors (standing in for age/sex/med_status/
    site_id/adhd_index), and asserts the two outputs are bit-identical.
    A model that is even accidentally phenotype-sensitive -- e.g. a
    future refactor that wires pheno in under a different flag -- would
    fail this immediately, rather than being caught only by a human
    reading the class definition.

    `model_name_or_spec` is either a MODEL_SPECS key or a raw
    {model_cls, kwargs} dict (so this can also be pointed at ADHDNet
    itself, as a negative control -- see Test M).
    """
    spec = MODEL_SPECS[model_name_or_spec] if isinstance(model_name_or_spec, str) else model_name_or_spec
    model = spec['model_cls'](**spec['kwargs']).to(DEVICE)
    model.eval()
    torch.manual_seed(0)
    vol = torch.randn(batch_size, *vol_shape, device=DEVICE)
    pheno_a = torch.randn(batch_size, 5, device=DEVICE)
    pheno_b = torch.randn(batch_size, 5, device=DEVICE) * 100.0 + 50.0  # wildly different values
    with torch.no_grad():
        out_a = model(vol, pheno_a)
        out_b = model(vol, pheno_b)
    identical = torch.equal(out_a, out_b)
    return dict(
        model_name=model_name_or_spec if isinstance(model_name_or_spec, str) else spec['model_cls'].__name__,
        phenotype_blind=identical,
        max_abs_diff=float((out_a - out_b).abs().max()),
    )


def _step_loss(model, spec, vol, pheno, label, criterion):
    if spec['use_consistency']:
        z1 = encode_representation(model, vol)
        logits = classify_from_repr(model, z1)
        loss_cls = criterion(logits, label)
        vol2 = consistency_augment(vol.clone())
        z2 = encode_representation(model, vol2)
        loss_con, _ = cosine_consistency_loss(z1, z2)
        return loss_cls + spec['lam'] * loss_con
    logits = model(vol, pheno)
    return criterion(logits, label)


def compute_fold_config_fingerprint(model_name, index_hash, split_hash, epochs, patience):
    spec = MODEL_SPECS[model_name]
    return dict(
        model_name=model_name, model_kwargs=spec['kwargs'], use_consistency=spec['use_consistency'],
        lam=spec['lam'], augment=spec['augment'], index_hash=index_hash, split_hash=split_hash,
        epochs_max=epochs, patience=patience, batch_size=BATCH_SIZE, lr=LR, seed=SEED,
    )


def check_fold_fingerprint_or_raise(current_fp, saved_fp, tag):
    """Same strict-validation design as phase2_trm_ablation.py /
    phase2_consistency.py's check_fingerprint_or_raise(), generalized to
    this module's fold fingerprint shape."""
    mismatches = {k: (saved_fp.get(k), current_fp[k]) for k in current_fp if saved_fp.get(k) != current_fp[k]}
    if mismatches:
        lines = "\n".join(f"    {k}: saved={old!r}  current={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"[{tag}] REFUSING to reuse a fold result -- configuration has changed since it "
            f"was produced:\n{lines}\n  Point OUT_DIR at a fresh, empty directory for this new "
            f"configuration."
        )


def require_fold_fingerprint_or_raise(saved_fp, tag, source_path):
    if saved_fp is None:
        raise RuntimeError(
            f"[{tag}] REFUSING to reuse -- {source_path} has no config_fingerprint (predates "
            f"this safeguard, or was produced by a different module version), so there is no "
            f"way to verify it matches the current configuration. Point OUT_DIR at a fresh, "
            f"empty directory."
        )


def train_one_fold(model_name, fold_tag, train_ds, val_ds, out_dir, index_hash, split_hash,
                    epochs=EPOCHS, patience=PATIENCE, resume=True, _stop_after_epoch=None):
    """
    Trains ONE model spec on ONE fold's train/val split; returns
    out-of-fold predictions for the val subjects plus the fold's
    aggregate result. Resumable and config-fingerprinted exactly like
    phase2_trm_ablation.py / phase2_consistency.py's per-variant training
    (refuses to reuse a fold result/checkpoint produced under a
    different -- or absent -- fingerprint).
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = MODEL_SPECS[model_name]
    paths = dict(
        latest=f'{out_dir}/{fold_tag}_latest_checkpoint.pt',
        best_model=f'{out_dir}/{fold_tag}_best_model.pt',
        best_preds=f'{out_dir}/{fold_tag}_best_preds.csv',
        results=f'{out_dir}/{fold_tag}_results.csv',
        config=f'{out_dir}/{fold_tag}_config.json',
        oof=f'{out_dir}/{fold_tag}_oof_predictions.csv',
    )
    current_fp = compute_fold_config_fingerprint(model_name, index_hash, split_hash, epochs, patience)

    if resume and os.path.isfile(paths['results']):
        saved_fp = None
        if os.path.isfile(paths['config']):
            saved_fp = json.load(open(paths['config'])).get('config_fingerprint')
        require_fold_fingerprint_or_raise(saved_fp, fold_tag, paths['config'])
        check_fold_fingerprint_or_raise(current_fp, saved_fp, fold_tag)
        return pd.read_csv(paths['results']).iloc[0].to_dict(), 'skipped'

    train_gen = torch.Generator()
    train_gen.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               generator=train_gen, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             worker_init_fn=worker_init_fn)

    cw = train_ds.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    set_seed(SEED)
    model = _build_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    best = dict(auroc=-1.0, epoch=0)
    best_probs, best_labels = None, None
    patience_ctr = 0

    if resume and os.path.isfile(paths['latest']):
        ckpt = torch.load(paths['latest'], map_location=DEVICE, weights_only=False)
        saved_fp = ckpt.get('config_fingerprint')
        require_fold_fingerprint_or_raise(saved_fp, fold_tag, paths['latest'])
        check_fold_fingerprint_or_raise(current_fp, saved_fp, fold_tag)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt['best']
        patience_ctr = ckpt['patience_ctr']
        # Phase 7, item 4/8: the best EPOCH's predictions must survive a resume even if no
        # epoch in THIS session beats it (best_probs/best_labels below are otherwise only
        # ever set inside the `if auroc > best['auroc']` branch of the loop below -- on a
        # resumed run whose remaining epochs never improve on the already-loaded best, that
        # branch never fires again, and best_probs/best_labels would stay None all the way
        # to the final OOF write, crashing there). Reload them from the sidecar written
        # alongside whichever checkpoint actually produced the CURRENT best['epoch'].
        if os.path.isfile(paths['best_preds']):
            best_preds_df = pd.read_csv(paths['best_preds'])
            best_probs, best_labels = best_preds_df['y_prob'].tolist(), best_preds_df['y_true'].tolist()
        print(f"  [{fold_tag}] RESUMING from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for vol, pheno, label in train_loader:
            vol, pheno, label = vol.to(DEVICE), pheno.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = _step_loss(model, spec, vol, pheno, label, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs, labels_list = [], []
        with torch.no_grad():
            for vol, pheno, label in val_loader:
                vol, pheno = vol.to(DEVICE), pheno.to(DEVICE)
                logits = model(vol, pheno)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
                labels_list.extend(label.numpy())
        try:
            auroc = roc_auc_score(labels_list, probs)
        except Exception:
            auroc = 0.5

        if auroc > best['auroc']:
            best = dict(auroc=auroc, epoch=epoch)
            best_probs, best_labels = list(probs), list(labels_list)
            patience_ctr = 0
            # Phase 7, item 3: best-model checkpoint saved SEPARATELY from the latest
            # (per-epoch, always-overwritten) checkpoint below -- matching the existing
            # convention in phase2_trm_ablation.py/phase2_consistency.py, which this
            # harness's earlier version did not (it only ever wrote `latest`, with no
            # standalone best-weights file). The sidecar prediction CSV lets a later
            # resume recover best_probs/best_labels without reloading/re-evaluating the
            # full model (see the RESUMING branch above).
            atomic_save({k: v.clone() for k, v in model.state_dict().items()}, paths['best_model'])
            atomic_to_csv(pd.DataFrame(dict(y_true=best_labels, y_prob=best_probs)), paths['best_preds'])
        else:
            patience_ctr += 1

        atomic_save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(),
                          scheduler_state_dict=scheduler.state_dict(),
                          best=best, patience_ctr=patience_ctr,
                          config_fingerprint=current_fp), paths['latest'])

        if _stop_after_epoch is not None and epoch >= _stop_after_epoch:
            return dict(auroc=best['auroc'], epoch=best['epoch']), 'interrupted'
        if patience_ctr >= patience:
            break

    val_ids = val_ds.df['subject_id'].tolist()
    if best_probs is None or best_labels is None:
        raise RuntimeError(
            f"[{fold_tag}] best_probs/best_labels are still unset after training -- this means "
            f"no epoch ever improved on the initial -1.0 AUROC sentinel, which should be "
            f"impossible for epochs >= 1 (see item 2: 'first-valid-checkpoint' sentinel fix "
            f"elsewhere in this package), or {paths['best_preds']} was missing on a resume with "
            f"zero remaining epochs. Investigate before trusting this fold's OOF predictions.")
    # THRESHOLD_SELECTION_POLICY (Phase 6): the decision threshold is the fixed constant
    # 0.5 below, applied identically to every fold of every model -- never fit, searched,
    # or adjusted using this (or any) held-out fold's own labels. The only thing chosen
    # using fold data is which EPOCH's checkpoint is "best" (by validation AUROC, which
    # is threshold-independent); the 0.5 cutoff itself is never tuned.
    best_preds = (np.array(best_probs) >= 0.5).astype(int)
    oof_df = pd.DataFrame(dict(subject_id=val_ids, y_true=best_labels, y_prob=best_probs,
                                y_pred=best_preds))
    atomic_to_csv(oof_df, paths['oof'])

    cm = confusion_matrix(best_labels, best_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision and sensitivity and (precision + sensitivity) else None)
    try:
        auprc = float(average_precision_score(best_labels, best_probs))
    except Exception:
        auprc = None
    accuracy = float(np.mean(np.array(best_labels) == best_preds))
    balanced_accuracy = (float((sensitivity + specificity) / 2)
                         if sensitivity is not None and specificity is not None else None)

    result = dict(model_name=model_name, fold_tag=fold_tag, auroc=best['auroc'], auprc=auprc,
                   accuracy=accuracy, balanced_accuracy=balanced_accuracy,
                   sensitivity=sensitivity, specificity=specificity, f1=f1,
                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   best_epoch=best['epoch'], n_val=len(val_ids),
                   params=sum(p.numel() for p in model.parameters()))
    atomic_to_csv(pd.DataFrame([result]), paths['results'])
    with open(paths['config'], 'w') as f:
        json.dump(dict(model_name=model_name, fold_tag=fold_tag, config_fingerprint=current_fp), f, indent=2)

    return result, 'fresh'


def build_repeated_cv_folds(raw_df, n_splits=5, n_repeats=2, seed=SEED, stratify_by_site=False,
                             min_per_combined_bin=2):
    """
    Item 4 / Phase 6: stratified REPEATED k-fold over subject-level
    labels. `stratify_by_site=True` stratifies by (label, site) jointly
    where every combined bin has at least `min_per_combined_bin`
    subjects (so every fold gets a representative mix of sites, not
    just diagnoses) -- falls back to label-only stratification (the
    default) for any site whose per-label counts are too small for
    joint stratification to be meaningful, logging which sites fell
    back rather than silently using one scheme for some subjects and
    another for others.
    """
    subject_ids = raw_df['subject_id'].values
    labels = raw_df['label'].values
    fallback_sites = []
    if stratify_by_site and 'site' in raw_df.columns:
        combined = raw_df['site'].astype(str) + '_' + raw_df['label'].astype(str)
        bin_counts = combined.value_counts()
        eligible_sites = set(raw_df['site'].unique())
        for site in raw_df['site'].unique():
            for label in (0, 1):
                key = f'{site}_{label}'
                if bin_counts.get(key, 0) < min_per_combined_bin:
                    eligible_sites.discard(site)
        fallback_sites = sorted(set(raw_df['site'].unique()) - eligible_sites)
        strat_labels = np.where(raw_df['site'].isin(eligible_sites), combined, raw_df['label'].astype(str)).astype(str)
    else:
        strat_labels = labels

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = []
    for i, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), strat_labels)):
        folds.append(dict(fold_id=f'cv{i:02d}', fold_type='cv',
                           train_ids=subject_ids[train_idx].tolist(),
                           val_ids=subject_ids[val_idx].tolist()))
    if stratify_by_site:
        for f in folds:
            f['stratified_by_site'] = True
            f['sites_fallen_back_to_label_only'] = fallback_sites
    return folds


def build_loso_folds(raw_df, min_per_class=5):
    """
    Item 4: leave-one-site-out, SKIPPING sites that don't have both
    diagnostic classes present in a usable count -- "handle sites
    containing only one diagnostic class appropriately" means excluding
    them from being a held-out fold (an AUROC on a single-class fold is
    undefined), not silently computing a meaningless number.
    """
    folds, skipped = [], []
    for site in sorted(raw_df['site'].unique()):
        site_df = raw_df[raw_df['site'] == site]
        counts = site_df['label'].value_counts()
        if counts.get(0, 0) < min_per_class or counts.get(1, 0) < min_per_class:
            skipped.append(dict(site=site, control=int(counts.get(0, 0)), adhd=int(counts.get(1, 0))))
            continue
        val_ids = site_df['subject_id'].tolist()
        train_ids = raw_df[raw_df['site'] != site]['subject_id'].tolist()
        folds.append(dict(fold_id=f'loso_{site}', fold_type='loso', train_ids=train_ids, val_ids=val_ids))
    return folds, skipped


def fold_distribution_report(fold, raw_df, id_col='subject_id', label_col='label', site_col='site'):
    """
    Phase 6: "Report the class distribution and site distribution for
    every fold." Returns train/val class counts and, if `site_col` is
    present, train/val site counts, for one fold dict (as produced by
    build_repeated_cv_folds()/build_loso_folds()). Also re-confirms no
    subject appears in both train and val (belt and suspenders on top
    of the fold builders' own construction).
    """
    train_df = raw_df[raw_df[id_col].isin(fold['train_ids'])]
    val_df = raw_df[raw_df[id_col].isin(fold['val_ids'])]
    overlap = set(fold['train_ids']) & set(fold['val_ids'])
    report = dict(
        fold_id=fold['fold_id'], fold_type=fold.get('fold_type', 'unknown'),
        n_train=len(train_df), n_val=len(val_df), subject_overlap=sorted(overlap),
        train_class_counts=train_df[label_col].value_counts().to_dict(),
        val_class_counts=val_df[label_col].value_counts().to_dict(),
    )
    if site_col in raw_df.columns:
        report['train_site_counts'] = train_df[site_col].value_counts().to_dict()
        report['val_site_counts'] = val_df[site_col].value_counts().to_dict()
    return report


def aggregate_fold_metrics(results_df, metrics=('auroc', 'auprc', 'accuracy', 'balanced_accuracy',
                                                 'sensitivity', 'specificity', 'f1'),
                            confidence=0.95):
    """
    Phase 6: mean/std/count + a normal-approximation confidence interval
    across folds, per model, computed SEPARATELY for repeated-CV and
    LOSO rows -- "Keep repeated-CV and LOSO results separate in all
    tables" -- never pooled into one number. Requires a `fold_type`
    column (both fold builders above tag it); raises rather than
    silently mixing if it's missing.
    """
    if 'fold_type' not in results_df.columns:
        raise ValueError("results_df has no 'fold_type' column -- cannot keep repeated-CV and "
                          "LOSO results separate without it.")
    from scipy import stats as _stats
    rows = []
    for fold_type, group in results_df.groupby('fold_type'):
        for model_name, model_group in group.groupby('model_name'):
            row = dict(fold_type=fold_type, model_name=model_name, n_folds=len(model_group))
            for m in metrics:
                if m not in model_group.columns:
                    continue
                vals = model_group[m].dropna().astype(float)
                if len(vals) == 0:
                    row[f'{m}_mean'] = row[f'{m}_std'] = row[f'{m}_ci_low'] = row[f'{m}_ci_high'] = None
                    continue
                mean, std, n = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, len(vals)
                if n > 1 and std > 0:
                    half_width = _stats.t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
                else:
                    half_width = 0.0
                row[f'{m}_mean'] = float(mean)
                row[f'{m}_std'] = float(std)
                row[f'{m}_ci_low'] = float(mean - half_width)
                row[f'{m}_ci_high'] = float(mean + half_width)
            rows.append(row)
    return pd.DataFrame(rows)


def run_controlled_comparison(index_file, out_dir, model_names=None, n_splits=5, n_repeats=2,
                               min_loso_per_class=5, epochs=EPOCHS, patience=PATIENCE,
                               site_filter=None, resume=True, _stop_after_epoch=None):
    """
    Item 4's full entrypoint: for each model in `model_names` (default:
    all of MODEL_SPECS), runs repeated-CV + LOSO folds, saving
    out-of-fold predictions and fold results under `out_dir` -- a NEW
    directory, never baseline/, phase2/, or phase2/consistency/.
    Train-only preprocessing stats (fit_stats) are recomputed per fold
    from that fold's own train_ids, exactly like every other script in
    this package -- no leakage. Resumable per fold via the same
    fingerprint-guarded checkpoint convention as the rest of Phase 2.

    THIS FUNCTION PERFORMS REAL TRAINING WHEN CALLED WITH REAL DATA AND A
    REAL EPOCH BUDGET. It is implemented and unit-tested here (see
    run_sanity_check()'s Tests I/J) but this module never invokes it with
    real data on its own -- see main()'s docstring.
    """
    model_names = model_names or list(MODEL_SPECS.keys())
    os.makedirs(out_dir, exist_ok=True)

    raw_df = pd.read_csv(index_file)
    if site_filter is not None:
        raw_df = raw_df[raw_df['site'] != site_filter].reset_index(drop=True)
    index_hash = hash_file(index_file)

    cv_folds = build_repeated_cv_folds(raw_df, n_splits=n_splits, n_repeats=n_repeats)
    loso_folds, skipped_sites = build_loso_folds(raw_df, min_per_class=min_loso_per_class)
    all_folds = cv_folds + loso_folds

    print(f"  {len(cv_folds)} repeated-CV folds + {len(loso_folds)} LOSO folds "
          f"({len(skipped_sites)} sites skipped for too few of one class: "
          f"{[s['site'] for s in skipped_sites]})")

    # Phase 6: class/site distribution for every fold, saved once (identical across
    # model_names, since every model trains on the SAME folds) -- not per model.
    distribution_rows = [fold_distribution_report(fold, raw_df) for fold in all_folds]
    atomic_to_csv(pd.DataFrame(distribution_rows), f'{out_dir}/fold_distributions.csv')
    for rep in distribution_rows:
        assert not rep['subject_overlap'], \
            f"fold {rep['fold_id']} has train/val subject overlap: {rep['subject_overlap']}"

    all_results = []
    for model_name in model_names:
        model_dir = f'{out_dir}/{model_name}'
        os.makedirs(model_dir, exist_ok=True)
        spec = MODEL_SPECS[model_name]
        for fold in all_folds:
            split_hash = hash_split(fold['train_ids'], fold['val_ids'])
            train_stats_ds = ADHDDataset(index_file, subject_ids=fold['train_ids'], augment=False)
            fit_stats = train_stats_ds.get_fit_stats()
            train_ds = ADHDDataset(index_file, subject_ids=fold['train_ids'],
                                    augment=spec['augment'], fit_stats=fit_stats)
            val_ds = ADHDDataset(index_file, subject_ids=fold['val_ids'],
                                  augment=False, fit_stats=fit_stats)
            result, status = train_one_fold(model_name, fold['fold_id'], train_ds, val_ds, model_dir,
                                             index_hash, split_hash, epochs=epochs, patience=patience,
                                             resume=resume, _stop_after_epoch=_stop_after_epoch)
            result = dict(result)
            result['model_name'] = model_name
            result['status'] = status
            result['fold_type'] = fold['fold_type']
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    atomic_to_csv(results_df, f'{out_dir}/controlled_comparison_all_folds.csv')
    # Phase 6: repeated-CV and LOSO are aggregated separately, never pooled into one row.
    summary_df = aggregate_fold_metrics(results_df)
    atomic_to_csv(summary_df, f'{out_dir}/controlled_comparison_summary.csv')
    return results_df, summary_df, skipped_sites


# ============================================================================
# SECTION 5 -- synthetic-only sanity checks (no real dataset touched)
# ============================================================================

def _make_synthetic_dataset(root, n_subjects=24, sites=('SiteA', 'SiteB', 'SiteOneClass'),
                             seed=0, vol_shape=(64, 64, 64)):
    """
    Builds a tiny, REAL on-disk synthetic dataset (actual .npy volumes +
    a real processed_index.csv), so sanity checks exercise the genuine
    ADHDDataset/DataLoader file-I/O path, not an in-memory mock. One site
    ('SiteOneClass') is deliberately given only control subjects, to
    exercise build_loso_folds()'s single-class-site exclusion (Test I).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(f'{root}/npy', exist_ok=True)
    rows = []
    for i in range(n_subjects):
        sid = f'SYN{i:04d}'
        site = sites[i % len(sites)]
        label = 0 if site == 'SiteOneClass' else int(i % 2 == 0)
        vol = rng.standard_normal(vol_shape).astype(np.float32)
        path = f'{root}/npy/{sid}.npy'
        np.save(path, vol)
        rows.append(dict(subject_id=sid, label=label, site=site, npy_path=os.path.abspath(path),
                          age=8 + (i % 10), sex=i % 2, med_status=1 + (i % 2),
                          site_id=i % len(sites), adhd_index=40 + (i % 20)))
    df = pd.DataFrame(rows)
    index_path = f'{root}/processed_index.csv'
    df.to_csv(index_path, index=False)
    return index_path, df


def run_sanity_check():
    """
    Synthetic-only tests for every section above (item 5: "First
    implement and test the audit utilities. Run lightweight synthetic
    checks only."). No real dataset is loaded; every temp directory is
    cleaned up automatically; nothing under baseline/, phase2/, or
    phase2/consistency/ is read or written.
    """
    import tempfile

    print(f"{'=' * 70}\n  EXPERIMENTAL AUDIT MODULE -- SYNTHETIC SANITY CHECKS\n{'=' * 70}")

    # ---- Test A: baseline input audit reflects the real model code ----
    print("\n  Test A -- baseline input audit (introspection, not prose):")
    report = audit_baseline_inputs()
    assert report['ADHDNet']['pheno_actually_wired_into_computation'] is True
    assert report['ADHDNetMRIOnly_existing_class']['pheno_actually_wired_into_computation'] is False
    assert report['ADHDProposedModelV2_default']['pheno_actually_wired_into_computation'] is False
    print(f"    ADHDNet multimodal={report['ADHDNet']['pheno_actually_wired_into_computation']}  "
          f"ADHDNetMRIOnly multimodal={report['ADHDNetMRIOnly_existing_class']['pheno_actually_wired_into_computation']}  "
          f"ADHDProposedModelV2(default) multimodal={report['ADHDProposedModelV2_default']['pheno_actually_wired_into_computation']}")
    print("  TEST A PASSED")

    # ---- Test B: split-identity verification detects matches and mismatches ----
    print("\n  Test B -- split identity verification (matching vs mismatched subject sets):")
    with tempfile.TemporaryDirectory(prefix='exp_audit_splittest_') as tmp:
        pd.DataFrame({'subject_id': ['S1', 'S2', 'S3']}).to_csv(f'{tmp}/baseline_val.csv', index=False)
        pd.DataFrame({'subject_id': ['S1', 'S2', 'S3']}).to_csv(f'{tmp}/matching_val.csv', index=False)
        pd.DataFrame({'subject_id': ['S1', 'S2', 'S4']}).to_csv(f'{tmp}/mismatched_val.csv', index=False)
        report = verify_split_identity(f'{tmp}/baseline_val.csv',
                                        {'stage_a': f'{tmp}/matching_val.csv',
                                         'stage_b': f'{tmp}/mismatched_val.csv'})
        assert report['stage_a']['identical_to_baseline'] is True
        assert report['stage_b']['identical_to_baseline'] is False
        assert report['stage_b']['missing_from_this_stage'] == ['S3']
        assert report['pairwise_identical']['stage_a_vs_stage_b'] is False
        print(f"    matching stage correctly identical: {report['stage_a']['identical_to_baseline']}  "
              f"mismatched stage correctly flagged: {not report['stage_b']['identical_to_baseline']}")
    print("  TEST B PASSED")

    # ---- Test C: TRM variant verification, end-to-end against a real (tiny) training run ----
    print("\n  Test C -- TRM variant verification against a real (synthetic-data) training run:")
    import phase2_trm_ablation as trm
    with tempfile.TemporaryDirectory(prefix='exp_audit_trmtest_') as tmp:
        train_ds = trm._SyntheticDS(8, seed=50)
        val_ds = trm._SyntheticDS(4, seed=51)
        trm.train_variant('A3 tokens+FastKAN', 'A3', EXPECTED_TRM_KWARGS['A3'], train_ds, val_ds,
                           out_dir=tmp, resume=False, epochs=1, patience=100,
                           index_hash='h', split_hash='s')
        report = verify_trm_variant('A3', tmp)
        assert report['fresh_build_matches_expected'] is True
        assert report['config_kwargs_match_expected'] is True
        assert report['checkpoint_matches_expected'] is True
        assert report['checkpoint_loads_into_expected_architecture'] is True
        assert report['results_params_match_expected'] is True
        assert report['all_consistent'] is True
        print(f"    A3's own real training-run artifacts all cross-check consistent: "
              f"{report['all_consistent']}")

        # Now corrupt the config to simulate a MISLABELED variant (A3's config.json
        # claims A4's kwargs) -- verify_trm_variant must catch it.
        cfg_path = f'{tmp}/A3_config.json'
        cfg = json.load(open(cfg_path))
        cfg['kwargs'] = EXPECTED_TRM_KWARGS['A4']
        json.dump(cfg, open(cfg_path, 'w'))
        bad_report = verify_trm_variant('A3', tmp)
        assert bad_report['config_kwargs_match_expected'] is False
        assert bad_report['all_consistent'] is False
        print(f"    Deliberately mislabeled config correctly caught: "
              f"config_kwargs_match_expected={bad_report['config_kwargs_match_expected']}")
    print("  TEST C PASSED")

    # ---- Test D: log header/result alignment catches the real failure mode ----
    print("\n  Test D -- log header/result alignment (the real run's A3/A4 artifact, reproduced):")
    variant_names = ['A3 tokens+FastKAN', 'A4 +MHA TRM (Phase 1)', 'A4-new +Pairwise TRM', 'A5-new +Pairwise TRM +AnatAttn']
    clean_log = "\n".join([
        "============================================================", variant_names[0],
        "============================================================", "  [fresh]  AUROC=0.70",
        "============================================================", variant_names[1],
        "============================================================", "  [fresh]  AUROC=0.70",
        "============================================================", variant_names[2],
        "============================================================", "  [fresh]  AUROC=0.69",
        "============================================================", variant_names[3],
        "============================================================", "  [fresh]  AUROC=0.68",
    ])
    clean_report = check_log_header_result_alignment(clean_log, variant_names)
    assert clean_report['likely_console_log_artifact'] is False
    assert clean_report['headers_missing'] == []

    # drop variant 2's header line, reproducing exactly what the real log showed
    broken_log = "\n".join([
        "============================================================", variant_names[0],
        "============================================================",
        "  [fresh]  AUROC=0.70  (actually A4's result)",
        "  [fresh]  AUROC=0.69",
        "============================================================", variant_names[2],
        "============================================================", "  [fresh]  AUROC=0.69",
        "============================================================", variant_names[3],
        "============================================================", "  [fresh]  AUROC=0.68",
    ])
    broken_report = check_log_header_result_alignment(broken_log, variant_names)
    assert broken_report['headers_missing'] == [variant_names[1]]
    assert broken_report['likely_console_log_artifact'] is True
    print(f"    Clean log: likely_console_log_artifact={clean_report['likely_console_log_artifact']}  "
          f"Broken log (header dropped): likely_console_log_artifact={broken_report['likely_console_log_artifact']}, "
          f"headers_missing={broken_report['headers_missing']}")
    print("  TEST D PASSED")

    # ---- Test E: calibration metrics + reliability diagram ----
    print("\n  Test E -- calibration metrics (ECE, Brier) + reliability diagram:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_caltest_') as tmp:
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 200)
        y_prob_perfect = y_true.astype(float) * 0.9 + 0.05
        y_prob_bad = np.full(200, 0.9)
        cal_good = calibration_metrics(y_true, y_prob_perfect)
        cal_bad = calibration_metrics(y_true, y_prob_bad)
        assert cal_good['ece'] < cal_bad['ece'], "well-separated probabilities should calibrate better than a constant 0.9"
        assert 0.0 <= cal_good['brier_score'] <= 1.0
        path = plot_reliability_diagram(cal_good, 'synthetic test', f'{tmp}/reliability.png')
        assert os.path.isfile(path)
        print(f"    ECE(well-separated)={cal_good['ece']:.4f} < ECE(constant 0.9)={cal_bad['ece']:.4f}: "
              f"{cal_good['ece'] < cal_bad['ece']}  reliability diagram written: {os.path.isfile(path)}")
    print("  TEST E PASSED")

    # ---- Test F: ROC/PR comparison, probability distributions, per-class metrics ----
    print("\n  Test F -- ROC/PR comparison, probability distributions, per-class metrics:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_rocpr_') as tmp:
        rng = np.random.default_rng(2)
        n = 60
        y_true = rng.integers(0, 2, n)
        y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.2, n) + 0.2, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        pd.DataFrame(dict(subject_id=[f'S{i}' for i in range(n)], y_true=y_true, y_prob=y_prob,
                           y_pred=y_pred)).to_csv(f'{tmp}/preds.csv', index=False)
        roc_report = roc_pr_comparison({'synthetic': f'{tmp}/preds.csv'})
        assert 0.0 <= roc_report['synthetic']['auroc'] <= 1.0
        assert len(roc_report['synthetic']['fpr']) == len(roc_report['synthetic']['tpr'])
        dist_report = probability_distributions(y_true, y_prob)
        assert set(dist_report.keys()) == {'control', 'adhd'}
        pc_report = per_class_metrics(y_true, y_pred)
        assert 0.0 <= pc_report['sensitivity_adhd_recall'] <= 1.0
        print(f"    AUROC={roc_report['synthetic']['auroc']:.4f}  AUPRC={roc_report['synthetic']['auprc']:.4f}  "
              f"sensitivity={pc_report['sensitivity_adhd_recall']:.4f}  specificity={pc_report['specificity_control_recall']:.4f}")
    print("  TEST F PASSED")

    # ---- Test G: embedding extraction + within/between-subject diagnostics (inference-only) ----
    print("\n  Test G -- embedding extraction + collapse diagnostics (inference-only, no training):")
    with tempfile.TemporaryDirectory(prefix='exp_audit_embedtest_') as tmp:
        index_path, _ = _make_synthetic_dataset(tmp, n_subjects=12, sites=('SiteA', 'SiteB'))
        val_ds = ADHDDataset(index_path, augment=False)
        model = ADHDProposedModelV2(trm_type='pairwise', use_attention=True).to(DEVICE)
        report, embeddings, subject_ids, labels, stability_df = representation_quality_report(
            model, val_ds, tag='synthetic')
        assert embeddings.shape == (12, 128)
        assert len(subject_ids) == 12
        assert 1 <= report['embedding']['effective_rank'] <= 128
        assert -1.0 <= report['embedding']['class_separation_gap'] <= 1.0
        assert stability_df.shape[0] == 12
        print(f"    embeddings shape={embeddings.shape}  effective_rank={report['embedding']['effective_rank']:.2f}  "
              f"class_separation_gap={report['embedding']['class_separation_gap']:.4f}  "
              f"within_subject_stability={report['embedding']['mean_within_subject_stability']:.4f}")
    print("  TEST G PASSED")

    # ---- Test H: fold trainer -- fresh / resume / fingerprint-mismatch / missing-fingerprint ----
    print("\n  Test H -- fold trainer: fresh run, skip-if-complete, fingerprint guards:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_foldtest_') as tmp_data, \
         tempfile.TemporaryDirectory(prefix='exp_audit_foldout_') as tmp_out:
        index_path, df = _make_synthetic_dataset(tmp_data, n_subjects=16, sites=('SiteA', 'SiteB'))
        train_ids = df['subject_id'].tolist()[:12]
        val_ids = df['subject_id'].tolist()[12:]
        fit_stats_ds = ADHDDataset(index_path, subject_ids=train_ids, augment=False)
        fit_stats = fit_stats_ds.get_fit_stats()
        train_ds = ADHDDataset(index_path, subject_ids=train_ids, augment=True, fit_stats=fit_stats)
        val_ds = ADHDDataset(index_path, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
        index_hash = hash_file(index_path)
        split_hash = hash_split(train_ids, val_ids)

        result, status = train_one_fold('A3', 'testfold', train_ds, val_ds, tmp_out,
                                         index_hash, split_hash, epochs=2, patience=100)
        assert status == 'fresh'
        assert os.path.isfile(f'{tmp_out}/testfold_oof_predictions.csv')
        oof = pd.read_csv(f'{tmp_out}/testfold_oof_predictions.csv')
        assert set(oof['subject_id']) == set(val_ids)

        result2, status2 = train_one_fold('A3', 'testfold', train_ds, val_ds, tmp_out,
                                           index_hash, split_hash, epochs=2, patience=100)
        assert status2 == 'skipped'
        print(f"    Fresh run: status={status}  Rerun (identical config): status={status2}")

        raised = False
        try:
            train_one_fold('A3', 'testfold', train_ds, val_ds, tmp_out,
                            'DIFFERENT_HASH', split_hash, epochs=2, patience=100)
        except RuntimeError as e:
            raised = True
            print(f"    Changed index_hash correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised, "a changed index_hash was NOT detected/refused!"

        cfg_path = f'{tmp_out}/testfold_config.json'
        cfg = json.load(open(cfg_path))
        del cfg['config_fingerprint']
        json.dump(cfg, open(cfg_path, 'w'))
        raised2 = False
        try:
            train_one_fold('A3', 'testfold', train_ds, val_ds, tmp_out,
                            index_hash, split_hash, epochs=2, patience=100)
        except RuntimeError as e:
            raised2 = True
            print(f"    Missing config_fingerprint correctly REFUSED: {str(e).splitlines()[0]}")
        assert raised2, "a missing config_fingerprint was NOT refused!"
    print("  TEST H PASSED")

    # ---- Test I: fold builders -- repeated CV shape + LOSO single-class exclusion ----
    print("\n  Test I -- repeated-CV fold count + LOSO single-class-site exclusion:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_foldbuild_') as tmp:
        index_path, df = _make_synthetic_dataset(tmp, n_subjects=30, sites=('SiteA', 'SiteB', 'SiteOneClass'))
        cv_folds = build_repeated_cv_folds(df, n_splits=3, n_repeats=2)
        assert len(cv_folds) == 6, f"expected 3*2=6 CV folds, got {len(cv_folds)}"
        for f in cv_folds:
            assert set(f['train_ids']).isdisjoint(f['val_ids'])

        loso_folds, skipped = build_loso_folds(df, min_per_class=2)
        loso_sites = {f['fold_id'] for f in loso_folds}
        assert 'loso_SiteOneClass' not in loso_sites, "single-class site must be EXCLUDED from LOSO"
        assert any(s['site'] == 'SiteOneClass' for s in skipped)
        assert 'loso_SiteA' in loso_sites and 'loso_SiteB' in loso_sites
        print(f"    {len(cv_folds)} CV folds (expected 6), LOSO sites included: {sorted(loso_sites)}, "
              f"skipped (single-class): {[s['site'] for s in skipped]}")
    print("  TEST I PASSED")

    # ---- Test J: run_controlled_comparison end-to-end on synthetic data ----
    print("\n  Test J -- run_controlled_comparison() end-to-end (tiny synthetic data, 1 model, 1 epoch):")
    with tempfile.TemporaryDirectory(prefix='exp_audit_e2e_data_') as tmp_data, \
         tempfile.TemporaryDirectory(prefix='exp_audit_e2e_out_') as tmp_out:
        index_path, df = _make_synthetic_dataset(tmp_data, n_subjects=20, sites=('SiteA', 'SiteB', 'SiteOneClass'))
        results_df, summary_df, skipped_sites = run_controlled_comparison(
            index_path, tmp_out, model_names=['A3'], n_splits=2, n_repeats=1,
            min_loso_per_class=2, epochs=1, patience=100)
        assert (results_df['status'] == 'fresh').all()
        assert not any(s == 'loso_SiteOneClass' for s in results_df['fold_tag'])
        assert os.path.isfile(f'{tmp_out}/controlled_comparison_all_folds.csv')
        assert os.path.isfile(f'{tmp_out}/controlled_comparison_summary.csv')
        n_cv = 2  # n_splits=2, n_repeats=1
        n_loso = 2  # SiteA, SiteB (SiteOneClass skipped)
        assert len(results_df) == n_cv + n_loso, f"expected {n_cv + n_loso} folds, got {len(results_df)}"
        print(f"    {len(results_df)} total folds run (expected {n_cv + n_loso}), "
              f"all fresh, SiteOneClass correctly excluded from LOSO")
        print(f"    Summary:\n{summary_df.to_string(index=False)}")
    print("  TEST J PASSED")

    # ---- Test K: every MODEL_SPEC trains one fold end-to-end, including the
    #      two code paths Test J alone doesn't reach: ADHDNetMRIOnly's own
    #      (vol, pheno) calling convention, and M1's two-view consistency loss ----
    print("\n  Test K -- every model spec (all 7) trains one fold end-to-end:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_allspecs_data_') as tmp_data, \
         tempfile.TemporaryDirectory(prefix='exp_audit_allspecs_out_') as tmp_out:
        index_path, df = _make_synthetic_dataset(tmp_data, n_subjects=16, sites=('SiteA', 'SiteB'))
        train_ids = df['subject_id'].tolist()[:12]
        val_ids = df['subject_id'].tolist()[12:]
        index_hash = hash_file(index_path)
        split_hash = hash_split(train_ids, val_ids)
        for model_name, spec in MODEL_SPECS.items():
            fit_stats = ADHDDataset(index_path, subject_ids=train_ids, augment=False).get_fit_stats()
            train_ds = ADHDDataset(index_path, subject_ids=train_ids, augment=spec['augment'], fit_stats=fit_stats)
            val_ds = ADHDDataset(index_path, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
            result, status = train_one_fold(model_name, f'{model_name}_fold', train_ds, val_ds, tmp_out,
                                             index_hash, split_hash, epochs=2, patience=100)
            assert status == 'fresh'
            assert os.path.isfile(f'{tmp_out}/{model_name}_fold_oof_predictions.csv')
            print(f"    {model_name:<20} params={result['params']:>7,}  auroc={result['auroc']:.4f}  status={status}")
    print("  TEST K PASSED")

    # ---- Test L: representation_collapse_diagnostics() on two engineered,
    #      known-ground-truth scenarios -- proves it actually discriminates
    #      "global alignment" from "genuine class-discrimination loss",
    #      not just that it runs without error ----
    print("\n  Test L -- collapse diagnostics distinguish global alignment from genuine collapse:")
    rng = np.random.default_rng(7)
    n_per_class = 40
    dim = 128

    # Scenario 1: GLOBAL ALIGNMENT. Two genuinely well-separated classes,
    # then a single large shared offset vector added to EVERY embedding
    # (simulating every representation being pulled toward a common
    # direction/region). Raw cosine similarity should look "collapsed"
    # (high between-class similarity, small raw gap) purely because of
    # the shared offset -- but centering removes it, and centroid
    # separation / the linear-projection AUC (both shift-invariant)
    # should reveal the original separation was never actually destroyed.
    mu0_true = rng.standard_normal(dim) * 0.3
    mu1_true = mu0_true + rng.standard_normal(dim) * 2.0  # well-separated true means
    class0 = mu0_true + rng.standard_normal((n_per_class, dim)) * 0.5
    class1 = mu1_true + rng.standard_normal((n_per_class, dim)) * 0.5
    shared_offset = rng.standard_normal(dim) * 20.0  # dominates raw cosine direction
    aligned_embeddings = np.concatenate([class0, class1]) + shared_offset
    aligned_labels = np.array([0] * n_per_class + [1] * n_per_class)

    aligned_report = representation_collapse_diagnostics(aligned_embeddings, aligned_labels)
    assert aligned_report['raw_pairwise']['between_class'] > 0.9, \
        "scenario setup check: the shared offset should make raw cosine similarity look collapsed"
    assert aligned_report['centered_pairwise']['gap'] > aligned_report['raw_pairwise']['gap'], \
        "centering must recover MORE separation than the raw (offset-dominated) similarity shows"
    assert aligned_report['linear_centroid_projection_auc'] > 0.95, \
        "a shift-invariant linear readout must still cleanly separate the two (truly separated) classes"
    print(f"    [global alignment] raw gap={aligned_report['raw_pairwise']['gap']:.4f}  "
          f"centered gap={aligned_report['centered_pairwise']['gap']:.4f}  "
          f"fisher_ratio={aligned_report['centroid_separation']['fisher_like_ratio']:.4f}  "
          f"projection_auc={aligned_report['linear_centroid_projection_auc']:.4f}  "
          f"(correctly identified as NOT genuinely collapsed)")

    # Scenario 2: GENUINE COLLAPSE. Both "classes" are drawn from the
    # IDENTICAL distribution (no real class signal at all) -- centering
    # must NOT manufacture separation that was never there, and the
    # linear-projection AUC must sit at chance.
    shared_mu = rng.standard_normal(dim) * 0.3
    collapsed_embeddings = shared_mu + rng.standard_normal((2 * n_per_class, dim)) * 0.5
    collapsed_labels = np.array([0] * n_per_class + [1] * n_per_class)

    collapsed_report = representation_collapse_diagnostics(collapsed_embeddings, collapsed_labels)
    assert collapsed_report['centroid_separation']['fisher_like_ratio'] < 0.3, \
        "genuinely identical-distribution classes must show a near-zero Fisher-like ratio"
    assert 0.3 < collapsed_report['linear_centroid_projection_auc'] < 0.7, \
        "genuinely identical-distribution classes must show a near-chance projection AUC"
    print(f"    [genuine collapse] raw gap={collapsed_report['raw_pairwise']['gap']:.4f}  "
          f"centered gap={collapsed_report['centered_pairwise']['gap']:.4f}  "
          f"fisher_ratio={collapsed_report['centroid_separation']['fisher_like_ratio']:.4f}  "
          f"projection_auc={collapsed_report['linear_centroid_projection_auc']:.4f}  "
          f"(correctly identified as genuinely collapsed -- no class signal to recover)")
    print("  TEST L PASSED")

    # ---- Test M: item 3 -- EVERY MRI-only model spec is behaviorally phenotype-blind,
    #      proven against a negative control (ADHDNet, which DOES use phenotype) ----
    print("\n  Test M -- phenotype blindness, behavioral proof (item 3), all 7 specs + negative control:")
    for model_name in MODEL_SPECS:
        result = verify_phenotype_blindness(model_name)
        assert result['phenotype_blind'] is True, \
            f"{model_name} is supposed to be MRI-only but its output changed with the phenotype input!"
        print(f"    {model_name:<20} phenotype_blind={result['phenotype_blind']}  "
              f"max_abs_diff={result['max_abs_diff']:.2e}")
    negative_control = verify_phenotype_blindness(dict(model_cls=ADHDNet, kwargs=dict(n_pheno=5, dropout=0.3)))
    assert negative_control['phenotype_blind'] is False, \
        "negative control FAILED: ADHDNet is supposed to BE phenotype-sensitive -- if this " \
        "assertion fires, verify_phenotype_blindness() itself has a bug (it isn't actually " \
        "testing anything, since even the known-multimodal model would pass)"
    print(f"    {'ADHDNet (negative control)':<20} phenotype_blind={negative_control['phenotype_blind']}  "
          f"max_abs_diff={negative_control['max_abs_diff']:.2e}  (correctly phenotype-SENSITIVE)")
    print("  TEST M PASSED")

    # ---- Test N: Phase 6 -- per-fold class/site distribution report + no-overlap check ----
    print("\n  Test N -- fold distribution report (class + site counts, no train/val overlap):")
    with tempfile.TemporaryDirectory(prefix='exp_audit_folddist_') as tmp:
        index_path, df = _make_synthetic_dataset(tmp, n_subjects=20, sites=('SiteA', 'SiteB', 'SiteOneClass'))
        cv_folds = build_repeated_cv_folds(df, n_splits=3, n_repeats=1)
        for fold in cv_folds:
            assert fold['fold_type'] == 'cv'
        loso_folds, _ = build_loso_folds(df, min_per_class=2)
        for fold in loso_folds:
            assert fold['fold_type'] == 'loso'
        report = fold_distribution_report(cv_folds[0], df)
        assert report['subject_overlap'] == []
        assert report['n_train'] + report['n_val'] == len(df)
        assert 'train_site_counts' in report and 'val_site_counts' in report
        print(f"    fold {report['fold_id']} ({report['fold_type']}): n_train={report['n_train']}  "
              f"n_val={report['n_val']}  subject_overlap={report['subject_overlap']}  "
              f"val_class_counts={report['val_class_counts']}")
    print("  TEST N PASSED")

    # ---- Test O: Phase 6 -- metric aggregation keeps CV and LOSO separate, with CIs ----
    print("\n  Test O -- aggregate_fold_metrics() keeps repeated-CV and LOSO separate:")
    synthetic_results = pd.DataFrame([
        dict(model_name='A3', fold_type='cv', auroc=0.70, auprc=0.6, accuracy=0.65,
             balanced_accuracy=0.65, sensitivity=0.6, specificity=0.7, f1=0.6),
        dict(model_name='A3', fold_type='cv', auroc=0.72, auprc=0.62, accuracy=0.66,
             balanced_accuracy=0.66, sensitivity=0.62, specificity=0.7, f1=0.61),
        dict(model_name='A3', fold_type='loso', auroc=0.55, auprc=0.5, accuracy=0.5,
             balanced_accuracy=0.5, sensitivity=0.5, specificity=0.5, f1=0.5),
    ])
    agg = aggregate_fold_metrics(synthetic_results)
    assert set(agg['fold_type']) == {'cv', 'loso'}
    cv_row = agg[(agg['fold_type'] == 'cv') & (agg['model_name'] == 'A3')].iloc[0]
    loso_row = agg[(agg['fold_type'] == 'loso') & (agg['model_name'] == 'A3')].iloc[0]
    assert cv_row['n_folds'] == 2 and loso_row['n_folds'] == 1
    assert abs(cv_row['auroc_mean'] - 0.71) < 1e-6
    assert cv_row['auroc_ci_low'] <= cv_row['auroc_mean'] <= cv_row['auroc_ci_high']
    no_fold_type_col = synthetic_results.drop(columns=['fold_type'])
    raised = False
    try:
        aggregate_fold_metrics(no_fold_type_col)
    except ValueError:
        raised = True
    assert raised, "aggregate_fold_metrics() must refuse to run without a fold_type column"
    print(f"    cv: n_folds={cv_row['n_folds']}  auroc_mean={cv_row['auroc_mean']:.4f}  "
          f"95% CI=[{cv_row['auroc_ci_low']:.4f}, {cv_row['auroc_ci_high']:.4f}]  |  "
          f"loso: n_folds={loso_row['n_folds']}  auroc_mean={loso_row['auroc_mean']:.4f}  "
          f"(kept in separate rows, never pooled)")
    print("  TEST O PASSED")

    # ---- Test P: site-stratified repeated CV correctly falls back for too-small sites ----
    print("\n  Test P -- site-stratified CV falls back to label-only for too-small sites:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_sitestrat_') as tmp:
        index_path, df = _make_synthetic_dataset(tmp, n_subjects=24, sites=('SiteA', 'SiteB', 'SiteOneClass'))
        folds = build_repeated_cv_folds(df, n_splits=2, n_repeats=1, stratify_by_site=True, min_per_combined_bin=2)
        assert all(f.get('stratified_by_site') for f in folds)
        assert 'SiteOneClass' in folds[0]['sites_fallen_back_to_label_only'], \
            "SiteOneClass (single-class) must fall back to label-only stratification, not join-stratify"
        print(f"    sites fallen back to label-only: {folds[0]['sites_fallen_back_to_label_only']} "
              f"(SiteOneClass correctly flagged)")
    print("  TEST P PASSED")

    # ---- Test Q: Phase 7, item 4/8 -- best predictions survive a resume whose remaining
    #      epochs never beat the already-loaded best (the exact bug this module had: before
    #      the fix, best_probs/best_labels were only ever set in-memory inside the `if auroc
    #      > best['auroc']` branch, so a resumed run that never re-triggers it crashed trying
    #      to np.array(None) at the final OOF write) ----
    print("\n  Test Q -- best predictions survive a resume that never re-beats the loaded best:")
    with tempfile.TemporaryDirectory(prefix='exp_audit_resumebest_') as tmp_data, \
         tempfile.TemporaryDirectory(prefix='exp_audit_resumebest_out_') as tmp_out:
        index_path, df = _make_synthetic_dataset(tmp_data, n_subjects=16, sites=('SiteA', 'SiteB'))
        train_ids = df['subject_id'].tolist()[:12]
        val_ids = df['subject_id'].tolist()[12:]
        fit_stats = ADHDDataset(index_path, subject_ids=train_ids, augment=False).get_fit_stats()
        train_ds = ADHDDataset(index_path, subject_ids=train_ids, augment=False, fit_stats=fit_stats)
        val_ds = ADHDDataset(index_path, subject_ids=val_ids, augment=False, fit_stats=fit_stats)
        index_hash, split_hash = hash_file(index_path), hash_split(train_ids, val_ids)

        global roc_auc_score
        original_roc_auc_score = roc_auc_score
        call_count = {'n': 0}

        def _epoch1_best_then_worse(y_true, y_score):
            call_count['n'] += 1
            return 0.99 if call_count['n'] == 1 else 0.10   # epoch 1 wins; every later epoch loses

        roc_auc_score = _epoch1_best_then_worse
        try:
            _, status1 = train_one_fold('A3', 'resumefold', train_ds, val_ds, tmp_out,
                                         index_hash, split_hash, epochs=4, patience=100,
                                         _stop_after_epoch=1)
            assert status1 == 'interrupted'
            best_preds_at_epoch1 = pd.read_csv(f'{tmp_out}/resumefold_best_preds.csv')

            result2, status2 = train_one_fold('A3', 'resumefold', train_ds, val_ds, tmp_out,
                                                index_hash, split_hash, epochs=4, patience=100)
            assert status2 == 'fresh', "epochs 2-4 should still run (not skipped) to prove they genuinely didn't beat epoch 1"
            assert result2['best_epoch'] == 1, \
                f"epoch 1 (AUROC 0.99) should remain best through epochs 2-4 (AUROC 0.10 each), got best_epoch={result2['best_epoch']}"
            oof = pd.read_csv(f'{tmp_out}/resumefold_oof_predictions.csv')
            assert np.allclose(oof['y_prob'].values, best_preds_at_epoch1['y_prob'].values), \
                "final OOF predictions must match epoch 1's (the true best), not a later, worse epoch's"
        finally:
            roc_auc_score = original_roc_auc_score
        print(f"    interrupted after epoch 1 (forced AUROC=0.99) -> resumed through epochs 2-4 "
              f"(forced AUROC=0.10 each, never beating epoch 1): status={status2}, "
              f"best_epoch={result2['best_epoch']} (correct), final OOF matches epoch 1's predictions "
              f"(correct) -- no crash, no silently-wrong 'last epoch' predictions.")
    print("  TEST Q PASSED")

    print(f"\n{'=' * 70}\n  ALL SYNTHETIC SANITY CHECKS (A-Q) PASSED ✓\n{'=' * 70}")
    print("  (No real dataset was loaded, nothing under baseline/, phase2/, or "
          "phase2/consistency/ was read or written. Section 4's harness is now "
          "implemented and unit-tested, but has NOT been run on real data.)")


def main():
    """
    Default entrypoint: synthetic sanity checks ONLY (item 5: "Do not run
    expensive experiments yet").

    To run Section 4's controlled comparison on REAL data once that is
    approved, call `run_controlled_comparison()` directly with your real
    `index_file` and a real `epochs`/`patience` budget, e.g.:

        import experimental_audit as ea
        ea.run_controlled_comparison(
            index_file='/kaggle/working/processed_index.csv',
            out_dir='/kaggle/working/exp_audit',
            model_names=['baseline_mri_only', 'A3', 'A4', 'A4-new', 'A5-new', 'M0', 'M1'],
            n_splits=5, n_repeats=2, min_loso_per_class=5, epochs=60, patience=10,
        )

    This module's own main() does NOT do this automatically -- that is a
    new, expensive, real-data training run (up to 119 folds x 7 models;
    see AUDIT_REPORT.md's compute estimate) and item 5 explicitly asks
    for the utilities to be implemented and sanity-tested first, not run.
    """
    run_sanity_check()


if __name__ == '__main__':
    main()
