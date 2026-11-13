# adhd_conf_extend

ADHD-200 MRI classification — conference-extension work: run-ready codebase,
real-data audit, and post-hoc analysis scripts.

- [`AUDIT_REPORT.md`](AUDIT_REPORT.md) — audit of the initial real-data
  results (896 subjects) before starting Phase 3: baseline comparability,
  TRM-variant verification, consistency-learning diagnostics, and a
  controlled MRI-only comparison design (now implemented as code — see
  `code/experimental_audit.py` below). Several items are marked
  **PENDING** real-run artifacts — see "Outstanding items" and "Which
  analyses need existing checkpoints vs. new training" at the end.
- [`ARCHITECTURE_AUDIT_REPORT.md`](ARCHITECTURE_AUDIT_REPORT.md) — the
  8-phase pipeline/architecture/reliability audit: preprocessing
  consistency, resolution, the fair MRI-only baseline, encoder/token/TRM/
  attention diagnostics, the M0/M1 consistency-loss investigation,
  augmentation, repeated-CV/LOSO reporting, and implementation
  reliability (including two real bugs found and fixed — see
  `code/experimental_audit.py` below). Deliverable 5 (final
  recommendation) is explicitly deferred pending real-data results.
- [`code/`](code) — Phase 1 (baseline `ADHDNet` + ablations) and Phase 2,
  Steps 1–2 (pairwise Token Relationship Module, representation-consistency
  learning). See `code/README.md` for setup and run instructions.
  - [`code/full_codebase.py`](code/full_codebase.py) — all 17 scripts
    above, flattened verbatim into one file under `# FILE: code/...`
    section headers, for single-file reading/supplementary submission.
    **Reference only — do not run this file directly**: several of the
    originals execute a full pipeline at module scope and reuse
    config-constant/class names across files for different purposes, so
    concatenated together they would collide. Run each script from
    `code/` on its own, as `code/README.md` describes.
  - [`code/full_codebase_notebook.py`](code/full_codebase_notebook.py) —
    the *other* codebase: all 20 `_adhd200_combined_*.py` files the real
    Kaggle notebook (`adhd200_notebook_updated.ipynb`) writes via
    `%%writefile` — the actual modules that produced the real results in
    `AUDIT_REPORT.md` — flattened the same way. **Reference only**: each
    section starts with its own `%%writefile` magic line (paste each into
    its own notebook cell), so this file isn't valid plain Python on its
    own either.
  - [`code/experimental_audit.py`](code/experimental_audit.py) — the
    separate experimental audit module: baseline-input introspection,
    split-identity verification, TRM-variant/checkpoint cross-checking,
    a reusable log-consistency checker, calibration/ROC-PR/embedding
    diagnostics, and a repeated-CV + leave-one-site-out controlled
    comparison harness covering all 7 models (`baseline_mri_only`, A3,
    A4, A4-new, A5-new, M0, M1). Imports the existing model classes and
    training utilities unmodified; defines no new model. `python
    experimental_audit.py` runs its synthetic-only sanity suite (17
    tests, A-Q — see
    [`experimental_audit_sanity_output.txt`](experimental_audit_sanity_output.txt)).
    Has not been run against real data.
  - [`code/architecture_audit.py`](code/architecture_audit.py) — encoder/
    token/TRM/attention diagnostics (Phase 3), a separate scale-preserving
    attention ablation class, a consistency-loss lambda sweep + warmup
    schedule (Phase 4), and an MRI-safe augmentation ablation (Phase 5).
    Imports the existing model classes unmodified; defines exactly one new
    class (`ScalePreservingAnatomicalAttention`), alongside, not instead
    of, the existing `AnatomicalAttention`. `python architecture_audit.py`
    runs its synthetic-only sanity suite (6 tests, A-F — see
    [`architecture_audit_sanity_output.txt`](architecture_audit_sanity_output.txt)).
  - [`code/preprocessing_audit.py`](code/preprocessing_audit.py) —
    NIfTI affine/voxel-spacing consistency checker, per-subject
    preprocessing-variant classification, site/class/QC/missing-scan
    reporting, representative-slice visualization, and a 64³-vs-96³
    resolution-comparison harness with gradient accumulation (Phase 1).
    Read-only against existing outputs except the (not-yet-run)
    resolution harness. `python preprocessing_audit.py` runs its
    synthetic-only sanity suite (5 tests, A-E — nibabel-written fake
    NIfTI files, no real MRI data).
- [`analysis/`](analysis) — read-only / inference-only scripts used by the
  audit:
  - `check_phenotype_leakage.py` — univariate AUROC + distribution checks
    for the baseline's phenotype features, to test whether `med_status`/
    `adhd_index` trivially separate the diagnostic label.
  - `extract_representations_m0_m1.py` — extracts per-subject 128-d
    representations from the trained M0/M1 checkpoints and computes
    collapse/discriminability diagnostics, without any training or
    architecture change.
  - [`notebook_cells_full.py`](analysis/notebook_cells_full.py) — the
    notebook-specific versions of the two scripts above (adapted import
    style, matching the real Kaggle notebook's `_adhd200_combined_*`
    naming convention), flattened into one file under per-cell section
    headers. Each section starts with a `%%writefile` Jupyter magic line
    and is meant to be pasted into its own notebook cell, not run as a
    plain script — see the file's own header.

Architecture is frozen pending the audit's review. Phase 3 has not started.
