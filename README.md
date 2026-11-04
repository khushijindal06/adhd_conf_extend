# adhd_conf_extend

ADHD-200 MRI classification — conference-extension work: run-ready codebase,
real-data audit, and post-hoc analysis scripts.

- [`AUDIT_REPORT.md`](AUDIT_REPORT.md) — audit of the initial real-data
  results (896 subjects) before starting Phase 3: baseline comparability,
  TRM-variant verification, consistency-learning diagnostics, and a
  proposed controlled MRI-only comparison design. Several items are marked
  **PENDING** real-run artifacts — see "Outstanding items" at the end.
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
- [`analysis/`](analysis) — read-only / inference-only scripts used by the
  audit:
  - `check_phenotype_leakage.py` — univariate AUROC + distribution checks
    for the baseline's phenotype features, to test whether `med_status`/
    `adhd_index` trivially separate the diagnostic label.
  - `extract_representations_m0_m1.py` — extracts per-subject 128-d
    representations from the trained M0/M1 checkpoints and computes
    collapse/discriminability diagnostics, without any training or
    architecture change.

Architecture is frozen pending the audit's review. Phase 3 has not started.
