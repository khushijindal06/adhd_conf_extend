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
    oof_df = pd.DataFrame(dict(subject_id=val_ids, y_true=best_labels, y_prob=best_probs,
                                y_pred=(np.array(best_probs) >= 0.5).astype(int)))
    atomic_to_csv(oof_df, paths['oof'])

    result = dict(model_name=model_name, fold_tag=fold_tag, auroc=best['auroc'], best_epoch=best['epoch'],
                   n_val=len(val_ids), params=sum(p.numel() for p in model.parameters()))
    atomic_to_csv(pd.DataFrame([result]), paths['results'])
    with open(paths['config'], 'w') as f:
        json.dump(dict(model_name=model_name, fold_tag=fold_tag, config_fingerprint=current_fp), f, indent=2)

    return result, 'fresh'


def build_repeated_cv_folds(raw_df, n_splits=5, n_repeats=2, seed=SEED):
    """Item 4: stratified REPEATED k-fold over subject-level labels."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    subject_ids = raw_df['subject_id'].values
    labels = raw_df['label'].values
    folds = []
    for i, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), labels)):
        folds.append(dict(fold_id=f'cv{i:02d}', train_ids=subject_ids[train_idx].tolist(),
                           val_ids=subject_ids[val_idx].tolist()))
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
        folds.append(dict(fold_id=f'loso_{site}', train_ids=train_ids, val_ids=val_ids))
    return folds, skipped


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
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    atomic_to_csv(results_df, f'{out_dir}/controlled_comparison_all_folds.csv')
    summary_df = results_df.groupby('model_name')['auroc'].agg(['mean', 'std', 'count']).reset_index()
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

    print(f"\n{'=' * 70}\n  ALL SYNTHETIC SANITY CHECKS (A-M) PASSED ✓\n{'=' * 70}")
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
