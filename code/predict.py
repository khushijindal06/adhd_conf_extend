
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