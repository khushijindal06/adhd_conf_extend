
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ADHDDataset(Dataset):
    def __init__(self, index_csv, subject_ids=None, fit_stats=None, augment=False):
        df = pd.read_csv(index_csv)
        df = df[df['site'] != 'Pittsburgh'].reset_index(drop=True)

        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)].reset_index(drop=True)

        # ── replace -999 with NaN everywhere ──────────────────────────────
        df.replace(-999, np.nan, inplace=True)

        # ── cast to numeric ────────────────────────────────────────────────
        for col in ['age', 'sex', 'med_status', 'site_id', 'adhd_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ── imputation statistics ────────────────────────────────────────────
        # Fit on whatever subset is passed when fit_stats is None (the caller
        # passes only the TRAINING subjects for that call), then reuse the
        # exact same median/mode for every other split via fit_stats. Computing
        # these independently per split leaks validation-set information into
        # the imputed values.
        if fit_stats is None:
            self.age_median        = df['age'].median()
            self.sex_mode          = df['sex'].mode()[0]
            self.med_status_mode   = df['med_status'].mode()[0]
            self.adhd_index_median = df['adhd_index'].median()
        else:
            self.age_median        = fit_stats['age_median']
            self.sex_mode          = fit_stats['sex_mode']
            self.med_status_mode   = fit_stats['med_status_mode']
            self.adhd_index_median = fit_stats['adhd_index_median']

        # ── impute missing values using the (train-derived) statistics ──────
        df['age']        = df['age'].fillna(self.age_median)
        df['sex']        = df['sex'].fillna(self.sex_mode)
        df['site_id']    = df['site_id'].fillna(0)
        df['adhd_index'] = df['adhd_index'].fillna(self.adhd_index_median)
        df['med_status'] = df['med_status'].fillna(self.med_status_mode)

        # ── z-score normalise continuous features ──────────────────────────
        # only age and adhd_index are continuous
        # sex, site_id, med_status are categorical — just scale to 0-1
        if fit_stats is None:
            self.age_mean      = df['age'].mean()
            self.age_std       = df['age'].std()        + 1e-8
            self.adhd_idx_mean = df['adhd_index'].mean()
            self.adhd_idx_std  = df['adhd_index'].std() + 1e-8
            # site_id: 0-5 → divide by 5
            # med_status: 1,2 → divide by 2
            # sex: 0,1 → already 0-1
        else:
            self.age_mean      = fit_stats['age_mean']
            self.age_std       = fit_stats['age_std']
            self.adhd_idx_mean = fit_stats['adhd_idx_mean']
            self.adhd_idx_std  = fit_stats['adhd_idx_std']

        df['age_norm']      = (df['age']        - self.age_mean)      / self.age_std
        df['adhd_idx_norm'] = (df['adhd_index'] - self.adhd_idx_mean) / self.adhd_idx_std
        df['site_norm']     = df['site_id']    / 5.0
        df['med_norm']      = df['med_status'] / 2.0
        df['sex_norm']      = df['sex']        / 1.0

        self.df      = df
        self.labels  = df['label'].values.astype(np.int64)
        self.augment = augment

    def get_fit_stats(self):
        return {
            # imputation statistics
            'age_median'        : self.age_median,
            'sex_mode'          : self.sex_mode,
            'med_status_mode'   : self.med_status_mode,
            'adhd_index_median' : self.adhd_index_median,
            # normalization statistics
            'age_mean'      : self.age_mean,
            'age_std'       : self.age_std,
            'adhd_idx_mean' : self.adhd_idx_mean,
            'adhd_idx_std'  : self.adhd_idx_std,
        }

    def class_weights(self):
        n      = len(self.labels)
        n_ctrl = (self.labels == 0).sum()
        n_adhd = (self.labels == 1).sum()
        return torch.tensor([n/(2*n_ctrl), n/(2*n_adhd)], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── MRI volume ─────────────────────────────────────────────────────
        vol = np.load(row['npy_path']).astype(np.float32)

        # ── Augmentation (training only) ───────────────────────────────────
        if self.augment:
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=0).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=1).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=2).copy()

        vol = torch.from_numpy(vol).unsqueeze(0)           # (1,64,64,64)

        # ── All 5 phenotypic features ──────────────────────────────────────
        # [age_norm, sex_norm, med_norm, site_norm, adhd_idx_norm]
        pheno = torch.tensor([
            float(row['age_norm']),
            float(row['sex_norm']),
            float(row['med_norm']),
            float(row['site_norm']),
            float(row['adhd_idx_norm']),
        ], dtype=torch.float32)                            # (5,)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return vol, pheno, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    INDEX_CSV = '/kaggle/working/processed_index.csv'
    ds = ADHDDataset(INDEX_CSV, augment=True)
    print(f"Total samples  : {len(ds)}")
    print(f"Class weights  : {ds.class_weights()}")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    vol_b, pheno_b, label_b = next(iter(loader))
    print(f"Volume shape   : {vol_b.shape}")       # (4,1,64,64,64)
    print(f"Pheno shape    : {pheno_b.shape}")     # (4,5)
    print(f"Labels         : {label_b}")
    print(f"Pheno sample   : {pheno_b[0]}")        # check values look sane
    print(f"\ndataset.py sanity check PASSED ✓")