import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from model import ADHDNet

OUT_DIR = '.'

plt.rcParams.update({'font.size': 10, 'savefig.dpi': 300})


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Architecture diagram
# (DS-Conv3D -> BN -> ReLU -> Pool) x3 -> 3D ECA -> GAP, fused with the
# phenotype MLP branch via adaptive gating.
# ══════════════════════════════════════════════════════════════════════════
def box(ax, xy, w, h, text, fc='#F2F2F2', ec='#555555', fontsize=9, fontweight='normal'):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                            boxstyle='round,pad=0.02,rounding_size=0.03',
                            facecolor=fc, edgecolor=ec, linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight)


def arrow(ax, xy1, xy2, color='#333333'):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle='-|>', mutation_scale=12,
                                  color=color, linewidth=1.3))


fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

# ── MRI branch (left column) ─────────────────────────────────────────────
mri_x = 3.0
nodes_mri = [
    (9.4, 'MRI Volume\n(1, 64³)', '#FFFFFF'),
    (8.2, 'DS-Conv3D Block 1\nBN → ReLU → Pool\n(24, 32³)', '#EAF2FB'),
    (7.0, 'DS-Conv3D Block 2\nBN → ReLU → Pool\n(48, 16³)', '#EAF2FB'),
    (5.8, 'DS-Conv3D Block 3\nBN → ReLU → Pool\n(96, 8³)', '#EAF2FB'),
    (4.6, '3D ECA\n(96, 8³)', '#FCE8D8'),
    (3.4, 'Global Avg Pool\n→ 96', '#F2F2F2'),
    (2.2, 'Linear → 128\nz_MRI', '#DCEBF7'),
]
for y, text, fc in nodes_mri:
    box(ax, (mri_x, y), 2.6, 0.85, text, fc=fc)
for (y1, *_), (y2, *_) in zip(nodes_mri[:-1], nodes_mri[1:]):
    arrow(ax, (mri_x, y1 - 0.43), (mri_x, y2 + 0.43))

# ── Phenotype branch (right column) ──────────────────────────────────────
phen_x = 7.6
nodes_phen = [
    (9.4, 'Phenotype Vector\n(5,)\nage, sex, med,\nsite, ADHD-index', '#FFFFFF'),
    (7.6, 'MLP\nFC → BN → ReLU\n→ Dropout → FC', '#EAF2FB'),
    (6.0, 'z_P\n(32,)', '#DCEBF7'),
]
for y, text, fc in nodes_phen:
    h = 1.3 if 'Phenotype' in text else (1.05 if 'MLP' in text else 0.85)
    box(ax, (phen_x, y), 2.6, h, text, fc=fc)
arrow(ax, (phen_x, 9.4 - 0.65), (phen_x, 7.6 + 0.52))
arrow(ax, (phen_x, 7.6 - 0.52), (phen_x, 6.0 + 0.43))

# ── Fusion + head (bottom, spanning both columns) ────────────────────────
merge_x = (mri_x + phen_x) / 2
box(ax, (merge_x, 1.0), 4.2, 0.95,
    'Adaptive Gated Fusion\ng=σ(FC→ReLU→FC)([z_MRI || z_P]);  '
    'z_P→128;  fused = g·z_MRI + (1-g)·z_P', fc='#FCE8D8', fontsize=8)
arrow(ax, (mri_x, 2.2 - 0.43), (merge_x - 0.3, 1.0 + 0.48))
arrow(ax, (phen_x, 6.0 - 0.43), (merge_x + 0.3, 1.0 + 0.48))

box(ax, (merge_x, -0.3), 3.2, 0.7, 'Classifier Head\nFC(128→64) → ReLU → Dropout → FC(64→2)',
    fc='#F2F2F2', fontsize=8)
arrow(ax, (merge_x, 1.0 - 0.48), (merge_x, -0.3 + 0.35))

ax.set_ylim(-0.8, 10)
ax.set_title('ADHDNet Architecture', fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_architecture.png')
print(f"Fig. 1 saved -> {OUT_DIR}/fig1_architecture.png")


# ══════════════════════════════════════════════════════════════════════════
# Table 1 — Layer-by-layer architecture summary (traced from the real model)
# ══════════════════════════════════════════════════════════════════════════
model = ADHDNet(n_pheno=5, dropout=0.3)
model.eval()

rows = []
hooks = []

def make_hook(name):
    def hook(module, inp, out):
        shape = tuple(out.shape[1:]) if torch.is_tensor(out) else None
        n_params = sum(p.numel() for p in module.parameters())
        rows.append((name, type(module).__name__, shape, n_params))
    return hook

named = [
    ('cnn.0 (DS-Conv Block 1)', model.cnn[0]),
    ('cnn.1 (DS-Conv Block 2)', model.cnn[1]),
    ('cnn.2 (DS-Conv Block 3)', model.cnn[2]),
    ('cnn.3 (3D ECA)',          model.cnn[3]),
    ('cnn.4 (GAP)',             model.cnn[4]),
    ('img_proj (Linear)',       model.img_proj),
    ('pheno_mlp',               model.pheno_mlp),
    ('gate',                    model.gate),
    ('pheno_proj (Linear)',     model.pheno_proj),
    ('head',                    model.head),
]
for name, mod in named:
    hooks.append(mod.register_forward_hook(make_hook(name)))

with torch.no_grad():
    model(torch.randn(1, 1, 64, 64, 64), torch.randn(1, 5))

for h in hooks:
    h.remove()

print(f"\n{'Layer':<28}{'Type':<16}{'Output shape':<18}{'Params':>10}")
print('-' * 72)
total_params = sum(p.numel() for p in model.parameters())
for name, typ, shape, n in rows:
    print(f"{name:<28}{typ:<16}{str(shape):<18}{n:>10,}")
print('-' * 72)
print(f"{'TOTAL':<62}{total_params:>10,}")

with open(f'{OUT_DIR}/table1_architecture.csv', 'w') as f:
    f.write("layer,type,output_shape,params\n")
    for name, typ, shape, n in rows:
        f.write(f'"{name}",{typ},"{shape}",{n}\n')
    f.write(f'"TOTAL",,,{total_params}\n')

print(f"\nTable 1 saved -> {OUT_DIR}/table1_architecture.csv")
