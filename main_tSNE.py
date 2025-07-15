import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import clip
from load import set_hparams, compute_description_encodings, compute_label_encodings

def load_or_initialise_results(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

# ─── Config ──────────────────────────────────────────────────────────────────────
dataset = 'eurosat'
method = 'd-clip'       # Options: ['clip', 'e-clip', 'd-clip', 'waffleclip', 'waffleclip+concepts', 'defntaxs']
results_file = 'results/experiment_results.json'
output_dir   = 'figs/tsne_vis'
os.makedirs(output_dir, exist_ok=True)

# TSNE / plotting params
PCA_DIMS     = 50
TSNE_PERP    = 10
TSNE_SEED    = 0
MARKER_SIZE  = 80
DESC_MARKER  = 'x'
CLIP_MARKER  = 'o'
DESC_COLOR   = 'tab:orange'
CLIP_COLOR   = 'tab:blue'
FONT_SIZE_DESC = 5
FONT_SIZE_CLIP = 6

# ─── Setup ───────────────────────────────────────────────────────────────────────
# Hyperparameters, data, model
hparams, tfms, dataset_loader, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(
    model_size='ViT-B/32',
    desc_type='gpt-3',
    dataset=dataset,
    method=method
)

seed_everything(hparams['seed'])

dataloader = DataLoader(dataset_loader,
                        batch_size=hparams['batch_size'],
                        shuffle=False,
                        num_workers=16,
                        pin_memory=True)

device = torch.device(hparams['device'])
model, preprocess = clip.load(hparams['model_size'],
                              device=device,
                              jit=False)
model.eval()
model.requires_grad_(False)


# ─── Encode ──────────────────────────────────────────────────────────────────────
print("Encoding descriptions and labels…")
desc_encs  = compute_description_encodings(model, gpt_descriptions, hparams)
label_encs = compute_label_encodings(model, hparams, label_to_classname)

# Build data matrix + labels
desc_points = torch.cat(list(desc_encs.values()), dim=0).cpu().numpy()
label_points = label_encs.cpu().numpy()

X = np.vstack([label_points, desc_points])
labels_type = (["CLIP-Standard"] * len(label_points) +
               ["Description-based"] * len(desc_points))
class_names = list(label_to_classname) + \
              [cls for cls, pts in desc_encs.items() for _ in range(pts.shape[0])]

# ─── Dimensionality reduction ───────────────────────────────────────────────────
print("Running PCA → tSNE…")
pca = PCA(n_components=PCA_DIMS, random_state=TSNE_SEED)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2,
            perplexity=TSNE_PERP,
            random_state=TSNE_SEED,
            init='pca')
X_2d = tsne.fit_transform(X_pca)


# ─── Plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
num_labels = len(label_points)

# Description-based
desc_idx = range(num_labels, len(X_2d))
ax.scatter(X_2d[desc_idx,0], X_2d[desc_idx,1],
           marker=DESC_MARKER,
           s=MARKER_SIZE * 0.5,
           alpha=0.6,
           label='Description-based',
           color=DESC_COLOR)
for i in desc_idx:
    ax.text(X_2d[i,0], X_2d[i,1], class_names[i],
            fontsize=FONT_SIZE_DESC, color=DESC_COLOR)

# CLIP-Standard
clip_idx = range(0, num_labels)
ax.scatter(X_2d[clip_idx,0], X_2d[clip_idx,1],
           marker=CLIP_MARKER,
           s=MARKER_SIZE,
           alpha=0.7,
           label='CLIP-Standard',
           color=CLIP_COLOR)
for i in clip_idx:
    if i % 5 == 0:  # annotate a subset to reduce clutter
        ax.text(X_2d[i,0], X_2d[i,1], class_names[i],
                fontsize=FONT_SIZE_CLIP, color='black')

ax.set_title(f"tSNE of CLIP vs Description-based Embeddings ({dataset}, {method})")
ax.set_xlabel("tSNE Component 1")
ax.set_ylabel("tSNE Component 2")
ax.grid(True)
ax.legend(loc='best')

# Save then show
out_path = os.path.join(output_dir, f"tsne_{method}_{dataset}.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print("Saved plot to", out_path)
plt.show()