import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import clip
from load import set_hparams, compute_description_encodings, compute_label_encodings

# ─── Config ──────────────────────────────────────────────────────────────────────
dataset           = 'eurosat'
METHODS           = ['d-clip', 'waffleclip', 'defntaxs']  # ← add as many as you like
MODEL_SIZE        = 'ViT-B/32'
DESC_TYPE         = 'gpt-3'
RESULTS_FILE      = 'results/experiment_results.json'
OUTPUT_DIR        = 'figs/tsne_vis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TSNE / plotting parameters
PCA_DIMS               = 50
TSNE_PERP              = 10
TSNE_SEED              = 0
MARKER_SIZE            = 80
BASELINE_MARKER        = 'o'
METHOD_MARKERS         = ['x','*','s','^','D','v','<','>','P']  # will cycle if you have >9 methods
ANNOTATE_BASELINE_EVERY= 5
FONT_BASELINE          = 6
FONT_METHOD            = 5

# ─── Setup model, data ───────────────────────────────────────────────────────────
# 1) get baseline hyperparams + class names
hparams_base, _, dataset_loader, _, _, _, _, label_to_classname, _ = set_hparams(
    model_size=MODEL_SIZE,
    desc_type=DESC_TYPE,
    dataset=dataset,
    method='clip'   # baseline
)
seed_everything(hparams_base['seed'])

device = torch.device(hparams_base['device'])
model, _ = clip.load(MODEL_SIZE, device=device, jit=False)
model.eval()
model.requires_grad_(False)

dataloader = DataLoader(dataset_loader,
                        batch_size=hparams_base['batch_size'],
                        shuffle=False,
                        num_workers=16,
                        pin_memory=True)

# ─── Build embeddings for each group ───────────────────────────────────────────────
all_embeddings   = []
all_group_names  = []
all_class_names  = []
marker_list      = []

# 1) Baseline CLIP labels
label_pts = compute_label_encodings(model, hparams_base, label_to_classname).cpu().numpy()
all_embeddings.append(label_pts)
all_group_names.append('CLIP-Standard')
all_class_names.append(list(label_to_classname))
marker_list.append(BASELINE_MARKER)

# 2) Each method’s description embeddings
for idx, method in enumerate(METHODS):
    # get method-specific descriptors
    hparams_m, _, _, _, _, gpt_descs_m, _, _, _ = set_hparams(
        model_size=MODEL_SIZE,
        desc_type=DESC_TYPE,
        dataset=dataset,
        method=method
    )
    desc_encs = compute_description_encodings(model, gpt_descs_m, hparams_m)
    emb = np.vstack([v.cpu().numpy() for v in desc_encs.values()])
    names = [cls for cls, v in desc_encs.items() for _ in range(v.shape[0])]

    all_embeddings.append(emb)
    all_group_names.append(method)
    all_class_names.append(names)
    marker_list.append(METHOD_MARKERS[idx % len(METHOD_MARKERS)])

# flatten into one array for TSNE
X           = np.vstack(all_embeddings)
flat_names  = sum(all_class_names, [])

# ─── PCA → tSNE ──────────────────────────────────────────────────────────────────
pca  = PCA(n_components=PCA_DIMS, random_state=TSNE_SEED)
X_pca = pca.fit_transform(X)
tsne = TSNE(n_components=2,
            perplexity=TSNE_PERP,
            random_state=TSNE_SEED,
            init='pca')
X_2d = tsne.fit_transform(X_pca)

# compute index ranges for each group
ranges = []
start = 0
for emb in all_embeddings:
    n = emb.shape[0]
    ranges.append(range(start, start+n))
    start += n

# ─── Plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
cmap = cm.get_cmap('tab10')

for i, (grp, rng) in enumerate(zip(all_group_names, ranges)):
    xs = X_2d[list(rng), 0]
    ys = X_2d[list(rng), 1]
    color = cmap(i % 10)

    ax.scatter(xs, ys,
               marker=marker_list[i],
               s=MARKER_SIZE,
               alpha=0.7,
               label=grp,
               color=color)

    # annotations
    if i == 0:
        # baseline: annotate every Nth point
        for j in rng:
            if j % ANNOTATE_BASELINE_EVERY == 0:
                ax.text(X_2d[j,0], X_2d[j,1],
                        flat_names[j],
                        fontsize=FONT_BASELINE,
                        color='black')
    else:
        # methods: annotate all
        for j in rng:
            ax.text(X_2d[j,0], X_2d[j,1],
                    flat_names[j],
                    fontsize=FONT_METHOD,
                    color=color)

ax.set_title(f"tSNE of CLIP + {', '.join(METHODS)} ({dataset})")
ax.set_xlabel("tSNE Component 1")
ax.set_ylabel("tSNE Component 2")
ax.grid(True)
ax.legend(loc='best')

out_path = os.path.join(OUTPUT_DIR, f"TEST_tsne_multi_{dataset}.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.show()
print("Saved to", out_path)
