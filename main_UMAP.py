import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from load import set_hparams, compute_description_encodings, compute_label_encodings

# ————————————————————————————————————————————————
# 1) Experiment settings (make sure these match your run)
dataset = 'eurosat'
method  = 'd-clip'

hparams, tfms, dataset_loader, dataset_classes, class_subcategories, \
    gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(
        model_size='ViT-B/32',
        desc_type='gpt-3',
        dataset=dataset,
        method=method
    )

device = torch.device(hparams['device'])
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

# ————————————————————————————————————————————————
# 2) Compute text embeddings (all on GPU, then move to CPU/NumPy)
desc_encs  = compute_description_encodings(model, gpt_descriptions, hparams)
label_encs = compute_label_encodings(model, hparams, label_to_classname)

# flatten out the description vectors into one big matrix
desc_points = torch.cat([v.to('cpu') for v in desc_encs.values()], dim=0).numpy()

# make a parallel list of class-names for those points
desc_names = []
for cls_key, v in desc_encs.items():
    try:
        # if the key is convertible to int, look up its human name
        cls_idx  = int(cls_key)
        cls_name = label_to_classname[cls_idx]
    except (ValueError, KeyError):
        # otherwise assume the key itself is already the class name
        cls_name = str(cls_key)
    # repeat that name once per descriptor
    desc_names += [cls_name] * v.shape[0]


# whole X matrix: first all the CLIP‐standard label embeddings, then the descriptions
X = np.vstack([
    label_encs.to('cpu').numpy(),
    desc_points
])

# matching labels for coloring/legend
labels = (
    ["CLIP-Standard"] * label_encs.shape[0]
  + ["Description-based"] * desc_points.shape[0]
)

# a single list of class names for annotation
class_names = (
    [label_to_classname[i] for i in range(label_encs.shape[0])]
  + desc_names
)

# ————————————————————————————————————————————————
# 3) UMAP reduction
reducer = UMAP(n_components=2, random_state=42)
X_2d   = reducer.fit_transform(X)  # shape (n_points, 2)

# ————————————————————————————————————————————————
# 4) Plot
plt.figure(figsize=(12, 9))

# separate indices
n_clip = label_encs.shape[0]
clip_idx = range(n_clip)
desc_idx = range(n_clip, n_clip + desc_points.shape[0])

# CLIP‐standard
plt.scatter(
    X_2d[clip_idx, 0],
    X_2d[clip_idx, 1],
    marker='o',
    label='CLIP-Standard',
    alpha=0.7,
    s=80
)
for i in clip_idx:
    plt.text(
        X_2d[i, 0],
        X_2d[i, 1],
        class_names[i],
        fontsize=6,
        color='black'
    )

# Description‐based
plt.scatter(
    X_2d[desc_idx, 0],
    X_2d[desc_idx, 1],
    marker='x',
    label='Description-based',
    alpha=0.6,
    s=40,
    color='tab:orange'
)
for i in desc_idx:
    # only label a subset if there are too many; here we skip the first 5 descriptors
    plt.text(
        X_2d[i, 0],
        X_2d[i, 1],
        class_names[i],
        fontsize=5,
        color='tab:orange'
    )

plt.title("UMAP of CLIP vs Description-based Text Embeddings")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.grid(True)
plt.legend(loc='best')

# ensure output folder exists
out_dir = f"figs/umap_vis"
os.makedirs(out_dir, exist_ok=True)
filename = f"{method}_embedding_umap_{dataset}.png"
plt.tight_layout()
plt.savefig(os.path.join(out_dir, filename), dpi=200)
plt.show()

print(f"Saved UMAP plot to {os.path.join(out_dir, filename)}")