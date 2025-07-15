import json
from load import *
import torch
import torchmetrics
from tqdm import tqdm
from torch.utils.data import DataLoader
import clip
from pytorch_lightning import seed_everything
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import functional as F

# --- Helper functions for results ---
def load_or_initialise_results(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


# --- Experiment setup ---
dataset = 'food101'
method = 'd-clip'

# Set the hyperparameters and data
hparams, tfms, dataset_loader, dataset_classes, class_subcategories, \
    gpt_descriptions, unmodify_dict, label_to_classname, n_classes = \
    set_hparams(model_size='ViT-B/32', desc_type='gpt-3', dataset=dataset, method=method)

# File for storing results
data_path = 'results/experiment_results.json'
results = load_or_initialise_results(data_path)

# Fix random seeds
dt = hparams.get('seed', 42)
seed_everything(dt)

# Prepare dataloader
bs = hparams['batch_size']
dataloader = DataLoader(dataset_loader, bs, shuffle=False, num_workers=16, pin_memory=True)

# Load CLIP model
device = torch.device(hparams.get('device', 'cpu'))
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

# Encode text: descriptions & labels
description_encodings = compute_description_encodings(model, gpt_descriptions, hparams)
label_encodings       = compute_label_encodings(model, hparams, label_to_classname)

# Alias for downstream
label_encs = label_encodings
desc_encs  = description_encodings

# --- Compute per-class mean image embeddings ---
num_classes = len(dataset_classes)
class_image_embeds = [[] for _ in range(num_classes)]
for images, labels in tqdm(dataloader, desc="Image → embeddings"):
    images = images.to(device)
    with torch.no_grad():
        embeds = model.encode_image(images)
        embeds = F.normalize(embeds, dim=-1)
    for emb, lbl in zip(embeds, labels.to(device)):
        class_image_embeds[lbl.item()].append(emb.cpu())

mean_img_embeds = []
for embeds in class_image_embeds:
    if embeds:
        m = torch.stack(embeds, dim=0).mean(dim=0)
        mean_img_embeds.append(F.normalize(m, dim=0).cpu().numpy())
    else:
        mean_img_embeds.append(np.zeros(model.visual.output_dim, dtype=float))
mean_img_embeds = np.stack(mean_img_embeds, axis=0)

# --- Build feature matrix for t-SNE ---
label_feats = label_encs.cpu().numpy()                 # (C, D)
desc_feats  = torch.cat(list(desc_encs.values()), dim=0).cpu().numpy()  # (N_desc, D)
all_feats   = np.concatenate([label_feats, desc_feats, mean_img_embeds], axis=0)

# 1) L2‐normalize each row
label_feats = normalize(label_feats, norm='l2')           # (C, D)
desc_feats  = normalize(desc_feats,  norm='l2')           # (N_desc, D)
mean_feats  = normalize(mean_img_embeds, norm='l2')       # (C, D)

# 2) concatenate
all_feats = np.vstack([label_feats, desc_feats, mean_feats])

from sklearn.preprocessing import StandardScaler

# 3) build your NxD matrix of [ label_feats; desc_feats; mean_img_embeds ]
all_feats = np.vstack([label_feats, desc_feats, mean_img_embeds])

# 4) zero-mean & unit-var each *dimension* (pulls both groups into the same ballpark)
scaler    = StandardScaler()
all_feats = scaler.fit_transform(all_feats)

# 5) now TSNE on the standardized data
all_2d = TSNE(n_components=2, random_state=42, perplexity=30, init='pca') \
            .fit_transform(all_feats)


# # 6) run t-SNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
# all_2d = tsne.fit_transform(all_feats)

C = label_feats.shape[0]
N = desc_feats.shape[0]
M = mean_feats.shape[0]

clip_2d = all_2d[:C]
desc_2d = all_2d[C:C+N]
mean_2d = all_2d[C+N:C+N+M]

# 7) plot
plt.figure(figsize=(12,10))
plt.scatter(clip_2d[:,0],  clip_2d[:,1],  s=80, marker='o', label='CLIP labels')
plt.scatter(desc_2d[:,0],  desc_2d[:,1],  s=40, marker='x', label='GPT descriptions')
plt.scatter(mean_2d[:,0],  mean_2d[:,1],  s=150, marker='X', edgecolor='k', color='r', label='Mean images')
plt.legend()
plt.title("t-SNE of joint CLIP space (all modalities)")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

filename = f"2_{method}_embedding_plot_{dataset}.png"
plt.savefig(f"figs/tsne_vis/{filename}", dpi=200, bbox_inches="tight")
print("Plot saved as ", filename, " in figs/tsne_vis/")
