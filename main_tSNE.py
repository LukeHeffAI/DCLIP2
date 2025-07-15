import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchmetrics
from tqdm import tqdm
import torch
import numpy as np
from load import *

def load_or_initialise_results(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    
def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

dataset = 'eurosat' # Options: ['eurosat', 'food101', 'cifar10', 'cifar100', 'imagenet-1k', 'imagenet-21k']
method = 'd-clip' # Options: ['clip', 'e-clip', 'd-clip', 'waffleclip', 'waffleclip+concepts', 'defntaxs']


# Set the hyperparameters
hparams, tfms, dataset_loader, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(  model_size='ViT-B/32',
                                                                                                                                                    desc_type='gpt-3',
                                                                                                                                                    dataset=dataset,
                                                                                                                                                    method=method)

results_file_path = 'results/experiment_results.json'
results = load_or_initialise_results(results_file_path)

# Initialize the environment
seed_everything(hparams['seed'])

# Prepare the data loader
bs = hparams['batch_size']
dataloader = DataLoader(dataset_loader, bs, shuffle=False, num_workers=16, pin_memory=True)  # Shuffle should be False for class-wise evaluation

# Load the model and preprocessing
print("Loading model...")
device = torch.device(hparams['device'])
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

# Encode descriptions and labels
print("Encoding descriptions...")
description_encodings = compute_description_encodings(model, gpt_descriptions, hparams)
label_encodings = compute_label_encodings(model, hparams, label_to_classname)

# Number of classes
num_classes = len(dataset_classes)

# Set experiment params (copy these from your experiment)
hparams, tfms, dataset_loader, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(
    model_size='ViT-B/32',
    desc_type='gpt-3',
    dataset=dataset,
    method=method
)
import clip
device = hparams['device']
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

# ----- Compute text embeddings -----
desc_encs = compute_description_encodings(model, gpt_descriptions, hparams)
label_encs = compute_label_encodings(model, hparams, label_to_classname)

# For all description embeddings:
desc_points = torch.cat([v for v in desc_encs.values()], dim=0)
desc_labels = []
for class_idx, (k, v) in enumerate(desc_encs.items()):
    desc_labels += [k] * v.shape[0]

X = torch.cat([label_encs, desc_points], dim=0).cpu().numpy()
labels = (["CLIP-Standard"] * len(label_to_classname)) + (["Description-based"] * len(desc_points))
class_names = list(label_to_classname) + desc_labels

# --- tSNE ---
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
X_2d = tsne.fit_transform(X)

scale = len(label_to_classname) / 20
x_size = 10 + scale
y_size = 8 + scale
plt.figure(figsize=(x_size, y_size))

# Plot Description-based embeddings first (orange markers, orange text)
desc_indices = range(len(label_to_classname), len(class_names))
plt.scatter(X_2d[desc_indices, 1], X_2d[desc_indices, 0], marker='x', color='tab:orange', 
           label='Description-based', alpha=0.6, s=50)
for i in desc_indices:
    if i > 5:
        plt.text(X_2d[i, 1], X_2d[i, 0], class_names[i], fontsize=5, color='tab:orange')

# Plot CLIP-Standard embeddings on top (blue markers, black text)
clip_indices = range(len(label_to_classname))
plt.scatter(X_2d[clip_indices, 1], X_2d[clip_indices, 0], marker='o', color='tab:blue', 
           label='CLIP-Standard', alpha=0.7, s=100)
for i in clip_indices:
    plt.text(X_2d[i, 1], X_2d[i, 0], class_names[i], fontsize=7, color='black')

plt.title("tSNE of CLIP vs Description-based Text Embeddings")
plt.legend()
plt.tight_layout()
plt.xlabel("tSNE Component 1")
plt.ylabel("tSNE Component 2")

# Save the plot
plt.grid()
plt.show()
filename = f"{method}_embedding_plot_{dataset}.png"
plt.savefig(f"figs/tsne_vis/test_{filename}", dpi=200, bbox_inches="tight")
print("Plot saved as ", filename, " in figs/tsne_vis/")