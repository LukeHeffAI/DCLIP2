import json
from load import *
import torchmetrics
from tqdm import tqdm
import torch

def load_or_initialise_results(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    
def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

dataset = 'food101'
method = 'd-clip'

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

# Evaluation metrics for overall and per-class accuracies
print("Evaluating...")
overall_lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
overall_lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

overall_clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
overall_clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

# Initialize dictionaries to track class-wise accuracy
class_wise_lang_accuracy = {i: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) for i in range(num_classes)}
class_wise_clip_accuracy = {i: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) for i in range(num_classes)}

# for batch_number, (images, labels) in enumerate(tqdm(dataloader)):    
#     images = images.to(device)
#     labels = labels.to(device)
    
#     # Encode images
#     image_encodings = model.encode_image(images)
#     image_encodings = F.normalize(image_encodings)
    
#     # Compute similarities and make predictions
#     image_labels_similarity = image_encodings @ label_encodings.T
#     clip_predictions = image_labels_similarity.argmax(dim=1)
    
#     # Update overall and class-wise accuracies for CLIP
#     overall_clip_accuracy_metric(image_labels_similarity, labels)
#     overall_clip_accuracy_metric_top5(image_labels_similarity, labels)
#     for i in range(num_classes):
#         class_mask = labels == i
#         if class_mask.any():
#             class_wise_clip_accuracy[i](clip_predictions[class_mask], labels[class_mask])

#     # Compute description-based predictions
#     image_description_similarity = [None]*n_classes
#     image_description_similarity_cumulative = [None]*n_classes

#     for i, (k, v) in enumerate(description_encodings.items()):
#         dot_product_matrix = image_encodings @ v.T
        
#         # # Penalise the dot product matrix based on the frequency of the descriptor
#         # if frequency_type == 'freq_exact':
#         #     for descriptor in gpt_descriptions[k]: # Iterate over the descriptors for this class
#         #         dot_product_matrix -= freq_exact[unmodify_dict[k][descriptor]]
#         # elif frequency_type == 'freq_approx':
#         #     for descriptor in gpt_descriptions[k]:
#         #         dot_product_matrix -= freq_exact[unmodify_dict[k][descriptor]]

#         # # Penalise the dot product matrix based on the normalised cosine similarity of the descriptor to all other descriptors
#         # if similarity_penalty_config == 'similarity_penalty':
#         #     for descriptor in gpt_descriptions[k]: # Iterate over the descriptors for this class
#         #         dot_product_matrix /= descriptor_self_similarity[unmodify_dict[k][descriptor]]
        
#         image_description_similarity[i] = dot_product_matrix
#         image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

#     cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
#     descr_predictions = cumulative_tensor.argmax(dim=1)
    
#     # Update overall and class-wise accuracies for descriptions
#     overall_lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
#     overall_lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
#     for i in range(num_classes):
#         class_mask = labels == i
#         if class_mask.any():
#             class_wise_lang_accuracy[i](descr_predictions[class_mask], labels[class_mask])

# # Print class-wise accuracies
# # print("\nClass-wise Description-based Accuracy:")
# # for i, acc in class_wise_lang_accuracy.items():
# #     class_name = dataset_classes[i]
# #     accuracy = 100 * acc.compute().item()
# #     print(f"Desc. Acc.: {accuracy:.3f}% - {class_name}")

# # print("\nClass-wise CLIP-Standard Accuracy:")
# # for i, acc in class_wise_clip_accuracy.items():
# #     class_name = dataset_classes[i]
# #     accuracy = 100 * acc.compute().item()
# #     print(f"CLIP Acc.: {accuracy:.3f}% - {class_name}")

# # acc_list = []
# trivial_count = 0
# # print("Compare accuracies of description and CLIP-Standard")
# # for i, acc_class_wise in class_wise_lang_accuracy.items():
# #     for j, acc_clip_class_wise in class_wise_clip_accuracy.items():
# #         if i == j:
# #             class_name = dataset_classes[i]
# #             acc = acc_class_wise.compute().item() - acc_clip_class_wise.compute().item()
# #             acc_list.append(acc)
# #             if acc > 0.01 or acc < -0.01:
# #                 print(f"Desc. Acc. - CLIP Acc.: {acc:.3f}% - {class_name}")
# #             else:
# #                 trivial_count += 1
# #                 print(f"Desc. Acc. - CLIP Acc.: Trivial - {class_name}")
# # print("Trivial count: ", trivial_count)

# # for i in range(len(sorted(acc_list))):
# #     print(f"{sorted(acc_list)[i]}")
# # print(sum(acc_list))

# class_wise_accuracies = {}
# differences = {}

# for i in range(num_classes):
#     class_name = dataset_classes[i]
#     desc_accuracy = 100 * class_wise_lang_accuracy[i].compute().item()
#     clip_accuracy = 100 * class_wise_clip_accuracy[i].compute().item()
    
#     # Calculate the difference between description-based and CLIP-standard accuracies
#     difference = desc_accuracy - clip_accuracy
    
#     # Store accuracies and their difference in the dictionary
#     class_wise_accuracies[class_name] = {
#         "Description-based Accuracy": desc_accuracy,
#         "CLIP-Standard Accuracy": clip_accuracy,
#         "Difference": difference
#     }
    
#     # Also store the difference separately for sorting
#     differences[class_name] = difference

# # Sort classes by the magnitude of difference
# sorted_classes_by_difference = sorted(differences, key=differences.get, reverse=True)

# # Reorganize the class-wise accuracies based on the sorted order
# sorted_class_wise_accuracies = {class_name: class_wise_accuracies[class_name] for class_name in sorted_classes_by_difference}

# # Print overall accuracies
# experimental_results = {}
# experimental_results["Class-wise Accuracies and Differences (Top 10)"] = [list(sorted_class_wise_accuracies.keys())[:10]]
# experimental_results["Class-wise Accuracies and Differences (Bottom 10)"] = [list(sorted_class_wise_accuracies.keys())[-10:]]
# experimental_results["Trivial Count"] = trivial_count
# experimental_results["Class-wise Accuracies and Differences"] = sorted_class_wise_accuracies
# experimental_results["Total Description-based Top-1 Accuracy: "] = 100*overall_lang_accuracy_metric.compute().item()
# experimental_results["Total Description-based Top-5 Accuracy: "] = 100*overall_lang_accuracy_metric_top5.compute().item()
# experimental_results["Total CLIP-Standard Top-1 Accuracy: "] = 100*overall_clip_accuracy_metric.compute().item()
# experimental_results["Total CLIP-Standard Top-5 Accuracy: "] = 100*overall_clip_accuracy_metric_top5.compute().item()

# # Ensure the structure 'model_size' > 'dataset' > 'freq_type'
# model_size = hparams['model_size']
# dataset_name = hparams['dataset_name']

# if model_size not in results:
#     results[model_size] = {}

# if dataset_name not in results[model_size]:
#     results[model_size][dataset_name] = {}

# if frequency_type not in results[model_size][dataset_name]:
#     results[model_size][dataset_name][frequency_type] = {}

# # Store results
# results[model_size][dataset_name][frequency_type] = experimental_results

# # Save the updated results
# # save_results(results, results_file_path)

# print(f"CLIP Model: {hparams['model_size']}",
#       f"|| Desc. Source: {hparams['desc_type']}",
#       f"|| Dataset being tested: {hparams['dataset']}",
#       f"|| Method: {hparams['method']}",
#       f"|| Cut Proportion: {cut_proportion}",
#       f"|| Freq. Penalisation Type: {frequency_type}",
#       f"|| Sim. Penalisation: {similarity_penalty_config}")
# print("Total Description-based Top-1 Accuracy: ", 100 * overall_lang_accuracy_metric.compute().item(), "%")
# print("Total Description-based Top-5 Accuracy: ", 100 * overall_lang_accuracy_metric_top5.compute().item(), "%")
# print("Total CLIP-Standard Top-1 Accuracy: ", 100 * overall_clip_accuracy_metric.compute().item(), "%")
# print("Total CLIP-Standard Top-5 Accuracy: ", 100 * overall_clip_accuracy_metric_top5.compute().item(), "%")
# print("Class-wise Accuracies and Differences (Top 10 and Bottom 10):\n", list(sorted_class_wise_accuracies.keys())[:10], "\n", list(sorted_class_wise_accuracies.keys())[-10:])



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
from load import set_hparams, compute_description_encodings, compute_label_encodings

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

# desc_encs = desc_encs.to(device)
# label_encs = label_encs.to(device)

# --- Prepare data for tSNE (mean of descriptions per class) ---
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

scale = len(class_names) / 100
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

# ---- Compute mean image embeddings per class ----
# Collect image embeddings by class
class_image_embeds = [[] for _ in range(len(dataset_classes))]

for images, labels in tqdm(dataloader, desc="Computing mean image embeddings per class"):
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        image_embeds = model.encode_image(images)
        image_embeds = F.normalize(image_embeds, dim=-1)
    for emb, label in zip(image_embeds, labels):
        class_image_embeds[label.item()].append(emb.cpu())

# Compute mean embedding for each class
mean_img_embeds = []
for embeds in class_image_embeds:
    if len(embeds) > 0:
        mean_embed = torch.stack(embeds, dim=0).mean(0)
        mean_embed = F.normalize(mean_embed, dim=0)
        mean_img_embeds.append(mean_embed)
    else:
        mean_img_embeds.append(torch.zeros_like(image_embeds[0]))

mean_img_embeds = torch.stack(mean_img_embeds, dim=0).numpy()

# ---- Project mean image embeddings into tSNE ----
# Use the *same* tsne object as for text (recommended for fair 2D mapping)
mean_img_embeds_2d = tsne.fit_transform(np.concatenate([X, mean_img_embeds], axis=0))[-len(mean_img_embeds):]

# ---- Plot mean image embedding for each class ----
plt.scatter(mean_img_embeds_2d[:, 1], mean_img_embeds_2d[:, 0], 
            marker='X', color='red', label='Mean Image Embedding', s=120, edgecolor='black', linewidth=1.2)
# Optional: Label a few for inspection
for i in range(len(mean_img_embeds_2d)):
    if i < 10 or len(mean_img_embeds_2d) < 20:  # Label all for tiny datasets, or first 10 for large
        plt.text(mean_img_embeds_2d[i, 1], mean_img_embeds_2d[i, 0], dataset_classes[i], fontsize=8, color='red')


# Save the plot
plt.grid()
plt.show()
filename = f"{method}_embedding_plot_{dataset}.png"
plt.savefig(f"figs/tsne_vis/{filename}", dpi=200, bbox_inches="tight")
print("Plot saved as ", filename, " in figs/tsne_vis/")