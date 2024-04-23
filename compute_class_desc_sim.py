import json
from load import *
import torchmetrics
import torch
from descriptor_analysis import compute_class_list, compute_descriptor_list
import csv

results_file_path = 'results/experiment_results.json'

descriptor_file_path = hparams['descriptor_fname']

replace_list = ["-"," "]

class_descriptor_dict = load_json(descriptor_file_path)
class_list = compute_class_list(class_descriptor_dict, sort_config=True)
descriptor_list = compute_descriptor_list(class_descriptor_dict, sort_config=True)

class_list = [c.replace(replace_list[0], replace_list[1]) for c in class_list]

print(class_list[0:5], "\n", descriptor_list[0:5])

seed_everything(hparams['seed'])

# Prepare the data loader
bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=16, pin_memory=True)

# Load the model and preprocessing
print("Loading model...")
device = torch.device(hparams['device'])
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

# Encode descriptions and labels
print("Encoding descriptions...")
description_encodings = F.normalize(model.encode_text(clip.tokenize(descriptor_list).to(device)))
label_encodings = F.normalize(model.encode_text(clip.tokenize(class_list).to(device)))

cosine_similarity = torch.mm(description_encodings, label_encodings.T)


# Load JSON data
with open(results_file_path, 'r', encoding='utf-8') as results_file:
    classification_results_dict = json.load(results_file)

# Extract the relevant nested dictionary
class_acc_diff = classification_results_dict['ViT-B/32'][hparams['dataset_name']]['null']['Class-wise Accuracies and Differences']

# Transform keys and retain the full dictionary structure for each class
classification_results = {k.split('.')[-1].replace('_', ' '): v for k, v in class_acc_diff.items()}

# Open the CSV file for writing
with open('results/descriptor_similarity_cub.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    
    # Prepare the initial header with class names only
    header = [''] + [class_name for class_name in class_list]
    writer.writerow(header)

    # Prepare and write rows for each accuracy metric
    for metric in ["Description-based Accuracy", "CLIP-Standard Accuracy", "Difference"]:
        row = [metric]
        for class_name in class_list:
            metric_value = str(classification_results.get(class_name, {}).get(metric, ""))
            row.append(metric_value)
        writer.writerow(row)

    # Additional empty row for separation (optional)
    writer.writerow([''])

    # Prepare the header for the descriptor section
    descriptor_header = ['Descriptor'] + class_list
    writer.writerow(descriptor_header)

    # Write each descriptor and its associated similarity scores
    for i, descriptor in enumerate(descriptor_list):
        row = [descriptor] + [str(similarity) for similarity in cosine_similarity[i].tolist()]
        writer.writerow(row)
