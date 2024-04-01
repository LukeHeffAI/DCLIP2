from load import *
import torchmetrics
from tqdm import tqdm
import torch

# Initialize the environment
seed_everything(hparams['seed'])

# Prepare the data loader
bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=16, pin_memory=True)  # Shuffle should be False for class-wise evaluation

# Load the model and preprocessing
print("Loading model...")
device = torch.device(hparams['device'])
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

# Encode descriptions and labels
print("Encoding descriptions...")
description_encodings = compute_description_encodings(model)
label_encodings = compute_label_encodings(model)

# Number of classes
num_classes = len(dataset.classes)

# TODO: Move this to load.py
descriptor_frequencies = load_json('descriptor_freq_analysis/descriptors_freq_cub')
total_descriptors_is = sum(descriptor_frequencies['freq_is'].values())
total_descriptors_contains = sum(descriptor_frequencies['freq_contains'].values())
frequency_proportion_is = {desc: freq/total_descriptors_is for desc, freq in descriptor_frequencies['freq_is'].items()}
frequency_proportion_contains = {desc: freq/total_descriptors_contains for desc, freq in descriptor_frequencies['freq_contains'].items()}

print("Total descriptors (is): ", total_descriptors_is)

# # Evaluation metrics for overall and per-class accuracies
# print("Evaluating...")
# overall_lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
# overall_lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

# overall_clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
# overall_clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

# # Initialize dictionaries to track class-wise accuracy
# class_wise_lang_accuracy = {i: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) for i in range(num_classes)}
# class_wise_clip_accuracy = {i: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) for i in range(num_classes)}

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

    


# Print overall accuracies
print("\nDataset being tested: ", hparams['dataset'])
print("Total Description-based Top-1 Accuracy: ", 100 * overall_lang_accuracy_metric.compute().item(), "%")
print("Total Description-based Top-5 Accuracy: ", 100 * overall_lang_accuracy_metric_top5.compute().item(), "%")
print("Total CLIP-Standard Top-1 Accuracy: ", 100 * overall_clip_accuracy_metric.compute().item(), "%")
print("Total CLIP-Standard Top-5 Accuracy: ", 100 * overall_clip_accuracy_metric_top5.compute().item(), "%")

# Print class-wise accuracies
print("\nClass-wise Description-based Accuracy:")
for i, acc in class_wise_lang_accuracy.items():
    class_name = dataset.classes[i]
    accuracy = 100 * acc.compute().item()
    print(f"Desc. Acc.: {accuracy:.3f}% - {class_name}")

print("\nClass-wise CLIP-Standard Accuracy:")
for i, acc in class_wise_clip_accuracy.items():
    class_name = dataset.classes[i]
    accuracy = 100 * acc.compute().item()
    print(f"CLIP Acc.: {accuracy:.3f}% - {class_name}")

acc_list = []
trivial_count = 0
print("Compare accuracies of description and CLIP-Standard")
for i, acc_class_wise in class_wise_lang_accuracy.items():
    for j, acc_clip_class_wise in class_wise_clip_accuracy.items():
        if i == j:
            class_name = dataset.classes[i]
            acc = acc_class_wise.compute().item() - acc_clip_class_wise.compute().item()
            acc_list.append(acc)
            if acc > 0.001 or acc < -0.001:
                print(f"Desc. Acc. - CLIP Acc.: {acc:.3f}% - {class_name}")
            else:
                trivial_count += 1
                print(f"Desc. Acc. - CLIP Acc.: Trivial - {class_name}")
print("Trivial count: ", trivial_count)

# for i in range(len(sorted(acc_list))):
#     print(f"{sorted(acc_list)[i]}")
# print(sum(acc_list))

# print("\nDataset being tested: ", hparams['dataset'])

# accuracy_logs = {}
# accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
# accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

# accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
# accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

