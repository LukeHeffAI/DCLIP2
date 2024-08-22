import json
from load import *
import torchmetrics
from tqdm import tqdm
import torch
import time
import numpy as np

def load_or_initialise_results(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    
def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

def run_experiment(cut_proportion, dataset_name, similarity_penalty_config, frequency_penalty_type, results_file_path, template_index):
    results = load_or_initialise_results(results_file_path)

    # Initialize the environment
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
    description_encodings = compute_description_encodings(model)
    label_encodings = compute_label_encodings(model)

    # Number of classes
    num_classes = len(dataset.classes)

    # Evaluation metrics for overall and per-class accuracies
    print("Evaluating...")
    print(f"Cut Proportion: {cut_proportion}", f"|| Dataset: {dataset_name}", f"|| Sim. Penalty: {similarity_penalty_config}", f"|| Freq. Penalty: {frequency_penalty_type}", f"|| Template: {imagenet_templates[template_index]}")
    overall_lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    overall_lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    overall_clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    overall_clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    # Initialize dictionaries to track class-wise accuracy
    class_wise_lang_accuracy = {i: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) for i in range(num_classes)}
    class_wise_clip_accuracy = {i: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) for i in range(num_classes)}

    for batch_number, (images, labels) in enumerate(tqdm(dataloader)):    
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        # Compute similarities and make predictions
        image_labels_similarity = image_encodings @ label_encodings.T
        clip_predictions = image_labels_similarity.argmax(dim=1)
        
        # Update overall and class-wise accuracies for CLIP
        overall_clip_accuracy_metric(image_labels_similarity, labels)
        overall_clip_accuracy_metric_top5(image_labels_similarity, labels)
        for i in range(num_classes):
            class_mask = labels == i
            if class_mask.any():
                class_wise_clip_accuracy[i](clip_predictions[class_mask], labels[class_mask])

        # Compute description-based predictions
        image_description_similarity = [None]*num_classes
        image_description_similarity_cumulative = [None]*num_classes

        for i, (k, v) in enumerate(description_encodings.items()):
            dot_product_matrix = image_encodings @ v.T
            
            if frequency_penalty_type:
                for descriptor in gpt_descriptions[k]:
                    freq = descriptors_freq[frequency_type].get(descriptor, 1)
                    norm_freq = freq / sum(descriptors_freq[frequency_type].values())
                    penalty_index = gpt_descriptions[k].index(descriptor)
                    dot_product_matrix[:, penalty_index] /= norm_freq

            if similarity_penalty_config:
                class_average_sim = average_cosine_similarities.get(k, 0)  # Default to 0 if not found
                dot_product_matrix -= class_average_sim
            
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

        cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
        descr_predictions = cumulative_tensor.argmax(dim=1)
        
        # Update overall and class-wise accuracies for descriptions
        overall_lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
        overall_lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
        for i in range(num_classes):
            class_mask = labels == i
            if class_mask.any():
                class_wise_lang_accuracy[i](descr_predictions[class_mask], labels[class_mask])

    # Save results for current experiment
    save_experiment_results(results, cut_proportion, dataset_name, similarity_penalty_config, frequency_penalty_type, overall_lang_accuracy_metric, overall_lang_accuracy_metric_top5, overall_clip_accuracy_metric, overall_clip_accuracy_metric_top5, class_wise_lang_accuracy, class_wise_clip_accuracy, imagenet_templates, template_index)

    # Save the updated results
    save_results(results, results_file_path)

prepend_results = []

def save_experiment_results(results, cut_proportion, dataset_name, similarity_penalty_config, frequency_penalty_type, overall_lang_accuracy_metric, overall_lang_accuracy_metric_top5, overall_clip_accuracy_metric, overall_clip_accuracy_metric_top5, class_wise_lang_accuracy, class_wise_clip_accuracy, imagenet_templates, template_index):
    class_wise_accuracies = {}
    differences = {}
    num_classes = len(class_wise_lang_accuracy)
    for i in range(num_classes):
        class_name = dataset.classes[i]
        desc_accuracy = 100 * class_wise_lang_accuracy[i].compute().item()
        clip_accuracy = 100 * class_wise_clip_accuracy[i].compute().item()
        
        # Calculate the difference between description-based and CLIP-standard accuracies
        difference = desc_accuracy - clip_accuracy
        
        # Store accuracies and their difference in the dictionary
        class_wise_accuracies[class_name] = {
            "Description-based Accuracy": desc_accuracy,
            "CLIP-Standard Accuracy": clip_accuracy,
            "Difference": difference
        }
        
        # Also store the difference separately for sorting
        differences[class_name] = difference

    # Sort classes by the magnitude of difference
    sorted_classes_by_difference = sorted(differences, key=differences.get, reverse=True)

    # Reorganize the class-wise accuracies based on the sorted order
    sorted_class_wise_accuracies = {class_name: class_wise_accuracies[class_name] for class_name in sorted_classes_by_difference}

    # Print overall accuracies
    experimental_results = {}
    experimental_results["Class-wise Accuracies and Differences (Top 10)"] = [list(sorted_class_wise_accuracies.keys())[:10]]
    experimental_results["Class-wise Accuracies and Differences (Bottom 10)"] = [list(sorted_class_wise_accuracies.keys())[-10:]]
    experimental_results["Class-wise Accuracies and Differences"] = sorted_class_wise_accuracies
    experimental_results["Total Description-based Top-1 Accuracy: "] = 100 * overall_lang_accuracy_metric.compute().item()
    experimental_results["Total Description-based Top-5 Accuracy: "] = 100 * overall_lang_accuracy_metric_top5.compute().item()
    experimental_results["Total CLIP-Standard Top-1 Accuracy: "] = 100 * overall_clip_accuracy_metric.compute().item()
    experimental_results["Total CLIP-Standard Top-5 Accuracy: "] = 100 * overall_clip_accuracy_metric_top5.compute().item()

    print('Results:', experimental_results["Total Description-based Top-1 Accuracy: "]
          , experimental_results["Total CLIP-Standard Top-1 Accuracy: "])
    prepend_results.append(experimental_results["Total Description-based Top-1 Accuracy: "])

    # Ensure the structure 'model_size' > 'dataset' > 'cut_proportion' > 'similarity_penalty_config' > 'frequency_penalty_type'
    model_size = hparams['model_size']

    if model_size not in results:
        results[model_size] = {}

    if dataset_name not in results[model_size]:
        results[model_size][dataset_name] = {}

    if str(cut_proportion) not in results[model_size][dataset_name]:
        results[model_size][dataset_name][str(cut_proportion)] = {}

    if str(similarity_penalty_config) not in results[model_size][dataset_name][str(cut_proportion)]:
        results[model_size][dataset_name][str(cut_proportion)][str(similarity_penalty_config)] = {}

    if str(frequency_penalty_type) not in results[model_size][dataset_name][str(cut_proportion)][str(similarity_penalty_config)]:
        results[model_size][dataset_name][str(cut_proportion)][str(similarity_penalty_config)][str(frequency_penalty_type)] = {}

    if imagenet_templates[template_index].split('{')[0].capitalize() not in results[model_size][dataset_name][str(cut_proportion)][str(similarity_penalty_config)][str(frequency_penalty_type)]:
        results[model_size][dataset_name][str(cut_proportion)][str(similarity_penalty_config)][str(frequency_penalty_type)][imagenet_templates[template_index].split('{')[0].capitalize()] = {}

    # Store results
    results[model_size][dataset_name][str(cut_proportion)][str(similarity_penalty_config)][str(frequency_penalty_type)] = experimental_results

# Main loop
results_file_path = 'results/experiment_loop_results.json'

# Example of variable configurations
cut_proportions = [# 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                   # 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
                   1.0]
datasets = ['cub',
            #*['cub_gpt4_{0}_desc'.format(i) for i in range(1, 9)]
            ]
similarity_penalty_configs = [False]
frequency_penalty_types = [None]

template_indicies = np.linspace(1, len(imagenet_templates)).astype(int)

total_experiment_count = len(cut_proportions) * len(datasets) * len(similarity_penalty_configs) * len(frequency_penalty_types) * len(template_indicies)

start_time = time.time()

experiment_tally = 0
for cut_proportion in cut_proportions:
    for dataset_name in datasets:
        for similarity_penalty_config in similarity_penalty_configs:
            for frequency_penalty_type in frequency_penalty_types:
                for template_index in template_indicies:
                    experiment_tally += 1
                    print(f"Running experiment {experiment_tally} of {total_experiment_count}...")
                    run_experiment(cut_proportion, dataset_name, similarity_penalty_config, frequency_penalty_type, results_file_path, template_index)
                    print("Results:", prepend_results)

end_time = time.time()
print(f"Total time taken to run {total_experiment_count} experiments: {end_time - start_time} seconds")
