import itertools
import copy
import json
from load import set_hparams, update_hparams, compute_description_encodings, compute_label_encodings, aggregate_similarity
from loading_helpers import seed_everything
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import clip
import torchmetrics
from tqdm import tqdm
from time import time

def load_existing_results(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

def run_experiments():

    # model_sizes = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    # desc_types = ['gpt3', 'gpt4o']
    # datasets = ['imagenet', 'imagenetv2', 'cub', 'eurosat', 'places365', 'food101', 'pets', 'dtd']
    # methods = ['clip', 'e-clip', 'd-clip', 'waffleclip', 'defntaxs']
    
    # Test with a smaller subset of the above parameters
    model_sizes = ['ViT-B/32', 'ViT-B/16']
    desc_types = ['gpt3']
    datasets = ['cub']
    methods = ['d-clip']
    
    # Path to the results file
    results_file_path = 'results/all_backbone_method_dataset_experiment_results.json'
    
    # Load existing results
    all_results = load_existing_results(results_file_path)

    failed_experiments = {}

    count = 1
    start_time = time()

    # Loop through all combinations of model_size, desc_type, dataset, and method
    for model_size, desc_type, dataset, method in itertools.product(model_sizes, desc_types, datasets, methods):
        # Check if this combination of model_size, desc_type, dataset, and method has already been run
        if desc_type in all_results and model_size in all_results[desc_type] and method in all_results[desc_type][model_size] and dataset in all_results[desc_type][model_size][method]:
            print(f"Skipping existing experiment: model_size={model_size}, desc_type={desc_type}, dataset={dataset}, method={method}")

            avg_runtime = (time() - start_time) / count
            count += 1
            print(f"Average runtime: {avg_runtime:.2f} seconds. Expected time for all experiments: {avg_runtime * len(model_sizes) * len(desc_types) * len(datasets) * len(methods):.2f} seconds")

            continue

        # Make a deep copy of the original hparams to avoid modifying the original
        # current_hparams = copy.deepcopy(hparams)
        
        # Set hparams for the current experiment
        hparams = set_hparams(model_size, desc_type, dataset, method)

        # Update hparams and other variables based on the current experiment using update_hparams
        hparams, tfms, dataset, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = update_hparams(hparams)
        
        
        print(f"\nRunning experiment with model_size: {model_size}, desc_type: {desc_type}, dataset: {dataset}, method: {method}")
        
        # Run the main experiment logic from main.py
        try:
            results = run_single_experiment(hparams, tfms, dataset, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes)
            avg_runtime = (time() - start_time) / count
            count += 1
            print(f"Average runtime: {avg_runtime:.2f} seconds. Expected time for all experiments: {avg_runtime * len(model_sizes) * len(desc_types) * len(datasets) * len(methods):.2f} seconds")
            # Store the results in the all_results dictionary
            if model_size not in all_results:
                all_results[model_size] = {}
            if method not in all_results[model_size]:
                all_results[model_size][method] = {}
            all_results[model_size][method][dataset] = results
        except Exception as e:
            failed_experiments[count] = [model_size, desc_type, dataset, method]
            print(f"Experiment failed for model_size: {model_size}, desc_type: {desc_type}, dataset: {dataset}, method: {method}")
            print(f"Error: {e}")

            avg_runtime = (time() - start_time) / count
            count += 1
            print(f"Average runtime: {avg_runtime:.2f} seconds. Expected time for all experiments: {avg_runtime * len(model_sizes) * len(desc_types) * len(datasets) * len(methods):.2f} seconds")
    
    # Save the updated results to the JSON file
    all_results["failed_experiments"] = failed_experiments
    save_results(all_results, results_file_path)
    
    print(f"All results have been saved to {results_file_path}")

def run_single_experiment(hparams, tfms, dataset, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes):

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

    n_classes = len(dataset_classes)

    # Encode descriptions and labels
    print("Encoding descriptions...")
    description_encodings = compute_description_encodings(model, gpt_descriptions, hparams, batch_size=32)
    label_encodings = compute_label_encodings(model, hparams)

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

    for batch_number, (images, labels) in enumerate(tqdm(dataloader)):    
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        # Compute similarities and make predictions
        image_labels_similarity = image_encodings @ label_encodings.T
        
        # Update overall accuracies for CLIP
        overall_clip_accuracy_metric(image_labels_similarity, labels)
        overall_clip_accuracy_metric_top5(image_labels_similarity, labels)

        # Compute description-based predictions
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes

        for i, (k, v) in enumerate(description_encodings.items()):
            dot_product_matrix = image_encodings @ v.T
            
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
        # Update overall accuracies for descriptions
        overall_lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
        overall_lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)

    class_wise_accuracies = {}

    for i in range(num_classes):
        class_name = dataset_classes[i]
        desc_accuracy = 100 * class_wise_lang_accuracy[i].compute().item()
        clip_accuracy = 100 * class_wise_clip_accuracy[i].compute().item()
        
        # Store accuracies and their difference in the dictionary
        class_wise_accuracies[class_name] = {
            "Description-based Accuracy": desc_accuracy,
            "CLIP-Standard Accuracy": clip_accuracy
        }

    # Print overall accuracies
    experimental_results = {}
    experimental_results[f"{hparams['method'].capitalize()} Accuracy: "] = 100*overall_lang_accuracy_metric.compute().item()
    experimental_results["CLIP Accuracy: "] = 100*overall_clip_accuracy_metric.compute().item()

    return experimental_results


start_time = time()
run_experiments()
end_time = time()

print(f"Total time taken: {end_time - start_time:.2f} seconds")