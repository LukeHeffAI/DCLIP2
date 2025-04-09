import itertools
import json
import os
from load import set_hparams, compute_description_encodings, compute_label_encodings, aggregate_similarity
from loading_helpers import seed_everything
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import clip
import torchmetrics
from tqdm import tqdm
from time import time
from create_subcategories_batch import create_subcategories

def load_existing_results(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_results(results, file_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

def run_experiments(num_runs=3, force_regenerate_subcategories=True, max_classes_per_subcategory=10):
    """
    Run multiple experiments across different model configurations.
    
    Args:
        num_runs: Number of times to repeat each experiment configuration
        force_regenerate_subcategories: Whether to regenerate subcategories before each run
        max_classes_per_subcategory: Maximum classes per subcategory for this run set
    """
    model_sizes = ['ViT-B/16']  # Choosing this order for medium range length of experiment, for best estimate of time for all experiments
    # model_sizes = ['ViT-B/32']
    desc_types = ['gpt-3']
    datasets = ['cub', 'eurosat', 'places365', 'food101', 'pets', 'dtd']
    # datasets = ['cub', 'eurosat', 'pets']
    # methods = ['clip', 'e-clip', 'd-clip', 'waffleclip', 'waffleclip+concepts', 'defntaxs']
    # methods = ['clip', 'e-clip', 'd-clip', 'defntaxs']
    methods = ['defntaxs']

    total_experiments = len(model_sizes) * len(desc_types) * len(datasets) * len(methods) * num_runs
    print(f"Conducting {num_runs} iterations of {len(model_sizes) * len(desc_types) * len(datasets) * len(methods)} experiment configurations ({total_experiments} total runs).")
    
    # Path to the results file
    results_file_path = 'results/subcat_hparam_tests.json'
    
    # Load existing results
    all_results = load_existing_results(results_file_path)
    
    # Track failures
    failed_experiments = all_results.get("failed_experiments", {})

    count = 1
    runs_completed = 0
    start_time = time()

    # Loop through all combinations of model configurations
    for model_size, desc_type, current_dataset, method in itertools.product(model_sizes, desc_types, datasets, methods):
        # Initialize result structure for this configuration if it doesn't exist
        if desc_type not in all_results:
            all_results[desc_type] = {}
        if model_size not in all_results[desc_type]:
            all_results[desc_type][model_size] = {}
        if method not in all_results[desc_type][model_size]:
            all_results[desc_type][model_size][method] = {}
        if current_dataset not in all_results[desc_type][model_size][method]:
            all_results[desc_type][model_size][method][current_dataset] = []
        
        # Get current results list for this configuration
        current_results = all_results[desc_type][model_size][method][current_dataset]
        
        # Filter results to only include those with matching max_classes_per_subcategory
        matching_results = [
            result for result in current_results 
            if result.get("Est. classes per subcategory") == max_classes_per_subcategory
        ]
        
        # Find the highest run_id already completed for this specific parameter value
        completed_run_ids = [result.get("run_id", 0) for result in matching_results] if matching_results else []
        
        max_run_id = max(completed_run_ids) if completed_run_ids else 0
        remaining_runs = max(0, num_runs - len(matching_results))
        runs_completed += len(matching_results)

        # Skip if all runs with this specific max_classes_per_subcategory are already completed
        if remaining_runs <= 0:
            print(f"Skipping {desc_type}/{model_size}/{method}/{current_dataset} - Already have {len(matching_results)} runs with max_classes_per_subcategory={max_classes_per_subcategory}")
            continue
        
        print(f"Found {len(matching_results)} completed runs for max_classes_per_subcategory={max_classes_per_subcategory}.")
        print(f"Running {remaining_runs} more runs to reach target of {num_runs}.")
        
        # Run the remaining experiments
        for i in range(remaining_runs):
            run_idx = max_run_id + i + 1
            
            try:
                # Set up parameters for this specific run
                seed = run_idx
                seed_everything(seed)
                hparams, _, _, _, _, _, _, _, _ = set_hparams(
                    model_size=model_size, 
                    desc_type=desc_type, 
                    dataset=current_dataset, 
                    method=method
                )
                
                # Load dataset
                print(f"\n\nRunning experiment {count} of {total_experiments - runs_completed}: {desc_type} {model_size} {method} {current_dataset} (Run {run_idx}/{num_runs})")
                
                # Set maximum classes per subcategory to the current loop value
                print(f"Using max_classes_per_subcategory: {max_classes_per_subcategory}")
                
                # Create subcategories
                if force_regenerate_subcategories and method in ["defntaxs"]:
                    create_subcategories(hparams, force=True, max_classes_per_subcategory=max_classes_per_subcategory)
                
                # Set hparams for the current experiment
                hparams, tfms, dataset_loader, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(model_size, desc_type, current_dataset, method)
                hparams['seed'] = hparams['seed'] + run_idx
                seed_everything(hparams['seed'])
                
                # Run the experiment
                print(f"Running experiment with model_size: {model_size}, desc_type: {desc_type}, dataset: {current_dataset}, method: {method}, run: {run_idx}")
                results = run_single_experiment(hparams, tfms, dataset_loader, dataset_classes, gpt_descriptions, label_to_classname, n_classes)
                
                # Add the parameter information to results
                results["Est. classes per subcategory"] = max_classes_per_subcategory
                results["run_id"] = run_idx
                results["timestamp"] = time()
                
                # Append results to the current dataset
                current_results.append(results)
                
                # Save after each experiment run
                all_results[desc_type][model_size][method][current_dataset] = current_results
                save_results(all_results, results_file_path)
                
                # Track progress and estimate remaining time
                count += 1
                elapsed_time = time() - start_time
                avg_time_per_exp = elapsed_time / count
                remaining_exps = total_experiments - runs_completed - count
                est_remaining_time = avg_time_per_exp * remaining_exps
                print(f"Progress: {count}/{total_experiments - runs_completed} experiments completed")
                print(f"Average time per experiment: {avg_time_per_exp:.2f} seconds")
                print(f"Estimated time remaining: {est_remaining_time:.2f} seconds ({est_remaining_time/3600:.2f} hours)")
                
            except Exception as e:
                # Record the failed experiment
                failure_key = f"{model_size}_{desc_type}_{current_dataset}_{method}_run{run_idx}"
                failed_experiments[failure_key] = {
                    "model_size": model_size, 
                    "desc_type": desc_type, 
                    "dataset": current_dataset,
                    "method": method,
                    "run_idx": run_idx,
                    "error": str(e)
                }
                
                print(f"Experiment failed: {failure_key}")
                print(f"Error: {e}")
                
                # Save failure information
                all_results["failed_experiments"] = failed_experiments
                save_results(all_results, results_file_path)
                
                # Continue with next experiment
                count += 1
    
    # Save final results
    save_results(all_results, results_file_path)
    print(f"All results have been saved to {results_file_path}")


def run_single_experiment(hparams, tfms, dataset_loader, dataset_classes, gpt_descriptions, label_to_classname, n_classes):
    """Run a single experiment with the given configuration."""


    # Prepare the data loader
    bs = hparams['batch_size']
    dataloader = DataLoader(dataset_loader, bs, shuffle=False, num_workers=16, pin_memory=True)

    # Load the model and preprocessing
    print("Loading model...")
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    # Encode descriptions and labels
    print("Encoding descriptions...")
    description_encodings = compute_description_encodings(model, gpt_descriptions, hparams, batch_size=32)
    label_encodings = compute_label_encodings(model, hparams, label_to_classname)

    # Number of classes
    num_classes = len(dataset_classes)

    # Evaluation metrics
    print("Evaluating...")
    overall_lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    overall_lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    overall_clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    overall_clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    for batch_number, (images, labels) in enumerate(tqdm(dataloader)):    
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        # Compute similarities and make predictions for CLIP
        image_labels_similarity = image_encodings @ label_encodings.T
        overall_clip_accuracy_metric(image_labels_similarity, labels)
        overall_clip_accuracy_metric_top5(image_labels_similarity, labels)

        # Compute description-based predictions
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes

        for i, (k, v) in enumerate(description_encodings.items()):
            dot_product_matrix = image_encodings @ v.T
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

        cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
        
        # Update overall accuracies for descriptions
        overall_lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
        overall_lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)

    # Calculate accuracy values
    method_top1 = 100*overall_lang_accuracy_metric.compute().item()
    method_top5 = 100*overall_lang_accuracy_metric_top5.compute().item()
    clip_top1 = 100*overall_clip_accuracy_metric.compute().item()
    clip_top5 = 100*overall_clip_accuracy_metric_top5.compute().item()

    # Prepare results dictionary with consistent keys
    experimental_results = {
        f"{hparams['method'].capitalize()} Top-1 Accuracy": method_top1,
        f"{hparams['method'].capitalize()} Top-5 Accuracy": method_top5,
        "CLIP Top-1 Accuracy": clip_top1,
        "CLIP Top-5 Accuracy": clip_top5,
        "seed": hparams['seed']
    }

    # Print results summary with matching keys
    print(f"\nResults for {hparams['method']} on {hparams['dataset_name']}:")
    print(f"{hparams['method'].capitalize()} Top-1 Accuracy: {method_top1:.2f}%")
    print(f"CLIP Top-1 Accuracy: {clip_top1:.2f}%")

    return experimental_results


if __name__ == "__main__":
    # Run experiments with specified number of runs per configuration
    # Change these parameters as needed
    num_runs = 6  # Number of times to run each configuration
    force_regenerate = True  # Whether to regenerate subcategories each time
    
    for max_classes_per_subcategory in [5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]:
        start_time = time()
        run_experiments(num_runs=num_runs, force_regenerate_subcategories=force_regenerate, max_classes_per_subcategory=max_classes_per_subcategory)
        end_time = time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds / {(end_time - start_time)/3600:.2f} hours")

    print("All experiments completed.")