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
from create_subcategories import create_subcategories


def load_existing_results(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_results(results, file_path):
    tmp = file_path + ".tmp"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=4)
    os.replace(tmp, file_path)


def run_experiments(num_runs=3, force_regenerate_subcategories=True, max_classes_per_subcategory=10):
    """
    Run multiple experiments across different model configurations.
    
    Args:
        num_runs: Number of times to repeat each experiment configuration
        force_regenerate_subcategories: Whether to regenerate subcategories before each run
        max_classes_per_subcategory: Maximum classes per subcategory for this run set
    """
    model_sizes     = ['ViT-B/32', 'ViT-B/16']  # Choosing this order for medium range length of experiment
    desc_types      = ['gpt-3']
    datasets        = ['cub', 'pets', 'dtd', 'food101', 'places365', 'eurosat']
    methods         = ['defntaxs']
    context_indices = list(range(5))  # Up to 5 context options for each dataset

    total_configs = len(model_sizes) * len(desc_types) * len(datasets) * len(methods) * len(context_indices)
    total_runs    = total_configs * num_runs
    print(f"Will attempt up to {total_runs} runs ({num_runs} per config)")
    
    results_file = 'results/subcat_hparam_tests_new.json'
    all_results = load_existing_results(results_file)

    # Ensure minimal nesting: method -> model_size -> list
    all_results.setdefault('failed_experiments', {})
    for method in methods:
        all_results.setdefault(method, {})
        for ms in model_sizes:
            all_results[method].setdefault(ms, [])

    start_time = time()
    completed = 0
    failures  = all_results['failed_experiments']

    # Main experiment loop
    for model_size, desc_type, current_dataset, method, context_idx in itertools.product(
        model_sizes, desc_types, datasets, methods, context_indices
    ):
        runs_list = all_results[method][model_size]

        # Identify existing runs matching this config
        matching = [r for r in runs_list
                    if r['desc_type']==desc_type
                    and r['dataset']==current_dataset
                    and r['context_idx']==context_idx
                    and r['max_classes_per_subcategory']==max_classes_per_subcategory]
        done  = len(matching)
        to_do = max(0, num_runs - done)

        if to_do == 0:
            print(f"Skipping {method}/{model_size}/{desc_type}/{current_dataset}/ctx{context_idx}: {done}/{num_runs} done")
            completed += done
            continue

        max_run_id = max((r.get('run_id', 0) for r in matching), default=0)
        print(f"{done}/{num_runs} exist for {current_dataset}; running {to_do} more (starting at run_id={max_run_id+1})")

        # Run the required experiments
        for i in range(to_do):
            run_id = max_run_id + i + 1
            try:
                # Generate subcategories if this is a new run and regeneration is forced
                if method == 'defntaxs' and force_regenerate_subcategories:
                    hparams, _, _, _, _, _, _, _, _ = set_hparams(
                                model_size=model_size,
                                desc_type=desc_type,
                                dataset=current_dataset,
                                method=method,
                                subcategory_context_idx=context_idx,
                                run_id=run_id,
                                max_classes_per_subcategory=max_classes_per_subcategory
                    )
                    # Check if subcategories already exist
                    if os.path.exists(f'class_analysis/json/versions/class_analysis_{hparams["dataset"]}_run{hparams["seed"]}_mcps{hparams["max_classes_per_subcategory"]}.json') != True:
                        # Create subcategories
                        print(f"Creating subcategories for {current_dataset} run {run_id} (ctx={context_idx}, max={max_classes_per_subcategory})")
                        create_subcategories(
                            hparams, force=True, max_workers=20,
                            max_classes_per_subcategory=max_classes_per_subcategory
                        )
                    else:
                        print(f"Subcategory data already exists for: {current_dataset}, run {run_id}, mcps={max_classes_per_subcategory}")

                seed_everything(run_id)
                hparams, tfms, ds_loader, ds_classes, class_subcats, gpt_descs, unmod, label_to_classname, n_classes = set_hparams(
                    model_size=model_size,
                    desc_type=desc_type,
                    dataset=current_dataset,
                    method=method,
                    subcategory_context_idx=context_idx,
                    run_id=run_id,
                    max_classes_per_subcategory=max_classes_per_subcategory
                )

                print(f"Starting run #{run_id}")
                results = run_single_experiment(
                    hparams, tfms, ds_loader, ds_classes,
                    gpt_descs, label_to_classname, n_classes
                )

                # Attach metadata
                results.update({
                    'desc_type': desc_type,
                    'model_size': model_size,
                    'method': method,
                    'dataset': current_dataset,
                    'context_idx': context_idx,
                    'max_classes_per_subcategory': max_classes_per_subcategory,
                    'run_id': run_id,
                    'seed': hparams['seed'],
                    'timestamp': time()
                })

                runs_list.append(results)
                save_results(all_results, results_file)
                completed += 1

            except Exception as e:
                key = f"{method}_{model_size}_{desc_type}_{current_dataset}_ctx{context_idx}_run{run_id}_mcps{max_classes_per_subcategory}"
                failures[key] = {
                    'desc_type': desc_type,
                    'model_size': model_size,
                    'method': method,
                    'dataset': current_dataset,
                    'context_idx': context_idx,
                    'run_id': run_id,
                    'error': str(e)
                }
                print(f"Failure recorded: {key}")
                save_results(all_results, results_file)

    elapsed = time() - start_time
    print(f"Done: {completed}/{total_runs} succeeded in {elapsed:.1f}s. Failures: {len(failures)}")


def run_single_experiment(hparams, tfms, dataset_loader, dataset_classes, gpt_descriptions, label_to_classname, n_classes):
    """Run a single experiment with the given configuration."""

    # Prepare the data loader
    bs = hparams['batch_size']
    dataloader = DataLoader(dataset_loader, bs, shuffle=False, num_workers=8, pin_memory=True)

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


# if __name__ == "__main__":
#     num_runs = 5
#     force_regenerate = True
    
#     # for max_classes_per_subcategory in [5, 8, 12, 18, 20, 25, 30, 40]:
#     # for max_classes_per_subcategory in [40, 30, 25, 20, 18, 12, 8, 5]:
#     for max_classes_per_subcategory in [40, 30, 25, 20, 18, 12, 8, 5]:
#         start_time = time()
#         run_experiments(num_runs=num_runs,
#                         force_regenerate_subcategories=force_regenerate,
#                         max_classes_per_subcategory=max_classes_per_subcategory)
#         end_time = time()
#         print(f"Total time taken: {end_time - start_time:.2f} seconds / {(end_time - start_time)/3600:.2f} hours")

#     print("All experiments completed.")

# The following code opens the existing results file, loads the results into a dictionary, deletes all entries containing the parameter 'dtd', and then saves the updated dictionary back to the file.
# This is due to an error in the dataset loading process that caused all results to be duplicates.
#
# For context, the results file is in the form: { "defntaxs": { "ViT-B/32": [ { "desc_type": "gpt-3", "dataset": "cub", "context_idx": 0, "max_classes_per_subcategory": 10, "run_id": 1, "seed": 42, "timestamp": 1234567890 } ] } }
if __name__ == "__main__":
    results_file = 'results/subcat_hparam_tests_new.json'
    all_results = load_existing_results(results_file)
    # Remove all entries with 'dtd' in the dataset name
    for method, method_results in all_results.items():
        for model_size, model_results in method_results.items():
            all_results[method][model_size] = [result for result in model_results if 'dtd' not in result['dataset']]
    # Save the updated results back to the file
    save_results(all_results, results_file)
