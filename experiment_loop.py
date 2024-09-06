# experiment_loop.py

import torch
from joblib import Parallel, delayed
from itertools import product
import torch.nn.functional as F
from loading_helpers import seed_everything, load_gpt_descriptions
from load import dataset_classes, compute_description_encodings, compute_label_encodings
from main import dataloader
import torchmetrics
import clip

# Hyperparameter lists
model_size_list = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] # Omitting 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'
dataset_name_list = ['imagenet', 'imagenetv2', 'cub', 'cub_reassignment', 'cub_reassignment_threshold', 'cub_gpt4_8_desc', 'eurosat', 'places365', 'food101', 'pets', 'dtd']
cut_proportion_list = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
similarity_penalty_config_list = ['no_similarity_penalty', 'similarity_penalty']
frequency_penalty_config_list = ['no_freq_penalty', 'freq_is', 'freq_contains']

# Cache for storing computed encodings to avoid redundancy
cache = {}

# Function to free GPU memory after each run
def clear_gpu_cache():
    torch.cuda.empty_cache()

# Function to compute and cache description encodings
def compute_and_cache(model, dataset_name):
    key = (model, dataset_name)
    if key not in cache:
        description_encodings = compute_description_encodings(model)
        label_encodings = compute_label_encodings(model)
        cache[key] = (description_encodings, label_encodings)
    return cache[key]

# Function to run the experiment for a single configuration
def run_experiment(model_size, dataset_name, cut_proportion, similarity_penalty, frequency_penalty):
    # Initialize hyperparameters
    hparams = {
        'model_size': model_size,
        'dataset': dataset_name,
        'cut_proportion': cut_proportion,
        'similarity_penalty_config': similarity_penalty,
        'frequency_type': frequency_penalty,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size': 64,
        'seed': 1
    }
    
    # Set the seed for reproducibility
    seed_everything(hparams['seed'])
    
    # Load model and data
    model, preprocess = clip.load(hparams['model_size'], device=hparams['device'], jit=False)
    model.eval()
    
    # Cache the encodings to avoid recomputation
    description_encodings, label_encodings = compute_and_cache(model, hparams['dataset'])
    
    # Metrics initialization
    overall_lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(dataset_classes)).to(hparams['device'])
    overall_clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(dataset_classes)).to(hparams['device'])

    # Running the experiment with data loader
    for images, labels in dataloader:
        images, labels = images.to(hparams['device']), labels.to(hparams['device'])
        
        # Encode images using the model
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        # Compute similarities
        image_labels_similarity = image_encodings @ label_encodings.T
        clip_predictions = image_labels_similarity.argmax(dim=1)
        
        # Update CLIP-based accuracy
        overall_clip_accuracy_metric(image_labels_similarity, labels)

        # Compute description-based similarity and accuracy
        cumulative_tensor = torch.stack([image_encodings @ desc.T for desc in description_encodings.values()], dim=1)
        descr_predictions = cumulative_tensor.argmax(dim=1)
        overall_lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    
    # Compute final results
    lang_accuracy = overall_lang_accuracy_metric.compute().item()
    clip_accuracy = overall_clip_accuracy_metric.compute().item()

    # Free GPU memory
    clear_gpu_cache()

    # Print results (or store them in a more elaborate result-tracking mechanism)
    # print(f"Model: {model_size}, Dataset: {dataset_name}, Cut: {cut_proportion}, SimPenalty: {similarity_penalty}, FreqPenalty: {frequency_penalty}")
    # print(f"Description-based Accuracy: {lang_accuracy * 100:.2f}%, CLIP Accuracy: {clip_accuracy * 100:.2f}%")

# Function to determine valid permutations based on the new conditions
def get_valid_permutations():
    valid_combinations = []

    for model_size, dataset_name in product(model_size_list, dataset_name_list):
        # 1. When `cut_proportion` is varied, both `frequency_penalty` and `similarity_penalty` must be 'no_freq_penalty' and 'no_similarity_penalty'.
        for cut_proportion in cut_proportion_list:
            if cut_proportion != 1:
                valid_combinations.append((model_size, dataset_name, cut_proportion, 'no_similarity_penalty', 'no_freq_penalty'))

        # 2. When `similarity_penalty` is set, `cut_proportion` must be 1 and `frequency_penalty` must be 'no_freq_penalty'.
        for similarity_penalty in similarity_penalty_config_list:
            if similarity_penalty != 'no_similarity_penalty':
                valid_combinations.append((model_size, dataset_name, 1, similarity_penalty, 'no_freq_penalty'))

        # 3. When `frequency_penalty` is set, `cut_proportion` must be 1 and `similarity_penalty` must be 'no_similarity_penalty'.
        for frequency_penalty in frequency_penalty_config_list:
            if frequency_penalty != 'no_freq_penalty':
                valid_combinations.append((model_size, dataset_name, 1, 'no_similarity_penalty', frequency_penalty))

    return valid_combinations

# Parallel execution of all valid permutations
def run_all_experiments():
    # Get valid permutations
    valid_combinations = get_valid_permutations()

    print(f"Running {len(valid_combinations)} experiments...")
    
    # Run experiments in parallel using all available cores
    Parallel(n_jobs=-1)(
        delayed(run_experiment)(model_size, dataset_name, cut_proportion, similarity_penalty, frequency_penalty)
        for model_size, dataset_name, cut_proportion, similarity_penalty, frequency_penalty in valid_combinations
    )

if __name__ == '__main__':
    run_all_experiments()