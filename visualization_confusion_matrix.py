import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from load import compute_description_encodings
from loading_helpers import load_gpt_descriptions, seed_everything, compute_class_list
import clip
import argparse
from torch.nn import functional as F
import pandas as pd
from load import set_hparams
import os
from tqdm import tqdm

def compute_text_method_confusion_matrix(model, method1, method2, dataset_classes, hparams, aggregation='mean'):
    """
    Compute a confusion matrix between two text prompting methods.
    
    Args:
        model: CLIP model
        method1: First text prompting method
        method2: Second text prompting method
        dataset_classes: List of class names
        hparams: Hyperparameters
        aggregation: How to aggregate similarities ('mean', 'max', or 'first')
        
    Returns:
        confusion_matrix: A matrix showing the similarity between text encodings
                         from the two methods
    """
    # Store original method
    original_method = hparams['method']
    device = hparams['device']
    
    # Set method to method1 and encode descriptions
    print(f"Encoding descriptions for method: {method1}")
    hparams['method'] = method1
    gpt_descriptions1, _ = load_gpt_descriptions(hparams)
    desc_encodings1 = compute_description_encodings(model, gpt_descriptions1, hparams)
    
    # Set method to method2 and encode descriptions
    print(f"Encoding descriptions for method: {method2}")
    hparams['method'] = method2
    gpt_descriptions2, _ = load_gpt_descriptions(hparams)
    desc_encodings2 = compute_description_encodings(model, gpt_descriptions2, hparams)
    
    # Restore original method
    hparams['method'] = original_method
    
    # Prepare array for confusion matrix
    n_classes = len(dataset_classes)
    confusion_matrix = torch.zeros((n_classes, n_classes), device=device)
    
    print("Computing confusion matrix...")
    for i, (class1, encodings1) in enumerate(tqdm(desc_encodings1.items())):
        for j, (class2, encodings2) in enumerate(desc_encodings2.items()):
            # Compute all pairwise similarities
            sim_matrix = encodings1 @ encodings2.T
            
            # Aggregate similarities based on method
            if aggregation == 'mean':
                sim = sim_matrix.mean().item()
            elif aggregation == 'max':
                sim = sim_matrix.max().item()
            elif aggregation == 'first':
                sim = sim_matrix[0, 0].item()
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
                
            confusion_matrix[i, j] = sim
            
    return confusion_matrix, dataset_classes

def visualize_confusion_matrix(confusion_matrix, class_names, method1, method2, save_path=None, 
                              normalize=True, figsize=(16, 14), title=None, annot=True):
    """
    Visualize the confusion matrix between two text prompting methods.
    
    Args:
        confusion_matrix: Confusion matrix tensor
        class_names: List of class names
        method1: Name of the first method (y-axis)
        method2: Name of the second method (x-axis)
        save_path: Path to save the visualization (optional)
        normalize: Whether to normalize the confusion matrix by row
        figsize: Figure size
        title: Custom title (optional)
        annot: Whether to annotate cells with values
    """
    # Convert to numpy for plotting
    conf_matrix = confusion_matrix.cpu().numpy()
    
    # Normalize if requested
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = conf_matrix / (row_sums + 1e-8)  # Avoid division by zero
    
    # Create DataFrame
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    
    # Set up plot
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(df_cm, annot=annot, cmap="viridis", fmt=".2f" if normalize else ".4f",
               xticklabels=True, yticklabels=True)
    
    # Set title and labels
    if title is None:
        title = f'Text Prompt Confusion Matrix: {method1} vs {method2}'
    plt.title(title, fontsize=16)
    plt.xlabel(f'Classes encoded with {method2}', fontsize=14)
    plt.ylabel(f'Classes encoded with {method1}', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Adjust layout to ensure all elements are visible
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    
    return plt

def generate_confusion_matrix_comparison(model_size='ViT-B/32', 
                                        desc_type='gpt-3',
                                        dataset='eurosat',
                                        method1='clip',
                                        method2='defntaxs',
                                        save_path=None,
                                        aggregation='mean',
                                        normalize=True):
    """
    Generate and visualize a confusion matrix comparing two text prompting methods.
    
    Args:
        model_size: CLIP model size
        desc_type: Type of descriptions to use
        dataset: Dataset to use
        method1: First method to compare
        method2: Second method to compare
        save_path: Path to save the visualization
        aggregation: How to aggregate similarities ('mean', 'max', or 'first')
        normalize: Whether to normalize the confusion matrix by row
    """
    # Set up hyperparameters
    hparams, _, _, _, _, _, _, _, _  = set_hparams(model_size=model_size, desc_type=desc_type, dataset=dataset, method=method1)  # Start with method1
    
    with open(f'{hparams['descriptor_fname']}.json', 'r') as f:
        class_descriptors = json.load(f)
    class_list = compute_class_list(class_descriptors)

    # Ensure reproducibility
    seed_everything(hparams['seed'])
    
    # Load the model
    print(f"Loading CLIP model: {model_size}")
    device = torch.device(hparams['device'])
    model, _ = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)
    
    # Compute the confusion matrix
    confusion_matrix, class_names = compute_text_method_confusion_matrix(
        model, method1, method2, class_list, hparams, aggregation)
    
    # Visualize the confusion matrix
    plt = visualize_confusion_matrix(
        confusion_matrix, class_names, method1, method2, save_path, normalize)
    
    plt.show()
    
    return confusion_matrix, class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confusion matrix between two text prompting methods")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model size")
    parser.add_argument("--desc_type", type=str, default="gpt-3", help="Description type")
    parser.add_argument("--dataset", type=str, default="eurosat", help="Dataset name")
    parser.add_argument("--method1", type=str, default="clip", help="First text prompting method")
    parser.add_argument("--method2", type=str, default="defntaxs", help="Second text prompting method")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the visualization")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max", "first"],
                       help="How to aggregate similarities")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the confusion matrix")
    
    args = parser.parse_args()
    
    generate_confusion_matrix_comparison(
        model_size=args.model,
        desc_type=args.desc_type,
        dataset=args.dataset,
        method1=args.method1,
        method2=args.method2,
        save_path=args.save_path,
        aggregation=args.aggregation,
        normalize=args.normalize
    )
