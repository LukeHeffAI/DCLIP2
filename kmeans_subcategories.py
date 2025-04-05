import os
import json
import numpy as np
from sklearn.cluster import KMeans
import torch
import clip
from openai import OpenAI
from collections import defaultdict

def compute_class_list(data:dict, sort_config = False):
    if sort_config:
        data = dict(sorted(data.items()))

    class_list = []
    for k in data.keys():
        class_list.append(k)

    if sort_config:
        class_list = sorted(class_list)

    return class_list

def get_dataset_terms(dataset_name):
    """
    Returns dataset-specific terminology for cluster naming.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'ImageNet', 'CUB-200', 'Food101')
        
    Returns:
        dict: Dictionary containing dataset-specific terms
    """
    # Default generic terms
    terms = {
        "class_type": "item",
        "group_prefix": "Group",
        "misc_group_name": "Miscellaneous Items"
    }
    
    # Dataset-specific overrides
    dataset_name = dataset_name.lower()
    
    if any(bird_term in dataset_name for bird_term in ["cub", "bird"]):
        terms = {
            "class_type": "bird",
            "group_prefix": "Bird Group",
            "misc_group_name": "Miscellaneous Birds"
        }
    elif any(food_term in dataset_name for food_term in ["food", "cuisine"]):
        terms = {
            "class_type": "food",
            "group_prefix": "Food Category",
            "misc_group_name": "Miscellaneous Food Items"
        }
    elif "imagenet" in dataset_name:
        terms = {
            "class_type": "object",
            "group_prefix": "Object Category",
            "misc_group_name": "Miscellaneous Objects"
        }
    elif "place" in dataset_name:
        terms = {
            "class_type": "place",
            "group_prefix": "Location Type",
            "misc_group_name": "Miscellaneous Places"
        }
    elif "aircraft" in dataset_name:
        terms = {
            "class_type": "aircraft",
            "group_prefix": "Aircraft Type",
            "misc_group_name": "Miscellaneous Aircraft"
        }
    elif "car" in dataset_name:
        terms = {
            "class_type": "car",
            "group_prefix": "Vehicle Category",
            "misc_group_name": "Miscellaneous Vehicles"
        }
    elif "flower" in dataset_name:
        terms = {
            "class_type": "flower",
            "group_prefix": "Floral Group",
            "misc_group_name": "Miscellaneous Flowers"
        }
    elif "dtd" in dataset_name or "texture" in dataset_name:
        terms = {
            "class_type": "texture",
            "group_prefix": "Texture Type",
            "misc_group_name": "Miscellaneous Textures"
        }
    elif "pet" in dataset_name:
        terms = {
            "class_type": "pet",
            "group_prefix": "Pet Category",
            "misc_group_name": "Miscellaneous Pets"
        }
    elif "eurosat" in dataset_name or "satellite" in dataset_name:
        terms = {
            "class_type": "land use",
            "group_prefix": "Land Category",
            "misc_group_name": "Miscellaneous Land Types"
        }
    elif "cifar10" in dataset_name:
        terms = {
            "class_type": "object",
            "group_prefix": "Object Group",
            "misc_group_name": "Miscellaneous Objects"
        }
    elif "cifar100" in dataset_name:
        terms = {
            "class_type": "object",
            "group_prefix": "Object Category",
            "misc_group_name": "Miscellaneous Objects"
        }
    elif "sun397" in dataset_name:
        terms = {
            "class_type": "scene",
            "group_prefix": "Scene Type",
            "misc_group_name": "Miscellaneous Scenes"
        }
    elif "caltech101" in dataset_name:
        terms = {
            "class_type": "object",
            "group_prefix": "Object Category",
            "misc_group_name": "Miscellaneous Objects"
        }
    
    return terms

def create_subcategories_with_kmeans(
    hparams,
    device,
    model_size,
    out_json_path,
    out_desc_path,
    fraction_clusters=0.15,  # Increased from 0.05 to get more clusters
    use_llm_for_cluster_names=True
):
    """
    1) Gets CLIP text embeddings for each class label.
    2) Runs k-means to find subcategories (clusters).
    3) (Optionally) uses an LLM to label each cluster.
    4) Saves a JSON file mapping subcategory_name -> [list_of_classes].
    5) Also saves subcategory descriptors (if you want them) to a second file.

    Args:
        hparams (dict): Hyperparameters including dataset name.
        device (torch.device): CUDA or CPU device.
        model_size (str): e.g. 'ViT-B/32'.
        out_json_path (str): Path to save the subcategory->classes mapping JSON.
        out_desc_path (str): Path to save subcategory-level descriptors JSON.
        fraction_clusters (float): e.g. 0.05 means #clusters ~ 5% of #classes.
        use_llm_for_cluster_names (bool): if True, call an LLM to name each cluster.

    Returns:
        cluster_dict (dict): { 'cluster_label': [classA, classB, ...], ... }
    """
    # Get dataset-specific terminology
    dataset_terms = get_dataset_terms(hparams['dataset_name'])
    
    # Load the descriptors file
    filename = f'descriptors/gpt-3/descriptors_{hparams["dataset"]}.json'
    with open(filename, 'r') as f:
        data = json.load(f)

    # Get all classes from the descriptors file
    all_classes = list(data.keys())
    print(f"Total classes found: {len(all_classes)}")
    
    # 1) Load CLIP and encode each class name as a text embedding
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_size, device=device, jit=False)
    model.eval()

    # Convert each class string into a token embedding
    all_embeddings = []
    classes = []  # To keep track of classes in same order as embeddings
    
    for cname in all_classes:
        # Optionally do the same prefix as usual, e.g. "A photo of a {cname}"
        text = f"{cname}"
        tokens = clip.tokenize([text]).to(device)
        try:
            with torch.no_grad():
                emb = model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2-normalize
            all_embeddings.append(emb[0].cpu().numpy())
            classes.append(cname)
        except Exception as e:
            print(f"Error processing {cname}: {e}")

    # 2) K-means over those text embeddings
    all_embeddings = np.stack(all_embeddings, axis=0)
    num_clusters = max(5, int(len(classes) * fraction_clusters))
    print(f"Creating {num_clusters} clusters")

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    assignments = kmeans.fit_predict(all_embeddings)

    # 3) Group classes by cluster ID
    cluster_map = defaultdict(list)
    for idx, class_name in enumerate(classes):
        cluster_id = assignments[idx]
        cluster_map[cluster_id].append(class_name)

    # 4) Use LLM to generate descriptive names per cluster
    cluster_dict = {}
    if use_llm_for_cluster_names:
        openai_client = OpenAI()

        # Get the class list for the prompt
        class_list = ', '.join(all_classes)

        for cluster_id, c_list in cluster_map.items():
            prompt_text = (
                f"You're helping categorize a dataset of {dataset_terms['class_type']}s. "
                f"Provide a short descriptive subcategory name for this group of {dataset_terms['class_type']}s: {c_list}. "
                f"The name should be specific to this subset and differentiate it from the broader set of {dataset_terms['class_type']}s within the {hparams['dataset_name']} dataset. "
                "Respond with only the category name, nothing else."
            )
            # For the first iteration, print the prompt text
            if cluster_id == 0:
                print(f"Prompt text for LLM: {prompt_text}")
                
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
                max_tokens=20
            )
            
            if response.choices[0].message.content is not None: cluster_name = response.choices[0].message.content.strip()

            # Just in case the LLM returns something messy
            if not cluster_name:
                cluster_name = f"{dataset_terms['group_prefix']} {cluster_id}"
                
            # Handle duplicate names by adding numbers
            if cluster_name in cluster_dict:
                base_name = cluster_name
                i = 1
                while f"{base_name} (Group {i})" in cluster_dict:
                    i += 1
                cluster_name = f"{base_name} (Group {i})"
                
            cluster_dict[cluster_name] = c_list
    else:
        # No LLM; use dataset-specific generic naming
        for cluster_id, c_list in cluster_map.items():
            cluster_name = f"{dataset_terms['group_prefix']} {cluster_id+1}"
            cluster_dict[cluster_name] = c_list

    # Verify all classes are included
    all_classes_in_clusters = [c for classes in cluster_dict.values() for c in classes]
    missing_classes = set(all_classes) - set(all_classes_in_clusters)
    
    if missing_classes:
        print(f"Warning: {len(missing_classes)} {dataset_terms['class_type']}s are missing from clusters: {missing_classes}")
        # Add missing classes to a miscellaneous category
        if dataset_terms['misc_group_name'] not in cluster_dict:
            cluster_dict[dataset_terms['misc_group_name']] = list(missing_classes)
        else:
            cluster_dict[dataset_terms['misc_group_name']].extend(list(missing_classes))

    # 5) Save subcategory->classes mapping
    with open(out_json_path, 'w') as f:
        json.dump(cluster_dict, f, indent=2)
        
    print(f"Saved clusters to {out_json_path}")
    print(f"Created {len(cluster_dict)} named clusters")

    return cluster_dict