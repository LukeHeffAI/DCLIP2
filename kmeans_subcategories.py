# kmeans_subcategories.py

import os
import json
import numpy as np
from sklearn.cluster import KMeans
import torch
import clip

# OPTIONAL: If you want to use an LLM to label subcategories:
from openai import OpenAI

def create_subcategories_with_kmeans(
    classes,
    device,
    model_size,
    out_json_path,
    out_desc_path,
    fraction_clusters=0.05,
    use_llm_for_cluster_names=True
):
    """
    1) Gets CLIP text embeddings for each class label.
    2) Runs k-means to find subcategories (clusters).
    3) (Optionally) uses an LLM to label each cluster.
    4) Saves a JSON file mapping subcategory_name -> [list_of_classes].
    5) Also saves subcategory descriptors (if you want them) to a second file.

    Args:
        classes (List[str]): The class names for your dataset.
        device (torch.device): CUDA or CPU device.
        model_size (str): e.g. 'ViT-B/32'.
        out_json_path (str): Path to save the subcategory->classes mapping JSON.
        out_desc_path (str): Path to save subcategory-level descriptors JSON.
        fraction_clusters (float): e.g. 0.05 means #clusters ~ 5% of #classes.
        use_llm_for_cluster_names (bool): if True, call an LLM to name each cluster.

    Returns:
        cluster_dict (dict): { 'cluster_label': [classA, classB, ...], ... }
    """

    # 1) Load CLIP and encode each class name as a text embedding
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_size, device=device, jit=False)
    model.eval()

    # Convert each class string into a token embedding
    all_embeddings = []
    for cname in classes:
        # Optionally do the same prefix as usual, e.g. "A photo of a {cname}"
        text = f"{cname}"
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2-normalize
        all_embeddings.append(emb[0].cpu().numpy())

    # 2) K-means over those text embeddings
    all_embeddings = np.stack(all_embeddings, axis=0)
    num_clusters = max(1, int(len(classes) * fraction_clusters))

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    assignments = kmeans.fit_predict(all_embeddings)

    # 3) Group classes by cluster ID
    cluster_map = {}
    for idx, class_name in enumerate(classes):
        cluster_id = assignments[idx]
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(class_name)

    # 4) (Optional) Use LLM to generate a short descriptive name per cluster
    #     or else just name them "subcat_0", "subcat_1", etc.
    cluster_dict = {}
    if use_llm_for_cluster_names:
        # Example using your existing OpenAI code
        openai_client = OpenAI()

        for cluster_id, c_list in cluster_map.items():
            prompt_text = (
                f"Provide a short descriptive subcategory name for "
                f"these classes: {c_list}. "
                "Respond with only the name, nothing else."
            )
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"user","content":prompt_text}],
                temperature=0.2,
                max_tokens=20
            )
            cluster_name = response.choices[0].message.content.strip()

            # Just in case the LLM returns something messy
            if not cluster_name:
                cluster_name = f"subcategory_{cluster_id}"
            # Clean or shorten cluster_name if needed
            cluster_dict[cluster_name] = c_list
    else:
        # No LLM; just label them "subcat_0", "subcat_1", ...
        for cluster_id, c_list in cluster_map.items():
            cluster_name = f"subcategory_{cluster_id}"
            cluster_dict[cluster_name] = c_list

    # 5) Save subcategory->classes mapping
    with open(out_json_path, 'w') as f:
        json.dump(cluster_dict, f, indent=2)

    # 6) If you want subcategory-level descriptors (like the old "class_analysis_xxx_descriptors.json"),
    #    you can create them yourself or leave them blank. For example:
    subcat_desc = {}
    for subcat_name in cluster_dict:
        # A trivial placeholder descriptor, or you can generate them with an LLM
        subcat_desc[subcat_name] = [
            "cluster descriptor text one",
            "cluster descriptor text two"
        ]
    with open(out_desc_path, 'w') as f:
        json.dump(subcat_desc, f, indent=2)

    return cluster_dict
