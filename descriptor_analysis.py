import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip
from load import hparams, dataset
from loading_helpers import compute_descriptor_list, load_json
from collections import Counter
from torch.utils.data import DataLoader, Subset
import json


def compute_freq_exact(data):
    """
    Compute a normalised frequency score for each descriptor based on its relative frequency
    in the dataset. The frequency is normalised by dividing by the maximum frequency.

    Args:
        data: The input dataset from which descriptors are extracted.

    Returns:
        A dictionary where keys are descriptors and values are normalised frequency scores (floats).
    """
    descriptor_list = compute_descriptor_list(data, sort_config=True)
    
    # Calculate the frequency of each descriptor
    descriptor_frequencies = Counter(descriptor_list)
    
    # Get the maximum frequency to normalise
    max_freq = max(descriptor_frequencies.values())
    
    # Normalise the frequency scores
    descriptor_normalised_frequencies = {desc: freq / max_freq for desc, freq in descriptor_frequencies.items()}
    
    return descriptor_normalised_frequencies

def compute_freq_approx(data):
    """
    Compute a normalised frequency score for each descriptor based on both exact matches and
    occurrences where the descriptor is a substring of another descriptor.
    
    Args:
        data: The input dataset from which descriptors are extracted.

    Returns:
        A dictionary where keys are descriptors and values are normalised frequency scores (floats).
    """
    # Get the list of descriptors from the data
    descriptor_list = compute_descriptor_list(data, sort_config=True)

    # Initialise the frequency dictionary
    descriptor_frequencies = {}

    # First, count exact matches
    for descriptor in descriptor_list:
        if descriptor in descriptor_frequencies:
            descriptor_frequencies[descriptor] += 1
        else:
            descriptor_frequencies[descriptor] = 1

    # Now, count occurrences where a descriptor is a substring of another descriptor
    for descriptor in set(descriptor_list):
        for potential_container in descriptor_list:
            if descriptor in potential_container and descriptor != potential_container:
                descriptor_frequencies[descriptor] += 1

    # Get the maximum frequency for normalisation
    max_freq = max(descriptor_frequencies.values())

    # Normalise the frequencies
    descriptor_normalised_frequencies = {desc: freq / max_freq for desc, freq in descriptor_frequencies.items()}

    return descriptor_normalised_frequencies

def load_clip():

    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    return model, device

def tokenise_descriptor(descriptor, model):
    """
    Tokenise and encode a single descriptor using the provided CLIP model.
    
    Args:
        descriptor (str): A textual description or descriptor of an image.
        model (clip.model): An instance of a CLIP model preloaded with a tokeniser and text encoder.
    
    Returns:
        torch.Tensor: The normalised encoding of the descriptor.
    """
    tokens = clip.tokenize([descriptor]).to(hparams['device'])
    with torch.no_grad():
        encoded_text = model.encode_text(tokens)
        normalised_text_encoding = torch.nn.functional.normalize(encoded_text, dim=1)
    return normalised_text_encoding

def compute_descriptor_sim_uniqueness_score(data):
    """
    Compute the uniqueness score of each descriptor based on the sum of the normalised cosine similarities between the descriptor and all other descriptors.
    This optimised version encodes all descriptors in a batch to take advantage of parallel GPU operations.
    """
    model, device = load_clip()
    
    descriptor_list = compute_descriptor_list(data, sort_config=True)

    descriptor_tensors = []
    for desc in tqdm(descriptor_list, desc="Tokenising Descriptors"):
        desc_tensor = tokenise_descriptor(desc, model)
        descriptor_tensors.append(desc_tensor)

    # Stack the descriptor tensors into a single tensor for parallel computation
    descriptor_tensors = torch.cat(descriptor_tensors, dim=0).to(device)

    with torch.no_grad():
        similarity_matrix = descriptor_tensors @ descriptor_tensors.T
        similarity_matrix.fill_diagonal_(0)  # Exclude self-similarity by setting the diagonal to 0

    # Sum similarities for each descriptor
    descriptor_sums = similarity_matrix.sum(dim=1).cpu().numpy()

    # Normalise the sums
    max_sum = max(descriptor_sums)
    descriptor_normalised_sums = {desc: float(descriptor_sums[i] / max_sum) for i, desc in enumerate(descriptor_list)}

    return descriptor_normalised_sums

def compute_text_image_cosine_similarity(data):
    """
    Compute the cosine similarity between each descriptor and all images,
    normalise these values, and save the results in JSON format.
    """
    
    model, device = load_clip()

    descriptor_list = compute_descriptor_list(data, sort_config=True)
    dataloader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=16, pin_memory=True)

    descriptor_sums = {desc: 0.0 for desc in descriptor_list}
    for desc in tqdm(descriptor_list, desc="Processing Descriptors"):
        desc_tensor = tokenise_descriptor(desc, model)
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Computing Similarities", leave=False):
                images = images.to(device)
                image_encodings = model.encode_image(images)
                image_encodings = F.normalize(image_encodings, dim=1)

                sim = (desc_tensor @ image_encodings.T).squeeze(0)
                descriptor_sums[desc] += sim.sum().item()  # Sum similarities for this batch

    # Normalise the sums
    max_sum = max(descriptor_sums.values())
    descriptor_normalised_sums = {k: v / max_sum for k, v in descriptor_sums.items()}

    return descriptor_normalised_sums

descriptor_file_path = hparams['descriptor_fname']

# for json_path in descriptor_file:
analysis = load_json(descriptor_file_path)

freq_exact = compute_freq_exact(analysis)
freq_contains = compute_freq_approx(analysis)
descriptor_self_similarity = compute_descriptor_sim_uniqueness_score(analysis)
# text_image_similarity = compute_text_image_cosine_similarity(analysis)

analysis_dict = {"freq_exact": freq_exact,
        "freq_approx": freq_contains,
        "descriptor-self-similarity": descriptor_self_similarity,
        # "text-image-similarity": text_image_similarity,
        }
    
# print(analysis_dict['freq_exact'])
output_path_name = descriptor_file_path.split("/")[-1].split(".")[0].split("_")[-1]
json_output_path = f'descriptor_analysis/descriptors_analysis_{output_path_name}.json'

print(f'Saving analysis to {json_output_path}')
with open(json_output_path, 'w') as f:
    json.dump(analysis_dict, f, indent=4)