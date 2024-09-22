import torch
import torch.nn.functional as F
from tqdm import tqdm
from load import *


def compute_freq_is(data):
    '''
    Compute the frequency of each value in the data when the value completely matches the key.    
    '''

    descriptor_list = compute_descriptor_list(data)
    descriptor_frequencies = {i: 0 for i in descriptor_list}
    for entry in descriptor_list:
        if entry in descriptor_frequencies:
            descriptor_frequencies[entry] += 1

    return descriptor_frequencies

def compute_freq_contains(data):
    '''
    Compute the frequency of each item in the value list in the data, including when the item is a substring of another item.
    '''
    descriptor_list = compute_descriptor_list(data)
    descriptor_frequencies = {}
    for descriptor in descriptor_list:
        if descriptor in descriptor_frequencies:
            descriptor_frequencies[descriptor] += 1
        else:
            descriptor_frequencies[descriptor] = 1
    for descriptor in set(descriptor_list):
        for potential_container in descriptor_list:
            if descriptor in potential_container and descriptor != potential_container:
                descriptor_frequencies[descriptor] += 1

    return descriptor_frequencies

def load_clip():

    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    return model, device

def tokenise_descriptor(descriptor, model):
    """
    Tokenize and encode a single descriptor using the provided CLIP model.
    
    Args:
        descriptor (str): A textual description or descriptor of an image.
        model (clip.model): An instance of a CLIP model preloaded with a tokeniser and text encoder.
    
    Returns:
        torch.Tensor: The normalized encoding of the descriptor.
    """
    tokens = clip.tokenize([descriptor]).to(hparams['device'])
    with torch.no_grad():
        encoded_text = model.encode_text(tokens)
        normalised_text_encoding = torch.nn.functional.normalize(encoded_text, dim=1)
    return normalised_text_encoding

def compute_descriptor_uniqueness_score_from_sim(data):
    """
    Compute the uniqueness score of each descriptor based on the sum of the normalised cosine similarities between the descriptor and all other descriptors.
    This optimized version encodes all descriptors in a batch to take advantage of parallel GPU operations.
    """
    model, device = load_clip()
    
    descriptor_list = compute_descriptor_list(data, sort_config=True)

    descriptor_tensors = []
    for desc in tqdm(descriptor_list, desc="Tokenizing Descriptors"):
        desc_tensor = tokenise_descriptor(desc, model)
        descriptor_tensors.append(desc_tensor)

    # Stack the descriptor tensors into a single tensor for parallel computation
    descriptor_tensors = torch.cat(descriptor_tensors, dim=0).to(device)

    with torch.no_grad():
        similarity_matrix = descriptor_tensors @ descriptor_tensors.T
        similarity_matrix.fill_diagonal_(0)  # Exclude self-similarity by setting the diagonal to 0

    # Sum similarities for each descriptor
    descriptor_sums = similarity_matrix.sum(dim=1).cpu().numpy()

    # Normalize the sums
    max_sum = max(descriptor_sums)
    descriptor_normalised_sums = {desc: float(descriptor_sums[i] / max_sum) for i, desc in enumerate(descriptor_list)}

    return descriptor_normalised_sums

def compute_text_image_cosine_similarity(data):
    """
    Compute the cosine similarity between each descriptor and all images,
    normalize these values, and save the results in JSON format.
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

    # Normalize the sums
    max_sum = max(descriptor_sums.values())
    descriptor_normalised_sums = {k: v / max_sum for k, v in descriptor_sums.items()}

    return descriptor_normalised_sums

descriptor_file_path = hparams['descriptor_fname']

# for json_path in descriptor_file:
analysis = load_json(descriptor_file_path)

freq_is = compute_freq_is(analysis)
freq_contains = compute_freq_contains(analysis)
descriptor_self_similarity = compute_descriptor_uniqueness_score_from_sim(analysis)
# text_image_similarity = compute_text_image_cosine_similarity(analysis)

analysis_dict = {"freq_is": freq_is,
        "freq_contains": freq_contains,
        "descriptor-self-similarity": descriptor_self_similarity,
        # "text-image-similarity": text_image_similarity,
        }
    
# print(analysis_dict['descriptor-self-similarity'])
output_path_name = descriptor_file_path.split("/")[-1].split(".")[0].split("_")[-1]
json_output_path = f'descriptor_analysis/descriptors_analysis_{output_path_name}.json'

print(f'Saving analysis to {json_output_path}')
with open(json_output_path, 'w') as f:
    json.dump(analysis_dict, f, indent=4)