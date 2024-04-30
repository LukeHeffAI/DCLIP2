import torch
import json
from tqdm import tqdm
from load import *

def compute_class_description_cosine_similarity(data):
    """
    Compute the cosine similarity between each class and all images,
    normalize these values, and save the results in JSON format.
    """
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()

    class_list = compute_class_list(data, sort_config=True)

    class_descriptor_dict = load_json(descriptor_file_path)
    class_list = compute_class_list(class_descriptor_dict, sort_config=True)
    descriptor_list = compute_descriptor_list(class_descriptor_dict, sort_config=True)

    class_list = [c.replace('-', ' ') for c in class_list]

    seed_everything(hparams['seed'])

    # Load the model and preprocessing
    print("Loading model...")
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    # Encode descriptions and labels
    print("Encoding descriptions...")
    description_encodings = F.normalize(model.encode_text(clip.tokenize(descriptor_list).to(device)))
    label_encodings = F.normalize(model.encode_text(clip.tokenize(class_list).to(device)))

    cosine_similarity = torch.mm(description_encodings, label_encodings.T)

    # Calculate average cosine similarity for each class
    average_cosine_similarity = cosine_similarity.mean(dim=0).tolist()

    return cosine_similarity, average_cosine_similarity

descriptor_file_path = hparams['descriptor_fname']
results_file_path = 'results/experiment_results.json'

class_descriptor_dict = load_json(descriptor_file_path)
cosine_similarity = compute_class_description_cosine_similarity(class_descriptor_dict)

print(cosine_similarity)