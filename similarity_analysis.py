## TODO: Combine this with the descriptor_analysis.py in a Jupyter notebook
## TODO: Update function to allow flexible similarity checking with analysis_type, regarding the cosine_similarity and what it compares

import torch
import json
import csv
from torch.nn import functional as F
from load import hparams, clip, seed_everything, load_json, compute_class_list, compute_descriptor_list, compute_description_encodings, compute_label_encodings

def model_setup():
    """
    Load the CLIP model and set up the device.
    """
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    return model, preprocess

def create_text_data(descriptor_file_path):
    """
    Load the class descriptions from the JSON file.
    """
    analysis_dict = load_json(descriptor_file_path)

    class_list = compute_class_list(analysis_dict, sort_config=True)
    descriptor_list = compute_descriptor_list(analysis_dict, sort_config=True)

    return class_list, descriptor_list

def encode_text_data(model, class_list:list, descriptor_list: list):
    """
    Encode the class descriptions and labels, as well as the descriptor prompts as a full sentence.
    """
    # TODO: Update how these functions work to allow for more flexible encoding
    # compute_label_encodings can't take descriptors as input

    class_encodings = F.normalize(model.encode_text(clip.tokenize(class_list).to(hparams['device'])))
    descriptor_encodings = F.normalize(model.encode_text(clip.tokenize(descriptor_list).to(hparams['device'])))

    return class_encodings, descriptor_encodings

def compute_cosine_similarity(encodings_1, encodings_2, analysis_object):

    if analysis_object == 'class_self_similarity':
        print('Computing class self-similarity...')
        cosine_similarity = torch.mm(encodings_1, encodings_1.T)
    elif analysis_object == 'descriptor_self_similarity':
        print('Computing descriptor self-similarity...')
        cosine_similarity = torch.mm(encodings_2, encodings_2.T)
    else:
        print('Computing cosine similarity between classes and descriptors...')
        cosine_similarity = torch.mm(encodings_1, encodings_2.T)

    return cosine_similarity

def structure_similarity_data(cosine_similarity, analysis_subject):
    """
    Compute the cosine similarity between each class and all images,
    normalise these values, and save the results in JSON format.
    """

    cosine_similarity_cpu = cosine_similarity.cpu().detach().numpy()

    if analysis_subject == 'class':
        key_list = class_list
    elif analysis_subject == 'descriptor':
        key_list = descriptor_list

    similarity = {}
    for i, key_name in enumerate(key_list):
        similarity[key_name] = {}
        similarity[key_name]["cosine_similarity_vector"] = cosine_similarity_cpu[i].tolist()
        similarity[key_name]["cosine_similarity_vector_sorted"] = sorted(cosine_similarity_cpu[i].tolist(), reverse=True)
        similarity[key_name]["average_cosine_similarity"] = cosine_similarity.mean(dim=0).tolist()[i]

    return similarity

analysis_subject = 'class'     # 'class' or 'descriptor
analysis_type = 'self_similarity'   # 'self_similarity' or 'cosine_similarity'
analysis_object = f'{analysis_subject}_{analysis_type}'
dataset_name = hparams['dataset']
descriptor_file_path = hparams['descriptor_fname']
output_path_name = f'{analysis_object}_{dataset_name}'

# Load the model and preprocessing
print("Loading model...")
model, preprocess = model_setup()
seed_everything(hparams['seed'])

# Load the class labels and descriptions
class_list, descriptor_list = create_text_data(descriptor_file_path)

# # Create the label and descriptor combined prompts
# wordify etc etc

# Encode descriptions and labels
print("Encoding descriptions...")
class_encodings, descriptor_encodings = encode_text_data(model, class_list, descriptor_list)

# Compute cosine similarity
cosine_similarity = compute_cosine_similarity(class_encodings, descriptor_encodings, analysis_object)

similarity_dict = structure_similarity_data(cosine_similarity, analysis_subject)

print('Saving results to .json and .xlsx formats...')
# Print results to .json, with the class names as the keys and the similarity to the descriptors as the values
json_output_path = f'{analysis_subject}_analysis/json/{output_path_name}.json'
with open(json_output_path, 'w') as f:
    json.dump(similarity_dict, f, indent=4)

# Print the same information to .csv, with the class names as the columns and the descriptors as the rows
csv_output_path = f'{analysis_subject}_analysis/xlsx/{output_path_name}.xlsx'
with open(csv_output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    header = ['Description'] + [class_name for class_name in class_list]
    writer.writerow(header)

    reference_list = descriptor_list if analysis_subject == 'descriptor' else class_list
    for index, reference in enumerate(reference_list):
        row = [reference] + [similarity_dict[name]["cosine_similarity_vector"][index] for name in reference_list]
        writer.writerow(row)