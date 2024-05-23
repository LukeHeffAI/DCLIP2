import torch
import json
import csv
from tqdm import tqdm
from load import *

def compute_class_description_cosine_similarity(descriptor_file_path):
    """
    Compute the cosine similarity between each class and all images,
    normalize these values, and save the results in JSON format.
    """
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()

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
    class_encodings = F.normalize(model.encode_text(clip.tokenize(class_list).to(device)))

    cosine_similarity = torch.mm(class_encodings, description_encodings.T)

    # Calculate average cosine similarity for each class
    average_cosine_similarity = cosine_similarity.mean(dim=0).tolist()

    cosine_similarity_cpu = cosine_similarity.cpu().detach().numpy()

    similarity = {}
    for i, class_name in enumerate(class_list):
        similarity[class_name] = {}
        similarity[class_name]["cosine_similarity_vector"] = cosine_similarity_cpu[i].tolist()
        similarity[class_name]["cosine_similarity_vector_sorted"] = sorted(cosine_similarity_cpu[i].tolist(), reverse=True)
        similarity[class_name]["average_cosine_similarity"] = average_cosine_similarity[i]

    return similarity, class_list, descriptor_list

# descriptor_file_path = hparams['descriptor_fname']

# class_descriptor_dict = load_json(descriptor_file_path)
# analysis_dict, class_list, descriptor_list = compute_class_description_cosine_similarity(class_descriptor_dict)

# output_path_name = descriptor_file_path.split("/")[-1].split(".")[0].split("_")[-1]
# json_output_path = f'class_analysis/class_analysis_{output_path_name}.json'
# with open(json_output_path, 'w') as f:
#     json.dump(analysis_dict, f, indent=4)

# # Print the same information to .csv, with the class names as the columns and the descriptors as the rows
# csv_output_path = f'class_analysis/class_analysis_{output_path_name}_test.xlsx'
# with open(csv_output_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
#     header = ['Description'] + [class_name for class_name in class_list]
#     writer.writerow(header)
#     for index, descriptor in enumerate(descriptor_list):
#         row = [descriptor] + [analysis_dict[class_name]["cosine_similarity_vector"][index] for class_name in class_list]
#         writer.writerow(row)