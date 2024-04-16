import torch
import json
from tqdm import tqdm
from loading_helpers import load_json
from load import *


def compute_descriptor_list(data):
    '''
    Compute the list of all descriptors in the data.
    '''
    descriptor_list = []
    for v in data.values():
        descriptor_list.extend(v)
    return descriptor_list

def compute_class_list(data):
    '''
    Compute the list of all classes in the data.
    '''
    class_list = []
    for k in data.keys():
        class_list.append(k)
    return class_list

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

def compute_cosine_similarity(data):

    num_classes = len(dataset.classes)
    # Load descriptor files
    # Create prompt templates ("{class_label} which is {descriptor}")
    # Compute similarity between descriptor and every image embedding
    # Aggregate similarity score for each descriptor
    # Normalise the similarity scores
    normalised_similarity = []*num_classes

    # Initialize the environment
    seed_everything(hparams['seed'])

    # Prepare the data loader
    bs = hparams['batch_size']
    dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=16, pin_memory=True)  # Shuffle should be False for class-wise evaluation

    # Load the model and preprocessing
    print("Loading model...")
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    # Encode descriptions and labels
    print("Encoding descriptions...")
    description_encodings = compute_description_encodings(model)

    # Number of classes
    num_classes = len(dataset.classes)

    for batch_number, (images, labels) in enumerate(tqdm(dataloader)):    
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)

        # Compute description-based predictions
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes

    # In main.py, where image_description_similarity is computed
        for i, (k, v) in enumerate(description_encodings.items()):
            dot_product_matrix = image_encodings @ v.T

    return normalised_similarity

descriptor_file = [
    'descriptors/descriptors_cub.json',
    # 'descriptors/descriptors_dtd.json',
    # 'descriptors/descriptors_eurosat.json',
    # 'descriptors/descriptors_food101.json',
    # 'descriptors/descriptors_imagenet.json',
    # 'descriptors/descriptors_pets.json',
    # 'descriptors/descriptors_places365.json',
]

for json_path in descriptor_file:
    data = load_json(json_path)

    freq_is = compute_freq_is(data)
    freq_contains = compute_freq_contains(data)

    freq = {"freq_is": freq_is,
            "freq_contains": freq_contains}
    
    output_path_name = json_path.split("/")[-1].split(".")[0].split("_")[-1]
    json_output_path = f'descriptor_freq_analysis/descriptors_analysis_{output_path_name}.json'
    with open(json_output_path, 'w') as f:
        json.dump(freq, f, indent=4)

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df_is = pd.DataFrame([(k, v) for k, v in freq_is.items() if v > 6], columns=['Value', 'Frequency'])
# df_is = df_is.sort_values(by='Frequency', ascending=False)
# # df_contains = pd.DataFrame([(k, v) for k, v in freq_contains.items() if v > 15], columns=['Value', 'Frequency'])
# # df_contains = df_contains.sort_values(by='Frequency', ascending=False)

# plt.figure(figsize=(15, 6))  # Adjust the size of the plot as needed
# sns.barplot(x='Value', y='Frequency', data=df_is, palette='viridis')
# # sns.barplot(x='Value', y='Frequency', data=df_contains, palette='viridis')

# # Adding plot title and labels for clarity
# plt.title('Frequency Distribution of Descriptors')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# # Rotating the x-axis labels for better readability if necessary
# plt.xticks(rotation=60)

# # Show the plot
# plt.show()