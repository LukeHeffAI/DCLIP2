import torch
import json
from tqdm import tqdm
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
    '''
    Compute the cosine similarity between each descriptor and all images,
    normalize these values, and save the results in JSON format.
    '''

    # Load descriptor files
    # Create prompt templates ("{class_label} which is {descriptor}")
    # Compute similarity between descriptor and every image embedding
    # Aggregate similarity score for each descriptor
    # Normalise the similarity scores

    # Initialize the environment
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()

    # Load descriptors and encode them
    descriptor_list = compute_descriptor_list(data)
    descriptor_encodings = compute_description_encodings(model)
    
    # Prepare data loader for images
    dataloader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Compute similarities
    descriptor_sums = {desc: 0 for desc in descriptor_list}
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Processing Images"):
            images = images.to(device)
            image_encodings = model.encode_image(images)
            image_encodings = F.normalize(image_encodings, dim=1)

            for desc, desc_encoding in descriptor_encodings.items():
                desc_encoding = desc_encoding.unsqueeze(0)  # Add batch dimension
                sim = (desc_encoding @ image_encodings.T).squeeze(0)  # Cosine similarity
                descriptor_sums[desc] += sim.sum().item()  # Sum similarities for this batch

    # Normalize the sums
    max_sum = max(descriptor_sums.values())
    descriptor_normalized_sums = {k: v / max_sum for k, v in descriptor_sums.items()}

    # Save the results
    save_path = 'descriptor_cosine_similarity.json'
    with open(save_path, 'w') as f:
        json.dump(descriptor_normalized_sums, f, indent=4)

    return descriptor_normalized_sums



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