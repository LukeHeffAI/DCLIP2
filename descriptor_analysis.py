import torch
import json
import time as t
from loading_helpers import load_json

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
    freq = {"freq_is": freq_is, "freq_contains": freq_contains}
    output_path_name = json_path.split("/")[-1].split(".")[0].split("_")[-1]
    json_output_path = f'descriptor_freq_analysis/descriptors_freq_{output_path_name}.json'
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