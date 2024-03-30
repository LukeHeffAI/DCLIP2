import torch
import json
import time as t
import seaborn as sns

def compute_frequencies_is(data):
    '''
    Compute the frequency of each value in the data when the value completely matches the key.    
    '''
    class_list = []
    descriptor_list = []
    descriptor_frequencies = {}
    for k, v in data.items():
        class_list.append(k)
        descriptor_list.extend(v)
    descriptor_frequencies = {i: 0 for i in descriptor_list}
    for entry in descriptor_list:
        if entry in descriptor_frequencies:
            descriptor_frequencies[entry] += 1
    return descriptor_frequencies

def compute_frequencies_contains(data):
    '''
    Compute the frequency of each item in the value list in the data, including when the item is a substring of another item.
    '''




def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

data = load_json('descriptors/descriptors_cub.json')
frq = compute_frequencies_is(data)
for k, v in frq.items():
    if v>3:
        print(v, "|", k)



# descriptor_file = [
#     'descriptors/descriptors_cub.json',
#     'descriptors/descriptors_dtd.json',
#     'descriptors/descriptors_eurosat.json',
#     'descriptors/descriptors_food101.json',
#     'descriptors/descriptors_imagenet.json',
#     'descriptors/descriptors_pets.json',
#     'descriptors/descriptors_places365.json',
# ]
    

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame([(k, v) for k, v in frq.items() if v > 4], columns=['Value', 'Frequency'])
df = df.sort_values(by='Frequency', ascending=False)

plt.figure(figsize=(15, 6))  # Adjust the size of the plot as needed
sns.barplot(x='Value', y='Frequency', data=df, palette='viridis')

# Adding plot title and labels for clarity
plt.title('Frequency Distribution of Descriptors')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Rotating the x-axis labels for better readability if necessary
plt.xticks(rotation=45)

# Show the plot
plt.show()
