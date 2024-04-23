import json
from load import *
import torchmetrics
import torch
from descriptor_analysis import compute_class_list, compute_descriptor_list

descriptor_file = [
    'descriptors/descriptors_cub.json',
    # 'descriptors/descriptors_cub_gpt4_full.json',
    # 'descriptors/descriptors_dtd.json',
    # 'descriptors/descriptors_eurosat.json',
    # 'descriptors/descriptors_food101.json',
    # 'descriptors/descriptors_imagenet.json',
    # 'descriptors/descriptors_pets.json',
    # 'descriptors/descriptors_places365.json',
]

class_descriptor_dict = load_json(descriptor_file[0])
class_list = compute_class_list(class_descriptor_dict)
descriptor_list = compute_descriptor_list(class_descriptor_dict)

print(class_list[0:5], "\n", descriptor_list[0:5])

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
description_encodings = F.normalize(model.encode_text(clip.tokenize(descriptor_list).to(device)))
label_encodings = F.normalize(model.encode_text(clip.tokenize(class_list).to(device)))

cosine_similarity = torch.mm(description_encodings, label_encodings.T)



print(cosine_similarity, cosine_similarity.shape)

# Save this information to a .csv file, with the following format:
# x-axis: class names
# y-axis: descriptor names
# values: cosine similarity

with open('results/descriptor_similarity_cub.csv', 'w') as file:
    file.write(','.join(class_list) + '\n')
    for i, descriptor in enumerate(descriptor_list):
        descriptor = f'"{descriptor}"'
        file.write(descriptor + ',' + ','.join([str(similarity) for similarity in cosine_similarity[i].tolist()]) + '\n')

