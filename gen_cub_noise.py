import json
import random
import copy

# Initialise the list of categories
with open ('descriptors/descriptors_cub.json', 'r') as cub:
    ds = json.load(cub)

# Initialise the list of noise descriptors
with open('descriptor_modifiers/noise.json', 'r') as noise:
    ds_noise = json.load(noise)
    noise = list(ds_noise.values())[0]

ds_pre_noise  = copy.deepcopy(ds)
ds_post_noise = copy.deepcopy(ds)
ds_noise      = copy.deepcopy(ds)

# Add the noise descriptors to the CUB descriptors
for k, v in ds_pre_noise.items():
    for i in range(len(v)):
        v[i] = noise[int(random.random()*len(noise))] + " " + v[i]

for k, v in ds_post_noise.items():
    for i in range(len(v)):
        v[i] += " " + noise[int(random.random()*len(noise))]

for k, v in ds_noise.items():
    for i in range(len(v)):
        v[i] += " " + noise[int(random.random()*len(noise))]
        v[i] = noise[int(random.random()*len(noise))] + " " + v[i]

# Save the new list of descriptors
with open('descriptors/descriptors_cub_gpt4_pre_noise.json', 'w') as cub_pre_noise:
    json.dump(ds_pre_noise, cub_pre_noise, indent=4)

with open('descriptors/descriptors_cub_gpt4_post_noise.json', 'w') as cub_post_noise:
    json.dump(ds_post_noise, cub_post_noise, indent=4)

with open('descriptors/descriptors_cub_gpt4_noise.json', 'w') as cub_noise:
    json.dump(ds_noise, cub_noise, indent=4)