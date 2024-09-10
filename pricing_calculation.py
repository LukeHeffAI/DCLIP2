import json

# Load all class-descriptor sets from json files
with open('descriptors/descriptors_imagenet.json', 'r') as file:
    descriptors_imagenet = json.load(file)

with open('descriptors/descriptors_cub.json', 'r') as file:
    descriptors_cub = json.load(file)

with open('descriptors/descriptors_eurosat.json', 'r') as file:
    descriptors_eurosat = json.load(file)

with open('descriptors/descriptors_places365.json', 'r') as file:
    descriptors_places365 = json.load(file)

with open('descriptors/descriptors_food101.json', 'r') as file:
    descriptors_food101 = json.load(file)

with open('descriptors/descriptors_pets.json', 'r') as file:
    descriptors_pets = json.load(file)

with open('descriptors/descriptors_dtd.json', 'r') as file:
    descriptors_dtd = json.load(file)

# Print sum of length of all keys in each dataset
class_count = (len(descriptors_imagenet.keys()) + len(descriptors_cub.keys()) + len(descriptors_eurosat.keys()) + len(descriptors_places365.keys()) + len(descriptors_food101.keys()) + len(descriptors_pets.keys()) + len(descriptors_dtd.keys()))

price_input = class_count * 700 * 0.005 / 1000  # number of classes * est. number of input tokens per class * price per 1000 tokens 
price_output = class_count * 140 * 0.015 / 1000 # number of classes * est. number of output tokens per class * price per 1000 tokens
price_total = price_input + price_output        # total price

price_aud = price_total * 1.48 # conversion rate from USD to AUD

print(f"Total classes: {class_count}")
print(f"Price in AUD: ${price_aud:.2f}")