from loading_helpers import compute_class_list
import json
from openai import OpenAI
from load import hparams
import time

time_start = time.time()

client = OpenAI()

filename = f'descriptors/gpt3/descriptors_{hparams['dataset']}.json'

with open(filename, 'r') as f:
    data = json.load(f)

class_list = compute_class_list(data, sort_config=False)
example_class_list = ["tiger shark", "night snake", "hermit crab"]

class_list = class_list
classes_assigned_to_subcategories = {}
count = 0

# Create a number of subcategories such that the maximum number of classes per subcategory is 20
n_classes = len(class_list)

if n_classes < 20:
    min_subcategories = 1
else:
    min_subcategories = int(n_classes / 10) + 1

max_classes_per_subcategory = n_classes // min_subcategories

context_prompt = f"The {hparams['dataset_name']} dataset is constructed from {len(class_list)} classes. You will create at minimum {min_subcategories} subcategories to group the classes by and assign at maximum {max_classes_per_subcategory} of the {hparams['dataset_name']} classes to each subcategory. For an example of a subcategory and its classes, a subcategory \"kitchen utensil\" may have the classes \"fork\", \"knife\", \"can opener\" and \"teaspoon\" assigned to it. Every class must be assigned to a subcategory, none can be missed."


def generate_subcategories_from(class_list, context_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": context_prompt + f"\n\nFirst, create the list of subcategories to assign these {hparams['dataset_name']} classes to, in the exact form of a Python list and nothing more, and stop there before assigning the classes.\n\n{hparams['dataset_name']} classes:\n{class_list}"
                }
            ]
            }
        ],
        temperature=0.2,
        max_tokens=min_subcategories*20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    print(response.choices[0].message.content)

    subcategories_list = str(response.choices[0].message.content).replace('\"', '').replace('\'', '').split("[")[1].split(']')[0].replace('\n', '').replace('_', ' ').lower().split(',')
    subcategories_list = [subcategory.strip() for subcategory in subcategories_list]

    print(f"List has {len(subcategories_list)} subcategories: {subcategories_list}")

    return subcategories_list

def refine_subcategories_from(class_list, category_list):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": context_prompt + f"\n\nFirst, create the list of subcategories to assign these {hparams['dataset_name']} classes to, in the exact form of a Python list and nothing more, and stop there before assigning the classes.\n\n{hparams['dataset_name']} classes:\n{class_list}"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": f'Here is a list of subcategories for the {hparams['dataset_name']} classes:\n\nsubcategories = {category_list}'
                }
            ]
            },            
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"The subcategories in this list are too coarse and will not differentiate the classes well. Breakdown the existing subcategories into more specific subcategories to better group the classes, e.g. instead of \"dog\" and \"cat\", use \"terrier\", \"retriever\", \"siamese\" and \"persian\". Use as many as needed to allow the classes to be as distinct as possible, and even removing overly broad subcategories like \"dogs\" and \"cats\". Once again, do not assign classes yet.\n\nSubcategories:\n{category_list}"
                }
            ]
            }
        ],
        temperature=0.4,
        max_tokens=min_subcategories*20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    print(response.choices[0].message.content)

    subcategories_list = str(response.choices[0].message.content).replace('\"', '').replace('\'', '').split("[")[1].split(']')[0].replace('\n', '').replace('_', ' ').lower().split(',')
    subcategories_list = [subcategory.strip() for subcategory in subcategories_list]

    print(f"List has {len(subcategories_list)} subcategories: {subcategories_list}")

    return subcategories_list

def allocate_classes_to(subcategories_list, context_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": context_prompt
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": f'Here is a list of subcategories for the {hparams['dataset_name']} classes:\n\nsubcategories = {subcategories_list}'
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Which of the subcategories in the above Python list should '{class_name}' be assigned to? It must be one of the subcategories in the list, not a new one. If a class could belong to multiple subcategories, assign it to the most unique/least likely subcategory. Respond with only the subcategory name."
                }
            ]
            }
        ],
        temperature=0.4,
        max_tokens=40,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    subcategory = str(response.choices[0].message.content).replace('\"', '').replace('\'', '')

    return subcategory


subcategories_list = generate_subcategories_from(class_list, context_prompt)

time_broad_subcategories = time.time()

if len(class_list) > 300:
    subcategories_list = refine_subcategories_from(class_list, subcategories_list)

time_fine_subcategories = time.time()

for class_name in class_list:
    subcategory = allocate_classes_to(subcategories_list, context_prompt)

    count += 1
    print(f"Class number {count} of {len(class_list)}: ", subcategory)

    if subcategory in classes_assigned_to_subcategories:
        classes_assigned_to_subcategories[subcategory].append(class_name)
    else:
        classes_assigned_to_subcategories[subcategory] = [class_name]

time_assigned = time.time()

print(classes_assigned_to_subcategories)

class_filename = f'class_analysis/json/class_analysis_{hparams['dataset']}.json'
print(f"Saving subcategories to {class_filename}")

with open(class_filename, 'w') as f:
    json.dump(classes_assigned_to_subcategories, f, indent=4)

time_end = time.time()

print(f"Time taken to generate broad subcategories: {time_broad_subcategories - time_start}")
print(f"Time taken to refine subcategories: {time_fine_subcategories - time_broad_subcategories}")
print(f"Time taken to assign classes: {time_assigned - time_fine_subcategories}")
print(f"Time taken to save classes: {time_end - time_assigned}")