from loading_helpers import compute_class_list
import json
from openai import OpenAI
from load import hparams

client = OpenAI()

filename = f'descriptors/gpt3/descriptors_{hparams['dataset']}.json'

with open(filename, 'r') as f:
    data = json.load(f)

class_list = compute_class_list(data, sort_config=False)
example_class_list = ["tiger shark", "night snake", "hermit crab"]

class_list = class_list
classes_assigned_to_subcategories = {}
count = 0

n_classes = len(class_list)
min_subcategories = min(n_classes, n_classes // 20)
max_classes_per_subcategory = n_classes // min_subcategories

context_prompt = f"The {hparams['dataset_name']} dataset is constructed from {len(class_list)} classes. You will create at minimum {min_subcategories} subcategories to group the classes by and assign at maximum {max_classes_per_subcategory} of the {hparams['dataset_name']} classes to each subcategory. For an example of a subcategory and its classes, a subcategory \"kitchen utensils\" may have the classes \"forks\", \"knife\", \"can opener\" and \"teaspoon\" assigned to it. Every class must be assigned to a subcategory, none can be missed."


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
        temperature=0,
        max_tokens=min_subcategories*20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    # print(response.choices[0].message.content)

    subcategories_list = str(response.choices[0].message.content).replace('\"', '').replace('\'', '').split('subcategories = ')[1].split("[")[1].split(']')[0].replace('\n', '').replace(' ', '').replace('_', ' ').split(',')

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
                "text": f'Here is a list of 40 subcategories for the {hparams['dataset_name']} classes:\n\nsubcategories = {subcategories_list}'
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Which of the subcategories in the above Python list should '{class_name}' be assigned to? It must be one of the subcategories in the list, not a new one. Respond with only the subcategory name."
                }
            ]
            }
        ],
        temperature=0,
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

for class_name in class_list:
    subcategory = allocate_classes_to(subcategories_list, context_prompt)

    count += 1
    print(f"Class number {count} of {len(class_list)}: ", subcategory)

    if subcategory in classes_assigned_to_subcategories:
        classes_assigned_to_subcategories[subcategory].append(class_name)
    else:
        classes_assigned_to_subcategories[subcategory] = [class_name]

print(classes_assigned_to_subcategories)

class_filename = f'class_analysis/json/class_analysis_{hparams['dataset']}.json'
print(f"Saving subcategories to {class_filename}")

with open(class_filename, 'w') as f:
    json.dump(classes_assigned_to_subcategories, f, indent=4)