import time

time.sleep(60*5)

from loading_helpers import compute_class_list
import json
from openai import OpenAI
import openai
from load import hparams
import concurrent.futures

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

n_classes = len(class_list)

# Create an estimate for the minimum number of subcategories to create. The LLM won't be able to accurately process this number anyway, but it usually gets close.
if n_classes < 20:
    min_subcategories = 2
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
        frequency_penalty=0.2,
        presence_penalty=0.2,
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

def allocate_classes_to(class_name, subcategories_list, context_prompt):
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
                "text": f"Which of the subcategories in the above Python list should '{class_name}' be assigned to? It must be one of the subcategories in the list, not a new one. If a class could belong to multiple subcategories, assign it to the most unique/least likely subcategory to increase the differentiation of classes. Respond with only the subcategory name."
                }
            ]
            }
        ],
        temperature=0.2,
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

while len(class_list) > 100 and len(subcategories_list) < (len(class_list) / 10) + 1:
    subcategories_list = refine_subcategories_from(class_list, subcategories_list)

time_fine_subcategories = time.time()

failed_classes = []

def process_class(class_name, subcategories_list, context_prompt, max_retries=10, initial_delay=3):
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            subcategory = allocate_classes_to(class_name, subcategories_list, context_prompt)
            return class_name, subcategory
        except openai.RateLimitError as e:
            retries += 1
            print(f"Rate limit hit for {class_name}. Retrying in {delay} seconds... (Attempt {retries} of {max_retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"Error processing {class_name}: {e}")
            return class_name, None
    return class_name, None


with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    future_to_class = {executor.submit(process_class, class_name, subcategories_list, context_prompt): class_name for class_name in class_list}
    
    count = 0
    for future in concurrent.futures.as_completed(future_to_class):
        class_name, subcategory = future.result()

        if subcategory:  # Only add to dictionary if subcategory is valid
            count += 1
            print(f"Class number {count} of {len(class_list)}: ", subcategory)

            if subcategory in classes_assigned_to_subcategories:
                classes_assigned_to_subcategories[subcategory].append(class_name)
            else:
                classes_assigned_to_subcategories[subcategory] = [class_name]
        else:
            # Log the failed class
            print(f"Failed to process class {class_name} after retries.")
            failed_classes.append(class_name)  # Track failed classes

# Wait for 10 seconds before retrying failed classes
if failed_classes:
    print("Waiting for 10 seconds before retrying failed classes...")
    time.sleep(120)
    print(f"Retrying {len(failed_classes)} failed classes...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_class_retry = {executor.submit(process_class, class_name, subcategories_list, context_prompt): class_name for class_name in failed_classes}
        
        for future in concurrent.futures.as_completed(future_to_class_retry):
            class_name, subcategory = future.result()

            if subcategory:  # Only add to dictionary if subcategory is valid
                print(f"Successfully retried class: {class_name}")

                if subcategory in classes_assigned_to_subcategories:
                    classes_assigned_to_subcategories[subcategory].append(class_name)
                else:
                    classes_assigned_to_subcategories[subcategory] = [class_name]
            else:
                print(f"Class {class_name} still failed after retries.")

time_assigned = time.time()

print(classes_assigned_to_subcategories)

class_filename = f'class_analysis/json/class_analysis_{hparams['dataset']}.json'
print(f"Saving subcategories to {class_filename}")

if failed_classes:
    failed_classes_filename = f'class_analysis/json/failed_classes_{hparams["dataset"]}.json'
    print(f"Saving failed classes to {failed_classes_filename}")

    with open(failed_classes_filename, 'w') as f:
        json.dump(failed_classes, f, indent=4)

with open(class_filename, 'w') as f:
    json.dump(classes_assigned_to_subcategories, f, indent=4)

time_end = time.time()

print(f"Time taken to generate broad subcategories: {time_broad_subcategories - time_start}")
print(f"Time taken to refine subcategories: {time_fine_subcategories - time_broad_subcategories}")
print(f"Time taken to assign classes: {time_assigned - time_fine_subcategories}")
print(f"Time taken to save classes: {time_end - time_assigned}")