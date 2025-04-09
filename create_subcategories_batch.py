from loading_helpers import compute_class_list
import json
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
load_dotenv()

import concurrent.futures
from tqdm import tqdm
import tenacity

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type((Exception))
)
def allocate_classes_to(class_name, subcategories_list, context_prompt, client):
    # Wrap parameters in a batch request list for the Batch API
    batch_request = [{
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": context_prompt}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Here is a list of subcategories for the classes:\n\nSubcategories = {subcategories_list}"}]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"Which of the subcategories in the above Python list should '{class_name}' be assigned to? "
                        "It must be one of the subcategories in the list, not a new one. "
                        "If a class could belong to multiple subcategories, assign it to the most unique/least likely subcategory. "
                        "If you are unsure, just choose the best fit, DO NOT ponder or talk about your choices. "
                        "Respond with ONLY the subcategory name and NOTHING ELSE, including comments or other notes."
                    )
                }]
            }
        ],
        "temperature": 0.4,
        "max_tokens": 40,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "text"}
    }]

    responses = client.batch.chat.completions.create(
        model="gpt-4o",
        requests=batch_request
    )
    response = responses[0]
    subcategory = str(response.choices[0].message.content).replace('\"', '').replace('\'', '')
    return subcategory

def assign_class_parallel(args):
    class_name, subcategories_list, context_prompt, client = args
    try:
        subcategory = allocate_classes_to(class_name, subcategories_list, context_prompt, client)
        return class_name, subcategory
    except Exception as e:
        print(f"Error assigning {class_name}: {e}")
        return class_name, None

def generate_subcategories_from(class_list, context_prompt, client):
    min_subcategories = len(class_list) // 20 + 1
    if min_subcategories < 1:
        min_subcategories = 1

    batch_request = [{
        "messages": [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": context_prompt + f"\n\nFirst, create the list of subcategories to assign these classes to, in the exact form of a Python list and nothing more, and stop there before assigning the classes. Respond with ONLY the list of subcategory names and NOTHING ELSE, including comments or other notes with the names.\n\nClasses:\n{class_list}"
                }]
            }
        ],
        "temperature": 0.2,
        "max_tokens": min_subcategories * 20,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "text"}
    }]

    responses = client.batch.chat.completions.create(
        model="gpt-4o",
        requests=batch_request
    )
    response = responses[0]
    subcategories_list = str(response.choices[0].message.content)
    subcategories_list = subcategories_list.replace('\"', '').replace('\'', '')
    subcategories_list = subcategories_list.split("[")[1].split(']')[0].replace('\n', '').replace('_', ' ').lower().split(',')
    subcategories_list = [subcategory.strip() for subcategory in subcategories_list]

    print(f"List has {len(subcategories_list)} subcategories, including: {subcategories_list[0:5]}")

    return subcategories_list

def refine_subcategories_from(class_list, category_list, context_prompt, client):
    max_classes_per_subcategory = 15
    min_subcategories = len(class_list) // max_classes_per_subcategory + 1
    if min_subcategories < 1:
        min_subcategories = 1

    batch_request = [{
        "messages": [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": context_prompt + f"\n\nFirst, create the list of subcategories to assign these classes to, in the exact form of a Python list and nothing more, and stop there before assigning the classes.\n\nClasses:\n{class_list}"
                }]
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"Here is a list of subcategories for the classes:\n\nSubcategories = {category_list}"
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"The subcategories in this list are too coarse and will not differentiate the classes well. "
                        f"Breakdown the existing subcategories into more specific subcategories to better group the classes. "
                        f"For example, instead of using \"dog\" and \"cat\", create a refined list of breeds or types. "
                        f"Ensure there are no more than {max_classes_per_subcategory} classes per subcategory (roughly {min_subcategories} minimum subcategories). "
                        "Respond with ONLY the list of subcategory names and NOTHING ELSE, including comments or notes."
                    )
                }]
            }
        ],
        "temperature": 0.4,
        "max_tokens": min_subcategories * 25,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "text"}
    }]

    responses = client.batch.chat.completions.create(
        model="gpt-4o",
        requests=batch_request
    )
    response = responses[0]
    subcategories_list = str(response.choices[0].message.content)
    subcategories_list = subcategories_list.replace('\"', '').replace('\'', '')
    subcategories_list = subcategories_list.split("[")[1].split(']')[0].replace('\n', '').replace('_', ' ').lower().split(',')
    subcategories_list = [subcategory.strip() for subcategory in subcategories_list]

    print(f"List has {len(subcategories_list)} subcategories, including: {subcategories_list[0:5]}")

    return subcategories_list

def refine_large_subcategories(classes_assigned_to_subcategories, max_classes_per_subcategory, context_prompt, client, max_workers=1):
    """
    Identify subcategories with too many classes and further refine them into more specific subcategories.
    """
    refined_assignments = {}
    large_subcategories = {subcat: classes for subcat, classes in classes_assigned_to_subcategories.items() 
                           if len(classes) > max_classes_per_subcategory}
    
    if not large_subcategories:
        return classes_assigned_to_subcategories
    
    print(f"Found {len(large_subcategories)} subcategories with too many classes")
    
    for subcategory, classes in large_subcategories.items():
        print(f"Refining subcategory '{subcategory}' with {len(classes)} classes (max: {max_classes_per_subcategory})")
        
        num_needed_subcategories = len(classes) // max_classes_per_subcategory + 1
        class_list_str = ", ".join(classes)
        
        batch_request = [{
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": (
                            f"{context_prompt}\n\nThe subcategory '{subcategory}' has too many classes ({len(classes)} total). "
                            f"Please create {num_needed_subcategories} more specific subcategories to replace it and better organize these classes. "
                            "Respond with ONLY the list of subcategory names and NOTHING ELSE, including comments or notes. "
                            f"Return only a Python list of the new subcategories.\n\nClasses in '{subcategory}':\n{class_list_str}\n\n"
                            f"Overall existing subcategories list:\n{list(classes_assigned_to_subcategories.keys())}"
                        )
                    }]
                }
            ],
            "temperature": 0.3,
            "max_tokens": num_needed_subcategories * 25,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {"type": "text"}
        }]

        responses = client.batch.chat.completions.create(
            model="gpt-4o",
            requests=batch_request
        )
        response = responses[0]
        refined_subcats = str(response.choices[0].message.content)
        refined_subcats = refined_subcats.replace('\"', '').replace('\'', '')
        refined_subcats = refined_subcats.split("[")[1].split(']')[0].replace('\n', '').replace('_', ' ').lower().split(',')
        refined_subcats = [subcat.strip() for subcat in refined_subcats]
        
        print(f"Created {len(refined_subcats)} refined subcategories: {refined_subcats[:5]}...")
        
        class_assignments = {subcat: [] for subcat in refined_subcats}
        
        args_list = [
            (
                class_name,
                refined_subcats,
                f"{context_prompt}\n\nWe are refining the subcategory '{subcategory}' into more specific subcategories.",
                client
            )
            for class_name in classes
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(assign_class_parallel, args_list),
                total=len(classes),
                desc=f"Refining '{subcategory}'"
            ))
        
        for class_name, assigned_subcat in results:
            if assigned_subcat and assigned_subcat in refined_subcats:
                class_assignments[assigned_subcat].append(class_name)
            else:
                print(f"Retrying allocation for {class_name} in refined subcategories")
                try:
                    assigned_subcat = allocate_classes_to(class_name, refined_subcats,
                                                          f"{context_prompt}\n\nWe are refining the subcategory '{subcategory}'.",
                                                          client)
                    if assigned_subcat in refined_subcats:
                        class_assignments[assigned_subcat].append(class_name)
                    else:
                        class_assignments[refined_subcats[0]].append(class_name)
                except Exception as e:
                    print(f"Error reassigning {class_name}: {e}")
                    class_assignments[refined_subcats[0]].append(class_name)
        
        for refined_subcat, assigned_classes in class_assignments.items():
            if assigned_classes:
                refined_assignments[refined_subcat] = assigned_classes
    
    for subcategory, classes in classes_assigned_to_subcategories.items():
        if subcategory not in large_subcategories:
            refined_assignments[subcategory] = classes
    
    return refined_assignments

def create_subcategories(hparams, force=False, max_workers=1, max_classes_per_subcategory=10):
    """
    Create subcategories for classes in the dataset.
    """
    time_start = time.time()
    
    class_filename = f'class_analysis/json/class_analysis_{hparams["dataset"]}.json'
    if os.path.exists(class_filename) and not force:
        print(f"Loading existing subcategories from {class_filename}")
        with open(class_filename, 'r') as f:
            return json.load(f)
    
    print(f"Creating new subcategories for {hparams['dataset']}")
    client = OpenAI()

    filename = f'descriptors/{hparams["desc_type"]}/descriptors_{hparams["dataset"]}.json'

    with open(filename, 'r') as f:
        data = json.load(f)

    class_list = compute_class_list(data, sort_config=False)
    classes_assigned_to_subcategories = {}

    n_classes = len(class_list)
    if n_classes < 21:
        min_subcategories = 1
    else:
        min_subcategories = int(n_classes / max_classes_per_subcategory)

    context_prompt = (
        f"The {hparams['dataset_name']} dataset is constructed from {len(class_list)} classes. "
        f"You will create at minimum {min_subcategories} subcategories to group the classes by and assign at maximum {max_classes_per_subcategory} of the {hparams['dataset_name']} classes to each subcategory. "
        'For example, a subcategory "kitchen utensil" may have the classes "fork", "knife", "can opener" and "teaspoon" assigned to it. Every class must be assigned to a subcategory, none can be missed.'
    )

    subcategories_list = generate_subcategories_from(class_list, context_prompt, client)
    time_broad_subcategories = time.time()

    args_list = [(class_name, subcategories_list, context_prompt, client) for class_name in class_list]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(assign_class_parallel, args_list),
            total=len(class_list),
            desc="Assigning classes"
        ))
    
    for class_name, subcategory in results:
        if subcategory:
            if subcategory in classes_assigned_to_subcategories:
                classes_assigned_to_subcategories[subcategory].append(class_name)
            else:
                classes_assigned_to_subcategories[subcategory] = [class_name]
        else:
            print(f"Retrying allocation for {class_name}")
            subcategory = allocate_classes_to(class_name, subcategories_list, context_prompt, client)
            if subcategory in classes_assigned_to_subcategories:
                classes_assigned_to_subcategories[subcategory].append(class_name)
            else:
                classes_assigned_to_subcategories[subcategory] = [class_name]

    time_assigned = time.time()

    print("Checking for subcategories with too many classes...")
    classes_assigned_to_subcategories = refine_large_subcategories(
        classes_assigned_to_subcategories,
        max_classes_per_subcategory,
        context_prompt,
        client,
        max_workers
    )

    time_refined = time.time()

    os.makedirs(os.path.dirname(class_filename), exist_ok=True)
    print(f"Saving subcategories to {class_filename}")

    with open(class_filename, 'w') as f:
        json.dump(classes_assigned_to_subcategories, f, indent=4)

    time_end = time.time()

    print(f"Time taken: {time_end - time_start:.2f} seconds")
    return classes_assigned_to_subcategories
