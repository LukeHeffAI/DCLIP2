from loading_helpers import compute_class_list
import json
from openai import OpenAI
import time
import os
import concurrent.futures
from tqdm import tqdm
import tenacity

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type((Exception))
)
def allocate_classes_to(class_name, subcategories_list, context_prompt, client):
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
                "text": f'Here is a list of subcategories for the classes:\n\nSubcategories = {subcategories_list}'
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
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": context_prompt + f"\n\nFirst, create the list of subcategories to assign these classes to, in the exact form of a Python list and nothing more, and stop there before assigning the classes.\n\nClasses:\n{class_list}"
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

def refine_subcategories_from(class_list, category_list, context_prompt, client):
    min_subcategories = len(class_list) // 20 + 1
    if min_subcategories < 1:
        min_subcategories = 1
        
    # Generate refined subcategories
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": context_prompt + f"\n\nFirst, create the list of subcategories to assign these classes to, in the exact form of a Python list and nothing more, and stop there before assigning the classes.\n\nClasses:\n{class_list}"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": f'Here is a list of subcategories for the classes:\n\nSubcategories = {category_list}'
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

def create_subcategories(hparams, force=False, max_workers=10):
    """
    Create subcategories for classes in the dataset.
    
    Args:
        hparams: Hyperparameters dictionary
        force: If True, regenerate subcategories even if they already exist
        max_workers: Maximum number of concurrent workers for parallelization
        
    Returns:
        classes_assigned_to_subcategories: Dictionary mapping subcategories to classes
    """
    time_start = time.time()
    
    # Check if subcategories already exist and if we should use them
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

    # Create a number of subcategories such that the maximum number of classes per subcategory is 20
    n_classes = len(class_list)

    if n_classes < 20:
        min_subcategories = 1
    else:
        min_subcategories = int(n_classes / 10) + 1

    max_classes_per_subcategory = n_classes // min_subcategories

    context_prompt = f"The {hparams['dataset_name']} dataset is constructed from {len(class_list)} classes. You will create at minimum {min_subcategories} subcategories to group the classes by and assign at maximum {max_classes_per_subcategory} of the {hparams['dataset_name']} classes to each subcategory. For an example of a subcategory and its classes, a subcategory \"kitchen utensil\" may have the classes \"fork\", \"knife\", \"can opener\" and \"teaspoon\" assigned to it. Every class must be assigned to a subcategory, none can be missed."

    # Generate initial subcategories
    subcategories_list = generate_subcategories_from(class_list, context_prompt, client)
    time_broad_subcategories = time.time()

    # Refine subcategories for large datasets
    if len(class_list) > 300 or len(subcategories_list) < min_subcategories:
        subcategories_list = refine_subcategories_from(class_list, subcategories_list, context_prompt, client)
    time_fine_subcategories = time.time()

    # Parallelize class allocation
    print(f"Allocating {len(class_list)} classes to subcategories in parallel (max_workers={max_workers})...")
    
    # Prepare arguments for parallel execution
    args_list = [(class_name, subcategories_list, context_prompt, client) for class_name in class_list]
    
    # Execute in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(assign_class_parallel, args_list), 
            total=len(class_list),
            desc="Assigning classes"
        ))
    
    # Process results
    for class_name, subcategory in results:
        if subcategory:
            if subcategory in classes_assigned_to_subcategories:
                classes_assigned_to_subcategories[subcategory].append(class_name)
            else:
                classes_assigned_to_subcategories[subcategory] = [class_name]
        else:
            # For classes that failed, try again sequentially
            print(f"Retrying allocation for {class_name}")
            subcategory = allocate_classes_to(class_name, subcategories_list, context_prompt, client)
            if subcategory in classes_assigned_to_subcategories:
                classes_assigned_to_subcategories[subcategory].append(class_name)
            else:
                classes_assigned_to_subcategories[subcategory] = [class_name]

    time_assigned = time.time()

    print(classes_assigned_to_subcategories)

    # Ensure directory exists
    os.makedirs(os.path.dirname(class_filename), exist_ok=True)
    print(f"Saving subcategories to {class_filename}")

    with open(class_filename, 'w') as f:
        json.dump(classes_assigned_to_subcategories, f, indent=4)

    time_end = time.time()

    print(f"Time taken to generate broad subcategories: {time_broad_subcategories - time_start}")
    print(f"Time taken to refine subcategories: {time_fine_subcategories - time_broad_subcategories}")
    print(f"Time taken to assign classes: {time_assigned - time_fine_subcategories}")
    print(f"Time taken to save classes: {time_end - time_assigned}")
    print(f"Total time: {time_end - time_start}")
    
    return classes_assigned_to_subcategories

if __name__ == "__main__":
    # If this script is run directly, use these settings
    from load import set_hparams, update_hparams
    
    # Set the hyperparameters
    hparams = set_hparams(model_size='ViT-B/32', desc_type='gpt-3', dataset='eurosat', method='defntaxs')
    
    # Update the hyperparameters
    hparams, _, _, _, _, _, _, _, _ = update_hparams(hparams)
    
    # Force regeneration of subcategories with parallel processing
    create_subcategories(hparams, force=True, max_workers=10)