import json
import numpy as np
import torch
from torch.nn import functional as F
import random
import pathlib

from descriptor_strings import openai_imagenet_classes
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder, Places365, CIFAR10, CIFAR100, FGVCAircraft, StanfordCars, Flowers102, SUN397, Caltech101
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from datasets import _transform, CUBDataset
from collections import OrderedDict
import clip

from loading_helpers import *

# ------------------------
# NEW HELPER FUNCTION
# ------------------------
def randomize_subcategories(class_subcategories, randomize_pct):
    """
    Randomize the subcategory assignment for a fraction (randomize_pct) of classes.
    
    Args:
        class_subcategories (dict): Original mapping of subcategory -> list of classes.
        randomize_pct (float): Fraction between 0 and 1 indicating how many classes to reassign.
    
    Returns:
        dict: A new subcategory mapping (subcategory -> list of classes) with randomized assignments.
    """
    # Create a flat mapping: class -> original subcategory
    class_to_subcat = {}
    for subcat, class_list in class_subcategories.items():
        for cls in class_list:
            class_to_subcat[cls] = subcat

    all_subcats = list(class_subcategories.keys())
    
    # For each class, with probability randomize_pct, choose a new subcategory (different from the original)
    for cls, orig_subcat in class_to_subcat.items():
        if random.random() < randomize_pct:
            possible = [s for s in all_subcats if s != orig_subcat]
            if possible:
                class_to_subcat[cls] = random.choice(possible)
    
    # Reassemble mapping: new_subcategories: subcategory -> list of classes
    new_subcategories = {}
    for cls, subcat in class_to_subcat.items():
        new_subcategories.setdefault(subcat, []).append(cls)
    
    return new_subcategories

def name_random_subcategory(subcategory_classes, client):
    """
    Generate a descriptive name for a subcategory based on its assigned classes.
    
    Args:
        subcategory_classes (list): List of class names assigned to this subcategory
        client: OpenAI client or similar LLM API client
    
    Returns:
        str: A descriptive name for the subcategory
    """
    import tenacity
    from openai import OpenAI
    
    # If no client provided, initialize one
    if client is None:
        client = OpenAI()
    
    # Use a retry decorator to handle API failures
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type((Exception))
    )
    def get_subcategory_name():
        class_list_str = ", ".join(subcategory_classes[0:-2]) + f", and {subcategory_classes[-1]}"
        
        # if len(subcategory_classes) > 10:
        #     class_list_str += f", and {len(subcategory_classes) - 10} more classes"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"I have randomly grouped these classes together out of a larger dataset: [{class_list_str}]. Please provide a short, descriptive subcategory name (1-3 words) that could reasonably encompass anything most of these items have in common. If nothing in common, still give a potential subcategory name. The name should be a noun or noun phrase that could fit into sentences like 'which is a type of ___' or 'belongs to the category of ___'. Respond ONLY with the subcategory name and nothing else, especially no comments or further notes/thoughts."
                    }
                ]
                }
            ],
            temperature=0.4,
            max_tokens=15,
            response_format={"type": "text"}
        )
        
        return response.choices[0].message.content.strip().lower() if response.choices[0].message.content is not None else response.choices[0].message.content
    
    try:
        return get_subcategory_name()
    except Exception as e:
        print(f"Error generating subcategory name: {e}")
        # Fallback to a generic name if LLM fails
        return f"miscellaneous items"

def generate_random_subcategories(classes, num_subcategories=None, name_subcategories=True, client=None):
    """
    Generate completely random subcategories and class assignments.
    
    Args:
        classes (list): List of class names
        num_subcategories (int): Number of subcategories to create. If None,
                                uses sqrt of number of classes (balanced allocation)
        name_subcategories (bool): Whether to use LLM to generate descriptive names
        client: OpenAI client for LLM naming (optional)
    
    Returns:
        dict: A dictionary mapping subcategory names to lists of classes
    """
    # Determine number of subcategories if not specified
    if num_subcategories is None:
        # Use square root of number of classes as heuristic for number of subcategories
        num_subcategories = max(3, int(np.sqrt(len(classes))))
    
    # Initialize temporary subcategory dictionary with numbered names
    temp_subcategories = {i: [] for i in range(num_subcategories)}
    
    # Randomly shuffle classes
    shuffled_classes = classes.copy()
    random.shuffle(shuffled_classes)
    
    # Assign classes evenly to subcategories
    for i, class_name in enumerate(shuffled_classes):
        subcategory_idx = i % num_subcategories
        temp_subcategories[subcategory_idx].append(class_name)
    
    # If we don't need descriptive names, return with generic names
    if not name_subcategories:
        return {f"random_subcategory_{i+1}": classes for i, classes in temp_subcategories.items()}
    
    # Import necessary libraries for LLM naming
    try:
        from openai import OpenAI
        import concurrent.futures
        from tqdm import tqdm
        
        if client is None:
            client = OpenAI()
    except ImportError as e:
        print(f"Required package not found: {e}. Using generic subcategory names.")
        return {f"random_subcategory_{i+1}": classes for i, classes in temp_subcategories.items()}
    
    # Generate descriptive names for each subcategory
    named_subcategories = {}
    print(f"Generating descriptive names for {num_subcategories} random subcategories in parallel...")
    
    # Prepare arguments for parallel execution
    subcategory_tasks = []
    for i, classes_list in temp_subcategories.items():
        subcategory_tasks.append((i, classes_list))
    
    # Process subcategories in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(subcategory_tasks))) as executor:
        # Create a function for the executor to run
        def process_subcategory(task):
            i, classes_list = task
            try:
                subcategory_name = name_random_subcategory(classes_list, client)
                return i, subcategory_name, classes_list, None
            except Exception as e:
                return i, None, classes_list, str(e)
        
        # Execute tasks in parallel with a progress bar
        results = list(tqdm(
            executor.map(process_subcategory, subcategory_tasks),
            total=len(subcategory_tasks),
            desc="Naming subcategories"
        ))
    
    # Process results
    used_names = set()
    for i, name, classes_list, error in results:
        if error:
            print(f"Error naming subcategory {i+1}: {error}")
            named_subcategories[f"miscellaneous items {i+1}"] = classes_list
            continue
            
        # Ensure the name is unique
        base_name = name
        counter = 2
        while name in used_names:
            name = f"{base_name} {counter}"
            counter += 1
        
        used_names.add(name)
        named_subcategories[name] = classes_list
        print(f"Named subcategory {i+1}/{num_subcategories}: '{name}' with {len(classes_list)} classes")
    
    return named_subcategories

# ------------------------
# FUNCTIONS FROM SET_HPARAMS AND OTHERS
# ------------------------
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)
    
def load_descriptors_frequency(hparams):
    freq_filename = hparams.get('descriptor_analysis_fname', None)
    if freq_filename:
        return load_json(freq_filename)
    return None

def compute_class_list(data:dict, sort_config = False):

    if sort_config:
        data = dict(sorted(data.items()))

    class_list = []
    for k in data.keys():
        class_list.append(k)

    if sort_config:
        class_list = sorted(class_list)

    return class_list

def compute_descriptor_list(data:dict, sort_config = False):

    if sort_config:
        data = dict(sorted(data.items()))
        
    descriptor_list = []
    for v in data.values():
        descriptor_list.extend(v)

    if sort_config:
        descriptor_list = sorted(descriptor_list)
    
    return descriptor_list
    

def wordify(string):
    word = string.replace('_', ' ')
    return word

def starts_with_vowel(word):
    vowels = ('a', 'e', 'i', 'o', 'u')
    return word.lower().startswith(vowels)

def make_descriptor_sentence(descriptor, hparams):
    if (hparams['category_name_inclusion'] == 'prepend'):
        if descriptor.startswith('a ') or descriptor.startswith('an ') or descriptor.startswith('the '):
            return f"which is {descriptor}"
        elif starts_with_vowel(descriptor.split(' ')[0]):
            return f"which is an {descriptor}"
        elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
            return f"which {descriptor}"
        elif descriptor.startswith('used'):
            return f"which is {descriptor}"
        else:
            return f"which has {descriptor}"
    elif hparams['category_name_inclusion'] == 'append':
        return f"{descriptor.capitalize()}, which is a description of a "
    
def modify_descriptor(descriptor, apply_changes, hparams):
    if apply_changes:
        return make_descriptor_sentence(descriptor, hparams)
    return descriptor

def truncate_label(label, proportion, method='len'):
    if frequency_type == None and similarity_penalty_config == None:
        if method == 'chr':
            cut_len = int(len(label) * proportion / len(label))
        elif method == 'len':
            cut_len = int(len(label) * proportion)
        return label[:cut_len]
    else:
        return label

def create_gibberish_descriptions(length, repeat=1):
    import string
    import random
    character_array = string.ascii_letters + string.digits
    gibberish_descriptions = ''.join(random.choices(character_array, k=length))
    return gibberish_descriptions

def append_subcategory_descriptor_to(str, hparams, subcategory):
    return f"{str}, with {subcategory}{hparams['after_text']}"

def load_gpt_descriptions(hparams, classes_to_load=None, cut_proportion=1):
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}

    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered

    # Use override subcategories if provided, otherwise load from file.
    if hparams['class_analysis_fname'] is not None:
        if 'class_subcategories_override' in hparams:
            subcategory_dict = hparams['class_subcategories_override']
        else:
            subcategory_dict = load_json(hparams['class_analysis_fname'])

    if hparams['category_name_inclusion'] is not None and (hparams['method'] == 'defntaxs+descriptors' or hparams['method'] == 'defntaxs_tax_descriptor'):
        subcategory_desc_dict = load_json(hparams['subcategory_desc_fname'])

        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)

        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']

            word_to_add = wordify(k)

            subcategory_to_add = None
            for subcategory, classes in subcategory_dict.items():
                if k in classes:
                    subcategory_to_add = subcategory
                    subcategory_descriptor_list = subcategory_desc_dict[subcategory]
                    break
            
            if subcategory_to_add:
                processed_descriptions = []
                if hparams['method'] == 'defntaxs+descriptors':
                    for subcategory_descriptor in subcategory_descriptor_list:
                        if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                            build_descriptor_string = lambda item: append_subcategory_descriptor_to(
                                f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['before_subcategory']}{subcategory_to_add}", hparams, subcategory_descriptor)
                        else:
                            build_descriptor_string = lambda item: append_subcategory_descriptor_to(
                                f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f', which is described as a {subcategory_to_add} texture'}", hparams, subcategory_descriptor)

                if hparams['method'] == 'defntaxs_tax_descriptor':
                    for subcategory_descriptor in subcategory_descriptor_list:
                        if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                            build_descriptor_string = lambda item: append_subcategory_descriptor_to(
                                f"{hparams['before_text']}{word_to_add}{hparams['before_subcategory']}{subcategory_to_add}", hparams, subcategory_descriptor)
                        else:
                            build_descriptor_string = lambda item: append_subcategory_descriptor_to(
                                f"{hparams['before_text']}{word_to_add}{f', which is described as a {subcategory_to_add} texture'}", hparams, subcategory_descriptor)

                processed_descriptions.extend([build_descriptor_string(item) for item in v])
                gpt_descriptions[k] = list(set(processed_descriptions))
                unmodify_dict[k] = {desc: item for desc, item in zip(gpt_descriptions[k], v)}

            if i == 0:
                print(f"Example description for class '{k}': \"{gpt_descriptions[k][0]}\"\n")

    elif hparams['category_name_inclusion'] is not None:
        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)

        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']

            word_to_add = wordify(k)
            subcategory_to_add = "unknown"
            
            if 'class_analysis_fname' in hparams and hparams['class_analysis_fname'] is not None:
                for subcategory, classes in subcategory_dict.items():
                    if k in classes:
                        subcategory_to_add = subcategory
                        break

            if (hparams['category_name_inclusion'] == 'append'):
                build_descriptor_string = lambda item: f"{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['between_text']}{word_to_add}"
            
            elif (hparams['category_name_inclusion'] == 'prepend'):

                if hparams['method'] == 'clip':
                    build_descriptor_string = lambda item: f"{word_to_add}"
                elif hparams['method'] == 'e-clip':
                    build_descriptor_string = lambda item: f"{'A photo of a '}{word_to_add}"
                elif (hparams['method'] == 'd-clip'):
                    build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                elif hparams['method'] == 'waffleclip':
                    build_descriptor_string = lambda item: f"a {word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)} {" "}{create_gibberish_descriptions(4)}"
                elif hparams['method'] == 'waffleclip+concepts':
                    if hparams['concept_phrase']:
                        build_descriptor_string = lambda item: f"A photo of {hparams['concept_phrase']}: a {word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)} {" "}{create_gibberish_descriptions(4)}"
                    else:
                        build_descriptor_string = lambda item: f"a {word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)} {" "}{create_gibberish_descriptions(4)}"
                elif (hparams['method'] == 'defntaxs'):
                    if 'before_subcategory' not in hparams:
                        hparams['before_subcategory'] = ', which is a type of '
                    if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['before_subcategory']}{subcategory_to_add}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f' which presents {subcategory_to_add} appearance when viewed'}{hparams['after_text']}"
                elif (hparams['method'] == 'waffletaxs'):
                    if 'before_subcategory' not in hparams:
                        hparams['before_subcategory'] = ', which is a type of '
                    if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['before_subcategory']}{create_gibberish_descriptions(length=8)}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f' which presents {create_gibberish_descriptions(length=8)} appearance when viewed'}{hparams['after_text']}"
                elif (hparams['method'] == 'taxclip'):
                    if 'before_subcategory' not in hparams:
                        hparams['before_subcategory'] = ', which is a type of '
                    if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(f'{create_gibberish_descriptions(4)} {" "}{create_gibberish_descriptions(4)}', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['before_subcategory']}{subcategory_to_add}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(f'{create_gibberish_descriptions(4)} {" "}{create_gibberish_descriptions(4)}', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f' which presents {subcategory_to_add} appearance when viewed'}{hparams['after_text']}"
                elif (hparams['method'] == 'defntaxs_sans_descriptor'):
                    if hparams['dataset_name'] == 'ImageNet' or hparams['dataset_name'] == 'ImageNetV2':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a type of {subcategory_to_add}'}{hparams['after_text']}"
                    elif hparams['dataset_name'] == 'Food101':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which would be found on a menu under \"{subcategory_to_add}\"'}{hparams['after_text']}"
                    elif hparams['dataset_name'] == 'EuroSAT':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a type of {subcategory_to_add}'}{f', from the EuroSAT dataset.'}"
                    elif hparams['dataset_name'] == 'Oxford Pets':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a breed of {subcategory_to_add}'}{hparams['after_text']}"
                    elif hparams['dataset_name'] == 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is described as a {subcategory_to_add} texture'}{hparams['after_text']}"
                    elif hparams['dataset_name'] == 'Caltech-UCSD Birds 200 (CUB-200)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which belongs to the genus of {subcategory_to_add}'}{hparams['after_text']}"
                    elif hparams['dataset_name'] == 'Places365 Scene Recognition':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a type of place for {subcategory_to_add}'}{hparams['after_text']}"                    
                elif (hparams['method'] == 'other'):
                    build_descriptor_string = lambda item: f"{word_to_add}{', '}{create_gibberish_descriptions(2)}"

            else:
                build_descriptor_string = lambda item: truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)

            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]

            if i == 0:
                print(f"Example description for class '{k}': \"{gpt_descriptions[k][0]}\"\n")
    return gpt_descriptions, unmodify_dict


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------------------
# Modified set_hparams: now supports random subcategory generation with descriptive naming
# ------------------------
def set_hparams(model_size, desc_type, dataset, method, randomize_pct=0.0, subcategory_mode="normal"):
    """
    Set hyperparameters for the experiment.
    
    Args:
        model_size: Model architecture to use
        desc_type: Type of descriptions
        dataset: Dataset to use
        method: Method for text prompt construction
        randomize_pct: Fraction of subcategories to randomize (0-1)
        subcategory_mode: One of "normal" (LLM-generated), "randomize" (perturb LLM),
                        or "random" (completely random subcategories)
    """
    hparams = {}

    hparams['model_size'] = model_size
    hparams['desc_type'] = desc_type
    hparams['dataset'] = dataset
    hparams['method'] = method
    hparams['randomize_subcat_pct'] = randomize_pct
    hparams['subcategory_mode'] = subcategory_mode

    # Set additional hyperparameters
    hparams['batch_size'] = 64*10
    hparams['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    hparams['category_name_inclusion'] = 'prepend'
    hparams['apply_descriptor_modification'] = True
    hparams['verbose'] = False
    hparams['image_size'] = 224
    if hparams['model_size'] == 'ViT-L/14@336px' and hparams['image_size'] != 336:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 336.')
        hparams['image_size'] = 336
    elif hparams['model_size'] == 'RN50x4' and hparams['image_size'] != 288:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
        hparams['image_size'] = 288
    elif hparams['model_size'] == 'RN50x16' and hparams['image_size'] != 384:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 384.')
        hparams['image_size'] = 384
    elif hparams['model_size'] == 'RN50x64' and hparams['image_size'] != 448:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 448.')
        hparams['image_size'] = 448

    hparams['seed'] = 1

    # Descriptor and analysis filenames
    hparams['descriptor_fname'] = None

    IMAGENET_DIR = '/home/luke/Documents/GitHub/data/ImageNet/'
    IMAGENETV2_DIR = '/home/luke/Documents/GitHub/data/ImageNetV2/'
    CUB_DIR = '/home/luke/Documents/GitHub/data/CUB/CUB_200_2011/'
    EUROSAT_DIR = '/home/luke/Documents/GitHub/data/EuroSAT/2750/'
    FOOD101_DIR = '/home/luke/Documents/GitHub/data/FOOD_101/food-101/food-101/'
    PETS_DIR = '/home/luke/Documents/GitHub/data/Oxford_Pets/'
    DTD_DIR = '/home/luke/Documents/GitHub/data/DTD/dtd/'
    PLACES_DIR = '/home/luke/Documents/GitHub/data/places_devkit/torch_download/'
    CIFAR10_DIR = '/home/luke/Documents/GitHub/data/CIFAR10/'
    CIFAR100_DIR = '/home/luke/Documents/GitHub/data/CIFAR100/'
    AIRCRAFT_DIR = '/home/luke/Documents/GitHub/data/FGVC-aircraft-2013b/data/'
    CARS_DIR = '/home/luke/Documents/GitHub/data/stanford_cars/'
    FLOWERS_DIR = '/home/luke/Documents/GitHub/data/Oxford_flowers/'
    SUN397_DIR = '/home/luke/Documents/GitHub/data/SUN397/'
    CALTECH101_DIR = '/home/luke/Documents/GitHub/data/Caltech101/'

    tfms = _transform(hparams['image_size'])

    if hparams['dataset'] == 'imagenet':
        hparams['dataset_name'] = 'ImageNet'
        hparams['concept_phrase'] = None
        dsclass = ImageNet        
        hparams['data_dir'] = pathlib.Path(IMAGENET_DIR)
        hparams['analysis_fname'] = 'analysis_imagenet'
        dataset_loader = dsclass(hparams['data_dir'], split='val', transform=tfms)
        classes_to_load = None
        hparams['descriptor_fname'] = 'descriptors_imagenet'
        hparams['before_subcategory'] = ' often categorized as a type of '
        hparams['after_text'] = hparams['label_after_text'] = f', from a large-scale image dataset with diverse categories for visual object recognition.'
            
    elif hparams['dataset'] == 'imagenetv2':
        hparams['dataset_name'] = 'ImageNetV2'
        hparams['concept_phrase'] = None
        dsclass = ImageNetV2
        hparams['data_dir'] = pathlib.Path(IMAGENETV2_DIR)
        hparams['analysis_fname'] = 'analysis_imagenet'
        dataset_loader = dsclass(location=str(hparams['data_dir']), transform=tfms)
        classes_to_load = openai_imagenet_classes
        hparams['descriptor_fname'] = 'descriptors_imagenet'
        hparams['before_subcategory'] = ', which is a type of '
        hparams['after_text'] = hparams['label_after_text'] = f', from a large-scale image dataset with diverse categories for visual object recognition.'

    elif hparams['dataset'] == 'cub':
        hparams['dataset_name'] = 'Caltech-UCSD Birds 200 (CUB-200)'
        hparams['concept_phrase'] = 'a bird'
        hparams['data_dir'] = pathlib.Path(CUB_DIR)
        hparams['analysis_fname'] = 'analysis_cub'
        dataset_loader = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
        classes_to_load = None
        hparams['descriptor_fname'] = 'descriptors_cub'
        hparams['before_subcategory'] = ', which belongs to the genus of '
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset of bird images.'

    elif hparams['dataset'] == 'cub_reassignment':
        hparams['dataset_name'] = 'CUB_reassignment'
        hparams['data_dir'] = pathlib.Path(CUB_DIR)
        hparams['analysis_fname'] = 'analysis_cub'
        dataset_loader = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
        classes_to_load = None
        hparams['descriptor_fname'] = 'descriptors_cub_reassignment'

    elif hparams['dataset'] == 'cub_reassignment_threshold':
        hparams['dataset_name'] = 'CUB_reassignment_threshold'
        hparams['data_dir'] = pathlib.Path(CUB_DIR)
        hparams['analysis_fname'] = 'analysis_cub'
        dataset_loader = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
        classes_to_load = None
        hparams['descriptor_fname'] = 'descriptors_cub_reassignment_threshold'

    elif hparams['dataset'].startswith('cub_gpt4'):
        hparams['dataset_name'] = 'CUB_GPT4_{}'.format(hparams['dataset'][-1].split('_')[2:-1])
        hparams['data_dir'] = pathlib.Path(CUB_DIR)
        hparams['analysis_fname'] = 'analysis_cub'
        dataset_loader = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
        classes_to_load = None
        hparams['descriptor_fname'] = f'descriptors_{hparams["dataset"]}riptors'
        
    elif hparams['dataset'] == 'eurosat':
        hparams['dataset_name'] = 'EuroSAT'
        hparams['concept_phrase'] = 'land use'
        hparams['data_dir'] = pathlib.Path(EUROSAT_DIR)
        hparams['analysis_fname'] = 'analysis_eurosat'
        dsclass = ImageFolder
        dataset_loader = dsclass(str(hparams['data_dir']), transform=tfms)
        hparams['descriptor_fname'] = 'descriptors_eurosat'
        classes_to_load = None
        hparams['before_subcategory'] = ', which is a type of '
        hparams['after_text'] = hparams['label_after_text'] = f', from the EuroSAT dataset.'
        
    elif hparams['dataset'] == 'places365':
        hparams['dataset_name'] = 'Places365 Scene Recognition'
        hparams['concept_phrase'] = 'a place'
        hparams['data_dir'] = pathlib.Path(PLACES_DIR)
        hparams['analysis_fname'] = 'analysis_places365'
        dataset_loader = Places365(hparams['data_dir'], split='val', small=True, download=False, transform=tfms)
        hparams['descriptor_fname'] = 'descriptors_places365'
        classes_to_load = None
        hparams['before_subcategory'] = ', which is a type of place for '
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing diverse scene images for environmental classification tasks.'
        
    elif hparams['dataset'] == 'food101':
        hparams['dataset_name'] = 'Food101'
        hparams['concept_phrase'] = 'a food'
        hparams['data_dir'] = pathlib.Path(FOOD101_DIR)
        hparams['analysis_fname'] = 'analysis_food101'
        dsclass = ImageFolder
        dataset_loader = dsclass(str(hparams['data_dir'] / 'images'), transform=tfms)
        hparams['descriptor_fname'] = 'descriptors_food101'
        classes_to_load = None
        hparams['before_subcategory'] = ', which would be found on a menu under '
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing 101 food categories.'

    elif hparams['dataset'] == 'pets':
        hparams['dataset_name'] = 'Oxford Pets'
        hparams['concept_phrase'] = 'a breed'
        hparams['data_dir'] = pathlib.Path(PETS_DIR)
        hparams['analysis_fname'] = 'analysis_pets'
        dsclass = ImageFolder
        dataset_loader = dsclass(str(hparams['data_dir'] / 'images'), transform=tfms)
        hparams['descriptor_fname'] = 'descriptors_pets'
        classes_to_load = None
        hparams['before_subcategory'] = ', which is a breed of '
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of dog and cat breeds.'
        
    elif hparams['dataset'] == 'dtd':
        hparams['dataset_name'] = 'Describable Textures Dataset (DTD)'
        hparams['concept_phrase'] = None
        hparams['data_dir'] = pathlib.Path(DTD_DIR)
        hparams['analysis_fname'] = 'analysis_dtd'
        dataset_loader = ImageFolder(str(hparams['data_dir'] / 'images'), transform=tfms)
        hparams['descriptor_fname'] = 'descriptors_dtd'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images categorized by visual textures.'

    elif hparams['dataset'] == 'cifar10':
        hparams['dataset_name'] = 'CIFAR-10'
        hparams['data_dir'] = pathlib.Path(CIFAR10_DIR)
        dataset_loader = CIFAR10(hparams['data_dir'], train=False, transform=tfms, download=True)
        hparams['descriptor_fname'] = 'descriptors_cifar10'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of 10 different classes.'

    elif hparams['dataset'] == 'cifar100':
        hparams['dataset_name'] = 'CIFAR-100'
        hparams['data_dir'] = pathlib.Path(CIFAR100_DIR)
        dataset_loader = CIFAR100(hparams['data_dir'], train=False, transform=tfms, download=True)
        hparams['descriptor_fname'] = 'descriptors_cifar100'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of 100 different classes.'

    elif hparams['dataset'] == 'aircraft':
        hparams['dataset_name'] = 'FGVC Aircraft'
        hparams['data_dir'] = pathlib.Path(AIRCRAFT_DIR)
        dataset_loader = FGVCAircraft(hparams['data_dir'], split='val', transform=tfms, download=False)
        hparams['descriptor_fname'] = 'descriptors_aircraft'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of aircrafts.'

    elif hparams['dataset'] == 'cars':
        hparams['dataset_name'] = 'Stanford Cars'
        hparams['data_dir'] = pathlib.Path(CARS_DIR)
        dataset_loader = StanfordCars(hparams['data_dir'], split='test', transform=tfms, download=True)
        hparams['descriptor_fname'] = 'descriptors_cars'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of cars.'

    elif hparams['dataset'] == 'flowers':
        hparams['dataset_name'] = 'Oxford Flowers'
        hparams['data_dir'] = pathlib.Path(FLOWERS_DIR)
        dataset_loader = Flowers102(str(hparams['data_dir'] / 'jpg'), split='test', transform=tfms, download=False)
        hparams['descriptor_fname'] = 'descriptors_flowers'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of flowers.'
        
    elif hparams['dataset'] == 'sun397':
        hparams['dataset_name'] = 'SUN397'
        hparams['data_dir'] = pathlib.Path(SUN397_DIR)
        dataset_loader = SUN397(hparams['data_dir'], transform=tfms, download=True)
        hparams['descriptor_fname'] = 'descriptors_sun397'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of scenes and places.'

    elif hparams['dataset'] == 'caltech101':
        hparams['dataset_name'] = 'Caltech101'
        hparams['data_dir'] = pathlib.Path(CALTECH101_DIR)
        dataset_loader = Caltech101(hparams['data_dir'], transform=tfms, download=True)
        hparams['descriptor_fname'] = 'descriptors_caltech101'
        classes_to_load = None
        hparams['after_text'] = hparams['label_after_text'] = f', from a dataset containing images of objects from 101 categories.'

    if hparams['dataset'] != 'imagenetv2':
        dataset_classes = dataset_loader.classes
    else:
        dataset_classes = classes_to_load

    hparams['before_text'] = ""
    hparams['label_before_text'] = ""
    hparams['between_text'] = ', '
    if hparams['dataset'] != 'eurosat': hparams['after_text'] = ''
    hparams['unmodify'] = True
    hparams['label_after_text'] = ''

    hparams['descriptor_fname'] = f'./descriptors/{hparams["desc_type"]}/{hparams["descriptor_fname"]}'
    hparams['descriptor_analysis_fname'] = './descriptor_analysis/descriptors_' + hparams['analysis_fname']
    hparams['class_analysis_fname'] = './class_analysis/json/class_' + hparams['analysis_fname']
    hparams['subcategory_desc_fname'] = './class_analysis/json/class_' + hparams['analysis_fname'] + '_descriptors'

    print("Loading class subcategories...")
    
    # Handle subcategory generation based on specified mode
    if subcategory_mode == "random":
        # Generate completely random subcategories with LLM-generated names
        print("Loading descriptor data to extract class list...")
        try:
            with open(hparams['descriptor_fname'] + '.json', 'r') as f:
                descriptor_data = json.load(f)
                class_list = compute_class_list(descriptor_data, sort_config=False)
            
            from openai import OpenAI
            client = OpenAI()
            
            print(f"Generating random subcategories with descriptive names for {len(class_list)} classes...")
            class_subcategories = generate_random_subcategories(class_list, client=client)
            print(f"Generated random subcategories with descriptive names")
            hparams['class_subcategories_override'] = class_subcategories
        except Exception as e:
            print(f"Error generating named subcategories: {e}")
            print("Using existing subcategories as fallback")
            with open(hparams['class_analysis_fname'] + '.json', 'r') as f:
                class_subcategories = json.load(f)
            hparams['class_subcategories_override'] = class_subcategories
    else:
        # Load existing subcategories
        with open(hparams['class_analysis_fname'] + '.json', 'r') as f:
            class_subcategories = json.load(f)
            
        # Apply randomization if requested
        if subcategory_mode == "randomize" and hparams['randomize_subcat_pct'] > 0:
            randomized_subcats = randomize_subcategories(class_subcategories, hparams['randomize_subcat_pct'])
            print(f"Randomized subcategory assignments for {hparams['randomize_subcat_pct']*100:.1f}% of classes.")
            hparams['class_subcategories_override'] = randomized_subcats
        else:
            hparams['class_subcategories_override'] = class_subcategories

    print("Creating descriptors from {}...".format(hparams['descriptor_fname'].split("/")[-1]))

    gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load, cut_proportion=cut_proportion)
    label_to_classname = list(gpt_descriptions.keys())

    n_classes = len(list(gpt_descriptions.keys()))

    return hparams, tfms, dataset_loader, dataset_classes, class_subcategories, gpt_descriptions, unmodify_dict, label_to_classname, n_classes

def penalty_metrics(hparams):
    if frequency_type == 'freq_exact' or frequency_type == 'freq_approx' or similarity_penalty_config == 'similarity_penalty':
        descriptors_stats = load_json(hparams['descriptor_analysis_fname'] + '.json')
        freq_exact = descriptors_stats['freq_exact']
        freq_approx = descriptors_stats['freq_approx']
        descriptor_self_similarity = descriptors_stats['descriptor-self-similarity']

    return freq_exact, freq_approx, descriptor_self_similarity


def compute_description_encodings(model, gpt_descriptions, hparams, batch_size=32):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        encodings = []
        for i in range(0, len(v), batch_size):
            batch = v[i:i + batch_size]
            tokens = clip.tokenize(batch).to(hparams['device'])
            encodings.append(F.normalize(model.encode_text(tokens)).cpu())
        description_encodings[k] = torch.cat(encodings).to(hparams['device'])
    return description_encodings

def compute_label_encodings(model, hparams, label_to_classname):
    # print(hparams['label_before_text'], hparams['label_after_text'], hparams['device'])
    label_encodings = F.normalize(model.encode_text(clip.tokenize([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(hparams['device'])))
    return label_encodings

# def compute_label_encodings(model): # Original function
#     label_encodings = F.normalize(model.encode_text(clip.tokenize([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(hparams['device'])))
#     return label_encodings

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': 
        return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': 
        return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': 
        return similarity_matrix_chunk.mean(dim=1)
    else: 
        raise ValueError("Unknown aggregate_similarity")
    
def print_descriptor_similarity(image_description_similarity, index, label, label_name, gpt_descriptions, unmodify_dict, label_type="provided"):
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(f"Average:\t\t{100.*aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key = lambda x: x[1], reverse=True):
        k = unmodify_dict[label_name][k]
        print(f"{k}\t{100.*v}")
        
def print_max_descriptor_similarity(image_description_similarity, index, label, label_name, gpt_descriptions, unmodify_dict):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")

