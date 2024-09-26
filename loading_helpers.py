import torch
import os

import numpy as np
import random

cut_proportion = 1

frequency_type = None
# Options:
# [ None,
#   'freq_exact',
#   'freq_approx']

similarity_penalty_config = None
# Options:
# [ None,
#   'similarity_penalty']

import json
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

def make_descriptor_sentence(descriptor, hparams):
    if (hparams['category_name_inclusion'] == 'prepend'):
        if descriptor.startswith('a ') or descriptor.startswith('an ') or descriptor.startswith('the '):
            return f"which is {descriptor}"
        elif descriptor.startswith('a') or descriptor.startswith('e') or descriptor.startswith('i') or descriptor.startswith('o') or descriptor.startswith('u'):
            return f"which is an {descriptor}"
        elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
            return f"which {descriptor}"
        elif descriptor.startswith('used'):
            return f"which is {descriptor}"
        else:
            return f"which has {descriptor}"
    elif hparams['category_name_inclusion'] == 'append':
        return f"{descriptor.capitalize()}, which is a description of a "
    
# def make_descriptor_sentence(descriptor):
#     return descriptor.replace('It', 'which').replace('.', ',')
    
def modify_descriptor(descriptor, apply_changes, hparams):
    if apply_changes:
        return make_descriptor_sentence(descriptor, hparams)
    return descriptor

def truncate_label(label, proportion, method='len'):
    '''
    Truncate the label to a certain proportion of its length.
    When method is 'chr', the proportion is the final number of characters in the output.
    When method is 'len', the proportion is the fraction of the total input characters in the output.
    '''
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
    gibberish_descriptions = ''
    gibberish_descriptions = gibberish_descriptions.join(random.choices(character_array, k=length))

    return gibberish_descriptions

def load_gpt_descriptions(hparams, classes_to_load=None, cut_proportion=1):
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}

    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered

    if hparams['class_analysis_fname'] is not None:
        subcategory_dict = load_json(hparams['class_analysis_fname'])

    if hparams['category_name_inclusion'] is not None:
        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)

        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']

            word_to_add = wordify(k)
            for subcategory, classes in subcategory_dict.items():
                if k in classes:
                    subcategory_to_add = subcategory
                    break

            if (hparams['category_name_inclusion'] == 'append'):
                build_descriptor_string = lambda item: f"{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['between_text']}{word_to_add}"
            elif (hparams['category_name_inclusion'] == 'prepend'):
                # Base structure
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                # Base structure, with ImageNet subcategory. Best (63.24%): "A photo of an tench, which is a freshwater fish, which is a type of fish" 
                build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f", which is a type of {subcategory_to_add}"}{hparams['after_text']}"
                # Base structure, with Food101 subcategory. Best (81.26%): "apple pie, which is a pie dish, which would be found on a menu under "desserts""
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f', which would be found on a menu under "{subcategory_to_add}"'}{hparams['after_text']}"
                # Base structure, with EuroSAT subcategory. Best (57.22%): "annual crop land, which has large, open fields, which is a type of agricultural area, from the EuroSAT dataset."
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f', which is a type of {subcategory_to_add}'}{hparams['after_text']}"
                # Base structure, with Oxford Pets subcategory. Best (87.48%): "A photo of a Abyssinian, which has black, grey, or brown fur, which is a breed of short-haired cats, from a dataset containing images of dog and cat breeds."
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f', which is a breed of {subcategory_to_add}'}{hparams['after_text']}"
                # Base structure, with Describable Textures subcategory. Best (45.88%): "banded, which is a repeating pattern of light and dark bands, which is described as a {subcategory} texture"
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f', which is described as a {subcategory_to_add} texture'}{hparams['after_text']}"
                # Base structure, with CUB subcategory. Best (54.02%): "Black-footed Albatross, which is a seabird, which belongs to the genus of albatrosses"
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f', which belongs to the genus of {subcategory_to_add}'}{hparams['after_text']}"
                # Base structure, with Places365 subcategory. Best (40.27%): "airfield, which is an airport, which is a type of air transportation" (note: "A photo of an airfield, which is an airport, which is a type of air transportation" achieved 41.09%)
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f'; a type of location used for {subcategory_to_add}'}{hparams['after_text']}"
                # Descriptor only
                # build_descriptor_string = lambda item: f"{item.capitalize()}"
                # # Class name only
                # build_descriptor_string = lambda item: f"{word_to_add}"
                # Class name with dataset name
                # build_descriptor_string = lambda item: f"{word_to_add}{f', from {hparams['dataset_name']} dataset'}"
                # Class name, plus
                # build_descriptor_string = lambda item: f"{word_to_add}{', '}{create_gibberish_descriptions(2)}"
                # Recreate CLIP paper approach
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}"
                # Recreate WaffleCLIP approach
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}"
                # # Ascend taxonomic class
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add.split(' ')[-1]}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                # # Class name, repetition
                # build_descriptor_string = lambda item: f"{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(word_to_add, hparams['apply_descriptor_modification'], hparams), cut_proportion)}"
                # lol test (works well for EuroSAT)
                # build_descriptor_string = lambda item: f'{word_to_add}{' lol'}{hparams['after_text']}'
                # Consistency test
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']} image"
            else:
                build_descriptor_string = lambda item: truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)

            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}

            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]

            # print an example the first time
            if i == 0: #verbose and 
                print(f"Example description for class '{k}': \"{gpt_descriptions[k][0]}\"\n")
    return gpt_descriptions, unmodify_dict


def seed_everything(seed: int):
    # import random, os
    # import numpy as np
    # import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
 
import matplotlib.pyplot as plt

stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means
  
def show_single_image(image):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    denorm_image = denormalize(image.unsqueeze(0).cpu(), *stats)
    ax.imshow(denorm_image.squeeze().permute(1, 2, 0).clamp(0,1))
    
    plt.show()