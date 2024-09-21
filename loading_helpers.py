import torch
import os

import numpy as np
import random

cut_proportion = 1

similarity_penalty_config = None
# Options:
# [ None,
#   'similarity_penalty']

frequency_type = None
# Options:
# [ None,
#   'freq_is',
#   'freq_contains']

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
        class_list = list(set(class_list))
        class_list = sorted(class_list)
    # else:
    #     class_list = list(set(class_list))

    return class_list

def compute_descriptor_list(data:dict, sort_config = False):

    if sort_config:
        data = dict(sorted(data.items()))
        
    descriptor_list = []
    for v in data.values():
        descriptor_list.extend(v)

    if sort_config:
        descriptor_list = list(set(descriptor_list))
        descriptor_list = sorted(descriptor_list)
    else:
        descriptor_list = list(set(descriptor_list))
    
    return descriptor_list
    

def wordify(string):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor, hparams):
    if (hparams['category_name_inclusion'] == 'prepend'):
        if descriptor.startswith('a') or descriptor.startswith('an'):
            return f"which is {descriptor}"
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

def load_gpt_descriptions(hparams, classes_to_load=None, cut_proportion=1):
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}

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
    
    def create_gibberish_descriptions(length):
        import string
        import random

        gibberish_descriptions = string.ascii_letters + ' ' * 5
        return ''.join(random.choices(gibberish_descriptions, k=length))

    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered

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

            if (hparams['category_name_inclusion'] == 'append'):
                build_descriptor_string = lambda item: f"{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['between_text']}{word_to_add}"
            elif (hparams['category_name_inclusion'] == 'prepend'):
                # Base structure
                build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                # Descriptor only
                # build_descriptor_string = lambda item: f"{item.capitalize()}"
                # # Class name only
                # build_descriptor_string = lambda item: f"{word_to_add}"
                # Class name, plus
                # build_descriptor_string = lambda item: f"{word_to_add}{', '}{create_gibberish_descriptions(2)}"
                # # Ascend taxonomic class
                # build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add.split(' ')[-1]}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                # # Class name, repetition
                # build_descriptor_string = lambda item: f"{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(word_to_add, hparams['apply_descriptor_modification'], hparams), cut_proportion)}"
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