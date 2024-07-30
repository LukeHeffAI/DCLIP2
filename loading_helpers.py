import torch
import os
import numpy as np
import random
import json
import itertools

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
    class_list = [k for k in data.keys()]
    return sorted(list(set(class_list))) if sort_config else list(set(class_list))

def compute_descriptor_list(data:dict, sort_config = False):
    if sort_config:
        data = dict(sorted(data.items()))
    descriptor_list = [item for sublist in data.values() for item in sublist]
    return sorted(list(set(descriptor_list))) if sort_config else list(set(descriptor_list))

def get_permutations(descriptors, max_permutations=100):
    return list(itertools.permutations(descriptors, min(len(descriptors), max_permutations)))

def format_descriptors(descriptors):
    if len(descriptors) > 1:
        return ', '.join(descriptors[:-1]) + ' and ' + descriptors[-1]
    return descriptors[0]

def wordify(string):
    return string.replace('_', ' ')

def make_descriptor_sentence(descriptor):
    if descriptor.startswith(('a', 'an')):
        return f"which is {descriptor}"
    elif descriptor.startswith(('has', 'often', 'typically', 'may', 'can')):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
def modify_descriptor(descriptor, apply_changes):
    return make_descriptor_sentence(descriptor) if apply_changes else descriptor

def load_gpt_descriptions(hparams, classes_to_load=None, cut_proportion=1):
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}

    def truncate_label(label, proportion):
        return label[:int(len(label) * proportion)]

    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered

    if hparams['category_name_inclusion'] is not None:
        if classes_to_load is not None:
            for k in list(gpt_descriptions.keys()):
                if k not in classes_to_load:
                    print(f"Skipping descriptions for \"{k}\", not in classes to load")
                    gpt_descriptions.pop(k)

        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if not v:
                v = ['']

            word_to_add = wordify(k)
            permutations = get_permutations(v)

            all_descriptor_strings = []
            for perm in permutations:
                # TODO: make truncatation happen before permutation
                combined_descriptor = format_descriptors([truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification']), cut_proportion) for item in perm])
                if hparams['category_name_inclusion'] == 'append':
                    descriptor_string = f"{combined_descriptor}{hparams['between_text']}{word_to_add}"
                elif hparams['category_name_inclusion'] == 'prepend':
                    descriptor_string = f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{combined_descriptor}{hparams['after_text']}"
                else:
                    descriptor_string = combined_descriptor

                all_descriptor_strings.append(descriptor_string)

            unmodify_dict[k] = {descriptor: v for descriptor in all_descriptor_strings}
            gpt_descriptions[k] = all_descriptor_strings

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
