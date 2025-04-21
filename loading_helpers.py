import torch
import os
import json
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

def append_subcategory_descriptor_to(str, hparams, subcategory):
    return f"{str}, with {subcategory}{hparams['after_text']}"

def load_gpt_descriptions(hparams, classes_to_load=None, cut_proportion=1):
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}

    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered

    if hparams['class_analysis_fname'] is not None:
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
                # Create the augmented descriptions for the current class
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

                # Add processed descriptions for this subcategory descriptor
                processed_descriptions.extend([build_descriptor_string(item) for item in v])

                # Deduplicate descriptions and update the dictionary
                gpt_descriptions[k] = list(set(processed_descriptions))
                unmodify_dict[k] = {desc: item for desc, item in zip(gpt_descriptions[k], v)}

                            # print an example the first time
            if i == 0: #verbose and 
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
            
            # Initialize subcategory_to_add with a default value
            subcategory_to_add = "unknown"
            
            # Try to find the subcategory for this class
            if 'class_analysis_fname' in hparams and hparams['class_analysis_fname'] is not None:
                for subcategory, classes in subcategory_dict.items():
                    if k in classes:
                        subcategory_to_add = subcategory
                        break

            if (hparams['category_name_inclusion'] == 'append'):
                build_descriptor_string = lambda item: f"{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['between_text']}{word_to_add}"
            
            elif (hparams['category_name_inclusion'] == 'prepend'):

                if hparams['method'] == 'clip':
                    # Class name only
                    build_descriptor_string = lambda item: f"{word_to_add}"

                elif hparams['method'] == 'e-clip':
                    # Recreate CLIP paper approach
                    build_descriptor_string = lambda item: f"{'A photo of a '}{word_to_add}"

                elif (hparams['method'] == 'd-clip'):
                    # Base structure
                    build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"

                elif hparams['method'] == 'waffleclip':
                    # Recreate WaffleCLIP approach
                    build_descriptor_string = lambda item: f"a {word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}"

                elif hparams['method'] == 'waffleclip+concepts':
                    # Recreate WaffleCLIP approach with concepts
                    if hparams['concept_phrase']:
                        build_descriptor_string = lambda item: f"A photo of {hparams['concept_phrase']}: a {word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}"
                    else:
                        build_descriptor_string = lambda item: f"a {word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor('', hparams['apply_descriptor_modification'], hparams), cut_proportion)}{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}"

                elif (hparams['method'] == 'defntaxs'):
                    # Make sure we have 'before_subcategory' in hparams
                    if 'before_subcategory' not in hparams:
                        hparams['before_subcategory'] = ', which is a type of '
                        
                    if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['before_subcategory']}{subcategory_to_add}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f' which presents {subcategory_to_add} appearance when viewed'}{hparams['after_text']}"

                elif (hparams['method'] == 'waffletaxs'):
                    # Make sure we have 'before_subcategory' in hparams
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
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(f"{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}", hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['before_subcategory']}{subcategory_to_add}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(f"{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}", hparams['apply_descriptor_modification'], hparams), cut_proportion)}{f' which presents {subcategory_to_add} appearance when viewed'}{hparams['after_text']}"

                elif (hparams['method'] == 'waffletaxs_swap'):
                    # Make sure we have 'before_subcategory' in hparams
                    if 'before_subcategory' not in hparams:
                        hparams['before_subcategory'] = ', which is a type of '
                        
                    if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['before_subcategory']}{create_gibberish_descriptions(length=8)}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f' which presents {create_gibberish_descriptions(length=8)} appearance'}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"

                elif (hparams['method'] == 'taxclip_swap'):
                    if 'before_subcategory' not in hparams:
                        hparams['before_subcategory'] = ', which is a type of '
                        
                    if hparams['dataset_name'] != 'Describable Textures Dataset (DTD)':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['before_subcategory']}{subcategory_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(f"{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}", hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"
                    else:
                        subcategory_to_add = f'an {subcategory_to_add}' if starts_with_vowel(subcategory_to_add) else f'a {subcategory_to_add}'
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f' which presents {subcategory_to_add} appearance'}{hparams['between_text']}{truncate_label(modify_descriptor(f"{create_gibberish_descriptions(4)}{" "}{create_gibberish_descriptions(4)}", hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"

                elif (hparams['method'] == 'defntaxs_sans_descriptor'):

                    # Best (63.48%): "tench, which is a freshwater fish, which is a type of freshwater fish"
                    # Best (v2) (55.90%): "tench, which is a freshwater fish, which is a type of freshwater fish"
                    if hparams['dataset_name'] == 'ImageNet' or hparams['dataset_name'] == 'ImageNetV2':
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f", which is a type of {subcategory_to_add}"}{hparams['after_text']}"
                    
                    elif hparams['dataset_name'] == 'Food101':
                    # Best (81.26%): "apple pie, which is a pie dish, which would be found on a menu under "desserts""
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which would be found on a menu under "{subcategory_to_add}"'}{hparams['after_text']}"
                    
                    elif hparams['dataset_name'] == 'EuroSAT':
                    # Best (57.22%): "annual crop land, which has large, open fields, which is a type of agricultural area, from the EuroSAT dataset."
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a type of {subcategory_to_add}'}{f', from the EuroSAT dataset.'}"

                    elif hparams['dataset_name'] == 'Oxford Pets':
                    # Best (87.48%): "A photo of a Abyssinian, which has black, grey, or brown fur, which is a breed of short-haired cats, from a dataset containing images of dog and cat breeds."
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a breed of {subcategory_to_add}'}{hparams['after_text']}"

                    elif hparams['dataset_name'] == 'Describable Textures Dataset (DTD)':
                    # Best (45.88%): "banded, which is a repeating pattern of light and dark bands, which is described as a {subcategory_to_add} texture"
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is described as a {subcategory_to_add} texture'}{hparams['after_text']}"

                    elif hparams['dataset_name'] == 'Caltech-UCSD Birds 200 (CUB-200)':
                    # Best (54.02%): "Black-footed Albatross, which is a seabird, which belongs to the genus of albatrosses"
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which belongs to the genus of {subcategory_to_add}'}{hparams['after_text']}"

                    elif hparams['dataset_name'] == 'Places365 Scene Recognition':
                    # Best (40.27%): "airfield, which is an airport, which is a type of air transportation" (note: "A photo of an airfield, which is an airport, which is a type of air transportation" achieved 41.09%)
                        build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{f', which is a type of place for {subcategory_to_add}'}{hparams['after_text']}"                    

                elif (hparams['method'] == 'other'):

                    # Descriptor only
                    build_descriptor_string = lambda item: f"{item.capitalize()}"

                    # Class name with dataset name
                    build_descriptor_string = lambda item: f"{word_to_add}{f', from {hparams['dataset_name']} dataset'}"

                    # Class name, plus
                    build_descriptor_string = lambda item: f"{word_to_add}{', '}{create_gibberish_descriptions(2)}"

                    # # Ascend taxonomic class
                    build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add.split(' ')[-1]}{hparams['between_text']}{truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)}{hparams['after_text']}"

                    # # Class name, repetition
                    build_descriptor_string = lambda item: f"{word_to_add}{hparams['between_text']}{truncate_label(modify_descriptor(word_to_add, hparams['apply_descriptor_modification'], hparams), cut_proportion)}"

            else:
                build_descriptor_string = lambda item: truncate_label(modify_descriptor(item, hparams['apply_descriptor_modification'], hparams), cut_proportion)

            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]

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

# stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means
  
# def show_single_image(image):
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.set_xticks([]); ax.set_yticks([])
#     denorm_image = denormalize(image.unsqueeze(0).cpu(), *stats)
#     ax.imshow(denorm_image.squeeze().permute(1, 2, 0).clamp(0,1))
    
#     plt.show()