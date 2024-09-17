import json
import numpy as np
import torch
from torch.nn import functional as F

from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor
import pathlib

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder, Places365
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from datasets import _transform, CUBDataset
from collections import OrderedDict
import clip

from loading_helpers import *

# hyperparameters
hparams = {}

hparams['model_size'] = "ViT-B/16"
# Options:
# ['RN50',
#  'RN101',
#  'RN50x4',
#  'RN50x16',
#  'RN50x64',
#  'ViT-B/32',
#  'ViT-B/16',
#  'ViT-L/14',
#  'ViT-L/14@336px']

hparams['desc_type'] = 'gpt3'
# Options:
# ['gpt3',
#  'gpt4]

hparams['dataset'] = 'eurosat'
# Options:
# ['imagenet',
#  'imagenetv2',
#  'cub',
#  'cub_reassignment',
#  'cub_reassignment_threshold',
#  'cub_gpt4_1_desc',
#  'cub_gpt4_2_desc',
#  'cub_gpt4_3_desc',
#  'cub_gpt4_4_desc',
#  'cub_gpt4_5_desc',
#  'cub_gpt4_6_desc',
#  'cub_gpt4_7_desc',
#  'cub_gpt4_8_desc',
#  'eurosat',
#  'places365',
#  'food101',
#  'pets',
#  'dtd']

hparams['batch_size'] = 64*10
hparams['device'] = "cuda" if torch.cuda.is_available() else "cpu"
hparams['category_name_inclusion'] = 'prepend' #'append' 'prepend'

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
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 384
elif hparams['model_size'] == 'RN50x64' and hparams['image_size'] != 448:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 448

hparams['before_text'] = ""
hparams['label_before_text'] = ""
hparams['between_text'] = ', '
# hparams['between_text'] = ' '
# hparams['between_text'] = ''
hparams['after_text'] = ''
hparams['unmodify'] = True
# hparams['after_text'] = '.'
# hparams['after_text'] = ' which is a type of bird.'
hparams['label_after_text'] = ''
# hparams['label_after_text'] = ' which is a type of bird.'
hparams['seed'] = 1

# classes_to_load = openai_imagenet_classes
hparams['descriptor_fname'] = None

IMAGENET_DIR = '/home/luke/Documents/GitHub/data/ImageNet/'
IMAGENETV2_DIR = '/home/luke/Documents/GitHub/data/ImageNetV2/'
CUB_DIR = '/home/luke/Documents/GitHub/data/CUB/CUB_200_2011/'
EUROSAT_DIR = '/home/luke/Documents/GitHub/data/EuroSAT/2750/'
FOOD101_DIR = '/home/luke/Documents/GitHub/data/FOOD_101/food-101/food-101/' # TODO: Fix shape issue when running main.py
PETS_DIR = '/home/luke/Documents/GitHub/data/Oxford_Pets/'
DTD_DIR = '/home/luke/Documents/GitHub/data/DTD/dtd/'
PLACES_DIR = '/home/luke/Documents/GitHub/data/places_devkit/torch_download/'
ESC_10_DIR = '/home/luke/Documents/GitHub/ImageBind/ESC-50-master/audio/ESC-10'


# PyTorch datasets
tfms = _transform(hparams['image_size'])



if hparams['dataset'] == 'imagenet':
    hparams['dataset_name'] = 'ImageNet'
    dsclass = ImageNet        
    hparams['data_dir'] = pathlib.Path(IMAGENET_DIR)
    hparams['analysis_fname'] = 'analysis_imagenet'
    # train_ds = ImageNet(hparams['data_dir'], split='val', transform=train_tfms)
    dataset = dsclass(hparams['data_dir'], split='val', transform=tfms)
    classes_to_load = None
    hparams['descriptor_fname'] = 'descriptors_imagenet'
    hparams['after_text'] = hparams['label_after_text'] = '.'
        
elif hparams['dataset'] == 'imagenetv2':
    hparams['dataset_name'] = 'ImageNetV2'
    dsclass = ImageNetV2
    hparams['data_dir'] = pathlib.Path(IMAGENETV2_DIR)
    hparams['analysis_fname'] = 'analysis_imagenet'
    dataset = dsclass(location=str(hparams['data_dir']), transform=tfms)
    classes_to_load = openai_imagenet_classes
    hparams['descriptor_fname'] = 'descriptors_imagenet'

elif hparams['dataset'] == 'cub':
    hparams['dataset_name'] = 'CUB'
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    hparams['analysis_fname'] = 'analysis_cub'
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_cub'

elif hparams['dataset'] == 'cub_reassignment':
    hparams['dataset_name'] = 'CUB_reassignment'
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    hparams['analysis_fname'] = 'analysis_cub'
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_cub_reassignment'

elif hparams['dataset'] == 'cub_reassignment_threshold':
    hparams['dataset_name'] = 'CUB_reassignment_threshold'
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    hparams['analysis_fname'] = 'analysis_cub'
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_cub_reassignment_threshold'

elif hparams['dataset'].startswith('cub_gpt4'):
    hparams['dataset_name'] = 'CUB_GPT4_{}'.format(hparams['dataset'][-1].split('_')[2:-1])
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    hparams['analysis_fname'] = 'analysis_cub'
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    classes_to_load = None
    hparams['descriptor_fname'] = f'descriptors_{hparams["dataset"]}riptors'
    
# I recommend using VISSL https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md to download these
    
elif hparams['dataset'] == 'eurosat':
    hparams['dataset_name'] = 'EuroSAT'
    # from extra_datasets.patching.eurosat import EuroSATVal
    hparams['data_dir'] = pathlib.Path(EUROSAT_DIR)
    hparams['analysis_fname'] = 'analysis_eurosat'
    # dataset = EuroSATVal(location=hparams['data_dir'], preprocess=tfms)
    # dataset = dataset.test_dataset
    dsclass = ImageFolder
    dataset = dsclass(str(hparams['data_dir']), transform=tfms)
    hparams['descriptor_fname'] = 'descriptors_eurosat'
    classes_to_load = None
    
elif hparams['dataset'] == 'places365':
    hparams['dataset_name'] = 'Places365'
    hparams['data_dir'] = pathlib.Path(PLACES_DIR)
    hparams['analysis_fname'] = 'analysis_places365'
    dataset = Places365(hparams['data_dir'], split='val', small=True, download=False, transform=tfms)
    hparams['descriptor_fname'] = 'descriptors_places365'
    classes_to_load = None
    
elif hparams['dataset'] == 'food101':
    hparams['dataset_name'] = 'Food101'
    hparams['data_dir'] = pathlib.Path(FOOD101_DIR)
    hparams['analysis_fname'] = 'analysis_food101'
    dsclass = ImageFolder
    dataset = dsclass(str(hparams['data_dir'] / 'images'), transform=tfms)
    hparams['descriptor_fname'] = 'descriptors_food101'
    classes_to_load = None

elif hparams['dataset'] == 'pets':
    hparams['dataset_name'] = 'Oxford Pets'
    hparams['data_dir'] = pathlib.Path(PETS_DIR)
    hparams['analysis_fname'] = 'analysis_pets'
    dsclass = ImageFolder
    dataset = dsclass(str(hparams['data_dir'] / 'images'), transform=tfms)
    hparams['descriptor_fname'] = 'descriptors_pets'
    classes_to_load = None
    
elif hparams['dataset'] == 'dtd':
    hparams['dataset_name'] = 'DTD'
    hparams['data_dir'] = pathlib.Path(DTD_DIR)
    hparams['analysis_fname'] = 'analysis_dtd'
    dataset = ImageFolder(str(hparams['data_dir'] / 'images'), transform=tfms)
    hparams['descriptor_fname'] = 'descriptors_dtd'
    classes_to_load = None

if hparams['dataset'] != 'imagenetv2':
    dataset_classes = dataset.classes
else:
    dataset_classes = classes_to_load
    
hparams['descriptor_fname'] = f'./descriptors/{hparams['desc_type']}/{hparams['descriptor_fname']}'
hparams['descriptor_analysis_fname'] = './descriptor_analysis/descriptors_' + hparams['analysis_fname']
hparams['class_analysis_fname'] = './class_analysis/json/class_' + hparams['analysis_fname']
    
print("Creating descriptors from {}...".format(hparams['descriptor_fname'].split("/")[-1]))

gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load, cut_proportion=cut_proportion)
label_to_classname = list(gpt_descriptions.keys())

print("Creating descriptor frequencies...")
descriptors_freq = load_descriptors_frequency(hparams)

n_classes = len(list(gpt_descriptions.keys()))

total_descriptors_is = sum(descriptors_freq['freq_is'].values())
total_descriptors_contains = sum(descriptors_freq['freq_contains'].values())
frequency_proportion_is = {desc: freq/total_descriptors_is for desc, freq in descriptors_freq['freq_is'].items()}
frequency_proportion_contains = {desc: freq/total_descriptors_contains for desc, freq in descriptors_freq['freq_contains'].items()}

if similarity_penalty_config:
    class_similarity_dict = load_json(hparams['class_analysis_fname'] + '.json')
    class_list = compute_class_list(class_similarity_dict, sort_config=True)
    average_cosine_similarities = {k: v['average_cosine_similarity'] for k, v in class_similarity_dict.items()}

def compute_description_encodings(model, batch_size=32):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        encodings = []
        for i in range(0, len(v), batch_size):
            batch = v[i:i + batch_size]
            tokens = clip.tokenize(batch).to(hparams['device'])
            encodings.append(F.normalize(model.encode_text(tokens)).cpu())
        description_encodings[k] = torch.cat(encodings).to(hparams['device'])
    return description_encodings

def compute_label_encodings(model):
    label_encodings = F.normalize(model.encode_text(clip.tokenize([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(hparams['device'])))
    return label_encodings

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': 
        return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': 
        return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': 
        return similarity_matrix_chunk.mean(dim=1)
    else: 
        raise ValueError("Unknown aggregate_similarity")

def show_from_indices(indices, images, labels=None, predictions=None, predictions2 = None, n=None, image_description_similarity=None, image_labels_similarity=None):
    if indices is None or not indices:
        print("No indices provided")
        return
    
    if n is not None:
        indices = indices[:n]
    
    for index in indices:
        show_single_image(images[index])
        print(f"Index: {index}")
        if labels is not None:
            true_label = labels[index]
            true_label_name = label_to_classname[true_label]
            print(f"True label: {true_label_name}")
        if predictions is not None:
            predicted_label = predictions[index]
            predicted_label_name = label_to_classname[predicted_label]
            print(f"Predicted label (ours): {predicted_label_name}")
        if predictions2 is not None:
            predicted_label2 = predictions2[index]
            predicted_label_name2 = label_to_classname[predicted_label2]
            print(f"Predicted label 2 (CLIP): {predicted_label_name2}")
        
        print("\n")
        
        if image_labels_similarity is not None:
            if labels is not None:
                print(f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name: 
                    print("Predicted label (ours) matches true label")
                else: 
                    print(f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches true label")
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else: 
                    print(f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}")
        
            print("\n")
        
        if image_description_similarity is not None:
            if labels is not None:
                print_descriptor_similarity(image_description_similarity, index, true_label, true_label_name, "true")
                print("\n")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) same as true label")
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label, predicted_label_name, "descriptor")
                print("\n")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) same as true label")
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label2, predicted_label_name2, "CLIP")
            print("\n")

def print_descriptor_similarity(image_description_similarity, index, label, label_name, label_type="provided"):
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(f"Average:\t\t{100.*aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key = lambda x: x[1], reverse=True):
        k = unmodify_dict[label_name][k]
        print(f"{k}\t{100.*v}")
        
def print_max_descriptor_similarity(image_description_similarity, index, label, label_name):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")
    
def show_misclassified_images(images, labels, predictions, n=None, 
                              image_description_similarity=None, 
                              image_labels_similarity=None,
                              true_label_to_consider: int = None, 
                              predicted_label_to_consider: int = None):
    misclassified_indices = yield_misclassified_indices(images, labels=labels, predictions=predictions, true_label_to_consider=true_label_to_consider, predicted_label_to_consider=predicted_label_to_consider)
    if misclassified_indices is None: return
    show_from_indices(misclassified_indices, images, labels, predictions, 
                      n=n,
                      image_description_similarity=image_description_similarity, 
                      image_labels_similarity=image_labels_similarity)

def yield_misclassified_indices(images, labels, predictions, true_label_to_consider=None, predicted_label_to_consider=None):
    misclassified_indicators = (predictions.cpu() != labels.cpu())
    if true_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (labels.cpu() == true_label_to_consider)
    if predicted_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (predictions.cpu() == predicted_label_to_consider)
        
    if misclassified_indicators.sum() == 0:
        output_string = 'No misclassified images found'
        if true_label_to_consider is not None:
            output_string += f' with true label {label_to_classname[true_label_to_consider]}'
        if predicted_label_to_consider is not None:
            output_string += f' with predicted label {label_to_classname[predicted_label_to_consider]}'
        print(output_string + '.')
        return
    
    misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
    return misclassified_indices


from PIL import Image
def predict_and_show_explanations(images, model, labels=None, description_encodings=None, label_encodings=None, device=None):
    if isinstance(images, Image):
        images = tfms(images)
        
    if images.device != device:
        images = images.to(device)
        labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    n_classes = len(description_encodings)
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    for i, (k, v) in enumerate(description_encodings.items()):
        dot_product_matrix = image_encodings @ v.T
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
    cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    show_from_indices(torch.arange(images.shape[0]), images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)
