## Edit names of folders and files in the local EuroSAT dataset to match the names in the EuroSAT descriptor file
# EuroSAT dataset folder names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
# EuroSAT dataset file path = /home/luke/Documents/GitHub/data/EuroSAT/2750
# Example EuroSAT dataset filenames = AnnualCrop_100.jpg, Highway_54.jpg, SeaLake_123.jpg
# EuroSAT descriptor file folder names = ["annual crop land", "forest", "brushland or shrubland", "highway or road", "industrial buildings or commercial buildings", "pasture land", "permanent crop land", "residential buildings or homes or apartments", "river", "lake or sea"]
# EuroSAT descriptor file path = /home/luke/Documents/GitHub/DCLIP2/descriptors/gpt3/descriptors_eurosat.json

import json
import os
import shutil

# Define the paths
dataset_path = '/home/luke/Documents/GitHub/data/EuroSAT/2750'
descriptor_path = '/home/luke/Documents/GitHub/DCLIP2/descriptors/gpt3/descriptors_eurosat.json'

# Load the descriptor file
with open(descriptor_path, 'r') as file:
    descriptors = json.load(file)

# Define the mapping between the EuroSAT dataset folder names and the EuroSAT descriptor file folder names
folder_mapping = {
    'AnnualCrop': 'annual_crop_land',
    'Forest': 'forest',
    'HerbaceousVegetation': 'brushland_or_shrubland',
    'Highway': 'highway or road',
    'Industrial': 'industrial_buildings_or_commercial_buildings',
    'Pasture': 'pasture_land',
    'PermanentCrop': 'permanent_crop_land',
    'Residential': 'residential_buildings_or_homes_or_apartments',
    'River': 'river',
    'SeaLake': 'lake_or_sea'
}

# Rename the files in the EuroSAT dataset
# Iterate through the folders in the EuroSAT dataset
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder) # Get the path of the folder
    if os.path.isdir(folder_path): # Check if the path is a folder
        for file in os.listdir(folder_path): # Iterate through the files in the folder
            file_name, file_extension = os.path.splitext(file) # Split the file name and extension
            if file_extension == '.jpg': # Check if the file is a .jpg file
                new_file_name = f'{folder_mapping[folder]}_{file_name.split("_")[-1]}{file_extension}' # Create the new file name
                old_path = os.path.join(folder_path, file)
                new_path = os.path.join(folder_path, new_file_name)
                os.rename(old_path, new_path)
                print(f'Renamed {old_path} to {new_path}')

# Rename the folders in the EuroSAT dataset
for folder in os.listdir(dataset_path):
    if folder in folder_mapping:
        old_path = os.path.join(dataset_path, folder)
        new_path = os.path.join(dataset_path, folder_mapping[folder])
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')

# # Move the files to the correct folders
# for folder in os.listdir(dataset_path):
#     folder_path = os.path.join(dataset_path, folder)
#     if os.path.isdir(folder_path):
#         for file in os.listdir(folder_path):
#             old_path = os.path.join(folder_path, file)
#             new_path = os.path.join(dataset_path, folder_mapping[folder], file)
#             shutil.move(old_path, new_path)
#             print(f'Moved {old_path} to {new_path}')

# # Remove the empty folders
# for folder in os.listdir(dataset_path):
#     folder_path = os.path.join(dataset_path, folder)
#     if os.path.isdir(folder_path) and not os.listdir(folder_path):
#         os.rmdir(folder_path)
#         print(f'Removed {folder_path}')