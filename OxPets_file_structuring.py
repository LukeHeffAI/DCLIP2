import os
import shutil
from pathlib import Path

# Paths
images_dir = Path("/home/luke/Documents/GitHub/data/Oxford_Pets/images")
annotations_path = Path("/home/luke/Documents/GitHub/data/Oxford_Pets/annotations/list.txt")


# Create folders and move images
def move_to_class_folders():
    # Create a dictionary to map image names to class directories
    class_dirs = {}

    # Read the annotations file
    with open(annotations_path, "r") as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or line.strip() == "":
                continue
            
            # Extract the image name and breed (class label)
            parts = line.strip().split()
            image_name, class_id, species, breed_id = parts[0], parts[1], parts[2], parts[3]
            
            # Determine breed name based on image naming convention
            breed_name = image_name.rsplit("_", 1)[0]  # Extract the breed name
            
            # Create class directory if it does not exist
            class_dir = images_dir / breed_name
            if breed_name not in class_dirs:
                class_dirs[breed_name] = class_dir
                os.makedirs(class_dir, exist_ok=True)

            # Move image to the corresponding class directory
            src_image_path = images_dir / f"{image_name}.jpg"
            dst_image_path = class_dir / f"{image_name}.jpg"
            if src_image_path.exists():
                shutil.move(str(src_image_path), str(dst_image_path))

def move_from_class_folders():
    '''
    This function moves all images back to the main images directory.
    '''
    for root, dirs, files in os.walk(images_dir):
        print(root, dirs, files)
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(images_dir, file)
            shutil.move(src, dst)
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

if __name__ == '__main__':
    move_to_class_folders()
    # move_from_class_folders()