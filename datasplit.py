import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuration
# ---------------------------
# Source directories for images and labels
source_image_dir = r"D:\BoneFracture\Dataset\BoneFractureYolo8\img_dataset"
source_label_dir = r"D:\BoneFracture\Dataset\BoneFractureYolo8\label_dataset"

# Destination directory for the new dataset
destination_dir = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2"

# Define file extensions for your images
image_extensions = ('.jpg', '.jpeg', '.png')

# Set new split ratios (80% train, 10% validation, 10% test)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)

# ---------------------------
# Gather all image files from the source image directory
# ---------------------------
all_images = [f for f in os.listdir(source_image_dir) if f.lower().endswith(image_extensions)]
print(f"Found {len(all_images)} images in the source image directory.")

# ---------------------------
# Split the dataset into train, validation, and test sets
# ---------------------------
# First, split off the test set
train_val, test = train_test_split(all_images, test_size=test_ratio, random_state=random_seed)
# Then split the remaining into train and validation.
# Adjust the validation fraction relative to the remaining images.
val_adjusted = val_ratio / (train_ratio + val_ratio)
train, val = train_test_split(train_val, test_size=val_adjusted, random_state=random_seed)

print(f"New split: {len(train)} training, {len(val)} validation, {len(test)} test images.")

# ---------------------------
# Function to copy images and their corresponding label files
# ---------------------------
def copy_files(file_list, subset):
    """Copy image files and corresponding label files to the destination subset folder."""
    subset_img_dir = os.path.join(destination_dir, subset, "images")
    subset_label_dir = os.path.join(destination_dir, subset, "labels")
    os.makedirs(subset_img_dir, exist_ok=True)
    os.makedirs(subset_label_dir, exist_ok=True)
    
    for filename in file_list:
        # Copy the image file
        src_img_path = os.path.join(source_image_dir, filename)
        dst_img_path = os.path.join(subset_img_dir, filename)
        shutil.copy(src_img_path, dst_img_path)
        
        # Copy the corresponding label file (same basename with .txt extension)
        base_name = os.path.splitext(filename)[0]
        label_filename = base_name + ".txt"
        src_label_path = os.path.join(source_label_dir, label_filename)
        dst_label_path = os.path.join(subset_label_dir, label_filename)
        
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            # If the label file does not exist, create an empty file (indicating no fracture)
            with open(dst_label_path, "w") as f:
                pass

# ---------------------------
# Copy files for each subset
# ---------------------------
copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

# ---------------------------
# Create a CSV file mapping images to labels
# ---------------------------
# For each image, the CSV will have columns for subset, filename, and label.
# For the label: if the corresponding .txt file is empty, record "no_fracture".
# Otherwise, assume the first token in the file represents the fracture class (e.g., 0-6).
data = []
for subset in ["train", "val", "test"]:
    subset_img_dir = os.path.join(destination_dir, subset, "images")
    subset_label_dir = os.path.join(destination_dir, subset, "labels")
    
    for filename in os.listdir(subset_img_dir):
        if not filename.lower().endswith(image_extensions):
            continue
        
        # Get the corresponding label file
        base_name = os.path.splitext(filename)[0]
        label_filename = base_name + ".txt"
        label_path = os.path.join(subset_label_dir, label_filename)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                content = f.read().strip()
        else:
            content = ""
        
        # Determine the label: if empty, set as "no_fracture"
        label = "no_fracture" if content == "" else content.split()[0]
        
        data.append({
            "subset": subset,
            "filename": filename,
            "label": label
        })

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(data)
csv_output_path = os.path.join(destination_dir, "dataset_labels.csv")
df.to_csv(csv_output_path, index=False)
print("CSV file with labels saved to:", csv_output_path)
