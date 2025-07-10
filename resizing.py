import os
from PIL import Image

def resize_image(img_path, target_size=(256, 256)):
    """
    Opens an image and resizes it to the specified dimensions.
    """
    with Image.open(img_path) as img:
        img = img.resize(target_size, Image.ANTIALIAS)
        return img

def resize_dataset(source_folder, target_folder, target_size=(256, 256)):
    """
    Resizes all images in source_folder and saves them to target_folder.
    """
    os.makedirs(target_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(image_extensions):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(target_folder, filename)

            try:
                resized_img = resize_image(src_path, target_size=target_size)
                resized_img.save(dst_path)
                print(f"Resized {filename} -> {dst_path}")
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    # Define dataset directories (adjust paths as needed)
    base_dataset_dir = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2"

    # Source folders (original images)
    train_source = os.path.join(base_dataset_dir, "train", "images")
    val_source   = os.path.join(base_dataset_dir, "val", "images")
    test_source  = os.path.join(base_dataset_dir, "test", "images")

    # Target folders (resized images)
    train_target = os.path.join(base_dataset_dir, "train", "resized_imagesV3")
    val_target   = os.path.join(base_dataset_dir, "val", "resized_imagesV3")
    test_target  = os.path.join(base_dataset_dir, "test", "resized_imagesV3")

    # Process each dataset split
    print("Resizing training images...")
    resize_dataset(train_source, train_target, target_size=(299, 299))

    print("Resizing validation images...")
    resize_dataset(val_source, val_target, target_size=(299, 299))

    print("Resizing test images...")
    resize_dataset(test_source, test_target, target_size=(299, 299))

    print("All images resized successfully!")
