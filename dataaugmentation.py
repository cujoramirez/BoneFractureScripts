import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from pathlib import Path

def create_augmentation_pipeline():
    """Creates an augmentation pipeline suitable for bone fracture images"""
    return A.Compose([
        # Geometric transforms - keeping anatomical validity in mind
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=0,
            p=0.5
        ),
        
        # Photometric augmentations - preserving fracture visibility
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1
            ),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=2)
        ], p=0.5),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10, 30)),
            A.GaussianBlur(blur_limit=(3, 3)),
            A.MedianBlur(blur_limit=3)
        ], p=0.3),
        
        # Maintain final size
        A.Resize(256, 256, always_apply=True)
    ])

def graham_preprocess(image, sigma=25):
    """Applies Ben Graham's preprocessing technique"""
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def crop_black_borders(image, tol=7):
    """Removes black borders from images"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = gray.mean(axis=2) > tol
    else:
        mask = image > tol
    
    if mask.sum() == 0:  # Prevent empty images
        return image
        
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return image[rmin:rmax+1, cmin:cmax+1]

def augment_dataset(
    input_dir,
    output_dir,
    augmentations_per_image=3,
    apply_graham=True
):
    """
    Augments a dataset of bone fracture images
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        augmentations_per_image: Number of augmentations per original image
        apply_graham: Whether to apply Graham preprocessing
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize augmentation pipeline
    aug_pipeline = create_augmentation_pipeline()
    
    # Process each image in the input directory
    image_files = list(Path(input_dir).rglob("*.jpg")) + list(Path(input_dir).rglob("*.png"))
    
    print(f"Found {len(image_files)} images. Creating {augmentations_per_image} augmentations each...")
    
    for img_path in tqdm(image_files):
        # Read and preprocess original image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove black borders
        image = crop_black_borders(image)
        
        # Apply Graham preprocessing if enabled
        if apply_graham:
            image = graham_preprocess(image)
        
        # Save preprocessed original
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        
        # Generate augmentations
        for i in range(augmentations_per_image):
            # Apply augmentation pipeline
            augmented = aug_pipeline(image=image)['image']
            
            # Create augmented image filename
            base_name = img_path.stem
            extension = img_path.suffix
            aug_name = f"{base_name}_aug_{i+1}{extension}"
            aug_path = Path(output_dir) / aug_name
            
            # Save augmented image
            cv2.imwrite(
                str(aug_path),
                cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            )

if __name__ == "__main__":
    # Example usage
    augment_dataset(
        input_dir=r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\resized_imagesV3",
        output_dir=r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\agumented_resizedV3",
        augmentations_per_image=3,
        apply_graham=True
    )