import os
import cv2
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MedicalImageDatasetCleaner:
    def __init__(self, dataset_path, label_path, report_path, output_dir=None):
        self.dataset_path = Path(dataset_path)
        self.label_path = Path(label_path)
        self.report_path = Path(report_path)
        self.output_dir = Path(output_dir or dataset_path) / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.report = self._load_validation_report()
    
    def _load_validation_report(self):
        """Load validation report from JSON file."""
        with open(self.report_path, 'r') as f:
            return json.load(f)
    
    def _remove_corrupted_images(self):
        """Remove corrupted images and corresponding YOLO labels."""
        print("\n🗑 Removing corrupted images and labels...")
        removed, failed = [], []
        
        for img in self.report['corrupted_images']:
            img_path = Path(img['path'])
            label_path = self.label_path / (img_path.stem + '.txt')
            try:
                if img_path.exists():
                    img_path.unlink()
                    removed.append(str(img_path))
                    if label_path.exists():
                        label_path.unlink()
                else:
                    failed.append((str(img_path), "File not found"))
            except Exception as e:
                failed.append((str(img_path), str(e)))
        
        print(f"\n✅ Removed {len(removed)} images and labels")
        print(f"❌ Failed to remove {len(failed)} files")
        
        if failed:
            for path, error in failed:
                print(f"- {path}: {error}")
        
        print("\n🛠 Cleaning completed.")

def main():
    DATASET_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\img_dataset"
    LABEL_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\label_dataset"
    REPORT_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\processed\medical_validation_report.json"
    
    cleaner = MedicalImageDatasetCleaner(dataset_path=DATASET_PATH, label_path=LABEL_PATH, report_path=REPORT_PATH)
    cleaner._remove_corrupted_images()

if __name__ == "__main__":
    main()
