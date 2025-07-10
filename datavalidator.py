import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import json
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from skimage.metrics import structural_similarity as ssim

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MedicalImageDatasetValidator:
    def __init__(self, dataset_path, label_csv=None, output_dir=None):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir or dataset_path) / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Medical imaging specific thresholds
        self.quality_thresholds = {
            'min_entropy': 1.5,          # Reduced for X-ray characteristics
            'max_uniform_ratio': 0.995,  # Increased tolerance
            'min_std_dev': 5.0,          # Reduced for medical imaging
            'min_resolution': (224, 224) # Match model input size
        }
        
        # Reporting
        self.report = {
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': [],
            'resolution_issues': [],
            'low_quality_images': [],
            'duplicates': [],
            'label_issues': []
        }

    def _calculate_color_entropy(self, image):
        """Calculate entropy for color images using YUV color space"""
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        channels = [yuv[..., i] for i in range(3)]
        return np.mean([self._channel_entropy(ch) for ch in channels])

    def _channel_entropy(self, channel):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / channel.size
        return -np.sum([p * np.log2(p) for p in hist if p > 0])

    def verify_image_integrity(self, image_path):
        """Enhanced medical image validation with GPU support"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'resolution': None,
            'color_channels': None
        }
        
        try:
            # Try multiple image loading methods
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError("OpenCV failed to read image")
            except:
                img = np.array(Image.open(image_path).convert('RGB'))
                
            # Validate basic properties
            if img.size == 0:
                validation_result['issues'].append("Zero-size image")
                validation_result['is_valid'] = False
                return validation_result
                
            # Check resolution
            h, w = img.shape[:2]
            validation_result['resolution'] = (w, h)
            if min(h, w) < min(self.quality_thresholds['min_resolution']):
                validation_result['issues'].append(f"Low resolution: {w}x{h}")
                validation_result['is_valid'] = False

            # Check color channels
            validation_result['color_channels'] = img.shape[2] if len(img.shape) == 3 else 1
            if validation_result['color_channels'] not in [1, 3]:
                validation_result['issues'].append(f"Invalid channel count: {validation_result['color_channels']}")
                validation_result['is_valid'] = False

            # GPU-based validation
            if self.device.type == 'cuda':
                try:
                    tensor_img = self.tensor_transform(Image.fromarray(img)).to(self.device)
                    if torch.isnan(tensor_img).any():
                        validation_result['issues'].append("NaN values in tensor conversion")
                        validation_result['is_valid'] = False
                except Exception as e:
                    validation_result['issues'].append(f"GPU processing failed: {str(e)}")
                    validation_result['is_valid'] = False

            # Medical image specific quality checks
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            std_dev = np.std(gray)
            entropy = self._calculate_color_entropy(img) if len(img.shape) == 3 else self._channel_entropy(gray)
            unique_ratio = len(np.unique(gray)) / gray.size
            
            if entropy < self.quality_thresholds['min_entropy']:
                validation_result['issues'].append(f"Low entropy: {entropy:.2f}")
                validation_result['is_valid'] = False
                
            if unique_ratio > self.quality_thresholds['max_uniform_ratio']:
                validation_result['issues'].append(f"High uniformity: {unique_ratio:.2%}")
                validation_result['is_valid'] = False
                
            if std_dev < self.quality_thresholds['min_std_dev']:
                validation_result['issues'].append(f"Low std dev: {std_dev:.2f}")
                validation_result['is_valid'] = False

            return validation_result
            
        except Exception as e:
            validation_result['issues'].append(f"Critical error: {str(e)}")
            validation_result['is_valid'] = False
            return validation_result

    def detect_duplicates(self, threshold=0.95):
        """Perceptual duplicate detection using SSIM"""
        duplicates = []
        image_files = list(self.dataset_path.glob('*.[jJ][pP][eE]?[gG]'))
        
        for i in range(len(image_files)):
            try:
                img1 = cv2.imread(str(image_files[i]))
                if img1 is None: continue
                
                for j in range(i+1, len(image_files)):
                    img2 = cv2.imread(str(image_files[j]))
                    if img2 is None: continue
                    
                    if img1.shape != img2.shape:
                        continue
                        
                    # Resize for efficiency
                    img1_resized = cv2.resize(img1, (128, 128))
                    img2_resized = cv2.resize(img2, (128, 128))
                    
                    # Calculate SSIM
                    similarity = ssim(img1_resized, img2_resized, 
                                    multichannel=True, 
                                    channel_axis=2,
                                    data_range=img2_resized.max() - img2_resized.min())
                    
                    if similarity > threshold:
                        duplicates.append((str(image_files[i]), str(image_files[j])))
                        
            except Exception as e:
                print(f"Duplicate check error: {e}")
                
        return duplicates

    def validate_dataset(self):
        """Comprehensive validation pipeline"""
        print(f"\n🔍 Starting Medical Dataset Validation on {self.device}...")
        
        # Process all image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(self.dataset_path.glob(ext))
            
        for img_path in image_paths:
            self.report['total_images'] += 1
            result = self.verify_image_integrity(img_path)
            
            if result['is_valid']:
                self.report['valid_images'] += 1
            else:
                record = {
                    'path': str(img_path),
                    'issues': result['issues'],
                    'resolution': result['resolution'],
                    'channels': result['color_channels']
                }
                if 'Low entropy' in result['issues'] or 'Low std dev' in result['issues']:
                    self.report['low_quality_images'].append(record)
                else:
                    self.report['corrupted_images'].append(record)
        
        # Enhanced duplicate detection
        self.report['duplicates'] = self.detect_duplicates()
        
        # Generate report
        self._generate_report()
        return self.report

    def _generate_report(self):
        """Enhanced reporting with statistics"""
        report_path = self.output_dir / 'medical_validation_report.json'
        
        # Calculate quality statistics
        stats = {
            'resolution_distribution': {},
            'channel_distribution': {},
            'entropy_stats': {'min': float('inf'), 'max': -float('inf'), 'sum': 0},
            'std_dev_stats': {'min': float('inf'), 'max': -float('inf'), 'sum': 0}
        }
        
        for item in self.report['low_quality_images'] + self.report['corrupted_images']:
            res = f"{item['resolution'][0]}x{item['resolution'][1]}"
            stats['resolution_distribution'][res] = stats['resolution_distribution'].get(res, 0) + 1
            
            channels = str(item['channels'])
            stats['channel_distribution'][channels] = stats['channel_distribution'].get(channels, 0) + 1
        
        self.report['statistics'] = stats
        
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=4)
            
        print("\n📊 Medical Dataset Validation Report")
        print(f"Total Images: {self.report['total_images']}")
        print(f"Valid Images: {self.report['valid_images']} ({self.report['valid_images']/self.report['total_images']:.1%})")
        print(f"Corrupted Images: {len(self.report['corrupted_images'])}")
        print(f"Low Quality Images: {len(self.report['low_quality_images'])}")
        print(f"Duplicate Groups: {len(self.report['duplicates'])}")
        print(f"Common Resolutions: {dict(sorted(stats['resolution_distribution'].items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"Channel Distribution: {stats['channel_distribution']}")

def main():
    DATASET_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\agumented_resized"
    
    validator = MedicalImageDatasetValidator(
        dataset_path=DATASET_PATH,
        output_dir=r"D:\BoneFracture\Dataset\BoneFractureYolo8\results"
    )
    
    report = validator.validate_dataset()

if __name__ == "__main__":
    main()