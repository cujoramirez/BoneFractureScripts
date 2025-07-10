import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Settings and Constants
# ---------------------------
NUM_CLASSES = 7
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model file paths
MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
RESNET_PATH = os.path.join(MODEL_SAVE_PATH, 'ResNet50_best.pth')
DENSENET_PATH = os.path.join(MODEL_SAVE_PATH, 'DenseNet121_best.pth')
EFFICIENTNET_PATH = os.path.join(MODEL_SAVE_PATH, 'EfficientNetB0_best.pth')
INCEPTION_PATH = os.path.join(MODEL_SAVE_PATH, 'InceptionV3_best.pth')
VIT_PATH = os.path.join(MODEL_SAVE_PATH, 'ViT_B_16_best.pth')
MOBILENET_PATH = os.path.join(MODEL_SAVE_PATH, 'MobileNetV3_best.pth')

# Test data paths
TEST_CSV = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\test_labels.csv"
TEST_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"

# ---------------------------
# Data Transforms
# ---------------------------
# For models expecting 224x224 images (ResNet, DenseNet, EfficientNet, ViT, MobileNet)
transform_224 = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
# For InceptionV3 (expects 299x299)
transform_299 = A.Compose([
    A.Resize(299, 299),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ---------------------------
# Custom Dataset for Ensemble Inference
# ---------------------------
class BoneFractureEnsembleDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.label_map = {'0': 0, '1': 1, '2': 2, '4': 3, '5': 4, '6': 5, 'no_fracture': 6}
        self.label_map.update({0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5})
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = row['label']
        if isinstance(label, (int, float)):
            label = str(int(label))
        label_idx = self.label_map.get(label, self.label_map['no_fracture'])
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_224 = transform_224(image=image)['image']
        image_299 = transform_299(image=image)['image']
        return {'img_224': image_224, 'img_299': image_299, 'label': label_idx}

# ---------------------------
# Model Definitions
# ---------------------------
class ResNet50Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50Model, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

class DenseNet121Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet121Model, self).__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        self.base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

class InceptionV3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(InceptionV3Model, self).__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.base_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
                                       aux_logits=False, init_weights=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        outputs = self.base_model(x)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        return self.classifier(outputs)

# --- Updated MobileNetV3 Model (matches your training script) ---
class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV3Model, self).__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace the final classifier layer (index 3) with a new Linear layer
        num_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Linear(num_features, num_classes)
        self.freeze_layers()
    
    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.classifier[3].parameters():
            param.requires_grad = True

    def unfreeze_last_two_blocks(self):
        # Unfreeze the last two blocks of the features (assumed to be the last 2 elements)
        for block in self.base_model.features[-2:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)

# --- Updated ViT Model (matches your training export) ---
class ViTModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTModel, self).__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(num_features, num_classes)
        # Note: In your training script, you may have computed weights internally.
    def forward(self, x):
        return self.base_model(x)

# ---------------------------
# Helper: Load Model with Optional strict=False for ViT
# ---------------------------
def load_model(model_class, model_path):
    model = model_class(NUM_CLASSES, pretrained=False)
    # Load weights only for safety; for ViT, use strict=False to ignore extra keys.
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    if model_class.__name__ == "ViTModel":
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------
# Ensemble Model Definition
# ---------------------------
class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        """
        models: dictionary with keys: 'resnet', 'densenet', 'efficientnet', 
                'inception', 'vit', 'mobilenet'
        weights: dictionary with relative weights.
                 Default weights (based on F1 scores):
                   DenseNet121: 1.00, ResNet50: 0.97, EfficientNetB0: 0.87,
                   InceptionV3: 0.84, ViT: 0.99, MobileNetV3: 0.90.
        """
        super(EnsembleModel, self).__init__()
        self.models = models
        if weights is None:
            self.weights = {
                'densenet': 1.00,
                'resnet': 0.97,
                'efficientnet': 0.87,
                'inception': 0.84,
                'vit': 0.99,
                'mobilenet': 0.90
            }
        else:
            self.weights = weights

    def forward(self, x_224, x_299):
        out_resnet = self.models['resnet'](x_224)
        out_densenet = self.models['densenet'](x_224)
        out_efficientnet = self.models['efficientnet'](x_224)
        out_vit = self.models['vit'](x_224)
        out_mobilenet = self.models['mobilenet'](x_224)
        out_inception = self.models['inception'](x_299)
        
        prob_resnet = F.softmax(out_resnet, dim=1)
        prob_densenet = F.softmax(out_densenet, dim=1)
        prob_efficientnet = F.softmax(out_efficientnet, dim=1)
        prob_vit = F.softmax(out_vit, dim=1)
        prob_mobilenet = F.softmax(out_mobilenet, dim=1)
        prob_inception = F.softmax(out_inception, dim=1)
        
        weighted_sum = (self.weights['resnet'] * prob_resnet +
                        self.weights['densenet'] * prob_densenet +
                        self.weights['efficientnet'] * prob_efficientnet +
                        self.weights['vit'] * prob_vit +
                        self.weights['mobilenet'] * prob_mobilenet +
                        self.weights['inception'] * prob_inception)
        total_weight = (self.weights['resnet'] +
                        self.weights['densenet'] +
                        self.weights['efficientnet'] +
                        self.weights['vit'] +
                        self.weights['mobilenet'] +
                        self.weights['inception'])
        ensemble_prob = weighted_sum / total_weight
        return ensemble_prob

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_ensemble(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            x_224 = batch['img_224'].to(device)
            x_299 = batch['img_299'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(x_224, x_299)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    try:
        y_true_onehot = pd.get_dummies(all_labels).values
        y_pred_onehot = pd.get_dummies(all_preds).values
        roc_auc = roc_auc_score(y_true_onehot, y_pred_onehot, average='weighted', multi_class='ovr')
    except Exception:
        roc_auc = None

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'all_preds': np.array(all_preds),
        'all_labels': np.array(all_labels)
    }
    return metrics

# ---------------------------
# Confusion Matrix Plotting
# ---------------------------
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ---------------------------
# Main Ensemble Inference
# ---------------------------
def main():
    test_dataset = BoneFractureEnsembleDataset(TEST_CSV, TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load individual models
    resnet_model = load_model(ResNet50Model, RESNET_PATH)
    densenet_model = load_model(DenseNet121Model, DENSENET_PATH)
    efficientnet_model = load_model(EfficientNetB0Model, EFFICIENTNET_PATH)
    inception_model = load_model(InceptionV3Model, INCEPTION_PATH)
    vit_model = load_model(ViTModel, VIT_PATH)
    mobilenet_model = load_model(MobileNetV3Model, MOBILENET_PATH)
    
    models = {
        'resnet': resnet_model,
        'densenet': densenet_model,
        'efficientnet': efficientnet_model,
        'inception': inception_model,
        'vit': vit_model,
        'mobilenet': mobilenet_model
    }
    weights = {
        'densenet': 1.00,    # F1 ~0.8407
        'resnet': 0.97,      # F1 ~0.8145
        'efficientnet': 0.87,# F1 ~0.7268
        'inception': 0.84,   # F1 ~0.7059
        'vit': 0.99,         # F1 ~0.8410 (or in the 0.83-0.84 range)
        'mobilenet': 0.90    # F1 ~0.7595
    }
    
    ensemble_model = EnsembleModel(models, weights)
    ensemble_model.to(DEVICE)
    
    metrics = evaluate_ensemble(ensemble_model, test_loader, DEVICE)
    print("\n--- Ensemble Test Metrics ---")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        print("Test ROC AUC: Could not be computed")
    
    target_names = ['elbow positive', 'fingers positive', 'forearm fracture',
                    'humerus', 'shoulder fracture', 'wrist positive', 'no fracture']
    report = classification_report(metrics['all_labels'], metrics['all_preds'], target_names=target_names)
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(metrics['all_labels'], metrics['all_preds'], target_names)

if __name__ == '__main__':
    main()
