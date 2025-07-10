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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Settings and Constants
# ---------------------------
NUM_CLASSES = 7
BATCH_SIZE = 8        # Use a small batch size to conserve VRAM
EPOCHS = 50           # Number of distillation training epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 3.0     # Higher temperature for softer teacher outputs
ALPHA = 0.9           # Weight for the distillation (KL) loss component
LEARNING_RATE = 5e-4   # Lower learning rate for stable training

# Paths for exported teacher models (adjust as needed)
MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
RESNET_PATH = os.path.join(MODEL_SAVE_PATH, 'ResNet50_best.pth')
DENSENET_PATH = os.path.join(MODEL_SAVE_PATH, 'DenseNet121_best.pth')
EFFICIENTNET_PATH = os.path.join(MODEL_SAVE_PATH, 'EfficientNetB0_best.pth')
INCEPTION_PATH = os.path.join(MODEL_SAVE_PATH, 'InceptionV3_best.pth')
VIT_PATH = os.path.join(MODEL_SAVE_PATH, 'ViT_B_16_best.pth')
MOBILENET_PATH = os.path.join(MODEL_SAVE_PATH, 'MobileNetV3_best.pth')

# CSV and image directories for training, validation, and testing
TRAIN_CSV = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\train_labels.csv"
TRAIN_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\agumented_resized"
VAL_CSV   = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\val\val_labels.csv"
VAL_DIR   = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\val\resized_images"
TEST_CSV  = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\test_labels.csv"
TEST_DIR  = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"

# ---------------------------
# Helper Function: Compute Specificity
# ---------------------------
def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specificities)

# ---------------------------
# Augmentation Pipelines
# ---------------------------
def create_augmentation_pipeline():
    """Online augmentation for training X-ray images."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(224, 224),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            interpolation=1,
            mask_interpolation=0,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
        ], p=0.2),
        A.CLAHE(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_validation_pipeline():
    """Simple pipeline for validation and test."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ---------------------------
# Dataset Definition for Distillation
# ---------------------------
class DistillationDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
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
        image = self.transform(image=image)['image']
        return image, label_idx

# ---------------------------
# Teacher Model Definitions
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
        feats = self.base_model(x)
        return self.classifier(feats)

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
        feats = self.base_model(x)
        return self.classifier(feats)

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
        feats = self.base_model(x)
        return self.classifier(feats)

class InceptionV3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(InceptionV3Model, self).__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.base_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True,
            init_weights=False
        )
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
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return self.classifier(outputs)

class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV3Model, self).__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Linear(num_features, num_classes)
        self.freeze_layers()
    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.classifier[3].parameters():
            param.requires_grad = True
    def forward(self, x):
        return self.base_model(x)

class ViTModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTModel, self).__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.base_model(x)

# ---------------------------
# Helper: Load Teacher Model
# ---------------------------
def load_teacher_model(model_class, model_path):
    model = model_class(NUM_CLASSES, pretrained=False)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    if model_class.__name__ in ["ViTModel", "InceptionV3Model"]:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------
# Build Teacher Ensemble
# ---------------------------
def build_teacher_ensemble():
    teachers = {
        'resnet': load_teacher_model(ResNet50Model, RESNET_PATH),
        'densenet': load_teacher_model(DenseNet121Model, DENSENET_PATH),
        'efficientnet': load_teacher_model(EfficientNetB0Model, EFFICIENTNET_PATH),
        'inception': load_teacher_model(InceptionV3Model, INCEPTION_PATH),
        'vit': load_teacher_model(ViTModel, VIT_PATH),
        'mobilenet': load_teacher_model(MobileNetV3Model, MOBILENET_PATH)
    }
    return teachers

def teacher_ensemble_predict(teachers, inputs):
    outputs = []
    with torch.no_grad():
        for name, model in teachers.items():
            out = model(inputs)
            outputs.append(F.softmax(out, dim=1))
    ensemble_output = torch.stack(outputs).mean(dim=0)
    return ensemble_output

# ---------------------------
# Distillation Loss Function
# ---------------------------
def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha):
    p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, targets)
    return alpha * kd_loss + (1 - alpha) * ce_loss

# ---------------------------
# Training Loop for Distillation (with Metrics)
# ---------------------------
def distillation_train(student, teachers, optimizer, dataloader, device, temperature, alpha):
    student.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_output = teacher_ensemble_predict(teachers, images)
        
        student_logits = student(images)
        loss = distillation_loss(student_logits, teacher_output, labels, temperature, alpha)
        loss.backward()
        optimizer.step()

        preds = student_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    acc = 100 * total_correct / total_samples
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    spec = compute_specificity(all_labels, all_preds)
    return avg_loss, acc, precision, recall, f1, spec

# ---------------------------
# Evaluation Function for Student Model
# ---------------------------
def evaluate_student(student, dataloader, device):
    student.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = student(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    acc = 100 * total_correct / total_samples
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    spec = compute_specificity(all_labels, all_preds)
    return avg_loss, acc, precision, recall, f1, spec, all_preds, all_labels

# ---------------------------
# Student Model Definition (Lightweight)
# ---------------------------
class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------------------
# Main Distillation Training and Evaluation Function
# ---------------------------
def main():
    # Create datasets and dataloaders for training, validation, and testing
    train_dataset = DistillationDataset(TRAIN_CSV, TRAIN_DIR, create_augmentation_pipeline())
    val_dataset   = DistillationDataset(VAL_CSV, VAL_DIR, create_validation_pipeline())
    test_dataset  = DistillationDataset(TEST_CSV, TEST_DIR, create_validation_pipeline())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Build teacher ensemble (teachers are frozen)
    teachers = build_teacher_ensemble()
    
    # Instantiate student model for distillation
    student = StudentModel(NUM_CLASSES)
    student.to(DEVICE)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.005)
    
    print("Starting Distillation Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_spec = distillation_train(
            student, teachers, optimizer, train_loader, DEVICE, TEMPERATURE, ALPHA
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_spec, _, _ = evaluate_student(student, val_loader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Prec: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, Spec: {train_spec:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Prec: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, Spec: {val_spec:.4f}")
    
    # Save the distilled student model
    torch.save(student.state_dict(), os.path.join(MODEL_SAVE_PATH, 'Student_Distilled.pth'))
    print("Student model saved.")
    
    # Final Evaluation on Test Set
    test_loss, test_acc, test_prec, test_rec, test_f1, test_spec, all_preds, all_labels = evaluate_student(student, test_loader, DEVICE)
    print("\n--- Test Set Evaluation ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Specificity: {test_spec:.4f}")
    
    target_names = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus', 'shoulder fracture', 'wrist positive', 'no fracture']
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nClassification Report on Test Set:")
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == '__main__':
    main()
