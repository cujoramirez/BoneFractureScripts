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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------------------
# Settings and Constants
# ---------------------------
NUM_CLASSES = 7
BATCH_SIZE = 4
EPOCHS = 1  # For demonstration; increase as needed.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters for mutual learning
lambda_mutual = 1.0    # Weight of the mutual loss term
temperature = 1.0      # Temperature for softening the predictions

# Paths for training data (adjust these paths accordingly)
TRAIN_CSV = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\train_labels.csv"
TRAIN_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\agumented_resized"

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
# Custom Dataset for Mutual Learning
# ---------------------------
class MutualLearningDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        # Use the same label mapping as in training
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
        # Produce two versions for different models
        img_224 = transform_224(image=image)['image']
        img_299 = transform_299(image=image)['image']
        return {'img_224': img_224, 'img_299': img_299, 'label': label_idx}

# ---------------------------
# Model Definitions (Placeholders)
# ---------------------------
# These should match your training model definitions.
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
        # Set aux_logits=True to match the pretrained weights.
        self.base_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True,
            init_weights=False
        )
        # Get number of features from the primary classifier.
        num_features = self.base_model.fc.in_features
        # Replace the final fully connected layer with an Identity so we can attach our own head.
        self.base_model.fc = nn.Identity()
        # Define your custom classifier head.
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
        # When aux_logits=True, the forward returns a tuple: (primary, aux)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # use only the main output
        return self.classifier(outputs)


class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV3Model, self).__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Linear(num_features, num_classes)
        # Freeze all parameters except the new head (as done in training)
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
# Helper: Load Model
# ---------------------------
def load_model(model_class, model_path):
    model = model_class(NUM_CLASSES, pretrained=False)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    # For ViT, allow strict=False
    if model_class.__name__ == "ViTModel":
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.train()  # Ensure training mode for mutual learning
    return model

# ---------------------------
# Mutual Learning Training Loop
# ---------------------------
def mutual_learning_train(models, optimizers, dataloader, device, lambda_mutual, temperature):
    """
    models: dictionary of models (keys: 'resnet', 'densenet', etc.)
    optimizers: dictionary of optimizers corresponding to each model.
    dataloader: training DataLoader.
    Returns: average loss and accuracy for the ensemble.
    """
    # Set all models to train
    for model in models.values():
        model.train()
    
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch in tqdm(dataloader, desc="Mutual Learning Training"):
        # Get labels (common to all)
        labels = batch['label'].to(device)
        batch_size = labels.size(0)
        # For each model, select the appropriate input: Inception uses 'img_299', others use 'img_224'
        outputs = {}
        for name, model in models.items():
            if name == "inception":
                inputs = batch['img_299'].to(device)
            else:
                inputs = batch['img_224'].to(device)
            outputs[name] = model(inputs)
        
        # Compute cross-entropy loss for each model
        ce_losses = {}
        for name, output in outputs.items():
            ce_losses[name] = F.cross_entropy(output, labels)
        
        # Compute mutual learning loss for each model
        mutual_losses = {}
        model_names = list(models.keys())
        for i, name_i in enumerate(model_names):
            out_i = outputs[name_i] / temperature
            log_p_i = F.log_softmax(out_i, dim=1)
            mutual_loss = 0.0
            count = 0
            for j, name_j in enumerate(model_names):
                if i == j:
                    continue
                out_j = outputs[name_j] / temperature
                p_j = F.softmax(out_j, dim=1)
                # KL divergence: KL(p_j || p_i)
                kl = F.kl_div(log_p_i, p_j, reduction='batchmean')
                mutual_loss += kl
                count += 1
            mutual_losses[name_i] = mutual_loss / count
        
        # Total loss per model: cross-entropy + lambda_mutual * mutual loss
        total_losses = {}
        for name in model_names:
            total_losses[name] = ce_losses[name] + lambda_mutual * mutual_losses[name]
        
        # Zero gradients for all optimizers
        for opt in optimizers.values():
            opt.zero_grad()
        # Backpropagate for each model; use retain_graph=True for all but last
        for idx, name in enumerate(model_names):
            if idx < len(model_names) - 1:
                total_losses[name].backward(retain_graph=True)
            else:
                total_losses[name].backward()
        # Step all optimizers
        for opt in optimizers.values():
            opt.step()
        
        # For monitoring, we can compute the ensemble output (average logits) and use it for accuracy.
        ensemble_logits = sum(outputs.values()) / len(outputs)
        preds = ensemble_logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += sum(loss.item() for loss in total_losses.values()) / len(total_losses) * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy

# ---------------------------
# Main Mutual Learning Training Function
# ---------------------------
def main():
    # Create dataset and dataloader for training mutual learning
    train_dataset = MutualLearningDataset(TRAIN_CSV, TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Instantiate all 6 models for mutual learning
    models = {
        'resnet': ResNet50Model(NUM_CLASSES, pretrained=True),
        'densenet': DenseNet121Model(NUM_CLASSES, pretrained=True),
        'efficientnet': EfficientNetB0Model(NUM_CLASSES, pretrained=True),
        'inception': InceptionV3Model(NUM_CLASSES, pretrained=True),
        'vit': ViTModel(NUM_CLASSES, pretrained=True),
        'mobilenet': MobileNetV3Model(NUM_CLASSES, pretrained=True)
    }
    # Put models to device and set them to train mode
    for model in models.values():
        model.to(DEVICE)
        model.train()
    
    # Create separate optimizers for each model (you can tune learning rates)
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
        for name, model in models.items()
    }
    
    # Mutual learning training loop
    for epoch in range(EPOCHS):
        avg_loss, acc = mutual_learning_train(models, optimizers, train_loader, DEVICE, lambda_mutual, temperature)
        print(f"Epoch {epoch+1}/{EPOCHS}: Avg Loss = {avg_loss:.4f}, Ensemble Accuracy = {acc:.2f}%")
    
    # (Optionally, save the final mutual learned models or perform evaluation on a validation/test set.)
    # For demonstration, we print a message.
    print("Mutual learning training complete.")

if __name__ == '__main__':
    main()
