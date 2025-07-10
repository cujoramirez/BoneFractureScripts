import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.swa_utils import AveragedModel, SWALR

# ---------------------------
# Constants and Settings
# ---------------------------
NUM_CLASSES = 7
MODEL_INPUT_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 32
NUM_EPOCHS = 50
BASE_LEARNING_RATE = 1e-5  # Base LR (will be overridden in phases)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
TRAIN_CSV = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\train_labels.csv"
TRAIN_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train\agumented_resized"
VAL_CSV = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\val\val_labels.csv"
VAL_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\val\resized_images"
TEST_CSV = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\test_labels.csv"
TEST_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"

# Path to save models
MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Cross-validation and enhancement settings
CROSS_VALIDATE = True
NUM_FOLDS = 5
EARLY_STOP_PATIENCE = 3  # epochs with no improvement in validation F1
USE_LR_FINDER = False   # Disabled by default; enable manually if desired
USE_SWA = True          # Use SWA for better generalization

# Enable cudnn benchmark for constant input sizes
torch.backends.cudnn.benchmark = True

# ---------------------------
# Data Augmentation Pipelines
# ---------------------------
def create_augmentation_pipeline():
    return A.Compose([
        A.RandomResizedCrop(
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
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
    return A.Compose([
        A.Resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ---------------------------
# Dataset Definition
# ---------------------------
class BoneFractureDataset(Dataset):
    def __init__(self, csv_path=None, img_dir=None, transform=None, class_weights=None, df=None):
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.class_weights = class_weights
        self.label_map = {'0': 0, '1': 1, '2': 2, '4': 3, '5': 4, '6': 5, 'no_fracture': 6}
        self.label_map.update({0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5})
        self._validate_labels()
        
    def _validate_labels(self):
        invalid_labels = []
        for idx, row in self.df.iterrows():
            label = row['label']
            if isinstance(label, (int, float)):
                label = str(int(label))
            if label not in self.label_map:
                invalid_labels.append((idx, label))
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        label = self.df.iloc[idx]['label']
        if isinstance(label, (int, float)):
            label = str(int(label))
        label_idx = self.label_map.get(label, self.label_map['no_fracture'])
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            return image, label_idx
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), -1

# ---------------------------
# Metrics Tracker with Additional Metrics
# ---------------------------
class MetricsTracker:
    def __init__(self):
        self.running_loss = 0
        self.running_corrects = 0
        self.total_samples = 0
        self.all_logits = []  # Store logits
        self.all_labels = []  # Store ground truth labels
        
    def update(self, outputs, labels, loss, batch_size):
        self.running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        self.running_corrects += torch.sum(preds == labels).item()
        self.total_samples += batch_size
        self.all_logits.extend(outputs.cpu().detach().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
    def get_metrics(self):
        logits = np.array(self.all_logits)
        # Stabilize softmax computation using log-sum-exp trick
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Debug: ensure each row sums to ~1
        # print("Row sums:", np.sum(probs, axis=1))
        preds_array = np.argmax(probs, axis=1)
        # One-hot encode true labels for roc_auc_score
        y_true_onehot = label_binarize(np.array(self.all_labels), classes=list(range(NUM_CLASSES)))
        try:
            roc_auc = roc_auc_score(y_true_onehot, probs, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"ROC AUC computation error: {e}")
            roc_auc = 0.0
        precision = precision_score(self.all_labels, preds_array, average='weighted', zero_division=0)
        recall = recall_score(self.all_labels, preds_array, average='weighted', zero_division=0)
        f1 = f1_score(self.all_labels, preds_array, average='weighted', zero_division=0)
        cm = confusion_matrix(self.all_labels, preds_array, labels=list(range(NUM_CLASSES)))
        spec_list = []
        for i in range(NUM_CLASSES):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            spec_list.append(spec)
        specificity = np.mean(spec_list)
        return {
            'loss': self.running_loss / self.total_samples,
            'accuracy': 100 * self.running_corrects / self.total_samples,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'all_preds': preds_array,
            'all_labels': np.array(self.all_labels)
        }

# ---------------------------
# Transfer Learning Model (ResNet50)
# ---------------------------
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
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
        self.freeze_layers()
    
    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def unfreeze_layer_group(self, layer_name):
        if hasattr(self.base_model, layer_name):
            for param in getattr(self.base_model, layer_name).parameters():
                param.requires_grad = True
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

# ---------------------------
# Trainer for Progressive Transfer Learning
# ---------------------------
class TransferLearningTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler()
        # Extended history to store extra metrics:
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_roc_auc': [], 'train_precision': [], 'train_recall': [], 'train_specificity': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_roc_auc': [], 'val_precision': [], 'val_recall': [], 'val_specificity': [], 'val_f1': []
        }
        if USE_SWA:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = None
    
    def setup_optimization(self, phase):
        if phase == 'initial':
            trainable_params = self.model.classifier.parameters()
            self.optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.005)
        elif phase == 'fine_tuning':
            classifier_params = {'params': self.model.classifier.parameters(), 'lr': 5e-4}
            base_params = {'params': filter(lambda p: p.requires_grad, self.model.base_model.parameters()), 'lr': 1e-5}
            self.optimizer = optim.AdamW([classifier_params, base_params], weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        if USE_SWA:
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=1e-4)
    
    def setup_optimization_phase3(self):
        classifier_params = {'params': self.model.classifier.parameters(), 'lr': 1e-5}
        base_params = {'params': filter(lambda p: p.requires_grad, self.model.base_model.parameters()), 'lr': 1e-5}
        self.optimizer = optim.AdamW([classifier_params, base_params], weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        if USE_SWA:
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=1e-4)
        
    def lr_finder(self, init_value=1e-6, final_value=10, beta=0.98, num_iters=100):
        import matplotlib.pyplot as plt
        self.model.train()
        lr = init_value
        mult = (final_value / init_value) ** (1/num_iters)
        losses = []
        lrs = []
        avg_loss = 0.0
        best_loss = float('inf')
        iterator = iter(self.train_loader)
        for i in range(num_iters):
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                inputs, labels = next(iterator)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**(i+1))
            losses.append(smoothed_loss)
            lrs.append(lr)
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            if smoothed_loss > 4 * best_loss:
                break
            self.scaler.scale(loss).backward()
            self.optimizer.step()
            try:
                self.scaler.update()
            except AssertionError:
                pass
            lr *= mult
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.show()
        
    def train_epoch(self):
        self.model.train()
        metrics = MetricsTracker()
        for inputs, labels in tqdm(self.train_loader, desc='Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            metrics.update(outputs, labels, loss, inputs.size(0))
        self.scheduler.step()
        if USE_SWA:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        return metrics.get_metrics()
    
    def train_phases(self, num_epochs, patience=EARLY_STOP_PATIENCE):
        best_val_f1 = 0
        epochs_without_improvement = 0
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.update_history(train_metrics, val_metrics)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Training - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}, "
                  f"AUC: {train_metrics['roc_auc']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, "
                  f"Recall (Sensitivity): {train_metrics['recall']:.4f}, "
                  f"Specificity: {train_metrics['specificity']:.4f}, "
                  f"F1-Score: {train_metrics['f1_score']:.4f}")
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['roc_auc']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall (Sensitivity): {val_metrics['recall']:.4f}, "
                  f"Specificity: {val_metrics['specificity']:.4f}, "
                  f"F1-Score: {val_metrics['f1_score']:.4f}")
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ResNet50_best.pth'))
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs with no improvement in validation F1.")
                break
        return self.history
    
    def progressive_training(self):
        print("Phase 1: Training classifier layers only...")
        self.setup_optimization('initial')
        if USE_LR_FINDER:
            print("Running learning rate finder for initial phase...")
            self.lr_finder()
        self.train_phases(5, patience=EARLY_STOP_PATIENCE)
        
        print("\nPhase 2: Fine-tuning with layer4...")
        self.model.unfreeze_layer_group('layer4')
        self.setup_optimization('fine_tuning')
        self.train_phases(5, patience=EARLY_STOP_PATIENCE)
        
        print("\nPhase 3: Fine-tuning with layer3...")
        self.model.unfreeze_layer_group('layer3')
        self.setup_optimization('fine_tuning')
        self.train_phases(NUM_EPOCHS - 10, patience=EARLY_STOP_PATIENCE)
        
        if USE_SWA:
            print("Swapping in SWA weights for evaluation...")
            from torch.optim.swa_utils import update_bn
            update_bn(self.train_loader, self.swa_model, device=self.device)
            self.model = self.swa_model.module
    
    def evaluate(self, loader):
        self.model.eval()
        metrics = MetricsTracker()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                metrics.update(outputs, labels, loss, inputs.size(0))
        return metrics.get_metrics()
    
    def update_history(self, train_metrics, val_metrics):
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['train_roc_auc'].append(train_metrics['roc_auc'])
        self.history['train_precision'].append(train_metrics['precision'])
        self.history['train_recall'].append(train_metrics['recall'])
        self.history['train_specificity'].append(train_metrics['specificity'])
        self.history['train_f1'].append(train_metrics['f1_score'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['val_roc_auc'].append(val_metrics['roc_auc'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_specificity'].append(val_metrics['specificity'])
        self.history['val_f1'].append(val_metrics['f1_score'])
    
    def print_epoch_summary(self, epoch, train_metrics, val_metrics):
        pass

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_basic_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_detailed_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training Metrics
    axs[0, 0].plot(epochs, history['train_roc_auc'], 'b-o')
    axs[0, 0].set_title('Training ROC AUC')
    axs[0, 1].plot(epochs, history['train_precision'], 'b-o')
    axs[0, 1].set_title('Training Precision')
    axs[0, 2].plot(epochs, history['train_recall'], 'b-o')
    axs[0, 2].set_title('Training Recall')
    
    # Validation Metrics
    axs[1, 0].plot(epochs, history['val_roc_auc'], 'r-o')
    axs[1, 0].set_title('Validation ROC AUC')
    axs[1, 1].plot(epochs, history['val_precision'], 'r-o')
    axs[1, 1].set_title('Validation Precision')
    axs[1, 2].plot(epochs, history['val_recall'], 'r-o')
    axs[1, 2].set_title('Validation Recall')
    
    plt.tight_layout()
    plt.show()
    
    # Plot specificity and F1 separately
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(epochs, history['train_specificity'], 'b-o', label='Train Specificity')
    axs[0].plot(epochs, history['val_specificity'], 'r-o', label='Val Specificity')
    axs[0].set_title('Specificity')
    axs[0].legend()
    axs[1].plot(epochs, history['train_f1'], 'b-o', label='Train F1')
    axs[1].plot(epochs, history['val_f1'], 'r-o', label='Val F1')
    axs[1].set_title('F1 Score')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def plot_fold_performance(fold_histories, final_history):
    # Assume fold_histories is a list of history dictionaries (one per fold)
    metrics = ['val_loss', 'val_acc', 'val_roc_auc', 'val_precision', 'val_recall', 'val_specificity', 'val_f1']
    metric_names = ['Loss', 'Accuracy (%)', 'ROC AUC', 'Precision', 'Recall', 'Specificity', 'F1 Score']
    num_folds = len(fold_histories)
    fold_final = {m: [] for m in metrics}
    
    for history in fold_histories:
        for m in metrics:
            fold_final[m].append(history[m][-1])
    
    # Plot bar charts for each metric comparing folds and final SWA performance.
    final_metrics = final_history
    x = np.arange(num_folds)  # Fold indices
    width = 0.35
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i, m in enumerate(metrics):
        axs[i].bar(x, fold_final[m], width, label='Fold Final')
        axs[i].bar(x + width, [final_metrics[m][-1]]*num_folds, width, label='Final SWA')
        axs[i].set_title(metric_names[i])
        axs[i].set_xticks(x + width/2)
        axs[i].set_xticklabels([f'Fold {j+1}' for j in range(num_folds)])
        axs[i].legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

# ---------------------------
# Model Export Functionality
# ---------------------------
def export_model(model, save_path, model_name):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'input_size': MODEL_INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'transform_params': {'mean': [0.485, 0.456, 0.406],
                             'std': [0.229, 0.224, 0.225]}
    }, model_path)
    model.eval()
    traced_model = torch.jit.script(model)
    traced_model_path = os.path.join(save_path, f"{model_name}_traced.pt")
    traced_model.save(traced_model_path)
    print(f"Model saved to {model_path}")
    print(f"TorchScript model saved to {traced_model_path}")

# ---------------------------
# Main Function
# ---------------------------
def main():
    train_csv = TRAIN_CSV
    train_dir = TRAIN_DIR
    val_csv = VAL_CSV
    val_dir = VAL_DIR
    test_csv = TEST_CSV
    test_dir = TEST_DIR
    
    val_transform = create_validation_pipeline()
    train_transform = create_augmentation_pipeline()
    
    if CROSS_VALIDATE:
        df_train = pd.read_csv(train_csv)
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        fold_metrics = []
        fold_histories = []
        fold_idx = 1
        for train_index, val_index in skf.split(df_train, df_train['label']):
            print(f"\n--- Fold {fold_idx} ---")
            train_df = df_train.iloc[train_index]
            val_df = df_train.iloc[val_index]
            train_dataset = BoneFractureDataset(df=train_df, img_dir=train_dir, transform=train_transform)
            val_dataset = BoneFractureDataset(df=val_df, img_dir=train_dir, transform=val_transform)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            model = TransferLearningModel(NUM_CLASSES, pretrained=True)
            model = model.to(DEVICE)
            trainer = TransferLearningTrainer(model, train_loader, val_loader, None, DEVICE)
            trainer.progressive_training()
            fold_val_metrics = trainer.evaluate(val_loader)
            print(f"Fold {fold_idx} Validation Metrics: {fold_val_metrics}")
            fold_metrics.append(fold_val_metrics)
            fold_histories.append(trainer.history)
            fold_idx += 1
        avg_loss = np.mean([m['loss'] for m in fold_metrics])
        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in fold_metrics])
        print("\n--- Cross-Validation Results ---")
        print(f"Average Validation Loss: {avg_loss:.4f}")
        print(f"Average Validation Accuracy: {avg_acc:.2f}%")
        print(f"Average Validation F1 Score: {avg_f1:.4f}")
    else:
        train_dataset = BoneFractureDataset(csv_path=train_csv, img_dir=train_dir, transform=train_transform)
        val_dataset = BoneFractureDataset(csv_path=val_csv, img_dir=val_dir, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        model = TransferLearningModel(NUM_CLASSES, pretrained=True)
        model = model.to(DEVICE)
        trainer = TransferLearningTrainer(model, train_loader, val_loader, None, DEVICE)
        trainer.progressive_training()
    
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'ResNet50_best.pth')):
        best_model_path = os.path.join(MODEL_SAVE_PATH, 'ResNet50_best.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=False))
        
        export_model(model, MODEL_SAVE_PATH, 'ResNet50')
        
        test_dataset = BoneFractureDataset(csv_path=TEST_CSV, img_dir=TEST_DIR, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_metrics = trainer.evaluate(test_loader)
        print("\nFinal Test Results:")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
        report = classification_report(test_metrics['all_labels'], test_metrics['all_preds'], target_names=[
            'elbow positive', 'fingers positive', 'forearm fracture',
            'humerus', 'shoulder fracture', 'wrist positive', 'no fracture'
        ])
        print("\nClassification Report:")
        print(report)
        plot_confusion_matrix(test_metrics['all_labels'], test_metrics['all_preds'], [
            'elbow positive', 'fingers positive', 'forearm fracture',
            'humerus', 'shoulder fracture', 'wrist positive', 'no fracture'
        ])
    else:
        print("No best model was saved.")
    
    # Plot the metrics:
    print("\nPlotting Basic History (Loss & Accuracy)...")
    if CROSS_VALIDATE:
        plot_basic_history(fold_histories[-1])
        print("\nPlotting Detailed Metrics...")
        plot_detailed_metrics(fold_histories[-1])
        print("\nPlotting Fold Performances vs. Final SWA Performance...")
        # Assuming final SWA performance is from the last epoch of the last fold's history
        plot_fold_performance(fold_histories, fold_histories[-1])
    else:
        plot_basic_history(trainer.history)
        plot_detailed_metrics(trainer.history)
    
if __name__ == '__main__':
    main()
