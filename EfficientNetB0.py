import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
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
# Use 224x224 as the model input size for EfficientNetB0 (typical for ImageNet)
MODEL_INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 60
BASE_LEARNING_RATE = 1e-3  # Base LR (will be overridden in phases)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths (adjust as needed)
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
    """Training augmentation pipeline with advanced transformations."""
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
    """Validation and test pipeline: fixed resizing and normalization."""
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
        
        # Label mapping: handles both numeric and string labels.
        self.label_map = {
            '0': 0, '1': 1, '2': 2, '4': 3, '5': 4, '6': 5, 'no_fracture': 6
        }
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
# Metrics Tracking
# ---------------------------
class MetricsTracker:
    def __init__(self):
        self.running_loss = 0
        self.running_corrects = 0
        self.total_samples = 0
        self.all_preds = []
        self.all_labels = []
        
    def update(self, outputs, labels, loss, batch_size):
        self.running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        self.running_corrects += torch.sum(preds == labels).item()
        self.total_samples += batch_size
        self.all_preds.extend(outputs.cpu().detach().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
    def get_metrics(self):
        preds_array = np.argmax(np.array(self.all_preds), axis=1)
        return {
            'loss': self.running_loss / self.total_samples,
            'accuracy': 100 * self.running_corrects / self.total_samples,
            'f1_score': f1_score(self.all_labels, preds_array, average='weighted'),
            'all_preds': preds_array,
            'all_labels': np.array(self.all_labels)
        }

# ---------------------------
# Transfer Learning Model (EfficientNetB0)
# ---------------------------
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        # Load EfficientNetB0 with pretrained weights
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # Get number of input features from the original classifier (which is a Sequential: [Dropout, Linear])
        num_features = self.base_model.classifier[1].in_features
        # Replace the original classifier with Identity so that we can add a custom head
        self.base_model.classifier = nn.Identity()
        # Custom classifier head with reduced dropout (0.2)
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
        """
        For EfficientNetB0, we assume the features are a Sequential module.
        We mimic the ResNet50 progressive unfreezing by:
          - For 'layer4': unfreeze the last 2 blocks of self.base_model.features.
          - For 'layer3': unfreeze the 2 blocks preceding those.
        """
        features = self.base_model.features
        total_blocks = len(features)
        if layer_name == 'layer4':
            # Unfreeze the last 2 blocks
            for i in range(total_blocks - 2, total_blocks):
                for param in features[i].parameters():
                    param.requires_grad = True
        elif layer_name == 'layer3':
            # Unfreeze the 2 blocks preceding the last 2 blocks
            for i in range(total_blocks - 4, total_blocks - 2):
                for param in features[i].parameters():
                    param.requires_grad = True
        else:
            for param in self.base_model.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        # The base_model returns the flattened features because classifier is Identity.
        features = self.base_model(x)
        return self.classifier(features)

# ---------------------------
# Trainer for Progressive Transfer Learning with FP16, AdamW, SWA, etc.
# ---------------------------
class TransferLearningTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Use label smoothing in the loss function (requires PyTorch 1.10+)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = None
        self.scheduler = None
        
        self.scaler = torch.amp.GradScaler()
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        if USE_SWA:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = None
        
    def setup_optimization(self, phase):
        if phase == 'initial':
            # Optimize only the classifier head with AdamW
            trainable_params = self.model.classifier.parameters()
            self.optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.005)
        elif phase == 'fine_tuning':
            # Fine-tuning: lower LR for both classifier and base layers
            classifier_params = {'params': self.model.classifier.parameters(), 'lr': 5e-4}
            base_params = {'params': filter(lambda p: p.requires_grad, self.model.base_model.parameters()), 'lr': 1e-5}
            self.optimizer = optim.AdamW([classifier_params, base_params], weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        if USE_SWA:
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=1e-4)
            
    def setup_optimization_phase3(self):
        # Further lower the LR for the classifier as well as the base layers
        classifier_params = {'params': self.model.classifier.parameters(), 'lr': 1e-5}
        base_params = {'params': filter(lambda p: p.requires_grad, self.model.base_model.parameters()), 'lr': 1e-5}
        self.optimizer = optim.AdamW([classifier_params, base_params], weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        if USE_SWA:
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=1e-4)
        
    def lr_finder(self, init_value=1e-6, final_value=10, beta=0.98, num_iters=100):
        """
        A minimal learning rate finder that increases the learning rate exponentially.
        This function runs for a few iterations on a single batch.
        """
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
            # Apply gradient clipping
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
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                epochs_without_improvement = 0
                # Save best model weights
                torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'EfficientNetB0_best.pth'))
            else:
                epochs_without_improvement += 1
            
            self.print_epoch_summary(epoch, train_metrics, val_metrics)
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
        # Train classifier head for 2 epochs
        self.train_phases(5, patience=EARLY_STOP_PATIENCE)
        
        print("\nPhase 2: Fine-tuning with layer4...")
        self.model.unfreeze_layer_group('layer4')
        self.setup_optimization('fine_tuning')
        # Fine-tune layer4 for 4 epochs
        self.train_phases(5, patience=EARLY_STOP_PATIENCE)
        
        print("\nPhase 3: Fine-tuning with layer3...")
        self.model.unfreeze_layer_group('layer3')
        self.setup_optimization_phase3()
        # Train remaining epochs
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
        self.history['train_f1'].append(train_metrics['f1_score'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1_score'])
    
    def print_epoch_summary(self, epoch, train_metrics, val_metrics):
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, F1: {train_metrics['f1_score']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, F1: {val_metrics['f1_score']:.4f}")

# ---------------------------
# Plotting Training History
# ---------------------------
def plot_history(history):
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
    plt.savefig('confusion_matrix.png')
    plt.close()

# ---------------------------
# Model Export Functionality
# ---------------------------
def export_model(model, save_path, model_name):
    """
    Export the complete model with architecture and weights.
    
    Args:
        model: The trained PyTorch model
        save_path: Directory to save the model
        model_name: Name of the model file
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save the complete model (architecture + weights)
    model_path = os.path.join(save_path, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'input_size': MODEL_INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'transform_params': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }, model_path)
    
    # Export to TorchScript for deployment
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
            # For cross-validation, use the validation pipeline on the validation set
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
    
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'EfficientNetB0_best.pth')):
        best_model_path = os.path.join(MODEL_SAVE_PATH, 'EfficientNetB0_best.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=False))
        
        export_model(model, MODEL_SAVE_PATH, 'EfficientNetB0')
        
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
    
    if not CROSS_VALIDATE:
        plot_history(trainer.history)
    else:
        plot_history(fold_histories[-1])

if __name__ == '__main__':
    main()
