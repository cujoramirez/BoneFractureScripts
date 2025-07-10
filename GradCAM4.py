import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models import inception_v3, Inception_V3_Weights

# ---------------------------
# Constants and Paths
# ---------------------------
NUM_CLASSES = 7
MODEL_INPUT_SIZE = 299  # InceptionV3’s input size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
# Use the .pth checkpoint for training (not the deployment TorchScript model)
PTH_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "InceptionV3.pth")
TEST_IMAGES_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"
OUTPUT_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\InceptionV3-Grad-CAM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class labels (must match training order)
LABEL_NAMES = [
    'elbow positive', 'fingers positive', 'forearm fracture',
    'humerus', 'shoulder fracture', 'wrist positive', 'no fracture'
]

# ---------------------------
# Helper Functions for Preprocessing and Display
# ---------------------------
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        return img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if mask.any():
            coords = np.argwhere(mask)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1
            return img[x0:x1, y0:y1, :]
        return img

def load_ben_color(path, sigmaX=10):
    """
    Loads an image from the given path, crops dark borders,
    resizes to MODEL_INPUT_SIZE, and applies Ben Graham's brightness adjustments.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to load image from {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

def preprocess_image(image):
    """
    Resize and normalize the image for model inference.
    Returns a tensor of shape (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).
    """
    image = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    return image_tensor.float()

def apply_heatmap(heatmap):
    """
    Convert a normalized heatmap (values between 0 and 1) to a colored heatmap.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_color

def overlay_heatmap_on_image(original, heatmap, alpha=0.5):
    """
    Overlay the colored heatmap on the original image.
    """
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ---------------------------
# TransferLearningModel for InceptionV3 with GradCAM++ Support
# ---------------------------
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        self.base_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None)
        if pretrained:
            self.base_model.aux_logits = False
            self.base_model.AuxLogits = None
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
        if layer_name == 'layer4':
            for block in ['Mixed_7a', 'Mixed_7b', 'Mixed_7c']:
                if hasattr(self.base_model, block):
                    for param in getattr(self.base_model, block).parameters():
                        param.requires_grad = True
        elif layer_name == 'layer3':
            for block in ['Mixed_6d', 'Mixed_6e']:
                if hasattr(self.base_model, block):
                    for param in getattr(self.base_model, block).parameters():
                        param.requires_grad = True
        else:
            for param in self.base_model.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        outputs = self.base_model(x)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        return self.classifier(outputs)

# ---------------------------
# Forward Functions for Feature Extraction
# ---------------------------
def forward_until_mixed7c(model, input_tensor):
    x = model.base_model.Conv2d_1a_3x3(input_tensor)
    x = model.base_model.Conv2d_2a_3x3(x)
    x = model.base_model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.base_model.Conv2d_3b_1x1(x)
    x = model.base_model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.base_model.Mixed_5b(x)
    x = model.base_model.Mixed_5c(x)
    x = model.base_model.Mixed_5d(x)
    x = model.base_model.Mixed_6a(x)
    x = model.base_model.Mixed_6b(x)
    x = model.base_model.Mixed_6c(x)
    x = model.base_model.Mixed_6d(x)
    x = model.base_model.Mixed_6e(x)
    x = model.base_model.Mixed_7a(x)
    x = model.base_model.Mixed_7b(x)
    x = model.base_model.Mixed_7c(x)
    return x

def forward_until_mixed7b(model, input_tensor):
    x = model.base_model.Conv2d_1a_3x3(input_tensor)
    x = model.base_model.Conv2d_2a_3x3(x)
    x = model.base_model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.base_model.Conv2d_3b_1x1(x)
    x = model.base_model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.base_model.Mixed_5b(x)
    x = model.base_model.Mixed_5c(x)
    x = model.base_model.Mixed_5d(x)
    x = model.base_model.Mixed_6a(x)
    x = model.base_model.Mixed_6b(x)
    x = model.base_model.Mixed_6c(x)
    x = model.base_model.Mixed_6d(x)
    x = model.base_model.Mixed_6e(x)
    x = model.base_model.Mixed_7a(x)
    x = model.base_model.Mixed_7b(x)
    return x

# ---------------------------
# GradCAM++ Implementation for InceptionV3
# ---------------------------
def generate_cam_inception_plus(model, input_tensor, target_class):
    """
    Computes GradCAM++ for InceptionV3.
    
    Args:
        model: The loaded InceptionV3 model (an instance of TransferLearningModel).
        input_tensor: Preprocessed input tensor of shape (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).
        target_class: Integer target class (if None, the predicted class is used).
    
    Returns:
        cam: GradCAM++ heatmap as a numpy array (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE) normalized to [0, 1].
        confidence: The model's confidence for the target class.
    """
    # Forward pass up to Mixed_7c
    features = forward_until_mixed7c(model, input_tensor)
    features = F.relu(features, inplace=False)
    features.requires_grad_()
    features.retain_grad()
    
    pooled = model.base_model.avgpool(features)
    flattened = torch.flatten(pooled, 1)
    logits = model.classifier(flattened)
    probabilities = torch.softmax(logits, dim=1)
    
    if target_class is None:
        target_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, target_class].item()
    
    model.zero_grad()
    # Use create_graph=True for higher-order derivatives
    score = logits[0, target_class]
    score.backward(retain_graph=True, create_graph=True)
    
    # First-order gradients
    gradients = features.grad
    # Compute second and third-order gradients (element-wise)
    second_gradients = gradients ** 2
    third_gradients = gradients ** 3
    
    # Global sum: sum over spatial dimensions for third gradients weighted by the feature map
    global_sum = torch.sum(features * third_gradients, dim=(2, 3), keepdim=True)
    denominator = 2 * second_gradients + global_sum + 1e-8
    alpha = second_gradients / denominator
    relu_gradients = F.relu(gradients)
    weights = torch.sum(alpha * relu_gradients, dim=(2, 3))
    
    cam = torch.sum(weights.unsqueeze(2).unsqueeze(3) * features, dim=1)
    cam = F.relu(cam)
    
    cam = cam.squeeze(0).cpu().detach().numpy()
    # If the CAM is nearly uniform, try using Mixed_7b features
    if np.abs(cam.max() - cam.min()) < 1e-6:
        print("Warning: GradCAM++ from Mixed_7c is nearly uniform. Recomputing using Mixed_7b...")
        features = forward_until_mixed7b(model, input_tensor)
        features = F.relu(features, inplace=False)
        features.requires_grad_()
        features.retain_grad()
        
        pooled = model.base_model.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        logits = model.classifier(flattened)
        probabilities = torch.softmax(logits, dim=1)
        if target_class is None:
            target_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, target_class].item()
        
        model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True, create_graph=True)
        
        gradients = features.grad
        second_gradients = gradients ** 2
        third_gradients = gradients ** 3
        global_sum = torch.sum(features * third_gradients, dim=(2,3), keepdim=True)
        denominator = 2 * second_gradients + global_sum + 1e-8
        alpha = second_gradients / denominator
        relu_gradients = F.relu(gradients)
        weights = torch.sum(alpha * relu_gradients, dim=(2,3))
        cam = torch.sum(weights.unsqueeze(2).unsqueeze(3) * features, dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze(0).cpu().detach().numpy()
    
    cam = cv2.resize(cam, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam, confidence

# ---------------------------
# Main GradCAM++ Visualization Routine
# ---------------------------
def main():
    # Load the model from the .pth checkpoint
    model = TransferLearningModel(NUM_CLASSES, pretrained=True)
    checkpoint = torch.load(PTH_MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    print("Loaded InceptionV3 model from .pth successfully.")
    
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print("No test images found.")
        return
    
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        print(f"Processing image: {img_path}")
        
        # Load display image using Ben Graham’s method
        try:
            display_image = load_ben_color(img_path)
        except Exception as e:
            print(f"Error processing display image {img_path}: {e}")
            continue
        
        # For model input, load a separate copy and preprocess it.
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            print(f"Unable to load image {img_path}. Skipping.")
            continue
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(rgb_image).to(DEVICE)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Compute GradCAM++ heatmap for the predicted class
        cam, _ = generate_cam_inception_plus(model, input_tensor, target_class=pred_class)
        heatmap = apply_heatmap(cam)
        overlay = overlay_heatmap_on_image(display_image, heatmap, alpha=0.5)
        
        # Create and save the visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(display_image)
        plt.title("Ben Graham Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title("GradCAM++ Heatmap")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPredicted: {LABEL_NAMES[pred_class]}\nConfidence: {confidence*100:.2f}%")
        plt.axis("off")
        plt.tight_layout()
        output_filename = os.path.splitext(img_file)[0] + "_gradcamplusplus.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved GradCAM++ visualization for {img_file} to {output_path}")

if __name__ == "__main__":
    main()
