import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models import densenet121  # DenseNet121 backbone

# ---------------------------
# Constants and Paths
# ---------------------------
NUM_CLASSES = 7
MODEL_INPUT_SIZE = 224  # DenseNet121’s input size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
# Use the .pth checkpoint (ensure this checkpoint was saved from a DenseNet121 model)
PTH_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "DenseNet121.pth")
TEST_IMAGES_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"
OUTPUT_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\DenseNet-Grad-CAM++"
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
# TransferLearningModel for DenseNet121 with GradCAM++ Support
# ---------------------------
class TransferLearningDenseNet121(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        """
        Uses DenseNet121 as the backbone.
        """
        super(TransferLearningDenseNet121, self).__init__()
        self.base_model = densenet121(pretrained=False)
        num_features = self.base_model.classifier.in_features
        # Remove the original classifier.
        self.base_model.classifier = nn.Identity()
        # Custom classifier head.
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
        # Forward pass through DenseNet121 features.
        features = self.base_model.features(x)  # shape: (B, C, H, W)
        # Apply ReLU without in-place operations.
        features = F.relu(features, inplace=False)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        out = self.classifier(flattened)
        return out

# ---------------------------
# GradCAM++ Implementation for DenseNet121
# ---------------------------
def generate_cam_densenet_plus(model, input_tensor, target_class):
    """
    Computes GradCAM++ for DenseNet121.
    
    Args:
        model: The DenseNet121 model instance.
        input_tensor: Preprocessed input tensor of shape (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).
        target_class: Integer target class (if None, the predicted class is used).
    
    Returns:
        cam: GradCAM++ heatmap as a numpy array (MODEL_INPUT_SIZE x MODEL_INPUT_SIZE) normalized to [0, 1].
        confidence: The model's confidence for the target class.
    """
    # Forward pass: extract convolutional features from the DenseNet121 backbone.
    features = model.base_model.features(input_tensor)  # (1, C, H, W)
    features = F.relu(features, inplace=False)
    # Enable gradient computation on features.
    features.requires_grad_()
    features.retain_grad()
    
    # Pool features and pass through classifier.
    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    flattened = torch.flatten(pooled, 1)
    logits = model.classifier(flattened)
    probabilities = torch.softmax(logits, dim=1)
    
    if target_class is None:
        target_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, target_class].item()
    
    model.zero_grad()
    score = logits[0, target_class]
    # Compute first-order gradients with create_graph=True.
    score.backward(retain_graph=True, create_graph=True)
    
    gradients = features.grad  # (1, C, H, W)
    # Compute element-wise second- and third-order gradients.
    second_gradients = gradients ** 2
    third_gradients = gradients ** 3
    
    # Global sum over spatial dimensions.
    global_sum = torch.sum(features * third_gradients, dim=(2, 3), keepdim=True)
    denominator = 2 * second_gradients + global_sum + 1e-8
    alpha = second_gradients / denominator  # (1, C, H, W)
    
    relu_gradients = F.relu(gradients)
    weights = torch.sum(alpha * relu_gradients, dim=(2, 3))  # (1, C)
    
    cam = torch.sum(weights.unsqueeze(2).unsqueeze(3) * features, dim=1)  # (1, H, W)
    cam = F.relu(cam)
    
    # Resize and normalize CAM.
    cam = cam.squeeze(0).cpu().detach().numpy()
    cam = cv2.resize(cam, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam, confidence

# ---------------------------
# Main GradCAM++ Visualization Routine
# ---------------------------
def main():
    # Load the DenseNet121 checkpoint.
    try:
        checkpoint = torch.load(PTH_MODEL_PATH, map_location=DEVICE,weights_only=False)
    except Exception as e:
        print(f"Error loading model from {PTH_MODEL_PATH}: {e}")
        return
    
    # If the checkpoint is a dictionary with 'model_state_dict', extract it.
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    print("DenseNet121 model state loaded successfully from .pth file.")
    
    # Create the DenseNet121 model and load the state dictionary.
    model = TransferLearningDenseNet121(NUM_CLASSES, pretrained=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("State dict loaded into DenseNet121 model for GradCAM++.")
    
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not test_images:
        print("No test images found.")
        return
    
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        print(f"Processing image: {img_path}")
        
        # Load a display image using Ben Graham's method.
        try:
            display_image = load_ben_color(img_path)
        except Exception as e:
            print(f"Error processing display image {img_path}: {e}")
            continue
        
        # Load and preprocess an image for model input.
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            print(f"Unable to load image {img_path}. Skipping.")
            continue
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(rgb_image).to(DEVICE)
        
        # Get model predictions.
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        label_text = LABEL_NAMES[pred_class] if pred_class < len(LABEL_NAMES) else f"Class {pred_class}"
        
        # Compute the GradCAM++ heatmap for the predicted class.
        cam, _ = generate_cam_densenet_plus(model, input_tensor, target_class=pred_class)
        heatmap = apply_heatmap(cam)
        overlay = overlay_heatmap_on_image(display_image, heatmap, alpha=0.5)
        
        # Create and save the visualization.
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
        plt.title(f"Overlay\nPredicted: {label_text}\nConfidence: {confidence*100:.2f}%")
        plt.axis("off")
        
        plt.tight_layout()
        output_filename = os.path.splitext(img_file)[0] + "_gradcamplusplus.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved GradCAM++ visualization for {img_file} to {output_path}")

if __name__ == "__main__":
    main()
