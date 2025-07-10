import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models import efficientnet_b0

# ---------------------------
# Constants and Paths
# ---------------------------
NUM_CLASSES = 7
MODEL_INPUT_SIZE = 224  # EfficientNetB0 input size.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
# Use the .pth checkpoint for GradCAM++
PTH_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "EfficientNetB0_best.pth")
TEST_IMAGES_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"
OUTPUT_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\EfficientNetB0-Grad-CAM"
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
    """Load an image, crop dark borders, resize, and adjust brightness."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to load image from {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    return image

def preprocess_image(image):
    """Resize and normalize image for model inference."""
    image = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2,0,1))
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    return image_tensor.float()

def apply_heatmap(heatmap):
    """Convert a normalized heatmap to a colored heatmap."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

def overlay_heatmap_on_image(original, heatmap, alpha=0.5):
    """Overlay the colored heatmap on the original image."""
    return cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

# ---------------------------
# Model Definition for EfficientNetB0
# ---------------------------
class TransferLearningEfficientNetB0(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        """
        Instantiate EfficientNetB0 without pretrained weights and define a custom classifier head.
        """
        super(TransferLearningEfficientNetB0, self).__init__()
        self.base_model = efficientnet_b0(weights=None)
        num_features = self.base_model.classifier[1].in_features
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
        # For inference, use the full forward pass.
        # Note: self.base_model(x) performs avgpool so that the classifier input is pooled.
        features = self.base_model(x)
        out = self.classifier(features)
        return out

# ---------------------------
# GradCAM++ Implementation for EfficientNetB0
# ---------------------------
def generate_cam_efficientnet(model, input_tensor, target_class):
    """
    Computes GradCAM++ for EfficientNetB0.
    
    Args:
        model: The fine-tuned EfficientNetB0 model.
        input_tensor: Preprocessed input tensor of shape (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).
        target_class: Integer target class (if None, use predicted class).
    
    Returns:
        cam: GradCAM++ heatmap as a numpy array normalized to [0, 1].
        confidence: The model's confidence for the target class.
    """
    # Ensure input tensor requires gradients.
    input_tensor.requires_grad_(True)
    # Unfreeze backbone parameters.
    for param in model.base_model.parameters():
        param.requires_grad = True

    # Get convolutional feature maps before pooling.
    features = model.base_model.features(input_tensor)  # (1, C, H, W)
    features = F.relu(features, inplace=False)
    # Do NOT detach the features.
    
    # Compute pooled features manually.
    pooled = F.adaptive_avg_pool2d(features, (1,1))
    flattened = torch.flatten(pooled, 1)
    logits = model.classifier(flattened)
    probabilities = torch.softmax(logits, dim=1)
    
    if target_class is None:
        target_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, target_class].item()
    
    model.zero_grad()
    score = logits[0, target_class]
    
    # Compute first-order gradients.
    first_gradients = torch.autograd.grad(score, features, create_graph=True)[0]
    print("First gradients norm: {:.6f}".format(first_gradients.norm().item()))
    
    # Compute second- and third-order gradients.
    second_gradients = torch.autograd.grad(first_gradients, features,
                                           grad_outputs=torch.ones_like(first_gradients),
                                           create_graph=True)[0]
    third_gradients = torch.autograd.grad(second_gradients, features,
                                          grad_outputs=torch.ones_like(second_gradients),
                                          create_graph=True)[0]
    
    # Check the norm of second gradients.
    second_norm = second_gradients.norm().item()
    print("Second gradients norm: {:.6f}".format(second_norm))
    
    # If the second derivatives are nearly zero, fall back to GradCAM.
    if second_norm < 1e-6:
        print("Second gradients nearly zero, using GradCAM weights.")
        weights = torch.mean(F.relu(first_gradients), dim=(2,3))
    else:
        global_sum = torch.sum(features * third_gradients, dim=(2,3), keepdim=True)
        denominator = 2 * second_gradients + global_sum + 1e-8
        alpha = second_gradients / denominator  # (1, C, H, W)
        weights = torch.sum(alpha * F.relu(first_gradients), dim=(2,3))
    
    # Compute the weighted combination of feature maps.
    cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * features, dim=1)
    cam = F.relu(cam)
    
    # Debug: print raw CAM stats.
    print("CAM raw: min={:.6f}, max={:.6f}".format(cam.min().item(), cam.max().item()))
    
    # Resize and normalize CAM.
    cam = cam.squeeze(0).cpu().detach().numpy()
    cam = cv2.resize(cam, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam, confidence

# ---------------------------
# Main GradCAM++ Visualization Routine
# ---------------------------
def main():
    # Load the model checkpoint.
    try:
        checkpoint = torch.load(PTH_MODEL_PATH, map_location=DEVICE)
    except Exception as e:
        print(f"Error loading model from {PTH_MODEL_PATH}: {e}")
        return
    
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    print("EfficientNetB0 model state loaded successfully from .pth file.")
    
    # Create and load the model.
    model = TransferLearningEfficientNetB0(NUM_CLASSES, pretrained=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("State dict loaded into EfficientNetB0 model for GradCAM++.")
    
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not test_images:
        print("No test images found.")
        return
    
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        print(f"Processing image: {img_path}")
        
        # Load display image.
        try:
            display_image = load_ben_color(img_path)
        except Exception as e:
            print(f"Error processing display image {img_path}: {e}")
            continue
        
        # Load and preprocess the image.
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            print(f"Unable to load image {img_path}. Skipping.")
            continue
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(rgb_image).to(DEVICE)
        
        # Get model prediction.
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        label_text = LABEL_NAMES[pred_class] if pred_class < len(LABEL_NAMES) else f"Class {pred_class}"
        
        # Compute GradCAM++ heatmap.
        cam, _ = generate_cam_efficientnet(model, input_tensor, target_class=pred_class)
        heatmap = apply_heatmap(cam)
        overlay = overlay_heatmap_on_image(display_image, heatmap, alpha=0.5)
        
        # Visualize and save.
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(display_image)
        plt.title("Ben Graham Image")
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.imshow(heatmap)
        plt.title("GradCAM++ Heatmap")
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPredicted: {label_text}\nConfidence: {confidence*100:.2f}%")
        plt.axis("off")
        
        plt.tight_layout()
        output_filename = os.path.splitext(img_file)[0] + "_gradcamplusplus.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved GradCAM++ visualization for {img_file} to {output_path}")

if __name__ == "__main__":
    main()
