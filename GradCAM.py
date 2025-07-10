import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

# ---------------------------
# Constants and Paths
# ---------------------------
NUM_CLASSES = 7
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
PTH_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "ResNet50_best.pth")
TEST_IMAGES_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"
OUTPUT_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\Grad-CAM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = [
    'elbow positive', 'fingers positive', 'forearm fracture',
    'humerus', 'shoulder fracture', 'wrist positive', 'no fracture'
]

# ---------------------------
# Helper Functions
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
    """Load image using Ben Graham's brightness adjustments."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to load image from {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    return image

def preprocess_image(image):
    """Resize and normalize image for model input."""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).float()

def apply_heatmap(heatmap):
    """Convert a normalized heatmap to a colored heatmap."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

def overlay_heatmap_on_image(original, heatmap, alpha=0.5):
    """Overlay the colored heatmap on the original image."""
    return cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

# ---------------------------
# Model Definition for ResNet50
# ---------------------------
class TransferLearningResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(TransferLearningResNet50, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
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
        out = self.classifier(features)
        return out

# ---------------------------
# GradCAM++ Implementation for ResNet50
# ---------------------------
def generate_cam_resnetplusplus(model, input_tensor, target_class):
    """
    Computes GradCAM++ for ResNet50.
    
    Manually runs forward pass up to layer4, then computes gradients.
    Falls back to standard GradCAM if second-order gradients vanish.
    
    Args:
        model: TransferLearningResNet50 model.
        input_tensor: Tensor of shape (1, 3, IMG_SIZE, IMG_SIZE).
        target_class: Target class index.
    Returns:
        cam: Heatmap (IMG_SIZE x IMG_SIZE) normalized to [0,1].
        confidence: Model confidence for the target class.
    """
    base = model.base_model
    x = base.conv1(input_tensor)
    x = base.bn1(x)
    x = base.relu(x)
    x = base.maxpool(x)
    x = base.layer1(x)
    x = base.layer2(x)
    x = base.layer3(x)
    x_conv = base.layer4(x)
    
    x_conv.requires_grad_(True)
    x_conv.retain_grad()
    
    x_pool = base.avgpool(x_conv)
    x_flat = torch.flatten(x_pool, 1)
    logits = model.classifier(x_flat)
    probabilities = torch.softmax(logits, dim=1)
    if target_class is None:
        target_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, target_class].item()
    
    model.zero_grad()
    score = logits[0, target_class]
    score.backward(retain_graph=True)
    
    first_gradients = torch.autograd.grad(score, x_conv, create_graph=True)[0]
    print("First gradients norm: {:.6f}".format(first_gradients.norm().item()))
    second_gradients = torch.autograd.grad(first_gradients, x_conv,
                                           grad_outputs=torch.ones_like(first_gradients),
                                           create_graph=True)[0]
    third_gradients = torch.autograd.grad(second_gradients, x_conv,
                                          grad_outputs=torch.ones_like(second_gradients),
                                          create_graph=True)[0]
    
    second_norm = second_gradients.norm().item()
    print("Second gradients norm: {:.6f}".format(second_norm))
    
    if second_norm < 1e-6:
        print("Second gradients nearly zero; falling back to GradCAM (average of first gradients).")
        weights = torch.mean(F.relu(first_gradients), dim=(2,3))  # Note: no keepdim.
    else:
        global_sum = torch.sum(x_conv * third_gradients, dim=(2,3), keepdim=True)
        denominator = 2 * second_gradients + global_sum + 1e-8
        alpha = second_gradients / denominator  # Shape: (1, C, H, W)
        weights = torch.sum(alpha * F.relu(first_gradients), dim=(2,3))
    
    cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * x_conv, dim=1)
    cam = F.relu(cam)
    
    print("CAM raw: min={:.6f}, max={:.6f}".format(cam.min().item(), cam.max().item()))
    
    cam = cam.squeeze(0).cpu().detach().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam, confidence

# ---------------------------
# Main Visualization Routine
# ---------------------------
def main():
    try:
        checkpoint = torch.load(PTH_MODEL_PATH, map_location=DEVICE)
    except Exception as e:
        print(f"Error loading model from {PTH_MODEL_PATH}: {e}")
        return

    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    print("ResNet50 model state loaded successfully from .pth file.")
    
    model = TransferLearningResNet50(NUM_CLASSES, pretrained=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("State dict loaded into ResNet50 model for GradCAM++.")
    
    for img_filename in os.listdir(TEST_IMAGES_DIR):
        if not img_filename.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        
        img_path = os.path.join(TEST_IMAGES_DIR, img_filename)
        print(f"Processing image: {img_path}")
        
        try:
            display_image = load_ben_color(img_path)
        except Exception as e:
            print(f"Error processing display image {img_path}: {e}")
            continue
        
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            print(f"Unable to load image {img_path}. Skipping.")
            continue
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(rgb_image).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_class = torch.max(probs, dim=1)
            confidence = conf.item()
            pred_class = pred_class.item()
        
        label_text = LABEL_NAMES[pred_class] if pred_class < len(LABEL_NAMES) else f"Class {pred_class}"
        
        cam, _ = generate_cam_resnetplusplus(model, input_tensor, target_class=pred_class)
        heatmap = apply_heatmap(cam)
        overlay = overlay_heatmap_on_image(display_image, heatmap, alpha=0.5)
        
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
        output_filename = os.path.splitext(img_filename)[0] + "_gradcamplusplus.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved GradCAM++ visualization for {img_filename} to {output_path}")

if __name__ == "__main__":
    main()
