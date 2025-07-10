import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shap  # pip install shap
import contextlib

from torchvision.models import inception_v3, Inception_V3_Weights

# ---------------------------
# Constants and Paths
# ---------------------------
NUM_CLASSES = 7
MODEL_INPUT_SIZE = 299  # InceptionV3’s input size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
PTH_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "InceptionV3.pth")
TEST_IMAGES_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"
OUTPUT_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\InceptionV3-Grad-CAM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    Ensures the heatmap is a 2D contiguous array of type uint8.
    """
    heatmap = np.squeeze(heatmap)
    if heatmap.ndim != 2:
        heatmap = heatmap[..., 0]
    heatmap_uint8 = np.ascontiguousarray(np.uint8(255 * heatmap))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_color

def overlay_heatmap_on_image(original, heatmap, alpha=0.5):
    """
    Overlay the colored heatmap on the original image.
    """
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlay

def denormalize_input_tensor(tensor):
    """
    Convert a normalized tensor (1,3,H,W) back to a numpy image (H,W,3) in [0,1].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy()[0].transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# ---------------------------
# Helper: Disable Inplace ReLU Recursively
# ---------------------------
def disable_inplace_relu(module):
    """
    Recursively sets all nn.ReLU modules to inplace=False.
    """
    for child_name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, nn.ReLU(inplace=False))
        else:
            disable_inplace_relu(child)

# ---------------------------
# Context Manager to Disable Inplace ReLU Globally for SHAP
# ---------------------------
@contextlib.contextmanager
def disable_inplace_relu_global():
    orig_relu = torch.nn.functional.relu
    def safe_relu(input, inplace=True):
        return orig_relu(input, inplace=False)
    torch.nn.functional.relu = safe_relu
    try:
        yield
    finally:
        torch.nn.functional.relu = orig_relu

# ---------------------------
# Function to Extract Convolutional Features
# ---------------------------
def forward_until_mixed7c(model, input_tensor):
    """
    Runs the input tensor through InceptionV3's layers up to Mixed_7c.
    Returns the convolutional feature maps.
    """
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
    return x  # Expected shape: (1, C, H, W)

# ---------------------------
# SHAP-CAM (Deep) Implementation
# ---------------------------
def generate_shapcam_deep(model, input_tensor, background_tensor, pred_class):
    """
    Computes a SHAP-CAM heatmap for the predicted class based on conv features.
    
    Steps:
      1. Compute conv features for input and background.
      2. Define a module (ConvToScore) that maps conv features to the predicted class score.
      3. Use SHAP's DeepExplainer on conv features to obtain per-channel SHAP values.
      4. Average the SHAP values spatially (with appropriate axis order) to obtain weights of shape (1, C, 1, 1).
      5. Compute the weighted sum of the conv features over channels, apply ReLU, upsample, and normalize.
    
    Returns:
        shap_cam: A heatmap of shape (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE) normalized to [0,1].
    """
    # 1. Compute conv features.
    conv_features = forward_until_mixed7c(model, input_tensor)          # Expected shape: (1, C, H, W)
    background_conv = forward_until_mixed7c(model, background_tensor)     # Expected shape: (1, C, H, W)
    
    # 2. Define a module that maps conv features to the predicted class score.
    class ConvToScore(nn.Module):
        def __init__(self, model, pred_class):
            super(ConvToScore, self).__init__()
            self.model = model
            self.pred_class = pred_class
        def forward(self, x):
            pooled = self.model.base_model.avgpool(x)        # (batch_size, C, 1, 1)
            flattened = torch.flatten(pooled, 1)              # (batch_size, C)
            logits = self.model.classifier(flattened)         # (batch_size, num_classes)
            return logits[:, self.pred_class:self.pred_class+1]  # (batch_size, 1)
    
    conv_to_score = ConvToScore(model, pred_class).to(DEVICE)
    
    # 3. Compute SHAP values for the conv features.
    with disable_inplace_relu_global():
        explainer_conv = shap.DeepExplainer(conv_to_score, background_conv)
        shap_conv_values = explainer_conv.shap_values(conv_features, check_additivity=False)
    # Expect a list with one element.
    shap_conv_values = shap_conv_values[0]
    
    # 4. Ensure the SHAP values have shape (batch, channels, H, W).
    if shap_conv_values.ndim == 4:
        # Sometimes the SHAP values may come out with a different axis order,
        # e.g. (channels, H, W, batch) instead of (batch, channels, H, W).
        # Check if the last dimension is 1 (batch size) but the first dimension is channels.
        if shap_conv_values.shape[0] != conv_features.shape[0] and shap_conv_values.shape[-1] == conv_features.shape[0]:
            # Transpose from (C, H, W, batch) to (batch, C, H, W)
            shap_conv_values = np.transpose(shap_conv_values, (3, 0, 1, 2))
    elif shap_conv_values.ndim == 3:
        # If no batch dimension, add one.
        shap_conv_values = shap_conv_values[None, ...]
    
    # 5. Average spatially over H, W with keepdims to obtain weights of shape (batch, C, 1, 1).
    channel_weights = np.mean(shap_conv_values, axis=(2,3), keepdims=True)
    
    # 6. Compute weighted sum of conv features.
    conv_features_np = conv_features.cpu().detach().numpy()  # shape: (1, C, H, W)
    weighted_sum = np.sum(channel_weights * conv_features_np, axis=1)  # shape: (1, H, W)
    shap_cam = np.maximum(weighted_sum, 0)  # Apply ReLU
    shap_cam = shap_cam[0]  # shape: (H, W)
    
    # Upsample to input resolution and normalize.
    shap_cam = cv2.resize(shap_cam, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    shap_cam = (shap_cam - shap_cam.min()) / (shap_cam.max() - shap_cam.min() + 1e-8)
    return shap_cam

# ---------------------------
# TransferLearningModel for InceptionV3 with SHAP-CAM Support (Re-defined)
# ---------------------------
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        self.base_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None)
        if pretrained:
            self.base_model.aux_logits = False
            self.base_model.AuxLogits = None
        disable_inplace_relu(self.base_model)
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
# Main SHAP-CAM Visualization Routine
# ---------------------------
def main():
    # Load the model from the .pth checkpoint.
    model = TransferLearningModel(NUM_CLASSES, pretrained=True)
    checkpoint = torch.load(PTH_MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    print("Loaded InceptionV3 model from .pth successfully.")
    
    # List test images.
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print("No test images found.")
        return

    # Prepare a background tensor for SHAP (use the first image in the test folder).
    bg_path = os.path.join(TEST_IMAGES_DIR, test_images[0])
    bg_bgr = cv2.imread(bg_path)
    if bg_bgr is None:
        raise ValueError(f"Unable to load background image from {bg_path}")
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    background_tensor = preprocess_image(bg_rgb).to(DEVICE)
    
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        print(f"Processing image: {img_path}")
        
        # Load display image using Ben Graham’s method.
        try:
            display_image = load_ben_color(img_path)
        except Exception as e:
            print(f"Error processing display image {img_path}: {e}")
            continue
        
        # Load a separate copy for model input and preprocess it.
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            print(f"Unable to load image {img_path}. Skipping.")
            continue
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(rgb_image).to(DEVICE)
        
        # Get model predictions.
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Compute SHAP-CAM heatmap for the predicted class using our deep approach.
        shap_cam = generate_shapcam_deep(model, input_tensor, background_tensor, pred_class)
        shap_heatmap = apply_heatmap(shap_cam)
        overlay = overlay_heatmap_on_image(display_image, shap_heatmap, alpha=0.5)
        
        # Create and save the SHAP-CAM visualization.
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(display_image)
        plt.title("Ben Graham Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(shap_heatmap)
        plt.title("SHAP-CAM Heatmap")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPredicted: {LABEL_NAMES[pred_class]}\nConfidence: {confidence*100:.2f}%")
        plt.axis("off")
        plt.tight_layout()
        output_filename = os.path.splitext(img_file)[0] + "_shapcam.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP-CAM visualization for {img_file} to {output_path}")

if __name__ == "__main__":
    main()
