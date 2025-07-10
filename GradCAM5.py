import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.models import vit_b_16, ViT_B_16_Weights

# ---------------------------
# Constants and Paths
# ---------------------------
NUM_CLASSES = 7
MODEL_INPUT_SIZE = 224  # ViT-B/16 expects 224x224 images by default
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\model"
# Use the .pth checkpoint from training
PTH_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "ViT_B_16_best.pth")
TEST_IMAGES_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test\images"
OUTPUT_DIR = r"D:\BoneFracture\Dataset\BoneFractureYolo8\results\ViT-Grad-CAM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class labels (order must match training)
LABEL_NAMES = [
    'elbow positive', 'fingers positive', 'forearm fracture',
    'humerus', 'shoulder fracture', 'wrist positive', 'no fracture'
]

# ---------------------------
# Helper Functions for Preprocessing and Display
# ---------------------------
def preprocess_image(image):
    """
    Resize and normalize the image for ViT-B16 inference.
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
    Both images must have the same dimensions.
    """
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ---------------------------
# TransferLearning Model for ViT-B/16
# ---------------------------
class TransferLearningViTModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningViTModel, self).__init__()
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace the classification head with a new linear layer
        num_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(num_features, num_classes)
        # (Optional) Freeze backbone parameters if needed
    def forward(self, x):
        return self.base_model(x)

# ---------------------------
# GradCAM++ Implementation for ViT-B/16
# ---------------------------
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        """
        model: an instance of TransferLearningViTModel
        target_layer: the layer to hook (we use base_model.conv_proj for spatial features)
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, input_tensor, target_class):
        """
        Computes GradCAM++ for the target class.
        Args:
            input_tensor: tensor of shape (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            target_class: integer index of the target class
        Returns:
            cam: GradCAM++ heatmap tensor (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            output: model logits
        """
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)  # shape: [1, num_classes]
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        activations = self.activations   # shape: [1, C, h, w]
        gradients = self.gradients       # same shape as activations
        epsilon = 1e-7
        grad2 = gradients ** 2
        grad3 = gradients ** 3
        numerator = grad2
        denominator = 2 * grad2 + (activations * grad3).sum(dim=(2,3), keepdim=True) + epsilon
        alpha = numerator / denominator
        weights = (alpha * torch.relu(gradients)).sum(dim=(2,3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + epsilon)
        return cam, output

# ---------------------------
# Main GradCAM++ Visualization Routine
# ---------------------------
def main():
    # Load the trained ViT model from checkpoint
    model = TransferLearningViTModel(NUM_CLASSES, pretrained=True)
    checkpoint = torch.load(PTH_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    print("Loaded ViT-B/16 model from checkpoint successfully.")
    
    # Initialize GradCAM++ using the patch embedding layer as the target.
    gradcam = GradCAMPlusPlus(model, model.base_model.conv_proj)
    
    # List test images
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print("No test images found.")
        return
    
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        print(f"Processing image: {img_path}")
        
        # Load image using OpenCV and convert to RGB
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            print(f"Failed to load image {img_path}. Skipping.")
            continue
        display_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Resize display image to MODEL_INPUT_SIZE (224x224) for proper overlay
        display_image = cv2.resize(display_image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        
        # Preprocess the image for model input
        input_tensor = preprocess_image(display_image).to(DEVICE)
        input_tensor.requires_grad_()  # Ensure gradients are tracked
        
        # Get model predictions
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        # Compute GradCAM++ heatmap for the predicted class
        cam, _ = gradcam.generate_cam(input_tensor, target_class=pred_class)
        cam_np = cam.squeeze().cpu().numpy()
        heatmap = apply_heatmap(cam_np)
        overlay = overlay_heatmap_on_image(display_image, heatmap, alpha=0.5)
        
        # Create and save the visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(display_image)
        plt.title("Original Image")
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
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved GradCAM++ visualization for {img_file} to {output_path}")
    
    gradcam.remove_hooks()

if __name__ == "__main__":
    main()
