"""
AstraMed Assist - Grad-CAM Implementation
==========================================
Generates class-discriminative activation maps.
Used for:
  1. Visual explainability (heatmap overlay on X-ray)
  2. Severity coefficient computation (Si)

Reference: Selvaraju et al. (ICCV 2017)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple


# ─── Hook-Based Grad-CAM ─────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM for a specified target layer.

    Usage:
        gradcam = GradCAM(model, target_layer=model.features.denseblock4)
        heatmap = gradcam(input_tensor, class_idx=0)  # class_idx: 0=pneumonia, 1=tb, 2=normal
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM for a given class.

        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor
            class_idx: Target class index (0=pneumonia, 1=tb, 2=normal)
            smooth: Apply Gaussian smoothing to heatmap

        Returns:
            heatmap: (H, W) float array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(input_tensor)

        # Target is the logit for the specified class
        target = logits[:, class_idx]
        target.backward()

        # Pool gradients over spatial dimensions
        gradients = self._gradients          # (1, C, H, W)
        activations = self._activations      # (1, C, H, W)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input size
        H, W = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if smooth:
            cam = cv2.GaussianBlur(cam, (7, 7), 0)
            cam = np.clip(cam, 0, 1)

        return cam


# ─── Target Layer Resolver ───────────────────────────────────────────────────

def get_gradcam_target_layer(model: nn.Module, backbone: str = "densenet121") -> nn.Module:
    """Return the last feature convolutional layer for Grad-CAM."""
    if "densenet" in backbone:
        # Last dense block in DenseNet
        return model.features.denseblock4
    elif "efficientnet" in backbone:
        # Last MBConv block
        return model.features[-1]
    else:
        # Fallback: last layer of features
        layers = list(model.features.children())
        return layers[-1]


# ─── Heatmap Overlay ─────────────────────────────────────────────────────────

def create_heatmap_overlay(
    original_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: (H, W, 3) uint8 RGB image
        cam: (H, W) float array [0, 1]
        alpha: heatmap transparency
        colormap: OpenCV colormap

    Returns:
        overlay: (H, W, 3) uint8 RGB blended image
    """
    h, w = original_img.shape[:2]

    # Resize CAM to match image
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap (JET: blue→green→yellow→red)
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), colormap
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # convert to RGB

    # Blend
    overlay = (alpha * heatmap.astype(float) + (1 - alpha) * original_img.astype(float))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) normalised tensor to uint8 RGB array."""
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ─── Severity Score Computation ──────────────────────────────────────────────

def compute_severity_score(
    cam: np.ndarray,
    threshold: float = 0.5,
    lung_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute severity coefficient Si from Grad-CAM activation map.

    Si = Σ[A(x,y) > τ] · M(x,y) / Σ M(x,y)

    Where:
        A(x,y) = Grad-CAM activation at pixel (x,y)
        τ       = activation threshold (default 0.5)
        M(x,y) = lung mask (all 1s if not provided)

    Returns:
        float in [0, 1] representing abnormality extent
    """
    if lung_mask is None:
        # Simple approximation: use centre region of image as lung proxy
        h, w = cam.shape
        lung_mask = np.zeros_like(cam, dtype=float)
        # Central 70% region (approximate lung bounds)
        margin_h, margin_w = int(h * 0.15), int(w * 0.10)
        lung_mask[margin_h:h - margin_h, margin_w:w - margin_w] = 1.0

    high_activation = (cam > threshold).astype(float)
    numerator = (high_activation * lung_mask).sum()
    denominator = lung_mask.sum()

    if denominator < 1e-6:
        return 0.0

    return float(numerator / denominator)


# ─── Multi-Class Heatmap Generation ─────────────────────────────────────────

def generate_all_heatmaps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backbone: str = "densenet121",
    threshold: float = 0.5,
) -> dict:
    """
    Generate Grad-CAM heatmaps for all 3 classes.

    Returns:
        dict with keys: 'pneumonia', 'tb', 'normal'
        Each value: {'cam': np.ndarray, 'severity': float, 'overlay': np.ndarray}
    """
    target_layer = get_gradcam_target_layer(model, backbone)
    gradcam = GradCAM(model, target_layer)

    class_names = ["pneumonia", "tb", "normal"]
    original_img = tensor_to_rgb(input_tensor.squeeze(0))

    results = {}
    for i, name in enumerate(class_names):
        cam = gradcam(input_tensor, class_idx=i)
        severity = compute_severity_score(cam, threshold=threshold)
        overlay = create_heatmap_overlay(original_img, cam)
        results[name] = {
            "cam": cam,
            "severity": severity,
            "overlay": overlay,
        }

    return results, original_img


# ─── PIL Conversion Helpers ──────────────────────────────────────────────────

def overlay_to_pil(overlay: np.ndarray) -> Image.Image:
    """Convert overlay numpy array to PIL Image."""
    return Image.fromarray(overlay)


def cam_to_pil(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> Image.Image:
    """Convert raw CAM to colourised PIL Image."""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap)
