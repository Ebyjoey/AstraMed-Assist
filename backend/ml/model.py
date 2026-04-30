"""
AstraMed Assist - Model Definition
====================================
DenseNet-121 backbone with 3-class sigmoid multi-label head.
Supports EfficientNet-B0 as alternative backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AstramedModel(nn.Module):
    """
    Multi-label chest X-ray classifier.

    Architecture:
        - Backbone: DenseNet-121 or EfficientNet-B0 (pretrained ImageNet)
        - Classifier head: Linear → BN → ReLU → Dropout → Linear(3)
        - Output activation: Sigmoid (per-class independent probability)

    Outputs 3 probabilities: [pneumonia, tb, normal]
    """

    BACKBONES = {
        "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, 1024),
        "densenet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1, 1664),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 1408),
    }

    def __init__(
        self,
        backbone: str = "densenet121",
        num_classes: int = 3,
        dropout: float = 0.5,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(self.BACKBONES)}")

        model_fn, weights, feature_dim = self.BACKBONES[backbone]

        # Load pretrained backbone
        base_model = model_fn(weights=weights)

        # Extract feature extractor (remove classifier)
        if "densenet" in backbone:
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif "efficientnet" in backbone:
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = feature_dim

        # Optionally freeze early layers (transfer learning with partial fine-tuning)
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Weight initialisation for classifier head
        self._init_classifier()

    def _freeze_layers(self, num_layers: int):
        """Freeze first `num_layers` parameter groups of the backbone."""
        params = list(self.features.parameters())
        for p in params[:num_layers]:
            p.requires_grad = False

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalized tensor
        Returns:
            logits: (B, 3) — raw logits before sigmoid
        """
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid-activated probabilities."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Returns last conv feature maps (for Grad-CAM)."""
        return self.features(x)

    def mc_forward(self, x: torch.Tensor, n_passes: int = 20) -> tuple:
        """
        Monte Carlo Dropout forward pass for uncertainty estimation.

        Args:
            x: (B, 3, 224, 224)
            n_passes: Number of stochastic forward passes

        Returns:
            mean_probs: (B, 3) mean probabilities
            uncertainty: (B, 3) variance across passes
        """
        self.train()  # Enable dropout during inference
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward(x)
                preds.append(torch.sigmoid(logits))
        preds = torch.stack(preds, dim=0)  # (n_passes, B, 3)
        mean_probs = preds.mean(dim=0)     # (B, 3)
        uncertainty = preds.var(dim=0)     # (B, 3)
        self.eval()
        return mean_probs, uncertainty


def load_model(
    checkpoint_path: str,
    backbone: str = "densenet121",
    num_classes: int = 3,
    device: str = "cpu",
) -> AstramedModel:
    """Load model from checkpoint file."""
    model = AstramedModel(backbone=backbone, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    # Quick smoke test
    model = AstramedModel(backbone="densenet121")
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Probs:  {torch.sigmoid(logits)}")
    mean_p, unc = model.mc_forward(x, n_passes=10)
    print(f"MC Mean: {mean_p}")
    print(f"MC Unc:  {unc}")
    params = count_parameters(model)
    print(f"Parameters: {params}")
