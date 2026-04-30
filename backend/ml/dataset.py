"""
AstraMed Assist - Dataset & Preprocessing
==========================================
PyTorch Dataset class with full preprocessing pipeline:
- Resize 224×224
- 3-channel (grayscale → RGB)
- ImageNet normalisation
- Augmentations: HFlip, ±10° rotation, contrast jitter, Gaussian noise
"""

import os
import io
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─── ImageNet Statistics ─────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Label Columns & Class Map ───────────────────────────────────────────────

LABEL_COLS = ["pneumonia", "tb", "normal"]
CLASS_NAMES = ["Pneumonia", "Tuberculosis", "Normal"]

# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224) -> T.Compose:
    """Augmented transform for training."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Lambda(lambda img: img.convert("RGB")),  # ensure 3-channel
        T.RandomHorizontalFlip(p=0.5),             # paper: no vertical flip
        T.RandomRotation(degrees=10),              # ±10° as per paper
        T.ColorJitter(brightness=0.2, contrast=0.2),  # contrast jitter
        T.ToTensor(),
        GaussianNoise(std=0.02),                   # custom Gaussian noise
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 224) -> T.Compose:
    """Deterministic transform for validation / test."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(img_size: int = 224) -> T.Compose:
    """Single-image inference transform (same as val)."""
    return get_val_transform(img_size)


class GaussianNoise:
    """Add Gaussian noise to a tensor image."""
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


# ─── Dataset Class ────────────────────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    Multi-label chest X-ray dataset.

    Args:
        csv_path: Path to CSV with columns (path, pneumonia, tb, normal, split)
        split: 'train', 'val', or 'test'
        transform: torchvision transform pipeline
        img_size: Target image size
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        transform: Optional[T.Compose] = None,
        img_size: int = 224,
    ):
        self.img_size = img_size
        self.split = split

        df = pd.read_csv(csv_path)
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

        # Validate required columns
        missing = [c for c in LABEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transform(img_size)
        else:
            self.transform = get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["path"]

        # Load and convert image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return black image on failure
            img = Image.new("RGB", (self.img_size, self.img_size), color=0)

        # Apply transforms
        img_tensor = self.transform(img)

        # Labels as float tensor for BCEWithLogitsLoss
        labels = torch.tensor(
            [row["pneumonia"], row["tb"], row["normal"]], dtype=torch.float32
        )
        return img_tensor, labels

    def get_class_weights(self) -> torch.Tensor:
        """Compute positive class weights for imbalanced BCE loss."""
        pos_counts = self.df[LABEL_COLS].sum(axis=0).values
        neg_counts = len(self.df) - pos_counts
        weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)
        return weights

    def get_sample_info(self, idx: int) -> dict:
        """Return metadata for a given sample."""
        row = self.df.iloc[idx]
        return {
            "path": row["path"],
            "pneumonia": int(row["pneumonia"]),
            "tb": int(row["tb"]),
            "normal": int(row["normal"]),
            "source": row.get("source", "unknown"),
        }


# ─── DataLoader Factory ──────────────────────────────────────────────────────

def build_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
) -> dict:
    """
    Create train / val / test DataLoaders.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = ChestXRayDataset(csv_path=csv_path, split=split, img_size=img_size)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders


# ─── Preprocessing Utilities ─────────────────────────────────────────────────

def preprocess_image_bytes(img_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    """
    Preprocess raw image bytes for inference.

    Args:
        img_bytes: Raw bytes of the image file
        img_size: Target size

    Returns:
        tensor: (1, 3, H, W) normalized tensor
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = get_inference_transform(img_size)
    tensor = transform(img).unsqueeze(0)  # Add batch dim
    return tensor


def preprocess_image_path(img_path: str, img_size: int = 224) -> torch.Tensor:
    """Preprocess image from file path for inference."""
    with open(img_path, "rb") as f:
        return preprocess_image_bytes(f.read(), img_size)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: (3, H, W) normalized tensor

    Returns:
        np.ndarray: (H, W, 3) uint8 image
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


if __name__ == "__main__":
    # Smoke test with a dummy CSV
    import tempfile, os

    # Create a tiny dummy dataset
    dummy_img = Image.new("RGB", (256, 256), color=(128, 64, 64))
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.jpg")
        dummy_img.save(img_path)

        df = pd.DataFrame([{
            "path": img_path,
            "pneumonia": 1,
            "tb": 0,
            "normal": 0,
            "split": "train"
        }])
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)

        ds = ChestXRayDataset(csv_path=csv_path, split="train")
        img, labels = ds[0]
        print(f"Image shape: {img.shape}")
        print(f"Labels: {labels}")
        print(f"Transform applied successfully!")
