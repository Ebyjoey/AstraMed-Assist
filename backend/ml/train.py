"""
AstraMed Assist - Training Loop
=================================
Complete training pipeline:
- BCEWithLogitsLoss
- Adam optimizer + cosine LR scheduler
- Early stopping
- Per-epoch metric logging (accuracy, F1, AUC) → CSV
- Best model checkpoint saving
"""

import os
import csv
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)

from backend.ml.model import AstramedModel, count_parameters
from backend.ml.dataset import build_dataloaders, LABEL_COLS, CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics for multi-label output.

    Args:
        labels: (N, 3) ground truth binary labels
        probs:  (N, 3) predicted probabilities
        threshold: decision threshold

    Returns:
        dict of metric name → value
    """
    preds = (probs >= threshold).astype(int)

    # Per-class AUC (handle cases where only one class present)
    aucs = []
    for i in range(labels.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    # Primary class prediction (argmax of probabilities)
    primary_true = labels.argmax(axis=1)
    primary_pred = probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(primary_true, primary_pred),
        "f1_macro": f1_score(primary_true, primary_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(primary_true, primary_pred, average="weighted", zero_division=0),
        "precision": precision_score(primary_true, primary_pred, average="macro", zero_division=0),
        "recall": recall_score(primary_true, primary_pred, average="macro", zero_division=0),
        "auc_pneumonia": aucs[0],
        "auc_tb": aucs[1],
        "auc_normal": aucs[2],
        "auc_mean": float(np.mean(aucs)),
    }
    return metrics


# ─── Training Step ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
) -> dict:
    """Run one training epoch. Returns loss and metric dict."""
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataloader. Returns loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    return metrics, all_labels, all_probs


# ─── Logger ──────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Writes per-epoch metrics to a CSV file."""

    FIELDNAMES = [
        "epoch", "train_loss", "val_loss",
        "train_accuracy", "val_accuracy",
        "train_f1_macro", "val_f1_macro",
        "train_auc_mean", "val_auc_mean",
        "lr", "epoch_time_s"
    ]

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log(self, epoch: int, train_m: dict, val_m: dict, lr: float, elapsed: float):
        row = {
            "epoch": epoch,
            "train_loss": f"{train_m['loss']:.4f}",
            "val_loss": f"{val_m['loss']:.4f}",
            "train_accuracy": f"{train_m['accuracy']:.4f}",
            "val_accuracy": f"{val_m['accuracy']:.4f}",
            "train_f1_macro": f"{train_m['f1_macro']:.4f}",
            "val_f1_macro": f"{val_m['f1_macro']:.4f}",
            "train_auc_mean": f"{train_m['auc_mean']:.4f}",
            "val_auc_mean": f"{val_m['auc_mean']:.4f}",
            "lr": f"{lr:.6f}",
            "epoch_time_s": f"{elapsed:.1f}",
        }
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─── Main Training Function ──────────────────────────────────────────────────

def train(
    csv_path: str,
    output_dir: str = "outputs",
    backbone: str = "densenet121",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    patience: int = 5,
    num_workers: int = 0,
    img_size: int = 224,
    use_amp: bool = True,
):
    """Full training pipeline."""
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ── Data ──
    loaders = build_dataloaders(csv_path, batch_size, num_workers, img_size)
    logger.info(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)}")

    # ── Model ──
    model = AstramedModel(backbone=backbone, num_classes=3, dropout=dropout)
    model.to(device)
    params = count_parameters(model)
    logger.info(f"Model: {backbone} | Params: {params['total']:,} | Trainable: {params['trainable']:,}")

    # ── Loss ──
    # Compute class weights from training data
    train_ds = loaders["train"].dataset
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # ── Optimiser ──
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # ── LR Scheduler ──
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    # ── AMP scaler ──
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # ── Logging ──
    training_logger = TrainingLogger(f"{output_dir}/logs/training_log.csv")
    early_stop = EarlyStopping(patience=patience)

    best_val_auc = 0.0
    best_epoch = 0

    # ── Training Loop ──
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scaler)
        val_metrics, _, _ = evaluate(model, loaders["val"], criterion, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        training_logger.log(epoch, train_metrics, val_metrics, current_lr, elapsed)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1_macro']:.4f}/{val_metrics['f1_macro']:.4f} | "
            f"AUC: {val_metrics['auc_mean']:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        scheduler.step()

        # ── Save Best Model ──
        if val_metrics["auc_mean"] > best_val_auc:
            best_val_auc = val_metrics["auc_mean"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "backbone": backbone,
                    "num_classes": 3,
                },
                f"{output_dir}/models/best_model.pth"
            )
            logger.info(f"  ✅ Best model saved (AUC={best_val_auc:.4f})")

        # ── Early Stopping ──
        if early_stop(val_metrics["loss"]):
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # ── Final Evaluation on Test Set ──
    logger.info("\n=== Final Test Evaluation ===")
    best_ckpt = f"{output_dir}/models/best_model.pth"
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_metrics, test_labels, test_probs = evaluate(model, loaders["test"], criterion, device)

    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Macro:  {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test AUC Mean:  {test_metrics['auc_mean']:.4f}")
    logger.info(f"  AUC Pneumonia: {test_metrics['auc_pneumonia']:.4f}")
    logger.info(f"  AUC TB:        {test_metrics['auc_tb']:.4f}")
    logger.info(f"  AUC Normal:    {test_metrics['auc_normal']:.4f}")
    logger.info(f"Best epoch: {best_epoch}")

    return model, test_metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AstraMed Model")
    parser.add_argument("--data", default="data/processed/master_dataset.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--backbone", default="densenet121", choices=["densenet121", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train(
        csv_path=args.data,
        output_dir=args.output,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
    )
