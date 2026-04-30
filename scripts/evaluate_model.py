"""
AstraMed Assist - Evaluation Script
=====================================
Computes full evaluation metrics on the test set:
  - Accuracy, Precision, Recall, F1 (macro/weighted)
  - AUC-ROC per class and mean
  - Sensitivity, Specificity per class
  - Confusion Matrix (saved as PNG)
  - ROC Curves (saved as PNG)
  - Per-class performance table

Usage:
    python scripts/evaluate_model.py \
        --model outputs/models/best_model.pth \
        --data data/processed/master_dataset.csv \
        --output outputs/evaluation
"""

import os
import argparse
import logging
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

from backend.ml.model import load_model
from backend.ml.dataset import build_dataloaders, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["pneumonia", "tb", "normal"]


def evaluate_model(
    model_path: str,
    csv_path: str,
    output_dir: str,
    backbone: str = "densenet121",
    batch_size: int = 32,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on: {device}")

    # Load model
    model = load_model(model_path, backbone=backbone, device=str(device))

    # Load test data
    loaders = build_dataloaders(csv_path, batch_size=batch_size)
    test_loader = loaders["test"]

    # Collect predictions
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)   # (N, 3)
    all_probs  = np.vstack(all_probs)    # (N, 3)

    # Convert to primary class predictions
    primary_true = all_labels.argmax(axis=1)
    primary_pred = all_probs.argmax(axis=1)

    # ── Overall Metrics ──
    acc       = accuracy_score(primary_true, primary_pred)
    prec_mac  = precision_score(primary_true, primary_pred, average="macro", zero_division=0)
    rec_mac   = recall_score(primary_true, primary_pred, average="macro", zero_division=0)
    f1_mac    = f1_score(primary_true, primary_pred, average="macro", zero_division=0)
    f1_wt     = f1_score(primary_true, primary_pred, average="weighted", zero_division=0)

    # ── Per-class AUC ──
    aucs = {}
    for i, cls in enumerate(CLASS_NAMES):
        try:
            aucs[cls] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            aucs[cls] = 0.5
    mean_auc = float(np.mean(list(aucs.values())))

    # ── Per-class Sensitivity & Specificity ──
    cm = confusion_matrix(primary_true, primary_pred)
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        per_class[cls] = {
            "precision": precision_score(primary_true == i, primary_pred == i, zero_division=0),
            "recall":    sensitivity,
            "specificity": specificity,
            "f1":        f1_score(primary_true == i, primary_pred == i, zero_division=0),
            "auc":       aucs[cls],
        }

    # ── Print Results ──
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:          {acc:.4f}")
    logger.info(f"Precision (Macro): {prec_mac:.4f}")
    logger.info(f"Recall (Macro):    {rec_mac:.4f}")
    logger.info(f"F1 Macro:          {f1_mac:.4f}")
    logger.info(f"F1 Weighted:       {f1_wt:.4f}")
    logger.info(f"Mean AUC:          {mean_auc:.4f}")
    logger.info("\nPer-Class Results:")
    for cls, m in per_class.items():
        logger.info(
            f"  {cls:12s}: Prec={m['precision']:.4f} Rec={m['recall']:.4f} "
            f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}"
        )

    # ── Save Results JSON ──
    results = {
        "overall": {
            "accuracy": acc, "precision_macro": prec_mac,
            "recall_macro": rec_mac, "f1_macro": f1_mac,
            "f1_weighted": f1_wt, "mean_auc": mean_auc,
        },
        "per_class": per_class,
        "aucs": aucs,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Confusion Matrix Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#F9FAFB')

    ax = axes[0]
    ax.set_facecolor('white')
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix\n(Accuracy: {acc:.4f})', fontsize=13, fontweight='bold')
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── ROC Curves ──
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    colors = ['#1A56DB', '#DC2626', '#7C3AED']

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        try:
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
            ax2.plot(fpr, tpr, linewidth=2.5, color=color,
                     label=f'{cls} (AUC={aucs[cls]:.3f})')
            ax2.fill_between(fpr, tpr, alpha=0.05, color=color)
        except Exception:
            pass

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.50)')
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title(f'ROC Curves\n(Mean AUC: {mean_auc:.4f})', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

    plt.suptitle('AstraMed Assist — Model Evaluation', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'),
                dpi=150, bbox_inches='tight', facecolor='#F9FAFB')
    plt.close()

    logger.info(f"\n✅ Results saved to: {output_dir}/metrics.json")
    logger.info(f"✅ Plots saved to: {output_dir}/evaluation_plots.png")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="outputs/models/best_model.pth")
    parser.add_argument("--data",     default="data/processed/master_dataset.csv")
    parser.add_argument("--output",   default="outputs/evaluation")
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        csv_path=args.data,
        output_dir=args.output,
        backbone=args.backbone,
        batch_size=args.batch_size,
    )
