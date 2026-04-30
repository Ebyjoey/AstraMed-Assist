"""
AstraMed Assist - Monte Carlo Dropout Uncertainty
===================================================
Estimates prediction uncertainty via stochastic inference.
The model is kept in train() mode so dropout layers remain active,
and we run N forward passes to sample the predictive distribution.

U = mean variance over N passes = E[Var(P | X)]
"""

import torch
import numpy as np
from typing import Tuple


def mc_dropout_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 20,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Monte Carlo Dropout inference.

    Args:
        model: AstramedModel with dropout layers
        input_tensor: (1, 3, 224, 224) preprocessed tensor
        n_passes: Number of stochastic forward passes
        device: torch device string

    Returns:
        mean_probs: (3,) mean probabilities [Pneumonia, TB, Normal]
        std_probs:  (3,) std deviation per class
        uncertainty: scalar — mean variance across classes
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # Enable dropout by setting model to train mode
    # but avoid BN update by manually enabling only dropout
    _enable_dropout(model)

    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            samples.append(probs.cpu().numpy())

    # Restore eval mode
    model.eval()

    samples = np.stack(samples, axis=0)  # (N, 1, 3)
    samples = samples.squeeze(1)         # (N, 3)

    mean_probs  = samples.mean(axis=0)   # (3,)
    var_probs   = samples.var(axis=0)    # (3,)
    std_probs   = samples.std(axis=0)    # (3,)
    uncertainty = float(var_probs.mean())

    return mean_probs, std_probs, uncertainty


def _enable_dropout(model: torch.nn.Module):
    """Enable dropout layers while keeping BN in eval mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def uncertainty_to_confidence(uncertainty: float, scale: float = 10.0) -> float:
    """
    Map uncertainty [0, ∞) to confidence [0, 1].
    Uses exponential decay: confidence = exp(-scale * uncertainty)
    """
    return float(np.exp(-scale * uncertainty))


def interpret_uncertainty(uncertainty: float) -> dict:
    """
    Provide human-readable interpretation of uncertainty.

    Returns:
        dict with 'level', 'label', 'recommendation'
    """
    if uncertainty < 0.01:
        return {
            "level": "very_low",
            "label": "Very Low Uncertainty",
            "recommendation": "Model is highly confident in its prediction.",
            "should_review": False,
        }
    elif uncertainty < 0.03:
        return {
            "level": "low",
            "label": "Low Uncertainty",
            "recommendation": "Model prediction is reliable.",
            "should_review": False,
        }
    elif uncertainty < 0.06:
        return {
            "level": "moderate",
            "label": "Moderate Uncertainty",
            "recommendation": "Consider supplementary clinical information.",
            "should_review": True,
        }
    elif uncertainty < 0.10:
        return {
            "level": "high",
            "label": "High Uncertainty",
            "recommendation": "Manual radiologist review strongly recommended.",
            "should_review": True,
        }
    else:
        return {
            "level": "very_high",
            "label": "Very High Uncertainty",
            "recommendation": "Model output unreliable — mandatory expert review required.",
            "should_review": True,
        }
