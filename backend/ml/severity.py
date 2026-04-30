"""
AstraMed Assist - Severity & Triage Scoring
============================================
Implements the composite triage formula from the paper:

    T = α·Σ(Pi·Si) + β·max(Pi) + γ·(1 − U)

where:
    Pi = disease probability (from model)
    Si = severity coefficient (from Grad-CAM)
    U  = uncertainty (from MC Dropout variance)
    α=0.5, β=0.3, γ=0.2

Triage categories:
    High   : T ≥ 0.65
    Medium : 0.35 ≤ T < 0.65
    Low    : T < 0.35
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


# ─── Triage Constants ────────────────────────────────────────────────────────

ALPHA = 0.5   # Weight for severity-weighted disease probability
BETA  = 0.3   # Weight for peak disease probability
GAMMA = 0.2   # Weight for prediction confidence (1 - uncertainty)

TRIAGE_HIGH_THRESHOLD   = 0.65
TRIAGE_MEDIUM_THRESHOLD = 0.35

CLASS_NAMES = ["Pneumonia", "Tuberculosis", "Normal"]


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class DiseaseResult:
    """Per-class prediction result."""
    name: str
    probability: float
    severity: float
    is_detected: bool   # probability > threshold


@dataclass
class TriageResult:
    """Complete triage assessment for a chest X-ray."""
    triage_score: float
    triage_level: str          # "High", "Medium", "Low"
    triage_emoji: str          # 🔴, 🟡, 🟢
    overall_severity: float    # Weighted severity [0, 1]
    uncertainty: float         # MC Dropout uncertainty
    confidence: float          # 1 - uncertainty
    diseases: Dict[str, DiseaseResult] = field(default_factory=dict)
    max_probability: float = 0.0
    primary_finding: str = "Normal"
    clinical_urgency: str = "Routine"


# ─── Triage Calculator ───────────────────────────────────────────────────────

class TriageCalculator:
    """
    Computes composite triage score from model outputs.

    Args:
        alpha: Weight for Σ(Pi·Si) term
        beta: Weight for max(Pi) term
        gamma: Weight for confidence term
        detection_threshold: Probability threshold for positive detection
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        beta: float  = BETA,
        gamma: float = GAMMA,
        detection_threshold: float = 0.5,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"Weights must sum to 1. Got α={alpha}, β={beta}, γ={gamma}"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = detection_threshold

    def compute(
        self,
        probabilities: np.ndarray,
        severities: np.ndarray,
        uncertainty: float,
    ) -> TriageResult:
        """
        Compute complete triage result.

        Args:
            probabilities: (3,) array [P_pneumonia, P_tb, P_normal]
            severities:    (3,) array [S_pneumonia, S_tb, S_normal]
            uncertainty:   scalar — MC Dropout variance (mean over classes)

        Returns:
            TriageResult
        """
        probabilities = np.clip(probabilities, 0, 1)
        severities = np.clip(severities, 0, 1)
        uncertainty = float(np.clip(uncertainty, 0, 1))

        # For triage, only pathological classes contribute (not Normal)
        # Indices: 0=Pneumonia, 1=TB
        pathological_mask = np.array([1.0, 1.0, 0.0])

        # Term 1: Severity-weighted disease probabilities Σ(Pi·Si)
        weighted_sum = float(np.sum(probabilities * severities * pathological_mask))

        # Term 2: Max disease probability (across pathological classes)
        max_prob = float(np.max(probabilities * pathological_mask + (1 - pathological_mask) * -1))
        max_prob = max(0.0, max_prob)

        # Term 3: Model confidence (1 - uncertainty)
        confidence = 1.0 - uncertainty

        # Composite triage score
        T = self.alpha * weighted_sum + self.beta * max_prob + self.gamma * confidence
        T = float(np.clip(T, 0, 1))

        # Triage level
        triage_level, triage_emoji = self._classify_triage(T)

        # Overall severity (weighted mean of pathological severities)
        if max_prob > 0:
            weights = probabilities * pathological_mask
            if weights.sum() > 1e-6:
                overall_severity = float(np.average(severities, weights=weights))
            else:
                overall_severity = 0.0
        else:
            overall_severity = 0.0

        # Per-disease results
        diseases = {}
        for i, name in enumerate(["Pneumonia", "Tuberculosis", "Normal"]):
            diseases[name] = DiseaseResult(
                name=name,
                probability=float(probabilities[i]),
                severity=float(severities[i]),
                is_detected=bool(probabilities[i] >= self.threshold),
            )

        # Primary finding
        detected = [n for n, d in diseases.items() if d.is_detected and n != "Normal"]
        if detected:
            primary = detected[0] if len(detected) == 1 else " + ".join(detected)
        elif diseases["Normal"].is_detected:
            primary = "Normal"
        else:
            # Pick highest probability
            primary = CLASS_NAMES[int(np.argmax(probabilities))]

        # Clinical urgency text
        urgency = self._clinical_urgency(triage_level, primary)

        return TriageResult(
            triage_score=round(T, 4),
            triage_level=triage_level,
            triage_emoji=triage_emoji,
            overall_severity=round(overall_severity, 4),
            uncertainty=round(uncertainty, 4),
            confidence=round(confidence, 4),
            diseases=diseases,
            max_probability=round(max_prob, 4),
            primary_finding=primary,
            clinical_urgency=urgency,
        )

    @staticmethod
    def _classify_triage(score: float) -> tuple:
        if score >= TRIAGE_HIGH_THRESHOLD:
            return "High", "🔴"
        elif score >= TRIAGE_MEDIUM_THRESHOLD:
            return "Medium", "🟡"
        else:
            return "Low", "🟢"

    @staticmethod
    def _clinical_urgency(level: str, finding: str) -> str:
        if level == "High":
            return f"Immediate radiologist review required — suspected {finding}"
        elif level == "Medium":
            return f"Scheduled review recommended — possible {finding}"
        else:
            return "Routine follow-up — no urgent findings detected"


# ─── Severity Interpretation ─────────────────────────────────────────────────

def severity_label(score: float) -> str:
    """Human-readable severity interpretation."""
    if score >= 0.7:
        return "Severe"
    elif score >= 0.4:
        return "Moderate"
    elif score >= 0.2:
        return "Mild"
    else:
        return "Minimal"


def probability_confidence_label(prob: float) -> str:
    """Human-readable confidence label for a probability."""
    if prob >= 0.85:
        return "Very High Confidence"
    elif prob >= 0.70:
        return "High Confidence"
    elif prob >= 0.50:
        return "Moderate Confidence"
    elif prob >= 0.30:
        return "Low Confidence"
    else:
        return "Very Low Confidence"


# ─── Standalone API ──────────────────────────────────────────────────────────

def compute_triage(
    probabilities: list,
    severities: list,
    uncertainty: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> dict:
    """
    Convenience function returning a JSON-serialisable triage dict.

    Args:
        probabilities: [P_pneumonia, P_tb, P_normal]
        severities:    [S_pneumonia, S_tb, S_normal]
        uncertainty:   MC Dropout mean variance

    Returns:
        dict suitable for API response
    """
    calc = TriageCalculator(alpha=alpha, beta=beta, gamma=gamma)
    result = calc.compute(
        np.array(probabilities),
        np.array(severities),
        uncertainty,
    )

    return {
        "triage_score": result.triage_score,
        "triage_level": result.triage_level,
        "triage_emoji": result.triage_emoji,
        "overall_severity": result.overall_severity,
        "severity_label": severity_label(result.overall_severity),
        "uncertainty": result.uncertainty,
        "confidence": result.confidence,
        "primary_finding": result.primary_finding,
        "clinical_urgency": result.clinical_urgency,
        "diseases": {
            name: {
                "probability": d.probability,
                "confidence_label": probability_confidence_label(d.probability),
                "severity": d.severity,
                "severity_label": severity_label(d.severity),
                "is_detected": d.is_detected,
            }
            for name, d in result.diseases.items()
        }
    }


if __name__ == "__main__":
    # Example
    result = compute_triage(
        probabilities=[0.87, 0.12, 0.05],  # High pneumonia
        severities=[0.65, 0.10, 0.02],
        uncertainty=0.04,
    )
    import json
    print(json.dumps(result, indent=2))
