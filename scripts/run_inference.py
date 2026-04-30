"""
AstraMed Assist - Standalone Inference
========================================
Run inference on a single chest X-ray from the command line.

Usage:
    python scripts/run_inference.py \
        --image path/to/xray.jpg \
        --model outputs/models/best_model.pth \
        --output_dir outputs/inference \
        --save_heatmap \
        --generate_report \
        --patient_name "John Doe" \
        --patient_age 45 \
        --patient_gender Male
"""

import os
import sys
import json
import argparse
import base64
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.model import load_model
from backend.ml.dataset import preprocess_image_path
from backend.ml.gradcam import generate_all_heatmaps, overlay_to_pil
from backend.ml.severity import compute_triage
from backend.ml.uncertainty import mc_dropout_inference, interpret_uncertainty


def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Image:  {args.image}")

    # Load model
    model = load_model(args.model, backbone=args.backbone, device=device)
    print(f"Model loaded: {args.model}")

    # Preprocess
    input_tensor = preprocess_image_path(args.image)
    input_tensor = input_tensor.to(device)

    # MC Dropout inference
    print(f"Running MC Dropout ({args.mc_passes} passes)...")
    mean_probs, std_probs, uncertainty = mc_dropout_inference(
        model, input_tensor, n_passes=args.mc_passes, device=device
    )

    # Grad-CAM
    print("Computing Grad-CAM heatmaps...")
    heatmap_results, original_img = generate_all_heatmaps(
        model, input_tensor, backbone=args.backbone, threshold=args.threshold
    )

    severities = [
        heatmap_results["pneumonia"]["severity"],
        heatmap_results["tb"]["severity"],
        heatmap_results["normal"]["severity"],
    ]

    # Triage
    triage = compute_triage(
        probabilities=mean_probs.tolist(),
        severities=severities,
        uncertainty=uncertainty,
    )

    # Uncertainty
    unc_info = interpret_uncertainty(uncertainty)

    # ── Print Results ──
    print("\n" + "="*55)
    print("  ASTRAMED ASSIST — INFERENCE RESULTS")
    print("="*55)
    print(f"  Image:          {os.path.basename(args.image)}")
    print(f"  Primary Finding: {triage['primary_finding']}")
    print(f"  Triage Level:   {triage['triage_level']} (score: {triage['triage_score']:.3f})")
    print(f"  Severity:       {triage['severity_label']} ({triage['overall_severity']:.3f})")
    print(f"  Confidence:     {triage['confidence']*100:.1f}%")
    print(f"  Uncertainty:    {uncertainty:.4f} ({unc_info['label']})")
    print()
    print("  Disease Probabilities:")
    print(f"    Pneumonia:    {mean_probs[0]*100:.1f}% ± {std_probs[0]*100:.1f}%")
    print(f"    Tuberculosis: {mean_probs[1]*100:.1f}% ± {std_probs[1]*100:.1f}%")
    print(f"    Normal:       {mean_probs[2]*100:.1f}% ± {std_probs[2]*100:.1f}%")
    print()
    print(f"  Recommendation: {triage['clinical_urgency']}")
    print("="*55)

    # ── Save Heatmap ──
    if args.save_heatmap:
        primary_idx = int(np.argmax(mean_probs))
        primary_key = ["pneumonia", "tb", "normal"][primary_idx]
        overlay = heatmap_results[primary_key]["overlay"]
        overlay_pil = overlay_to_pil(overlay)
        heatmap_path = os.path.join(args.output_dir, "heatmap_overlay.png")
        overlay_pil.save(heatmap_path)
        print(f"Heatmap saved: {heatmap_path}")

    # ── Save JSON Results ──
    results = {
        "image": args.image,
        "probabilities": {
            "Pneumonia":    float(mean_probs[0]),
            "Tuberculosis": float(mean_probs[1]),
            "Normal":       float(mean_probs[2]),
        },
        "std": {
            "Pneumonia":    float(std_probs[0]),
            "Tuberculosis": float(std_probs[1]),
            "Normal":       float(std_probs[2]),
        },
        "severities": {
            "Pneumonia":    float(severities[0]),
            "Tuberculosis": float(severities[1]),
            "Normal":       float(severities[2]),
        },
        "triage": triage,
        "uncertainty": {"value": float(uncertainty), **unc_info},
        "timestamp": datetime.now().isoformat(),
    }

    json_path = os.path.join(args.output_dir, "inference_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {json_path}")

    # ── Generate PDF Report ──
    if args.generate_report:
        try:
            from backend.utils.pdf_report import generate_report
            patient_info = {
                "name": args.patient_name,
                "age":  str(args.patient_age) if args.patient_age else "",
                "gender": args.patient_gender,
                "patient_id": args.patient_id or "CLI-INFERENCE",
                "referring_physician": args.physician,
                "clinical_indication": args.indication,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            with open(args.image, "rb") as f:
                img_bytes = f.read()

            # Get heatmap bytes
            primary_idx = int(np.argmax(mean_probs))
            primary_key = ["pneumonia", "tb", "normal"][primary_idx]
            import io
            buf = io.BytesIO()
            overlay_to_pil(heatmap_results[primary_key]["overlay"]).save(buf, format="PNG")
            heat_bytes = buf.getvalue()

            report_path = os.path.join(args.output_dir, "clinical_report.pdf")
            generate_report(report_path, patient_info, triage, img_bytes, heat_bytes)
            print(f"PDF report saved: {report_path}")
        except Exception as e:
            print(f"Report generation failed: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstraMed Standalone Inference")
    parser.add_argument("--image",      required=True, help="Path to chest X-ray image")
    parser.add_argument("--model",      default="outputs/models/best_model.pth")
    parser.add_argument("--backbone",   default="densenet121")
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--mc_passes",  type=int, default=20)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--save_heatmap",    action="store_true")
    parser.add_argument("--generate_report", action="store_true")
    parser.add_argument("--patient_name",    default="Anonymous")
    parser.add_argument("--patient_id",      default="")
    parser.add_argument("--patient_age",     type=int, default=None)
    parser.add_argument("--patient_gender",  default="")
    parser.add_argument("--physician",       default="")
    parser.add_argument("--indication",      default="Routine screening")
    args = parser.parse_args()

    run_inference(args)
