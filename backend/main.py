"""
AstraMed Assist - FastAPI Backend
===================================
Endpoints:
  POST /predict         — classify chest X-ray
  POST /generate-report — create PDF report

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Internal modules
from backend.ml.model import AstramedModel, load_model
from backend.ml.dataset import preprocess_image_bytes, IMAGENET_MEAN, IMAGENET_STD
from backend.ml.gradcam import generate_all_heatmaps, overlay_to_pil
from backend.ml.severity import compute_triage
from backend.ml.uncertainty import mc_dropout_inference, interpret_uncertainty
from backend.utils.pdf_report import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH   = os.environ.get("MODEL_PATH",   "outputs/models/best_model.pth")
BACKBONE     = os.environ.get("BACKBONE",     "densenet121")
REPORTS_DIR  = os.environ.get("REPORTS_DIR",  "outputs/reports")
HEATMAPS_DIR = os.environ.get("HEATMAPS_DIR", "outputs/heatmaps")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AstraMed Assist API",
    description="Multi-label chest X-ray classification with triage and reporting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ────────────────────────────────────────────────────────────

model: Optional[AstramedModel] = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH, backbone=BACKBONE, device=DEVICE)
            logger.info(f"✅ Model loaded from {MODEL_PATH} on {DEVICE}")
        except Exception as e:
            logger.warning(f"Failed to load model from {MODEL_PATH}: {e}")
            logger.warning("Creating a fresh (untrained) model for demo purposes.")
            model = AstramedModel(backbone=BACKBONE)
            model.to(DEVICE)
            model.eval()
    else:
        logger.warning(f"No model checkpoint found at {MODEL_PATH}.")
        logger.warning("Creating a fresh (untrained) model for demo purposes.")
        model = AstramedModel(backbone=BACKBONE)
        model.to(DEVICE)
        model.eval()


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


# ─── Predict Endpoint ─────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)"),
    mc_passes: int   = Form(default=20, description="Monte Carlo Dropout passes"),
    threshold: float = Form(default=0.5, description="Detection threshold"),
):
    """
    Classify a chest X-ray image.

    Returns:
    - Disease probabilities (Pneumonia, TB, Normal)
    - Severity scores per class
    - Triage level (High / Medium / Low)
    - Grad-CAM heatmap as base64 PNG
    - Uncertainty from Monte Carlo Dropout
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read image
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        # Preprocess
        input_tensor = preprocess_image_bytes(img_bytes).to(DEVICE)

        # Monte Carlo Dropout inference
        mean_probs, std_probs, uncertainty = mc_dropout_inference(
            model, input_tensor, n_passes=mc_passes, device=DEVICE
        )

        # Grad-CAM heatmaps + severity
        heatmap_results, original_img = generate_all_heatmaps(
            model, input_tensor, backbone=BACKBONE, threshold=threshold
        )

        severities = [
            heatmap_results["pneumonia"]["severity"],
            heatmap_results["tb"]["severity"],
            heatmap_results["normal"]["severity"],
        ]

        # Triage computation
        triage = compute_triage(
            probabilities=mean_probs.tolist(),
            severities=severities,
            uncertainty=uncertainty,
        )

        # Get primary class heatmap overlay (whichever has highest probability)
        primary_idx = int(np.argmax(mean_probs))
        primary_key = ["pneumonia", "tb", "normal"][primary_idx]
        primary_overlay = heatmap_results[primary_key]["overlay"]

        # Encode heatmap overlay as base64
        overlay_pil = overlay_to_pil(primary_overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Also save heatmap to disk
        heatmap_id = str(uuid.uuid4())[:8]
        heatmap_path = os.path.join(HEATMAPS_DIR, f"{heatmap_id}_heatmap.png")
        overlay_pil.save(heatmap_path)

        # Uncertainty interpretation
        unc_info = interpret_uncertainty(uncertainty)

        return {
            "success": True,
            "probabilities": {
                "Pneumonia":    round(float(mean_probs[0]), 4),
                "Tuberculosis": round(float(mean_probs[1]), 4),
                "Normal":       round(float(mean_probs[2]), 4),
            },
            "std_probabilities": {
                "Pneumonia":    round(float(std_probs[0]), 4),
                "Tuberculosis": round(float(std_probs[1]), 4),
                "Normal":       round(float(std_probs[2]), 4),
            },
            "severities": {
                "Pneumonia":    round(float(severities[0]), 4),
                "Tuberculosis": round(float(severities[1]), 4),
                "Normal":       round(float(severities[2]), 4),
            },
            "triage": triage,
            "uncertainty": {
                "value": round(uncertainty, 6),
                **unc_info,
            },
            "heatmap": {
                "primary_class": primary_key,
                "image_base64": heatmap_b64,
                "mime_type": "image/png",
                "saved_path": heatmap_path,
            },
            "metadata": {
                "model_backbone": BACKBONE,
                "device": DEVICE,
                "mc_passes": mc_passes,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
            }
        }

    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─── Report Endpoint ──────────────────────────────────────────────────────────

@app.post("/generate-report")
async def generate_report_endpoint(
    file: UploadFile = File(..., description="Chest X-ray image"),
    patient_name: str = Form(default="Anonymous"),
    patient_id: str   = Form(default=""),
    age: str          = Form(default=""),
    gender: str       = Form(default=""),
    referring_physician: str = Form(default=""),
    clinical_indication: str = Form(default="Routine screening"),
    mc_passes: int    = Form(default=20),
    threshold: float  = Form(default=0.5),
):
    """
    Generate a complete clinical PDF report for a chest X-ray.

    Runs the full inference pipeline and composes a structured PDF.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        # Full inference pipeline (same as /predict)
        input_tensor = preprocess_image_bytes(img_bytes).to(DEVICE)
        mean_probs, std_probs, uncertainty = mc_dropout_inference(
            model, input_tensor, n_passes=mc_passes, device=DEVICE
        )
        heatmap_results, original_img = generate_all_heatmaps(
            model, input_tensor, backbone=BACKBONE, threshold=threshold
        )
        severities = [
            heatmap_results["pneumonia"]["severity"],
            heatmap_results["tb"]["severity"],
            heatmap_results["normal"]["severity"],
        ]
        triage = compute_triage(
            probabilities=mean_probs.tolist(),
            severities=severities,
            uncertainty=uncertainty,
        )

        # Primary heatmap
        primary_idx = int(np.argmax(mean_probs))
        primary_key = ["pneumonia", "tb", "normal"][primary_idx]
        primary_overlay = heatmap_results[primary_key]["overlay"]
        overlay_pil = overlay_to_pil(primary_overlay)
        heatmap_buf = io.BytesIO()
        overlay_pil.save(heatmap_buf, format="PNG")
        heatmap_bytes = heatmap_buf.getvalue()

        # Generate PDF
        report_id = str(uuid.uuid4())[:12]
        report_filename = f"AstraMed_Report_{report_id}.pdf"
        report_path = os.path.join(REPORTS_DIR, report_filename)

        patient_info = {
            "name": patient_name or "Anonymous",
            "patient_id": patient_id or report_id,
            "age": age,
            "gender": gender,
            "referring_physician": referring_physician,
            "clinical_indication": clinical_indication,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        generate_report(
            output_path=report_path,
            patient_info=patient_info,
            predictions=triage,
            original_img_bytes=img_bytes,
            heatmap_img_bytes=heatmap_bytes,
        )

        logger.info(f"Report generated: {report_path}")
        return FileResponse(
            path=report_path,
            media_type="application/pdf",
            filename=report_filename,
            headers={"X-Report-ID": report_id},
        )

    except Exception as e:
        logger.exception(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ─── List Reports ─────────────────────────────────────────────────────────────

@app.get("/reports")
async def list_reports():
    """List all generated reports."""
    reports = []
    for f in Path(REPORTS_DIR).glob("*.pdf"):
        reports.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
        })
    return {"reports": sorted(reports, key=lambda x: x["created"], reverse=True)}


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
