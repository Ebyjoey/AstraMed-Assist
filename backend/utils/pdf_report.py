"""
AstraMed Assist - PDF Report Generator
========================================
Generates structured clinical PDF reports using ReportLab.

Report contains:
  1. Header with AstraMed branding + patient info
  2. AI predictions with confidence bars
  3. Severity + triage assessment
  4. X-ray image with Grad-CAM heatmap
  5. Clinical summary
  6. Uncertainty + confidence metrics
  7. AI-generated disclaimer
"""

import os
import io
import base64
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.pdfgen import canvas as rlcanvas

# ─── Colour Palette ──────────────────────────────────────────────────────────

BRAND_BLUE   = colors.HexColor("#1A56DB")
BRAND_DARK   = colors.HexColor("#0F2044")
BRAND_LIGHT  = colors.HexColor("#EFF6FF")
ACCENT_RED   = colors.HexColor("#DC2626")
ACCENT_AMBER = colors.HexColor("#D97706")
ACCENT_GREEN = colors.HexColor("#16A34A")
GRAY_900     = colors.HexColor("#111827")
GRAY_600     = colors.HexColor("#4B5563")
GRAY_100     = colors.HexColor("#F3F4F6")
WHITE        = colors.white
BLACK        = colors.black

# ─── Triage Colours ──────────────────────────────────────────────────────────

TRIAGE_COLORS = {
    "High":   ACCENT_RED,
    "Medium": ACCENT_AMBER,
    "Low":    ACCENT_GREEN,
}

# ─── Styles ──────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "Title", fontSize=20, fontName="Helvetica-Bold",
            textColor=BRAND_DARK, spaceAfter=4, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", fontSize=11, fontName="Helvetica",
            textColor=GRAY_600, spaceAfter=12, alignment=TA_LEFT,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader", fontSize=12, fontName="Helvetica-Bold",
            textColor=BRAND_BLUE, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", fontSize=10, fontName="Helvetica",
            textColor=GRAY_900, leading=15, alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "Small", fontSize=8, fontName="Helvetica",
            textColor=GRAY_600, leading=12,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", fontSize=8, fontName="Helvetica-Oblique",
            textColor=GRAY_600, leading=12, borderPad=8,
            borderColor=GRAY_600, borderWidth=0.5,
        ),
        "label": ParagraphStyle(
            "Label", fontSize=9, fontName="Helvetica-Bold",
            textColor=GRAY_600, spaceAfter=2,
        ),
        "value": ParagraphStyle(
            "Value", fontSize=11, fontName="Helvetica-Bold",
            textColor=GRAY_900,
        ),
        "finding_high": ParagraphStyle(
            "FindingHigh", fontSize=13, fontName="Helvetica-Bold",
            textColor=ACCENT_RED,
        ),
        "finding_medium": ParagraphStyle(
            "FindingMedium", fontSize=13, fontName="Helvetica-Bold",
            textColor=ACCENT_AMBER,
        ),
        "finding_low": ParagraphStyle(
            "FindingLow", fontSize=13, fontName="Helvetica-Bold",
            textColor=ACCENT_GREEN,
        ),
    }
    return styles


# ─── Canvas Callback ─────────────────────────────────────────────────────────

class HeaderFooterCanvas:
    """Adds header/footer to every page."""

    def __init__(self, filename, **kwargs):
        self.canvas = rlcanvas.Canvas(filename, **kwargs)
        self.pages = []
        self.width, self.height = A4

    def showPage(self):
        self.pages.append(dict(self.canvas.__dict__))
        self.canvas._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.canvas.__dict__.update(page)
            self._draw_header()
            self._draw_footer(page_count)
            self.canvas.showPage()
        self.canvas.save()

    def _draw_header(self):
        c = self.canvas
        c.setFillColor(BRAND_DARK)
        c.rect(0, self.height - 22*mm, self.width, 22*mm, fill=True, stroke=False)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(15*mm, self.height - 12*mm, "AstraMed Assist")
        c.setFont("Helvetica", 8)
        c.drawString(15*mm, self.height - 18*mm, "AI-Powered Chest X-Ray Analysis System")
        c.setFont("Helvetica", 8)
        c.drawRightString(self.width - 15*mm, self.height - 12*mm,
                          f"Report Date: {datetime.now().strftime('%B %d, %Y  %H:%M')}")

    def _draw_footer(self, total_pages):
        c = self.canvas
        c.setFillColor(GRAY_100)
        c.rect(0, 0, self.width, 14*mm, fill=True, stroke=False)
        c.setFillColor(GRAY_600)
        c.setFont("Helvetica-Oblique", 7)
        c.drawString(15*mm, 8*mm,
                     "CONFIDENTIAL — AI-generated report for clinical decision support only. "
                     "Not a substitute for professional medical diagnosis.")
        c.setFont("Helvetica", 8)
        c.drawRightString(self.width - 15*mm, 8*mm,
                          f"Page {c._pageNumber} of {total_pages}")


# ─── Report Builder ──────────────────────────────────────────────────────────

def generate_report(
    output_path: str,
    patient_info: dict,
    predictions: dict,
    original_img_bytes: bytes,
    heatmap_img_bytes: Optional[bytes] = None,
) -> str:
    """
    Generate a structured clinical PDF report.

    Args:
        output_path: Where to save the PDF
        patient_info: dict with keys: name, age, gender, patient_id, referring_physician, date
        predictions: dict from compute_triage() — probabilities, severities, triage, etc.
        original_img_bytes: Raw bytes of uploaded X-ray
        heatmap_img_bytes: Optional heatmap overlay image bytes

    Returns:
        Path to the saved PDF
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    styles = build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=28*mm,
        bottomMargin=20*mm,
    )

    story = []

    # ── Patient Information Block ─────────────────────────────────────────────
    story.append(Paragraph("CHEST X-RAY ANALYSIS REPORT", styles["title"]))
    story.append(Paragraph(
        "Multi-Label AI Classification: Pneumonia · Tuberculosis · Normal",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=10))

    # Patient info table
    pid = patient_info.get("patient_id", "N/A")
    patient_data = [
        ["Patient Name", patient_info.get("name", "N/A"),
         "Patient ID",   pid],
        ["Age",          patient_info.get("age", "N/A"),
         "Gender",       patient_info.get("gender", "N/A")],
        ["Referring Physician", patient_info.get("referring_physician", "N/A"),
         "Study Date",   patient_info.get("date", datetime.now().strftime("%Y-%m-%d"))],
        ["Clinical Indication", patient_info.get("clinical_indication", "Routine screening"),
         "Modality",     "Digital Chest X-Ray (CXR)"],
    ]

    pt_table = Table(patient_data, colWidths=[42*mm, 58*mm, 42*mm, 40*mm])
    pt_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), GRAY_100),
        ("BACKGROUND",  (0, 0), (0, -1), BRAND_LIGHT),
        ("BACKGROUND",  (2, 0), (2, -1), BRAND_LIGHT),
        ("TEXTCOLOR",   (0, 0), (0, -1), BRAND_DARK),
        ("TEXTCOLOR",   (2, 0), (2, -1), BRAND_DARK),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("PADDING",     (0, 0), (-1, -1), 6),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [GRAY_100, WHITE]),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 12))

    # ── Triage Summary ────────────────────────────────────────────────────────
    story.append(Paragraph("TRIAGE ASSESSMENT", styles["section_header"]))

    triage_level = predictions.get("triage_level", "Low")
    triage_score = predictions.get("triage_score", 0.0)
    triage_color = TRIAGE_COLORS.get(triage_level, ACCENT_GREEN)
    primary_finding = predictions.get("primary_finding", "Normal")
    urgency = predictions.get("clinical_urgency", "Routine follow-up")

    triage_data = [
        [
            Paragraph(f"<b>TRIAGE LEVEL</b>", ParagraphStyle("", fontSize=9, textColor=GRAY_600)),
            Paragraph(f"<b>TRIAGE SCORE</b>", ParagraphStyle("", fontSize=9, textColor=GRAY_600)),
            Paragraph(f"<b>PRIMARY FINDING</b>", ParagraphStyle("", fontSize=9, textColor=GRAY_600)),
            Paragraph(f"<b>SEVERITY</b>", ParagraphStyle("", fontSize=9, textColor=GRAY_600)),
        ],
        [
            Paragraph(f"<b>{triage_level.upper()}</b>",
                      ParagraphStyle("", fontSize=18, fontName="Helvetica-Bold", textColor=triage_color)),
            Paragraph(f"<b>{triage_score:.3f}</b>",
                      ParagraphStyle("", fontSize=18, fontName="Helvetica-Bold", textColor=GRAY_900)),
            Paragraph(f"<b>{primary_finding}</b>",
                      ParagraphStyle("", fontSize=14, fontName="Helvetica-Bold", textColor=GRAY_900)),
            Paragraph(f"<b>{predictions.get('severity_label', 'Minimal')}</b>",
                      ParagraphStyle("", fontSize=14, fontName="Helvetica-Bold",
                                     textColor=ACCENT_RED if predictions.get('overall_severity', 0) > 0.5 else GRAY_900)),
        ]
    ]

    triage_table = Table(triage_data, colWidths=[45*mm, 40*mm, 55*mm, 42*mm])
    triage_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), BRAND_LIGHT),
        ("BACKGROUND",   (0, 1), (0, 1), colors.HexColor("#FEF2F2") if triage_level == "High"
                                   else colors.HexColor("#FFFBEB") if triage_level == "Medium"
                                   else colors.HexColor("#F0FDF4")),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",      (0, 0), (-1, -1), 10),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("LINEABOVE",    (0, 1), (-1, 1), 2, triage_color),
    ]))
    story.append(triage_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Clinical Recommendation:</b> {urgency}", styles["body"]))
    story.append(Spacer(1, 10))

    # ── Disease Probability Table ─────────────────────────────────────────────
    story.append(Paragraph("DISEASE CLASSIFICATION RESULTS", styles["section_header"]))

    diseases = predictions.get("diseases", {})
    prob_header = ["Disease", "Probability", "Confidence", "Severity", "Status"]
    prob_rows = [prob_header]

    for dname in ["Pneumonia", "Tuberculosis", "Normal"]:
        d = diseases.get(dname, {})
        prob = d.get("probability", 0.0)
        detected = d.get("is_detected", False)
        status_color = ACCENT_RED if (detected and dname != "Normal") else \
                       ACCENT_GREEN if (detected and dname == "Normal") else GRAY_600

        prob_rows.append([
            dname,
            f"{prob * 100:.1f}%",
            d.get("confidence_label", "—"),
            d.get("severity_label", "Minimal"),
            "DETECTED" if detected else "Not Detected",
        ])

    prob_table = Table(prob_rows, colWidths=[38*mm, 28*mm, 52*mm, 30*mm, 34*mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 0), (0, -1), "LEFT"),
        ("PADDING",      (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_100]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME",     (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 12))

    # ── Images ────────────────────────────────────────────────────────────────
    story.append(Paragraph("RADIOGRAPHIC IMAGES", styles["section_header"]))

    img_row_content = []

    # Original X-ray
    try:
        orig_img = Image.open(io.BytesIO(original_img_bytes)).convert("RGB")
        orig_img_resized = orig_img.resize((280, 280))
        orig_buf = io.BytesIO()
        orig_img_resized.save(orig_buf, format="JPEG")
        orig_buf.seek(0)

        orig_rl = RLImage(orig_buf, width=75*mm, height=75*mm)
        orig_label = Paragraph("Original Chest X-Ray", ParagraphStyle(
            "", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=GRAY_600
        ))
        img_row_content.append([orig_rl, orig_label])
    except Exception:
        img_row_content.append([Paragraph("X-ray image unavailable", styles["small"]), Paragraph("", styles["small"])])

    # Heatmap overlay
    if heatmap_img_bytes:
        try:
            heat_img = Image.open(io.BytesIO(heatmap_img_bytes)).convert("RGB")
            heat_img_resized = heat_img.resize((280, 280))
            heat_buf = io.BytesIO()
            heat_img_resized.save(heat_buf, format="JPEG")
            heat_buf.seek(0)

            heat_rl = RLImage(heat_buf, width=75*mm, height=75*mm)
            heat_label = Paragraph("Grad-CAM Activation Heatmap", ParagraphStyle(
                "", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=GRAY_600
            ))
            img_row_content.append([heat_rl, heat_label])
        except Exception:
            img_row_content.append([Paragraph("Heatmap unavailable", styles["small"]), Paragraph("", styles["small"])])
    else:
        img_row_content.append([Paragraph("Heatmap not generated", styles["small"]), Paragraph("", styles["small"])])

    # Layout the two images side by side
    img_table_data = [
        [img_row_content[0][0], img_row_content[1][0]],
        [img_row_content[0][1], img_row_content[1][1]],
    ]
    img_layout = Table(img_table_data, colWidths=[90*mm, 90*mm])
    img_layout.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_LIGHT),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("GRID",   (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))

    # Heatmap legend
    legend = Paragraph(
        "<b>Heatmap Interpretation:</b> Red/Warm regions indicate areas most influential "
        "for the model's prediction. These correspond to radiographic abnormalities "
        "such as consolidations, opacities, or cavitary lesions.",
        styles["small"]
    )

    story.append(KeepTogether([img_layout, Spacer(1, 6), legend]))
    story.append(Spacer(1, 12))

    # ── Uncertainty & Confidence ──────────────────────────────────────────────
    story.append(Paragraph("MODEL CONFIDENCE & UNCERTAINTY", styles["section_header"]))

    conf = predictions.get("confidence", 0.0)
    unc = predictions.get("uncertainty", 0.0)

    conf_data = [
        ["Metric", "Value", "Interpretation"],
        ["Model Confidence", f"{conf * 100:.1f}%",
         "High" if conf > 0.85 else "Moderate" if conf > 0.65 else "Low"],
        ["Prediction Uncertainty", f"{unc:.4f}",
         "Low" if unc < 0.03 else "Moderate" if unc < 0.06 else "High"],
        ["MC Dropout Passes", "20 stochastic forward passes", "Bayesian approximation"],
        ["Confidence Interval", "95% CI: Accuracy 0.901–0.915 | AUC 0.971–0.980",
         "Reported on 20,000 image test set"],
    ]

    conf_table = Table(conf_data, colWidths=[52*mm, 60*mm, 70*mm])
    conf_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 0), (0, -1), "LEFT"),
        ("PADDING",      (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_100]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 12))

    # ── Clinical Summary ──────────────────────────────────────────────────────
    story.append(Paragraph("CLINICAL SUMMARY", styles["section_header"]))

    summary = _generate_clinical_summary(predictions, patient_info)
    story.append(Paragraph(summary, styles["body"]))
    story.append(Spacer(1, 10))

    # ── Model Information ─────────────────────────────────────────────────────
    story.append(Paragraph("AI MODEL INFORMATION", styles["section_header"]))

    model_info = [
        ["Architecture",    "DenseNet-121 (pretrained ImageNet)"],
        ["Output",          "3-class sigmoid multi-label classifier"],
        ["Training Data",   "NIH ChestX-ray14, CheXpert, RSNA Pneumonia, TBX11K (n=20,000)"],
        ["Performance",     "Accuracy: 90.83% | F1: 91.53% | AUC: 0.9756"],
        ["Triage Formula",  "T = 0.5·Σ(Pi·Si) + 0.3·max(Pi) + 0.2·(1-U)"],
        ["Reference",       "Sakthi U. et al., ICAIHC 2026"],
    ]

    mi_table = Table(model_info, colWidths=[48*mm, 134*mm])
    mi_table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (0, -1), BRAND_DARK),
        ("PADDING",    (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT, WHITE]),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
    ]))
    story.append(mi_table)
    story.append(Spacer(1, 12))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT_AMBER, spaceAfter=8))
    disclaimer_text = (
        "<b>⚠ IMPORTANT DISCLAIMER:</b> This report is AI-generated by AstraMed Assist "
        "and is intended solely as a clinical decision-support tool. It does NOT constitute a "
        "medical diagnosis and must NOT be used as a standalone basis for clinical decisions. "
        "All findings must be reviewed and validated by a qualified, licensed radiologist or "
        "physician before any clinical action is taken. The AI model may produce false positives "
        "or false negatives. This system has not received regulatory approval (FDA/CE/CDSCO) "
        "as a medical device and is intended for research and decision-support purposes only. "
        "Patient management decisions remain the sole responsibility of the treating clinician."
    )
    story.append(Paragraph(disclaimer_text, styles["disclaimer"]))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story)
    return output_path


def _generate_clinical_summary(predictions: dict, patient_info: dict) -> str:
    """Generate a plain-English clinical summary from predictions."""
    name = patient_info.get("name", "The patient")
    age = patient_info.get("age", "")
    gender = patient_info.get("gender", "")

    diseases = predictions.get("diseases", {})
    triage = predictions.get("triage_level", "Low")
    primary = predictions.get("primary_finding", "Normal")
    urgency = predictions.get("clinical_urgency", "Routine follow-up")
    severity_label = predictions.get("severity_label", "Minimal")
    confidence = predictions.get("confidence", 0.0)

    pneu = diseases.get("Pneumonia", {})
    tb   = diseases.get("Tuberculosis", {})
    norm = diseases.get("Normal", {})

    findings = []
    if pneu.get("is_detected"):
        findings.append(
            f"Pneumonia (probability {pneu.get('probability', 0)*100:.1f}%, "
            f"severity: {pneu.get('severity_label', 'Mild')})"
        )
    if tb.get("is_detected"):
        findings.append(
            f"Tuberculosis (probability {tb.get('probability', 0)*100:.1f}%, "
            f"severity: {tb.get('severity_label', 'Mild')})"
        )

    desc = age + ("-year-old " if age else "") + gender.lower() + " " if (age or gender) else ""

    if findings:
        finding_str = " and ".join(findings)
        summary = (
            f"The AI system analysis of {name}'s {desc}chest X-ray demonstrates findings "
            f"consistent with {finding_str}. "
            f"The radiographic severity is classified as {severity_label.lower()}, with an "
            f"overall triage score of {predictions.get('triage_score', 0):.3f} ({triage} priority). "
            f"The model confidence for this prediction is {confidence * 100:.1f}%. "
            f"{urgency}. "
            f"Grad-CAM activation maps highlight the regions of highest predictive significance. "
            f"These findings should be correlated with clinical history, symptoms, and additional "
            f"investigations as deemed appropriate by the attending physician."
        )
    else:
        summary = (
            f"The AI system analysis of {name}'s {desc}chest X-ray does not demonstrate "
            f"significant findings consistent with Pneumonia or Tuberculosis. "
            f"The overall appearance is most consistent with a Normal chest X-ray "
            f"(probability {norm.get('probability', 0)*100:.1f}%). "
            f"Triage classification: {triage} priority (score: {predictions.get('triage_score', 0):.3f}). "
            f"Model confidence: {confidence * 100:.1f}%. "
            f"Routine follow-up is recommended as clinically indicated. These findings should be "
            f"reviewed in the context of the full clinical picture."
        )

    return summary
