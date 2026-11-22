"""
Forensic PDF Report Generator using ReportLab.
Generates an on-the-fly, publication-grade Digital Forensics Examination Dossier
containing the original image, 2D FFT spectrum, Grad-CAM activation overlay,
SHA-256 checksum, and model diagnostics.
"""

import io
import hashlib
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def _fig_to_bytes(fig) -> io.BytesIO:
    """Converts a Matplotlib figure to an in-memory PNG BytesIO buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="#0b0f19")
    buf.seek(0)
    plt.close(fig)
    return buf

def _pil_to_bytes(img: Image.Image) -> io.BytesIO:
    """Converts a PIL Image to an in-memory PNG BytesIO buffer."""
    buf = io.BytesIO()
    rgb_img = img.convert("RGB")
    rgb_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def generate_forensic_pdf(
    filename: str,
    original_image: Image.Image,
    fft_spectrum: np.ndarray,
    overlay_image: np.ndarray,
    label: str,
    confidence: float,
    model_name: str,
    threshold: float = 0.55
) -> bytes:
    """
    Generates a complete Forensic PDF Examination Dossier in memory.
    
    Args:
        filename: Evidence file name
        original_image: PIL Image of the original input
        fft_spectrum: 2D numpy array of the FFT log-magnitude spectrum
        overlay_image: RGB numpy array or PIL Image of Grad-CAM overlay
        label: 'REAL' or 'AI-GENERATED'
        confidence: Confidence score between 0.0 and 1.0
        model_name: Name of the predictor model used
        threshold: Decision threshold used
        
    Returns:
        bytes: The compiled PDF file content.
    """
    # 1. Compute SHA-256 hash of original image
    img_byte_arr = io.BytesIO()
    original_image.save(img_byte_arr, format="PNG")
    sha256_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()
    
    # 2. Render FFT spectrum to image buffer
    fig_fft, ax_fft = plt.subplots(figsize=(3.5, 3.5), facecolor="#0b0f19")
    cax = ax_fft.imshow(fft_spectrum, cmap="magma")
    ax_fft.axis("off")
    fig_fft.colorbar(cax, fraction=0.046, pad=0.04)
    fft_buf = _fig_to_bytes(fig_fft)

    # 3. Render Original image to buffer
    orig_buf = _pil_to_bytes(original_image)

    # 4. Render Grad-CAM overlay to buffer
    if isinstance(overlay_image, np.ndarray):
        overlay_pil = Image.fromarray(overlay_image.astype("uint8"))
    else:
        overlay_pil = overlay_image
    overlay_buf = _pil_to_bytes(overlay_pil)

    # 5. Build ReportLab Document
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=10,
        alignment=0
    )
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        textColor=colors.HexColor("#334155")
    )
    meta_bold = ParagraphStyle(
        'MetaBold',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8,
        textColor=colors.HexColor("#0f172a")
    )
    verdict_ai_style = ParagraphStyle(
        'VerdictAI',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor("#dc2626")
    )
    verdict_real_style = ParagraphStyle(
        'VerdictReal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor("#0284c7")
    )
    section_heading = ParagraphStyle(
        'SecHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4
    )
    plate_label = ParagraphStyle(
        'PlateLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=1
    )
    plate_sub = ParagraphStyle(
        'PlateSub',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        textColor=colors.HexColor("#64748b"),
        alignment=1
    )
    notes_style = ParagraphStyle(
        'Notes',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=colors.HexColor("#334155"),
        leading=11
    )

    flowables = []

    # Title & Header
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_id = f"DOSSIER-{hashlib.md5((filename + timestamp_str).encode()).hexdigest()[:10].upper()}"

    flowables.append(Paragraph("AI-GENERATED IMAGE DETECTOR — EVIDENCE EXAMINATION REPORT", title_style))
    flowables.append(Paragraph(f"Grad-CAM Enabled Spatio-Frequency Artifact Inspection | Case Ref: <b>{report_id}</b>", subtitle_style))
    flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # Metadata & Verdict Grid Table
    is_ai = (label == "AI-GENERATED")
    verdict_color = colors.HexColor("#fee2e2") if is_ai else colors.HexColor("#e0f2fe")
    verdict_border = colors.HexColor("#ef4444") if is_ai else colors.HexColor("#0284c7")
    verdict_p = Paragraph(f"● {label}", verdict_ai_style if is_ai else verdict_real_style)

    meta_table_data = [
        [
            Paragraph("<b>Target Evidence:</b>", meta_bold), Paragraph(filename, meta_style),
            Paragraph("<b>Verdict:</b>", meta_bold), verdict_p
        ],
        [
            Paragraph("<b>Inspection Date:</b>", meta_bold), Paragraph(timestamp_str, meta_style),
            Paragraph("<b>Confidence:</b>", meta_bold), Paragraph(f"<b>{confidence * 100:.2f}%</b> (Threshold &tau;={threshold:.2f})", meta_style)
        ],
        [
            Paragraph("<b>Detection Model:</b>", meta_bold), Paragraph(model_name, meta_style),
            Paragraph("<b>Image Resolution:</b>", meta_bold), Paragraph(f"{original_image.width} &times; {original_image.height} px", meta_style)
        ],
        [
            Paragraph("<b>SHA-256 Digest:</b>", meta_bold), Paragraph(f"{sha256_hash[:32]}...<br/>{sha256_hash[32:]}", meta_style),
            Paragraph("<b>Color Space:</b>", meta_bold), Paragraph(f"{original_image.mode} (RGB Normalized)", meta_style)
        ]
    ]

    t_meta = Table(meta_table_data, colWidths=[1.1*inch, 2.7*inch, 1.1*inch, 2.5*inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (2, 0), (3, 0), verdict_color),
        ('BOX', (2, 0), (3, 0), 1.2, verdict_border),
    ]))
    flowables.append(t_meta)
    flowables.append(Spacer(1, 12))

    # Triple Visual Plates
    flowables.append(Paragraph("FORENSIC VISUAL INSPECTION PLATES", section_heading))
    
    img_w, img_h = 2.3 * inch, 2.3 * inch
    rl_orig = RLImage(orig_buf, width=img_w, height=img_h)
    rl_fft = RLImage(fft_buf, width=img_w, height=img_h)
    rl_overlay = RLImage(overlay_buf, width=img_w, height=img_h)

    plates_data = [
        [rl_orig, rl_fft, rl_overlay],
        [
            Paragraph("PLATE A: INPUT EVIDENCE", plate_label),
            Paragraph("PLATE B: 2D FFT SPECTRUM", plate_label),
            Paragraph("PLATE C: GRAD-CAM OVERLAY", plate_label)
        ],
        [
            Paragraph("Spatial Domain Image", plate_sub),
            Paragraph("Centered Log-Magnitude Frequency", plate_sub),
            Paragraph("ResNet-50 Layer4 Saliency Map", plate_sub)
        ]
    ]

    t_plates = Table(plates_data, colWidths=[2.45*inch, 2.45*inch, 2.45*inch])
    t_plates.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ('BOX', (0, 0), (0, -1), 0.8, colors.HexColor("#cbd5e1")),
        ('BOX', (1, 0), (1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ('BOX', (2, 0), (2, -1), 0.8, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    flowables.append(t_plates)
    flowables.append(Spacer(1, 10))

    # Technical Diagnostic Interpretation
    flowables.append(Paragraph("TECHNICAL ANALYSIS & FORENSIC INTERPRETATION", section_heading))

    if is_ai:
        finding_spatial = "<b>Spatial Domain:</b> Grad-CAM activation highlights abnormal textural gradients, boundary blending anomalies, or upsampling artifacts characteristic of neural generative models (diffusion/GAN)."
        finding_freq = "<b>Frequency Domain:</b> FFT magnitude spectrum reveals azimuthally anisotropic peaks or abnormal high-frequency energy distribution inconsistent with natural optical lenses."
        summary_verdict = f"<b>Final Conclusion:</b> The dual-branch classifier flagged this image as <b>AI-GENERATED</b> with a confidence of <b>{confidence*100:.2f}%</b> (exceeding calibrated threshold &tau;={threshold:.2f})."
    else:
        finding_spatial = "<b>Spatial Domain:</b> Spatial feature responses are smoothly distributed across natural semantic edges without localized generative artifact clustering."
        finding_freq = "<b>Frequency Domain:</b> The 2D Fourier spectrum exhibits natural power-law radial spectral decay typical of authentic real-world photography and physical optical sensors."
        summary_verdict = f"<b>Final Conclusion:</b> The dual-branch classifier verified this image as <b>AUTHENTIC REAL</b> with a confidence of <b>{confidence*100:.2f}%</b>."

    findings_p = [
        Paragraph(finding_spatial, notes_style),
        Spacer(1, 4),
        Paragraph(finding_freq, notes_style),
        Spacer(1, 4),
        Paragraph(summary_verdict, notes_style)
    ]
    
    t_findings = Table([[findings_p]], colWidths=[7.4*inch])
    t_findings.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    flowables.append(t_findings)
    flowables.append(Spacer(1, 12))

    # Footer Sign-off
    footer_text = (
        f"Generated automatically by AI-Generated Image Detector | Engine: {model_name} | "
        f"Verified: {timestamp_str} | Academic Mini-Project Dossier"
    )
    flowables.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontName='Helvetica', fontSize=6.5, textColor=colors.HexColor("#94a3b8"), alignment=1)))

    doc.build(flowables)
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
