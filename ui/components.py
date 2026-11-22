"""
UI Components and Visual Blocks for the Digital Forensic AI Console.
Provides headers, verdict cards, image plate grids, and architecture summaries.
"""

import io
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

def render_forensic_header(active_engine_name: str, device_name: str):
    """Renders the top forensic console status banner."""
    header_html = f"""
    <div class="forensic-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.75rem;">
            <div>
                <h1 class="forensic-title">
                    AI-Generated Image Detector
                </h1>
                <div class="forensic-subtitle">
                    AI-Generated Image Detection with Spatial Grad-CAM Analysis
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.6rem; background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.3); padding: 0.35rem 0.75rem; border-radius: 4px;">
                <span class="spec-dot"></span>
                <span style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #38bdf8; font-weight: 600;">System Operational</span>
            </div>
        </div>
        <div class="forensic-specs">
            <div class="forensic-spec-item">
                <span style="color: #64748b;">Detector:</span>
                <span style="color: #f1f5f9; font-weight: 500;">{active_engine_name}</span>
            </div>
            <div class="forensic-spec-item">
                <span style="color: #64748b;">Compute:</span>
                <span style="color: #00f0ff; font-weight: 500;">{device_name.upper()}</span>
            </div>
            <div class="forensic-spec-item">
                <span style="color: #64748b;">Analysis:</span>
                <span style="color: #f1f5f9; font-weight: 500;">Spatial Grad-CAM</span>
            </div>
            <div class="forensic-spec-item">
                <span style="color: #64748b;">Frequency:</span>
                <span style="color: #f1f5f9; font-weight: 500;">Fourier Transform Spectrum</span>
            </div>
            <div class="forensic-spec-item">
                <span style="color: #64748b;">Threshold:</span>
                <span style="color: #f59e0b; font-weight: 500;">0.55 (Calibrated)</span>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_top_navbar(current_page: str):
    """
    Renders the unified top horizontal navigation bar:
    - Project identity on the left
    - Horizontal navigation links in the center
    - Quick CTA 'Start Detection' on the right
    """
    from ui.pipeline import navigate_to

    col_brand, col_nav, col_cta = st.columns([1.5, 5.0, 1.2], vertical_alignment="center")

    with col_brand:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.55rem; padding: 0.15rem 0;">
                <div style="font-size: 1.4rem;">🔬</div>
                <div>
                    <div style="font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.05rem; color: #ffffff; letter-spacing: -0.01em; line-height: 1.2;">
                        AI Image Detector
                    </div>
                    <div style="font-size: 0.68rem; color: #00f0ff; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;">
                        Spatio-Frequency Fusion
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_nav:
        nav_options = ["Home", "Detect Image", "Quadrant Split", "Analysis", "Segment Analysis", "Performance", "History & Reports", "About"]
        selected = st.pills(
            "Primary Navigation",
            options=nav_options,
            default=current_page,
            required=True,
            label_visibility="collapsed"
        )
        if selected and selected != current_page:
            navigate_to(selected)

    with col_cta:
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            navigate_to("Detect Image", "Single Image Detection")


def render_prediction_result_card(
    image: Image.Image,
    label: str,
    confidence: float,
    on_fft_click=None,
    on_gradcam_click=None,
    on_report_click=None,
    on_reset_click=None
):
    """
    Renders the streamlined, prominent prediction result layout:
    1. Uploaded image preview
    2. One clear, prominent result: REAL or AI-GENERATED
    3. Model confidence percentage
    4. Compact visual confidence bar
    5. User-friendly explanation
    6. Action buttons (FFT Analysis, Grad-CAM, Generate Report, Analyze Another)
    """
    is_ai = (label == "AI-GENERATED")
    res_color = "#ff2d55" if is_ai else "#10b981"
    bg_gradient = (
        "linear-gradient(135deg, rgba(30, 15, 25, 0.85), rgba(15, 23, 42, 0.95))"
        if is_ai else
        "linear-gradient(135deg, rgba(10, 30, 25, 0.85), rgba(15, 23, 42, 0.95))"
    )
    border_color = "rgba(255, 45, 85, 0.45)" if is_ai else "rgba(16, 185, 129, 0.45)"
    badge_label = "AI-GENERATED" if is_ai else "AUTHENTIC REAL"
    badge_icon = "🤖" if is_ai else "🌿"
    
    explanation_text = (
        "The Spatio-Frequency Fusion model detected synthetic artifact patterns in the frequency spectrum and spatial boundary transitions characteristic of generative diffusion models."
        if is_ai else
        "The Spatio-Frequency Fusion model verified continuous organic frequency distribution and natural camera sensor characteristics consistent with authentic photography."
    )

    col_img, col_res = st.columns([1, 1.4], gap="medium")

    with col_img:
        st.image(image, caption=f"Selected Image ({image.width}×{image.height}px)", use_container_width=True)

    with col_res:
        st.markdown(
            f"""
            <div style="background: {bg_gradient}; border: 1px solid {border_color}; border-radius: 8px; padding: 1.35rem 1.6rem; margin-bottom: 0.85rem;">
                <div style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;">
                    Detection Result
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {res_color}; margin: 0.25rem 0 0.4rem 0; letter-spacing: -0.01em;">
                    {badge_icon} {badge_label}
                </div>
                <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.6rem;">
                    <span style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">{confidence * 100:.1f}%</span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Model Confidence</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.1); border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 1rem;">
                    <div style="background: {res_color}; width: {confidence * 100:.1f}%; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.55;">
                    {explanation_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        act1, act2 = st.columns(2)
        with act1:
            if st.button("🌊 View FFT Analysis", use_container_width=True):
                if on_fft_click: on_fft_click()
            if st.button("📄 Generate Report", use_container_width=True):
                if on_report_click: on_report_click()
        with act2:
            if st.button("🔥 View Grad-CAM", use_container_width=True):
                if on_gradcam_click: on_gradcam_click()
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                if on_reset_click: on_reset_click()



def render_evidence_plates(image: Image.Image, fft_spectrum: np.ndarray, overlay: np.ndarray, heatmap: np.ndarray):
    """
    Renders the three side-by-side forensic visual plates:
    1. Original Input Image
    2. Grad-CAM Heatmap
    3. Grad-CAM Spatial Heatmap Overlay
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="image-plate-box">
                <div class="image-plate-label" style="font-family: 'Inter', sans-serif;">
                    <span>Plate 1: Original Evidence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        sha = hashlib.sha256(img_bytes.getvalue()).hexdigest()[:16]
        st.caption(f"Dim: {image.width}×{image.height}px | SHA-256: {sha}...")

    with col2:
        st.markdown(
            """
            <div class="image-plate-box">
                <div class="image-plate-label" style="font-family: 'Inter', sans-serif;">
                    <span>Plate 2: Grad-CAM Heatmap</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(heatmap, use_container_width=True)
        st.caption("Class-discriminative activation localization.")

    with col3:
        st.markdown(
            """
            <div class="image-plate-box">
                <div class="image-plate-label" style="font-family: 'Inter', sans-serif;">
                    <span>Plate 3: Heatmap Overlay</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(overlay, use_container_width=True)
        st.caption("Overlay highlighting synthesis artifacts.")


def render_radar_scanner(message: str = "INITIALIZING FORENSIC DEEP INFERENCE ENGINE..."):
    """Renders a high-tech forensic radar spinner."""
    return f"""
    <div class="forensic-scanner">
        <div class="radar-circle"></div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #00f0ff; letter-spacing: 0.08em; font-weight: 600;">
            {message}
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #64748b; margin-top: 0.4rem;">
            Analyzing image properties...
        </div>
    </div>
    """

def render_page_header(title: str, subtitle: str, module_badge: str = "MODULE"):
    """Renders a standard module page header for multi-page consistency."""
    header_html = f"""
    <div style="margin-bottom: 1.25rem; border-bottom: 1px solid rgba(0, 240, 255, 0.15); padding-bottom: 0.75rem;">
        <span style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #00f0ff; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;">
            {module_badge}
        </span>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 1.45rem; font-weight: 700; color: #ffffff; margin: 0.2rem 0 0.35rem 0;">
            {title}
        </h2>
        <p style="font-size: 0.84rem; color: #94a3b8; margin: 0; line-height: 1.5;">
            {subtitle}
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_disclaimer_notice():
    """Renders a standard probabilistic disclaimer banner."""
    st.markdown(
        """
        <div style="margin-top: 1.5rem; padding: 0.65rem 0.9rem; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 6px; font-size: 0.75rem; color: #94a3b8; line-height: 1.5;">
            <b style="color: #38bdf8;">Forensic Notice:</b> Model outputs represent probabilistic estimations based on spatio-frequency feature representations. They serve as investigative indicators and should be evaluated alongside contextual provenance, EXIF metadata, and cross-modal forensic verification.
        </div>
        """,
        unsafe_allow_html=True
    )

