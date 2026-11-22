"""
Home View / Overview Dashboard
Digital Forensic AI System Command Center.
"""

import os
import sys
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.pipeline import (
    get_compute_device,
    get_calibrated_threshold,
    ACTIVE_ENGINE_NAME,
    navigate_to
)
from ui.components import render_page_header, render_disclaimer_notice

def render_home_view():
    render_page_header(
        title="AI-Generated Image Detection Command Center",
        subtitle="Spatio-Frequency Dual-Branch Neural Architecture for Detecting Latent Diffusion Synthesis with Visual Explainability.",
        module_badge="SYSTEM OVERVIEW"
    )

    # Hero Callout & Primary Action
    c_hero, c_act = st.columns([2.5, 1.2])
    with c_hero:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.8)); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 8px; padding: 1.15rem 1.4rem; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
                <div style="font-size: 0.72rem; color: #00f0ff; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;">
                    Dual-Branch Spatio-Frequency Fusion
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff; margin: 0.35rem 0 0.5rem 0;">
                    Grad-CAM Enabled Detection of AI-Generated Images
                </div>
                <div style="font-size: 0.83rem; color: #cbd5e1; line-height: 1.6;">
                    Countering modern generative synthesis (Stable Diffusion, ADM, Midjourney) by fusing RGB spatial feature maps with 2D Fast Fourier Transform magnitude spectra through a learned gated attention mechanism.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_act:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 1.1rem 1.2rem; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; font-weight: 600;">PRIMARY WORKFLOW</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🚀 Start Single Detection", type="primary", use_container_width=True, key="home_btn_single"):
            navigate_to("Detect Image", "Single Image Detection")
        if st.button("⚡ Batch Evidence Scanner", use_container_width=True, key="home_btn_batch"):
            navigate_to("Detect Image", "Batch Evidence Scanner")
        if st.button("🧩 Quadrant Split Analysis", use_container_width=True, key="home_btn_quad"):
            navigate_to("Quadrant Split")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Session Activity Telemetry
    history = st.session_state.history
    total_analyzed = len(history)

    st.markdown("### 📈 Live Session Telemetry")
    if total_analyzed > 0:
        ai_count = sum(1 for h in history if h.get("label") == "AI-GENERATED")
        real_count = total_analyzed - ai_count
        avg_conf = sum(h.get("confidence", 0.0) for h in history) / total_analyzed
        avg_latency = sum(h.get("latency_ms", 0.0) for h in history) / total_analyzed

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Images Analyzed", total_analyzed)
        m2.metric("AI-Generated", f"{ai_count} ({ai_count/total_analyzed*100:.1f}%)")
        m3.metric("Authentic Real", f"{real_count} ({real_count/total_analyzed*100:.1f}%)")
        m4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        m5.metric("Avg Latency", f"{avg_latency:.1f} ms")
    else:
        st.info("ℹ️ No media has been inspected in the active session yet. Navigate to 'Detect Image' to analyze photographic evidence.")

    st.markdown("---")

    # System & Model Status
    tau = get_calibrated_threshold()
    dev = get_compute_device()

    st.markdown("### 🛡️ Detection System Specifications")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f"""
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; color: #64748b;">PRIMARY MODEL</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin-top: 0.2rem;">Spatio-Frequency Fusion</div>
                <div style="font-size: 0.72rem; color: #10b981; margin-top: 0.3rem;">● Operational</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; color: #64748b;">FEATURE DOMAINS</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-top: 0.2rem;">Spatial RGB + 2D FFT</div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.3rem;">Gated Feature Fusion</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; color: #64748b;">DECISION THRESHOLD</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #f59e0b; margin-top: 0.2rem;">τ = {tau:.2f} (Calibrated)</div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.3rem;">Optimized Precision Guard</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            f"""
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; color: #64748b;">COMPUTE HARDWARE</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #00f0ff; margin-top: 0.2rem;">{dev.upper()} ACCELERATION</div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.3rem;">PyTorch 2.x Architecture</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Quick Access Hub
    st.markdown("### 🗂️ Quick Laboratory Access")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.85rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8;">🔬 Image Detection</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Run single-image detection, batch evidence triage, or dual-evidence comparative inspection.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Image Detection →", key="hub_detect", use_container_width=True):
            navigate_to("Detect Image", "Single Image Detection")

    with col_b:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.85rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #00f0ff;">🧩 Quadrant Split</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Partition portrait, preserve 1 real quadrant, synthesize 3 photorealistic AI quadrants & predict.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Quadrant Split →", key="hub_quadrant", use_container_width=True):
            navigate_to("Quadrant Split")

    with col_c:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.85rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #a78bfa;">🌊 Deep Analysis Lab</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Inspect 2D Fourier power spectra, Grad-CAM spatial heatmaps, and test perturbation resilience.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Deep Analysis →", key="hub_analysis", use_container_width=True):
            navigate_to("Analysis", "Spatio-Frequency / FFT")

    with col_d:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.85rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #34d399;">📊 Evaluation & Reports</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Inspect verified benchmark matrices, confusion matrices, audit logs, and export PDF dossiers.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Performance & Reports →", key="hub_perf", use_container_width=True):
            navigate_to("Performance", "Model Performance Dashboard")

    st.markdown("---")

    # Project Workflow Pipeline Overview
    st.markdown("### 🔄 End-to-End Spatio-Frequency Pipeline")
    st.markdown(
        """
        <div style="background: rgba(8, 12, 22, 0.9); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 6px; padding: 1.25rem; font-family: 'Inter', sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div style="text-align: center; flex: 1; min-width: 140px; background: rgba(255,255,255,0.02); padding: 0.75rem; border-radius: 4px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.75rem; color: #00f0ff; font-weight: 700;">STEP 1</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;">Evidence Ingestion</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem;">224×224 RGB Normalization & Cryptographic Hash</div>
                </div>
                <div style="color: #64748b; font-size: 1.1rem;">➔</div>
                <div style="text-align: center; flex: 1.3; min-width: 160px; background: rgba(255,255,255,0.02); padding: 0.75rem; border-radius: 4px; border: 1px solid rgba(56, 189, 248, 0.3);">
                    <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">STEP 2</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;">Dual Representation</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem;">Spatial ResNet-50 + 2D FFT Magnitude Spectrum</div>
                </div>
                <div style="color: #64748b; font-size: 1.1rem;">➔</div>
                <div style="text-align: center; flex: 1.3; min-width: 160px; background: rgba(255,255,255,0.02); padding: 0.75rem; border-radius: 4px; border: 1px solid rgba(167, 139, 250, 0.3);">
                    <div style="font-size: 0.75rem; color: #a78bfa; font-weight: 700;">STEP 3</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;">Gated Attention Fusion</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem;">Dynamic α weighting + Unified Head (τ = 0.55)</div>
                </div>
                <div style="color: #64748b; font-size: 1.1rem;">➔</div>
                <div style="text-align: center; flex: 1; min-width: 140px; background: rgba(255,255,255,0.02); padding: 0.75rem; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <div style="font-size: 0.75rem; color: #10b981; font-weight: 700;">STEP 4</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;">Visual Saliency & Dossier</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem;">Grad-CAM Heatmap + ReportLab PDF Export</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if history:
        st.markdown("---")
        st.markdown("### 📋 Recent Detection Activity")
        df_recent = pd.DataFrame([
            {
                "Timestamp": h.get("timestamp"),
                "Filename": h.get("filename"),
                "Verdict": h.get("label"),
                "Confidence": f"{h.get('confidence', 0.0)*100:.1f}%",
                "Latency": f"{h.get('latency_ms', 0.0):.1f} ms"
            }
            for h in history[:5]
        ])
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

    render_disclaimer_notice()
