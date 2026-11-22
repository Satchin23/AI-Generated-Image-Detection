"""
About View
Academic Documentation, Architecture Topology, and Viva Voce Guide.
"""

import os
import sys
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.components import render_page_header, render_disclaimer_notice

def render_about_view():
    render_page_header(
        title="Project Documentation & Academic Dossier",
        subtitle="Technical documentation covering the spatio-frequency dual-branch network, anti-leakage dataset protocols, and reproducible execution instructions.",
        module_badge="ABOUT & SPECIFICATIONS"
    )

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.8)); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.72rem; color: #00f0ff; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;">Academic Viva Dossier</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; margin: 0.3rem 0 0.5rem 0;">
                Grad-CAM Enabled Detection of AI-Generated Images Using Spatio-Frequency Features
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                A deep learning forensic system designed to differentiate authentic camera captures from diffusion-synthesized imagery (Stable Diffusion v1.4, v1.5, ADM, Midjourney) by uniting spatial semantic representations with 2D Fourier power spectra under a learned gated attention fusion mechanism.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("1. Project Motivation & Problem Statement", expanded=True):
        st.markdown(
            """
            - **The Challenge:** Modern text-to-image latent diffusion models (LDMs) generate photorealistic imagery that bypasses human visual inspection and classical spatial-only CNNs.
            - **The Scientific Hypothesis:** While generative decoders can fool spatial pixel inspections, their convolutional deconvolution, transpose layers, and VAE decoders leave structural, periodic artifacts in the frequency domain (2D FFT magnitude spectrum).
            - **The Solution:** A dual-branch spatio-frequency network uniting a ResNet-50 spatial backbone with a 4-stage frequency CNN via gated attention fusion, paired with an operational Grad-CAM explainability pipeline.
            """
        )

    with st.expander("2. System Architecture Topology & Data Flow", expanded=True):
        st.markdown(
            """
            <div style="background: rgba(8, 12, 22, 0.95); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 1.5rem; text-align: center; font-family: 'Inter', sans-serif;">
                <div style="font-size: 0.75rem; color: #00f0ff; font-weight: 700; letter-spacing: 0.1em; margin-bottom: 1rem;">SPATIO-FREQUENCY DUAL-BRANCH PIPELINE TOPOLOGY</div>
                
                <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1.25rem;">
                    <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid #38bdf8; padding: 0.75rem 1.25rem; border-radius: 6px;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">Input Image Evidence</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">224 × 224 RGB</div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0;">
                    <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 6px; padding: 1rem; text-align: left;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8;">SPATIAL FEATURE EXTRACTION</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin: 0.4rem 0;">• Normalized RGB Tensor [3, 224, 224]</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin: 0.4rem 0;">• ResNet-50 Backbone (layer1 → layer4)</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin: 0.4rem 0;">• Grad-CAM Hooks on layer4[-1] Bottleneck</div>
                        <div style="font-size: 0.74rem; color: #38bdf8; font-weight: 600; margin-top: 0.5rem;">➔ Spatial Feature Embedding z_s (256-dim)</div>
                    </div>

                    <div style="background: rgba(167, 139, 250, 0.08); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; padding: 1rem; text-align: left;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #a78bfa;">FREQUENCY FEATURE EXTRACTION</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin: 0.4rem 0;">• 2D FFT Transform + Quadrant Shift</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin: 0.4rem 0;">• Log-Magnitude Normalization log(1 + |F|)</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin: 0.4rem 0;">• 4-Stage Convolutional Feature Extractor</div>
                        <div style="font-size: 0.74rem; color: #a78bfa; font-weight: 600; margin-top: 0.5rem;">➔ Frequency Feature Embedding z_f (256-dim)</div>
                    </div>
                </div>

                <div style="background: rgba(236, 72, 153, 0.08); border: 1px solid rgba(236, 72, 153, 0.4); border-radius: 6px; padding: 0.85rem; margin: 1rem 0;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #ec4899;">GATED ATTENTION FUSION MECHANISM</div>
                    <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 0.3rem;">
                        Gate α = Sigmoid( Linear( [z_s || z_f] ) ) &nbsp; | &nbsp; z_fused = α · z_s + (1 - α) · z_f
                    </div>
                </div>

                <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-top: 1rem;">
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.5); padding: 0.6rem 1.5rem; border-radius: 6px;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #10b981;">Binary Classification (τ = 0.55)</div>
                        <div style="font-size: 0.72rem; color: #ffffff;">REAL vs AI-GENERATED</div>
                    </div>
                    <div style="background: rgba(255, 45, 85, 0.15); border: 1px solid rgba(255, 45, 85, 0.5); padding: 0.6rem 1.5rem; border-radius: 6px;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #ff2d55;">Grad-CAM Saliency Map</div>
                        <div style="font-size: 0.72rem; color: #ffffff;">Layer4 Backpropagation Heatmap</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("3. Dataset Partitioning & Anti-Leakage Protocol", expanded=False):
        st.markdown(
            """
            - **Total Dataset:** 10,936 verified images.
            - **Authentic Photographic Class:** 2,500 real photographs drawn from ImageNet-1k (ILSVRC).
            - **Synthetic AI-Generated Class:** 8,436 images across ADM, Stable Diffusion v1.4, Stable Diffusion v1.5, and Midjourney.
            - **Strict Anti-Leakage Protocol:**
              - **Held-Out OOD Isolation:** Midjourney (2,250 images) is **strictly isolated** as an unseen out-of-distribution benchmark. It is **never used** during training, validation, or threshold calibration.
              - **Zero Cross-Split Contamination:** Train (80%), Val (10%), Test (10%) splits are partitioned by deterministic hash seeds to prevent identical scenes across splits.
            """
        )

    with st.expander("4. Training Methodology & Threshold Calibration", expanded=False):
        st.markdown(
            """
            - **Dual Synchronous DataLoader:** Uses synchronized batches feeding identical image augmentations to both the spatial ResNet-50 branch and the 2D FFT transform.
            - **Train-Only Augmentations:** Random JPEG compression (Q=50-95, p=0.3), Gaussian blur (σ≤1.5, p=0.3), downscale/upscale (0.5x-1.0x, p=0.3). Validation and test sets remain strictly unperturbed.
            - **Threshold Calibration:** The decision threshold τ is calibrated on the clean validation set to maximize F1-score with an F1 < 0.30 collapse guard. The verified threshold is **τ = 0.55**.
            """
        )

    with st.expander("5. Technology Stack & Execution Commands", expanded=True):
        st.markdown(
            """
            ```bash
            # 1. Run unit test suite
            pytest tests/test_inference.py -v

            # 2. Launch Streamlit web application
            streamlit run app.py

            # 3. Compile full documentation PDF
            python build_pdf_report.py
            ```
            """
        )

    render_disclaimer_notice()
