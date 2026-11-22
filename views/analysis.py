"""
Deep Analysis View
Spatio-Frequency (2D FFT) Spectra, Grad-CAM Saliency, and Image Quality Lab.
"""

import os
import sys
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.pipeline import (
    analyze_single_image,
    get_benchmark_samples,
    load_spatial_model,
    navigate_to
)
from ui.components import render_page_header, render_disclaimer_notice
from evaluation.robustness import (
    apply_jpeg_compression,
    apply_gaussian_blur,
    apply_downsampling,
    apply_contrast
)
from config.config import IMAGE_SIZE

def render_analysis_view():
    render_page_header(
        title="Visual & Spectral Analysis Lab",
        subtitle="Examine 2D Fourier power spectra, Grad-CAM spatial activations, and test sensitivity to transmission perturbations.",
        module_badge="ANALYSIS"
    )

    subtabs = ["Spatio-Frequency / FFT", "Grad-CAM Explainability", "Image Quality Lab"]
    current_sub = st.session_state.get("analysis_subtab", "Spatio-Frequency / FFT")
    if current_sub not in subtabs:
        current_sub = "Spatio-Frequency / FFT"

    selected_sub = st.pills(
        "Analysis Module",
        options=subtabs,
        default=current_sub,
        required=True,
        label_visibility="collapsed"
    )
    if selected_sub and selected_sub != current_sub:
        st.session_state.analysis_subtab = selected_sub
        st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Ensure an active image is available
    active_res = st.session_state.active_single_result
    if active_res is None:
        st.info("ℹ️ No active image loaded. Select a benchmark sample or upload an image below to initiate deep analysis:")
        bench_samples = get_benchmark_samples()
        c1, c2 = st.columns([1, 1])
        with c1:
            s_choice = st.selectbox("Benchmark Sample", list(bench_samples.keys()), key="an_sample_sel")
            if st.button("Load Benchmark Sample", key="btn_load_bench_an"):
                active_res = analyze_single_image(bench_samples[s_choice], os.path.basename(bench_samples[s_choice]), record_history=True)
                st.session_state.active_single_result = active_res
                st.rerun()
        with c2:
            u_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], key="an_upl")
            if u_file is not None and st.button("Analyze Uploaded Image", key="btn_load_upl_an"):
                active_res = analyze_single_image(u_file, u_file.name, record_history=True)
                st.session_state.active_single_result = active_res
                st.rerun()
        return

    # Render selected sub-tool
    if selected_sub == "Spatio-Frequency / FFT":
        render_fft_tool(active_res)
    elif selected_sub == "Grad-CAM Explainability":
        render_gradcam_tool(active_res)
    else:
        render_quality_tool(active_res)

    render_disclaimer_notice()


def render_fft_tool(active_res):
    pil_image = active_res["image"]
    fft_spec = active_res["fft"]

    c_back, c_cmap, _ = st.columns([1, 1.2, 1.8])
    with c_back:
        if st.button("⬅️ Back to Detection", use_container_width=True):
            navigate_to("Detect Image", "Single Image Detection")
    with c_cmap:
        cmap_choice = st.selectbox("Colormap", ["magma", "viridis", "inferno", "plasma", "gray"], index=0)

    st.markdown("---")
    col_orig, col_fft = st.columns(2)

    with col_orig:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 6px; padding: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; font-size: 0.85rem; color: #38bdf8;">Spatial Domain (RGB Pixel Space)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(pil_image, caption=f"Active File: {active_res['filename']}", use_container_width=True)
        st.markdown(
            f"""
            <div style="font-size: 0.76rem; color: #94a3b8; background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 4px; margin-top: 0.5rem; line-height: 1.6;">
                <div>• Resolution: <b>{pil_image.width} × {pil_image.height} px</b></div>
                <div>• Format: <b>{pil_image.mode}</b> (3 channels)</div>
                <div>• Model Prediction: <b style="color:{'#ff2d55' if active_res['label'] == 'AI-GENERATED' else '#10b981'};">{active_res['label']}</b> ({active_res['confidence']*100:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_fft:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(167, 139, 250, 0.3); border-radius: 6px; padding: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; font-size: 0.85rem; color: #a78bfa;">Frequency Domain (2D FFT Log-Magnitude)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#080c14")
        im = ax.imshow(fft_spec, cmap=cmap_choice)
        ax.axis("off")
        cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7, colors="#94a3b8")
        st.pyplot(fig)
        plt.close(fig)

        min_v = float(np.min(fft_spec))
        max_v = float(np.max(fft_spec))
        mean_v = float(np.mean(fft_spec))
        st.markdown(
            f"""
            <div style="font-size: 0.76rem; color: #94a3b8; background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 4px; margin-top: 0.5rem; line-height: 1.6;">
                <div>• Transform: <b>2D FFT with Quadrant Center Shift</b></div>
                <div>• Dynamic Range: <b>[{min_v:.2f}, {max_v:.2f}]</b></div>
                <div>• Spectral Energy Mean: <b>{mean_v:.3f}</b></div>
                <div>• Dimensions: <b>{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} Frequency Bins</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🔬 Spectral Feature Significance")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.9rem; line-height: 1.6; font-size: 0.82rem;">
                <div style="font-weight: 700; color: #00f0ff; margin-bottom: 0.4rem;">Low vs High Frequency Distribution</div>
                <div>• <b>Center (Low Frequency):</b> Ambient lighting, smooth skin tones, macroscopic structures. Both real cameras and generative models exhibit smooth central power.</div>
                <div style="margin-top: 0.4rem;">• <b>Periphery (High Frequency):</b> Fine textures and edges. AI models often generate unnatural radial spikes or periodic phase rolls.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with e2:
        st.markdown(
            """
            <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.9rem; line-height: 1.6; font-size: 0.82rem;">
                <div style="font-weight: 700; color: #ec4899; margin-bottom: 0.4rem;">Why Frequency Features Detect AI Synthesis</div>
                <div>• <b>Upsampling Harmonics:</b> Transpose convolution and latent VAE decoders apply uniform spatial interpolation, inducing characteristic checkerboard harmonics in Fourier space.</div>
                <div style="margin-top: 0.4rem;">• <b>Sensor Noise Inconsistency:</b> Real camera sensors leave Photo-Response Non-Uniformity (PRNU) noise patterns absent in synthetic decoders.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_gradcam_tool(active_res):
    pil_image = active_res["image"]
    heatmap = active_res["heatmap"]

    c_back, c_alpha, c_rep = st.columns([1, 1.5, 1])
    with c_back:
        if st.button("⬅️ Back to Detection", use_container_width=True, key="cam_back_btn"):
            navigate_to("Detect Image", "Single Image Detection")
    with c_alpha:
        alpha_val = st.slider("Overlay Opacity (α)", min_value=0.1, max_value=0.9, value=st.session_state.get("alpha_val", 0.5), step=0.05)
        st.session_state.alpha_val = alpha_val
    with c_rep:
        if st.button("📄 Generate Report →", use_container_width=True, key="cam_rep_btn"):
            navigate_to("History & Reports", "Formal PDF Reports")

    spatial_pred = load_spatial_model()
    overlay, _ = spatial_pred.gradcam.overlay(pil_image, heatmap, alpha=alpha_val)
    active_res["overlay"] = overlay

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Original Input**")
        st.image(pil_image, use_container_width=True)
    with c2:
        st.markdown("**2. Raw Grad-CAM Heatmap**")
        st.image(heatmap, use_container_width=True)
        st.caption("Normalized saliency [0, 1] from layer4[-1]")
    with c3:
        st.markdown("**3. Spatial Heatmap Overlay**")
        st.image(overlay, use_container_width=True)
        st.caption(f"Blended with α = {alpha_val:.2f}")

    st.markdown("---")
    st.markdown(
        """
        <div style="background: rgba(13, 19, 32, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 6px; padding: 0.9rem; line-height: 1.6; font-size: 0.82rem;">
            <b style="color: #38bdf8;">Visual Explanation Guide:</b> Grad-CAM computes gradients with respect to convolutional feature maps in ResNet-50 layer4 to visualize regions that heavily influenced the model's prediction. Warmer colors (red/yellow) indicate focal regions that triggered the decision.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_quality_tool(active_res):
    orig_img = active_res["image"]
    
    st.markdown("### 🎛️ Transmission Perturbation Testing")
    st.caption("Evaluate how digital transmission distortions (lossy compression, blur, downsampling) affect model confidence.")

    p1, p2 = st.columns(2)
    with p1:
        dist_type = st.selectbox(
            "Perturbation Type",
            ["JPEG Compression", "Gaussian Blur", "Downsampling & Rescaling", "Contrast Enhancement"]
        )
    with p2:
        if dist_type == "JPEG Compression":
            q = st.slider("JPEG Quality (Q)", 10, 95, 60, 5)
            trans_img = apply_jpeg_compression(orig_img, quality=q)
            desc = f"JPEG (Q={q})"
        elif dist_type == "Gaussian Blur":
            r = st.slider("Blur Radius", 0.5, 4.0, 1.5, 0.25)
            trans_img = apply_gaussian_blur(orig_img, radius=r)
            desc = f"Gaussian Blur (r={r})"
        elif dist_type == "Downsampling & Rescaling":
            s = st.slider("Downscale Factor", 0.2, 0.8, 0.5, 0.05)
            trans_img = apply_downsampling(orig_img, scale=s)
            desc = f"Downsampling ({s:.2f}×)"
        else:
            cf = st.slider("Contrast Factor", 0.5, 2.5, 1.4, 0.1)
            trans_img = apply_contrast(orig_img, factor=cf)
            desc = f"Contrast ({cf:.1f}×)"

    st.markdown("---")
    v1, v2 = st.columns(2)
    with v1:
        st.image(orig_img, caption="Original Evidence", use_container_width=True)
    with v2:
        st.image(trans_img, caption=f"Perturbed Evidence: {desc}", use_container_width=True)

    if st.button("🧪 Evaluate Model on Perturbed Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing perturbed image..."):
            pert_res = analyze_single_image(trans_img, f"{active_res['filename']}_{desc}", record_history=True)

        if pert_res:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original Result", active_res["label"])
            m2.metric("Perturbed Result", pert_res["label"])
            delta_conf = (pert_res["confidence"] - active_res["confidence"]) * 100
            m3.metric("Original Conf", f"{active_res['confidence']*100:.1f}%")
            m4.metric("Perturbed Conf", f"{pert_res['confidence']*100:.1f}%", f"{delta_conf:+.1f}%")
