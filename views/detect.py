"""
Detect Image View
Primary AI Image Detection, Batch Evidence Triage, and Side-by-Side Comparison.
Focused exclusively on the Dual-Branch Spatio-Frequency Fusion Model.
"""

import os
import sys
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.pipeline import (
    analyze_single_image,
    get_benchmark_samples,
    get_calibrated_threshold,
    navigate_to
)
from ui.components import (
    render_page_header,
    render_prediction_result_card,
    render_evidence_plates,
    render_radar_scanner,
    render_disclaimer_notice
)
from ui.pdf_generator import generate_forensic_pdf
from ui.gauge import render_confidence_gauge

def render_detect_view():
    render_page_header(
        title="AI Image Detection Center",
        subtitle="Evaluate photographic media using the Dual-Branch Spatio-Frequency Fusion model to detect latent diffusion synthesis.",
        module_badge="DETECTION"
    )

    # Sub-tab Navigation
    subtabs = ["Single Image Detection", "Batch Evidence Scanner", "Image Comparison Lab"]
    current_sub = st.session_state.get("detect_subtab", "Single Image Detection")
    if current_sub not in subtabs:
        current_sub = "Single Image Detection"

    selected_sub = st.pills(
        "Detection Mode",
        options=subtabs,
        default=current_sub,
        required=True,
        label_visibility="collapsed"
    )
    if selected_sub and selected_sub != current_sub:
        st.session_state.detect_subtab = selected_sub
        st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if selected_sub == "Single Image Detection":
        render_single_detection()
    elif selected_sub == "Batch Evidence Scanner":
        render_batch_scanner()
    else:
        render_comparison_lab()

    render_disclaimer_notice()


def render_single_detection():
    active_res = st.session_state.active_single_result

    # If no active result, show upload and sample loader
    if active_res is None:
        u_col1, u_col2 = st.columns([2, 1])

        with u_col1:
            uploaded_file = st.file_uploader(
                "Upload Photographic Evidence (JPG, JPEG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="single_file_upl",
                help="Input files are validated, normalized to 224x224 RGB, and analyzed securely in memory."
            )

        with u_col2:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.35rem;'>OR LOAD VERIFIED BENCHMARK SAMPLE</div>", unsafe_allow_html=True)
            bench_samples = get_benchmark_samples()
            sample_choice = st.selectbox(
                "Benchmark Test Samples",
                ["None"] + list(bench_samples.keys()),
                key="sample_sel_detect"
            )

        target_file = None
        target_name = None

        if uploaded_file is not None:
            target_file = uploaded_file
            target_name = uploaded_file.name
        elif sample_choice != "None" and sample_choice in bench_samples:
            target_file = bench_samples[sample_choice]
            target_name = os.path.basename(bench_samples[sample_choice])

        if target_file is not None:
            # Preview before analysis
            try:
                preview_img = Image.open(target_file).convert("RGB")
                st.markdown("---")
                p_col1, p_col2 = st.columns([1, 1.4], gap="medium")
                with p_col1:
                    st.image(preview_img, caption=f"Selected: {target_name} ({preview_img.width}×{preview_img.height}px)", use_container_width=True)
                with p_col2:
                    st.markdown(
                        f"""
                        <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 1.25rem 1.4rem;">
                            <div style="font-size: 0.75rem; color: #00f0ff; font-weight: 700; text-transform: uppercase;">Ready for Analysis</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin: 0.3rem 0;">Evidence Ingestion Complete</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.6;">
                                File: <b>{target_name}</b><br/>
                                Format: <b>RGB {preview_img.width}×{preview_img.height}px</b><br/>
                                Engine: <b>Dual-Branch Spatio-Frequency Fusion</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    if st.button("⚡ Analyze Image", type="primary", use_container_width=True):
                        spinner_slot = st.empty()
                        spinner_slot.markdown(render_radar_scanner(f"ANALYZING: {target_name}..."), unsafe_allow_html=True)
                        res = analyze_single_image(target_file, target_name, alpha=0.5, record_history=True)
                        spinner_slot.empty()
                        if res is not None:
                            st.session_state.active_single_result = res
                            st.rerun()
            except Exception as e:
                st.error(f"Error validating image file: {e}")
        else:
            st.info("💡 Drag and drop an image above or select a verified benchmark sample to begin detection.")

    else:
        # Prediction Result Page: Clean, Streamlined, No Redundant Output
        def go_to_fft():
            navigate_to("Analysis", "Spatio-Frequency / FFT")

        def go_to_gradcam():
            navigate_to("Analysis", "Grad-CAM Explainability")

        def go_to_report():
            navigate_to("History & Reports", "Formal PDF Reports")

        def reset_image():
            st.session_state.active_single_result = None
            st.rerun()

        render_prediction_result_card(
            image=active_res["image"],
            label=active_res["label"],
            confidence=active_res["confidence"],
            on_fft_click=go_to_fft,
            on_gradcam_click=go_to_gradcam,
            on_report_click=go_to_report,
            on_reset_click=reset_image
        )

        st.markdown("---")
        # Visual Plates: Original, FFT, Grad-CAM
        st.markdown("### 📷 Visual Saliency & Frequency Plates")
        render_evidence_plates(
            image=active_res["image"],
            fft_spectrum=active_res["fft"],
            overlay=active_res["overlay"],
            heatmap=active_res["heatmap"]
        )


def render_batch_scanner():
    st.markdown("### ⚡ Batch Evidence Scanner")
    st.caption("Upload multiple evidence images simultaneously for high-throughput batch triage.")

    batch_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upl"
    )

    if batch_files:
        st.write(f"📁 **{len(batch_files)}** files selected for batch triage.")
        if st.button("⚡ Execute Batch Triage", type="primary", key="btn_exec_batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_results = []
            
            t_start = time.time()
            for idx, file in enumerate(batch_files):
                status_text.markdown(f"**Scanning ({idx+1}/{len(batch_files)}):** `{file.name}`...")
                try:
                    res = analyze_single_image(file, file.name, alpha=0.5, record_history=True)
                    if res is not None:
                        batch_results.append(res)
                except Exception as e:
                    st.warning(f"Error scanning {file.name}: {e}")
                progress_bar.progress((idx + 1) / len(batch_files))

            elapsed = time.time() - t_start
            status_text.success(f"✅ Triage complete! Processed {len(batch_results)} images in {elapsed:.2f}s ({elapsed/max(1, len(batch_results))*1000:.1f}ms/image).")
            st.session_state.batch_results = batch_results

    if st.session_state.get("batch_results"):
        results = st.session_state.batch_results
        total_count = len(results)
        ai_count = sum(1 for r in results if r["label"] == "AI-GENERATED")
        real_count = total_count - ai_count
        avg_conf = np.mean([r["confidence"] for r in results]) if results else 0.0

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Triaged", total_count)
        m2.metric("Synthetic (AI)", f"{ai_count} ({ai_count/total_count*100:.1f}%)")
        m3.metric("Authentic (Real)", f"{real_count} ({real_count/total_count*100:.1f}%)")
        m4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

        chart_data = pd.DataFrame({
            "Category": ["AI-Generated", "Authentic Real"],
            "Count": [ai_count, real_count]
        }).set_index("Category")
        st.bar_chart(chart_data, color=["#38bdf8"])

        st.markdown("---")
        st.markdown("### 📋 Triage Results")

        for item in results:
            is_ai = (item["label"] == "AI-GENERATED")
            icon = "🤖" if is_ai else "🌿"
            color = "#ff2d55" if is_ai else "#00f0ff"
            with st.expander(f"{icon} {item['filename']} — {item['label']} ({item['confidence']*100:.1f}%)"):
                c1, c2, c3 = st.columns([1, 1, 1.2])
                with c1:
                    st.image(item["image"], caption="Evidence", use_container_width=True)
                with c2:
                    fig, ax = plt.subplots(figsize=(3, 3), facecolor="#080c14")
                    ax.imshow(item["fft"], cmap="magma")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("2D FFT Spectrum")
                with c3:
                    st.image(item["overlay"], caption="Grad-CAM Overlay", use_container_width=True)
                    st.write(f"Result: <b style='color:{color};'>{item['label']}</b> ({item['confidence']*100:.1f}%)", unsafe_allow_html=True)
                    if st.button("Inspect in Single Detection", key=f"insp_{item['id']}"):
                        st.session_state.active_single_result = item
                        navigate_to("Detect Image", "Single Image Detection")

        csv_rows = [
            {
                "Filename": r["filename"],
                "Verdict": r["label"],
                "Confidence": f"{r['confidence']:.4f}",
                "Model": r["model_name"],
                "Latency_ms": f"{r['latency_ms']:.1f}",
                "Timestamp": r["timestamp"]
            }
            for r in results
        ]
        df_export = pd.DataFrame(csv_rows)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Batch Summary as CSV",
            data=csv_bytes,
            file_name=f"batch_detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_comparison_lab():
    st.markdown("### ⚖️ Image Comparison Lab")
    st.caption("Compare two suspect images side by side through the Spatio-Frequency Fusion model.")

    bench_samples = get_benchmark_samples()
    bench_keys = ["(Upload Custom File)"] + list(bench_samples.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Sample A")
        ch_a = st.selectbox("Source for A", bench_keys, key="cmp_a")
        file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png", "webp"], key="upl_cmp_a") if ch_a == "(Upload Custom File)" else bench_samples[ch_a]

    with col_b:
        st.markdown("#### Sample B")
        ch_b = st.selectbox("Source for B", bench_keys, index=min(2, len(bench_keys)-1), key="cmp_b")
        file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png", "webp"], key="upl_cmp_b") if ch_b == "(Upload Custom File)" else bench_samples[ch_b]

    if st.button("⚖️ Run Side-by-Side Comparison", type="primary", use_container_width=True):
        if file_a is not None and file_b is not None:
            name_a = file_a.name if hasattr(file_a, "name") else os.path.basename(file_a)
            name_b = file_b.name if hasattr(file_b, "name") else os.path.basename(file_b)

            with st.spinner("Analyzing Sample A and Sample B..."):
                ra = analyze_single_image(file_a, name_a, record_history=True)
                rb = analyze_single_image(file_b, name_b, record_history=True)
            st.session_state.comparison_results = {"ra": ra, "rb": rb}
        else:
            st.error("Please supply both Image A and Image B.")

    if st.session_state.get("comparison_results"):
        comp = st.session_state.comparison_results
        ra = comp.get("ra")
        rb = comp.get("rb")
        if ra and rb:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### Image A: `{ra['filename']}`")
                g_a = render_confidence_gauge(ra["label"], ra["confidence"], threshold=ra["threshold"])
                st.markdown(g_a, unsafe_allow_html=True)
                st.markdown(f"**Result:** `{'🤖 AI-GENERATED' if ra['label'] == 'AI-GENERATED' else '🌿 REAL'}` ({ra['confidence']*100:.1f}%)")
                st.image(ra["image"], caption="Image A Input", use_container_width=True)
                st.image(ra["overlay"], caption="Image A Grad-CAM", use_container_width=True)

            with c2:
                st.markdown(f"#### Image B: `{rb['filename']}`")
                g_b = render_confidence_gauge(rb["label"], rb["confidence"], threshold=rb["threshold"])
                st.markdown(g_b, unsafe_allow_html=True)
                st.markdown(f"**Result:** `{'🤖 AI-GENERATED' if rb['label'] == 'AI-GENERATED' else '🌿 REAL'}` ({rb['confidence']*100:.1f}%)")
                st.image(rb["image"], caption="Image B Input", use_container_width=True)
                st.image(rb["overlay"], caption="Image B Grad-CAM", use_container_width=True)
