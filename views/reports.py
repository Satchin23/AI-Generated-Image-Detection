"""
Reports & History View with Comprehensive SQLite Database Explorer.
Provides persistent audit trail tracking, multi-table database inspector,
human-in-the-loop ground truth verification, and formal forensic PDF dossier generation.
"""

import os
import sys
import io
import hashlib
from datetime import datetime
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.pipeline import (
    analyze_single_image,
    get_benchmark_samples,
    navigate_to
)
from ui.components import render_page_header, render_disclaimer_notice
from ui.pdf_generator import generate_forensic_pdf

from database.db_manager import (
    get_db_connection,
    init_db,
    get_all_detections,
    get_detection_by_id,
    delete_detection,
    clear_all_detections,
    get_batch_sessions,
    get_models_registry,
    get_benchmark_generators,
    get_audit_logs,
    insert_audit_log,
    get_verification_feedback,
    insert_verification_feedback,
    get_system_configs,
    get_database_summary_stats,
    execute_custom_query,
    DB_PATH
)


def render_reports_view():
    render_page_header(
        title="Forensic Audit Trail & SQLite Database Explorer",
        subtitle="Review continuous database audit logs, inspect relational tables, run queries, and export publication-grade PDF dossiers.",
        module_badge="SQLITE DB & AUDIT"
    )

    subtabs = ["Detection History", "Database Explorer & Analytics", "Formal PDF Reports"]
    current_sub = st.session_state.get("reports_subtab", "Detection History")
    if current_sub not in subtabs:
        current_sub = "Detection History"

    selected_sub = st.pills(
        "Report Module",
        options=subtabs,
        default=current_sub,
        required=True,
        label_visibility="collapsed"
    )
    if selected_sub and selected_sub != current_sub:
        st.session_state.reports_subtab = selected_sub
        st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if selected_sub == "Detection History":
        render_history_tab()
    elif selected_sub == "Database Explorer & Analytics":
        render_database_explorer_tab()
    else:
        render_pdf_tab()

    render_disclaimer_notice()


def render_history_tab():
    """Renders detection history loaded directly from SQLite database."""
    init_db()

    f_col1, f_col2, f_col3 = st.columns([1.8, 1.5, 1.2])
    with f_col1:
        search = st.text_input("🔍 Search Filename / Record ID / Hash", "")
    with f_col2:
        verdict = st.radio("Verdict Filter", ["ALL", "AI-GENERATED", "REAL"], horizontal=True)
    with f_col3:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear DB Detections", use_container_width=True, type="secondary"):
            cleared = clear_all_detections()
            st.session_state.history = []
            st.session_state.active_single_result = None
            st.session_state.batch_results = []
            st.success(f"Cleared {cleared} detection records from SQLite database.")
            st.rerun()

    db_records = get_all_detections(limit=100, verdict_filter=verdict, search_query=search)

    if not db_records:
        st.info("📜 No examination records matching filter criteria in SQLite database. Run an analysis in 'Detect Image' to record live examinations.")
        if st.button("🚀 Start Detection", type="primary"):
            navigate_to("Detect Image", "Single Image Detection")
        return

    st.write(f"Displaying **{len(db_records)}** persistent examination records from SQLite (`detections` table):")

    for idx, item in enumerate(db_records):
        is_ai = (item.get("verdict") == "AI-GENERATED")
        b_col = "#ff2d55" if is_ai else "#10b981"
        b_txt = "🤖 AI-GENERATED" if is_ai else "🌿 REAL"
        rec_id = item.get("record_id", f"REC-{idx}")

        with st.container():
            st.markdown(
                f"""
                <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 3px solid {b_col}; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; flex-wrap: wrap;">
                        <span style="font-size: 0.88rem; font-weight: 700; color: #ffffff;">#{len(db_records) - idx} // {item.get('filename')}</span>
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <span style="font-family: monospace; font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 0.15rem 0.45rem; border-radius: 3px;">{rec_id}</span>
                            <span style="font-size: 0.72rem; color: #64748b;">{item.get('exam_timestamp')}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-family: monospace; margin-bottom: 0.5rem; word-break: break-all;">
                        SHA-256: {item.get('file_hash_sha256')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            c_info, c_metrics, c_act = st.columns([2.2, 1.8, 1.2])
            with c_info:
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; line-height: 1.6;">
                        <div>Classification: <b style="color: {b_col};">{b_txt}</b></div>
                        <div>Confidence: <b>{item.get('confidence', 0.0)*100:.2f}%</b> (τ={item.get('calibrated_threshold', 0.55):.2f})</div>
                        <div>Category / Generator: <b>{item.get('generator_family', 'Unknown')}</b> ({item.get('dataset_category', 'General')})</div>
                        <div>Examiner Notes: <span style="color:#94a3b8; font-style:italic;">{item.get('examiner_notes', 'N/A')}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_metrics:
                st.markdown(
                    f"""
                    <div style="font-size: 0.8rem; line-height: 1.6; background: rgba(0,0,0,0.25); padding: 0.5rem; border-radius: 4px;">
                        <div>Spatial Score: <code>{item.get('spatial_score', 0.0):.4f}</code></div>
                        <div>Frequency Score: <code>{item.get('frequency_score', 0.0):.4f}</code></div>
                        <div>Fusion Score: <code>{item.get('fusion_score', 0.0):.4f}</code></div>
                        <div>Latency: <b>{item.get('latency_ms', 0.0):.1f} ms</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_act:
                # Check for in-memory item with images
                mem_match = next((h for h in st.session_state.history if h.get("id") == rec_id or h.get("filename") == item.get("filename")), None)
                if mem_match:
                    if st.button("🔍 Open In Viewer", key=f"hist_open_{rec_id}_{idx}", use_container_width=True):
                        st.session_state.active_single_result = mem_match
                        navigate_to("Detect Image", "Single Image Detection")

                if st.button("🗑️ Delete", key=f"hist_del_{rec_id}_{idx}", use_container_width=True):
                    delete_detection(rec_id)
                    st.toast(f"Deleted record {rec_id}")
                    st.rerun()

    # CSV Export of SQLite Detections Table
    st.markdown("---")
    df_export = pd.DataFrame(db_records)
    csv_data = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export SQLite Detections Table (CSV)",
        data=csv_data,
        file_name=f"sqlite_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def render_database_explorer_tab():
    """Comprehensive interactive explorer for all 7 SQLite relational tables."""
    init_db()
    stats = get_database_summary_stats()

    # Database Header KPI Banner
    st.markdown("### 📊 SQLite Database Overview & Architecture")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Total Tables", stats["total_tables"])
    with k2:
        st.metric("Total Rows", stats["total_rows"])
    with k3:
        st.metric("Scanned Records", stats["detections_count"])
    with k4:
        st.metric("AI Detection Rate", f"{stats['ai_ratio_pct']}%")
    with k5:
        st.metric("Mean Latency", f"{stats['avg_latency_ms']} ms")
    with k6:
        st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")

    st.caption(f"📁 Database File: `{stats['db_file_path']}`")

    st.markdown("---")

    # Table Selection & Viewer
    table_options = [
        "detections (26 cols - Core Scan Records)",
        "batch_sessions (12 cols - Evaluation Batches)",
        "models_registry (14 cols - Model Benchmarks)",
        "benchmark_generators (9 cols - AI Engine Signatures)",
        "forensic_audit_logs (7 cols - Tamper & Security Logs)",
        "verification_feedback (7 cols - Human Reviews)",
        "system_configs (6 cols - Calibrations & Params)"
    ]

    selected_option = st.selectbox(
        "🗄️ Select Relational Table to Inspect",
        table_options,
        index=0
    )

    table_name = selected_option.split(" ")[0]

    # Fetch table data
    conn = get_db_connection()
    try:
        df_table = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    finally:
        conn.close()

    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.markdown(f"#### Table: `{table_name}` ({len(df_table)} rows, {len(df_table.columns)} columns)")
    with t_col2:
        csv_t = df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {table_name}.csv",
            data=csv_t,
            file_name=f"{table_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Render interactive DataFrame
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        height=min(450, 45 + len(df_table) * 36)
    )

    st.markdown("---")

    # Expandable Sections: Add Verification & SQL Sandbox
    col_v, col_q = st.columns(2)

    with col_v:
        st.markdown("### ✍️ Record Human Ground-Truth Feedback")
        st.caption("Add expert examiner validation to a detection record in SQLite.")
        with st.form("feedback_form"):
            detections_list = get_all_detections(limit=30)
            if detections_list:
                rec_choices = [f"{d['record_id']} - {d['filename']} ({d['verdict']})" for d in detections_list]
                chosen_rec_str = st.selectbox("Target Detection Record", rec_choices)
                target_rec_id = chosen_rec_str.split(" - ")[0]
            else:
                target_rec_id = st.text_input("Target Record ID", "DET-00109")

            human_verdict = st.selectbox(
                "Human Examiner Verdict",
                ["CONFIRMED_AI", "CONFIRMED_REAL", "FALSE_POSITIVE", "FALSE_NEGATIVE", "DISPUTED"]
            )
            reviewer = st.text_input("Examiner Name / Role", "Forensic Evaluator")
            rating = st.slider("Verification Confidence Rating", 1, 5, 5)
            notes = st.text_area("Examiner Justification & Notes", "Artifacts visually verified under high magnification.")

            submitted = st.form_submit_button("💾 Save Feedback to SQLite", type="primary", use_container_width=True)
            if submitted:
                fb_id = insert_verification_feedback(
                    record_id=target_rec_id,
                    human_verdict=human_verdict,
                    reviewer_name=reviewer,
                    review_notes=notes,
                    confidence_rating=rating
                )
                insert_audit_log(
                    event_type="DETECTION_RUN",
                    severity="INFO",
                    details=f"Human feedback recorded for {target_rec_id}: {human_verdict}",
                    target_record_id=target_rec_id
                )
                st.success(f"Feedback successfully saved to `verification_feedback` table (Row ID #{fb_id})!")
                st.rerun()

    with col_q:
        st.markdown("### ⚡ SQL Query Sandbox (Read-Only)")
        st.caption("Run custom SQL queries against the 7 relational tables.")

        preset_queries = {
            "Top 5 High-Confidence AI Scans": "SELECT record_id, filename, confidence, generator_family, latency_ms FROM detections WHERE verdict='AI-GENERATED' ORDER BY confidence DESC LIMIT 5;",
            "Model Registry Benchmark Rankings": "SELECT model_id, model_name, accuracy, f1_score, auc_roc, is_active FROM models_registry ORDER BY accuracy DESC;",
            "Generators by Spectral Energy Ratio": "SELECT generator_name, generator_family, spectral_energy_ratio, detection_rate FROM benchmark_generators ORDER BY spectral_energy_ratio DESC;",
            "Recent Security Audit Trail Logs": "SELECT id, event_type, severity, details, target_record_id, created_at FROM forensic_audit_logs ORDER BY id DESC LIMIT 5;"
        }

        chosen_preset = st.selectbox("Preset Query Templates", list(preset_queries.keys()))
        custom_sql = st.text_area("SQL Statement", value=preset_queries[chosen_preset], height=85)

        if st.button("🚀 Execute SQL Query", type="primary", use_container_width=True):
            try:
                cols, rows = execute_custom_query(custom_sql)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Query returned 0 rows.")
            except Exception as e:
                st.error(f"SQL Error: {e}")


def render_pdf_tab():
    active_res = st.session_state.active_single_result
    history = st.session_state.history

    selected_target = None

    if active_res is not None:
        st.markdown(f"**Target Evidence:** `{active_res['filename']}` ({active_res['label']} - {active_res['confidence']*100:.1f}%)")
        selected_target = active_res
    elif history:
        names = [f"#{i+1}: {h['filename']} ({h['label']} - {h['confidence']*100:.1f}%)" for i, h in enumerate(history)]
        sel_i = st.selectbox("Select Evidence from History", range(len(names)), format_func=lambda i: names[i])
        selected_target = history[sel_i]
    else:
        st.info("ℹ️ No active evidence or session history to generate a report.")
        bench = get_benchmark_samples()
        c_b, _ = st.columns([1, 1])
        with c_b:
            b_choice = st.selectbox("Load Benchmark Sample", list(bench.keys()))
            if st.button("Generate Report for Benchmark Sample", type="primary"):
                selected_target = analyze_single_image(bench[b_choice], os.path.basename(bench[b_choice]), record_history=True)
                st.session_state.active_single_result = selected_target
                st.rerun()

    if selected_target:
        st.markdown("---")
        img_bytes = io.BytesIO()
        selected_target["image"].save(img_bytes, format="PNG")
        sha = hashlib.sha256(img_bytes.getvalue()).hexdigest()

        is_ai = (selected_target["label"] == "AI-GENERATED")
        col = "#ff2d55" if is_ai else "#10b981"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 1rem; font-size: 0.82rem; line-height: 1.7;">
                    <div><b>Evidence File:</b> {selected_target['filename']}</div>
                    <div><b>Timestamp:</b> {selected_target['timestamp']}</div>
                    <div><b>SHA-256 Checksum:</b> <code style="font-size:0.75rem;">{sha}</code></div>
                    <div><b>Resolution:</b> {selected_target['image'].width} × {selected_target['image'].height} px</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div style="background: rgba(13, 19, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 1rem; font-size: 0.82rem; line-height: 1.7;">
                    <div><b>Classification Result:</b> <b style="color:{col}; font-size:0.95rem;">{selected_target['label']}</b></div>
                    <div><b>Model Confidence:</b> <b>{selected_target['confidence']*100:.2f}%</b></div>
                    <div><b>Calibrated Threshold:</b> τ = {selected_target['threshold']:.2f}</div>
                    <div><b>Model:</b> {selected_target['model_name']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📷 Plates Included in Final Dossier")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.image(selected_target["image"], caption="Plate 1: Original Evidence", use_container_width=True)
        with p2:
            st.image(selected_target["fft"], caption="Plate 2: 2D FFT Magnitude", use_container_width=True)
        with p3:
            st.image(selected_target["overlay"], caption="Plate 3: Grad-CAM Saliency", use_container_width=True)

        st.markdown("---")
        pdf_bytes = generate_forensic_pdf(
            filename=selected_target["filename"],
            original_image=selected_target["image"],
            fft_spectrum=selected_target["fft"],
            overlay_image=selected_target["overlay"],
            label=selected_target["label"],
            confidence=selected_target["confidence"],
            model_name=selected_target["model_name"],
            threshold=selected_target["threshold"]
        )

        st.download_button(
            label="📄 Download Formal Forensic Examination Dossier (PDF)",
            data=pdf_bytes,
            file_name=f"FORENSIC_DOSSIER_{selected_target['id']}_{selected_target['label']}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
